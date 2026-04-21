"""Live monitoring dashboard for muninn benchmark deployments.

Plotly Dash app that auto-refreshes every 15 seconds, showing:
- SQS queue depth (visible + in-flight + DLQ)
- ASG instance count and states with spot pricing
- Per-instance heartbeat status and current phase
- CloudWatch time-series chart (backed by metrics, survives refresh)
- Scaling events from ASG activity history
- CloudWatch log viewer filterable by level (ERROR/WARN/INFO)

Usage:
    uv run benchmarks/infra/dashboard.py
    # Opens at http://localhost:8060
"""

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import boto3
import dash
import plotly.graph_objects as go
import yaml
from botocore.exceptions import ClientError
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "benchmarks" / "infra" / "config.yml"

REFRESH_INTERVAL_MS = 15_000  # 15 seconds
DASHBOARD_PORT = 8060

# ── WCAG AA color palette (Tailwind v3) ───────────────────────
BG_PAGE = "#0f172a"
BG_CARD = "#1e293b"
BG_TABLE_HEAD = "#0f172a"
BORDER = "#334155"
TEXT_PRIMARY = "#f1f5f9"
TEXT_SECONDARY = "#cbd5e1"
TEXT_MUTED = "#94a3b8"
ACCENT_RED = "#f87171"
ACCENT_VIOLET = "#a78bfa"
ACCENT_AMBER = "#fbbf24"
ACCENT_GREEN = "#4ade80"
ACCENT_CYAN = "#22d3ee"
ROW_OK = "#052e16"
ROW_STALE = "#450a0a"
ROW_WARN = "#451a03"
ROW_TERM = "#18181b"


# ── Config ────────────────────────────────────────────────────────


def _load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sanitize_branch(branch: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "-", branch).strip("-")[:64]


# ── AWS Data Fetchers ─────────────────────────────────────────────


def _get_stack_outputs(cfg: dict) -> dict:
    branch = cfg["repo"]["branch"]
    safe_branch = _sanitize_branch(branch)
    stack_name = f"MuninnBench-{safe_branch}"
    cf = boto3.client("cloudformation", region_name=cfg["aws"]["ec2_region"])
    try:
        stack = cf.describe_stacks(StackName=stack_name)
        return {o["OutputKey"]: o["OutputValue"] for o in stack["Stacks"][0].get("Outputs", [])}
    except ClientError:
        return {}


def _get_queue_stats(cfg: dict, outputs: dict) -> dict:
    sqs = boto3.client("sqs", region_name=cfg["aws"]["ec2_region"])
    result = {"visible": 0, "in_flight": 0, "dlq": 0}
    queue_url = outputs.get("QueueUrl")
    dlq_url = outputs.get("DlqUrl")
    if queue_url:
        try:
            resp = sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["All"])
            attrs = resp.get("Attributes", {})
            result["visible"] = int(attrs.get("ApproximateNumberOfMessages", 0))
            result["in_flight"] = int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0))
        except ClientError:
            pass
    if dlq_url:
        try:
            resp = sqs.get_queue_attributes(QueueUrl=dlq_url, AttributeNames=["All"])
            attrs = resp.get("Attributes", {})
            result["dlq"] = int(attrs.get("ApproximateNumberOfMessages", 0))
        except ClientError:
            pass
    return result


# Cache on-demand prices per instance type to avoid repeated API calls
_ondemand_price_cache: dict[str, float] = {}


def _get_ondemand_price(instance_type: str, region: str) -> float:
    """Look up on-demand price for an instance type via the EC2 pricing API."""
    cache_key = f"{instance_type}:{region}"
    if cache_key in _ondemand_price_cache:
        return _ondemand_price_cache[cache_key]

    # Map EC2 region codes to pricing API location names
    region_names = {
        "us-east-1": "US East (N. Virginia)",
        "us-west-2": "US West (Oregon)",
        "ap-southeast-2": "Asia Pacific (Sydney)",
        "eu-west-1": "EU (Ireland)",
    }
    location = region_names.get(region, region)

    try:
        pricing = boto3.client("pricing", region_name="us-east-1")
        resp = pricing.get_products(
            ServiceCode="AmazonEC2",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
                {"Type": "TERM_MATCH", "Field": "location", "Value": location},
                {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
                {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
                {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
                {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
            ],
            MaxResults=1,
        )
        for price_item in resp.get("PriceList", []):
            product = json.loads(price_item)
            for term in product.get("terms", {}).get("OnDemand", {}).values():
                for dim in term.get("priceDimensions", {}).values():
                    price = float(dim["pricePerUnit"]["USD"])
                    _ondemand_price_cache[cache_key] = price
                    return price
    except (ClientError, Exception):
        pass

    _ondemand_price_cache[cache_key] = 0.0
    return 0.0


def _get_asg_instances(cfg: dict, outputs: dict) -> list[dict]:
    """Get ASG instance details including spot and on-demand pricing."""
    asg_name = outputs.get("AsgName")
    if not asg_name:
        return []

    ec2_region = cfg["aws"]["ec2_region"]
    asg_client = boto3.client("autoscaling", region_name=ec2_region)
    ec2 = boto3.client("ec2", region_name=ec2_region)

    try:
        resp = asg_client.describe_auto_scaling_groups(AutoScalingGroupNames=[asg_name])
        asg = resp["AutoScalingGroups"][0]
    except (ClientError, IndexError):
        return []

    instance_ids = [i["InstanceId"] for i in asg.get("Instances", [])]
    if not instance_ids:
        return []

    ec2_resp = ec2.describe_instances(InstanceIds=instance_ids)
    instances = []
    for res in ec2_resp["Reservations"]:
        for inst in res["Instances"]:
            lifecycle = inst.get("InstanceLifecycle", "on-demand")
            itype = inst["InstanceType"]
            az = inst["Placement"]["AvailabilityZone"]

            # Get current spot price for this instance type + AZ
            spot_price = "n/a"
            if lifecycle == "spot":
                try:
                    price_resp = ec2.describe_spot_price_history(
                        InstanceTypes=[itype],
                        AvailabilityZone=az,
                        ProductDescriptions=["Linux/UNIX"],
                        MaxResults=1,
                    )
                    if price_resp.get("SpotPriceHistory"):
                        spot_price = f"${price_resp['SpotPriceHistory'][0]['SpotPrice']}/hr"
                except ClientError:
                    pass

            # Compute uptime and accumulated cost
            launch_dt = inst["LaunchTime"]
            uptime_s = (
                datetime.now(UTC) - launch_dt.replace(tzinfo=UTC if launch_dt.tzinfo is None else launch_dt.tzinfo)
            ).total_seconds()
            uptime_hrs = uptime_s / 3600

            price_per_hr = 0.0
            if lifecycle == "spot" and spot_price != "n/a":
                price_per_hr = float(spot_price.replace("$", "").replace("/hr", ""))

            ondemand_price = _get_ondemand_price(itype, ec2_region)
            savings_pct = (
                round((1 - price_per_hr / ondemand_price) * 100) if ondemand_price > 0 and price_per_hr > 0 else 0
            )

            accumulated_cost = price_per_hr * uptime_hrs

            instances.append(
                {
                    "instance_id": inst["InstanceId"],
                    "state": inst["State"]["Name"],
                    "type": itype,
                    "az": az,
                    "launch_time": inst["LaunchTime"].isoformat(),
                    "ip": inst.get("PublicIpAddress", "n/a"),
                    "lifecycle": lifecycle,
                    "spot_price": spot_price,
                    "ondemand_price": f"${ondemand_price:.3f}/hr" if ondemand_price > 0 else "n/a",
                    "savings_pct": f"{savings_pct}%" if savings_pct > 0 else "n/a",
                    "uptime_hrs": round(uptime_hrs, 2),
                    "accumulated_cost": round(accumulated_cost, 4),
                    "price_per_hr": price_per_hr,
                }
            )

    return instances


def _get_heartbeats(cfg: dict, instance_ids: list[str]) -> dict[str, dict]:
    s3 = boto3.client("s3", region_name=cfg["aws"]["s3_region"])
    bucket = cfg["aws"]["s3_bucket"]
    heartbeats = {}
    for iid in instance_ids:
        try:
            resp = s3.get_object(Bucket=bucket, Key=f"heartbeat/{iid}.json")
            hb = json.loads(resp["Body"].read())
            ts = datetime.fromisoformat(hb["timestamp"].replace("Z", "+00:00"))
            age = (datetime.now(UTC) - ts).total_seconds()
            heartbeats[iid] = {
                "phase": hb.get("phase", "?"),
                "age_s": round(age),
                "run_id": hb.get("run_id", "?"),
                "status": "STALE" if age > 60 else "OK",
            }
        except ClientError:
            heartbeats[iid] = {"phase": "n/a", "age_s": -1, "run_id": "n/a", "status": "NO HB"}
    return heartbeats


def _get_cloudwatch_timeseries(cfg: dict, outputs: dict, hours: float = 1.0) -> dict:
    cw = boto3.client("cloudwatch", region_name=cfg["aws"]["ec2_region"])
    queue_name = (outputs.get("QueueUrl") or "").rsplit("/", 1)[-1]
    dlq_name = (outputs.get("DlqUrl") or "").rsplit("/", 1)[-1]
    asg_name = outputs.get("AsgName", "")

    now = datetime.now(UTC)
    start = now - timedelta(hours=hours)

    if hours <= 3:
        period = 60
    elif hours <= 24:
        period = 300
    else:
        period = 3600

    result = {"timestamps": [], "visible": [], "inflight": [], "dlq": [], "in_service": [], "desired": []}
    if not queue_name:
        return result

    queries = [
        {
            "Id": "visible",
            "MetricStat": {
                "Metric": {
                    "Namespace": "AWS/SQS",
                    "MetricName": "ApproximateNumberOfMessagesVisible",
                    "Dimensions": [{"Name": "QueueName", "Value": queue_name}],
                },
                "Period": period,
                "Stat": "Maximum",
            },
        },
        {
            "Id": "inflight",
            "MetricStat": {
                "Metric": {
                    "Namespace": "AWS/SQS",
                    "MetricName": "ApproximateNumberOfMessagesNotVisible",
                    "Dimensions": [{"Name": "QueueName", "Value": queue_name}],
                },
                "Period": period,
                "Stat": "Maximum",
            },
        },
    ]
    if dlq_name:
        queries.append(
            {
                "Id": "dlq",
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/SQS",
                        "MetricName": "ApproximateNumberOfMessagesVisible",
                        "Dimensions": [{"Name": "QueueName", "Value": dlq_name}],
                    },
                    "Period": period,
                    "Stat": "Maximum",
                },
            }
        )
    if asg_name:
        queries.append(
            {
                "Id": "in_service",
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/AutoScaling",
                        "MetricName": "GroupInServiceInstances",
                        "Dimensions": [{"Name": "AutoScalingGroupName", "Value": asg_name}],
                    },
                    "Period": period,
                    "Stat": "Maximum",
                },
            }
        )
        queries.append(
            {
                "Id": "desired",
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/AutoScaling",
                        "MetricName": "GroupDesiredCapacity",
                        "Dimensions": [{"Name": "AutoScalingGroupName", "Value": asg_name}],
                    },
                    "Period": period,
                    "Stat": "Maximum",
                },
            }
        )

    try:
        resp = cw.get_metric_data(MetricDataQueries=queries, StartTime=start, EndTime=now)
    except ClientError:
        return result

    ts_fmt = "%H:%M" if hours <= 24 else "%m-%d %H:%M"
    series = {}
    for metric_result in resp.get("MetricDataResults", []):
        mid = metric_result["Id"]
        for ts_val, val in zip(metric_result.get("Timestamps", []), metric_result.get("Values", []), strict=False):
            ts_str = ts_val.strftime(ts_fmt)
            series.setdefault(
                ts_str, {"visible": 0, "inflight": 0, "dlq": 0, "in_service": 0, "desired": 0, "_sort": ts_val}
            )
            series[ts_str][mid] = int(val)

    for ts_str in sorted(series.keys(), key=lambda k: series[k].get("_sort", k)):
        result["timestamps"].append(ts_str)
        result["visible"].append(series[ts_str]["visible"])
        result["inflight"].append(series[ts_str]["inflight"])
        result["dlq"].append(series[ts_str]["dlq"])
        result["in_service"].append(series[ts_str]["in_service"])
        result["desired"].append(series[ts_str]["desired"])

    return result


def _get_asg_events(cfg: dict, outputs: dict, max_events: int = 20) -> list[dict]:
    asg_name = outputs.get("AsgName")
    if not asg_name:
        return []
    asg_client = boto3.client("autoscaling", region_name=cfg["aws"]["ec2_region"])
    try:
        resp = asg_client.describe_scaling_activities(AutoScalingGroupName=asg_name, MaxRecords=max_events)
    except ClientError:
        return []

    events = []
    for act in resp.get("Activities", []):
        cause = act.get("Cause", "")
        ts = act["StartTime"].strftime("%H:%M:%S UTC")
        description = act.get("Description", "")

        if "spot interruption" in cause.lower() or "spot instance" in description.lower():
            event_type = "SPOT RECLAIM"
        elif "unhealthy" in cause.lower():
            event_type = "UNHEALTHY"
        elif "shrinking" in cause.lower() or ("changing the desired capacity from" in cause and "to 0" in cause):
            event_type = "SCALE IN"
        elif "launching" in description.lower() or "Launching" in description:
            event_type = "SCALE OUT"
        elif "terminating" in description.lower() or "Terminating" in description:
            event_type = "TERMINATE"
        else:
            event_type = "ACTIVITY"

        instance_id = ""
        match = re.search(r"i-[0-9a-f]+", cause)
        if match:
            instance_id = match.group(0)[:12]

        capacity = ""
        cap_match = re.search(r"from (\d+) to (\d+)", cause)
        if cap_match:
            capacity = f"{cap_match.group(1)}->{cap_match.group(2)}"

        events.append(
            {
                "time": ts,
                "type": event_type,
                "instance": instance_id,
                "capacity": capacity,
                "status": act.get("StatusCode", ""),
            }
        )

    return events


def _get_log_stream_instance_ids(cfg: dict, max_streams: int = 20) -> list[str]:
    """Get instance IDs that have CloudWatch log streams (current + recent)."""
    logs_client = boto3.client("logs", region_name=cfg["aws"]["ec2_region"])
    try:
        resp = logs_client.describe_log_streams(
            logGroupName="/muninn/benchmarks",
            orderBy="LastEventTime",
            descending=True,
            limit=max_streams,
        )
        return [s["logStreamName"] for s in resp.get("logStreams", []) if s.get("logStreamName", "").startswith("i-")]
    except ClientError:
        return []


def _get_cloudwatch_logs(cfg: dict, instance_ids: list[str], level_filter: str, minutes: int = 60) -> list[dict]:
    """Fetch CloudWatch logs for one or more instances, filtered by level."""
    if not instance_ids:
        return []

    logs_client = boto3.client("logs", region_name=cfg["aws"]["ec2_region"])
    start_time = int((datetime.now(UTC) - timedelta(minutes=minutes)).timestamp() * 1000)

    filter_pattern = ""
    if level_filter == "ERROR":
        filter_pattern = "?ERROR ?Error ?error ?FAILED ?Traceback"
    elif level_filter == "WARN":
        filter_pattern = "?WARNING ?WARN ?warn ?STALE"

    kwargs: dict = {
        "logGroupName": "/muninn/benchmarks",
        "logStreamNames": instance_ids,
        "startTime": start_time,
        "limit": 200,
    }
    if filter_pattern:
        kwargs["filterPattern"] = filter_pattern

    try:
        resp = logs_client.filter_log_events(**kwargs)
    except ClientError:
        return []

    entries = []
    for event in resp.get("events", []):
        ts = datetime.fromtimestamp(event["timestamp"] / 1000, tz=UTC).strftime("%H:%M:%S")
        stream = event.get("logStreamName", "")[:12]
        msg = event["message"].rstrip()
        entries.append({"time": ts, "instance": stream, "message": msg})

    return entries


# ── Dash App ──────────────────────────────────────────────────────


def create_app() -> dash.Dash:
    cfg = _load_config()
    branch = cfg["repo"]["branch"]

    app = dash.Dash(__name__, title=f"Muninn Benchmarks - {branch}")

    app.layout = html.Div(
        style={
            "fontFamily": "monospace",
            "padding": "20px",
            "backgroundColor": BG_PAGE,
            "color": TEXT_PRIMARY,
            "minHeight": "100vh",
        },
        children=[
            html.H1("Muninn Benchmark Dashboard", style={"color": ACCENT_RED}),
            html.P(f"Branch: {branch}", style={"color": TEXT_SECONDARY, "fontSize": "14px"}),
            dcc.Interval(id="refresh", interval=REFRESH_INTERVAL_MS, n_intervals=0),
            html.Div(id="last-updated", style={"color": TEXT_MUTED, "fontSize": "12px", "marginBottom": "20px"}),
            # ── Metric Cards ──────────────────────────────────
            html.Div(
                style={"display": "flex", "gap": "20px", "marginBottom": "30px", "flexWrap": "wrap"},
                children=[
                    _metric_card("sqs-visible", "Queue Visible", ACCENT_RED),
                    _metric_card("sqs-inflight", "In Flight", ACCENT_VIOLET),
                    _metric_card("sqs-dlq", "Dead Letter", ACCENT_AMBER),
                    _metric_card("asg-running", "Running Instances", ACCENT_GREEN),
                ],
            ),
            # ── Time Series Chart ─────────────────────────────
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "15px", "marginBottom": "10px"},
                children=[
                    html.H3("Queue & Workers Over Time", style={"color": ACCENT_RED, "margin": "0"}),
                    dcc.RadioItems(
                        id="time-range",
                        options=[
                            {"label": label, "value": v}
                            for label, v in [("1h", 1), ("3h", 3), ("12h", 12), ("1d", 24), ("3d", 72), ("7d", 168)]
                        ],
                        value=72,
                        inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "12px", "cursor": "pointer", "color": TEXT_SECONDARY},
                        persistence=True,
                        persistence_type="local",
                    ),
                ],
            ),
            dcc.Graph(id="queue-chart", style={"height": "250px"}),
            dcc.Graph(id="workers-chart", style={"height": "250px", "marginTop": "10px"}),
            # ── Workers Table ─────────────────────────────────
            html.H3("Workers", style={"color": ACCENT_RED, "marginBottom": "10px"}),
            html.Div(id="instance-table"),
            # ── Scaling Events ────────────────────────────────
            html.H3("Scaling Events", style={"color": ACCENT_RED, "marginTop": "30px", "marginBottom": "10px"}),
            html.Div(id="event-table"),
            # ── CloudWatch Logs Viewer ────────────────────────
            html.H3("CloudWatch Logs", style={"color": ACCENT_CYAN, "marginTop": "30px", "marginBottom": "10px"}),
            html.Div(
                style={
                    "display": "flex",
                    "gap": "15px",
                    "alignItems": "center",
                    "marginBottom": "10px",
                    "flexWrap": "wrap",
                },
                children=[
                    dcc.Dropdown(
                        id="log-instance-ids",
                        multi=True,
                        placeholder="Select instance(s)...",
                        style={"minWidth": "400px", "backgroundColor": BG_CARD, "color": TEXT_PRIMARY},
                        persistence=True,
                        persistence_type="local",
                    ),
                    dcc.RadioItems(
                        id="log-level",
                        options=[{"label": level, "value": level} for level in ["INFO", "WARN", "ERROR"]],
                        value="INFO",
                        inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "12px", "cursor": "pointer", "color": TEXT_SECONDARY},
                        persistence=True,
                        persistence_type="local",
                    ),
                    html.Button(
                        "Fetch Logs",
                        id="fetch-logs-btn",
                        style={
                            "backgroundColor": ACCENT_CYAN,
                            "color": BG_PAGE,
                            "border": "none",
                            "padding": "8px 16px",
                            "borderRadius": "4px",
                            "cursor": "pointer",
                            "fontWeight": "bold",
                        },
                    ),
                ],
            ),
            html.Div(
                id="log-viewer",
                style={
                    "backgroundColor": BG_CARD,
                    "padding": "15px",
                    "borderRadius": "8px",
                    "maxHeight": "400px",
                    "overflowY": "auto",
                    "fontSize": "12px",
                    "whiteSpace": "pre-wrap",
                    "fontFamily": "monospace",
                    "color": TEXT_SECONDARY,
                    "border": f"1px solid {BORDER}",
                },
            ),
        ],
    )

    # ── Main dashboard callback ───────────────────────────────
    @app.callback(
        [
            Output("sqs-visible-value", "children"),
            Output("sqs-inflight-value", "children"),
            Output("sqs-dlq-value", "children"),
            Output("asg-running-value", "children"),
            Output("queue-chart", "figure"),
            Output("workers-chart", "figure"),
            Output("instance-table", "children"),
            Output("event-table", "children"),
            Output("log-instance-ids", "options"),
            Output("last-updated", "children"),
        ],
        [Input("refresh", "n_intervals"), Input("time-range", "value")],
    )
    def update_dashboard(n_intervals, time_range_hours):
        now = datetime.now(UTC)
        ts = now.strftime("%H:%M:%S UTC")

        try:
            outputs = _get_stack_outputs(cfg)
            queue = _get_queue_stats(cfg, outputs)
            instances = _get_asg_instances(cfg, outputs)
            instance_ids = [i["instance_id"] for i in instances]
            heartbeats = _get_heartbeats(cfg, instance_ids)
            cw_data = _get_cloudwatch_timeseries(cfg, outputs, hours=float(time_range_hours or 1))
            asg_events = _get_asg_events(cfg, outputs, max_events=20)
        except Exception as e:
            empty_fig = go.Figure()
            empty_fig.update_layout(template="plotly_dark", paper_bgcolor=BG_PAGE, plot_bgcolor=BG_CARD)
            return (
                "?",
                "?",
                "?",
                "?",
                empty_fig,
                empty_fig,
                html.P(f"Error: {e}", style={"color": ACCENT_RED}),
                "",
                [],
                f"Error at {ts}",
            )

        # Running instance count (from EC2 API, not CloudWatch — no lag)
        running_count = sum(1 for i in instances if i["state"] == "running")

        # ── Instance table with spot pricing ──────────────
        rows = []
        for inst in instances:
            iid = inst["instance_id"]
            hb = heartbeats.get(iid, {})
            phase = hb.get("phase", "n/a")
            age = hb.get("age_s", -1)
            hb_status = hb.get("status", "?")
            age_str = f"{age}s" if age >= 0 else "n/a"

            rows.append(
                {
                    "Instance": iid,
                    "State": inst["state"],
                    "Type": inst["type"],
                    "Lifecycle": inst["lifecycle"],
                    "Spot $/hr": inst["spot_price"],
                    "OnDemand $/hr": inst["ondemand_price"],
                    "Saving": inst["savings_pct"],
                    "Uptime": f"{inst['uptime_hrs']:.1f}h",
                    "Cost": f"${inst['accumulated_cost']:.3f}",
                    "Phase": phase,
                    "Heartbeat": f"{hb_status} ({age_str})",
                }
            )

        if rows:
            table = dash_table.DataTable(
                data=rows,
                columns=[{"name": c, "id": c} for c in rows[0].keys()],
                style_header={
                    "backgroundColor": BG_TABLE_HEAD,
                    "color": TEXT_PRIMARY,
                    "fontWeight": "bold",
                    "border": f"1px solid {BORDER}",
                },
                style_cell={
                    "backgroundColor": BG_CARD,
                    "color": TEXT_PRIMARY,
                    "border": f"1px solid {BORDER}",
                    "fontSize": "13px",
                    "padding": "8px",
                },
                style_data_conditional=[
                    {
                        "if": {"filter_query": '{Heartbeat} contains "STALE"'},
                        "backgroundColor": ROW_STALE,
                        "color": ACCENT_RED,
                    },
                    {
                        "if": {"filter_query": '{Heartbeat} contains "OK"'},
                        "backgroundColor": ROW_OK,
                        "color": ACCENT_GREEN,
                    },
                    {
                        "if": {"filter_query": '{State} eq "terminated"'},
                        "color": TEXT_MUTED,
                        "backgroundColor": ROW_TERM,
                    },
                ],
            )
        else:
            table = html.P("No workers running", style={"color": TEXT_MUTED, "padding": "20px"})

        # ── Chart 1: Queue depth ─────────────────────────
        chart_layout = {
            "template": "plotly_dark",
            "paper_bgcolor": BG_PAGE,
            "plot_bgcolor": BG_CARD,
            "font": {"color": TEXT_PRIMARY},
            "margin": {"l": 50, "r": 50, "t": 30, "b": 30},
            "legend": {"orientation": "h", "y": 1.15, "font": {"color": TEXT_SECONDARY}},
            "xaxis": {"title": None, "gridcolor": BORDER, "tickfont": {"color": TEXT_MUTED}},
        }

        queue_fig = go.Figure()
        queue_fig.add_trace(
            go.Scatter(
                x=cw_data["timestamps"],
                y=cw_data["visible"],
                name="Visible",
                mode="lines",
                line={"color": ACCENT_RED, "width": 2},
                fill="tozeroy",
                fillcolor="rgba(248,113,113,0.1)",
            )
        )
        queue_fig.add_trace(
            go.Scatter(
                x=cw_data["timestamps"],
                y=cw_data["inflight"],
                name="In Flight",
                mode="lines",
                line={"color": ACCENT_VIOLET, "width": 2},
            )
        )
        queue_fig.add_trace(
            go.Scatter(
                x=cw_data["timestamps"],
                y=cw_data["dlq"],
                name="Dead Letter",
                mode="lines",
                line={"color": ACCENT_AMBER, "width": 2, "dash": "dot"},
            )
        )
        queue_fig.update_layout(
            **chart_layout,
            yaxis={"title": "Messages", "gridcolor": BORDER, "rangemode": "tozero", "tickfont": {"color": TEXT_MUTED}},
        )

        # ── Chart 2: Workers + cumulative cost ───────────
        workers_fig = go.Figure()
        workers_fig.add_trace(
            go.Scatter(
                x=cw_data["timestamps"],
                y=cw_data["in_service"],
                name="In Service",
                mode="lines",
                line={"color": ACCENT_GREEN, "width": 3},
            )
        )
        workers_fig.add_trace(
            go.Scatter(
                x=cw_data["timestamps"],
                y=cw_data["desired"],
                name="Desired",
                mode="lines",
                line={"color": ACCENT_GREEN, "width": 1, "dash": "dash"},
            )
        )

        # Cumulative spot cost
        avg_spot_price = sum(i["price_per_hr"] for i in instances if i["price_per_hr"] > 0)
        if avg_spot_price == 0:
            avg_spot_price = 0.084  # fallback: t3.xlarge spot
        period_hours = (1 if time_range_hours <= 3 else 5 if time_range_hours <= 24 else 60) / 60
        cumulative_cost = []
        running_total = 0.0
        for in_svc in cw_data["in_service"]:
            running_total += in_svc * avg_spot_price * period_hours
            cumulative_cost.append(round(running_total, 4))
        if cumulative_cost:
            workers_fig.add_trace(
                go.Scatter(
                    x=cw_data["timestamps"],
                    y=cumulative_cost,
                    name="Cumulative $",
                    mode="lines",
                    line={"color": ACCENT_CYAN, "width": 2},
                    yaxis="y2",
                )
            )

        workers_layout = {**chart_layout, "margin": {"l": 50, "r": 60, "t": 30, "b": 30}}
        workers_fig.update_layout(
            **workers_layout,
            yaxis={"title": "Workers", "gridcolor": BORDER, "rangemode": "tozero", "tickfont": {"color": TEXT_MUTED}},
            yaxis2={
                "title": "Cost ($)",
                "overlaying": "y",
                "side": "right",
                "gridcolor": BORDER,
                "rangemode": "tozero",
                "tickfont": {"color": ACCENT_CYAN},
                "showgrid": False,
            },
        )

        # ── Scaling events table ──────────────────────────
        if asg_events:
            event_table = dash_table.DataTable(
                data=asg_events,
                columns=[{"name": c, "id": c} for c in ["time", "type", "instance", "capacity", "status"]],
                style_header={
                    "backgroundColor": BG_TABLE_HEAD,
                    "color": TEXT_PRIMARY,
                    "fontWeight": "bold",
                    "border": f"1px solid {BORDER}",
                },
                style_cell={
                    "backgroundColor": BG_CARD,
                    "color": TEXT_PRIMARY,
                    "border": f"1px solid {BORDER}",
                    "fontSize": "13px",
                    "padding": "6px",
                },
                style_data_conditional=[
                    {
                        "if": {"filter_query": '{type} eq "SPOT RECLAIM"'},
                        "backgroundColor": ROW_STALE,
                        "color": ACCENT_RED,
                    },
                    {"if": {"filter_query": '{type} eq "SCALE OUT"'}, "backgroundColor": ROW_OK, "color": ACCENT_GREEN},
                    {
                        "if": {"filter_query": '{type} eq "SCALE IN"'},
                        "backgroundColor": ROW_WARN,
                        "color": ACCENT_AMBER,
                    },
                    {
                        "if": {"filter_query": '{type} eq "UNHEALTHY"'},
                        "backgroundColor": ROW_STALE,
                        "color": ACCENT_RED,
                    },
                ],
                page_size=10,
            )
        else:
            event_table = html.P("No scaling events", style={"color": TEXT_MUTED, "padding": "20px"})

        # ── Log stream dropdown options ─────────────────
        # Combine running instance IDs + recent CloudWatch log streams
        running_ids = [i["instance_id"] for i in instances]
        log_stream_ids = _get_log_stream_instance_ids(cfg)
        all_ids = list(dict.fromkeys(running_ids + log_stream_ids))  # dedupe, preserve order
        log_options = [
            {"label": f"{iid} {'(running)' if iid in running_ids else '(recent)'}", "value": iid} for iid in all_ids
        ]

        return (
            str(queue["visible"]),
            str(queue["in_flight"]),
            str(queue["dlq"]),
            str(running_count),
            queue_fig,
            workers_fig,
            table,
            event_table,
            log_options,
            f"Last updated: {ts} (every {REFRESH_INTERVAL_MS // 1000}s)",
        )

    # ── CloudWatch logs callback (on-demand, not auto-refresh) ─
    @app.callback(
        Output("log-viewer", "children"),
        [Input("fetch-logs-btn", "n_clicks")],
        [dash.State("log-instance-ids", "value"), dash.State("log-level", "value")],
        prevent_initial_call=True,
    )
    def fetch_logs(n_clicks, instance_ids, level):
        if not instance_ids:
            return "Select instance(s) from the dropdown and click Fetch Logs."

        entries = _get_cloudwatch_logs(cfg, instance_ids, level, minutes=60)

        if not entries:
            ids_str = ", ".join(instance_ids)
            return f"No {level} logs found for {ids_str} in the last 60 minutes."

        lines = []
        for e in entries:
            lines.append(f"[{e['time']}] {e['instance']} | {e['message']}")
        return "\n".join(lines)

    return app


def _metric_card(id_prefix: str, label: str, accent: str) -> html.Div:
    return html.Div(
        style={
            "backgroundColor": BG_CARD,
            "borderRadius": "8px",
            "padding": "20px",
            "textAlign": "center",
            "minWidth": "150px",
            "borderLeft": f"4px solid {accent}",
        },
        children=[
            html.Div(
                id=f"{id_prefix}-value",
                style={"fontSize": "36px", "fontWeight": "bold", "color": accent},
                children="...",
            ),
            html.Div(label, style={"fontSize": "12px", "color": TEXT_SECONDARY, "marginTop": "5px"}),
        ],
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Muninn benchmark monitoring dashboard")
    parser.add_argument(
        "--port", type=int, default=DASHBOARD_PORT, help=f"Port to serve on (default: {DASHBOARD_PORT})"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    app = create_app()
    log.info("Dashboard: http://localhost:%d", args.port)
    app.run(debug=False, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
