"""Live monitoring dashboard for muninn benchmark deployments.

Plotly Dash app that auto-refreshes every 15 seconds, showing:
- SQS queue depth (visible + in-flight + DLQ)
- ASG instance count and states
- Per-instance heartbeat status and current phase
- Timeline of phase transitions

Usage:
    uv run benchmarks/infra/dashboard.py
    # Opens at http://localhost:8050
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import dash
import yaml
from botocore.exceptions import ClientError
import plotly.graph_objects as go
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "benchmarks" / "infra" / "config.yml"

REFRESH_INTERVAL_MS = 15_000  # 15 seconds

# ── WCAG AA color palette (Tailwind v3) ───────────────────────
# All text colors pass 4.5:1 contrast ratio on their backgrounds.
BG_PAGE = "#0f172a"         # slate-900
BG_CARD = "#1e293b"         # slate-800
BG_TABLE_HEAD = "#0f172a"   # slate-900
BORDER = "#334155"          # slate-700
TEXT_PRIMARY = "#f1f5f9"    # slate-100 (14.5:1 on slate-900)
TEXT_SECONDARY = "#cbd5e1"  # slate-300 (9.1:1 on slate-900)
TEXT_MUTED = "#94a3b8"      # slate-400 (5.6:1 on slate-900)
ACCENT_RED = "#f87171"      # red-400 (5.3:1 on slate-800)
ACCENT_VIOLET = "#a78bfa"   # violet-400 (5.8:1 on slate-800)
ACCENT_AMBER = "#fbbf24"    # amber-400 (11.2:1 on slate-800)
ACCENT_GREEN = "#4ade80"    # green-400 (8.9:1 on slate-800)
ROW_OK = "#052e16"          # green-950 (green tint, white text readable)
ROW_STALE = "#450a0a"       # red-950 (red tint, white text readable)
ROW_WARN = "#451a03"        # amber-950 (amber tint, white text readable)
ROW_TERM = "#18181b"        # zinc-900 (dimmed row)


# ── Config ────────────────────────────────────────────────────────


def _load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sanitize_branch(branch: str) -> str:

    return re.sub(r"[^a-zA-Z0-9]", "-", branch).strip("-")[:64]


# ── AWS Data Fetchers ─────────────────────────────────────────────


def _get_stack_outputs(cfg: dict) -> dict:
    """Get CloudFormation stack outputs for the current branch."""
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
    """Get SQS queue statistics."""
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


def _get_asg_instances(cfg: dict, outputs: dict) -> list[dict]:
    """Get ASG instance details."""
    asg_name = outputs.get("AsgName")
    if not asg_name:
        return []

    asg_client = boto3.client("autoscaling", region_name=cfg["aws"]["ec2_region"])
    ec2 = boto3.client("ec2", region_name=cfg["aws"]["ec2_region"])

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
            instances.append({
                "instance_id": inst["InstanceId"],
                "state": inst["State"]["Name"],
                "type": inst["InstanceType"],
                "az": inst["Placement"]["AvailabilityZone"],
                "launch_time": inst["LaunchTime"].isoformat(),
                "ip": inst.get("PublicIpAddress", "n/a"),
            })

    return instances


def _get_heartbeats(cfg: dict, instance_ids: list[str]) -> dict[str, dict]:
    """Get S3 heartbeats for all instances."""
    s3 = boto3.client("s3", region_name=cfg["aws"]["s3_region"])
    bucket = cfg["aws"]["s3_bucket"]
    heartbeats = {}

    for iid in instance_ids:
        try:
            resp = s3.get_object(Bucket=bucket, Key=f"heartbeat/{iid}.json")
            hb = json.loads(resp["Body"].read())
            ts = datetime.fromisoformat(hb["timestamp"].replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            heartbeats[iid] = {
                "phase": hb.get("phase", "?"),
                "age_s": round(age),
                "run_id": hb.get("run_id", "?"),
                "status": "STALE" if age > 60 else "OK",
            }
        except ClientError:
            heartbeats[iid] = {"phase": "n/a", "age_s": -1, "run_id": "n/a", "status": "NO HB"}

    return heartbeats


def _get_asg_desired(cfg: dict, outputs: dict) -> int:
    """Get ASG desired capacity."""
    asg_name = outputs.get("AsgName")
    if not asg_name:
        return 0
    asg_client = boto3.client("autoscaling", region_name=cfg["aws"]["ec2_region"])
    try:
        resp = asg_client.describe_auto_scaling_groups(AutoScalingGroupNames=[asg_name])
        return resp["AutoScalingGroups"][0]["DesiredCapacity"]
    except (ClientError, IndexError):
        return 0


def _get_cloudwatch_timeseries(cfg: dict, outputs: dict, hours: float = 1.0) -> dict:
    """Fetch SQS + ASG metrics from CloudWatch for the time-series chart.

    Returns aligned time series from CloudWatch (not ephemeral in-memory state),
    so a page refresh always shows the full history.
    """
    cw = boto3.client("cloudwatch", region_name=cfg["aws"]["ec2_region"])
    queue_name = (outputs.get("QueueUrl") or "").rsplit("/", 1)[-1]
    dlq_name = (outputs.get("DlqUrl") or "").rsplit("/", 1)[-1]
    asg_name = outputs.get("AsgName", "")

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)

    # CloudWatch limits to 1440 data points per query.
    # Scale the period to keep within that limit.
    if hours <= 3:
        period = 60       # 1-minute granularity
    elif hours <= 24:
        period = 300      # 5-minute granularity
    else:
        period = 3600     # 1-hour granularity

    result = {"timestamps": [], "visible": [], "inflight": [], "dlq": [], "workers": []}

    if not queue_name:
        return result

    queries = [
        {
            "Id": "visible",
            "MetricStat": {
                "Metric": {"Namespace": "AWS/SQS", "MetricName": "ApproximateNumberOfMessagesVisible", "Dimensions": [{"Name": "QueueName", "Value": queue_name}]},
                "Period": period, "Stat": "Maximum",
            },
        },
        {
            "Id": "inflight",
            "MetricStat": {
                "Metric": {"Namespace": "AWS/SQS", "MetricName": "ApproximateNumberOfMessagesNotVisible", "Dimensions": [{"Name": "QueueName", "Value": queue_name}]},
                "Period": period, "Stat": "Maximum",
            },
        },
    ]

    if dlq_name:
        queries.append({
            "Id": "dlq",
            "MetricStat": {
                "Metric": {"Namespace": "AWS/SQS", "MetricName": "ApproximateNumberOfMessagesVisible", "Dimensions": [{"Name": "QueueName", "Value": dlq_name}]},
                "Period": period, "Stat": "Maximum",
            },
        })

    if asg_name:
        queries.append({
            "Id": "workers",
            "MetricStat": {
                "Metric": {"Namespace": "AWS/AutoScaling", "MetricName": "GroupInServiceInstances", "Dimensions": [{"Name": "AutoScalingGroupName", "Value": asg_name}]},
                "Period": period, "Stat": "Maximum",
            },
        })

    try:
        resp = cw.get_metric_data(
            MetricDataQueries=queries,
            StartTime=start,
            EndTime=now,
        )
    except ClientError:
        return result

    # Build aligned series — CloudWatch returns each metric separately
    ts_fmt = "%H:%M" if hours <= 24 else "%m-%d %H:%M"
    series = {}
    for metric_result in resp.get("MetricDataResults", []):
        mid = metric_result["Id"]
        timestamps = metric_result.get("Timestamps", [])
        values = metric_result.get("Values", [])
        for ts_val, val in zip(timestamps, values):
            ts_str = ts_val.strftime(ts_fmt)
            series.setdefault(ts_str, {"visible": 0, "inflight": 0, "dlq": 0, "workers": 0, "_sort": ts_val})
            series[ts_str][mid] = int(val)

    # Sort by actual timestamp and flatten
    for ts_str in sorted(series.keys(), key=lambda k: series[k].get("_sort", k)):
        result["timestamps"].append(ts_str)
        result["visible"].append(series[ts_str]["visible"])
        result["inflight"].append(series[ts_str]["inflight"])
        result["dlq"].append(series[ts_str]["dlq"])
        result["workers"].append(series[ts_str]["workers"])

    return result


def _get_asg_events(cfg: dict, outputs: dict, max_events: int = 20) -> list[dict]:
    """Fetch ASG scaling activities: scale-up, scale-in, spot reclaim, unhealthy."""
    asg_name = outputs.get("AsgName")
    if not asg_name:
        return []

    asg_client = boto3.client("autoscaling", region_name=cfg["aws"]["ec2_region"])
    try:
        resp = asg_client.describe_scaling_activities(
            AutoScalingGroupName=asg_name,
            MaxRecords=max_events,
        )
    except ClientError:
        return []

    events = []
    for act in resp.get("Activities", []):
        cause = act.get("Cause", "")
        ts = act["StartTime"].strftime("%H:%M:%S UTC")
        description = act.get("Description", "")

        # Classify the event
        if "spot interruption" in cause.lower() or "spot instance" in description.lower():
            event_type = "SPOT RECLAIM"
        elif "unhealthy" in cause.lower():
            event_type = "UNHEALTHY"
        elif "shrinking" in cause.lower() or "changing the desired capacity from" in cause and "to 0" in cause:
            event_type = "SCALE IN"
        elif "launching" in description.lower() or "Launching" in description:
            event_type = "SCALE OUT"
        elif "terminating" in description.lower() or "Terminating" in description:
            event_type = "TERMINATE"
        else:
            event_type = "ACTIVITY"

        # Extract instance ID if present
        instance_id = ""
        if "instance" in cause.lower():
            match = re.search(r"i-[0-9a-f]+", cause)
            if match:
                instance_id = match.group(0)[:12]

        # Extract capacity change
        capacity = ""
        cap_match = re.search(r"from (\d+) to (\d+)", cause)
        if cap_match:
            capacity = f"{cap_match.group(1)}->{cap_match.group(2)}"

        events.append({
            "time": ts,
            "type": event_type,
            "instance": instance_id,
            "capacity": capacity,
            "status": act.get("StatusCode", ""),
        })

    return events


# ── Dash App ──────────────────────────────────────────────────────


def create_app() -> dash.Dash:
    """Create and configure the Dash app."""
    cfg = _load_config()
    branch = cfg["repo"]["branch"]

    app = dash.Dash(
        __name__,
        title=f"Muninn Benchmarks - {branch}",
    )

    app.layout = html.Div(
        style={"fontFamily": "monospace", "padding": "20px", "backgroundColor": BG_PAGE, "color": TEXT_PRIMARY, "minHeight": "100vh"},
        children=[
            html.H1("Muninn Benchmark Dashboard", style={"color": ACCENT_RED}),
            html.P(f"Branch: {branch}", style={"color": TEXT_SECONDARY, "fontSize": "14px"}),

            dcc.Interval(id="refresh", interval=REFRESH_INTERVAL_MS, n_intervals=0),

            html.Div(id="last-updated", style={"color": TEXT_MUTED, "fontSize": "12px", "marginBottom": "20px"}),

            # ── Queue Stats ───────────────────────────────────
            html.Div(
                style={"display": "flex", "gap": "20px", "marginBottom": "30px"},
                children=[
                    _metric_card("sqs-visible", "Queue Visible", ACCENT_RED, BG_CARD, TEXT_SECONDARY),
                    _metric_card("sqs-inflight", "In Flight", ACCENT_VIOLET, BG_CARD, TEXT_SECONDARY),
                    _metric_card("sqs-dlq", "Dead Letter", ACCENT_AMBER, BG_CARD, TEXT_SECONDARY),
                    _metric_card("asg-desired", "ASG Workers", ACCENT_GREEN, BG_CARD, TEXT_SECONDARY),
                ],
            ),

            # ── Time Series Chart (backed by CloudWatch) ─────
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "15px", "marginBottom": "10px"},
                children=[
                    html.H3("Queue & Workers Over Time", style={"color": ACCENT_RED, "margin": "0"}),
                    dcc.RadioItems(
                        id="time-range",
                        options=[
                            {"label": "1h", "value": 1},
                            {"label": "3h", "value": 3},
                            {"label": "12h", "value": 12},
                            {"label": "1d", "value": 24},
                            {"label": "3d", "value": 72},
                            {"label": "7d", "value": 168},
                        ],
                        value=72,
                        inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "12px", "cursor": "pointer", "color": TEXT_SECONDARY},
                    ),
                ],
            ),
            dcc.Graph(id="timeseries-chart", style={"height": "350px"}),

            # ── Instance Table ────────────────────────────────
            html.H3("Workers", style={"color": ACCENT_RED, "marginBottom": "10px"}),
            html.Div(id="instance-table"),

            # ── ASG Event Log (scaling activities) ────────────
            html.H3("Scaling Events", style={"color": ACCENT_RED, "marginTop": "30px", "marginBottom": "10px"}),
            html.Div(id="event-table"),
        ],
    )

    @app.callback(
        [
            Output("sqs-visible-value", "children"),
            Output("sqs-inflight-value", "children"),
            Output("sqs-dlq-value", "children"),
            Output("asg-desired-value", "children"),
            Output("timeseries-chart", "figure"),
            Output("instance-table", "children"),
            Output("event-table", "children"),
            Output("last-updated", "children"),
        ],
        [Input("refresh", "n_intervals"), Input("time-range", "value")],
    )
    def update_dashboard(n_intervals, time_range_hours):
        now = datetime.now(timezone.utc)
        ts = now.strftime("%H:%M:%S UTC")

        try:
            outputs = _get_stack_outputs(cfg)
            queue = _get_queue_stats(cfg, outputs)
            instances = _get_asg_instances(cfg, outputs)
            desired = _get_asg_desired(cfg, outputs)
            instance_ids = [i["instance_id"] for i in instances]
            heartbeats = _get_heartbeats(cfg, instance_ids)
            cw_data = _get_cloudwatch_timeseries(cfg, outputs, hours=float(time_range_hours or 1))
            asg_events = _get_asg_events(cfg, outputs, max_events=20)
        except Exception as e:
            empty_fig = go.Figure()
            empty_fig.update_layout(template="plotly_dark", paper_bgcolor=BG_PAGE, plot_bgcolor=BG_CARD)
            return "?", "?", "?", "?", empty_fig, html.P(f"Error: {e}"), "", f"Error at {ts}"

        # ── Instance table ────────────────────────────────
        rows = []
        for inst in instances:
            iid = inst["instance_id"]
            hb = heartbeats.get(iid, {})
            phase = hb.get("phase", "n/a")
            age = hb.get("age_s", -1)
            hb_status = hb.get("status", "?")
            age_str = f"{age}s" if age >= 0 else "n/a"

            rows.append({
                "Instance": iid,
                "State": inst["state"],
                "Type": inst["type"],
                "AZ": inst["az"],
                "IP": inst["ip"],
                "Phase": phase,
                "Heartbeat": f"{hb_status} ({age_str})",
            })

        if rows:
            table = dash_table.DataTable(
                data=rows,
                columns=[{"name": c, "id": c} for c in rows[0].keys()],
                style_header={"backgroundColor": BG_TABLE_HEAD, "color": TEXT_PRIMARY, "fontWeight": "bold", "border": f"1px solid {BORDER}"},
                style_cell={"backgroundColor": BG_CARD, "color": TEXT_PRIMARY, "border": f"1px solid {BORDER}", "fontSize": "13px", "padding": "8px"},
                style_data_conditional=[
                    {"if": {"filter_query": '{Heartbeat} contains "STALE"'}, "backgroundColor": ROW_STALE, "color": ACCENT_RED},
                    {"if": {"filter_query": '{Heartbeat} contains "OK"'}, "backgroundColor": ROW_OK, "color": ACCENT_GREEN},
                    {"if": {"filter_query": '{State} eq "terminated"'}, "color": TEXT_MUTED, "backgroundColor": ROW_TERM},
                ],
            )
        else:
            table = html.P("No workers running", style={"color": TEXT_MUTED, "padding": "20px"})

        # ── Time series chart (from CloudWatch, survives page refresh) ─
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cw_data["timestamps"], y=cw_data["visible"],
            name="Queue Visible", mode="lines",
            line={"color": ACCENT_RED, "width": 2},
            fill="tozeroy", fillcolor="rgba(248,113,113,0.1)",
        ))
        fig.add_trace(go.Scatter(
            x=cw_data["timestamps"], y=cw_data["inflight"],
            name="In Flight", mode="lines",
            line={"color": ACCENT_VIOLET, "width": 2},
        ))
        fig.add_trace(go.Scatter(
            x=cw_data["timestamps"], y=cw_data["dlq"],
            name="Dead Letter", mode="lines",
            line={"color": ACCENT_AMBER, "width": 2, "dash": "dot"},
        ))
        fig.add_trace(go.Scatter(
            x=cw_data["timestamps"], y=cw_data["workers"],
            name="ASG Workers", mode="lines",
            line={"color": ACCENT_GREEN, "width": 3},
            yaxis="y2",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=BG_PAGE,
            plot_bgcolor=BG_CARD,
            font={"color": TEXT_PRIMARY},
            margin={"l": 50, "r": 50, "t": 30, "b": 40},
            legend={"orientation": "h", "y": 1.12, "font": {"color": TEXT_SECONDARY}},
            xaxis={"title": None, "gridcolor": BORDER, "tickfont": {"color": TEXT_MUTED}},
            yaxis={"title": "Messages", "gridcolor": BORDER, "rangemode": "tozero", "tickfont": {"color": TEXT_MUTED}},
            yaxis2={"title": "Workers", "overlaying": "y", "side": "right", "gridcolor": BORDER, "rangemode": "tozero", "tickfont": {"color": TEXT_MUTED}},
        )

        # ── Scaling events table (from ASG describe-scaling-activities) ─
        if asg_events:
            event_table = dash_table.DataTable(
                data=asg_events,
                columns=[
                    {"name": "Time", "id": "time"},
                    {"name": "Event", "id": "type"},
                    {"name": "Instance", "id": "instance"},
                    {"name": "Capacity", "id": "capacity"},
                    {"name": "Status", "id": "status"},
                ],
                style_header={"backgroundColor": BG_TABLE_HEAD, "color": TEXT_PRIMARY, "fontWeight": "bold", "border": f"1px solid {BORDER}"},
                style_cell={"backgroundColor": BG_CARD, "color": TEXT_PRIMARY, "border": f"1px solid {BORDER}", "fontSize": "13px", "padding": "6px"},
                style_data_conditional=[
                    {"if": {"filter_query": '{type} eq "SPOT RECLAIM"'}, "backgroundColor": ROW_STALE, "color": ACCENT_RED},
                    {"if": {"filter_query": '{type} eq "SCALE OUT"'}, "backgroundColor": ROW_OK, "color": ACCENT_GREEN},
                    {"if": {"filter_query": '{type} eq "SCALE IN"'}, "backgroundColor": ROW_WARN, "color": ACCENT_AMBER},
                    {"if": {"filter_query": '{type} eq "UNHEALTHY"'}, "backgroundColor": ROW_STALE, "color": ACCENT_RED},
                ],
                page_size=10,
            )
        else:
            event_table = html.P("No scaling events", style={"color": TEXT_MUTED, "padding": "20px"})

        return (
            str(queue["visible"]),
            str(queue["in_flight"]),
            str(queue["dlq"]),
            str(desired),
            fig,
            table,
            event_table,
            f"Last updated: {ts} (every {REFRESH_INTERVAL_MS // 1000}s)",
        )

    return app


def _metric_card(id_prefix: str, label: str, accent: str, bg: str, label_color: str) -> html.Div:
    """Create a metric card with a large number and label. All colors WCAG AA compliant."""
    return html.Div(
        style={
            "backgroundColor": bg,
            "borderRadius": "8px",
            "padding": "20px",
            "textAlign": "center",
            "minWidth": "150px",
            "borderLeft": f"4px solid {accent}",
        },
        children=[
            html.Div(id=f"{id_prefix}-value", style={"fontSize": "36px", "fontWeight": "bold", "color": accent}, children="..."),
            html.Div(label, style={"fontSize": "12px", "color": label_color, "marginTop": "5px"}),
        ],
    )


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    app = create_app()
    log.info("Dashboard: http://localhost:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
