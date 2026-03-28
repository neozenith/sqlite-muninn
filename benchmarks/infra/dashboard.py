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
from datetime import datetime, timezone
from pathlib import Path

import boto3
import dash
import yaml
from botocore.exceptions import ClientError
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "benchmarks" / "infra" / "config.yml"

REFRESH_INTERVAL_MS = 15_000  # 15 seconds


# ── Config ────────────────────────────────────────────────────────


def _load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sanitize_branch(branch: str) -> str:
    import re

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
        style={"fontFamily": "monospace", "padding": "20px", "backgroundColor": "#1a1a2e", "color": "#eee", "minHeight": "100vh"},
        children=[
            html.H1(f"Muninn Benchmark Dashboard", style={"color": "#e94560"}),
            html.P(f"Branch: {branch}", style={"color": "#aaa", "fontSize": "14px"}),

            dcc.Interval(id="refresh", interval=REFRESH_INTERVAL_MS, n_intervals=0),

            html.Div(id="last-updated", style={"color": "#666", "fontSize": "12px", "marginBottom": "20px"}),

            # ── Queue Stats ───────────────────────────────────
            html.Div(
                style={"display": "flex", "gap": "20px", "marginBottom": "30px"},
                children=[
                    _metric_card("sqs-visible", "Queue Visible", "#e94560"),
                    _metric_card("sqs-inflight", "In Flight", "#533483"),
                    _metric_card("sqs-dlq", "Dead Letter", "#7a1533"),
                    _metric_card("asg-desired", "ASG Workers", "#2d4a22"),
                ],
            ),

            # ── Instance Table ────────────────────────────────
            html.H3("Workers", style={"color": "#e94560", "marginBottom": "10px"}),
            html.Div(id="instance-table"),

            # ── Event Log ─────────────────────────────────────
            html.H3("Event Log", style={"color": "#e94560", "marginTop": "30px", "marginBottom": "10px"}),
            html.Div(
                id="event-log",
                style={
                    "backgroundColor": "#0f3460",
                    "padding": "15px",
                    "borderRadius": "8px",
                    "maxHeight": "300px",
                    "overflowY": "auto",
                    "fontSize": "13px",
                    "whiteSpace": "pre-wrap",
                },
            ),

            # Hidden store for event history
            dcc.Store(id="event-store", data=[]),
        ],
    )

    @app.callback(
        [
            Output("sqs-visible-value", "children"),
            Output("sqs-inflight-value", "children"),
            Output("sqs-dlq-value", "children"),
            Output("asg-desired-value", "children"),
            Output("instance-table", "children"),
            Output("event-log", "children"),
            Output("event-store", "data"),
            Output("last-updated", "children"),
        ],
        [Input("refresh", "n_intervals")],
        [dash.State("event-store", "data")],
    )
    def update_dashboard(n_intervals, events):
        now = datetime.now(timezone.utc)
        ts = now.strftime("%H:%M:%S UTC")

        try:
            outputs = _get_stack_outputs(cfg)
            queue = _get_queue_stats(cfg, outputs)
            instances = _get_asg_instances(cfg, outputs)
            desired = _get_asg_desired(cfg, outputs)
            instance_ids = [i["instance_id"] for i in instances]
            heartbeats = _get_heartbeats(cfg, instance_ids)
        except Exception as e:
            return "?", "?", "?", "?", html.P(f"Error: {e}"), "", events, f"Error at {ts}"

        # Build instance table rows
        rows = []
        for inst in instances:
            iid = inst["instance_id"]
            hb = heartbeats.get(iid, {})
            phase = hb.get("phase", "n/a")
            age = hb.get("age_s", -1)
            hb_status = hb.get("status", "?")

            # Track phase changes in event log
            event_key = f"{iid}:{phase}"
            if events and not any(e.get("key") == event_key for e in events[-20:]):
                events.append({"key": event_key, "time": ts, "instance": iid[:12], "phase": phase})

            age_str = f"{age}s" if age >= 0 else "n/a"
            status_color = "#4ade80" if hb_status == "OK" else "#ef4444" if hb_status == "STALE" else "#666"

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
                style_header={"backgroundColor": "#16213e", "color": "#eee", "fontWeight": "bold"},
                style_cell={"backgroundColor": "#0f3460", "color": "#eee", "border": "1px solid #16213e", "fontSize": "13px", "padding": "8px"},
                style_data_conditional=[
                    {"if": {"filter_query": '{Heartbeat} contains "STALE"'}, "backgroundColor": "#7a1533"},
                    {"if": {"filter_query": '{Heartbeat} contains "OK"'}, "backgroundColor": "#1a3a1a"},
                    {"if": {"filter_query": '{State} eq "terminated"'}, "color": "#666"},
                ],
            )
        else:
            table = html.P("No workers running", style={"color": "#666", "padding": "20px"})

        # Format event log (newest first)
        event_lines = []
        for e in reversed(events[-30:]):
            event_lines.append(f"[{e['time']}] {e['instance']} -> {e['phase']}")
        event_text = "\n".join(event_lines) if event_lines else "No events yet"

        return (
            str(queue["visible"]),
            str(queue["in_flight"]),
            str(queue["dlq"]),
            str(desired),
            table,
            event_text,
            events[-50:],  # keep last 50 events
            f"Last updated: {ts} (every {REFRESH_INTERVAL_MS // 1000}s)",
        )

    return app


def _metric_card(id_prefix: str, label: str, color: str) -> html.Div:
    """Create a metric card with a large number and label."""
    return html.Div(
        style={
            "backgroundColor": "#0f3460",
            "borderRadius": "8px",
            "padding": "20px",
            "textAlign": "center",
            "minWidth": "150px",
            "borderLeft": f"4px solid {color}",
        },
        children=[
            html.Div(id=f"{id_prefix}-value", style={"fontSize": "36px", "fontWeight": "bold", "color": color}, children="..."),
            html.Div(label, style={"fontSize": "12px", "color": "#aaa", "marginTop": "5px"}),
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
