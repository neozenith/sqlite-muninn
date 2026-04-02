"""Lambda: prune stale muninn benchmark AMIs and their backing snapshots.

Triggered weekly by EventBridge, or invoked manually:
    aws lambda invoke --function-name MuninnAmiCleanup /dev/stdout

Environment variables:
    MAX_AGE_DAYS  — AMIs older than this are pruned (default: 7)
    PROJECT_TAG   — only prune AMIs with this project tag value (default: muninn-benchmarks)
"""

import os
from datetime import UTC, datetime, timedelta

import boto3


def handler(event, context):
    """Prune AMIs older than MAX_AGE_DAYS with the project tag."""
    ec2 = boto3.client("ec2")
    max_age_days = int(os.environ.get("MAX_AGE_DAYS", "7"))
    project_tag = os.environ.get("PROJECT_TAG", "muninn-benchmarks")

    cutoff = datetime.now(UTC) - timedelta(days=max_age_days)

    images = ec2.describe_images(
        Owners=["self"],
        Filters=[{"Name": "tag:project", "Values": [project_tag]}],
    )

    pruned = 0
    kept = 0

    for img in images["Images"]:
        ami_id = img["ImageId"]
        name = img.get("Name", "unnamed")
        created = datetime.fromisoformat(img["CreationDate"].replace("Z", "+00:00"))
        age_days = (datetime.now(UTC) - created).days

        if created >= cutoff:
            print(f"KEEP  {ami_id} ({name}) — {age_days}d old")
            kept += 1
            continue

        # Collect snapshot IDs before deregistering
        snap_ids = [
            bdm["Ebs"]["SnapshotId"]
            for bdm in img.get("BlockDeviceMappings", [])
            if "Ebs" in bdm and "SnapshotId" in bdm["Ebs"]
        ]

        print(f"PRUNE {ami_id} ({name}) — {age_days}d old, {len(snap_ids)} snapshot(s)")

        ec2.deregister_image(ImageId=ami_id)
        for snap_id in snap_ids:
            try:
                ec2.delete_snapshot(SnapshotId=snap_id)
                print(f"  deleted snapshot {snap_id}")
            except Exception as e:
                print(f"  failed to delete snapshot {snap_id}: {e}")

        pruned += 1

    summary = f"Pruned {pruned}, kept {kept} (threshold: {max_age_days}d)"
    print(summary)
    return {"pruned": pruned, "kept": kept, "summary": summary}
