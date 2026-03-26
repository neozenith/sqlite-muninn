"""S3 mirror for additive-only benchmark data synchronization.

When enabled via --s3-bucket, provides transparent sync between local
benchmark directories and S3 under a configurable prefix.

Design principles:
- Additive only: never deletes. Missing file = "don't have it", not "delete it".
- Selective: only syncs files needed by the current action.
- Listing = union: list operations check both local and S3 without downloading.
- Size-based comparison: compares file sizes to detect out-of-sync files.

boto3 is imported lazily (inside the client property) so this module can be
imported freely without requiring AWS credentials or the boto3 package until
S3 operations are actually performed.
"""

import fnmatch
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Project root derived independently to avoid circular imports with common.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class S3Mirror:
    """Additive-only S3 mirror for benchmark harness data.

    Maps local paths to S3 keys via:
        local: {PROJECT_ROOT}/benchmarks/vectors/X.npy
        s3:    s3://{bucket}/{prefix}/benchmarks/vectors/X.npy

        local: {PROJECT_ROOT}/models/X.gguf
        s3:    s3://{bucket}/{prefix}/models/X.gguf
    """

    def __init__(self, bucket: str | None, prefix: str = "prep"):
        self.bucket = bucket
        self.prefix = prefix
        self._client = None
        self._listing_cache: dict[str, list[dict]] = {}

    @property
    def enabled(self) -> bool:
        """True if an S3 bucket is configured."""
        return self.bucket is not None

    @property
    def client(self):
        """Lazy-initialized boto3 S3 client."""
        if self._client is None:
            import boto3

            self._client = boto3.client("s3")
        return self._client

    def _local_to_s3_key(self, local_path: Path) -> str:
        """Convert a local filesystem path to its S3 key."""
        rel = local_path.resolve().relative_to(_PROJECT_ROOT)
        return f"{self.prefix}/{rel.as_posix()}"

    def _s3_key_to_local(self, s3_key: str) -> Path:
        """Convert an S3 key back to its local filesystem path."""
        rel = s3_key[len(self.prefix) + 1 :]  # strip "{prefix}/"
        return _PROJECT_ROOT / rel

    @staticmethod
    def _is_not_found(error: Exception) -> bool:
        """Check if a boto3 error is a 404 (key not found)."""
        resp = getattr(error, "response", None)
        if resp is None:
            return False
        return resp.get("Error", {}).get("Code") == "404"

    def ensure_local(self, path: Path) -> bool:
        """Ensure a file exists locally, downloading from S3 if needed.

        Returns True if the file is available locally after the call.
        Returns False if the file does not exist locally or in S3.
        """
        if path.exists():
            return True
        if not self.enabled:
            return False

        s3_key = self._local_to_s3_key(path)

        # Check existence in S3
        try:
            head = self.client.head_object(Bucket=self.bucket, Key=s3_key)
        except Exception as e:
            if self._is_not_found(e):
                return False
            raise

        s3_size = head["ContentLength"]
        size_mb = s3_size / (1024 * 1024)
        log.info("S3 ↓ downloading %s (%.1f MB) ...", path.name, size_mb)

        # Download to temp file, then atomic rename
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".s3tmp")
        try:
            self.client.download_file(self.bucket, s3_key, str(tmp_path))
            tmp_path.rename(path)
            log.info("S3 ↓ %s complete", path.name)
            return True
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def sync_to_s3(self, path: Path) -> None:
        """Upload a local file to S3 if not already synced (compared by size).

        No-op if the mirror is disabled, the file doesn't exist locally,
        or the S3 object already has the same size.
        """
        if not self.enabled or not path.exists():
            return

        s3_key = self._local_to_s3_key(path)
        local_size = path.stat().st_size

        # Check if S3 already has it at the same size
        try:
            resp = self.client.head_object(Bucket=self.bucket, Key=s3_key)
            if resp["ContentLength"] == local_size:
                log.debug("S3 = %s (size match, skip)", path.name)
                return
            log.info(
                "S3 ↑ %s (size changed: %d → %d)",
                path.name,
                resp["ContentLength"],
                local_size,
            )
        except Exception as e:
            if self._is_not_found(e):
                size_mb = local_size / (1024 * 1024)
                log.info("S3 ↑ %s (new, %.1f MB)", path.name, size_mb)
            else:
                raise

        self.client.upload_file(str(path), self.bucket, s3_key)
        log.info("S3 ↑ %s complete", path.name)

        # Invalidate listing cache for the parent prefix
        parent_prefix = self._local_to_s3_key(path.parent) + "/"
        self._listing_cache.pop(parent_prefix, None)

    def list_union(self, local_dir: Path, pattern: str) -> list[Path]:
        """Return sorted union of local and S3 files matching a glob pattern.

        Listing does NOT trigger downloads. S3-only files are returned as
        Path objects pointing to where they WOULD exist locally. Call
        ensure_local() before reading any S3-only paths.
        """
        local_files: set[Path] = set()
        if local_dir.exists():
            local_files = set(local_dir.glob(pattern))

        if not self.enabled:
            return sorted(local_files)

        s3_prefix = self._local_to_s3_key(local_dir) + "/"
        s3_objects = self._list_s3_prefix(s3_prefix)

        for obj in s3_objects:
            key = obj["Key"]
            # Only match direct children (skip nested subdirectories)
            rel_key = key[len(s3_prefix) :]
            if "/" in rel_key:
                continue
            if fnmatch.fnmatch(rel_key, pattern):
                local_path = self._s3_key_to_local(key)
                local_files.add(local_path)

        return sorted(local_files)

    def _list_s3_prefix(self, prefix: str) -> list[dict]:
        """List S3 objects under a prefix (cached per session to minimize API calls)."""
        if prefix in self._listing_cache:
            return self._listing_cache[prefix]

        objects: list[dict] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                objects.append({"Key": obj["Key"], "Size": obj["Size"]})

        self._listing_cache[prefix] = objects
        return objects


# ── Module-level singleton ─────────────────────────────────────────

_mirror: S3Mirror | None = None


def get_s3_mirror() -> S3Mirror:
    """Get the S3 mirror singleton (disabled by default)."""
    global _mirror
    if _mirror is None:
        _mirror = S3Mirror(bucket=None)
    return _mirror


def set_s3_bucket(bucket: str) -> None:
    """Configure the S3 mirror with a bucket name. Call once from CLI setup."""
    global _mirror
    _mirror = S3Mirror(bucket=bucket)
    log.info("S3 mirror enabled: s3://%s/%s/", bucket, _mirror.prefix)
