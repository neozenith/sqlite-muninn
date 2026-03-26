"""Base PrepTask ABC for all data preparation steps.

A PrepTask represents one unit of data preparation (download, transform, cache).
The harness calls run() which checks cache status, then delegates to fetch() + transform().

Subclasses MUST implement:
    task_id   — unique identifier (e.g., "text:3300", "vector:MiniLM:ag_news")
    label     — human-readable description
    outputs() — list of file paths this task produces
    fetch()   — download or generate source data

Subclasses MAY override:
    transform() — post-fetch transformation (default: no-op)
    status()    — cache status check (default: checks all outputs() exist)
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

log = logging.getLogger(__name__)


class PrepTask(ABC):
    """Base class for all data preparation tasks."""

    @property
    @abstractmethod
    def task_id(self) -> str:
        """Unique identifier for this prep task (e.g., 'text:3300')."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Human-readable description for status display."""

    @abstractmethod
    def outputs(self) -> list[Path]:
        """File paths this task produces. Used by status() to check readiness."""

    @abstractmethod
    def fetch(self, force: bool = False) -> None:
        """Download or generate source data."""

    def transform(self) -> None:  # noqa: B027
        """Post-fetch transformation. Override if needed; default is no-op."""

    def status(self) -> str:
        """Check whether all outputs exist. Returns 'READY' or 'MISSING'."""
        if all(p.exists() for p in self.outputs()):
            return "READY"
        return "MISSING"

    def run(self, force: bool = False) -> None:
        """Execute the full prep lifecycle: check cache → fetch → transform.

        When --s3-bucket is configured:
        - Before fetch: tries to download missing outputs from S3
        - After fetch+transform: uploads generated outputs to S3
        """
        from benchmarks.harness.s3_mirror import get_s3_mirror

        mirror = get_s3_mirror()

        if not force:
            # Try S3 download for any missing outputs before checking cache
            for p in self.outputs():
                mirror.ensure_local(p)
            if self.status() == "READY":
                log.info("  %s: cached (skip)", self.task_id)
                return

        self.fetch(force=force)
        self.transform()

        # Upload generated outputs to S3
        for p in self.outputs():
            mirror.sync_to_s3(p)
