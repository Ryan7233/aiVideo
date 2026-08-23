"""Retention sweep for generated media and job records.

Nothing ever cleaned up after a run: every clip, every downloaded source and
every collage stayed in ``output_data`` forever, and job rows accumulated
alongside them. This sweeps files past their age limit, staying strictly inside
the managed runtime directories.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

from core import job_store
from core.runtime import (
    DOWNLOAD_DIR,
    OUTPUT_DIR,
    PHOTO_UPLOAD_DIR,
    VIDEO_UPLOAD_DIR,
    _is_within,
)

logger = logging.getLogger(__name__)


def _days(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        logger.warning("%s is not a number; using %s", name, default)
        return default


def output_retention_days() -> float:
    return _days("OUTPUT_RETENTION_DAYS", 7)


def upload_retention_days() -> float:
    return _days("UPLOAD_RETENTION_DAYS", 30)


def job_retention_days() -> float:
    return _days("JOB_RETENTION_DAYS", 30)


def sweep_interval_hours() -> float:
    return _days("RETENTION_SWEEP_INTERVAL_HOURS", 6)


@dataclass
class SweepReport:
    removed_files: int = 0
    freed_bytes: int = 0
    removed_jobs: int = 0
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "removed_files": self.removed_files,
            "freed_bytes": self.freed_bytes,
            "freed_mb": round(self.freed_bytes / (1024 * 1024), 2),
            "removed_jobs": self.removed_jobs,
            "errors": self.errors,
        }


def _sweep_directory(
    directory: Path,
    max_age_days: float,
    report: SweepReport,
    *,
    recursive: bool,
    skip: Iterable[Path] = (),
) -> None:
    """Delete files under ``directory`` older than ``max_age_days``."""
    if max_age_days <= 0 or not directory.is_dir():
        return

    cutoff = time.time() - max_age_days * 86400
    skip_roots = [item.resolve() for item in skip]
    candidates = directory.rglob("*") if recursive else directory.glob("*")

    for path in candidates:
        try:
            if not path.is_file():
                continue
            resolved = path.resolve()
            # Never step outside the managed tree, and never touch a directory
            # that has its own retention rule (uploads live under output_data).
            if not _is_within(resolved, (directory,)):
                continue
            if any(_is_within(resolved, (root,)) for root in skip_roots):
                continue
            if path.stat().st_mtime >= cutoff:
                continue
            size = path.stat().st_size
            path.unlink()
            report.removed_files += 1
            report.freed_bytes += size
        except FileNotFoundError:
            continue  # raced with another sweep or a job cleaning up
        except OSError as exc:
            report.errors.append(f"{path}: {exc}")


def run_sweep() -> Dict[str, object]:
    """Apply every retention rule once and report what was reclaimed."""
    report = SweepReport()

    # output_data holds generated artifacts, but PHOTO_UPLOAD_DIR sits inside
    # it and keeps its own, longer window.
    _sweep_directory(
        OUTPUT_DIR,
        output_retention_days(),
        report,
        recursive=True,
        skip=(PHOTO_UPLOAD_DIR,),
    )
    _sweep_directory(PHOTO_UPLOAD_DIR, upload_retention_days(), report, recursive=True)
    _sweep_directory(VIDEO_UPLOAD_DIR, upload_retention_days(), report, recursive=True)
    _sweep_directory(DOWNLOAD_DIR, output_retention_days(), report, recursive=True)

    # 0 disables the rule, exactly as it does for the file sweeps above.
    job_days = job_retention_days()
    if job_days > 0:
        try:
            report.removed_jobs = job_store.delete_jobs_older_than(job_days)
        except Exception as exc:
            report.errors.append(f"job store: {exc}")

    summary = report.as_dict()
    if report.removed_files or report.removed_jobs:
        logger.info("Retention sweep reclaimed %s", summary)
    return summary
