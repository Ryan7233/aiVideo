"""Durable job records.

Until now nothing about a run was persisted: the API held every result in the
request that produced it, so a restart, a timeout or a dropped connection lost
the work outright. This is a small SQLite table -- no ORM, no migrations
framework -- that records what was asked for, what happened, and where the
output landed.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from core.runtime import DATA_ROOT

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("AIVIDEO_DB_PATH", str(DATA_ROOT / "aivideo.db")))

PENDING = "pending"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"

TERMINAL_STATES = frozenset({SUCCEEDED, FAILED, CANCELLED})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    status       TEXT NOT NULL,
    params       TEXT NOT NULL,
    result       TEXT,
    error        TEXT,
    progress     TEXT,
    backend      TEXT,
    external_id  TEXT,
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""

_init_lock = threading.Lock()
_initialised = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """One short-lived connection per operation; WAL keeps readers unblocked."""
    _ensure_schema()
    connection = sqlite3.connect(str(DB_PATH), timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _ensure_schema() -> None:
    global _initialised
    if _initialised:
        return
    with _init_lock:
        if _initialised:
            return
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(DB_PATH), timeout=30)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA)
            connection.commit()
        finally:
            connection.close()
        _initialised = True


def reset_for_tests(path: Optional[Path] = None) -> None:
    """Point the store at a different file and rebuild the schema."""
    global DB_PATH, _initialised
    if path is not None:
        DB_PATH = Path(path)
    _initialised = False
    _ensure_schema()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    job = dict(row)
    for field in ("params", "result", "progress"):
        raw = job.get(field)
        job[field] = json.loads(raw) if raw else None
    return job


def create_job(kind: str, params: Dict[str, Any], backend: str = "thread") -> str:
    job_id = uuid.uuid4().hex
    with _connect() as connection:
        connection.execute(
            "INSERT INTO jobs (id, kind, status, params, backend, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, kind, PENDING, json.dumps(params, ensure_ascii=False), backend, _now()),
        )
    logger.info("Created job %s (%s)", job_id, kind)
    return job_id


def mark_running(job_id: str, external_id: Optional[str] = None) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE jobs SET status = ?, started_at = COALESCE(started_at, ?),"
            " external_id = COALESCE(?, external_id) WHERE id = ?",
            (RUNNING, _now(), external_id, job_id),
        )


def mark_succeeded(job_id: str, result: Dict[str, Any]) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE jobs SET status = ?, result = ?, finished_at = ? WHERE id = ?",
            (SUCCEEDED, json.dumps(result, ensure_ascii=False, default=str), _now(), job_id),
        )


def mark_failed(job_id: str, error: str) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
            (FAILED, error[:4000], _now(), job_id),
        )
    logger.warning("Job %s failed: %s", job_id, error[:400])


def mark_cancelled(job_id: str) -> bool:
    """Cancel a job that has not reached a terminal state yet."""
    with _connect() as connection:
        cursor = connection.execute(
            "UPDATE jobs SET status = ?, finished_at = ? WHERE id = ? AND status IN (?, ?)",
            (CANCELLED, _now(), job_id, PENDING, RUNNING),
        )
        return cursor.rowcount > 0


def set_progress(job_id: str, progress: Dict[str, Any]) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE jobs SET progress = ? WHERE id = ?",
            (json.dumps(progress, ensure_ascii=False, default=str), job_id),
        )


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as connection:
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_dict(row) if row else None


def list_jobs(limit: int = 50, kind: Optional[str] = None,
              status: Optional[str] = None) -> List[Dict[str, Any]]:
    query = "SELECT * FROM jobs"
    clauses, args = [], []
    if kind:
        clauses.append("kind = ?")
        args.append(kind)
    if status:
        clauses.append("status = ?")
        args.append(status)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
    args.append(max(1, min(limit, 200)))
    with _connect() as connection:
        rows = connection.execute(query, args).fetchall()
    return [_row_to_dict(row) for row in rows]


def is_cancelled(job_id: str) -> bool:
    job = get_job(job_id)
    return bool(job and job["status"] == CANCELLED)


def delete_jobs_older_than(days: int) -> int:
    """Drop finished job records past the retention window."""
    cutoff = datetime.fromtimestamp(time.time() - days * 86400, tz=timezone.utc).isoformat()
    with _connect() as connection:
        cursor = connection.execute(
            "DELETE FROM jobs WHERE finished_at IS NOT NULL AND finished_at < ?", (cutoff,)
        )
        return cursor.rowcount
