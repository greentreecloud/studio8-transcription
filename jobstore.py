"""
jobstore.py
───────────
Shared SQLite job queue.
Used by: app.py (submit + read), watchfolder.py (submit),
         worker.py (claim + update).
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(os.environ.get("OUTPUT_DIR", "./output")) / "jobstore.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(
        str(DB_PATH),
        check_same_thread=False,
        timeout=10,
    )
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


def _get_priority(source: str) -> int:
    try:
        from settings import cfg

        if source == "upload":
            return cfg.priority.manual_upload
        return cfg.priority.watchfolder_default
    except Exception:
        return 10 if source == "upload" else 5


def _default_speakers_json() -> bool:
    try:
        from settings import cfg

        return bool(cfg.output.speakers_json_default)
    except Exception:
        return False


def init_db():
    with _connect() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id             TEXT PRIMARY KEY,
                source         TEXT NOT NULL DEFAULT 'upload',
                filename       TEXT NOT NULL,
                filepath       TEXT NOT NULL,
                status         TEXT NOT NULL DEFAULT 'queued',
                priority       INTEGER NOT NULL DEFAULT 5,
                mode           TEXT NOT NULL DEFAULT 'single',
                output_dir     TEXT DEFAULT '',
                output_dirs    TEXT DEFAULT '',
                language       TEXT DEFAULT '',
                speakers_json  INTEGER NOT NULL DEFAULT 0,
                retry_count    INTEGER NOT NULL DEFAULT 0,
                step           INTEGER DEFAULT 0,
                total          INTEGER DEFAULT 5,
                message        TEXT DEFAULT '',
                error          TEXT DEFAULT '',
                output         TEXT DEFAULT '',
                created_at     TEXT NOT NULL,
                updated_at     TEXT NOT NULL
            )
            """
        )

        # Safe in-place migrations for existing installations.
        for col, definition in [
            ("priority", "INTEGER NOT NULL DEFAULT 5"),
            ("mode", "TEXT NOT NULL DEFAULT 'single'"),
            ("output_dir", "TEXT DEFAULT ''"),
            ("output_dirs", "TEXT DEFAULT ''"),
            ("language", "TEXT DEFAULT ''"),
            ("speakers_json", "INTEGER NOT NULL DEFAULT 0"),
            ("retry_count", "INTEGER NOT NULL DEFAULT 0"),
        ]:
            try:
                con.execute(
                    f"ALTER TABLE jobs ADD COLUMN {col} {definition}"
                )
            except sqlite3.OperationalError:
                # Column already exists.
                pass

        con.commit()


def submit_job(
    job_id: str,
    filename: str,
    filepath: str,
    source: str = "upload",
    priority: int | None = None,
    mode: str = "single",
    output_dir: str = "",
    output_dirs: list | None = None,
    language: str | None = None,
    speakers_json: bool | None = None,
):
    """
    Add a new job to the queue.

    `speakers_json=None` means: use the global config default.
    Watchfolder workflows normally pass their explicit per-workflow value.
    """
    now = datetime.now().isoformat(timespec="seconds")

    if priority is None:
        priority = _get_priority(source)

    if speakers_json is None:
        speakers_json = _default_speakers_json()

    with _connect() as con:
        con.execute(
            """
            INSERT INTO jobs
                (
                    id,
                    source,
                    filename,
                    filepath,
                    status,
                    priority,
                    mode,
                    output_dir,
                    output_dirs,
                    language,
                    speakers_json,
                    created_at,
                    updated_at
                )
            VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                source,
                filename,
                filepath,
                priority,
                mode,
                output_dir,
                json.dumps(output_dirs or []),
                language or "",
                1 if speakers_json else 0,
                now,
                now,
            ),
        )
        con.commit()


def claim_next_job() -> dict | None:
    """
    Atomically claim the next queued job.
    Highest priority first; FIFO within equal priority.
    """
    with _connect() as con:
        row = con.execute(
            """
            SELECT *
            FROM jobs
            WHERE status = 'queued'
            ORDER BY priority DESC, created_at ASC
            LIMIT 1
            """
        ).fetchone()

        if not row:
            return None

        now = datetime.now().isoformat(timespec="seconds")
        con.execute(
            """
            UPDATE jobs
            SET status='running', updated_at=?
            WHERE id=?
            """,
            (now, row["id"]),
        )
        con.commit()

    return dict(row)


def update_job(
    job_id: str,
    status: str,
    step: int = 0,
    total: int = 5,
    message: str = "",
    error: str = "",
    output: str = "",
):
    now = datetime.now().isoformat(timespec="seconds")
    with _connect() as con:
        con.execute(
            """
            UPDATE jobs
            SET
                status=?,
                step=?,
                total=?,
                message=?,
                error=?,
                output=?,
                updated_at=?
            WHERE id=?
            """,
            (
                status,
                step,
                total,
                message,
                error,
                output,
                now,
                job_id,
            ),
        )
        con.commit()


def get_job(job_id: str) -> dict | None:
    with _connect() as con:
        row = con.execute(
            "SELECT * FROM jobs WHERE id=?",
            (job_id,),
        ).fetchone()
    return dict(row) if row else None


def get_jobs(
    limit: int = 100,
    source: str | None = None,
) -> list[dict]:
    with _connect() as con:
        if source:
            rows = con.execute(
                """
                SELECT *
                FROM jobs
                WHERE source=?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (source, limit),
            ).fetchall()
        else:
            rows = con.execute(
                """
                SELECT *
                FROM jobs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    return [dict(row) for row in rows]


def delete_job(job_id: str) -> bool:
    with _connect() as con:
        cur = con.execute(
            "DELETE FROM jobs WHERE id=?",
            (job_id,),
        )
        con.commit()
    return cur.rowcount > 0


def delete_jobs(status: str | None = None) -> int:
    with _connect() as con:
        if status:
            cur = con.execute(
                """
                DELETE FROM jobs
                WHERE status=? AND status != 'running'
                """,
                (status,),
            )
        else:
            cur = con.execute(
                """
                DELETE FROM jobs
                WHERE status != 'running'
                """
            )
        con.commit()
    return cur.rowcount


def retry_job(
    job_id: str,
    max_retries: int = 3,
) -> tuple[bool, str]:
    job = get_job(job_id)

    if not job:
        return False, "Job not found"

    if job["status"] != "error":
        return (
            False,
            f"Job is not in error state (status: {job['status']})",
        )

    retry_count = job.get("retry_count", 0)
    if retry_count >= max_retries:
        return False, f"Maximum retries ({max_retries}) reached"

    now = datetime.now().isoformat(timespec="seconds")
    with _connect() as con:
        con.execute(
            """
            UPDATE jobs
            SET
                status='queued',
                step=0,
                message='Retry attempt',
                error='',
                retry_count=retry_count+1,
                updated_at=?
            WHERE id=?
            """,
            (now, job_id),
        )
        con.commit()

    return (
        True,
        f"Retry {retry_count + 1}/{max_retries} queued",
    )


def reset_stale_jobs():
    now = datetime.now().isoformat(timespec="seconds")
    with _connect() as con:
        con.execute(
            """
            UPDATE jobs
            SET
                status='queued',
                step=0,
                message='Requeued after restart',
                updated_at=?
            WHERE status='running'
            """,
            (now,),
        )
        con.commit()
