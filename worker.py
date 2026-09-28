"""
worker.py
─────────
Single GPU worker. Polls SQLite and processes one job at a time.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import time
from pathlib import Path

import torch

from jobstore import (
    claim_next_job,
    get_job,
    init_db,
    reset_stale_jobs,
    update_job,
)
from settings import cfg
from transcribe import batch_transcribe, transcribe


POLL_INTERVAL = int(
    os.environ.get(
        "WORKER_POLL",
        str(cfg.runtime.worker_poll),
    )
)
GPU_SETTLE = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WORKER] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


class JobCancelledError(Exception):
    pass


def is_cancelled(job_id: str) -> bool:
    return get_job(job_id) is None


def make_progress(job_id: str):
    def progress(step: int, total: int, msg: str):
        if is_cancelled(job_id):
            raise JobCancelledError(
                f"Job {job_id} was cancelled via GUI"
            )
        update_job(
            job_id,
            "running",
            step=step,
            total=total,
            message=msg,
        )

    return progress


def cleanup_gpu():
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass
    time.sleep(GPU_SETTLE)


def _speakers_json_path(transcript: Path) -> Path:
    stem = transcript.stem
    if stem.endswith("_transcript"):
        stem = stem[: -len("_transcript")]
    return transcript.with_name(
        f"{stem}_speakers.json"
    )


def _job_speakers_json(job: dict) -> bool:
    value = job.get("speakers_json", 0)
    if isinstance(value, str):
        return value.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
            "ja",
        }
    return bool(value)


def deliver(
    out: Path,
    job: dict,
    filepath: str,
):
    """
    Copy transcript and, when enabled/present, speakers JSON to all additional
    output destinations.
    """
    raw = job.get("output_dirs", "[]")
    try:
        extra_dirs = (
            json.loads(raw)
            if isinstance(raw, str)
            else (raw or [])
        )
    except Exception:
        extra_dirs = []

    artifacts = [out]
    if _job_speakers_json(job):
        sidecar = _speakers_json_path(out)
        if sidecar.exists():
            artifacts.append(sidecar)

    for dest_dir in extra_dirs:
        for artifact in artifacts:
            try:
                if dest_dir == "same_as_source":
                    source_path = Path(filepath)
                    if source_path.is_dir():
                        dest = source_path / artifact.name
                    else:
                        dest = (
                            source_path.parent
                            / artifact.name
                        )
                else:
                    dest = (
                        Path(dest_dir)
                        / artifact.name
                    )
                    dest.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                if dest.resolve() != artifact.resolve():
                    shutil.copy2(
                        str(artifact),
                        str(dest),
                    )
                    log.info(
                        "Delivered to: %s",
                        dest,
                    )

            except Exception as exc:
                log.error(
                    "Delivery failed for %s: %s",
                    dest_dir,
                    exc,
                )


def process(job: dict):
    job_id = job["id"]
    filename = job["filename"]
    filepath = job["filepath"]
    mode = job.get("mode", "single")
    output_dir = (
        job.get("output_dir")
        or cfg.runtime.output_dir
    )
    language = job.get("language") or None
    speakers_json = _job_speakers_json(job)

    log.info(
        "Starting [%s]: %s [%s] lang=%s speakers_json=%s",
        mode,
        filename,
        job_id,
        language or "auto",
        "yes" if speakers_json else "no",
    )

    cleanup_gpu()
    progress = make_progress(job_id)

    try:
        if mode == "batch":
            if not Path(filepath).is_dir():
                raise FileNotFoundError(
                    f"Batch folder not found: {filepath}"
                )

            if output_dir == "same_as_source":
                out_path = (
                    Path(filepath)
                    / f"{Path(filepath).name}_transcript.txt"
                )
            else:
                out_path = (
                    Path(output_dir)
                    / f"{Path(filepath).name}_transcript.txt"
                )

            out = batch_transcribe(
                folder=filepath,
                output_path=str(out_path),
                progress=progress,
                language=language,
                emit_speakers_json=speakers_json,
            )

        else:
            if not Path(filepath).exists():
                raise FileNotFoundError(
                    f"File not found: {filepath}"
                )

            if output_dir == "same_as_source":
                out_path = (
                    Path(filepath).parent
                    / f"{Path(filename).stem}_transcript.txt"
                )
            else:
                out_path = (
                    Path(output_dir)
                    / f"{Path(filename).stem}_transcript.txt"
                )

            out = transcribe(
                filepath,
                original_name=filename,
                output_path=str(out_path),
                progress=progress,
                language=language,
                emit_speakers_json=speakers_json,
            )

        if is_cancelled(job_id):
            raise JobCancelledError(
                f"Job {job_id} was cancelled via GUI"
            )

        deliver(out, job, filepath)

        update_job(
            job_id,
            "done",
            step=5,
            total=5,
            message="Transcript complete",
            output=str(out),
        )
        log.info("Done: %s", out.name)

    except JobCancelledError:
        log.info(
            "Cancelled: %s [%s] — cleaning up GPU memory",
            filename,
            job_id,
        )

    except Exception as exc:
        if not is_cancelled(job_id):
            update_job(
                job_id,
                "error",
                message="Failed",
                error=str(exc),
            )
        log.exception(
            "Error processing %s: %s",
            filename,
            exc,
        )

    finally:
        try:
            path = Path(filepath)
            if (
                "/staging/" in filepath
                and path.exists()
                and path.is_file()
            ):
                path.unlink()
                log.info(
                    "Removed staging file: %s",
                    path.name,
                )
        except Exception as exc:
            log.warning(
                "Could not remove staging file %s: %s",
                filepath,
                exc,
            )

        cleanup_gpu()
        log.info(
            "GPU memory released after: %s",
            filename,
        )


def run():
    init_db()
    reset_stale_jobs()

    log.info(
        "Worker started — polling every %ds",
        POLL_INTERVAL,
    )
    log.info(
        "Exclusive GPU access guaranteed "
        "(one job at a time)"
    )

    while True:
        job = claim_next_job()
        if job:
            process(job)
        else:
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
