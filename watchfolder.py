"""
watchfolder.py
──────────────
Reads watchfolders.yaml and monitors configured sources.

Per workflow:
    speakers_json: false   # default
    speakers_json: true    # emit *_speakers.json

Single mode:
    each media file → one transcript.

Batch mode:
    .done marker inside a card folder → one combined transcript.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
from fnmatch import fnmatch
from pathlib import Path

import yaml
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from jobstore import init_db, submit_job
from settings import cfg
from transcribe import SUPPORTED_EXTENSIONS


CONFIG_PATH = Path(
    os.environ.get(
        "WATCHFOLDERS_CONFIG_PATH",
        "./watchfolders.yaml",
    )
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WATCHFOLDER] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

STAGING_COPY_MAX_BYTES = int(
    os.environ.get(
        "STAGING_COPY_MAX_BYTES",
        str(int(cfg.runtime.staging_copy_max_gb * 1024**3)),
    )
)


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
        "ja",
    }


def workflow_speakers_json(config: dict) -> bool:
    """
    Per-workflow override, falling back to the global config default.
    """
    if "speakers_json" in config:
        return _as_bool(config.get("speakers_json"), False)
    return bool(cfg.output.speakers_json_default)


# ─── Staging ──────────────────────────────────────────────────────────────────

def stage_or_reference(
    src: Path,
    staging_dir: Path,
    job_id: str,
) -> str:
    """
    Small files are copied to staging. Large files are referenced directly.
    """
    try:
        size = src.stat().st_size
    except OSError:
        size = 0

    if size > STAGING_COPY_MAX_BYTES:
        log.info(
            "File %s is %.1f GB — exceeds staging threshold %.1f GB; "
            "referencing source directly",
            src.name,
            size / 1024**3,
            STAGING_COPY_MAX_BYTES / 1024**3,
        )
        return str(src)

    staging_dir.mkdir(parents=True, exist_ok=True)
    staged = staging_dir / f"{job_id}_{src.name}"
    shutil.copy2(str(src), str(staged))
    return str(staged)


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config() -> list[dict]:
    if not CONFIG_PATH.exists():
        log.warning(
            "Config not found: %s — using default workflow",
            CONFIG_PATH,
        )
        return [
            {
                "name": "General Intake",
                "mode": "single",
                "path": "./watchfolder",
                "output": "./output",
                "priority": 5,
                "enabled": True,
                "speakers_json": False,
            }
        ]

    with open(CONFIG_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    valid: list[dict] = []
    for entry in data.get("watchfolders", []):
        if not entry.get("enabled", True):
            continue

        mode = entry.get("mode", "single")
        if mode not in ("single", "batch"):
            log.warning(
                "Unknown mode '%s' for '%s' — skipping",
                mode,
                entry.get("name"),
            )
            continue

        # Make the default explicit inside runtime config.
        entry = dict(entry)
        entry["speakers_json"] = workflow_speakers_json(entry)
        valid.append(entry)

    return valid


def get_batch_config(folder_path: str) -> dict | None:
    folder = Path(folder_path).resolve()

    for entry in load_config():
        if entry.get("mode") != "batch":
            continue

        base = Path(entry["path"]).resolve()
        try:
            folder.relative_to(base)
            if fnmatch(
                folder.name,
                entry.get("subfolder_glob", "*"),
            ):
                return entry
        except ValueError:
            continue

    return None


# ─── File settle tracking ─────────────────────────────────────────────────────

class PendingFiles:
    def __init__(self, settle_time: int = 5):
        self._files: dict[str, float] = {}
        self._settle = settle_time

    def touch(self, path: str):
        self._files[path] = time.time()

    def settled(self) -> list[str]:
        now = time.time()
        ready = [
            path
            for path, timestamp in self._files.items()
            if now - timestamp >= self._settle
        ]
        for path in ready:
            del self._files[path]
        return ready


# ─── Shared enqueue helpers ───────────────────────────────────────────────────

def _resolve_single_output(config: dict, src: Path) -> str:
    raw_output = config.get(
        "output",
        cfg.runtime.output_dir,
    )
    if raw_output == "same_as_source":
        return str(src.parent)
    return str(raw_output)


def _single_already_done(
    config: dict,
    src: Path,
    output_dir: str,
) -> bool:
    transcript = (
        Path(output_dir)
        / f"{src.stem}_transcript.txt"
    )
    return transcript.exists()


def _submit_single(
    config: dict,
    src: Path,
    staging_dir: Path,
) -> str | None:
    output_dir = _resolve_single_output(config, src)

    if _single_already_done(config, src, output_dir):
        log.info(
            "[%s] Skipping (transcript exists): %s",
            config["name"],
            src.name,
        )
        return None

    job_id = str(uuid.uuid4())[:8]
    staged_path = stage_or_reference(
        src,
        staging_dir,
        job_id,
    )

    priority = int(config.get("priority", 5))
    language = config.get("language") or None
    speakers_json = workflow_speakers_json(config)

    submit_job(
        job_id,
        filename=src.name,
        filepath=staged_path,
        source="watchfolder",
        priority=priority,
        output_dir=output_dir,
        language=language,
        speakers_json=speakers_json,
    )

    log.info(
        "[%s] Queued: %s [%s] "
        "(priority %s, lang=%s, speakers_json=%s, output=%s)",
        config["name"],
        src.name,
        job_id,
        priority,
        language or "auto",
        "yes" if speakers_json else "no",
        output_dir,
    )
    return job_id


# ─── Single mode: watchdog ────────────────────────────────────────────────────

class SingleModeHandler(FileSystemEventHandler):
    def __init__(self, config: dict):
        self.config = config
        self.pending = PendingFiles(cfg.runtime.settle_time)
        self.submitted: set[str] = set()
        self._staging = (
            Path(cfg.runtime.output_dir)
            / "staging"
        )

    def on_created(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._handle(event.dest_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    def _handle(self, path: str):
        if Path(path).suffix.lower() not in SUPPORTED_EXTENSIONS:
            return
        if path in self.submitted:
            return

        log.info(
            "[%s] New file: %s",
            self.config["name"],
            Path(path).name,
        )
        self.pending.touch(path)

    def process_settled(self):
        for path in self.pending.settled():
            self._enqueue(path)

    def _enqueue(self, path: str):
        if path in self.submitted:
            return

        src = Path(path)
        self.submitted.add(path)

        job_id = _submit_single(
            self.config,
            src,
            self._staging,
        )
        if job_id is None:
            return

    def scan_existing(self):
        watch_path = Path(self.config["path"])
        if not watch_path.exists():
            return

        files = [
            str(path)
            for path in watch_path.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if files:
            log.info(
                "[%s] %d existing file(s) found",
                self.config["name"],
                len(files),
            )
            for path in files:
                self._enqueue(path)


# ─── Single mode: polling ─────────────────────────────────────────────────────

class SingleModePoller:
    """
    Polling alternative for CIFS/NFS mounts.
    """

    def __init__(
        self,
        config: dict,
        poll_interval: int = 10,
    ):
        self.config = config
        self.poll_interval = int(
            config.get(
                "poll_interval",
                poll_interval,
            )
        )
        self.submitted: set[str] = set()
        self._staging = (
            Path(cfg.runtime.output_dir)
            / "staging"
        )

    def scan(self):
        watch_path = Path(self.config["path"])
        if not watch_path.exists():
            log.warning(
                "[%s] Path not reachable: %s",
                self.config["name"],
                watch_path,
            )
            return

        for path in watch_path.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if str(path) in self.submitted:
                continue
            self._enqueue(path)

    def _enqueue(self, path: str):
        if path in self.submitted:
            return

        src = Path(path)
        self.submitted.add(path)

        _submit_single(
            self.config,
            src,
            self._staging,
        )


# ─── Batch mode ───────────────────────────────────────────────────────────────

class BatchModePoller:
    """
    Poll parent folder for .done files located inside card subfolders.
    """

    def __init__(
        self,
        config: dict,
        poll_interval: int = 10,
    ):
        self.config = config
        self.done_ext = config.get(
            "done_extension",
            ".done",
        )
        self.delete_done = _as_bool(
            config.get("delete_done_file", True),
            True,
        )
        self.poll_interval = poll_interval
        self.submitted: set[str] = set()

    def scan(self):
        watch_path = Path(self.config["path"])
        if not watch_path.exists():
            log.warning(
                "[%s] Path not reachable: %s",
                self.config["name"],
                watch_path,
            )
            return

        for done_file in watch_path.rglob(
            f"*{self.done_ext}"
        ):
            key = str(done_file.resolve())
            if key in self.submitted:
                continue
            self._enqueue_batch(done_file)

    def _enqueue_batch(self, done_file: Path):
        key = str(done_file.resolve())
        card_folder = done_file.parent
        folder_name = card_folder.name

        if not card_folder.is_dir():
            log.warning(
                "[%s] Marker '%s' found but folder '%s' "
                "does not exist — skipping",
                self.config["name"],
                done_file.name,
                card_folder,
            )
            self.submitted.add(key)
            return

        media_files = [
            path
            for path in card_folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not media_files:
            log.warning(
                "[%s] No media files in '%s' — skipping",
                self.config["name"],
                folder_name,
            )
            self.submitted.add(key)
            return

        self.submitted.add(key)

        job_id = str(uuid.uuid4())[:8]
        priority = int(
            self.config.get("priority", 7)
        )
        language = self.config.get("language") or None
        speakers_json = workflow_speakers_json(
            self.config
        )

        raw_outputs = self.config.get("outputs") or []
        if (
            not raw_outputs
            and self.config.get("output")
        ):
            raw_outputs = [self.config["output"]]
        if not raw_outputs:
            raw_outputs = [cfg.runtime.output_dir]

        primary_output = raw_outputs[0]
        extra_outputs = raw_outputs[1:]

        submit_job(
            job_id,
            filename=folder_name,
            filepath=str(card_folder),
            source="watchfolder",
            priority=priority,
            mode="batch",
            output_dir=primary_output,
            output_dirs=extra_outputs,
            language=language,
            speakers_json=speakers_json,
        )

        log.info(
            "[%s] Batch queued: '%s' (%d clips) [%s] "
            "(priority %s, lang=%s, speakers_json=%s, outputs=%d)",
            self.config["name"],
            folder_name,
            len(media_files),
            job_id,
            priority,
            language or "auto",
            "yes" if speakers_json else "no",
            len(raw_outputs),
        )

        if self.delete_done:
            self._remove_marker(done_file)

    def _remove_marker(self, done_file: Path):
        removed = False

        for attempt in range(10):
            try:
                processed = done_file.with_suffix(
                    ".done.processed"
                )
                done_file.rename(processed)
                processed.unlink()
                log.info(
                    "[%s] Removed marker: %s",
                    self.config["name"],
                    done_file.name,
                )
                removed = True
                break

            except OSError as exc:
                if getattr(exc, "errno", None) in (
                    13,  # EACCES
                    16,  # EBUSY
                ):
                    log.debug(
                        "[%s] Marker locked, retrying "
                        "(attempt %d/10)",
                        self.config["name"],
                        attempt + 1,
                    )
                    time.sleep(1)
                else:
                    log.warning(
                        "[%s] Could not remove marker: %s",
                        self.config["name"],
                        exc,
                    )
                    break

        if not removed:
            log.warning(
                "[%s] Marker still locked after 10s; "
                "already in submitted set",
                self.config["name"],
            )


# ─── Main loop ────────────────────────────────────────────────────────────────

def run():
    init_db()
    entries = load_config()

    single_entries = [
        entry
        for entry in entries
        if entry.get("mode", "single") == "single"
    ]
    batch_entries = [
        entry
        for entry in entries
        if entry.get("mode") == "batch"
    ]

    poll_interval = cfg.runtime.batch_poll_interval

    log.info(
        "Loaded %d single-mode, %d batch-mode entries",
        len(single_entries),
        len(batch_entries),
    )

    observer = Observer()
    single_handlers: list[SingleModeHandler] = []
    single_pollers: list[SingleModePoller] = []

    for entry in single_entries:
        watch_path = Path(entry["path"])

        if entry.get("poll", False):
            poller = SingleModePoller(
                entry,
                poll_interval=poll_interval,
            )
            poller.scan()
            single_pollers.append(poller)
            log.info(
                "  Single (poll): '%s' → %s "
                "(every %ss, speakers_json=%s)",
                entry["name"],
                watch_path,
                poller.poll_interval,
                "yes"
                if workflow_speakers_json(entry)
                else "no",
            )
        else:
            if not watch_path.exists():
                watch_path.mkdir(
                    parents=True,
                    exist_ok=True,
                )

            handler = SingleModeHandler(entry)
            handler.scan_existing()
            observer.schedule(
                handler,
                str(watch_path),
                recursive=False,
            )
            single_handlers.append(handler)

            log.info(
                "  Single (watch): '%s' → %s "
                "(speakers_json=%s)",
                entry["name"],
                watch_path,
                "yes"
                if workflow_speakers_json(entry)
                else "no",
            )

    batch_pollers: list[BatchModePoller] = []

    for entry in batch_entries:
        poller = BatchModePoller(
            entry,
            poll_interval=poll_interval,
        )
        poller.scan()
        batch_pollers.append(poller)

        log.info(
            "  Batch: '%s' → %s "
            "(polling every %ss, trigger: *%s, speakers_json=%s)",
            entry["name"],
            entry["path"],
            poll_interval,
            entry.get("done_extension", ".done"),
            "yes"
            if workflow_speakers_json(entry)
            else "no",
        )

    observer.start()
    log.info("Watchfolder running (Ctrl+C to stop)")

    tick = 0

    try:
        while True:
            time.sleep(1)
            tick += 1

            for handler in single_handlers:
                handler.process_settled()

            for poller in single_pollers:
                if tick % poller.poll_interval == 0:
                    poller.scan()

            if tick % poll_interval == 0:
                for poller in batch_pollers:
                    poller.scan()

    except KeyboardInterrupt:
        log.info("Watchfolder shutting down…")
        observer.stop()

    observer.join()


if __name__ == "__main__":
    run()
