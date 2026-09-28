"""
settings.py
───────────
Loads config.yaml and provides typed access to all settings.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "./config.yaml"))


@dataclass
class Branding:
    organization: str = "Studio 8, Washington"
    app_name: str = "Transcription"
    footer: str = "WhisperX · pyannote · Studio 8, Washington"
    system_subtitle: str = "System Status"
    queue_subtitle: str = "Queue"


@dataclass
class Colors:
    bg: str = "#080808"
    surface: str = "#111111"
    surface2: str = "#1a1a1a"
    border: str = "#333333"
    border2: str = "#484848"
    text: str = "#f0f0f0"
    muted: str = "#888888"
    green: str = "#39d98a"
    amber: str = "#ffcc00"
    red: str = "#ff5555"
    blue: str = "#58a6ff"

    def as_css_vars(self) -> str:
        return f"""
    --bg:       {self.bg};
    --surface:  {self.surface};
    --surface2: {self.surface2};
    --border:   {self.border};
    --border2:  {self.border2};
    --text:     {self.text};
    --muted:    {self.muted};
    --green:    {self.green};
    --amber:    {self.amber};
    --red:      {self.red};
    --blue:     {self.blue};
    --mono:     'IBM Plex Mono', monospace;
    --sans:     'IBM Plex Sans', sans-serif;""".strip()


@dataclass
class Model:
    whisper_model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    default_language: Optional[str] = None
    upload_languages: list = field(default_factory=lambda: [
        ["en", "English"],
        ["de", "Deutsch"],
        ["es", "Español"],
    ])
    batch_size: int = 2
    tc_interval: float = 60.0


@dataclass
class Multilingual:
    # Used only when no language is forced for the job.
    enabled: bool = True

    # LID windows inside VAD speech regions.
    lid_window_seconds: float = 8.0
    lid_overlap_seconds: float = 2.0
    lid_batch_size: int = 8

    # Hysteresis: a candidate language must be confident and repeated.
    lid_min_confidence: float = 0.70
    switch_confirm_windows: int = 2

    # Prevent pathological tiny language blocks after boundary estimation.
    min_block_seconds: float = 3.0


@dataclass
class Output:
    # Global default. Individual watchfolder workflows may override this with
    # `speakers_json: true|false` in watchfolders.yaml.
    speakers_json_default: bool = False


@dataclass
class Runtime:
    output_dir: str = "/opt/transcription/output"
    settle_time: int = 5
    batch_poll_interval: int = 10
    worker_poll: int = 3
    update_branch: str = "main"
    staging_copy_max_gb: float = 5.0


@dataclass
class Watchdog:
    poll_interval: int = 30
    stuck_job_timeout: int = 3600
    staging_max_age_hours: int = 24


@dataclass
class Priority:
    manual_upload: int = 10
    watchfolder_default: int = 5
    max_retries: int = 3


@dataclass
class Transcript:
    date_label: str = "Date"
    file_label: str = "File"
    duration_label: str = "Duration"
    language_label: str = "Language"
    timecode_label: str = "Timecode start"
    separator_char: str = "─"
    separator_length: int = 44
    fallback_speaker: str = "SPEAKER_A"

    @property
    def separator(self) -> str:
        return self.separator_char * self.separator_length


@dataclass
class Config:
    version: str = "1.0.0"
    branding: Branding = field(default_factory=Branding)
    colors: Colors = field(default_factory=Colors)
    model: Model = field(default_factory=Model)
    multilingual: Multilingual = field(default_factory=Multilingual)
    output: Output = field(default_factory=Output)
    runtime: Runtime = field(default_factory=Runtime)
    watchdog: Watchdog = field(default_factory=Watchdog)
    priority: Priority = field(default_factory=Priority)
    transcript: Transcript = field(default_factory=Transcript)


def _merge(dataclass_instance, data: dict):
    """Recursively merge dict values into a dataclass instance."""
    for key, value in data.items():
        if hasattr(dataclass_instance, key):
            attr = getattr(dataclass_instance, key)
            if hasattr(attr, "__dataclass_fields__") and isinstance(value, dict):
                _merge(attr, value)
            else:
                setattr(dataclass_instance, key, value)


def load_config() -> Config:
    config = Config()
    if not CONFIG_PATH.exists():
        return config

    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if "version" in data:
            config.version = str(data["version"])

        for section in (
            "branding",
            "colors",
            "model",
            "multilingual",
            "output",
            "runtime",
            "watchdog",
            "priority",
            "transcript",
        ):
            if section in data:
                _merge(getattr(config, section), data[section])

    except Exception as e:
        print(
            f"[settings] Warning: could not load {CONFIG_PATH}: {e} "
            "— using defaults"
        )

    return config


cfg = load_config()
