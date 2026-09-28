"""
transcribe.py
─────────────
Studio8 transcription pipeline.

Auto-language mode:
    ffprobe
      → WhisperX VAD
      → overlapping LID windows
      → hysteresis / language blocks
      → ASR per language block
      → alignment once per language
      → full-file pyannote diarization
      → speaker assignment by time overlap
      → TXT
      → optional speakers JSON

Forced-language mode skips LID and behaves like a single-language workflow.
"""

from __future__ import annotations

import gc
import json
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
import whisperx
from pyannote.audio import Pipeline

from multilingual import (
    LanguageBlock,
    LIDWindow,
    analyse_languages,
    language_blocks_to_json,
    lid_windows_to_json,
)
from settings import cfg


HF_TOKEN = os.environ.get("HF_TOKEN", "")
DEVICE = os.environ.get("DEVICE", cfg.model.device)
COMPUTE_TYPE = (
    cfg.model.compute_type
    if cfg.model.compute_type
    else ("float16" if DEVICE == "cuda" else "int8")
)
MODEL_SIZE = os.environ.get("WHISPER_MODEL", cfg.model.whisper_model)
LANGUAGE = os.environ.get("LANGUAGE", cfg.model.default_language)
OUTPUT_DIR = Path(cfg.runtime.output_dir)

SUPPORTED_EXTENSIONS = {
    ".mp3",
    ".mp4",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".mxf",
    ".mov",
    ".mts",
    ".m2ts",
    ".avi",
    ".mkv",
    ".webm",
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Memory helpers ────────────────────────────────────────────────────────────

def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


# ─── Timecode / metadata helpers ──────────────────────────────────────────────

def _ffprobe_meta(filepath: str) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_entries",
        "format_tags=timecode,creation_time:"
        "stream_tags=timecode:"
        "stream=r_frame_rate:"
        "format=duration",
        filepath,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return json.loads(result.stdout) if result.stdout.strip() else {}
    except Exception:
        return {}


def _duration_seconds(filepath: str) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                filepath,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return float(result.stdout.strip() or "0")
    except Exception:
        return 0.0


def _parse_timecode(tc_str: str | None) -> tuple[int, int, int, int] | None:
    if not tc_str:
        return None

    value = tc_str.strip()
    match = re.match(
        r"(\d{2}):(\d{2}):(\d{2})[:;](\d{2,3})$",
        value,
    )
    if match:
        return (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
        )

    match = re.match(r"(\d{2}):(\d{2}):(\d{2})$", value)
    if match:
        return (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            0,
        )

    return None


def _detect_fps(meta: dict[str, Any]) -> float:
    try:
        for stream in meta.get("streams", []):
            rate = stream.get("r_frame_rate", "")
            if rate and rate != "0/0":
                return float(Fraction(rate))
    except Exception:
        pass
    return 25.0


def _tc_to_seconds(tc: tuple[int, int, int, int], fps: float) -> float:
    h, m, s, f = tc
    return h * 3600 + m * 60 + s + f / fps


def _seconds_to_smpte(total_seconds: float, fps: float) -> str:
    total_frames = round(total_seconds * fps)
    fps_int = max(1, round(fps))
    frames = total_frames % fps_int
    total_sec = total_frames // fps_int
    seconds = total_sec % 60
    total_min = total_sec // 60
    minutes = total_min % 60
    hours = total_min // 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"


def _seconds_to_hhmmss(total_seconds: float) -> str:
    td = timedelta(seconds=round(total_seconds))
    total = int(td.total_seconds())
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def extract_timecode_info(filepath: str) -> dict[str, Any]:
    meta = _ffprobe_meta(filepath)
    fps = _detect_fps(meta)
    tc_str = None

    for location in [
        meta.get("format", {}).get("tags", {}),
        *[
            stream.get("tags", {})
            for stream in meta.get("streams", [])
        ],
    ]:
        if not isinstance(location, dict):
            continue
        tc_str = location.get("timecode") or location.get("TIMECODE")
        if tc_str:
            break

    tc = _parse_timecode(tc_str)
    return {
        "start_seconds": _tc_to_seconds(tc, fps) if tc else 0.0,
        "fps": fps,
        "tc_string": tc_str,
        "has_timecode": tc is not None,
    }


def format_timestamp(
    relative_seconds: float,
    tc_info: dict[str, Any],
) -> str:
    absolute = tc_info["start_seconds"] + relative_seconds
    if tc_info["has_timecode"]:
        return _seconds_to_smpte(absolute, tc_info["fps"])
    return _seconds_to_hhmmss(absolute)


# ─── Language helpers ─────────────────────────────────────────────────────────

def _unique_languages_from_segments(
    segments: list[dict[str, Any]],
) -> list[str]:
    languages: list[str] = []
    for segment in segments:
        language = str(segment.get("language") or "").strip()
        if language and language != "unknown" and language not in languages:
            languages.append(language)
    return languages


def _language_value(languages: list[str]) -> str:
    if not languages:
        return "unknown"
    if len(languages) == 1:
        return languages[0]
    return "multilingual"


def _language_header(languages: list[str]) -> str:
    if not languages:
        return "unknown"
    if len(languages) == 1:
        return languages[0]
    return f"multilingual ({', '.join(languages)})"


def _block_for_time(
    blocks: list[LanguageBlock],
    t: float,
) -> LanguageBlock | None:
    for block in blocks:
        if block.start <= t <= block.end:
            return block
    if not blocks:
        return None
    return min(
        blocks,
        key=lambda b: abs(((b.start + b.end) / 2.0) - t),
    )


def _decorate_segment_language(
    segment: dict[str, Any],
    blocks: list[LanguageBlock],
    forced_language: str | None = None,
) -> dict[str, Any]:
    start = float(segment.get("start", 0.0) or 0.0)
    end = float(segment.get("end", start) or start)
    midpoint = (start + end) / 2.0

    block = _block_for_time(blocks, midpoint)
    if block is not None:
        segment["language"] = block.language
        segment["language_confidence"] = round(
            float(block.confidence),
            4,
        )
        segment["language_boundary"] = block.boundary
    elif forced_language:
        segment["language"] = forced_language
        segment["language_confidence"] = 1.0
        segment["language_boundary"] = "forced"
    else:
        segment.setdefault("language", "unknown")
        segment.setdefault("language_confidence", 0.0)
        segment.setdefault("language_boundary", "unknown")

    return segment


# ─── ASR / multilingual transcription ─────────────────────────────────────────

def _load_asr_model(language: str | None):
    return whisperx.load_model(
        MODEL_SIZE,
        DEVICE,
        compute_type=COMPUTE_TYPE,
        language=language,
    )


def _single_language_transcription(
    model,
    audio,
    language: str | None,
    duration_sec: float,
) -> tuple[
    list[dict[str, Any]],
    list[LanguageBlock],
    list[LIDWindow],
]:
    """
    Forced language, or legacy auto-language fallback when multilingual mode is
    disabled.
    """
    if language:
        detected = language
    else:
        detected = model.detect_language(audio)

    result = model.transcribe(
        audio,
        batch_size=cfg.model.batch_size,
        language=detected,
        task="transcribe",
    )

    block = LanguageBlock(
        start=0.0,
        end=duration_sec,
        language=detected,
        confidence=1.0 if language else 0.0,
        boundary="forced" if language else "file",
    )

    segments = [
        _decorate_segment_language(
            dict(segment),
            [block],
            forced_language=language,
        )
        for segment in result.get("segments", [])
    ]
    return segments, [block], []


def _multilingual_transcription(
    model,
    audio,
    duration_sec: float,
) -> tuple[
    list[dict[str, Any]],
    list[LanguageBlock],
    list[LIDWindow],
]:
    turns, lid_windows, blocks = analyse_languages(
        model,
        audio,
        window_seconds=cfg.multilingual.lid_window_seconds,
        overlap_seconds=cfg.multilingual.lid_overlap_seconds,
        lid_batch_size=cfg.multilingual.lid_batch_size,
        min_confidence=cfg.multilingual.lid_min_confidence,
        confirm_windows=cfg.multilingual.switch_confirm_windows,
        min_block_seconds=cfg.multilingual.min_block_seconds,
    )

    if not blocks:
        # No usable VAD speech turns. Preserve legacy behaviour as a failsafe.
        detected = model.detect_language(audio)
        blocks = [
            LanguageBlock(
                start=0.0,
                end=duration_sec,
                language=detected,
                confidence=0.0,
                boundary="fallback",
            )
        ]

    raw_segments: list[dict[str, Any]] = []

    for index, block in enumerate(blocks, start=1):
        a = max(0, int(round(block.start * 16000)))
        b = min(len(audio), int(round(block.end * 16000)))
        clip = audio[a:b]
        if len(clip) == 0:
            continue

        print(
            f"      ASR language block {index}/{len(blocks)}: "
            f"{block.language} "
            f"{block.start:.2f}-{block.end:.2f}s "
            f"(LID {block.confidence:.2f}, {block.boundary})"
        )

        result = model.transcribe(
            clip,
            batch_size=cfg.model.batch_size,
            language=block.language,
            task="transcribe",
        )

        for segment in result.get("segments", []):
            item = dict(segment)
            item["start"] = float(item.get("start", 0.0)) + block.start
            item["end"] = float(item.get("end", item["start"])) + block.start
            item["language"] = block.language
            item["language_confidence"] = round(block.confidence, 4)
            item["language_boundary"] = block.boundary
            raw_segments.append(item)

    raw_segments.sort(key=lambda s: float(s.get("start", 0.0)))
    return raw_segments, blocks, lid_windows


def _transcribe_with_languages(
    model,
    audio,
    forced_language: str | None,
    duration_sec: float,
) -> tuple[
    list[dict[str, Any]],
    list[LanguageBlock],
    list[LIDWindow],
]:
    if forced_language:
        print(f"      Language forced: {forced_language}")
        return _single_language_transcription(
            model,
            audio,
            forced_language,
            duration_sec,
        )

    if not cfg.multilingual.enabled:
        print("      Multilingual LID disabled — using one detected language")
        return _single_language_transcription(
            model,
            audio,
            None,
            duration_sec,
        )

    print(
        "      Multilingual LID enabled: "
        f"{cfg.multilingual.lid_window_seconds:.1f}s windows, "
        f"{cfg.multilingual.lid_overlap_seconds:.1f}s overlap, "
        f"{cfg.multilingual.switch_confirm_windows} confirmations"
    )
    return _multilingual_transcription(
        model,
        audio,
        duration_sec,
    )


# ─── Alignment ────────────────────────────────────────────────────────────────

def _align_segments_by_language(
    segments: list[dict[str, Any]],
    audio,
    blocks: list[LanguageBlock],
    forced_language: str | None = None,
) -> list[dict[str, Any]]:
    """
    Align all segments of the same language in one model load.

    Alignment models are intentionally loaded sequentially, not kept
    simultaneously in VRAM. This is effectively a per-language cache for the
    current file while remaining safe on smaller GPUs.
    """
    if not segments:
        return []

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        language = str(
            segment.get("language")
            or forced_language
            or "unknown"
        )
        grouped[language].append(segment)

    aligned_all: list[dict[str, Any]] = []

    for language, language_segments in grouped.items():
        if language == "unknown":
            aligned_all.extend(language_segments)
            continue

        print(
            f"      Alignment {language}: "
            f"{len(language_segments)} segment(s)"
        )

        model_a = None
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=language,
                device=DEVICE,
            )
            aligned = whisperx.align(
                language_segments,
                model_a,
                metadata,
                audio,
                DEVICE,
                return_char_alignments=False,
            )
            produced = aligned.get("segments", [])

            for segment in produced:
                item = dict(segment)
                _decorate_segment_language(
                    item,
                    blocks,
                    forced_language=forced_language,
                )
                aligned_all.append(item)

        except Exception as exc:
            # Missing/unsupported alignment language must not throw away a valid
            # transcription.
            print(
                f"      WARNING: Alignment unavailable for '{language}': "
                f"{exc}. Keeping ASR timestamps."
            )
            for segment in language_segments:
                aligned_all.append(dict(segment))

        finally:
            if model_a is not None:
                del model_a
            _release_cuda()

    aligned_all.sort(key=lambda s: float(s.get("start", 0.0)))
    return aligned_all


# ─── Diarization ──────────────────────────────────────────────────────────────

def _assign_speakers(
    segments: list[dict[str, Any]],
    audio,
) -> list[dict[str, Any]]:
    """
    Run diarization once over the complete file, then map speaker labels onto
    already-transcribed/aligned language segments using time overlap.
    """
    if not segments:
        return segments

    if not HF_TOKEN:
        print("      WARNING: No HF_TOKEN set — diarization skipped.")
        for segment in segments:
            segment["speaker"] = cfg.transcript.fallback_speaker
        return segments

    import pandas as pd

    diarize_model = None
    waveform = None

    try:
        diarize_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN,
        )
        diarize_model.to(torch.device(DEVICE))

        waveform = torch.from_numpy(audio).unsqueeze(0)
        diarize_output = diarize_model(
            {
                "waveform": waveform,
                "sample_rate": 16000,
            }
        )

        annotation = getattr(
            diarize_output,
            "speaker_diarization",
            diarize_output,
        )

        diarize_df = pd.DataFrame(
            [
                {
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                }
                for turn, _, speaker in annotation.itertracks(
                    yield_label=True
                )
            ]
        )

        result = whisperx.assign_word_speakers(
            diarize_df,
            {"segments": segments},
        )
        return result.get("segments", segments)

    finally:
        if waveform is not None:
            del waveform
        if diarize_model is not None:
            del diarize_model
        _release_cuda()


# ─── Complete per-file processing ─────────────────────────────────────────────

def _process_media(
    filepath: str,
    language: str | None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    float,
    list[str],
    list[LanguageBlock],
    list[LIDWindow],
]:
    tc_info = extract_timecode_info(filepath)
    duration_sec = _duration_seconds(filepath)
    audio = whisperx.load_audio(filepath)

    model = _load_asr_model(language)

    try:
        raw_segments, language_blocks, lid_windows = (
            _transcribe_with_languages(
                model,
                audio,
                language,
                duration_sec,
            )
        )
    finally:
        del model
        _release_cuda()

    aligned_segments = _align_segments_by_language(
        raw_segments,
        audio,
        language_blocks,
        forced_language=language,
    )

    segments_with_speakers = _assign_speakers(
        aligned_segments,
        audio,
    )

    # Re-apply language metadata after diarization in case a dependency version
    # returns freshly constructed segment dicts.
    for segment in segments_with_speakers:
        _decorate_segment_language(
            segment,
            language_blocks,
            forced_language=language,
        )

    languages = _unique_languages_from_segments(
        segments_with_speakers
    )

    del audio
    _release_cuda()

    return (
        segments_with_speakers,
        tc_info,
        duration_sec,
        languages,
        language_blocks,
        lid_windows,
    )


# ─── Transcript rendering ─────────────────────────────────────────────────────

def _render_segments(
    segments: list[dict[str, Any]],
    tc_info: dict[str, Any],
) -> list[str]:
    lines: list[str] = []

    current_speaker = None
    current_start = None
    last_tc_at = None
    current_text_parts: list[str] = []

    def flush_segment():
        nonlocal current_speaker
        nonlocal current_start
        nonlocal current_text_parts

        if current_speaker and current_text_parts:
            tc = format_timestamp(current_start, tc_info)
            lines.append(f"[{tc}] {current_speaker}")
            lines.append(" ".join(current_text_parts).strip())
            lines.append("")

    for segment in segments:
        speaker = (
            str(segment.get("speaker", "SPEAKER_?"))
            .upper()
            .replace(" ", "_")
        )
        text = str(segment.get("text", "") or "").strip()
        start = float(segment.get("start", 0.0) or 0.0)

        if not text:
            continue

        if speaker != current_speaker:
            flush_segment()
            current_speaker = speaker
            current_start = start
            last_tc_at = start
            current_text_parts = [text]

        elif (
            last_tc_at is not None
            and (start - last_tc_at) >= cfg.model.tc_interval
        ):
            flush_segment()
            current_start = start
            last_tc_at = start
            current_text_parts = [text]

        else:
            current_text_parts.append(text)

    flush_segment()
    return lines


# ─── JSON output ──────────────────────────────────────────────────────────────

def _speakers_json_path(transcript_path: Path) -> Path:
    stem = transcript_path.stem
    if stem.endswith("_transcript"):
        stem = stem[: -len("_transcript")]
    return transcript_path.with_name(f"{stem}_speakers.json")


def _build_speakers_payload(
    *,
    media_file: str,
    duration_sec: float,
    languages: list[str],
    tc_info: dict[str, Any],
    segments: list[dict[str, Any]],
    language_blocks: list[LanguageBlock],
    lid_windows: list[LIDWindow],
) -> dict[str, Any]:
    speakers: dict[str, dict[str, Any]] = {}

    for segment in segments:
        speaker_id = (
            str(segment.get("speaker", "SPEAKER_?"))
            .upper()
            .replace(" ", "_")
        )
        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", start) or start)
        if end < start:
            end = start

        entry = speakers.setdefault(
            speaker_id,
            {
                "id": speaker_id,
                "total_duration": 0.0,
                "segment_count": 0,
                "segments": [],
            },
        )

        item = {
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "language": segment.get("language", "unknown"),
            "language_confidence": round(
                float(segment.get("language_confidence", 0.0) or 0.0),
                4,
            ),
            "language_boundary": segment.get(
                "language_boundary",
                "unknown",
            ),
            "text": str(segment.get("text", "") or "").strip(),
        }

        entry["segments"].append(item)
        entry["total_duration"] += end - start
        entry["segment_count"] += 1

    for speaker in speakers.values():
        speaker["total_duration"] = round(
            float(speaker["total_duration"]),
            3,
        )

    return {
        "schema": "studio8-speakers",
        "version": 2,
        "media_file": media_file,
        "duration": round(duration_sec, 3),
        "language": _language_value(languages),
        "languages": languages,
        "timecode": {
            "has_timecode": bool(tc_info.get("has_timecode")),
            "start_seconds": round(
                float(tc_info.get("start_seconds", 0.0)),
                3,
            ),
            "fps": float(tc_info.get("fps", 25.0)),
            "tc_string": tc_info.get("tc_string"),
        },
        "language_blocks": language_blocks_to_json(language_blocks),
        # LID windows are useful for tuning/diagnostics and explain why a
        # boundary was chosen.
        "lid_windows": lid_windows_to_json(lid_windows),
        "speaker_count": len(speakers),
        "speakers": sorted(
            speakers.values(),
            key=lambda item: item["id"],
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


# ─── Public single-file API ───────────────────────────────────────────────────

def transcribe(
    filepath: str,
    original_name: str | None = None,
    output_path: str | None = None,
    progress=None,
    language: str | None = None,
    emit_speakers_json: bool | None = None,
) -> Path:
    """
    Transcribe one audio/video file.

    `language=None` enables automatic multilingual segmentation when
    cfg.multilingual.enabled is true.

    `emit_speakers_json=None` uses cfg.output.speakers_json_default.
    """
    def report(step: int, msg: str):
        print(f"[{step}/5] {msg}")
        if progress:
            progress(step, 5, msg)

    filepath = str(filepath)
    input_path = Path(filepath)
    display_name = original_name or input_path.name
    name_stem = (
        Path(original_name).stem
        if original_name
        else input_path.stem
    )

    forced_language = language or LANGUAGE

    if emit_speakers_json is None:
        emit_speakers_json = cfg.output.speakers_json_default

    report(1, f"Reading metadata: {display_name}")
    tc_info_preview = extract_timecode_info(filepath)
    if tc_info_preview["has_timecode"]:
        print(
            f"      Timecode found: {tc_info_preview['tc_string']} "
            f"@ {tc_info_preview['fps']:.2f} fps"
        )
    else:
        print("      No timecode found — using 00:00:00 as fallback")

    report(
        2,
        f"Loading Whisper model ({MODEL_SIZE}, {DEVICE}) / language analysis",
    )
    # _process_media owns model lifetime and includes LID/ASR.
    report(3, "Transcribing language blocks…")
    (
        segments,
        tc_info,
        duration_sec,
        languages,
        language_blocks,
        lid_windows,
    ) = _process_media(
        filepath,
        forced_language,
    )

    report(4, "Alignment complete; preparing speaker-labelled output…")
    report(5, "Writing transcript…")

    t = cfg.transcript
    if tc_info["has_timecode"]:
        tc_header = (
            f"{t.timecode_label}: {tc_info['tc_string']} "
            f"({tc_info['fps']:.2f} fps)"
        )
    else:
        tc_header = (
            f"{t.timecode_label}: 00:00:00 (no TC in file)"
        )

    lines = [
        f"{t.date_label}:     {datetime.now().strftime('%Y-%m-%d')}",
        f"{t.file_label}:     {display_name}",
        f"{t.duration_label}: {_seconds_to_hhmmss(duration_sec)}",
        f"{t.language_label}: {_language_header(languages)}",
        tc_header,
        t.separator,
        "",
    ]
    lines.extend(_render_segments(segments, tc_info))

    out_file = (
        OUTPUT_DIR / f"{name_stem}_transcript.txt"
        if output_path is None
        else Path(output_path)
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n✓ Transcript saved: {out_file}")

    if emit_speakers_json:
        payload = _build_speakers_payload(
            media_file=display_name,
            duration_sec=duration_sec,
            languages=languages,
            tc_info=tc_info,
            segments=segments,
            language_blocks=language_blocks,
            lid_windows=lid_windows,
        )
        json_path = _speakers_json_path(out_file)
        _write_json(json_path, payload)
        print(f"✓ Speakers JSON saved: {json_path}")
    else:
        print("  Speakers JSON: disabled for this workflow")

    return out_file


# ─── Batch API ────────────────────────────────────────────────────────────────

def _transcribe_single_for_batch(
    filepath: str,
    model=None,
    diarize_model=None,
    progress_label: str = "",
    progress=None,
    language: str | None = None,
):
    """
    Compatibility helper.

    The multilingual implementation deliberately manages model lifetimes per
    file, so externally preloaded `model`/`diarize_model` are ignored.
    """
    if progress:
        progress(f"Processing {Path(filepath).name}")

    (
        segments,
        tc_info,
        duration_sec,
        languages,
        language_blocks,
        lid_windows,
    ) = _process_media(
        str(filepath),
        language or LANGUAGE,
    )

    return (
        segments,
        tc_info,
        _language_value(languages),
        duration_sec,
    )


def batch_transcribe(
    folder: str,
    output_path: str | None = None,
    progress=None,
    language: str | None = None,
    emit_speakers_json: bool | None = None,
) -> Path:
    """
    Transcribe all supported media files in a folder into one TXT transcript.
    Each file runs multilingual LID independently when no language is forced.
    """
    folder_path = Path(folder)
    folder_name = folder_path.name
    forced_language = language or LANGUAGE

    if emit_speakers_json is None:
        emit_speakers_json = cfg.output.speakers_json_default

    files = sorted(
        [
            p
            for p in folder_path.iterdir()
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=lambda p: p.name,
    )

    if not files:
        raise ValueError(
            f"No supported media files found in {folder}"
        )

    total_steps = len(files) + 1

    def report(step: int, msg: str):
        print(f"[{step}/{total_steps}] {msg}")
        if progress:
            progress(step, total_steps, msg)

    report(1, f"Found {len(files)} file(s) — starting")

    t = cfg.transcript
    lines = [
        f"{t.date_label}:     {datetime.now().strftime('%Y-%m-%d')}",
        f"Folder:   {folder_name}",
        f"Files:    {len(files)} clips (alphabetical)",
        f"{t.duration_label}: —",
        f"{t.language_label}: —",
        t.separator,
        "",
    ]

    total_duration = 0.0
    all_languages: list[str] = []
    json_files: list[dict[str, Any]] = []

    for index, filepath in enumerate(files, start=1):
        report(
            index + 1,
            f"Processing {filepath.name} ({index}/{len(files)})",
        )

        try:
            (
                segments,
                tc_info,
                duration_sec,
                languages,
                language_blocks,
                lid_windows,
            ) = _process_media(
                str(filepath),
                forced_language,
            )
        except Exception as exc:
            print(f"    ERROR: {filepath.name}: {exc}")
            _release_cuda()
            sep_label = f"── {filepath.name} "
            lines.append(
                sep_label
                + "─" * max(
                    0,
                    cfg.transcript.separator_length
                    - len(sep_label),
                )
            )
            lines.append(f"[ERROR] {exc}")
            lines.append("")
            continue

        total_duration += duration_sec

        for lang in languages:
            if lang not in all_languages:
                all_languages.append(lang)

        sep_label = f"── {filepath.name} "
        lines.append(
            sep_label
            + "─" * max(
                0,
                cfg.transcript.separator_length
                - len(sep_label),
            )
        )
        lines.append("")
        lines.extend(_render_segments(segments, tc_info))

        if emit_speakers_json:
            json_files.append(
                _build_speakers_payload(
                    media_file=filepath.name,
                    duration_sec=duration_sec,
                    languages=languages,
                    tc_info=tc_info,
                    segments=segments,
                    language_blocks=language_blocks,
                    lid_windows=lid_windows,
                )
            )

    lines[3] = (
        f"{t.duration_label}: "
        f"{_seconds_to_hhmmss(total_duration)} (total)"
    )
    lines[4] = (
        f"{t.language_label}: "
        f"{_language_header(all_languages)}"
    )

    out_file = (
        Path(cfg.runtime.output_dir)
        / f"{folder_name}_transcript.txt"
        if output_path is None
        else Path(output_path)
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n✓ Batch transcript saved: {out_file}")

    if emit_speakers_json:
        payload = {
            "schema": "studio8-speakers-batch",
            "version": 2,
            "folder": folder_name,
            "file_count": len(json_files),
            "duration": round(total_duration, 3),
            "language": _language_value(all_languages),
            "languages": all_languages,
            "files": json_files,
        }
        json_path = _speakers_json_path(out_file)
        _write_json(json_path, payload)
        print(f"✓ Batch speakers JSON saved: {json_path}")
    else:
        print("  Speakers JSON: disabled for this workflow")

    return out_file


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <file> [output.txt]")
        raise SystemExit(1)

    out = sys.argv[2] if len(sys.argv) > 2 else None
    transcribe(
        sys.argv[1],
        original_name=Path(sys.argv[1]).name,
        output_path=out,
    )
