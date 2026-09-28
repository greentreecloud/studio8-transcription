"""
multilingual.py
───────────────
Language segmentation helpers for Studio8.

Pipeline:
    WhisperX VAD speech turns
      → overlapping fixed LID windows
      → Whisper/faster-whisper language ID
      → hysteresis
      → language blocks

This module intentionally does not perform speaker diarization. Speaker
diarization remains a separate full-file pass and is mapped onto the aligned
transcript afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable

import numpy as np
import torch

SAMPLE_RATE = 16000


@dataclass
class VADTurn:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class LIDWindow:
    start: float
    end: float
    vad_index: int
    language: str = "unknown"
    confidence: float = 0.0

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0


@dataclass
class LanguageBlock:
    start: float
    end: float
    language: str
    confidence: float
    boundary: str = "start"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration"] = round(self.duration, 3)
        return data


def _unique_turns(turns: Iterable[tuple[float, float]]) -> list[VADTurn]:
    result: list[VADTurn] = []
    seen: set[tuple[int, int]] = set()

    for start, end in turns:
        start = max(0.0, float(start))
        end = max(start, float(end))
        if end <= start:
            continue

        key = (round(start * 1000), round(end * 1000))
        if key in seen:
            continue
        seen.add(key)
        result.append(VADTurn(start, end))

    result.sort(key=lambda x: (x.start, x.end))
    return result


def extract_vad_turns(asr_model, audio: np.ndarray) -> list[VADTurn]:
    """
    Extract the fine VAD speech turns from the VAD model already owned by the
    WhisperX ASR pipeline.

    WhisperX normally merges VAD turns into larger chunks before ASR. Here we
    deliberately recover the underlying speech-turn candidates instead, because
    speaker turns must not define language boundaries.
    """
    if audio is None or len(audio) == 0:
        return []

    vad_model = asr_model.vad_model
    vad_params = getattr(asr_model, "_vad_params", {}) or {}
    onset = float(vad_params.get("vad_onset", 0.5))
    offset = vad_params.get("vad_offset", 0.363)

    # Mirror WhisperX's own VAD invocation contract.
    if hasattr(vad_model, "preprocess_audio"):
        waveform = vad_model.preprocess_audio(audio)
    else:
        waveform = torch.from_numpy(audio).unsqueeze(0)

    raw = vad_model(
        {
            "waveform": waveform,
            "sample_rate": SAMPLE_RATE,
        }
    )

    if raw is None:
        return []

    # Ask WhisperX's VAD helper to expose all internal turns in a single
    # container. The nested `segments` list keeps the actual fine VAD turns.
    duration = len(audio) / SAMPLE_RATE
    huge_chunk = max(30.0, duration + 1.0)

    merge_chunks = getattr(vad_model, "merge_chunks", None)
    if merge_chunks is None:
        return _unique_turns(
            (float(seg.start), float(seg.end))
            for seg in raw
            if hasattr(seg, "start") and hasattr(seg, "end")
        )

    merged = merge_chunks(
        raw,
        huge_chunk,
        onset=onset,
        offset=offset,
    )

    fine: list[tuple[float, float]] = []
    for item in merged or []:
        if isinstance(item, dict):
            nested = item.get("segments") or []
            if nested:
                fine.extend((float(a), float(b)) for a, b in nested)
            elif "start" in item and "end" in item:
                fine.append((float(item["start"]), float(item["end"])))
        elif hasattr(item, "start") and hasattr(item, "end"):
            fine.append((float(item.start), float(item.end)))

    return _unique_turns(fine)


def make_lid_windows(
    turns: list[VADTurn],
    window_seconds: float = 8.0,
    overlap_seconds: float = 2.0,
) -> list[LIDWindow]:
    """
    Create LID windows from VAD turns.

    Short VAD turns become one window. Long uninterrupted speech is split into
    fixed overlapping windows so language changes without pauses can still be
    detected.
    """
    window_seconds = max(1.0, float(window_seconds))
    overlap_seconds = max(0.0, float(overlap_seconds))
    if overlap_seconds >= window_seconds:
        overlap_seconds = max(0.0, window_seconds - 0.5)

    stride = window_seconds - overlap_seconds
    windows: list[LIDWindow] = []

    for vad_index, turn in enumerate(turns):
        if turn.duration <= window_seconds:
            windows.append(
                LIDWindow(
                    start=turn.start,
                    end=turn.end,
                    vad_index=vad_index,
                )
            )
            continue

        starts: list[float] = []
        s = turn.start
        while s + window_seconds < turn.end:
            starts.append(s)
            s += stride

        final_start = max(turn.start, turn.end - window_seconds)
        if not starts or abs(final_start - starts[-1]) > 0.25:
            starts.append(final_start)

        for start in starts:
            windows.append(
                LIDWindow(
                    start=start,
                    end=min(turn.end, start + window_seconds),
                    vad_index=vad_index,
                )
            )

    windows.sort(key=lambda x: (x.start, x.end))
    return windows


def _parse_language_token(token: str) -> str:
    token = str(token)
    if token.startswith("<|") and token.endswith("|>"):
        return token[2:-2]
    return token


def detect_language_windows(
    asr_model,
    audio: np.ndarray,
    windows: list[LIDWindow],
    batch_size: int = 8,
) -> list[LIDWindow]:
    """
    Run Whisper language ID for all candidate windows.

    Uses the CTranslate2/faster-whisper model already loaded by WhisperX.
    Windows are padded to Whisper's 30-second encoder input, then encoded in
    batches. This avoids starting a new model for every LID window.
    """
    if not windows:
        return windows

    from whisperx.audio import N_SAMPLES, log_mel_spectrogram

    fw_model = asr_model.model
    model_n_mels = fw_model.feat_kwargs.get("feature_size")
    n_mels = model_n_mels if model_n_mels is not None else 80

    batch_size = max(1, int(batch_size))

    for offset in range(0, len(windows), batch_size):
        batch = windows[offset : offset + batch_size]
        features: list[torch.Tensor] = []

        for window in batch:
            a = max(0, int(round(window.start * SAMPLE_RATE)))
            b = min(len(audio), int(round(window.end * SAMPLE_RATE)))
            clip = audio[a:b]

            if len(clip) > N_SAMPLES:
                clip = clip[:N_SAMPLES]

            padding = max(0, N_SAMPLES - len(clip))
            mel = log_mel_spectrogram(
                clip,
                n_mels=n_mels,
                padding=padding,
            )
            features.append(mel)

        stacked = torch.stack(features)
        encoder_output = fw_model.encode(stacked)
        results = fw_model.model.detect_language(encoder_output)

        for window, candidates in zip(batch, results):
            if not candidates:
                window.language = "unknown"
                window.confidence = 0.0
                continue

            language_token, probability = candidates[0]
            window.language = _parse_language_token(language_token)
            window.confidence = float(probability)

    return windows


def _estimated_boundary(previous: LIDWindow, first_new: LIDWindow) -> float:
    """
    Estimate a boundary between two overlapping windows.

    If they overlap, use the midpoint of the overlap. If they do not, use the
    midpoint of the gap.
    """
    left = first_new.start
    right = previous.end
    if right >= left:
        return (left + right) / 2.0
    return (right + left) / 2.0


def _block_confidence(
    language: str,
    start: float,
    end: float,
    windows: list[LIDWindow],
) -> float:
    values = [
        w.confidence
        for w in windows
        if w.language == language and start <= w.center <= end
    ]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def build_language_blocks(
    turns: list[VADTurn],
    windows: list[LIDWindow],
    min_confidence: float = 0.70,
    confirm_windows: int = 2,
    min_block_seconds: float = 3.0,
) -> list[LanguageBlock]:
    """
    Convert LID windows into stable language blocks using hysteresis.

    A switch is accepted only after `confirm_windows` consecutive confident
    windows agree on the new language. A switch that starts at a new VAD turn
    receives a natural `vad` boundary; otherwise the boundary is marked
    `estimated`.
    """
    if not turns or not windows:
        return []

    min_confidence = float(min_confidence)
    confirm_windows = max(1, int(confirm_windows))
    min_block_seconds = max(0.0, float(min_block_seconds))

    confident = [w for w in windows if w.confidence >= min_confidence]
    initial = confident[0] if confident else windows[0]
    current_language = initial.language

    events: list[tuple[float, str, str, float]] = []
    pending_language: str | None = None
    pending_windows: list[LIDWindow] = []
    previous_stable: LIDWindow = initial

    for window in windows:
        # Low-confidence disagreements do not move the state machine.
        if (
            window.language == "unknown"
            or window.confidence < min_confidence
        ):
            continue

        if window.language == current_language:
            pending_language = None
            pending_windows = []
            previous_stable = window
            continue

        if pending_language == window.language:
            pending_windows.append(window)
        else:
            pending_language = window.language
            pending_windows = [window]

        if len(pending_windows) < confirm_windows:
            continue

        first_new = pending_windows[0]

        if first_new.vad_index != previous_stable.vad_index:
            boundary = first_new.start
            boundary_kind = "vad"
        else:
            boundary = _estimated_boundary(previous_stable, first_new)
            boundary_kind = "estimated"

        confidence = sum(w.confidence for w in pending_windows) / len(
            pending_windows
        )
        events.append(
            (
                float(boundary),
                pending_language,
                boundary_kind,
                float(confidence),
            )
        )

        current_language = pending_language
        previous_stable = pending_windows[-1]
        pending_language = None
        pending_windows = []

    start = turns[0].start
    end = turns[-1].end

    blocks: list[LanguageBlock] = []
    current_language = initial.language
    cursor = start
    boundary_kind = "start"

    for boundary, new_language, new_boundary_kind, switch_conf in events:
        boundary = min(max(boundary, cursor), end)
        if boundary > cursor:
            conf = _block_confidence(
                current_language,
                cursor,
                boundary,
                windows,
            )
            blocks.append(
                LanguageBlock(
                    start=cursor,
                    end=boundary,
                    language=current_language,
                    confidence=conf,
                    boundary=boundary_kind,
                )
            )
        cursor = boundary
        current_language = new_language
        boundary_kind = new_boundary_kind

    if end > cursor:
        conf = _block_confidence(
            current_language,
            cursor,
            end,
            windows,
        )
        blocks.append(
            LanguageBlock(
                start=cursor,
                end=end,
                language=current_language,
                confidence=conf,
                boundary=boundary_kind,
            )
        )

    # Merge pathological tiny blocks. Prefer a same-language neighbour;
    # otherwise merge into the longer adjacent block.
    if min_block_seconds > 0 and len(blocks) > 1:
        changed = True
        while changed and len(blocks) > 1:
            changed = False
            for i, block in enumerate(list(blocks)):
                if block.duration >= min_block_seconds:
                    continue

                left = blocks[i - 1] if i > 0 else None
                right = blocks[i + 1] if i + 1 < len(blocks) else None

                if left and left.language == block.language:
                    left.end = block.end
                    left.confidence = max(left.confidence, block.confidence)
                    blocks.pop(i)
                elif right and right.language == block.language:
                    right.start = block.start
                    blocks.pop(i)
                elif left and right:
                    target = left if left.duration >= right.duration else right
                    if target is left:
                        left.end = block.end
                    else:
                        right.start = block.start
                    blocks.pop(i)
                elif left:
                    left.end = block.end
                    blocks.pop(i)
                elif right:
                    right.start = block.start
                    blocks.pop(i)
                changed = True
                break

    # Remove zero-length artefacts and sort.
    blocks = [b for b in blocks if b.end > b.start]
    blocks.sort(key=lambda b: (b.start, b.end))
    return blocks


def analyse_languages(
    asr_model,
    audio: np.ndarray,
    *,
    window_seconds: float,
    overlap_seconds: float,
    lid_batch_size: int,
    min_confidence: float,
    confirm_windows: int,
    min_block_seconds: float,
) -> tuple[list[VADTurn], list[LIDWindow], list[LanguageBlock]]:
    turns = extract_vad_turns(asr_model, audio)
    windows = make_lid_windows(
        turns,
        window_seconds=window_seconds,
        overlap_seconds=overlap_seconds,
    )
    detect_language_windows(
        asr_model,
        audio,
        windows,
        batch_size=lid_batch_size,
    )
    blocks = build_language_blocks(
        turns,
        windows,
        min_confidence=min_confidence,
        confirm_windows=confirm_windows,
        min_block_seconds=min_block_seconds,
    )
    return turns, windows, blocks


def lid_windows_to_json(windows: list[LIDWindow]) -> list[dict[str, Any]]:
    return [
        {
            "start": round(w.start, 3),
            "end": round(w.end, 3),
            "language": w.language,
            "confidence": round(w.confidence, 4),
            "vad_index": w.vad_index,
        }
        for w in windows
    ]


def language_blocks_to_json(
    blocks: list[LanguageBlock],
) -> list[dict[str, Any]]:
    result = []
    for block in blocks:
        result.append(
            {
                "start": round(block.start, 3),
                "end": round(block.end, 3),
                "duration": round(block.duration, 3),
                "language": block.language,
                "confidence": round(block.confidence, 4),
                "boundary": block.boundary,
            }
        )
    return result
