"""문장 끝이 미세하게 잘린 cue를 찾아 종료 시간을 늘립니다.

Whisper의 단어 타임스탬프는 마지막 자음이나 여운보다 조금 일찍 끝나는
경우가 있습니다. 이 모듈은 cue 종료 직후에 이어지는 짧은 음성 에너지만
감지해서 종료 시간을 연장합니다.

- 실제로 소리가 나던 문장만 대상으로 합니다. (끝이 무음이면 건드리지 않음)
- 다음 cue의 시작은 침범하지 않습니다.
- 무음 뒤에 떨어져 있는 다른 발화는 이어 붙이지 않습니다.
- 연장 폭이 너무 크면(문장이 통째로 빠진 경우) 건드리지 않습니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class TimedCue(Protocol):
    """`index`, `start`, `end`만 있으면 Cue/CueRecord 모두 사용할 수 있습니다."""

    index: int
    start: float
    end: float


@dataclass(frozen=True)
class CueSpan:
    """분석용으로 시각 정보만 복사한 cue."""

    index: int
    start: float
    end: float


@dataclass(frozen=True)
class TailFix:
    """한 문장의 종료 시간 보정 결과."""

    index: int
    old_end: float
    new_end: float

    @property
    def extension(self) -> float:
        return max(0.0, self.new_end - self.old_end)


@dataclass(frozen=True)
class TailFixConfig:
    """끝 잘림 감지 기준.

    기본값은 16 kHz 한국어/영어 강의·시험 음원에서 조정한 값입니다.
    """

    frame_seconds: float = 0.02
    hop_seconds: float = 0.01
    min_extension: float = 0.05
    max_extension: float = 0.70
    tail_padding: float = 0.04
    inside_window: float = 0.30
    speech_gate_db: float = -50.0
    noise_floor_percentile: float = 10.0
    noise_margin_db: float = 8.0
    min_threshold_db: float = -50.0
    start_tolerance: float = 0.03
    max_silence_gap: float = 0.04
    next_gap: float = 0.03
    scan_margin: float = 0.30


def _frame_levels(
    audio: np.ndarray,
    sample_rate: int,
    config: TailFixConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """프레임별 RMS를 dB로 계산해 (dB, 시작 시각)을 돌려줍니다."""

    frame_len = max(1, int(round(config.frame_seconds * sample_rate)))
    hop_len = max(1, int(round(config.hop_seconds * sample_rate)))
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)

    if samples.size < frame_len:
        padded = np.zeros(frame_len, dtype=np.float64)
        padded[: samples.size] = samples
        energy = np.array([np.mean(padded * padded)])
        return 10.0 * np.log10(np.maximum(energy, 1e-18)), np.array([0.0])

    frame_count = 1 + (samples.size - frame_len) // hop_len
    levels = np.empty(frame_count, dtype=np.float64)
    times = np.arange(frame_count, dtype=np.float64) * hop_len / float(sample_rate)

    # 긴 음원에서도 메모리가 급증하지 않도록 프레임을 나눠 계산합니다.
    frames_per_chunk = max(1, 2_000_000 // frame_len)
    for offset in range(0, frame_count, frames_per_chunk):
        count = min(frames_per_chunk, frame_count - offset)
        first = offset * hop_len
        last = first + (count - 1) * hop_len + frame_len
        block = samples[first:last]
        windows = np.lib.stride_tricks.sliding_window_view(block, frame_len)
        windows = windows[::hop_len][:count].astype(np.float64)
        windows *= windows
        energy = windows.mean(axis=1)
        levels[offset : offset + count] = 10.0 * np.log10(
            np.maximum(energy, 1e-18)
        )

    return levels, times


def _percentile(values: np.ndarray, percentile: float, default: float) -> float:
    if values.size == 0:
        return default
    return float(np.percentile(values, percentile))


def analyze_tail_clipping(
    audio: np.ndarray,
    cues: Sequence[TimedCue],
    *,
    sample_rate: int = 16_000,
    config: TailFixConfig | None = None,
) -> list[TailFix]:
    """끝이 미세하게 잘린 cue를 찾아 연장할 종료 시간을 반환합니다.

    입력 cue는 수정하지 않습니다. 실제 반영은 호출하는 쪽에서
    `fix.new_end`를 cue.end에 넣으면 됩니다.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate는 0보다 커야 합니다.")
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0 or not cues:
        return []

    config = config or TailFixConfig()
    if config.frame_seconds <= 0 or config.hop_seconds <= 0:
        raise ValueError("frame_seconds와 hop_seconds는 0보다 커야 합니다.")
    if config.max_extension < config.min_extension:
        raise ValueError("max_extension은 min_extension보다 커야 합니다.")

    levels, times = _frame_levels(samples, sample_rate, config)
    if levels.size == 0:
        return []

    duration = float(samples.size) / float(sample_rate)
    noise_floor = _percentile(levels, config.noise_floor_percentile, -180.0)
    threshold = max(noise_floor + config.noise_margin_db, config.min_threshold_db)

    fixes: list[TailFix] = []
    for position, cue in enumerate(cues):
        start = float(cue.start)
        end = float(cue.end)
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue

        next_start = (
            float(cues[position + 1].start)
            if position + 1 < len(cues)
            else duration
        )
        if not np.isfinite(next_start):
            next_start = duration

        # 다음 cue 시작 바로 앞까지만 본다.
        scan_end = min(
            next_start - config.next_gap,
            end + config.max_extension + config.scan_margin,
        )
        if scan_end - end < config.min_extension:
            continue

        after_mask = (times >= end - 1e-6) & (times < scan_end)
        after_db = levels[after_mask]
        after_times = times[after_mask]
        if after_db.size < 3:
            continue

        # 끝부분이 실제 발화였는지 확인합니다. 끝이 무음인 cue는 제외합니다.
        inside_mask = (times >= max(start, end - config.inside_window)) & (
            times + config.frame_seconds <= end + 1e-6
        )
        inside_db = levels[inside_mask]
        if inside_db.size == 0:
            continue
        if _percentile(inside_db, 90.0, -180.0) < config.speech_gate_db:
            continue

        above = after_db > threshold
        tolerance_frames = max(
            1, int(round(config.start_tolerance / config.hop_seconds))
        )
        first: int | None = None
        for index in range(min(tolerance_frames + 1, above.size)):
            if bool(above[index]):
                first = index
                break
        if first is None:
            # 종료 직후가 무음이면 자연스러운 문장 끝으로 봅니다.
            continue

        last = first
        index = first + 1
        while index < above.size:
            if bool(above[index]):
                last = index
            elif (index - last) * config.hop_seconds > config.max_silence_gap:
                break
            index += 1

        raw_extension = (float(after_times[last]) + config.frame_seconds) - end
        if raw_extension < config.min_extension:
            continue
        if raw_extension > config.max_extension:
            # 0.7초보다 길게 이어지면 단순한 꼬리 잘림이 아니라
            # 다음 발화가 통째로 빠진 것으로 보고 건드리지 않습니다.
            continue

        limit = end + config.max_extension
        if next_start > end:
            limit = min(limit, next_start - config.next_gap)
        new_end = min(end + raw_extension + config.tail_padding, limit)
        new_end = round(new_end, 3)
        old_end = round(end, 3)
        if new_end - old_end < config.min_extension:
            continue

        fixes.append(
            TailFix(index=int(cue.index), old_end=old_end, new_end=new_end)
        )

    return fixes
