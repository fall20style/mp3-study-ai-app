"""PyTorch Whisper 기반 음성 인식.

GPU가 없는 노트북을 기준으로 CPU에서 동작하도록 기본 설정되어 있습니다.
`--device auto`를 주면 CUDA가 있을 때만 GPU를 사용합니다.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)


class ASRError(RuntimeError):
    pass


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float | None = None


@dataclass
class ASRSegment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    no_speech_prob: float | None = None
    avg_logprob: float | None = None


@dataclass
class ASRResult:
    language: str
    duration: float
    model: str
    device: str
    segments: list[ASRSegment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ASRResult":
        segments = []
        for seg in data.get("segments", []):
            words = [Word(**w) for w in seg.get("words", [])]
            segments.append(
                ASRSegment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=str(seg.get("text", "")),
                    words=words,
                    no_speech_prob=seg.get("no_speech_prob"),
                    avg_logprob=seg.get("avg_logprob"),
                )
            )
        return cls(
            language=str(data.get("language", "")),
            duration=float(data.get("duration", 0.0)),
            model=str(data.get("model", "")),
            device=str(data.get("device", "cpu")),
            segments=segments,
        )


def resolve_device(requested: str = "auto") -> str:
    """'auto' | 'cpu' | 'cuda' | 'mps' → 실제 사용할 device."""

    import torch

    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA를 사용할 수 없어 CPU로 진행합니다.")
        return "cpu"
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if not (mps and mps.is_available()):
            log.warning("MPS를 사용할 수 없어 CPU로 진행합니다.")
            return "cpu"
    return requested


def configure_threads(threads: int = 0) -> int:
    """PyTorch CPU 스레드 수를 설정합니다. 0이면 논리 코어 수를 사용합니다."""

    import torch

    n = int(threads) if threads and threads > 0 else (os.cpu_count() or 4)
    try:
        torch.set_num_threads(n)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, n // 2))
    except Exception:
        # 이미 초기화된 경우 재설정할 수 없습니다.
        pass
    return n


_SAMPLE_RATE = 16_000


def _segments_from_result(
    result: dict[str, Any], *, offset: float = 0.0
) -> list[ASRSegment]:
    """Whisper 결과 dict를 ASRSegment 목록으로 변환합니다."""

    segments: list[ASRSegment] = []
    for seg in result.get("segments", []):
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = float(seg["start"]) + offset
        end = float(seg["end"]) + offset
        no_speech = seg.get("no_speech_prob")
        avg_logprob = seg.get("avg_logprob")
        # 무음/환각으로 보이는 구간은 버립니다.
        if (
            no_speech is not None
            and no_speech > 0.85
            and avg_logprob is not None
            and avg_logprob < -0.8
        ):
            log.debug("무음 추정 구간 제외: %.1f-%.1f %r", start, end, text)
            continue

        words: list[Word] = []
        for w in seg.get("words", []) or []:
            wtext = str(w.get("word", w.get("text", ""))).strip()
            if not wtext:
                continue
            ws = float(w.get("start", seg["start"])) + offset
            we = float(w.get("end", seg["end"])) + offset
            if we < ws:
                ws, we = we, ws
            words.append(
                Word(
                    start=ws,
                    end=we,
                    text=wtext,
                    probability=float(w["probability"]) if w.get("probability") is not None else None,
                )
            )

        segments.append(
            ASRSegment(
                start=start,
                end=end,
                text=text,
                words=words,
                no_speech_prob=float(no_speech) if no_speech is not None else None,
                avg_logprob=float(avg_logprob) if avg_logprob is not None else None,
            )
        )
    return segments


def _speech_regions(
    audio: np.ndarray,
    start: float,
    end: float,
    *,
    frame_seconds: float = 0.25,
    min_speech: float = 1.5,
    merge_gap: float = 1.2,
) -> list[tuple[float, float]]:
    """무음 구간 안에서 에너지 기준으로 말소리 구간을 찾습니다.

    Whisper가 30초 창 앞쪽의 긴 무음에 걸려 뒤쪽의 한국어 안내를
    통째로 건너뛰는 경우가 있어, 그 구간만 다시 인식하는 데 씁니다.
    """

    lo = max(0.0, start)
    hi = min(len(audio) / _SAMPLE_RATE, end)
    if hi - lo < min_speech:
        return []

    frame = max(1, int(frame_seconds * _SAMPLE_RATE))
    first = int(lo * _SAMPLE_RATE)
    last = int(hi * _SAMPLE_RATE)
    dbs: list[float] = []
    for pos in range(first, last - frame + 1, frame):
        chunk = audio[pos : pos + frame]
        rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
        dbs.append(20.0 * np.log10(max(rms, 1e-9)))
    if not dbs:
        return []

    values = np.asarray(dbs)
    floor = float(np.percentile(values, 10))
    # 디지털 무음(-180dB)부터 잡음이 있는 녹음까지 어느 정도 맞춥니다.
    threshold = min(max(-45.0, floor + 12.0), -28.0)
    voiced = values > threshold

    runs: list[tuple[float, float]] = []
    i = 0
    while i < len(voiced):
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j < len(voiced) and voiced[j]:
            j += 1
        rs = lo + i * frame_seconds
        re = min(hi, lo + j * frame_seconds)
        if runs and rs - runs[-1][1] <= merge_gap:
            runs[-1] = (runs[-1][0], re)
        else:
            runs.append((rs, re))
        i = j

    regions: list[tuple[float, float]] = []
    for rs, re in runs:
        if re - rs < min_speech:
            continue
        # Whisper는 한 번에 30초까지만 처리합니다.
        while re - rs > 28.0:
            regions.append((rs, rs + 28.0))
            rs += 28.0
        regions.append((rs, re))
    return regions


def _fill_skipped_gaps(
    model: Any,
    audio: np.ndarray,
    segments: list[ASRSegment],
    *,
    language: str | None,
    beam_size: int,
) -> list[ASRSegment]:
    """Whisper가 건너뛴 긴 무음 구간에서 말소리만 골라 다시 인식합니다."""

    if len(segments) < 2:
        return []

    bounds = [(0.0, segments[0].start)]
    for prev, nxt in zip(segments, segments[1:]):
        bounds.append((prev.end, nxt.start))
    bounds.append((segments[-1].end, len(audio) / _SAMPLE_RATE))

    added: list[ASRSegment] = []
    for gap_start, gap_end in bounds:
        if gap_end - gap_start < 6.0:
            continue
        for rs, re in _speech_regions(audio, gap_start, gap_end):
            chunk = audio[int(rs * _SAMPLE_RATE) : int(re * _SAMPLE_RATE)]
            if chunk.size == 0:
                continue
            result = model.transcribe(
                chunk,
                language=language,
                task="transcribe",
                fp16=False,
                verbose=None,
                word_timestamps=True,
                condition_on_previous_text=False,
                temperature=0.0,
                beam_size=beam_size,
                no_speech_threshold=0.9,
                logprob_threshold=None,
                initial_prompt=None,
            )
            for seg in _segments_from_result(result, offset=rs):
                # 음악/잡음에서 나온 환각을 걸러냅니다.
                if seg.no_speech_prob is not None and seg.no_speech_prob > 0.75:
                    continue
                if seg.avg_logprob is not None and seg.avg_logprob < -1.0:
                    continue
                seg.start = max(gap_start, seg.start)
                seg.end = min(gap_end, seg.end)
                if seg.end - seg.start < 0.4:
                    continue
                added.append(seg)
                log.info(
                    "      건너뛴 구간 재인식: %.1f-%.1f %r",
                    seg.start,
                    seg.end,
                    seg.text,
                )
    return added


def _hangul_ratio(text: str) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(1 for ch in chars if "\uac00" <= ch <= "\ud7a3") / len(chars)


def _rescue_sparse_segments(
    model: Any,
    audio: np.ndarray,
    segments: list[ASRSegment],
    *,
    beam_size: int,
) -> list[ASRSegment]:
    """짧은 영어가 긴 구간을 차지한 환각을 한국어 안내로 다시 인식합니다.

    Whisper가 한국어 안내를 영어 문장 하나로 덮어쓰는 경우(예: 28초 구간에
    "The answer is the answer.")가 있어, 단어 밀도가 비정상적으로 낮고
    무음 확률이 높은 구간만 한국어로 다시 읽습니다.
    """

    rescued: list[ASRSegment] = []
    for seg in segments:
        duration = seg.end - seg.start
        words = len(seg.text.split())
        if duration < 12.0 or words / max(duration, 1e-6) >= 0.5:
            continue
        if _hangul_ratio(seg.text) > 0.1:
            continue
        if seg.no_speech_prob is not None and seg.no_speech_prob < 0.5:
            continue

        total = len(audio) / _SAMPLE_RATE
        window_end = min(total, seg.end + 1.5)
        # 긴 무음이 앞에 있으면 Whisper가 30초 창 끝에서 안내를 잘라 먹으므로,
        # 말소리 구간만 골라 짧게 다시 읽습니다.
        regions = _speech_regions(
            audio, seg.start, window_end, min_speech=1.5
        ) or [(seg.start, window_end)]

        candidates: list[ASRSegment] = []
        for region_start, region_end in regions:
            lo = max(0.0, region_start - 0.5)
            hi = min(total, region_end + 0.5)
            chunk = audio[int(lo * _SAMPLE_RATE) : int(hi * _SAMPLE_RATE)]
            if chunk.size == 0:
                continue
            result = model.transcribe(
                chunk,
                language="ko",
                task="transcribe",
                fp16=False,
                verbose=None,
                word_timestamps=True,
                condition_on_previous_text=False,
                temperature=0.0,
                beam_size=beam_size,
                no_speech_threshold=0.9,
                logprob_threshold=None,
                initial_prompt=None,
            )
            for item in _segments_from_result(result, offset=lo):
                if (
                    _hangul_ratio(item.text) >= 0.3
                    and item.end - item.start >= 0.4
                ):
                    candidates.append(item)
        if not candidates:
            continue
        merged: list[ASRSegment] = []
        for item in candidates:
            item.start = max(seg.start, item.start)
            item.end = min(window_end, item.end)
            if item.end - item.start < 0.5:
                continue
            if merged and item.start - merged[-1].end <= 0.4:
                merged[-1].text = f"{merged[-1].text} {item.text}".strip()
                merged[-1].end = item.end
                merged[-1].words.extend(item.words)
            else:
                merged.append(item)
        if not merged:
            continue
        rescued.extend(merged)
        log.info(
            "      영어로 덮인 한국어 안내 복구: %.1f-%.1f %r",
            seg.start,
            seg.end,
            " ".join(item.text for item in merged),
        )

    if not rescued:
        return segments

    kept = [
        seg
        for seg in segments
        if not (
            seg.end - seg.start >= 12.0
            and len(seg.text.split()) / max(seg.end - seg.start, 1e-6) < 0.5
            and _hangul_ratio(seg.text) <= 0.1
            and seg.no_speech_prob is not None
            and seg.no_speech_prob >= 0.5
            and any(
                item.start < seg.end and item.end > seg.start for item in rescued
            )
        )
    ]
    return sorted(kept + rescued, key=lambda s: s.start)


def _progress_tqdm(callback: Callable[[float], None]) -> Any:
    """whisper 내부 tqdm을 감싸 진행률(0~1)을 콜백으로 전달합니다."""

    import tqdm as tqdm_module

    base = tqdm_module.tqdm

    class ProgressTqdm(base):  # type: ignore[misc, valid-type]
        def update(self, n: int = 1):  # noqa: D102
            super().update(n)
            total = float(self.total or 0.0)
            if total > 0:
                callback(min(1.0, float(self.n) / total))

    return SimpleNamespace(tqdm=ProgressTqdm)


def transcribe(
    audio: np.ndarray,
    *,
    model_name: str = "base",
    language: str | None = "en",
    device: str = "auto",
    threads: int = 0,
    beam_size: int = 1,
    initial_prompt: str | None = None,
    verbose: bool | None = False,
    progress: Callable[[float], None] | None = None,
) -> ASRResult:
    """Whisper로 전사하고 단어 단위 타임스탬프를 반환합니다."""

    try:
        import whisper
    except ImportError as exc:
        raise ASRError(
            "openai-whisper가 설치되어 있지 않습니다.\n"
            "  pip install -r requirements.txt"
        ) from exc

    configure_threads(threads)
    dev = resolve_device(device)
    fp16 = dev == "cuda"

    log.info("Whisper 모델 로드: %s (device=%s)", model_name, dev)
    try:
        model = whisper.load_model(model_name, device=dev)
    except Exception as exc:
        raise ASRError(
            f"Whisper 모델 '{model_name}'을 불러오지 못했습니다: {exc}\n"
            "모델 이름(base, small, medium, large-v3 등)과 인터넷 연결을 확인하세요."
        ) from exc

    lang = None if not language or language.lower() == "auto" else language.lower()
    log.info("음성 인식 시작 (%.1f초 오디오)", audio.size / 16_000)
    whisper_transcribe = None
    original_tqdm = None
    try:
        if progress is not None:
            import importlib

            whisper_transcribe = importlib.import_module("whisper.transcribe")
            original_tqdm = whisper_transcribe.tqdm
            whisper_transcribe.tqdm = _progress_tqdm(progress)
        result = model.transcribe(
            audio,
            language=lang,
            task="transcribe",
            fp16=fp16,
            verbose=verbose,
            word_timestamps=True,
            condition_on_previous_text=False,
            temperature=0.0,
            beam_size=beam_size,
            no_speech_threshold=0.9,
            logprob_threshold=None,
            initial_prompt=initial_prompt,
        )
    except Exception as exc:
        raise ASRError(f"Whisper 음성 인식에 실패했습니다: {exc}") from exc
    finally:
        if whisper_transcribe is not None and original_tqdm is not None:
            whisper_transcribe.tqdm = original_tqdm

    segments = _segments_from_result(result)
    if not segments:
        raise ASRError("음성에서 문장을 인식하지 못했습니다. 오디오와 모델 설정을 확인하세요.")

    # Whisper가 긴 무음 뒤의 한국어 안내를 통째로 건너뛰는 경우 보완
    added = _fill_skipped_gaps(
        model, audio, segments, language=lang, beam_size=beam_size
    )
    if added:
        segments = sorted(segments + added, key=lambda s: s.start)
        log.info("      건너뛴 구간에서 %d개 구간 추가 인식", len(added))

    # 긴 구간을 차지한 영어 환각을 한국어 안내로 되돌립니다.
    segments = _rescue_sparse_segments(
        model, audio, segments, beam_size=beam_size
    )
    if progress is not None:
        progress(1.0)

    return ASRResult(
        language=str(result.get("language", lang or "unknown")),
        duration=audio.size / 16_000,
        model=model_name,
        device=dev,
        segments=segments,
    )
