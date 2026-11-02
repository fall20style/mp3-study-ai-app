"""PyTorch Whisper 기반 음성 인식.

GPU가 없는 노트북을 기준으로 CPU에서 동작하도록 기본 설정되어 있습니다.
`--device auto`를 주면 CUDA가 있을 때만 GPU를 사용합니다.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any

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
    try:
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

    segments: list[ASRSegment] = []
    for seg in result.get("segments", []):
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        no_speech = seg.get("no_speech_prob")
        avg_logprob = seg.get("avg_logprob")
        # 무음/환각으로 보이는 구간은 버립니다.
        if (
            no_speech is not None
            and no_speech > 0.85
            and avg_logprob is not None
            and avg_logprob < -0.8
        ):
            log.debug("무음 추정 구간 제외: %.1f-%.1f %r", seg["start"], seg["end"], text)
            continue

        words: list[Word] = []
        for w in seg.get("words", []) or []:
            wtext = str(w.get("word", w.get("text", ""))).strip()
            if not wtext:
                continue
            ws = float(w.get("start", seg["start"]))
            we = float(w.get("end", seg["end"]))
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
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=text,
                words=words,
                no_speech_prob=float(no_speech) if no_speech is not None else None,
                avg_logprob=float(avg_logprob) if avg_logprob is not None else None,
            )
        )

    if not segments:
        raise ASRError("음성에서 문장을 인식하지 못했습니다. 오디오와 모델 설정을 확인하세요.")

    return ASRResult(
        language=str(result.get("language", lang or "unknown")),
        duration=audio.size / 16_000,
        model=model_name,
        device=dev,
        segments=segments,
    )
