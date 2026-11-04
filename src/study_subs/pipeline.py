"""전체 파이프라인: 디코딩 → Whisper → 문장 묶기 → 번역 → 자막 출력."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .asr import ASRResult, transcribe
from .audio import audio_duration, decode_audio
from .sentences import Cue, build_cues
from .subtitles import now_text, write_outputs
from .translate import build_translator

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float | None], None]
CancelFn = Callable[[], bool]


class CancelledError(RuntimeError):
    """사용자가 프로젝트 생성을 취소했습니다."""


def _emit(progress: ProgressFn | None, message: str, fraction: float | None) -> None:
    if progress is None:
        return
    try:
        progress(message, fraction)
    except Exception:  # noqa: BLE001 - 진행률 표시 실패가 작업을 막지 않게
        pass


def _cancelled(cancel: CancelFn | None) -> bool:
    return bool(cancel and cancel())


@dataclass
class RunConfig:
    input_path: Path
    out_dir: Path = Path("output")
    model: str = "base"
    language: str = "en"
    device: str = "auto"
    threads: int = 0
    beam_size: int = 1
    translator: str = "google"
    translate_workers: int = 3
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    max_cue_seconds: float = 12.0
    max_cue_chars: int = 180
    formats: tuple[str, ...] = ("srt", "md", "txt", "csv", "json")
    srt_mode: str = "bilingual"
    force_asr: bool = False
    force_translate: bool = False
    title: str | None = None
    initial_prompt: str | None = None
    whisper_verbose: bool | None = False


def _safe_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name).strip(" .")
    return cleaned or "subtitle"


def _fingerprint(path: Path, config: RunConfig) -> dict:
    stat = path.stat()
    return {
        "version": 3,
        "size": stat.st_size,
        "mtime": round(stat.st_mtime, 3),
        "model": config.model,
        "language": config.language,
        "beam_size": config.beam_size,
        "initial_prompt": config.initial_prompt,
    }


def _load_asr_cache(path: Path, fingerprint: dict) -> ASRResult | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("fingerprint") != fingerprint:
            log.info("ASR 캐시가 현재 설정/파일과 달라 새로 인식합니다.")
            return None
        return ASRResult.from_dict(data["result"])
    except Exception as exc:
        log.warning("ASR 캐시를 읽지 못했습니다(%s): %s", path, exc)
        return None


def _save_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    tmp.replace(path)


def run(
    config: RunConfig,
    *,
    progress: ProgressFn | None = None,
    cancel: CancelFn | None = None,
) -> tuple[dict[str, Path], list[Cue], dict]:
    input_path = Path(config.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {input_path}")

    stem = _safe_name(input_path.stem)
    out_dir = Path(config.out_dir)
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    _emit(progress, "오디오 준비 중…", 0.02)
    log.info("[1/4] 오디오 디코딩: %s", input_path)
    audio = decode_audio(input_path, timeout=900)
    duration = audio_duration(audio)
    log.info("      오디오 길이: %s (%.1f초)", _mmss(duration), duration)
    _emit(progress, "오디오 준비 완료", 0.04)

    fingerprint = _fingerprint(input_path, config)
    asr_cache = (
        cache_dir
        / f"{stem}.{_safe_name(config.model)}.{_safe_name(config.language)}.asr.json"
    )
    asr_result = None if config.force_asr else _load_asr_cache(asr_cache, fingerprint)

    if asr_result is None:
        log.info(
            "[2/4] Whisper 음성 인식 (model=%s, language=%s, device=%s)",
            config.model,
            config.language,
            config.device,
        )
        _emit(progress, "Whisper 모델 준비 중…", 0.06)
        t0 = time.time()
        asr_result = transcribe(
            audio,
            model_name=config.model,
            language=config.language,
            device=config.device,
            threads=config.threads,
            beam_size=config.beam_size,
            initial_prompt=config.initial_prompt,
            verbose=config.whisper_verbose,
            progress=(
                None
                if progress is None
                else lambda f: _emit(progress, "음성 인식 중…", 0.06 + 0.77 * f)
            ),
        )
        _save_json_atomic(
            asr_cache,
            {"fingerprint": fingerprint, "result": asr_result.to_dict()},
        )
        log.info(
            "      인식 완료: %d개 구간, %.1f초 소요 (캐시: %s)",
            len(asr_result.segments),
            time.time() - t0,
            asr_cache,
        )
    else:
        log.info(
            "[2/4] ASR 캐시 사용: %d개 구간 (%s)",
            len(asr_result.segments),
            asr_cache,
        )
    _emit(progress, "음성 인식 완료", 0.84)
    if _cancelled(cancel):
        raise CancelledError("사용자가 프로젝트 생성을 취소했습니다.")

    log.info("[3/4] 문장 단위 자막 생성")
    _emit(progress, "문장 정리 중…", 0.86)
    cues = build_cues(
        asr_result.segments,
        duration=duration,
        max_seconds=config.max_cue_seconds,
        max_chars=config.max_cue_chars,
    )
    log.info("      %d개 문장 cue", len(cues))
    _emit(progress, "문장 정리 완료", 0.88)

    translator_label = "none"
    if config.translator.lower() not in ("none", "off", "no"):
        log.info("[4/4] 한국어 번역 (%s)", config.translator)
        translation_cache = (
            cache_dir / f"{stem}.{config.translator.lower()}.ko.json"
        )
        if config.force_translate:
            translation_cache.unlink(missing_ok=True)
        _emit(progress, "번역 모델 준비 중…", 0.90)
        translator = build_translator(
            config.translator,
            cache_path=translation_cache,
            workers=config.translate_workers,
            nllb_model=config.nllb_model,
            device=config.device,
            threads=config.threads,
        )
        translator_label = translator.label
        t0 = time.time()
        english_cues = [cue for cue in cues if cue.lang == "en"]
        _emit(progress, "한국어 번역 중…", 0.92)
        translations = translator.translate_batch(
            [cue.text for cue in english_cues],
            progress=(
                None
                if progress is None
                else lambda done, total: _emit(
                    progress,
                    f"한국어 번역 중… {done}/{total}",
                    0.92 + 0.05 * min(1.0, done / max(1, total)),
                )
            ),
        )
        for cue, ko in zip(english_cues, translations):
            cue.translation = ko
        missing = sum(1 for cue in english_cues if not cue.translation.strip())
        log.info(
            "      번역 완료: %d/%d 문장, %.1f초 소요 (한국어 안내 %d개는 그대로 유지)",
            len(english_cues) - missing,
            len(english_cues),
            time.time() - t0,
            len(cues) - len(english_cues),
        )
        if missing:
            log.warning(
                "      번역이 비어 있는 cue가 %d개 있습니다. 다시 실행하면 캐시를 재사용합니다.",
                missing,
            )
    else:
        log.info("[4/4] 번역 건너뜀 (--translator none)")

    _emit(progress, "번역 완료", 0.97)
    if _cancelled(cancel):
        raise CancelledError("사용자가 프로젝트 생성을 취소했습니다.")

    meta = {
        "title": config.title or input_path.stem,
        "source": str(input_path.resolve()),
        "model": asr_result.model,
        "device": asr_result.device,
        "language": asr_result.language,
        "translator": translator_label,
        "duration": round(duration, 3),
        "generated_at": now_text(),
        "cue_count": len(cues),
    }

    log.info("자막 파일 저장: %s", out_dir)
    _emit(progress, "자막 파일 저장 중…", 0.98)
    paths = write_outputs(
        cues,
        out_dir,
        stem,
        formats=config.formats,
        meta=meta,
        srt_mode=config.srt_mode,
    )
    _emit(progress, "완료", 1.0)
    log.info("완료: %.1f초 소요", time.time() - started)
    return paths, cues, meta


def _mmss(seconds: float) -> str:
    total = max(0, int(seconds))
    m, s = divmod(total, 60)
    return f"{m:02d}:{s:02d}"
