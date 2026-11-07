"""명령줄 인터페이스."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from . import __version__
from .asr import ASRError
from .audio import AudioError
from .pipeline import RunConfig, run
from .subtitles import format_mmss
from .translate import TranslationError

log = logging.getLogger(__name__)

FORMATS = {"srt", "vtt", "txt", "md", "csv", "json"}


def _parse_formats(value: str) -> tuple[str, ...]:
    items = tuple(
        dict.fromkeys(x.strip().lower() for x in value.split(",") if x.strip())
    )
    if not items:
        raise argparse.ArgumentTypeError("출력 형식을 하나 이상 지정하세요.")
    unknown = set(items) - FORMATS
    if unknown:
        raise argparse.ArgumentTypeError(
            f"지원하지 않는 형식: {', '.join(sorted(unknown))} "
            f"(가능: {', '.join(sorted(FORMATS))})"
        )
    return items


def _parse_device(value: str) -> str:
    value = value.strip().lower()
    if value in ("auto", "cpu", "cuda", "mps"):
        return value
    if value == "xpu" or re.fullmatch(r"xpu:[0-9]+", value):
        return value
    raise argparse.ArgumentTypeError(
        "장치는 auto, cpu, cuda, xpu, xpu:N, mps 중 하나여야 합니다."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="study-subs",
        description=(
            "영어 음성 파일을 Whisper(PyTorch)로 인식하고 한국어로 번역해 "
            "'[분:초] 영어 / 한국어' 자막을 만듭니다. CUDA/Intel XPU가 있으면 "
            "자동으로 GPU를 사용하고, 없으면 CPU로 동작합니다."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            "  python main.py data/ksat_2026_english.mp3\n"
            "  python main.py data/ksat_2026_english.mp3 --model base --out-dir output\n"
            "  python main.py data/ksat_2026_english.mp3 --translator none --formats srt,txt\n"
            "  python main.py data/ksat_2026_english.mp3 --translator nllb  # 오프라인\n"
        ),
    )
    p.add_argument("input", type=Path, help="입력 오디오 파일 (mp3, wav, m4a 등)")
    p.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path("output"),
        help="출력 폴더 (기본: output)",
    )
    p.add_argument(
        "--model",
        default="small",
        help="Whisper 모델: tiny, base, small, medium, large-v3 (기본: small)",
    )
    p.add_argument(
        "--language",
        default="en",
        help="음성 언어. 'auto'면 자동 감지 (기본: en)",
    )
    p.add_argument(
        "--device",
        default="auto",
        type=_parse_device,
        metavar="{auto,cpu,cuda,xpu,xpu:N,mps}",
        help=(
            "추론 장치 (기본: auto → CUDA/Intel XPU/MPS 중 사용 가능한 GPU, "
            "없으면 CPU). Intel GPU가 여러 개면 xpu:0, xpu:1로 지정"
        ),
    )
    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="PyTorch CPU 스레드 수 (0: 논리 코어 전체)",
    )
    p.add_argument(
        "--beam-size",
        type=int,
        default=1,
        choices=(1, 2, 3, 4, 5),
        help="Whisper beam search 크기. 1이 가장 빠름 (기본: 1)",
    )
    p.add_argument(
        "--translator",
        default="google",
        choices=("google", "nllb", "none"),
        help="번역기: google(온라인, 기본), nllb(오프라인), none(영어만)",
    )
    p.add_argument(
        "--translate-workers",
        type=int,
        default=3,
        help="Google 번역 동시 요청 수 (기본: 3, 최대 8)",
    )
    p.add_argument(
        "--nllb-model",
        default="facebook/nllb-200-distilled-600M",
        help="--translator nllb일 때 사용할 모델",
    )
    p.add_argument(
        "--max-cue-seconds",
        type=float,
        default=12.0,
        help="한 자막의 최대 길이(초). 길면 쉼표/단어 경계에서 분할 (기본: 12)",
    )
    p.add_argument(
        "--max-cue-chars",
        type=int,
        default=180,
        help="한 자막의 최대 글자 수 (기본: 180)",
    )
    p.add_argument(
        "--formats",
        type=_parse_formats,
        default=("srt", "md", "txt", "csv", "json"),
        help="출력 형식 쉼표 목록 (기본: srt,md,txt,csv,json)",
    )
    p.add_argument(
        "--srt-mode",
        default="bilingual",
        choices=("bilingual", "english", "korean"),
        help="SRT/VTT 자막 언어 (기본: bilingual)",
    )
    p.add_argument(
        "--title",
        default=None,
        help="출력물 제목 (기본: 파일명)",
    )
    p.add_argument(
        "--initial-prompt",
        default=None,
        help="Whisper에 줄 초기 힌트 문장 (고유명사/말투 유도용, 선택)",
    )
    p.add_argument(
        "--force-asr",
        action="store_true",
        help="ASR 캐시를 무시하고 다시 인식",
    )
    p.add_argument(
        "--force-translate",
        action="store_true",
        help="번역 캐시를 무시하고 다시 번역",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="진행 로그와 Whisper 세그먼트 출력을 최소화",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="디버그 로그와 상세 오류 출력",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _setup_logging(quiet: bool, debug: bool) -> None:
    level = logging.DEBUG if debug else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    # Windows 콘솔(cp949)에서도 한글/영어가 깨지지 않도록 UTF-8로 전환
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.quiet, args.debug)

    if args.translate_workers < 1:
        parser.error("--translate-workers는 1 이상이어야 합니다.")
    if args.max_cue_seconds <= 0:
        parser.error("--max-cue-seconds는 0보다 커야 합니다.")
    if args.max_cue_chars < 20:
        parser.error("--max-cue-chars는 20 이상이어야 합니다.")
    if not args.input.exists():
        parser.error(f"입력 파일이 없습니다: {args.input}")

    config = RunConfig(
        input_path=args.input,
        out_dir=args.out_dir,
        model=args.model,
        language=args.language,
        device=args.device,
        threads=args.threads,
        beam_size=args.beam_size,
        translator=args.translator,
        translate_workers=args.translate_workers,
        nllb_model=args.nllb_model,
        max_cue_seconds=args.max_cue_seconds,
        max_cue_chars=args.max_cue_chars,
        formats=args.formats,
        srt_mode=args.srt_mode,
        force_asr=args.force_asr,
        force_translate=args.force_translate,
        title=args.title,
        initial_prompt=args.initial_prompt,
        whisper_verbose=True if args.debug else (None if args.quiet else False),
    )

    try:
        paths, cues, meta = run(config)
    except KeyboardInterrupt:
        print("\n중단되었습니다.", file=sys.stderr)
        return 130
    except (AudioError, ASRError, TranslationError, FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        if args.debug:
            raise
        return 1
    except Exception as exc:  # noqa: BLE001
        log.error("예상치 못한 오류: %s", exc)
        if args.debug:
            raise
        return 1

    print()
    print(f"완료: {len(cues)}개 문장 → {Path(args.out_dir)}")
    for fmt, path in paths.items():
        print(f"  {fmt:4s} {path}")
    print()
    print("미리보기:")
    for cue in cues[:5]:
        print(f"  [{format_mmss(cue.start)}] {cue.text}")
        if cue.translation:
            print(f"           {cue.translation}")
    if len(cues) > 5:
        print(f"  ... 외 {len(cues) - 5}개")
    return 0
