"""음원과 생성 파일을 학습 프로젝트 폴더로 모으고 project.json을 만듭니다.

예:
    python make_project.py `
      --name "2026학년도 수능 영어 듣기" `
      --audio data/lecture.mp3 `
      --out-dir output

    (--project-dir 생략 시 data/project/<음원 파일명>/)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from study_subs.project import create_project, default_project_root  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="학습 프로젝트 폴더 만들기")
    parser.add_argument("--audio", required=True, help="mp3/wav/m4a 음원 경로")
    parser.add_argument("--out-dir", default="output", help="자막 생성 결과 폴더")
    parser.add_argument(
        "--project-dir",
        help="프로젝트 폴더 (기본: data/project/<음원 파일명>)",
    )
    parser.add_argument("--name", help="프로젝트 이름 (기본: 음원 파일명)")
    parser.add_argument("--created-at", help="생성일 (기본: 현재 시각)")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="원본을 옮기지 않고 복사합니다.",
    )
    args = parser.parse_args(argv)

    audio = Path(args.audio)
    project_dir = (
        Path(args.project_dir)
        if args.project_dir
        else default_project_root() / audio.stem
    )
    name = args.name or audio.stem

    project = create_project(
        project_dir,
        name=name,
        audio=audio,
        out_dir=Path(args.out_dir),
        created_at=args.created_at,
        move=not args.copy,
    )
    print(f"프로젝트 생성: {project.directory}")
    print(f"  이름: {project.name}")
    print(f"  생성일: {project.created_at}")
    print(f"  문장: {project.cue_count}개")
    print(f"  음원: {project.audio.name}")
    print("  실행: python gui.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
