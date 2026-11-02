"""프로젝트 루트에서 실행하는 진입점.

    python main.py data/ksat_2026_english.mp3
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from study_subs.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
