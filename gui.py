"""영어 듣기 학습 GUI 진입점.

    python gui.py
    python gui.py --project data/project/ksat_2026_english
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from study_subs.gui import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
