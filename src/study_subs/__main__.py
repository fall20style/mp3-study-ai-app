"""`PYTHONPATH=src python -m study_subs ...` 실행 지원."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
