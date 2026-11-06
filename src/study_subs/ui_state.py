"""프로젝트별 GUI 상태를 저장합니다.

지금은 문장 목록의 마지막 선택 위치만 기록합니다. 상태 파일은 프로젝트
폴더 안에 숨김 파일로 두어, 같은 프로젝트를 다시 열면 이어서 볼 수 있게
합니다. 파일이 없거나 손상되어도 GUI 동작에는 영향을 주지 않습니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STATE_FILE = ".ui_state.json"


def state_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / STATE_FILE


def load_last_index(project_dir: str | Path) -> int | None:
    """저장된 마지막 선택 위치를 돌려줍니다. 없거나 잘못됐으면 None."""

    try:
        data = json.loads(state_path(project_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("last_index")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def save_last_index(project_dir: str | Path, index: int) -> None:
    """마지막 선택 위치를 저장합니다.

    프로젝트 폴더가 읽기 전용이거나 저장에 실패해도 학습을 막지 않도록
    예외를 밖으로 내보내지 않습니다.
    """

    if index < 0:
        return
    path = state_path(project_dir)
    data: dict[str, Any] = {}
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(current, dict):
            data.update(current)
    except (OSError, ValueError):
        pass

    data["last_index"] = int(index)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
