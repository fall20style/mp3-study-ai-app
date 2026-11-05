"""학습 프로젝트 폴더와 project.json 메타데이터 관리.

프로젝트 폴더 하나에 음원, 자막 파일, cue JSON을 함께 두고
`project.json`에 이름/생성일/파일 목록을 기록합니다.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_FILE = "project.json"
SUBTITLE_EXTENSIONS = (".srt", ".vtt", ".txt", ".md", ".csv")


@dataclass
class CueRecord:
    index: int
    start: float
    end: float
    text: str
    translation: str = ""
    question: int | None = None
    lang: str = "en"


@dataclass
class Project:
    directory: Path
    name: str
    created_at: str
    audio: Path
    cues: list[CueRecord]
    meta: dict[str, Any]
    data: dict[str, Any]

    @property
    def cue_count(self) -> int:
        return len(self.cues)

    @property
    def duration(self) -> float:
        try:
            return float(self.meta.get("duration", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def subtitle_path(self, extension: str) -> Path | None:
        name = (self.data.get("subtitles") or {}).get(extension.lstrip("."))
        if not name:
            return None
        path = self.directory / name
        return path if path.exists() else None


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _cue_from_dict(item: dict[str, Any], fallback_index: int) -> CueRecord:
    text = item.get("english")
    if text is None:
        text = item.get("text", "")
    translation = item.get("korean")
    if translation is None:
        translation = item.get("translation", "")
    return CueRecord(
        index=int(item.get("index", fallback_index)),
        start=float(item.get("start", 0.0)),
        end=float(item.get("end", 0.0)),
        text=str(text or ""),
        translation=str(translation or ""),
        question=item.get("question"),
        lang=str(item.get("language", "en")),
    )


def load_cues(path: Path) -> tuple[list[CueRecord], dict[str, Any]]:
    data = _read_json(path)
    if isinstance(data, list):
        raw_cues = data
        meta: dict[str, Any] = {}
    else:
        raw_cues = data.get("cues", [])
        meta = data.get("meta", {}) or {}
    cues = [
        _cue_from_dict(item, i + 1)
        for i, item in enumerate(raw_cues)
        if isinstance(item, dict)
    ]
    return cues, meta


def load_project(directory: str | Path) -> Project:
    directory = Path(directory)
    project_path = directory / PROJECT_FILE
    if not project_path.exists():
        raise FileNotFoundError(f"project.json이 없습니다: {project_path}")

    data = _read_json(project_path)
    slug = str(data.get("slug") or directory.name)
    audio_name = str(data.get("audio") or f"{slug}.mp3")
    cues_name = str(data.get("cues") or f"{slug}.json")

    audio = directory / audio_name
    if not audio.exists():
        raise FileNotFoundError(f"음원 파일이 없습니다: {audio}")
    cues_path = directory / cues_name
    if not cues_path.exists():
        raise FileNotFoundError(f"cue JSON이 없습니다: {cues_path}")

    cues, meta = load_cues(cues_path)
    merged_meta = dict(meta)
    merged_meta.update(data.get("meta", {}) or {})
    return Project(
        directory=directory,
        name=str(data.get("name") or slug),
        created_at=str(data.get("created_at", "")),
        audio=audio,
        cues=cues,
        meta=merged_meta,
        data=data,
    )


def project_cues_path(project: Project) -> Path:
    """프로젝트의 cue JSON 경로를 돌려줍니다."""

    name = project.data.get("cues")
    if not name:
        slug = str(project.data.get("slug") or project.directory.name)
        name = f"{slug}.json"
    return project.directory / str(name)


def save_project_cues(project: Project) -> list[Path]:
    """현재 `project.cues`를 cue JSON과 기존 자막 파일에 다시 저장합니다.

    종료 시간만 바꾼 뒤 SRT/VTT/TXT/MD/CSV를 한 번에 갱신할 때 사용합니다.
    `project.json`의 메타데이터는 그대로 두고, cue JSON의 기존 meta도
    보존합니다.
    """

    from .sentences import Cue
    from .subtitles import (
        write_csv,
        write_json,
        write_markdown,
        write_srt,
        write_txt,
        write_vtt,
    )

    cues_path = project_cues_path(project)
    try:
        _, meta = load_cues(cues_path)
    except (FileNotFoundError, ValueError):
        meta = dict(project.meta)

    cues = [
        Cue(
            index=cue.index,
            start=cue.start,
            end=cue.end,
            text=cue.text,
            translation=cue.translation,
            question=cue.question,
            lang=cue.lang,
        )
        for cue in project.cues
    ]

    paths: list[Path] = [cues_path]
    write_json(cues, cues_path, meta)

    subtitles = project.data.get("subtitles")
    targets: list[tuple[str, str]] = []
    if isinstance(subtitles, dict) and subtitles:
        for extension in SUBTITLE_EXTENSIONS:
            fmt = extension.lstrip(".")
            name = subtitles.get(fmt)
            if name:
                targets.append((fmt, str(name)))
    else:
        # project.json에 자막 목록이 없으면 실제로 존재하는 파일만 갱신합니다.
        slug = str(project.data.get("slug") or project.directory.name)
        for extension in SUBTITLE_EXTENSIONS:
            fmt = extension.lstrip(".")
            name = f"{slug}{extension}"
            if (project.directory / name).exists():
                targets.append((fmt, name))

    for fmt, name in targets:
        path = project.directory / name
        if fmt == "srt":
            write_srt(cues, path)
        elif fmt == "vtt":
            write_vtt(cues, path)
        elif fmt == "txt":
            write_txt(cues, path, meta)
        elif fmt == "md":
            write_markdown(cues, path, meta)
        elif fmt == "csv":
            write_csv(cues, path)
        else:
            continue
        paths.append(path)

    return paths


def default_project_root(start: str | Path | None = None) -> Path:
    """`data/project`의 기본 위치를 찾습니다.

    GUI를 어디서 실행하든 같은 폴더를 쓰도록, 현재 폴더에서 위로 올라가며
    저장소 표시(pyproject.toml, .git)를 찾습니다.
    """

    here = Path(start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate / "data" / "project"
    return here / "data" / "project"


def discover_projects(base: str | Path | None = None) -> list[Project]:
    base = Path(base) if base is not None else default_project_root()
    if not base.exists():
        return []
    projects: list[Project] = []
    for path in sorted(base.iterdir()):
        if not (path / PROJECT_FILE).exists():
            continue
        try:
            projects.append(load_project(path))
        except Exception:
            continue
    projects.sort(key=lambda p: p.created_at, reverse=True)
    return projects


def _unique_destination(directory: Path, source: Path) -> Path:
    dest = directory / source.name
    if not dest.exists() or dest.resolve() == source.resolve():
        return dest
    stem, suffix = source.stem, source.suffix
    n = 2
    while True:
        candidate = directory / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def _move_into(directory: Path, source: Path) -> Path:
    dest = _unique_destination(directory, source)
    if dest.resolve() == source.resolve():
        return dest
    shutil.move(str(source), str(dest))
    return dest


def _copy_into(directory: Path, source: Path) -> Path:
    dest = _unique_destination(directory, source)
    if dest.resolve() == source.resolve():
        return dest
    shutil.copy2(source, dest)
    return dest


def create_project(
    project_dir: str | Path,
    *,
    name: str,
    audio: str | Path,
    out_dir: str | Path,
    created_at: str | None = None,
    move: bool = True,
) -> Project:
    """음원과 생성 파일을 한 폴더로 모으고 project.json을 만듭니다."""

    directory = Path(project_dir)
    directory.mkdir(parents=True, exist_ok=True)

    audio_src = Path(audio)
    if not audio_src.exists():
        raise FileNotFoundError(f"음원 파일이 없습니다: {audio_src}")

    move_fn = _move_into if move else _copy_into
    audio_dest = move_fn(directory, audio_src)
    slug = audio_dest.stem

    out = Path(out_dir)
    if out.exists() and out.resolve() != directory.resolve():
        for path in out.iterdir():
            if path.name == ".cache":
                cache_dest = directory / ".cache"
                if cache_dest.exists():
                    # 기존 캐시는 덮어쓰지 않고 파일 단위로 합칩니다.
                    cache_dest.mkdir(exist_ok=True)
                    for child in path.iterdir():
                        if child.is_file():
                            shutil.move(str(child), str(cache_dest / child.name))
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    shutil.move(str(path), str(cache_dest))
                continue
            if path.is_file():
                move_fn(directory, path)

    cues_path = directory / f"{slug}.json"
    cues, meta = load_cues(cues_path) if cues_path.exists() else ([], {})

    existing: dict[str, Any] = {}
    project_path = directory / PROJECT_FILE
    if project_path.exists():
        try:
            existing = _read_json(project_path)
        except Exception:
            existing = {}

    subtitles: dict[str, str] = {}
    for ext in SUBTITLE_EXTENSIONS:
        candidate = directory / f"{slug}{ext}"
        if candidate.exists():
            subtitles[ext.lstrip(".")] = candidate.name

    data: dict[str, Any] = {
        "schema_version": 1,
        "name": name,
        "slug": slug,
        "created_at": created_at or existing.get("created_at") or now_iso(),
        "audio": audio_dest.name,
        "cues": cues_path.name if cues_path.exists() else None,
        "subtitles": subtitles,
        "cue_count": len(cues),
        "duration": meta.get("duration"),
        "meta": {
            "model": meta.get("model"),
            "translator": meta.get("translator"),
            "language": meta.get("language"),
            "generated_at": meta.get("generated_at"),
        },
    }
    _write_json_atomic(directory / PROJECT_FILE, data)
    return load_project(directory)
