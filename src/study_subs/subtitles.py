"""자막 파일 출력: SRT, VTT, TXT(공부용), Markdown, CSV, JSON."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from .sentences import Cue


def format_mmss(seconds: float) -> str:
    """시작 시간을 '분:초'로 표시합니다. 예: 00:09, 12:34."""

    total = max(0, int(float(seconds)))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def format_srt_time(seconds: float, *, sep: str = ",") -> str:
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def _one_line(text: str) -> str:
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split())


def _text_lines(cue: Cue, mode: str = "bilingual") -> list[str]:
    en = _one_line(cue.text)
    ko = _one_line(cue.translation)
    # 한국어 안내 방송은 번역하지 않고 한 번만 표시합니다.
    if cue.lang == "ko":
        if mode == "english":
            return []
        return [ko or en]
    if mode == "english":
        return [en] if en else ([ko] if ko else [])
    if mode == "korean":
        return [ko] if ko else ([en] if en else [])
    if en and ko:
        return [en, ko]
    return [en or ko]


def write_srt(cues: Sequence[Cue], path: str | Path, *, mode: str = "bilingual") -> None:
    blocks: list[str] = []
    number = 0
    for cue in cues:
        lines = _text_lines(cue, mode)
        if not lines:
            continue
        number += 1
        blocks.append(
            f"{number}\n"
            f"{format_srt_time(cue.start)} --> {format_srt_time(cue.end)}\n"
            + "\n".join(lines)
        )
    Path(path).write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def write_vtt(cues: Sequence[Cue], path: str | Path, *, mode: str = "bilingual") -> None:
    blocks: list[str] = ["WEBVTT\n"]
    for cue in cues:
        lines = _text_lines(cue, mode)
        if not lines:
            continue
        blocks.append(
            f"{format_srt_time(cue.start, sep='.')} --> "
            f"{format_srt_time(cue.end, sep='.')}\n"
            + "\n".join(lines)
        )
    Path(path).write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def _header_lines(meta: dict) -> list[str]:
    lines = [
        f"제목: {meta.get('title', '')}",
        f"원본: {meta.get('source', '')}",
        f"모델: whisper-{meta.get('model', '')} ({meta.get('device', 'cpu')})",
        f"번역: {meta.get('translator', 'none')}",
        f"길이: {format_mmss(meta.get('duration', 0))}",
        f"생성: {meta.get('generated_at', '')}",
    ]
    return lines


def write_txt(cues: Sequence[Cue], path: str | Path, meta: dict) -> None:
    """[분:초] 영어 / 한국어 형태의 공부용 텍스트."""

    out: list[str] = []
    out.extend(_header_lines(meta))
    out.append("=" * 72)
    out.append("")

    last_question: int | None = None
    for cue in cues:
        if cue.question is not None and cue.question != last_question:
            if out and out[-1] != "":
                out.append("")
            out.append(f"--- {cue.question}번 ---")
            out.append("")
            last_question = cue.question
        out.append(f"[{format_mmss(cue.start)}] {_one_line(cue.text)}")
        if cue.lang == "ko" and cue.text:
            out[-1] = f"[{format_mmss(cue.start)}] (한국어) {_one_line(cue.text)}"
        if cue.translation:
            out.append(f"          {_one_line(cue.translation)}")
        out.append("")

    Path(path).write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


def write_markdown(cues: Sequence[Cue], path: str | Path, meta: dict) -> None:
    has_questions = any(cue.question is not None for cue in cues)
    out: list[str] = [f"# {meta.get('title', '')}", ""]
    for line in _header_lines(meta)[1:]:
        out.append(f"- {line}")
    out.extend(["", "---", ""])

    last_question: int | None = None
    for cue in cues:
        if has_questions and cue.question is not None and cue.question != last_question:
            out.extend([f"## {cue.question}번", ""])
            last_question = cue.question
        out.append(f"**[{format_mmss(cue.start)}]** {_one_line(cue.text)}")
        if cue.lang == "ko" and cue.text:
            out[-1] = (
                f"**[{format_mmss(cue.start)}]** (한국어) {_one_line(cue.text)}"
            )
        if cue.translation:
            out.extend(["", f"> {_one_line(cue.translation)}"])
        out.append("")

    Path(path).write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


def write_csv(cues: Sequence[Cue], path: str | Path) -> None:
    # Excel에서 한글이 깨지지 않도록 utf-8-sig 사용
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "start",
                "end",
                "start_mmss",
                "end_mmss",
                "question",
                "language",
                "english",
                "korean",
            ]
        )
        for cue in cues:
            writer.writerow(
                [
                    cue.index,
                    f"{cue.start:.2f}",
                    f"{cue.end:.2f}",
                    format_mmss(cue.start),
                    format_mmss(cue.end),
                    cue.question if cue.question is not None else "",
                    cue.lang,
                    _one_line(cue.text),
                    _one_line(cue.translation),
                ]
            )


def write_json(cues: Sequence[Cue], path: str | Path, meta: dict) -> None:
    data = {
        "meta": meta,
        "cues": [
            {
                "index": cue.index,
                "start": round(cue.start, 3),
                "end": round(cue.end, 3),
                "start_mmss": format_mmss(cue.start),
                "end_mmss": format_mmss(cue.end),
                "question": cue.question,
                "language": cue.lang,
                "english": _one_line(cue.text),
                "korean": _one_line(cue.translation),
            }
            for cue in cues
        ],
    }
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_outputs(
    cues: Sequence[Cue],
    out_dir: str | Path,
    stem: str,
    *,
    formats: Iterable[str],
    meta: dict,
    srt_mode: str = "bilingual",
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for fmt in formats:
        fmt = fmt.lower().strip()
        if fmt == "srt":
            path = out / f"{stem}.srt"
            write_srt(cues, path, mode=srt_mode)
        elif fmt == "vtt":
            path = out / f"{stem}.vtt"
            write_vtt(cues, path, mode=srt_mode)
        elif fmt == "txt":
            path = out / f"{stem}.txt"
            write_txt(cues, path, meta)
        elif fmt == "md":
            path = out / f"{stem}.md"
            write_markdown(cues, path, meta)
        elif fmt == "csv":
            path = out / f"{stem}.csv"
            write_csv(cues, path)
        elif fmt == "json":
            path = out / f"{stem}.json"
            write_json(cues, path, meta)
        else:
            raise ValueError(f"지원하지 않는 출력 형식: {fmt}")
        paths[fmt] = path
    return paths


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
