"""Whisper 단어 타임스탬프를 공부용 문장 자막으로 묶는 모듈.

Whisper의 세그먼트는 20~30초씩 길게 나오는 경우가 많아서, 단어 단위
타임스탬프와 문장부호를 이용해 한 문장씩 끊습니다. 너무 긴 문장은
쉼표/세미콜론 우선으로 나누고, 그래도 길면 단어 경계에서 자릅니다.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from .asr import ASRSegment, Word


@dataclass
class Cue:
    index: int
    start: float
    end: float
    text: str
    translation: str = ""
    question: int | None = None
    is_number: bool = False
    lang: str = "en"


_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """Whisper가 한국어 안내를 그대로 받아쓴 경우를 구분합니다."""

    hangul = len(_HANGUL_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    if hangul == 0:
        return "en"
    if latin == 0:
        return "ko"
    return "ko" if hangul > latin else "en"


_TERMINAL_RE = re.compile(r"[.!?…]+[\"'”’)\]]*$")
_ABBREVIATIONS = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "st.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "u.s.",
    "u.k.",
    "no.",
    "fig.",
    "inc.",
    "ltd.",
    "co.",
    "al.",
}
_PUNCT_ONLY_RE = re.compile(r"^[\s,.!?;:'\"“”‘’()\[\]{}…-]+$")
_QUESTION_PREFIX_RE = re.compile(r"^(\d{1,2})\s*[.)]\s+(.+)$")
_NUMBER_ONLY_RE = re.compile(r"^\[?(\d{1,2})\]?\s*(?:번)?\s*[.)]?$")


def _normalize_word(text: str) -> str:
    text = html.unescape(str(text))
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_sentence_end(text: str) -> bool:
    t = text.strip()
    if not t or not _TERMINAL_RE.search(t):
        return False

    # 소수점/숫자 뒤 마침표는 문장 끝이 아닙니다. (예: 3.14)
    if re.search(r"\d\.\d", t):
        return False

    core = t.rstrip("\"'”’)]}")
    low = core.lower()

    # "1." 같은 문제 번호는 다음 문장과 붙여서 한 cue로 만듭니다.
    if re.fullmatch(r"\d{1,2}\s*[.)]", core):
        return False
    if low in _ABBREVIATIONS:
        return False
    # U.S., e.g., i.e. 같은 약어
    if re.fullmatch(r"(?:[A-Za-z]\.){2,}", core):
        return False
    return True


def _join_words(tokens: Iterable[str]) -> str:
    """Whisper 단어 토큰을 자연스러운 문장으로 결합합니다."""

    out = ""
    no_space_before = set(".,!?;:%)]}")
    no_space_after = set("([{")
    for raw in tokens:
        t = raw.strip()
        if not t:
            continue
        if not out:
            out = t
        elif t[0] in no_space_before:
            out += t
        elif out[-1] in no_space_after:
            out += t
        else:
            out += " " + t

    out = re.sub(r"\s+([,!.?;:%\)\]}])", r"\1", out)
    out = re.sub(r"([\(\[\{])\s+", r"\1", out)
    out = re.sub(r"\s+-(?=[A-Za-z0-9])", "-", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _chars(words: Sequence[Word]) -> int:
    return len(_join_words(w.text for w in words))


def _split_words(
    words: list[Word],
    *,
    max_seconds: float,
    max_chars: int,
) -> list[list[Word]]:
    """한 문장이 너무 길면 읽기 좋은 단위로 재귀적으로 나눕니다."""

    if len(words) <= 1:
        return [words]

    duration = words[-1].end - words[0].start
    if duration <= max_seconds and _chars(words) <= max_chars:
        return [words]

    # 1순위: 한도 안에서 가장 뒤에 있는 쉼표/세미콜론/콜론
    split_at: int | None = None
    for i in range(1, len(words)):
        prefix = words[:i]
        if prefix[-1].text.rstrip().endswith((",", ";", ":")):
            if (
                words[i - 1].end - words[0].start <= max_seconds
                and _chars(prefix) <= max_chars
            ):
                split_at = i

    # 2순위: 한도를 넘는 첫 단어 직전
    if split_at is None:
        for i in range(1, len(words)):
            if (
                words[i].end - words[0].start > max_seconds
                or _chars(words[:i]) > max_chars
            ):
                split_at = i
                break

    if split_at is None or split_at >= len(words):
        split_at = max(1, len(words) - 1)

    left = words[:split_at]
    right = words[split_at:]
    return _split_words(left, max_seconds=max_seconds, max_chars=max_chars) + _split_words(
        right, max_seconds=max_seconds, max_chars=max_chars
    )


def build_cues(
    segments: Sequence[ASRSegment],
    *,
    duration: float | None = None,
    max_seconds: float = 12.0,
    max_chars: int = 180,
) -> list[Cue]:
    """ASR 세그먼트를 문장 단위 Cue 목록으로 변환합니다."""

    words: list[Word] = []
    for seg in segments:
        seg_words = list(seg.words)
        if not seg_words:
            text = _normalize_word(seg.text)
            if text:
                seg_words = [Word(start=seg.start, end=seg.end, text=text)]
        for w in seg_words:
            text = _normalize_word(w.text)
            if not text:
                continue
            words.append(
                Word(
                    start=max(0.0, float(w.start)),
                    end=max(0.0, float(w.end)),
                    text=text,
                    probability=w.probability,
                )
            )

    if not words:
        return []

    # 문장부호만 있는 토큰은 앞 단어에 붙입니다. (예: "Hello" + "!")
    merged: list[Word] = []
    for w in words:
        if merged and _PUNCT_ONLY_RE.fullmatch(w.text):
            prev = merged[-1]
            prev.text = prev.text + w.text
            prev.end = max(prev.end, w.end)
        else:
            merged.append(w)

    raw: list[tuple[float, float, str]] = []
    current: list[Word] = []

    def flush(part: list[Word]) -> None:
        if not part:
            return
        text = _join_words(w.text for w in part)
        if not text:
            return
        start = part[0].start
        end = max(part[-1].end, start + 0.4)
        raw.append((start, end, text))

    for w in merged:
        current.append(w)
        if _is_sentence_end(w.text):
            flush(current)
            current = []
        elif (
            current[-1].end - current[0].start > max_seconds
            or _chars(current) > max_chars
        ):
            parts = _split_words(
                current, max_seconds=max_seconds, max_chars=max_chars
            )
            for part in parts[:-1]:
                flush(part)
            current = parts[-1]

    flush(current)

    # 문제 번호("1.", "2번" 등)를 다음 cue의 question으로 옮깁니다.
    pending_question: int | None = None
    cues: list[Cue] = []
    for start, end, text in raw:
        number_only = _NUMBER_ONLY_RE.fullmatch(text)
        if number_only and len(text) <= 8:
            pending_question = int(number_only.group(1))
            continue

        question: int | None = None
        m = _QUESTION_PREFIX_RE.match(text)
        if m:
            question = int(m.group(1))
            text = m.group(2).strip()
            if not text:
                pending_question = question
                continue
        if pending_question is not None:
            question = pending_question
            pending_question = None

        cues.append(
            Cue(
                index=0,
                start=start,
                end=end,
                text=text,
                question=question,
                lang=detect_language(text),
            )
        )

    # 같은 문장이 연속으로 반복되는 Whisper 환각 제거
    deduped: list[Cue] = []
    for cue in cues:
        if (
            deduped
            and _norm_text(deduped[-1].text) == _norm_text(cue.text)
            and cue.start - deduped[-1].end < 1.0
        ):
            deduped[-1].end = max(deduped[-1].end, cue.end)
            continue
        deduped.append(cue)

    total = float(duration) if duration and duration > 0 else deduped[-1].end
    cleaned: list[Cue] = []
    for cue in deduped:
        if cue.start >= total:
            continue
        cue.start = max(0.0, min(cue.start, total))
        cue.end = max(cue.start + 0.4, min(cue.end, total))
        cleaned.append(cue)

    # 다음 cue와의 간격이 짧으면 자막이 사라지지 않도록 살짝 늘립니다.
    for i, cue in enumerate(cleaned):
        if i + 1 < len(cleaned):
            nxt = cleaned[i + 1]
            gap = nxt.start - cue.end
            if 0.0 <= gap <= 0.8:
                cue.end = max(cue.end, nxt.start - 0.05)
            elif gap < 0.0:
                cue.end = max(cue.start + 0.4, nxt.start - 0.05)

    for i, cue in enumerate(cleaned, start=1):
        cue.index = i
    return cleaned


def _norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())
