"""핵심 로직 단위 테스트 (모델 다운로드 없이 실행 가능)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from study_subs.asr import ASRSegment, Word  # noqa: E402
from study_subs.sentences import build_cues, detect_language  # noqa: E402
from study_subs.subtitles import format_mmss, format_srt_time, write_srt  # noqa: E402


def seg(start: float, end: float, words: list[tuple[float, float, str]]) -> ASRSegment:
    return ASRSegment(
        start=start,
        end=end,
        text=" ".join(w[2] for w in words),
        words=[Word(start=a, end=b, text=t) for a, b, t in words],
    )


class TimestampTests(unittest.TestCase):
    def test_mmss_floor(self) -> None:
        self.assertEqual(format_mmss(9.99), "00:09")
        self.assertEqual(format_mmss(65.2), "01:05")
        self.assertEqual(format_mmss(-3), "00:00")

    def test_srt_time(self) -> None:
        self.assertEqual(format_srt_time(0), "00:00:00,000")
        self.assertEqual(format_srt_time(65.5), "00:01:05,500")
        self.assertEqual(format_srt_time(3723.004), "01:02:03,004")


class SentenceTests(unittest.TestCase):
    def test_splits_on_terminal_punctuation(self) -> None:
        cues = build_cues(
            [
                seg(
                    0,
                    2.2,
                    [
                        (0.0, 0.4, "Hello,"),
                        (0.5, 1.0, "viewers."),
                        (1.1, 1.3, "How"),
                        (1.4, 1.8, "are"),
                        (1.9, 2.2, "you?"),
                    ],
                )
            ],
            duration=3.0,
        )
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].text, "Hello, viewers.")
        self.assertEqual(cues[1].text, "How are you?")
        self.assertAlmostEqual(cues[0].start, 0.0)
        self.assertAlmostEqual(cues[1].start, 1.1)

    def test_abbreviation_is_not_sentence_end(self) -> None:
        cues = build_cues(
            [
                seg(
                    0,
                    2.0,
                    [
                        (0.0, 0.3, "Mr."),
                        (0.4, 0.8, "Kim"),
                        (0.9, 1.3, "is"),
                        (1.4, 2.0, "here."),
                    ],
                )
            ],
            duration=3.0,
        )
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "Mr. Kim is here.")

    def test_question_number_is_moved_to_metadata(self) -> None:
        cues = build_cues(
            [
                seg(
                    0,
                    3.0,
                    [
                        (0.0, 0.3, "1."),
                        (0.5, 0.9, "Listen"),
                        (1.0, 1.4, "carefully."),
                    ],
                )
            ],
            duration=4.0,
        )
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].question, 1)
        self.assertEqual(cues[0].text, "Listen carefully.")

    def test_long_sentence_splits(self) -> None:
        words = []
        text = []
        for i in range(16):
            words.append((float(i), float(i) + 0.4, f"word{i},"))
            text.append(words[-1][2])
        cues = build_cues(
            [seg(0, 16, words)],
            duration=17.0,
            max_seconds=4.0,
            max_chars=40,
        )
        self.assertGreater(len(cues), 1)
        for cue in cues:
            self.assertLessEqual(cue.end - cue.start, 5.0)

    def test_duplicate_segments_are_merged(self) -> None:
        cues = build_cues(
            [
                seg(0, 1.0, [(0.0, 0.5, "Okay.")]),
                seg(0.6, 1.5, [(0.6, 1.0, "Okay.")]),
            ],
            duration=2.0,
        )
        self.assertEqual(len(cues), 1)

    def test_korean_guide_is_detected(self) -> None:
        self.assertEqual(detect_language("다음을 듣고 고르시오."), "ko")
        self.assertEqual(detect_language("Only4Health 채널입니다."), "en")
        cues = build_cues(
            [seg(0, 3.0, [(0.0, 2.5, "다음을 듣고 고르시오.")])],
            duration=4.0,
        )
        self.assertEqual(cues[0].lang, "ko")

    def test_hyphenated_word_is_joined(self) -> None:
        cues = build_cues(
            [
                seg(
                    0,
                    2.0,
                    [
                        (0.0, 0.5, "problem"),
                        (0.5, 0.8, "-solving"),
                        (0.9, 1.3, "skills"),
                        (1.4, 1.8, "matter."),
                    ],
                )
            ],
            duration=3.0,
        )
        self.assertEqual(cues[0].text, "problem-solving skills matter.")


class OutputTests(unittest.TestCase):
    def test_srt_writes_bilingual_lines(self) -> None:
        cues = build_cues(
            [seg(0, 1.2, [(0.0, 0.5, "Hello.")])],
            duration=2.0,
        )
        cues[0].translation = "안녕하세요."
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.srt"
            write_srt(cues, path, mode="bilingual")
            data = path.read_text(encoding="utf-8")
            self.assertIn("00:00:00,000 -->", data)
            self.assertIn("Hello.", data)
            self.assertIn("안녕하세요.", data)

    def test_srt_english_mode_skips_korean_guide(self) -> None:
        cues = build_cues(
            [
                seg(0, 1.0, [(0.0, 0.8, "다음을 듣고 고르시오.")]),
                seg(1.2, 2.5, [(1.2, 2.2, "Hello.")]),
            ],
            duration=3.0,
        )
        cues[0].lang = "ko"
        cues[1].translation = "안녕하세요."
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.srt"
            write_srt(cues, path, mode="english")
            data = path.read_text(encoding="utf-8")
            self.assertNotIn("다음을", data)
            self.assertIn("Hello.", data)
            self.assertIn("1\n00:00:01", data)


if __name__ == "__main__":
    unittest.main()
