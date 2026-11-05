"""핵심 로직 단위 테스트 (모델 다운로드 없이 실행 가능)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from study_subs.asr import (  # noqa: E402
    ASRSegment,
    Word,
    _fill_skipped_gaps,
    _progress_tqdm,
    _rescue_sparse_segments,
    _speech_regions,
)
from study_subs.gui import DARK_THEME, LIGHT_THEME, THEMES  # noqa: E402
from study_subs.project import (  # noqa: E402
    create_project,
    default_project_root,
    discover_projects,
    load_project,
    save_project_cues,
)
from study_subs.sentences import build_cues, detect_language  # noqa: E402
from study_subs.subtitles import format_mmss, format_srt_time, write_srt  # noqa: E402
from study_subs.tail_fix import CueSpan, TailFixConfig, analyze_tail_clipping  # noqa: E402


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

    def test_korean_number_prefix_is_question(self) -> None:
        cues = build_cues(
            [
                seg(
                    0,
                    3.0,
                    [(0.0, 2.5, "13번 대화를 듣고 고르시오.")],
                )
            ],
            duration=4.0,
        )
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].question, 13)
        self.assertEqual(cues[0].text, "대화를 듣고 고르시오.")
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


class GapFillTests(unittest.TestCase):
    def _audio(self, total: float, voiced: list[tuple[float, float]]) -> "np.ndarray":
        import numpy as np

        audio = np.zeros(int(total * 16000), dtype=np.float32)
        t = np.arange(int(total * 16000)) / 16000.0
        for start, end in voiced:
            mask = (t >= start) & (t < end)
            audio[mask] = 0.3 * np.sin(2 * np.pi * 220 * t[mask])
        return audio

    def test_speech_regions_skip_silence(self) -> None:
        audio = self._audio(12.0, [(6.0, 9.0)])
        regions = _speech_regions(audio, 0.0, 12.0)
        self.assertEqual(len(regions), 1)
        self.assertAlmostEqual(regions[0][0], 6.0, delta=0.3)
        self.assertAlmostEqual(regions[0][1], 9.0, delta=0.3)

    def test_speech_regions_empty_for_silence(self) -> None:
        self.assertEqual(_speech_regions(self._audio(10.0, []), 0.0, 10.0), [])

    def test_fill_gaps_offsets_and_filters_hallucination(self) -> None:
        audio = self._audio(25.0, [(8.0, 12.0)])
        segments = [
            ASRSegment(start=0.0, end=2.0, text="First."),
            ASRSegment(start=18.0, end=20.0, text="Second."),
        ]

        class FakeModel:
            def __init__(self) -> None:
                self.calls = 0

            def transcribe(self, chunk, **kwargs):
                self.calls += 1
                return {
                    "segments": [
                        {
                            "start": 0.1,
                            "end": 3.9,
                            "text": "다음을 듣고 고르시오.",
                            "no_speech_prob": 0.1,
                            "avg_logprob": -0.2,
                            "words": [],
                        },
                        {
                            "start": 0.2,
                            "end": 3.5,
                            "text": "환각",
                            "no_speech_prob": 0.95,
                            "avg_logprob": -1.4,
                            "words": [],
                        },
                    ]
                }

        model = FakeModel()
        added = _fill_skipped_gaps(
            model, audio, segments, language="ko", beam_size=1
        )
        self.assertEqual(model.calls, 1)
        self.assertEqual(len(added), 1)
        self.assertAlmostEqual(added[0].start, 8.0, delta=0.3)
        self.assertAlmostEqual(added[0].end, 12.0, delta=0.3)
        self.assertEqual(added[0].text, "다음을 듣고 고르시오.")

    def test_rescue_replaces_sparse_english_with_korean_guide(self) -> None:
        audio = self._audio(30.0, [(1.0, 9.0)])

        class FakeModel:
            def __init__(self) -> None:
                self.calls = 0

            def transcribe(self, chunk, **kwargs):
                self.calls += 1
                assert kwargs["language"] == "ko"
                return {
                    "segments": [
                        {
                            "start": 1.0,
                            "end": 8.0,
                            "text": "13번 대화를 듣고 고르시오.",
                            "no_speech_prob": 0.2,
                            "avg_logprob": -0.3,
                            "words": [],
                        }
                    ]
                }

        model = FakeModel()
        segments = [
            ASRSegment(
                start=0.0,
                end=28.0,
                text="The answer is the answer.",
                no_speech_prob=0.8,
            ),
            ASRSegment(start=28.0, end=30.0, text="Next.", no_speech_prob=0.05),
        ]
        out = _rescue_sparse_segments(model, audio, segments, beam_size=1)
        self.assertEqual(model.calls, 1)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].text, "13번 대화를 듣고 고르시오.")
        self.assertEqual(out[1].text, "Next.")

    def test_rescue_keeps_confident_long_segment(self) -> None:
        audio = self._audio(30.0, [(1.0, 29.0)])

        class FakeModel:
            def transcribe(self, chunk, **kwargs):  # pragma: no cover
                raise AssertionError("신뢰도가 높은 구간은 다시 읽지 않아야 합니다.")

        segments = [
            ASRSegment(
                start=0.0,
                end=28.0,
                text="A long but confident sentence.",
                no_speech_prob=0.05,
            )
        ]
        out = _rescue_sparse_segments(
            FakeModel(), audio, segments, beam_size=1
        )
        self.assertEqual(out, segments)

    def test_progress_tqdm_maps_to_fraction(self) -> None:
        seen: list[float] = []
        bar = _progress_tqdm(seen.append).tqdm(total=4)
        bar.update(2)
        bar.update(2)
        bar.close()
        self.assertEqual(seen, [0.5, 1.0])


class ProjectTests(unittest.TestCase):
    def _make_outputs(self, directory: Path, stem: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        audio = directory / f"{stem}.mp3"
        audio.write_bytes(b"ID3fake")
        (directory / f"{stem}.json").write_text(
            '{"cues": [{"index": 1, "start": 0.0, "end": 1.0,'
            ' "english": "Hello.", "korean": "안녕하세요.", "language": "en"}],'
            ' "meta": {"model": "base", "translator": "nllb", "duration": 1.0}}',
            encoding="utf-8",
        )
        (directory / f"{stem}.srt").write_text("1\n", encoding="utf-8")
        return audio

    def test_create_project_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = self._make_outputs(tmp_path / "out", "sample")
            project_dir = tmp_path / "data" / "project" / "sample"
            project = create_project(
                project_dir,
                name="샘플 프로젝트",
                audio=audio,
                out_dir=audio.parent,
                move=True,
            )
            self.assertTrue((project_dir / "project.json").exists())
            self.assertTrue((project_dir / "sample.mp3").exists())
            self.assertTrue((project_dir / "sample.srt").exists())
            self.assertFalse(audio.exists())
            self.assertEqual(project.name, "샘플 프로젝트")
            self.assertEqual(project.cue_count, 1)
            self.assertEqual(project.cues[0].translation, "안녕하세요.")
            self.assertEqual(project.data["subtitles"]["srt"], "sample.srt")
            self.assertTrue(project.created_at)

    def test_create_project_preserves_created_at(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = self._make_outputs(tmp_path, "sample")
            first = create_project(
                tmp_path,
                name="샘플",
                audio=audio,
                out_dir=tmp_path,
                created_at="2020-01-01T00:00:00+09:00",
            )
            second = create_project(
                tmp_path,
                name="샘플",
                audio=tmp_path / "sample.mp3",
                out_dir=tmp_path,
            )
            self.assertEqual(first.created_at, "2020-01-01T00:00:00+09:00")
            self.assertEqual(second.created_at, first.created_at)

    def test_discover_projects_skips_plain_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = self._make_outputs(tmp_path, "sample")
            create_project(tmp_path / "good", name="좋음", audio=audio, out_dir=tmp_path)
            (tmp_path / "plain").mkdir()
            projects = discover_projects(tmp_path)
            self.assertEqual([p.name for p in projects], ["좋음"])

    def test_default_project_root_finds_repository(self) -> None:
        self.assertEqual(default_project_root(ROOT / "tests"), ROOT / "data" / "project")

    def test_save_project_cues_rewrites_json_and_existing_subtitles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "lecture.mp3").write_bytes(b"ID3fake")
            cue_data = {
                "meta": {
                    "title": "강의",
                    "model": "base",
                    "translator": "nllb",
                    "duration": 4.0,
                    "generated_at": "2026-01-01 00:00:00",
                },
                "cues": [
                    {
                        "index": 1,
                        "start": 0.0,
                        "end": 1.0,
                        "question": None,
                        "language": "en",
                        "english": "Hello.",
                        "korean": "안녕하세요.",
                    }
                ],
            }
            (directory / "lecture.json").write_text(
                json.dumps(cue_data, ensure_ascii=False), encoding="utf-8"
            )
            (directory / "custom.srt").write_text("old\n", encoding="utf-8")
            project_data = {
                "schema_version": 1,
                "name": "강의",
                "slug": "lecture",
                "created_at": "2026-01-01T00:00:00+09:00",
                "audio": "lecture.mp3",
                "cues": "lecture.json",
                "subtitles": {"srt": "custom.srt"},
                "cue_count": 1,
                "duration": 4.0,
                "meta": {"model": "base", "translator": "nllb"},
            }
            project_path = directory / "project.json"
            project_path.write_text(
                json.dumps(project_data, ensure_ascii=False), encoding="utf-8"
            )
            before = project_path.read_text(encoding="utf-8")

            project = load_project(directory)
            project.cues[0].end = 2.5
            paths = save_project_cues(project)

            self.assertIn(directory / "lecture.json", paths)
            self.assertIn(directory / "custom.srt", paths)
            self.assertEqual(project_path.read_text(encoding="utf-8"), before)

            reloaded = load_project(directory)
            self.assertAlmostEqual(reloaded.cues[0].end, 2.5)
            self.assertEqual(reloaded.cues[0].translation, "안녕하세요.")
            self.assertEqual(reloaded.meta.get("title"), "강의")
            self.assertIn(
                "00:00:00,000 --> 00:00:02,500",
                (directory / "custom.srt").read_text(encoding="utf-8"),
            )


class TailFixTests(unittest.TestCase):
    SAMPLE_RATE = 16_000

    def _tone(self, seconds: float, amplitude: float = 0.2) -> np.ndarray:
        count = int(round(seconds * self.SAMPLE_RATE))
        t = np.arange(count, dtype=np.float64)
        return (amplitude * np.sin(2 * np.pi * 440.0 * t / self.SAMPLE_RATE)).astype(
            np.float32
        )

    def _audio(self) -> np.ndarray:
        audio = np.zeros(int(9.0 * self.SAMPLE_RATE), dtype=np.float32)

        def add(start: float, end: float) -> None:
            a = int(start * self.SAMPLE_RATE)
            b = int(end * self.SAMPLE_RATE)
            audio[a:b] = self._tone((b - a) / self.SAMPLE_RATE)

        add(0.10, 1.20)  # 1: 1.00에서 잘림
        add(1.80, 2.30)  # 2: 종료 지점은 무음
        add(2.90, 3.40)  # 3: 3.10에서 잘림
        add(4.80, 5.20)  # 4: cue 구간 자체가 무음
        add(5.60, 6.70)  # 5: 0.9초가 이어져 단순 꼬리로 보기 어려움
        add(6.90, 7.20)  # 6: 무음 뒤 다른 발화
        add(7.40, 7.70)
        return audio

    def test_only_short_speech_tails_are_extended(self) -> None:
        cues = [
            CueSpan(1, 0.10, 1.00),
            CueSpan(2, 1.80, 2.50),
            CueSpan(3, 2.90, 3.10),
            CueSpan(4, 4.50, 4.80),
            CueSpan(5, 5.60, 5.80),
            CueSpan(6, 6.90, 7.20),
        ]
        fixes = analyze_tail_clipping(self._audio(), cues)
        self.assertEqual([fix.index for fix in fixes], [1, 3])
        self.assertAlmostEqual(fixes[0].new_end, 1.25, places=2)
        self.assertAlmostEqual(fixes[1].new_end, 3.45, places=2)

    def test_next_cue_start_is_never_crossed(self) -> None:
        audio = np.zeros(int(3.0 * self.SAMPLE_RATE), dtype=np.float32)
        audio[int(0.10 * self.SAMPLE_RATE) : int(1.80 * self.SAMPLE_RATE)] = (
            self._tone(1.70)
        )
        cues = [CueSpan(1, 0.10, 1.00), CueSpan(2, 1.35, 1.80)]
        fixes = analyze_tail_clipping(
            audio, cues, config=TailFixConfig(max_extension=0.70)
        )
        self.assertEqual(len(fixes), 1)
        self.assertAlmostEqual(fixes[0].new_end, 1.32, places=2)


class ThemeTests(unittest.TestCase):
    def test_light_and_dark_share_the_same_palette_keys(self) -> None:
        self.assertEqual(set(LIGHT_THEME), set(DARK_THEME))
        self.assertIs(THEMES["light"], LIGHT_THEME)
        self.assertIs(THEMES["dark"], DARK_THEME)


if __name__ == "__main__":
    unittest.main()
