"""Vulkan whisper.cpp 백엔드 단위 테스트 (실행 파일/모델 다운로드 없이)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from study_subs import whisper_vulkan  # noqa: E402
from study_subs.asr import is_vulkan_device, resolve_asr_backend  # noqa: E402
from study_subs.cli import _parse_device  # noqa: E402


def token(text: str, start_ms: int, end_ms: int, p: float = 0.9) -> dict:
    return {
        "text": text,
        "offsets": {"from": start_ms, "to": end_ms},
        "p": p,
    }


class TokenMergeTests(unittest.TestCase):
    def test_leading_space_starts_new_word(self) -> None:
        words = whisper_vulkan._merge_tokens(
            [
                token(" Hello", 5000, 6200),
                token(" viewers", 6200, 8000),
                token(",", 8000, 8200),
                token(" it", 8200, 8600),
                token("'s", 8600, 8900),
            ]
        )
        self.assertEqual(
            [w["word"] for w in words], ["Hello", "viewers,", "it's"]
        )
        self.assertAlmostEqual(words[0]["start"], 5.0)
        self.assertAlmostEqual(words[2]["end"], 8.9)

    def test_special_tokens_are_skipped(self) -> None:
        words = whisper_vulkan._merge_tokens(
            [
                token("[_BEG_]", 0, 0),
                token("<|en|>", 0, 0),
                token(" Hi", 0, 400),
            ]
        )
        self.assertEqual([w["word"] for w in words], ["Hi"])


class JsonParseTests(unittest.TestCase):
    def test_parse_segments_with_words(self) -> None:
        data = {
            "result": {"language": "en"},
            "transcription": [
                {
                    "offsets": {"from": 5000, "to": 9200},
                    "text": " Hello viewers, it's Ryan.",
                    "tokens": [
                        token("[_BEG_]", 5000, 5000, 0.9),
                        token(" Hello", 6020, 6430, 0.6),
                        token(" viewers", 6430, 8170, 0.8),
                        token(",", 8780, 9000, 0.7),
                        token(" it", 9000, 9200, 0.9),
                        token("'s", 9200, 9300, 0.5),
                    ],
                }
            ],
        }
        result = whisper_vulkan.parse_json_result(data, requested_language="en")
        self.assertEqual(result["language"], "en")
        self.assertEqual(len(result["segments"]), 1)
        segment = result["segments"][0]
        self.assertAlmostEqual(segment["start"], 5.0)
        self.assertAlmostEqual(segment["end"], 9.2)
        self.assertEqual(
            [w["word"] for w in segment["words"]],
            ["Hello", "viewers,", "it's"],
        )
        self.assertIsNotNone(segment["avg_logprob"])

    def test_empty_transcription(self) -> None:
        result = whisper_vulkan.parse_json_result({"transcription": []})
        self.assertEqual(result["segments"], [])


class JsonFileTests(unittest.TestCase):
    def test_reads_utf8_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text(
                '{"result": {"language": "en"}}', encoding="utf-8"
            )
            data = whisper_vulkan._read_json_file(path)
        self.assertEqual(data["result"]["language"], "en")

    def test_invalid_byte_is_replaced_instead_of_raising(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_bytes(b'{"text": "ok \xeb bad"}')
            data = whisper_vulkan._read_json_file(path)
        self.assertIn("ok", data["text"])
        self.assertIn("\ufffd", data["text"])


class ModelPathTests(unittest.TestCase):
    def test_known_alias_maps_to_ggml_file(self) -> None:
        with mock.patch.dict("os.environ", {"STUDY_SUBS_CACHE_DIR": "C:/tmp/x"}):
            self.assertEqual(
                whisper_vulkan.model_path("large-v3").name,
                "ggml-large-v3.bin",
            )
            self.assertEqual(
                whisper_vulkan.model_path("base").name, "ggml-base.bin"
            )

    def test_large_alias_uses_v3_weights(self) -> None:
        with mock.patch.dict("os.environ", {"STUDY_SUBS_CACHE_DIR": "C:/tmp/x"}):
            self.assertEqual(
                whisper_vulkan.model_path("large").name, "ggml-large-v3.bin"
            )

    def test_custom_name_allowed(self) -> None:
        with mock.patch.dict("os.environ", {"STUDY_SUBS_CACHE_DIR": "C:/tmp/x"}):
            self.assertEqual(
                whisper_vulkan.model_path("base.en").name, "ggml-base.en.bin"
            )

    def test_path_traversal_rejected(self) -> None:
        with self.assertRaises(ValueError):
            whisper_vulkan.model_path("../evil")


class DeviceDetectionTests(unittest.TestCase):
    def test_parse_vulkan_devices(self) -> None:
        output = (
            "ggml_vulkan: Found 2 Vulkan devices:\n"
            "ggml_vulkan: 0 = Intel(R) Iris(R) Xe Graphics (Intel Corporation) "
            "| uma: 1 | fp16: 1 | bf16: 0\n"
            "ggml_vulkan: 1 = Intel(R) Arc(TM) A350M Graphics (Intel Corporation) "
            "| uma: 0 | fp16: 1 | bf16: 0\n"
        )
        devices = whisper_vulkan._parse_vulkan_devices(output)
        self.assertEqual([d[0] for d in devices], [0, 1])
        self.assertIn("Arc", devices[1][1])
        self.assertTrue(devices[0][2])
        self.assertFalse(devices[1][2])

    def test_device_index(self) -> None:
        self.assertIsNone(whisper_vulkan._device_index("vulkan"))
        self.assertEqual(whisper_vulkan._device_index("vulkan:2"), 2)
        self.assertIsNone(whisper_vulkan._device_index("vulkan:x"))


class BackendSelectionTests(unittest.TestCase):
    def test_vulkan_device_detection(self) -> None:
        self.assertTrue(is_vulkan_device("vulkan"))
        self.assertTrue(is_vulkan_device("Vulkan:1"))
        self.assertFalse(is_vulkan_device("xpu"))
        self.assertFalse(is_vulkan_device("cpu"))

    def test_explicit_vulkan_passthrough(self) -> None:
        self.assertEqual(resolve_asr_backend("vulkan:1"), "vulkan:1")

    def test_env_forces_cpu_for_auto(self) -> None:
        with mock.patch.dict("os.environ", {"STUDY_SUBS_ASR_BACKEND": "cpu"}):
            self.assertEqual(resolve_asr_backend("auto"), "cpu")

    def test_cli_accepts_vulkan(self) -> None:
        self.assertEqual(_parse_device("vulkan"), "vulkan")
        self.assertEqual(_parse_device("VULKAN:1"), "vulkan:1")
        with self.assertRaises(Exception):
            _parse_device("vulkan:x")


if __name__ == "__main__":
    unittest.main()
