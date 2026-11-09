"""Intel Arc GPU(whisper.cpp Vulkan) 음성 인식 백엔드.

PyTorch XPU는 최신 Intel 그래픽 드라이버를 요구하지만, Vulkan 경로는
이 노트북의 2023년 드라이버에서도 Intel Arc A350M에서 동작합니다.
(같은 원리로 llama.cpp Vulkan이 동작한다는 것을 확인한 뒤 추가했습니다.)

whisper.cpp 실행 파일과 GGML 모델은 처음 사용할 때 자동으로 내려받아
`%LOCALAPPDATA%\\study_subs\\whisper-vulkan`(Windows)에 캐시합니다.
직접 빌드한 실행 파일을 쓰려면 `STUDY_SUBS_WHISPER_CLI` 환경 변수로
`whisper-cli.exe` 경로를 지정하세요.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
import wave
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000
_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# whisper.cpp의 공식 릴리스에는 Vulkan 빌드가 없어, Vulkan DLL이 포함된
# 사전 빌드 바이너리를 사용합니다. 다른 빌드를 쓰려면 STUDY_SUBS_WHISPER_CLI.
_RUNTIME_URL = (
    "https://github.com/eviscerations/whisper-windows-mcp/releases/download/"
    "v1.4.0/whisper-vulkan-win-x64.zip"
)
_RUNTIME_VERSION = "v1.4.0"
_MODEL_URL = (
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{name}"
)

_MODEL_FILES = {
    "tiny": "ggml-tiny.bin",
    "tiny.en": "ggml-tiny.en.bin",
    "base": "ggml-base.bin",
    "base.en": "ggml-base.en.bin",
    "small": "ggml-small.bin",
    "small.en": "ggml-small.en.bin",
    "medium": "ggml-medium.bin",
    "medium.en": "ggml-medium.en.bin",
    "large-v1": "ggml-large-v1.bin",
    "large-v2": "ggml-large-v2.bin",
    "large-v3": "ggml-large-v3.bin",
    "large": "ggml-large-v3.bin",
    "large-v3-turbo": "ggml-large-v3-turbo.bin",
}


def cache_root() -> Path:
    """런타임/모델 캐시 폴더를 돌려줍니다."""

    override = os.environ.get("STUDY_SUBS_CACHE_DIR")
    if override:
        base = Path(override)
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        base = base / "study_subs"
    else:
        base = Path.home() / ".cache" / "study_subs"
    return base / "whisper-vulkan"


def runtime_dir() -> Path:
    return cache_root() / "runtime"


def models_dir() -> Path:
    return cache_root() / "models"


def is_supported() -> bool:
    """이 플랫폼에서 Vulkan 백엔드를 쓸 수 있는지 확인합니다."""

    return sys.platform == "win32" and sys.maxsize > 2**32


def cli_path() -> Path | None:
    """설치된 whisper-cli.exe 경로를 찾습니다. 없으면 None."""

    override = os.environ.get("STUDY_SUBS_WHISPER_CLI")
    if override:
        path = Path(override).expanduser()
        return path if path.is_file() else None
    exe = runtime_dir() / "whisper-cli.exe"
    if exe.is_file():
        return exe
    if runtime_dir().is_dir():
        for found in runtime_dir().rglob("whisper-cli.exe"):
            return found
    return None


def model_path(model_name: str) -> Path:
    """모델 이름에 대응하는 GGML 파일 경로를 돌려줍니다."""

    key = (model_name or "small").strip().lower()
    filename = _MODEL_FILES.get(key)
    if filename is None:
        # 사용자 지정 이름은 경로 조작이 없는 경우에만 허용합니다.
        if not re.fullmatch(r"[A-Za-z0-9._-]+", key):
            raise ValueError(f"지원하지 않는 Whisper 모델 이름입니다: {model_name!r}")
        filename = f"ggml-{key}.bin"
    return models_dir() / filename


def _download(
    url: str,
    destination: Path,
    *,
    progress: Callable[[float], None] | None = None,
    description: str = "다운로드",
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "study-subs/1.0"})
    log.info("%s 시작: %s", description, url)
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length") or 0)
        received = 0
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                received += len(chunk)
                if total and progress is not None:
                    progress(min(1.0, received / total))
                    if received // (25 * 1024 * 1024) != (received - len(chunk)) // (
                        25 * 1024 * 1024
                    ):
                        log.info(
                            "      %s %d%% (%.0f/%.0f MB)",
                            description,
                            int(100 * received / total),
                            received / 1024 / 1024,
                            total / 1024 / 1024,
                        )
    if total and destination.stat().st_size < total:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"{description}가 중간에 끊겼습니다: {url}")
    return destination


def ensure_runtime(
    *, progress: Callable[[float], None] | None = None
) -> Path:
    """Vulkan whisper-cli 런타임을 준비하고 실행 파일 경로를 돌려줍니다."""

    existing = cli_path()
    if existing is not None:
        return existing

    if not is_supported():
        raise RuntimeError(
            "Vulkan whisper.cpp 런타임은 Windows x64에서만 자동 설치됩니다.\n"
            "다른 OS에서는 --device cpu 또는 PyTorch GPU를 사용하세요."
        )

    runtime_dir().mkdir(parents=True, exist_ok=True)
    archive = runtime_dir() / f"whisper-vulkan-{_RUNTIME_VERSION}.zip"
    if not archive.is_file() or archive.stat().st_size < 1_000_000:
        _download(
            _RUNTIME_URL,
            archive,
            progress=progress,
            description="Vulkan Whisper 런타임",
        )

    log.info("Vulkan Whisper 런타임 압축 해제: %s", archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(runtime_dir())
    archive.unlink(missing_ok=True)

    exe = cli_path()
    if exe is None:
        raise RuntimeError("압축을 풀었지만 whisper-cli.exe를 찾지 못했습니다.")

    vulkan_dll = exe.parent / "ggml-vulkan.dll"
    if not vulkan_dll.is_file():
        raise RuntimeError(
            "whisper-cli.exe는 찾았지만 ggml-vulkan.dll이 없습니다. "
            "Vulkan이 포함된 빌드인지 확인하세요."
        )
    return exe


def ensure_model(
    model_name: str,
    *,
    progress: Callable[[float], None] | None = None,
) -> Path:
    """GGML 모델을 준비하고 경로를 돌려줍니다."""

    path = model_path(model_name)
    if path.is_file() and path.stat().st_size > 1_000_000:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    _download(
        _MODEL_URL.format(name=path.name),
        partial,
        progress=progress,
        description=f"Whisper {model_name} 모델",
    )
    os.replace(partial, path)
    return path


def _parse_vulkan_devices(output: str) -> list[tuple[int, str, bool]]:
    """`ggml_vulkan: 0 = ... | uma: 1` 줄에서 장치 목록을 뽑습니다."""

    devices: list[tuple[int, str, bool]] = []
    for line in output.splitlines():
        match = re.search(r"ggml_vulkan:\s*(\d+)\s*=\s*([^|]+)\|(.*)", line)
        if not match:
            continue
        index = int(match.group(1))
        name = match.group(2).strip()
        uma = "uma: 1" in match.group(3)
        devices.append((index, name, uma))
    return devices


def detect_device(
    cli: Path | None = None, *, preferred: str = "arc"
) -> tuple[int, str]:
    """Vulkan 장치 중 Intel Arc(없으면 외장 GPU)를 고릅니다."""

    override = os.environ.get("STUDY_SUBS_VULKAN_DEVICE")
    exe = cli or cli_path()
    if exe is None:
        if override:
            return int(override), "vulkan"
        raise RuntimeError("whisper-cli.exe가 없어 Vulkan 장치를 확인할 수 없습니다.")

    try:
        proc = subprocess.run(
            [str(exe), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            creationflags=_CREATE_NO_WINDOW,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
    except Exception:
        log.debug("Vulkan 장치 목록을 읽지 못했습니다.", exc_info=True)
        output = ""

    devices = _parse_vulkan_devices(output)
    if override:
        index = int(override)
        for dev_index, name, _uma in devices:
            if dev_index == index:
                return index, name
        return index, "vulkan"

    for index, name, _uma in devices:
        if preferred in name.lower():
            return index, name
    for index, name, uma in devices:
        if not uma:
            return index, name
    if devices:
        return devices[0][0], devices[0][1]
    return 0, "vulkan"


def _device_index(device: str) -> int | None:
    if ":" not in device:
        return None
    try:
        return max(0, int(device.split(":", 1)[1]))
    except (TypeError, ValueError):
        return None


def _write_wav(audio: np.ndarray, path: Path) -> None:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(_SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def _merge_tokens(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """whisper.cpp 토큰을 단어 단위로 합칩니다.

    토큰은 보통 앞 공백으로 단어 시작을 표시합니다. `'s`, `.`, `,`처럼
    앞 공백이 없는 조각은 직전 단어에 붙입니다.
    """

    words: list[dict[str, Any]] = []
    for token in tokens:
        raw = str(token.get("text", ""))
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("[_") and stripped.endswith("]"):
            continue
        if stripped.startswith("<|") and stripped.endswith("|>"):
            continue
        offsets = token.get("offsets") or {}
        start = float(offsets.get("from", 0)) / 1000.0
        end = float(offsets.get("to", start * 1000.0)) / 1000.0
        end = max(start, end)
        probability = token.get("p")
        starts_new = not words or raw[:1].isspace()
        if starts_new:
            words.append(
                {
                    "word": stripped,
                    "start": start,
                    "end": end,
                    "probability": float(probability)
                    if probability is not None
                    else None,
                }
            )
        else:
            words[-1]["word"] += stripped
            words[-1]["end"] = max(float(words[-1]["end"]), end)
            if probability is not None:
                previous = words[-1]["probability"]
                words[-1]["probability"] = (
                    float(probability)
                    if previous is None
                    else min(float(previous), float(probability))
                )
    return words


def _read_json_file(path: Path) -> dict[str, Any]:
    """whisper.cpp JSON을 읽습니다.

    모델이 간혹 잘못된 UTF-8 바이트를 섞어 내보내면 `read_text()`가
    UnicodeDecodeError로 전사 전체를 실패시킵니다. 잘못된 바이트만
    대체 문자로 바꿔 나머지 결과는 그대로 사용합니다.
    """

    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        log.warning("whisper.cpp JSON에 잘못된 UTF-8 바이트가 있어 대체합니다: %s", path)
        text = raw.decode("utf-8-sig", errors="replace")
    return json.loads(text)


def parse_json_result(
    data: dict[str, Any], *, requested_language: str | None = None
) -> dict[str, Any]:
    """whisper.cpp JSON(=`-ojf`)을 Whisper dict 형태로 변환합니다."""

    segments: list[dict[str, Any]] = []
    for segment in data.get("transcription", []) or []:
        offsets = segment.get("offsets") or {}
        start = float(offsets.get("from", 0)) / 1000.0
        end = float(offsets.get("to", start * 1000.0)) / 1000.0
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        tokens = list(segment.get("tokens", []) or [])
        words = _merge_tokens(tokens)
        probabilities = [
            float(token["p"])
            for token in tokens
            if token.get("p") is not None and float(token["p"]) > 0
        ]
        avg_logprob = (
            sum(math.log(max(p, 1e-6)) for p in probabilities) / len(probabilities)
            if probabilities
            else None
        )
        segments.append(
            {
                "start": start,
                "end": max(start, end),
                "text": text,
                "words": words,
                "no_speech_prob": None,
                "avg_logprob": avg_logprob,
            }
        )

    result = data.get("result") or {}
    language = str(result.get("language") or requested_language or "")
    return {"language": language, "segments": segments}


def transcribe_audio(
    audio: np.ndarray,
    *,
    model_name: str = "small",
    language: str | None = "en",
    device: str = "vulkan",
    threads: int = 0,
    beam_size: int = 1,
    initial_prompt: str | None = None,
    progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Vulkan whisper.cpp로 전사하고 Whisper 호환 dict를 돌려줍니다."""

    def report(value: float) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, value)))

    def runtime_progress(fraction: float) -> None:
        report(0.02 * fraction)

    def model_progress(fraction: float) -> None:
        report(0.02 + 0.10 * fraction)

    exe = ensure_runtime(progress=runtime_progress)
    model = ensure_model(model_name, progress=model_progress)

    index = _device_index(device)
    if index is None:
        index, _name = detect_device(exe)
    log.info(
        "Vulkan Whisper 전사: model=%s device=%d language=%s",
        model_name,
        index,
        language or "auto",
    )

    cli_threads = int(threads) if threads and threads > 0 else min(8, os.cpu_count() or 4)
    lang = (language or "auto").lower()
    with tempfile.TemporaryDirectory(prefix="study_subs_whisper_") as tmp:
        tmp_dir = Path(tmp)
        wav_path = tmp_dir / "audio.wav"
        output_base = tmp_dir / "result"
        _write_wav(audio, wav_path)

        command = [
            str(exe),
            "-m",
            str(model),
            "-f",
            str(wav_path),
            "-l",
            lang,
            "-dev",
            str(index),
            "-t",
            str(cli_threads),
            "-bs",
            str(max(1, int(beam_size))),
            "-oj",
            "-ojf",
            "-of",
            str(output_base),
            "-pp",
        ]
        if initial_prompt:
            command.extend(["--prompt", initial_prompt])

        log.debug("whisper.cpp 명령: %s", command)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=_CREATE_NO_WINDOW,
        )
        tail: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            stripped = line.rstrip()
            if stripped:
                tail.append(stripped)
                del tail[:-30]
                log.debug("[whisper.cpp] %s", stripped)
            match = re.search(r"progress\s*=\s*(\d+)%", stripped)
            if match:
                report(0.15 + 0.85 * int(match.group(1)) / 100.0)
        returncode = process.wait()
        if returncode != 0:
            details = "\n".join(tail[-15:])
            raise RuntimeError(
                "whisper.cpp가 비정상 종료했습니다 "
                f"(exit={returncode}).\n{details}"
            )

        json_path = output_base.with_suffix(".json")
        if not json_path.is_file():
            raise RuntimeError("whisper.cpp JSON 결과 파일을 찾지 못했습니다.")
        data = _read_json_file(json_path)

    return parse_json_result(data, requested_language=lang if lang != "auto" else None)
