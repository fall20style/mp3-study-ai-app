"""오디오 디코딩 유틸리티.

imageio-ffmpeg에 포함된 ffmpeg 실행 파일을 사용하므로 시스템에 ffmpeg를
따로 설치할 필요가 없습니다. (Windows 노트북 기준)
"""

from __future__ import annotations

import functools
import shutil
import subprocess
from pathlib import Path

import numpy as np


class AudioError(RuntimeError):
    """오디오를 읽지 못했을 때 발생합니다."""


@functools.lru_cache(maxsize=1)
def find_ffmpeg() -> str:
    """ffmpeg 실행 파일 경로를 찾습니다.

    1) imageio-ffmpeg에 번들된 바이너리
    2) PATH에 있는 ffmpeg
    """

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise AudioError(
        "ffmpeg를 찾을 수 없습니다. `pip install imageio-ffmpeg`로 설치하거나 "
        "시스템 ffmpeg를 PATH에 추가하세요."
    )


def decode_audio(
    path: str | Path,
    sample_rate: int = 16_000,
    *,
    max_seconds: float | None = None,
    timeout: float | None = None,
) -> np.ndarray:
    """오디오 파일을 16 kHz 모노 float32 배열로 디코딩합니다.

    Whisper는 내부적으로 16 kHz 모노 입력을 기대합니다. 원본이 12 kHz
    MP3처럼 낮은 샘플레이트여도 ffmpeg가 리샘플링합니다.
    """

    src = Path(path)
    if not src.exists():
        raise AudioError(f"입력 파일이 없습니다: {src}")

    cmd = [
        find_ffmpeg(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vn",
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
    ]
    if max_seconds is not None and max_seconds > 0:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd.append("-")

    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise AudioError(f"ffmpeg 디코딩 실패: {stderr[:500]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioError("ffmpeg 디코딩 시간이 초과되었습니다.") from exc

    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise AudioError("오디오 샘플을 읽지 못했습니다. 파일 형식을 확인하세요.")
    return audio


def audio_duration(audio: np.ndarray, sample_rate: int = 16_000) -> float:
    return float(audio.size) / float(sample_rate)
