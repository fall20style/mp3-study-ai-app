"""영어 → 한국어 번역 백엔드.

기본값은 Google 번역의 공개 엔드포인트를 사용합니다. 모델 다운로드가
없고 CPU 부담이 없어 노트북에서 가장 빠르지만, 네트워크가 필요하고
문장이 외부로 전송됩니다.

완전 오프라인으로 쓰려면 `--translator nllb`를 사용하세요. 이 경우
transformers와 facebook/nllb-200-distilled-600M 모델(약 2.4 GB)이
필요합니다.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Protocol, Sequence

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int], None]

GOOGLE_ENDPOINT = "https://translate.googleapis.com/translate_a/single"


class TranslationError(RuntimeError):
    pass


class Translator(Protocol):
    label: str

    def translate_batch(
        self, texts: Sequence[str], progress: ProgressFn | None = None
    ) -> list[str]:
        ...


class JsonCache:
    """번역 결과를 문장 해시 기준으로 저장해 재실행을 빠르게 합니다."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._data: dict[str, str] = {}
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("번역 캐시를 읽지 못했습니다(%s): %s", self.path, exc)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> str | None:
        return self._data.get(self._key(text))

    def set(self, text: str, translated: str) -> None:
        with self._lock:
            self._data[self._key(text)] = translated

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            data = dict(self._data)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self.path)


class NoopTranslator:
    label = "none"

    def translate_batch(
        self, texts: Sequence[str], progress: ProgressFn | None = None
    ) -> list[str]:
        return ["" for _ in texts]


class GoogleTranslator:
    label = "google"

    def __init__(
        self,
        cache_path: str | Path | None = None,
        *,
        source_lang: str = "en",
        target_lang: str = "ko",
        workers: int = 3,
        timeout: float = 20.0,
    ):
        try:
            import requests  # noqa: F401
        except ImportError as exc:
            raise TranslationError(
                "Google 번역에는 requests가 필요합니다. `pip install requests`"
            ) from exc
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.workers = max(1, min(int(workers), 8))
        self.timeout = timeout
        self.cache = JsonCache(cache_path)
        self._local = threading.local()

    def _session(self):
        import requests

        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36"
                    )
                }
            )
            self._local.session = session
        return session

    def _request(self, text: str, max_retries: int = 4) -> str:
        params = {
            "client": "gtx",
            "sl": self.source_lang,
            "tl": self.target_lang,
            "dt": "t",
            "q": text,
        }
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = self._session().get(
                    GOOGLE_ENDPOINT, params=params, timeout=self.timeout
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise TranslationError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()
                parts = data[0] or []
                translated = "".join(
                    str(part[0]) for part in parts if part and part[0]
                )
                translated = re.sub(r"\s+", " ", html.unescape(translated)).strip()
                if not translated:
                    raise TranslationError("빈 번역 결과")
                return translated
            except Exception as exc:  # noqa: BLE001 - 재시도 후 최종 실패 처리
                last_exc = exc
                if attempt + 1 < max_retries:
                    wait = (2**attempt) + random.uniform(0, 0.8)
                    log.debug(
                        "번역 재시도 %d/%d (%.1fs 후): %s",
                        attempt + 1,
                        max_retries,
                        wait,
                        exc,
                    )
                    time.sleep(wait)
        raise TranslationError(f"Google 번역 실패: {last_exc}")

    def translate_batch(
        self, texts: Sequence[str], progress: ProgressFn | None = None
    ) -> list[str]:
        texts = list(texts)
        result: list[str] = [""] * len(texts)

        # 캐시에 있는 것은 먼저 채웁니다.
        missing: dict[str, list[int]] = {}
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            cached = self.cache.get(text)
            if cached is not None:
                result[i] = cached
            else:
                missing.setdefault(text, []).append(i)

        if not missing:
            return result

        log.info(
            "Google 번역: %d개 문장 (캐시 %d개, 신규 %d개)",
            len(texts),
            len(texts) - sum(len(v) for v in missing.values()),
            len(missing),
        )
        failures = 0
        new_count = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(self._request, text): text for text in missing
            }
            for done, future in enumerate(as_completed(futures), start=1):
                text = futures[future]
                try:
                    translated = future.result()
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    log.warning("번역 실패 (%d자): %s", len(text), exc)
                    continue
                self.cache.set(text, translated)
                new_count += 1
                for idx in missing[text]:
                    result[idx] = translated
                if new_count % 20 == 0:
                    self.cache.save()
                if progress is not None:
                    progress(done, len(futures))
                if done % 25 == 0 or done == len(futures):
                    log.info("  번역 진행 %d/%d", done, len(futures))

        self.cache.save()
        if failures:
            log.warning(
                "번역하지 못한 문장이 %d개 있습니다. 네트워크/요청 한도를 확인한 뒤 "
                "다시 실행하면 캐시된 문장은 건너뜁니다.",
                failures,
            )
        return result


class NLLBTranslator:
    """오프라인 PyTorch 번역 (facebook/nllb-200-distilled-600M)."""

    label = "nllb"

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        cache_path: str | Path | None = None,
        *,
        source_lang: str = "eng_Latn",
        target_lang: str = "kor_Hang",
        device: str = "auto",
        threads: int = 0,
        batch_size: int = 8,
        num_beams: int = 1,
    ):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:
            raise TranslationError(
                "NLLB 번역에는 transformers와 sentencepiece가 필요합니다.\n"
                "  pip install transformers sentencepiece sacremoses"
            ) from exc

        from .asr import configure_threads, resolve_device

        configure_threads(threads)
        self.torch = torch
        self.device = resolve_device(device)
        self.batch_size = max(1, int(batch_size))
        self.num_beams = max(1, int(num_beams))
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.cache = JsonCache(cache_path)
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        log.info("NLLB 모델 로드: %s (device=%s)", model_name, self.device)
        tokenizer_kwargs = {"src_lang": self.source_lang}
        model_kwargs = {"torch_dtype": self.dtype}
        try:
            # 이미 받아 둔 모델은 네트워크 확인 없이 바로 재사용합니다.
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=True, **tokenizer_kwargs
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, local_files_only=True, **model_kwargs
            )
            log.info("NLLB 로컬 캐시 사용 (재다운로드 없음)")
        except Exception:
            log.info("NLLB 모델을 처음 내려받습니다 (약 2.4GB, 1회만)")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, **tokenizer_kwargs
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, **model_kwargs
            )
        self.model.to(self.device)
        self.model.eval()
        self.forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(
            self.target_lang
        )

    def _translate_chunk(self, chunk: list[str]) -> list[str]:
        inputs = self.tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                forced_bos_token_id=self.forced_bos_token_id,
                num_beams=self.num_beams,
                max_new_tokens=256,
            )
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)

    def translate_batch(
        self, texts: Sequence[str], progress: ProgressFn | None = None
    ) -> list[str]:
        texts = list(texts)
        result: list[str] = [""] * len(texts)
        missing: dict[str, list[int]] = {}
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            cached = self.cache.get(text)
            if cached is not None:
                result[i] = cached
            else:
                missing.setdefault(text, []).append(i)

        if not missing:
            return result

        unique = list(missing)
        log.info(
            "NLLB 번역: %d개 문장 (캐시 %d개, 신규 %d개)",
            len(texts),
            len(texts) - sum(len(v) for v in missing.values()),
            len(unique),
        )
        for start in range(0, len(unique), self.batch_size):
            chunk = unique[start : start + self.batch_size]
            translated = self._translate_chunk(chunk)
            for text, ko in zip(chunk, translated):
                ko = re.sub(r"\s+", " ", ko).strip()
                self.cache.set(text, ko)
                for idx in missing[text]:
                    result[idx] = ko
            done = min(start + len(chunk), len(unique))
            if progress is not None:
                progress(done, len(unique))
            if done % 16 == 0 or done == len(unique):
                log.info("  번역 진행 %d/%d", done, len(unique))
            self.cache.save()
        return result


def build_translator(
    name: str,
    cache_path: str | Path | None = None,
    *,
    workers: int = 3,
    nllb_model: str = "facebook/nllb-200-distilled-600M",
    device: str = "auto",
    threads: int = 0,
) -> Translator:
    name = (name or "google").lower()
    if name in ("none", "off", "no"):
        return NoopTranslator()
    if name == "google":
        return GoogleTranslator(cache_path=cache_path, workers=workers)
    if name == "nllb":
        return NLLBTranslator(
            model_name=nllb_model,
            cache_path=cache_path,
            device=device,
            threads=threads,
        )
    raise TranslationError(
        f"알 수 없는 번역기: {name} (google, nllb, none 중 하나)"
    )
