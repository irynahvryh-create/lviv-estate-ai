import collections
import math
import os
import re
import time
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

_EMBED_ITEMS_PER_MIN = 100
_EMBED_SAFE = 90
_EMBED_WINDOW = 60.0

def _retry_delay_seconds(exc) -> float:
    m = re.search(r"retry in ([\d.]+)s", str(exc))
    return float(m.group(1)) + 1.0 if m else 50.0

class EmbeddingClient:
    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or os.environ.get("EMBEDDING_PROVIDER", "gemini")).lower()
        self.calls = 0
        self.texts_embedded = 0
        self._recent: collections.deque = collections.deque()

        if self.provider == "gemini":
            self._init_gemini()
        else:
            raise ValueError(f"Unknown provider {self.provider!r}")

    def _items_in_window(self) -> int:
        cutoff = time.time() - _EMBED_WINDOW
        while self._recent and self._recent[0][0] < cutoff:
            self._recent.popleft()
        return sum(count for _, count in self._recent)

    def _throttle(self, n: int) -> None:
        if self.provider != "gemini":
            return
        while self._items_in_window() + n > _EMBED_SAFE and self._recent:
            wait = _EMBED_WINDOW - (time.time() - self._recent[0][0]) + 0.5
            if wait <= 0:
                continue
            print(f"  [embedding quota: waiting {wait:.0f}s]")
            time.sleep(wait)

    def _init_gemini(self) -> None:
        from google import genai
        self.model = "gemini-embedding-2"
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def _embed_batch_gemini(self, texts, task_type, output_dimensionality):
        from google.genai import errors, types

        contents = [types.Content(parts=[types.Part(text=t)]) for t in texts]
        config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=output_dimensionality)
        
        for attempt in range(1, 6):
            try:
                response = self._client.models.embed_content(
                    model=self.model, contents=contents, config=config
                )
                self.calls += 1
                return [e.values for e in response.embeddings]
            except errors.ClientError as exc:
                if getattr(exc, "code", None) != 429 or attempt == 5:
                    raise
                delay = _retry_delay_seconds(exc)
                print(f"  [429 embedding quota; waiting {delay:.0f}s, retry {attempt}/4]")
                time.sleep(delay)

    def embed(self, text: str, *, task_type: Optional[str] = None, output_dimensionality: Optional[int] = None) -> list[float]:
        return self.embed_batch([text], task_type=task_type, output_dimensionality=output_dimensionality)[0]

    def embed_batch(self, texts: list[str], *, task_type: Optional[str] = None, output_dimensionality: Optional[int] = None) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_SAFE):
            window = texts[start : start + _EMBED_SAFE]
            self._throttle(len(window))
            got = self._embed_batch_gemini(window, task_type, output_dimensionality)
            if len(got) != len(window):
                raise RuntimeError(f"sent {len(window)} texts, got {len(got)} vectors back")
            vectors.extend(got)
            self._recent.append((time.time(), len(window)))
            self.texts_embedded += len(window)
        return vectors

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b)