"""Text embedders for rag mode, plus a small on-disk embedding cache.

- ``hash``: deterministic hashing-trick bag of words. Offline, no model download.
  A lexical baseline that makes rag mode runnable anywhere (CI included).
- ``local``: sentence-transformers model (``pip install rag-benchmark[local-embed]``).
- ``gemini``: Google Gemini embeddings via the ``google-genai`` SDK.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import zlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rag_benchmark.datasets import l2_normalize

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_EMBED_MODEL = "gemini-embedding-001"
DEFAULT_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_TOKEN = re.compile(r"[a-z0-9]+")
# fmt: off
_STOPWORDS = frozenset((
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does", "for",
    "from", "has", "have", "how", "i", "if", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "their", "there", "these", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "you", "your",
))
# fmt: on


class Embedder(ABC):
    """Turns texts into float32 vectors. Rows are L2-normalised."""

    #: Short name used in reports and cache keys.
    name: str = "embedder"
    #: Whether results are worth caching on disk (false for cheap embedders).
    cacheable: bool = True

    @property
    def signature(self) -> str:
        """Identifies the model + settings; part of the cache key."""
        return self.name

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed corpus chunks."""

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Embed questions. Override when the model distinguishes queries from documents."""
        return self.embed_documents(texts)


class HashingEmbedder(Embedder):
    """Signed hashing-trick term frequencies with sublinear scaling.

    It captures word overlap only (no synonyms, no semantics), which makes it a
    useful floor: a real embedding model should beat it on the same corpus.
    """

    name = "hash"
    cacheable = False

    def __init__(self, dim: int = 384):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    @property
    def signature(self) -> str:
        return f"hash:{self.dim}"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN.findall(text.lower()):
                if token in _STOPWORDS:
                    continue
                h = zlib.crc32(token.encode("utf-8"))
                out[row, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        out = np.sign(out) * np.log1p(np.abs(out))
        return l2_normalize(out)


class SentenceTransformerEmbedder(Embedder):
    """Local sentence-transformers model; runs on CPU or GPU, no API key."""

    name = "local"

    def __init__(self, model: str | None = None, batch_size: int = 64, device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "the local embedder needs sentence-transformers: "
                'pip install "rag-benchmark[local-embed]"'
            ) from exc
        self.model_name = model or DEFAULT_LOCAL_MODEL
        self.batch_size = batch_size
        self._model = SentenceTransformer(self.model_name, device=device)

    @property
    def signature(self) -> str:
        return f"local:{self.model_name}"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return l2_normalize(np.asarray(vectors, dtype=np.float32))


def _is_retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code in (429, 500, 502, 503, 504)


class GeminiEmbedder(Embedder):
    """Gemini embeddings through ``google-genai``.

    The model comes from ``model`` or ``GEMINI_EMBED_MODEL`` (default
    ``gemini-embedding-001``), the optional output size from ``dim`` or
    ``GEMINI_EMBED_DIM``. Documents and queries use the ``RETRIEVAL_DOCUMENT`` and
    ``RETRIEVAL_QUERY`` task types. Vectors are re-normalised because truncated
    outputs (dim < native size) are not unit length.
    """

    name = "gemini"

    def __init__(
        self,
        model: str | None = None,
        dim: int | None = None,
        batch_size: int = 100,
        client: Any | None = None,
        max_retries: int = 5,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model = model or os.environ.get("GEMINI_EMBED_MODEL") or DEFAULT_GEMINI_EMBED_MODEL
        env_dim = os.environ.get("GEMINI_EMBED_DIM")
        self.dim = dim if dim is not None else (int(env_dim) if env_dim else None)
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._sleep = sleep
        if client is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not set; the gemini embedder needs it")
            from google import genai

            client = genai.Client(api_key=api_key)
        self._client = client

    @property
    def signature(self) -> str:
        return f"gemini:{self.model}:{self.dim or 'default'}"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_QUERY")

    def _embed(self, texts: Sequence[str], task_type: str) -> np.ndarray:
        config: dict[str, Any] = {"task_type": task_type}
        if self.dim:
            config["output_dimensionality"] = self.dim
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = self._call(batch, config)
            embeddings = response.embeddings or []
            if len(embeddings) != len(batch):
                raise RuntimeError(f"expected {len(batch)} embeddings, got {len(embeddings)}")
            vectors.extend(list(e.values) for e in embeddings)
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def _call(self, batch: list[str], config: dict[str, Any]) -> Any:
        for attempt in range(self.max_retries + 1):
            try:
                return self._client.models.embed_content(
                    model=self.model, contents=batch, config=config
                )
            except Exception as exc:
                if attempt == self.max_retries or not _is_retryable(exc):
                    raise
                delay = min(2.0**attempt, 30.0)
                logger.warning("Gemini embed failed (%s); retrying in %.0fs", exc, delay)
                self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover


def make_embedder(kind: str, model: str | None = None, dim: int | None = None) -> Embedder:
    """Build an embedder by name: ``hash``, ``local`` or ``gemini``."""
    if kind == "hash":
        return HashingEmbedder(dim or 384)
    if kind == "local":
        return SentenceTransformerEmbedder(model)
    if kind == "gemini":
        return GeminiEmbedder(model=model, dim=dim)
    raise ValueError(f"unknown embedder {kind!r}; choose hash, local or gemini")


class EmbeddingCache:
    """Caches embeddings as ``.npy`` files keyed by embedder signature + texts.

    Saves API quota and time when the same corpus is benchmarked repeatedly.
    Pass ``directory=None`` to disable.
    """

    def __init__(self, directory: str | Path | None):
        self.directory = Path(directory) if directory else None

    def _key(self, embedder: Embedder, kind: str, texts: Sequence[str]) -> str:
        h = hashlib.sha256()
        h.update(f"{embedder.signature}\x1e{kind}\x1e".encode())
        for text in texts:
            h.update(text.encode("utf-8"))
            h.update(b"\x1f")
        return h.hexdigest()[:32]

    def embed(self, embedder: Embedder, texts: Sequence[str], kind: str) -> np.ndarray:
        """``kind`` is ``documents`` or ``queries``."""
        compute = embedder.embed_queries if kind == "queries" else embedder.embed_documents
        if self.directory is None or not embedder.cacheable:
            return compute(texts)
        path = self.directory / f"{self._key(embedder, kind, texts)}.npy"
        if path.exists():
            cached = np.load(path, allow_pickle=False)
            if cached.shape[0] == len(texts):
                logger.info("embedding cache hit: %s", path.name)
                return cached
        vectors = compute(texts)
        self.directory.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors
