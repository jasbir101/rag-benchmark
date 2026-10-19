"""Embedders. Gemini and sentence-transformers are replaced by fakes: no network."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from rag_benchmark.embedders import (
    DEFAULT_GEMINI_EMBED_MODEL,
    Embedder,
    EmbeddingCache,
    GeminiEmbedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    make_embedder,
)


class TestHashingEmbedder:
    def test_deterministic_and_normalised(self):
        emb = HashingEmbedder(dim=64)
        a = emb.embed_documents(["the quick brown fox", "lazy dogs sleep"])
        b = emb.embed_documents(["the quick brown fox", "lazy dogs sleep"])
        np.testing.assert_array_equal(a, b)
        assert a.shape == (2, 64) and a.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-5)

    def test_overlapping_text_is_closer(self):
        emb = HashingEmbedder(dim=256)
        q, near, far = emb.embed_documents(
            ["sourdough starter feeding", "feed the sourdough starter daily", "bicycle chain lube"]
        )
        assert q @ near > q @ far

    def test_text_without_tokens_is_zero_vector(self):
        assert not HashingEmbedder(8).embed_documents(["the of and"]).any()


class _FakeModels:
    def __init__(self, dim: int, failures: list[Exception] | None = None):
        self.dim = dim
        self.calls: list[dict] = []
        self.failures = list(failures or [])

    def embed_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": list(contents), "config": dict(config)})
        if self.failures:
            raise self.failures.pop(0)
        rng = np.random.default_rng(len(self.calls))
        values = rng.standard_normal((len(contents), self.dim)) * 3.0  # not unit length
        return SimpleNamespace(embeddings=[SimpleNamespace(values=v.tolist()) for v in values])


def _fake_client(dim=8, failures=None):
    return SimpleNamespace(models=_FakeModels(dim, failures))


class _ApiError(Exception):
    def __init__(self, code: int):
        super().__init__(f"HTTP {code}")
        self.code = code


class TestGeminiEmbedder:
    def test_batches_and_task_types(self):
        client = _fake_client()
        emb = GeminiEmbedder(client=client, batch_size=2)
        docs = emb.embed_documents(["a", "b", "c", "d", "e"])
        queries = emb.embed_queries(["q"])
        calls = client.models.calls
        assert [len(c["contents"]) for c in calls] == [2, 2, 1, 1]
        assert {c["config"]["task_type"] for c in calls[:3]} == {"RETRIEVAL_DOCUMENT"}
        assert calls[3]["config"]["task_type"] == "RETRIEVAL_QUERY"
        assert docs.shape == (5, 8) and queries.shape == (1, 8)
        np.testing.assert_allclose(np.linalg.norm(docs, axis=1), 1.0, rtol=1e-5)

    def test_default_model_and_no_dim(self):
        client = _fake_client()
        emb = GeminiEmbedder(client=client)
        emb.embed_documents(["x"])
        call = client.models.calls[0]
        assert call["model"] == DEFAULT_GEMINI_EMBED_MODEL
        assert "output_dimensionality" not in call["config"]

    def test_model_and_dim_from_environment(self, monkeypatch):
        monkeypatch.setenv("GEMINI_EMBED_MODEL", "some-future-embedding-model")
        monkeypatch.setenv("GEMINI_EMBED_DIM", "768")
        client = _fake_client()
        emb = GeminiEmbedder(client=client)
        emb.embed_documents(["x"])
        call = client.models.calls[0]
        assert call["model"] == "some-future-embedding-model"
        assert call["config"]["output_dimensionality"] == 768
        assert emb.signature == "gemini:some-future-embedding-model:768"

    def test_explicit_arguments_win_over_environment(self, monkeypatch):
        monkeypatch.setenv("GEMINI_EMBED_MODEL", "env-model")
        emb = GeminiEmbedder(model="arg-model", dim=256, client=_fake_client())
        assert (emb.model, emb.dim) == ("arg-model", 256)

    def test_requires_api_key_without_client(self):
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            GeminiEmbedder()

    def test_retries_rate_limits(self):
        sleeps: list[float] = []
        client = _fake_client(failures=[_ApiError(429), _ApiError(503)])
        emb = GeminiEmbedder(client=client, sleep=sleeps.append)
        assert emb.embed_documents(["x"]).shape == (1, 8)
        assert sleeps == [1.0, 2.0]

    def test_does_not_retry_client_errors(self):
        client = _fake_client(failures=[_ApiError(400)])
        emb = GeminiEmbedder(client=client, sleep=lambda _: None)
        with pytest.raises(_ApiError):
            emb.embed_documents(["x"])
        assert len(client.models.calls) == 1

    def test_gives_up_after_max_retries(self):
        client = _fake_client(failures=[_ApiError(429)] * 3)
        emb = GeminiEmbedder(client=client, max_retries=2, sleep=lambda _: None)
        with pytest.raises(_ApiError):
            emb.embed_documents(["x"])


def test_sentence_transformer_embedder_uses_model(monkeypatch):
    created = {}

    class FakeSentenceTransformer:
        def __init__(self, name, device=None):
            created["name"] = name

        def encode(self, texts, **kwargs):
            created["kwargs"] = kwargs
            return np.ones((len(texts), 4), dtype=np.float32)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    emb = SentenceTransformerEmbedder("some/local-model")
    vectors = emb.embed_documents(["a", "b"])
    assert created["name"] == "some/local-model"
    assert created["kwargs"]["normalize_embeddings"] is True
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)
    assert emb.signature == "local:some/local-model"


def test_sentence_transformer_missing_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match="local-embed"):
        SentenceTransformerEmbedder()


def test_make_embedder():
    assert isinstance(make_embedder("hash", dim=32), HashingEmbedder)
    with pytest.raises(ValueError):
        make_embedder("word2vec")


class _CountingEmbedder(Embedder):
    name = "counting"

    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return np.ones((len(texts), 3), dtype=np.float32)


class TestEmbeddingCache:
    def test_second_call_is_served_from_disk(self, tmp_path):
        emb = _CountingEmbedder()
        cache = EmbeddingCache(tmp_path)
        first = cache.embed(emb, ["a", "b"], "documents")
        second = cache.embed(emb, ["a", "b"], "documents")
        assert emb.calls == 1
        np.testing.assert_array_equal(first, second)
        assert len(list(tmp_path.glob("*.npy"))) == 1

    def test_key_depends_on_texts_and_kind(self, tmp_path):
        emb = _CountingEmbedder()
        cache = EmbeddingCache(tmp_path)
        cache.embed(emb, ["a"], "documents")
        cache.embed(emb, ["b"], "documents")
        cache.embed(emb, ["a"], "queries")
        assert emb.calls == 3

    def test_disabled_cache_writes_nothing(self, tmp_path):
        emb = _CountingEmbedder()
        cache = EmbeddingCache(None)
        cache.embed(emb, ["a"], "documents")
        cache.embed(emb, ["a"], "documents")
        assert emb.calls == 2
        assert not list(tmp_path.iterdir())
