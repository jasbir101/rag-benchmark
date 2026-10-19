"""Milvus and Pinecone: skipped cleanly without configuration, and adapter logic
verified against in-memory fakes of their SDKs (no network, no accounts)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from rag_benchmark.adapters import IndexConfig, milvus_backend
from rag_benchmark.adapters.milvus_backend import MilvusStore
from rag_benchmark.adapters.pinecone_backend import PineconeStore
from rag_benchmark.datasets import exact_knn
from rag_benchmark.metrics import recall_at_k
from rag_benchmark.runner import run_benchmark

K = 10


def _top_k(matrix: np.ndarray, query: np.ndarray, k: int) -> list[int]:
    scores = matrix @ query
    return np.argsort(-scores, kind="stable")[:k].tolist()


# -- skip behaviour -----------------------------------------------------------------


def _pretend_installed(monkeypatch, *modules: str) -> None:
    """Make the optional SDK imports succeed, so these tests behave the same whether
    or not the milvus/pinecone extras are installed."""
    for name in modules:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))


def test_milvus_unavailable_without_uri(monkeypatch):
    _pretend_installed(monkeypatch, "pymilvus")
    reason = MilvusStore.unavailable_reason()
    assert reason is not None and "MILVUS_URI" in reason


def test_milvus_lite_uri_rejected_on_windows(monkeypatch):
    _pretend_installed(monkeypatch, "pymilvus")
    monkeypatch.setattr(milvus_backend.sys, "platform", "win32")
    monkeypatch.setenv("MILVUS_URI", "./milvus_demo.db")
    reason = MilvusStore.unavailable_reason()
    assert reason is not None and "Milvus Lite" in reason


def test_pinecone_unavailable_without_api_key(monkeypatch):
    _pretend_installed(monkeypatch, "pinecone")
    reason = PineconeStore.unavailable_reason()
    assert reason is not None and "PINECONE_API_KEY" in reason


def test_blank_api_key_counts_as_unset(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "   ")
    assert PineconeStore.unavailable_reason() is not None


def test_missing_sdk_is_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymilvus", None)  # makes `import pymilvus` fail
    monkeypatch.setenv("MILVUS_URI", "http://localhost:19530")
    assert "not installed" in MilvusStore.unavailable_reason()


def test_runner_skips_unconfigured_backends_once(small_dataset, monkeypatch):
    _pretend_installed(monkeypatch, "pymilvus", "pinecone")
    configs = [
        IndexConfig("milvus", "hnsw a", {"index": "hnsw"}),
        IndexConfig("milvus", "hnsw b", {"index": "hnsw", "ef_search": 128}),
        IndexConfig("pinecone", "default", {}),
        IndexConfig("faiss", "flat", {"index": "flat"}),
    ]
    results = run_benchmark(small_dataset, configs, K, warmup=0)
    assert [(r.backend, r.status) for r in results] == [
        ("milvus", "skipped"),
        ("pinecone", "skipped"),
        ("faiss", "ok"),
    ]
    assert "MILVUS_URI" in results[0].note


# -- Milvus adapter against a fake pymilvus ------------------------------------------


class _FakeSchema:
    def __init__(self) -> None:
        self.fields: list[tuple[str, str, dict]] = []

    def add_field(self, name, dtype, **kwargs):
        self.fields.append((name, dtype, kwargs))


class _FakeIndexParams(list):
    def add_index(self, field_name, index_type="", metric_type="", params=None, **_):
        self.append(
            {"field": field_name, "type": index_type, "metric": metric_type, "params": params}
        )


class FakeMilvusClient:
    last: FakeMilvusClient | None = None

    def __init__(self, uri: str, token: str = ""):
        self.uri, self.token = uri, token
        self.collections: dict[str, dict] = {}
        self.search_params: list[dict] = []
        self.closed = False
        FakeMilvusClient.last = self

    @classmethod
    def create_schema(cls, **kwargs):
        return _FakeSchema()

    @classmethod
    def prepare_index_params(cls):
        return _FakeIndexParams()

    def create_collection(self, name, schema=None, **kwargs):
        self.collections[name] = {"schema": schema, "rows": [], "index": None, "loaded": False}

    def insert(self, name, data):
        self.collections[name]["rows"].extend(data)

    def flush(self, name):
        pass

    def create_index(self, name, index_params):
        self.collections[name]["index"] = index_params[0]

    def load_collection(self, name):
        self.collections[name]["loaded"] = True

    def search(self, name, data, limit, search_params, anns_field):
        coll = self.collections[name]
        assert coll["loaded"], "search before load"
        self.search_params.append(search_params)
        ids = np.array([r["id"] for r in coll["rows"]])
        matrix = np.array([r["vector"] for r in coll["rows"]], dtype=np.float32)
        return [
            [{"id": int(ids[i]), "distance": 0.0} for i in _top_k(matrix, np.array(q), limit)]
            for q in data
        ]

    def drop_collection(self, name):
        del self.collections[name]

    def close(self):
        self.closed = True


@pytest.fixture
def fake_pymilvus(monkeypatch):
    module = types.ModuleType("pymilvus")
    module.MilvusClient = FakeMilvusClient
    module.DataType = SimpleNamespace(INT64="INT64", FLOAT_VECTOR="FLOAT_VECTOR")
    monkeypatch.setitem(sys.modules, "pymilvus", module)
    monkeypatch.setenv("MILVUS_URI", "http://milvus.invalid:19530")
    monkeypatch.setenv("MILVUS_TOKEN", "placeholder-token")
    return module


def test_milvus_adapter_roundtrip(fake_pymilvus, small_dataset):
    assert MilvusStore.unavailable_reason() is None
    params = {"index": "hnsw", "M": 16, "ef_construction": 64, "ef_search": 32}
    store = MilvusStore(small_dataset.dim, "cosine", params)
    store.build(small_dataset.base)
    got = store.search(small_dataset.queries, K)

    client = FakeMilvusClient.last
    assert client.uri == "http://milvus.invalid:19530"
    assert client.token == "placeholder-token"
    coll = client.collections[store.collection]
    assert len(coll["rows"]) == small_dataset.n
    assert coll["index"] == {
        "field": "vector",
        "type": "HNSW",
        "metric": "COSINE",
        "params": {"M": 16, "efConstruction": 64},
    }
    assert client.search_params[-1] == {"metric_type": "COSINE", "params": {"ef": 32}}

    truth = exact_knn(small_dataset.base, small_dataset.queries, K, "cosine")
    assert recall_at_k(got, truth, K) == 1.0

    store.close()
    assert store.collection not in client.collections
    assert client.closed


def test_milvus_ivf_params(fake_pymilvus):
    store = MilvusStore(8, "l2", {"index": "ivf_flat", "nlist": 64, "nprobe": 8})
    assert store._index_params() == {"nlist": 64}
    assert store._search_params() == {"metric_type": "L2", "params": {"nprobe": 8}}


def test_milvus_through_runner(fake_pymilvus, small_dataset):
    configs = [IndexConfig("milvus", "hnsw", {"index": "hnsw"})]
    [result] = run_benchmark(small_dataset, configs, K, warmup=2)
    assert result.status == "ok", result.note
    assert result.recall_at_k == 1.0


# -- Pinecone adapter against a fake SDK ---------------------------------------------


class FakePineconeIndex:
    def __init__(self, dimension: int, lag: int = 0):
        self.dimension = dimension
        self.lag = lag  # stats calls that report stale (zero) counts
        self.namespaces: dict[str, dict[str, list[float]]] = {}

    def upsert(self, vectors, namespace="", show_progress=True):
        for vec_id, values in vectors:
            self.namespaces.setdefault(namespace, {})[vec_id] = values

    def describe_index_stats(self):
        stale = self.lag > 0
        self.lag = max(0, self.lag - 1)
        counts = {
            ns: SimpleNamespace(vector_count=0 if stale else len(v))
            for ns, v in self.namespaces.items()
        }
        return SimpleNamespace(dimension=self.dimension, namespaces=counts)

    def query(self, vector, top_k, namespace=""):
        records = self.namespaces[namespace]
        ids = list(records)
        matrix = np.array([records[i] for i in ids], dtype=np.float32)
        best = _top_k(matrix, np.array(vector, dtype=np.float32), top_k)
        return SimpleNamespace(matches=[SimpleNamespace(id=ids[i], score=0.0) for i in best])

    def delete(self, delete_all=False, namespace=""):
        assert delete_all
        self.namespaces.pop(namespace, None)


class FakePinecone:
    indexes: ClassVar[dict[str, FakePineconeIndex]] = {}
    created: ClassVar[list[dict]] = []
    deleted: ClassVar[list[str]] = []
    lag = 0

    def __init__(self, api_key: str):
        assert api_key

    def has_index(self, name):
        return name in self.indexes

    def create_index(self, name, dimension, metric, spec):
        FakePinecone.created.append(
            {"name": name, "dimension": dimension, "metric": metric, "spec": spec}
        )
        self.indexes[name] = FakePineconeIndex(dimension, lag=FakePinecone.lag)

    def Index(self, name):
        return self.indexes[name]

    def delete_index(self, name):
        FakePinecone.deleted.append(name)
        del self.indexes[name]


@pytest.fixture
def fake_pinecone(monkeypatch):
    FakePinecone.indexes, FakePinecone.created, FakePinecone.deleted = {}, [], []
    FakePinecone.lag = 2
    module = types.ModuleType("pinecone")
    module.Pinecone = FakePinecone
    module.ServerlessSpec = lambda cloud, region: {"cloud": cloud, "region": region}
    monkeypatch.setitem(sys.modules, "pinecone", module)
    monkeypatch.setenv("PINECONE_API_KEY", "placeholder-key")
    return FakePinecone


def test_pinecone_creates_index_waits_and_cleans_up(fake_pinecone, small_dataset):
    sleeps: list[float] = []
    store = PineconeStore(small_dataset.dim, "cosine", {}, sleep=sleeps.append)
    store.build(small_dataset.base)
    assert sleeps, "should poll until the upserted vectors are visible"
    assert fake_pinecone.created == [
        {
            "name": "rag-benchmark",
            "dimension": small_dataset.dim,
            "metric": "cosine",
            "spec": {"cloud": "aws", "region": "us-east-1"},
        }
    ]

    got = store.search(small_dataset.queries[:10], K)
    truth = exact_knn(small_dataset.base, small_dataset.queries[:10], K, "cosine")
    assert recall_at_k(got, truth, K) == 1.0

    index = fake_pinecone.indexes["rag-benchmark"]
    store.close()
    assert store.namespace not in index.namespaces
    assert fake_pinecone.deleted == ["rag-benchmark"]


def test_pinecone_reuses_existing_index_and_keeps_it(fake_pinecone, monkeypatch):
    monkeypatch.setenv("PINECONE_INDEX", "my-index")
    fake_pinecone.indexes["my-index"] = FakePineconeIndex(dimension=8)
    data = np.random.default_rng(0).standard_normal((30, 8)).astype(np.float32)
    store = PineconeStore(8, "l2", {}, sleep=lambda _: None)
    store.build(data)
    store.close()
    assert fake_pinecone.created == []
    assert fake_pinecone.deleted == []


def test_pinecone_dimension_mismatch(fake_pinecone):
    fake_pinecone.indexes["rag-benchmark"] = FakePineconeIndex(dimension=16)
    store = PineconeStore(8, "cosine", {}, sleep=lambda _: None)
    with pytest.raises(ValueError, match="dimension"):
        store.build(np.ones((4, 8), dtype=np.float32))
