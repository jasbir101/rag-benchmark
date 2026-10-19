import numpy as np
import pytest

from rag_benchmark.adapters.faiss_backend import FaissStore, resolve_nlist
from rag_benchmark.datasets import exact_knn
from rag_benchmark.metrics import recall_at_k

K = 10


def _recall(store, dataset):
    store.build(dataset.base)
    got = store.search(dataset.queries, K)
    truth = exact_knn(dataset.base, dataset.queries, K, dataset.metric)
    return got, recall_at_k(got, truth, K)


@pytest.mark.parametrize("fixture", ["small_dataset", "small_l2_dataset"])
def test_flat_is_exact(request, fixture):
    dataset = request.getfixturevalue(fixture)
    with FaissStore(dataset.dim, dataset.metric, {"index": "flat"}) as store:
        got, recall = _recall(store, dataset)
    assert recall == 1.0
    assert got.shape == (dataset.n_queries, K)
    assert got.dtype == np.int64


def test_hnsw_high_recall(small_dataset):
    params = {"index": "hnsw", "M": 16, "ef_construction": 100, "ef_search": 128}
    with FaissStore(small_dataset.dim, "cosine", params) as store:
        _, recall = _recall(store, small_dataset)
    assert recall >= 0.95


def test_ivf_probing_every_list_is_exact(small_dataset):
    params = {"index": "ivf_flat", "nlist": 16, "nprobe": 16}
    with FaissStore(small_dataset.dim, "cosine", params) as store:
        _, recall = _recall(store, small_dataset)
    assert recall == 1.0


def test_ivf_fewer_probes_never_beats_exhaustive(small_dataset):
    recalls = []
    for nprobe in (1, 16):
        with FaissStore(
            small_dataset.dim, "cosine", {"index": "ivf_flat", "nlist": 32, "nprobe": nprobe}
        ) as store:
            recalls.append(_recall(store, small_dataset)[1])
    assert recalls[0] <= recalls[1]


def test_single_query_vector_is_accepted(small_dataset):
    with FaissStore(small_dataset.dim, "cosine") as store:
        store.build(small_dataset.base)
        assert store.search(small_dataset.queries[0], 5).shape == (1, 5)


def test_search_before_build_raises():
    with pytest.raises(RuntimeError):
        FaissStore(8).search(np.zeros((1, 8), dtype=np.float32), 1)


def test_wrong_dimension_raises(small_dataset):
    store = FaissStore(small_dataset.dim + 1)
    with pytest.raises(ValueError):
        store.build(small_dataset.base)


def test_unknown_index_type():
    with pytest.raises(ValueError):
        FaissStore(8, "cosine", {"index": "pq"})


def test_resolve_nlist():
    assert resolve_nlist("auto", 20_000) == 512  # 4*sqrt(n)=566, capped at n // 39
    assert resolve_nlist("auto", 1_000_000) == 4_000
    assert resolve_nlist(128, 20_000) == 128
    with pytest.raises(ValueError):
        resolve_nlist(500, 100)


def test_is_available():
    assert FaissStore.unavailable_reason() is None
