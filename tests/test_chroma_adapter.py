import numpy as np
import pytest

from rag_benchmark.adapters.chroma_backend import ChromaStore
from rag_benchmark.datasets import exact_knn
from rag_benchmark.metrics import recall_at_k

K = 10


@pytest.mark.parametrize("fixture", ["small_dataset", "small_l2_dataset"])
def test_hnsw_high_recall(request, fixture):
    dataset = request.getfixturevalue(fixture)
    params = {"M": 16, "ef_construction": 100, "ef_search": 128}
    with ChromaStore(dataset.dim, dataset.metric, params) as store:
        store.build(dataset.base)
        got = store.search(dataset.queries, K)
    truth = exact_knn(dataset.base, dataset.queries, K, dataset.metric)
    assert got.shape == (dataset.n_queries, K)
    assert got.dtype == np.int64
    assert recall_at_k(got, truth, K) >= 0.95


def test_ids_are_row_numbers(small_dataset):
    with ChromaStore(small_dataset.dim, "cosine") as store:
        store.build(small_dataset.base)
        # a base vector's nearest neighbour is itself
        got = store.search(small_dataset.base[[3, 42, 1999]], 1)
    assert got[:, 0].tolist() == [3, 42, 1999]


def test_close_drops_collection(small_dataset):
    import chromadb
    from chromadb.config import Settings

    store = ChromaStore(small_dataset.dim, "cosine")
    store.build(small_dataset.base[:100])
    name = store._collection_name
    store.close()
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    assert name not in [c.name if hasattr(c, "name") else c for c in client.list_collections()]


def test_hnsw_configuration_mapping():
    store = ChromaStore(8, "l2", {"M": 24, "ef_construction": 150, "ef_search": 40})
    assert store._hnsw_config() == {
        "space": "l2",
        "max_neighbors": 24,
        "ef_construction": 150,
        "ef_search": 40,
    }


def test_search_before_build_raises():
    with pytest.raises(RuntimeError):
        ChromaStore(4).search(np.zeros((1, 4), dtype=np.float32), 1)
