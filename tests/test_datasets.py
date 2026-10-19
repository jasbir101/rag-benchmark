import numpy as np
import pytest

from rag_benchmark.datasets import (
    VectorDataset,
    exact_knn,
    l2_normalize,
    load_npy,
    make_synthetic,
    similarity_scores,
)


class TestSynthetic:
    def test_same_seed_is_identical(self):
        a = make_synthetic(n=500, dim=16, n_queries=20, seed=3)
        b = make_synthetic(n=500, dim=16, n_queries=20, seed=3)
        np.testing.assert_array_equal(a.base, b.base)
        np.testing.assert_array_equal(a.queries, b.queries)

    def test_different_seed_differs(self):
        a = make_synthetic(n=500, dim=16, n_queries=20, seed=3)
        b = make_synthetic(n=500, dim=16, n_queries=20, seed=4)
        assert not np.allclose(a.base, b.base)

    def test_shapes_and_dtype(self):
        ds = make_synthetic(n=300, dim=8, n_queries=10)
        assert ds.base.shape == (300, 8)
        assert ds.queries.shape == (10, 8)
        assert ds.base.dtype == np.float32
        assert (ds.n, ds.dim, ds.n_queries) == (300, 8, 10)

    def test_cosine_vectors_are_unit_length(self):
        ds = make_synthetic(n=200, dim=8, n_queries=5, metric="cosine")
        np.testing.assert_allclose(np.linalg.norm(ds.base, axis=1), 1.0, rtol=1e-5)

    def test_l2_vectors_are_not_normalised(self):
        ds = make_synthetic(n=200, dim=8, n_queries=5, metric="l2", cluster_std=0.5)
        assert not np.allclose(np.linalg.norm(ds.base, axis=1), 1.0)

    def test_queries_are_not_base_rows(self):
        ds = make_synthetic(n=200, dim=8, n_queries=5)
        for q in ds.queries:
            assert not np.any(np.all(ds.base == q, axis=1))

    @pytest.mark.parametrize("kwargs", [{"n": 0}, {"dim": 0}, {"cluster_std": -1.0}])
    def test_invalid_arguments(self, kwargs):
        with pytest.raises(ValueError):
            make_synthetic(**kwargs)


def _brute_force(base, queries, k, metric):
    if metric == "cosine":
        base, queries = l2_normalize(base), l2_normalize(queries)
        scores = queries.astype(np.float64) @ base.astype(np.float64).T
    else:
        diff = queries[:, None, :].astype(np.float64) - base[None, :, :].astype(np.float64)
        scores = -np.sum(diff**2, axis=2)
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


class TestExactKnn:
    @pytest.mark.parametrize("metric", ["cosine", "l2"])
    def test_matches_full_sort(self, metric):
        rng = np.random.default_rng(0)
        base = rng.standard_normal((400, 12)).astype(np.float32)
        queries = rng.standard_normal((25, 12)).astype(np.float32)
        np.testing.assert_array_equal(
            exact_knn(base, queries, 10, metric), _brute_force(base, queries, 10, metric)
        )

    def test_k_equal_to_n(self):
        rng = np.random.default_rng(1)
        base = rng.standard_normal((20, 4)).astype(np.float32)
        ids = exact_knn(base, base[:3], 20, "l2")
        assert ids.shape == (3, 20)
        assert ids[:, 0].tolist() == [0, 1, 2]  # each vector is its own nearest neighbour

    def test_ties_broken_by_lower_id(self):
        base = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        ids = exact_knn(base, np.array([[1.0, 0.0]], dtype=np.float32), 2, "cosine")
        assert ids.tolist() == [[0, 1]]

    def test_invalid_k(self):
        base = np.zeros((5, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            exact_knn(base, base, 6, "l2")
        with pytest.raises(ValueError):
            exact_knn(base, base, 1, "dot")


def test_similarity_scores_pads_with_minus_inf():
    base = np.eye(3, dtype=np.float32)
    queries = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    scores = similarity_scores(base, queries, np.array([[0, 1, -1]]), "cosine")
    assert scores[0, 0] == pytest.approx(1.0)
    assert scores[0, 1] == pytest.approx(0.0)
    assert scores[0, 2] == -np.inf


class TestLoadNpy:
    def test_holds_out_queries(self, tmp_path):
        data = np.random.default_rng(0).standard_normal((100, 6)).astype(np.float32)
        np.save(tmp_path / "vecs.npy", data)
        ds = load_npy(tmp_path / "vecs.npy", n_queries=10, metric="l2")
        assert ds.n == 90 and ds.n_queries == 10
        rows = {tuple(r) for r in ds.base.tolist()}
        assert not any(tuple(q) in rows for q in ds.queries.tolist())

    def test_separate_query_file(self, tmp_path):
        rng = np.random.default_rng(0)
        np.save(tmp_path / "base.npy", rng.standard_normal((50, 4)).astype(np.float32))
        np.save(tmp_path / "q.npy", rng.standard_normal((8, 4)).astype(np.float32))
        ds = load_npy(tmp_path / "base.npy", n_queries=5, queries_path=tmp_path / "q.npy")
        assert ds.n == 50 and ds.n_queries == 5

    def test_rejects_non_matrix(self, tmp_path):
        np.save(tmp_path / "bad.npy", np.zeros(10, dtype=np.float32))
        with pytest.raises(ValueError):
            load_npy(tmp_path / "bad.npy")


def test_dataset_validates_dimensions():
    with pytest.raises(ValueError, match="dimension"):
        VectorDataset("x", np.zeros((3, 4)), np.zeros((2, 5)))
