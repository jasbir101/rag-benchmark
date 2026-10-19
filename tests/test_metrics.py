import numpy as np
import pytest

from rag_benchmark.metrics import (
    first_relevant_rank,
    hit_rate_at_k,
    mrr_at_k,
    percentile,
    recall_at_k,
    summarize_latencies,
)


class TestRecall:
    def test_perfect(self):
        truth = np.array([[1, 2, 3], [4, 5, 6]])
        assert recall_at_k(truth[:, ::-1], truth, 3) == 1.0

    def test_partial(self):
        truth = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        got = np.array([[1, 2, 9, 10], [5, 11, 12, 13]])
        # (2 + 1) hits over 2 queries * 4
        assert recall_at_k(got, truth, 4) == pytest.approx(3 / 8)

    def test_only_first_k_columns_count(self):
        truth = np.array([[1, 2, 3, 4]])
        got = np.array([[9, 1, 2, 3]])
        assert recall_at_k(got, truth, 2) == pytest.approx(0.5)  # [9, 1] vs [1, 2]

    def test_padding_never_matches(self):
        truth = np.array([[0, 1, 2]])
        got = np.array([[0, -1, -1]])
        assert recall_at_k(got, truth, 3) == pytest.approx(1 / 3)

    def test_duplicates_counted_once(self):
        truth = np.array([[1, 2]])
        got = np.array([[1, 1]])
        assert recall_at_k(got, truth, 2) == pytest.approx(0.5)

    def test_ties_count_with_scores(self):
        # ids 2 and 3 are equally close; the backend returned 3 instead of 2.
        truth = np.array([[1, 2]])
        got = np.array([[1, 3]])
        truth_scores = np.array([[0.9, 0.5]])
        got_scores = np.array([[0.9, 0.5]])
        assert recall_at_k(got, truth, 2) == 0.5
        assert (
            recall_at_k(got, truth, 2, retrieved_scores=got_scores, truth_scores=truth_scores)
            == 1.0
        )

    def test_worse_item_is_not_a_tie(self):
        truth = np.array([[1, 2]])
        got = np.array([[1, 3]])
        assert (
            recall_at_k(
                got,
                truth,
                2,
                retrieved_scores=np.array([[0.9, 0.4]]),
                truth_scores=np.array([[0.9, 0.5]]),
            )
            == 0.5
        )

    @pytest.mark.parametrize("k", [0, 4])
    def test_invalid_k(self, k):
        with pytest.raises(ValueError):
            recall_at_k(np.zeros((1, 3)), np.zeros((1, 3)), k)

    def test_query_count_mismatch(self):
        with pytest.raises(ValueError, match="mismatch"):
            recall_at_k(np.zeros((2, 3)), np.zeros((3, 3)), 3)


class TestRankMetrics:
    def test_first_relevant_rank(self):
        assert first_relevant_rank(["a", "b", "c"], "b") == 2
        assert first_relevant_rank(["a", "b", "b"], "b") == 2
        assert first_relevant_rank(["a"], "z") is None

    def test_hit_rate(self):
        ranks = [1, 3, None, 5]
        assert hit_rate_at_k(ranks, 3) == pytest.approx(0.5)
        assert hit_rate_at_k(ranks, 5) == pytest.approx(0.75)

    def test_mrr(self):
        ranks = [1, 2, None, 4]
        assert mrr_at_k(ranks, 10) == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)
        # rank 4 falls outside k=3
        assert mrr_at_k(ranks, 3) == pytest.approx((1 + 0.5) / 4)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            hit_rate_at_k([], 3)
        with pytest.raises(ValueError):
            mrr_at_k([], 3)


class TestLatency:
    def test_percentile_linear_interpolation(self):
        values = list(range(1, 101))  # 1..100
        assert percentile(values, 50) == pytest.approx(50.5)
        assert percentile(values, 95) == pytest.approx(95.05)
        assert percentile(values, 99) == pytest.approx(99.01)
        assert percentile(values, 0) == 1
        assert percentile(values, 100) == 100

    def test_percentile_single_value(self):
        assert percentile([7.0], 99) == 7.0

    def test_percentile_errors(self):
        with pytest.raises(ValueError):
            percentile([], 50)
        with pytest.raises(ValueError):
            percentile([1.0], 101)

    def test_summarize_converts_seconds_to_ms(self):
        stats = summarize_latencies([0.001, 0.002, 0.003, 0.004])
        assert stats.p50_ms == pytest.approx(2.5)
        assert stats.mean_ms == pytest.approx(2.5)
        assert stats.p99_ms == pytest.approx(3.97)
        # without a wall time, qps = n / sum(latencies) = 4 / 0.010
        assert stats.qps == pytest.approx(400.0)

    def test_summarize_uses_wall_time(self):
        stats = summarize_latencies([0.001] * 10, wall_time_s=0.02)
        assert stats.qps == pytest.approx(500.0)

    def test_summarize_empty(self):
        with pytest.raises(ValueError):
            summarize_latencies([])
