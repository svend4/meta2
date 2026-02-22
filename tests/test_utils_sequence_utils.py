"""Тесты для puzzle_reconstruction/utils/sequence_utils.py."""
import numpy as np
import pytest

from puzzle_reconstruction.utils.sequence_utils import (
    SequenceConfig,
    align_sequences,
    batch_rank,
    invert_sequence,
    kendall_tau_distance,
    longest_increasing,
    normalize_sequence,
    rank_sequence,
    segment_by_threshold,
    sliding_scores,
)


# ─── TestSequenceConfig ───────────────────────────────────────────────────────

class TestSequenceConfig:
    def test_defaults(self):
        cfg = SequenceConfig()
        assert cfg.window == 3
        assert cfg.agg == "mean"
        assert cfg.threshold == pytest.approx(0.5)

    def test_window_zero_raises(self):
        with pytest.raises(ValueError, match="window"):
            SequenceConfig(window=0)

    def test_window_negative_raises(self):
        with pytest.raises(ValueError, match="window"):
            SequenceConfig(window=-1)

    def test_window_1_valid(self):
        cfg = SequenceConfig(window=1)
        assert cfg.window == 1

    def test_window_10_valid(self):
        cfg = SequenceConfig(window=10)
        assert cfg.window == 10

    def test_invalid_agg_raises(self):
        with pytest.raises(ValueError, match="agg"):
            SequenceConfig(agg="median")

    def test_all_valid_aggs(self):
        for agg in ("mean", "max", "min", "sum"):
            cfg = SequenceConfig(agg=agg)
            assert cfg.agg == agg

    def test_threshold_below_0_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            SequenceConfig(threshold=-0.01)

    def test_threshold_above_1_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            SequenceConfig(threshold=1.01)

    def test_threshold_0_valid(self):
        cfg = SequenceConfig(threshold=0.0)
        assert cfg.threshold == 0.0

    def test_threshold_1_valid(self):
        cfg = SequenceConfig(threshold=1.0)
        assert cfg.threshold == 1.0


# ─── TestRankSequence ─────────────────────────────────────────────────────────

class TestRankSequence:
    def test_returns_ndarray(self):
        result = rank_sequence(np.array([3.0, 1.0, 2.0]))
        assert isinstance(result, np.ndarray)

    def test_shape_preserved(self):
        arr = np.array([5.0, 2.0, 8.0, 1.0])
        result = rank_sequence(arr)
        assert result.shape == arr.shape

    def test_ascending_rank(self):
        result = rank_sequence(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_descending_rank(self):
        result = rank_sequence(np.array([3.0, 2.0, 1.0]))
        np.testing.assert_array_equal(result, [3.0, 2.0, 1.0])

    def test_single_element_rank_1(self):
        result = rank_sequence(np.array([42.0]))
        assert result[0] == pytest.approx(1.0)

    def test_tied_ranks_average(self):
        # [1, 1, 3] → positions 1,2 tie → rank 1.5
        result = rank_sequence(np.array([1.0, 1.0, 3.0]))
        assert result[0] == pytest.approx(1.5)
        assert result[1] == pytest.approx(1.5)
        assert result[2] == pytest.approx(3.0)

    def test_all_tied(self):
        result = rank_sequence(np.array([5.0, 5.0, 5.0]))
        assert np.all(result == pytest.approx(2.0))  # (1+2+3)/3 = 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            rank_sequence(np.array([]))

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            rank_sequence(np.array([[1.0, 2.0]]))

    def test_min_rank_is_1(self):
        result = rank_sequence(np.array([10.0, 5.0, 8.0, 1.0]))
        assert float(result.min()) == pytest.approx(1.0)

    def test_max_rank_equals_n(self):
        arr = np.array([10.0, 5.0, 8.0, 1.0])
        result = rank_sequence(arr)
        assert float(result.max()) == pytest.approx(float(len(arr)))

    def test_dtype_float64(self):
        result = rank_sequence(np.array([3.0, 1.0, 2.0]))
        assert result.dtype == np.float64

    def test_list_input_accepted(self):
        result = rank_sequence(np.asarray([3, 1, 2]))
        assert isinstance(result, np.ndarray)


# ─── TestNormalizeSequence ────────────────────────────────────────────────────

class TestNormalizeSequence:
    def test_returns_ndarray(self):
        result = normalize_sequence(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, np.ndarray)

    def test_min_is_zero(self):
        result = normalize_sequence(np.array([5.0, 10.0, 15.0]))
        assert float(result.min()) == pytest.approx(0.0)

    def test_max_is_one(self):
        result = normalize_sequence(np.array([5.0, 10.0, 15.0]))
        assert float(result.max()) == pytest.approx(1.0)

    def test_all_values_in_0_1(self):
        arr = np.array([3.0, 7.0, 1.0, 9.0, 5.0])
        result = normalize_sequence(arr)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_constant_returns_zeros(self):
        result = normalize_sequence(np.array([4.0, 4.0, 4.0]))
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_two_elements(self):
        result = normalize_sequence(np.array([0.0, 10.0]))
        np.testing.assert_allclose(result, [0.0, 1.0])

    def test_single_element_returns_zero(self):
        result = normalize_sequence(np.array([7.0]))
        assert result[0] == pytest.approx(0.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            normalize_sequence(np.array([]))

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            normalize_sequence(np.array([[1.0, 2.0]]))

    def test_negative_values(self):
        result = normalize_sequence(np.array([-3.0, 0.0, 3.0]))
        assert float(result.min()) == pytest.approx(0.0)
        assert float(result.max()) == pytest.approx(1.0)


# ─── TestInvertSequence ───────────────────────────────────────────────────────

class TestInvertSequence:
    def test_returns_ndarray(self):
        result = invert_sequence(np.array([0.0, 0.5, 1.0]))
        assert isinstance(result, np.ndarray)

    def test_zeros_become_ones(self):
        result = invert_sequence(np.array([0.0, 0.0]))
        np.testing.assert_allclose(result, [1.0, 1.0])

    def test_ones_become_zeros(self):
        result = invert_sequence(np.array([1.0, 1.0]))
        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_midpoint_preserved(self):
        result = invert_sequence(np.array([0.5]))
        assert result[0] == pytest.approx(0.5)

    def test_is_1_minus_x(self):
        arr = np.array([0.1, 0.3, 0.7, 0.9])
        result = invert_sequence(arr)
        np.testing.assert_allclose(result, 1.0 - arr)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            invert_sequence(np.array([]))

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            invert_sequence(np.array([[0.5, 0.5]]))


# ─── TestSlidingScores ────────────────────────────────────────────────────────

class TestSlidingScores:
    def test_returns_ndarray(self):
        result = sliding_scores(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, np.ndarray)

    def test_same_length(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sliding_scores(arr)
        assert len(result) == len(arr)

    def test_mean_agg_central_element(self):
        arr = np.array([0.0, 0.0, 3.0, 0.0, 0.0])
        cfg = SequenceConfig(window=3, agg="mean")
        result = sliding_scores(arr, cfg)
        assert result[2] == pytest.approx(1.0)  # (0+3+0)/3

    def test_max_agg(self):
        arr = np.array([1.0, 5.0, 2.0])
        cfg = SequenceConfig(window=3, agg="max")
        result = sliding_scores(arr, cfg)
        assert result[1] == pytest.approx(5.0)

    def test_min_agg(self):
        arr = np.array([3.0, 1.0, 5.0])
        cfg = SequenceConfig(window=3, agg="min")
        result = sliding_scores(arr, cfg)
        assert result[1] == pytest.approx(1.0)

    def test_sum_agg(self):
        arr = np.array([1.0, 2.0, 3.0])
        cfg = SequenceConfig(window=3, agg="sum")
        result = sliding_scores(arr, cfg)
        assert result[1] == pytest.approx(6.0)

    def test_window_1_returns_same(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        cfg = SequenceConfig(window=1, agg="mean")
        result = sliding_scores(arr, cfg)
        np.testing.assert_allclose(result, arr)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            sliding_scores(np.array([]))

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            sliding_scores(np.array([[1.0, 2.0]]))

    def test_none_cfg_uses_defaults(self):
        result = sliding_scores(np.array([1.0, 2.0, 3.0]), cfg=None)
        assert result.ndim == 1


# ─── TestAlignSequences ───────────────────────────────────────────────────────

class TestAlignSequences:
    def test_returns_tuple(self):
        result = align_sequences(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_same_length_unchanged(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        ra, rb = align_sequences(a, b)
        np.testing.assert_allclose(ra, a)
        np.testing.assert_allclose(rb, b)

    def test_shorter_upsampled(self):
        a = np.array([0.0, 1.0])
        b = np.array([0.0, 0.5, 1.0])
        ra, rb = align_sequences(a, b)
        assert len(ra) == 3
        assert len(rb) == 3

    def test_target_len_explicit(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0, 5.0, 6.0])
        ra, rb = align_sequences(a, b, target_len=5)
        assert len(ra) == 5
        assert len(rb) == 5

    def test_target_len_1(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        ra, rb = align_sequences(a, b, target_len=1)
        assert len(ra) == 1
        assert len(rb) == 1

    def test_target_len_0_raises(self):
        a = np.array([1.0])
        b = np.array([2.0])
        with pytest.raises(ValueError):
            align_sequences(a, b, target_len=0)

    def test_2d_a_raises(self):
        with pytest.raises(ValueError):
            align_sequences(np.array([[1.0, 2.0]]), np.array([1.0, 2.0]))

    def test_2d_b_raises(self):
        with pytest.raises(ValueError):
            align_sequences(np.array([1.0, 2.0]), np.array([[1.0, 2.0]]))

    def test_default_target_is_max_len(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        ra, rb = align_sequences(a, b)
        assert len(ra) == 5
        assert len(rb) == 5


# ─── TestKendallTauDistance ───────────────────────────────────────────────────

class TestKendallTauDistance:
    def test_identity_is_zero(self):
        p = np.array([0, 1, 2, 3])
        assert kendall_tau_distance(p, p) == 0

    def test_reversed_is_max(self):
        p = np.array([0, 1, 2, 3])
        q = np.array([3, 2, 1, 0])
        # max inversions = n*(n-1)/2 = 6
        assert kendall_tau_distance(p, q) == 6

    def test_single_swap(self):
        p = np.array([0, 1, 2])
        q = np.array([0, 2, 1])
        assert kendall_tau_distance(p, q) == 1

    def test_returns_int(self):
        p = np.array([0, 1, 2])
        q = np.array([2, 0, 1])
        result = kendall_tau_distance(p, q)
        assert isinstance(result, int)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            kendall_tau_distance(np.array([0, 1]), np.array([0, 1, 2]))

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            kendall_tau_distance(np.array([[0, 1]]), np.array([0, 1]))

    def test_nonneg(self):
        p = np.array([0, 1, 2, 3, 4])
        q = np.array([2, 4, 1, 3, 0])
        result = kendall_tau_distance(p, q)
        assert result >= 0

    def test_symmetric(self):
        p = np.array([0, 1, 2, 3])
        q = np.array([1, 3, 0, 2])
        assert kendall_tau_distance(p, q) == kendall_tau_distance(q, p)


# ─── TestLongestIncreasing ────────────────────────────────────────────────────

class TestLongestIncreasing:
    def test_empty_returns_0(self):
        assert longest_increasing(np.array([])) == 0

    def test_single_element_returns_1(self):
        assert longest_increasing(np.array([5.0])) == 1

    def test_sorted_ascending_returns_n(self):
        assert longest_increasing(np.array([1.0, 2.0, 3.0, 4.0])) == 4

    def test_sorted_descending_returns_1(self):
        assert longest_increasing(np.array([4.0, 3.0, 2.0, 1.0])) == 1

    def test_all_equal_returns_1(self):
        assert longest_increasing(np.array([3.0, 3.0, 3.0])) == 1

    def test_known_case(self):
        # [10, 9, 2, 5, 3, 7, 101, 18] → LIS = [2,3,7,18] or [2,3,7,101] → len 4
        arr = np.array([10.0, 9.0, 2.0, 5.0, 3.0, 7.0, 101.0, 18.0])
        assert longest_increasing(arr) == 4

    def test_returns_int(self):
        result = longest_increasing(np.array([1.0, 3.0, 2.0]))
        assert isinstance(result, int)

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            longest_increasing(np.array([[1.0, 2.0]]))

    def test_two_elements_increasing(self):
        assert longest_increasing(np.array([1.0, 2.0])) == 2

    def test_two_elements_decreasing(self):
        assert longest_increasing(np.array([2.0, 1.0])) == 1


# ─── TestSegmentByThreshold ───────────────────────────────────────────────────

class TestSegmentByThreshold:
    def test_returns_list(self):
        result = segment_by_threshold(np.array([0.1, 0.9, 0.1]))
        assert isinstance(result, list)

    def test_all_below_threshold_empty(self):
        cfg = SequenceConfig(threshold=0.9)
        result = segment_by_threshold(np.array([0.1, 0.2, 0.3]), cfg)
        assert result == []

    def test_all_above_threshold_one_segment(self):
        cfg = SequenceConfig(threshold=0.0)
        result = segment_by_threshold(np.array([0.5, 0.8, 0.9]), cfg)
        assert len(result) == 1
        assert result[0] == (0, 2)

    def test_single_spike(self):
        cfg = SequenceConfig(threshold=0.6)
        arr = np.array([0.1, 0.9, 0.1])
        result = segment_by_threshold(arr, cfg)
        assert result == [(1, 1)]

    def test_two_segments(self):
        cfg = SequenceConfig(threshold=0.5)
        arr = np.array([0.8, 0.8, 0.1, 0.1, 0.9, 0.9])
        result = segment_by_threshold(arr, cfg)
        assert len(result) == 2

    def test_segment_end_indices_inclusive(self):
        cfg = SequenceConfig(threshold=0.5)
        arr = np.array([0.1, 0.8, 0.9, 0.1])
        result = segment_by_threshold(arr, cfg)
        assert result[0] == (1, 2)

    def test_segment_at_end_of_array(self):
        cfg = SequenceConfig(threshold=0.5)
        arr = np.array([0.1, 0.1, 0.8, 0.9])
        result = segment_by_threshold(arr, cfg)
        assert result[-1] == (2, 3)

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            segment_by_threshold(np.array([[0.5, 0.5]]))

    def test_none_cfg_uses_defaults(self):
        result = segment_by_threshold(np.array([0.0, 1.0, 0.0]))
        assert isinstance(result, list)


# ─── TestBatchRank ────────────────────────────────────────────────────────────

class TestBatchRank:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            batch_rank([])

    def test_returns_list(self):
        result = batch_rank([np.array([1.0, 2.0, 3.0])])
        assert isinstance(result, list)

    def test_length_matches_input(self):
        seqs = [np.array([3.0, 1.0, 2.0]), np.array([5.0, 2.0]), np.array([1.0])]
        result = batch_rank(seqs)
        assert len(result) == 3

    def test_each_element_is_ndarray(self):
        seqs = [np.array([3.0, 1.0]), np.array([2.0, 4.0, 1.0])]
        result = batch_rank(seqs)
        for arr in result:
            assert isinstance(arr, np.ndarray)

    def test_each_element_same_length_as_input(self):
        seqs = [np.array([3.0, 1.0, 2.0]), np.array([4.0, 5.0])]
        result = batch_rank(seqs)
        for inp, out in zip(seqs, result):
            assert len(out) == len(inp)

    def test_ranks_are_correct(self):
        result = batch_rank([np.array([3.0, 1.0, 2.0])])
        np.testing.assert_array_equal(result[0], [3.0, 1.0, 2.0])

    def test_single_sequence(self):
        result = batch_rank([np.array([10.0, 5.0, 8.0])])
        assert len(result) == 1
