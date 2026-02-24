"""Extra tests for puzzle_reconstruction/utils/sequence_utils.py."""
from __future__ import annotations

import pytest
import numpy as np

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _seq(n=10, val=0.5) -> np.ndarray:
    return np.full(n, val)


def _ramp(n=10) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


# ─── SequenceConfig ───────────────────────────────────────────────────────────

class TestSequenceConfigExtra:
    def test_default_window(self):
        assert SequenceConfig().window == 3

    def test_default_agg(self):
        assert SequenceConfig().agg == "mean"

    def test_default_threshold(self):
        assert SequenceConfig().threshold == pytest.approx(0.5)

    def test_window_zero_raises(self):
        with pytest.raises(ValueError):
            SequenceConfig(window=0)

    def test_window_negative_raises(self):
        with pytest.raises(ValueError):
            SequenceConfig(window=-1)

    def test_invalid_agg_raises(self):
        with pytest.raises(ValueError):
            SequenceConfig(agg="median")

    def test_valid_aggs(self):
        for agg in ("mean", "max", "min", "sum"):
            cfg = SequenceConfig(agg=agg)
            assert cfg.agg == agg

    def test_threshold_below_zero_raises(self):
        with pytest.raises(ValueError):
            SequenceConfig(threshold=-0.01)

    def test_threshold_above_one_raises(self):
        with pytest.raises(ValueError):
            SequenceConfig(threshold=1.01)

    def test_threshold_at_boundaries(self):
        assert SequenceConfig(threshold=0.0).threshold == pytest.approx(0.0)
        assert SequenceConfig(threshold=1.0).threshold == pytest.approx(1.0)


# ─── rank_sequence ────────────────────────────────────────────────────────────

class TestRankSequenceExtra:
    def test_returns_ndarray(self):
        assert isinstance(rank_sequence(np.array([3.0, 1.0, 2.0])), np.ndarray)

    def test_length_preserved(self):
        seq = np.array([1.0, 2.0, 3.0])
        assert len(rank_sequence(seq)) == 3

    def test_sorted_ascending_ranks(self):
        seq = np.array([10.0, 20.0, 30.0])
        ranks = rank_sequence(seq)
        assert ranks[0] < ranks[1] < ranks[2]

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            rank_sequence(np.zeros((2, 3)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            rank_sequence(np.array([]))

    def test_all_same_tied_rank(self):
        seq = np.array([5.0, 5.0, 5.0])
        ranks = rank_sequence(seq)
        assert np.allclose(ranks, ranks[0])

    def test_single_element(self):
        ranks = rank_sequence(np.array([7.0]))
        assert ranks[0] == pytest.approx(1.0)


# ─── normalize_sequence ───────────────────────────────────────────────────────

class TestNormalizeSequenceExtra:
    def test_returns_ndarray(self):
        assert isinstance(normalize_sequence(_ramp()), np.ndarray)

    def test_min_is_zero(self):
        out = normalize_sequence(_ramp())
        assert out.min() == pytest.approx(0.0)

    def test_max_is_one(self):
        out = normalize_sequence(_ramp())
        assert out.max() == pytest.approx(1.0)

    def test_constant_returns_zeros(self):
        out = normalize_sequence(_seq(5, 3.0))
        assert np.allclose(out, 0.0)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            normalize_sequence(np.zeros((2, 3)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            normalize_sequence(np.array([]))

    def test_dtype_float64(self):
        assert normalize_sequence(_ramp()).dtype == np.float64

    def test_length_preserved(self):
        seq = _ramp(15)
        assert len(normalize_sequence(seq)) == 15


# ─── invert_sequence ──────────────────────────────────────────────────────────

class TestInvertSequenceExtra:
    def test_returns_ndarray(self):
        assert isinstance(invert_sequence(_ramp()), np.ndarray)

    def test_inverts_values(self):
        seq = np.array([0.0, 0.5, 1.0])
        out = invert_sequence(seq)
        np.testing.assert_allclose(out, [1.0, 0.5, 0.0])

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            invert_sequence(np.zeros((2, 3)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            invert_sequence(np.array([]))

    def test_double_invert_identity(self):
        seq = _ramp(10)
        np.testing.assert_allclose(invert_sequence(invert_sequence(seq)), seq)

    def test_length_preserved(self):
        seq = _ramp(8)
        assert len(invert_sequence(seq)) == 8


# ─── sliding_scores ───────────────────────────────────────────────────────────

class TestSlidingScoresExtra:
    def test_returns_ndarray(self):
        assert isinstance(sliding_scores(_ramp()), np.ndarray)

    def test_same_length(self):
        seq = _ramp(12)
        out = sliding_scores(seq)
        assert len(out) == 12

    def test_constant_seq_same_output(self):
        seq = _seq(10, 0.7)
        cfg = SequenceConfig(window=3, agg="mean")
        out = sliding_scores(seq, cfg=cfg)
        assert np.allclose(out, 0.7)

    def test_max_agg(self):
        seq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cfg = SequenceConfig(window=3, agg="max")
        out = sliding_scores(seq, cfg=cfg)
        assert len(out) == 5

    def test_min_agg(self):
        seq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cfg = SequenceConfig(window=3, agg="min")
        out = sliding_scores(seq, cfg=cfg)
        assert len(out) == 5

    def test_sum_agg(self):
        seq = np.ones(8)
        cfg = SequenceConfig(window=3, agg="sum")
        out = sliding_scores(seq, cfg=cfg)
        assert len(out) == 8

    def test_none_cfg(self):
        out = sliding_scores(_ramp(10), cfg=None)
        assert len(out) == 10


# ─── align_sequences ──────────────────────────────────────────────────────────

class TestAlignSequencesExtra:
    def test_returns_tuple_2(self):
        a = _ramp(5)
        b = _ramp(8)
        result = align_sequences(a, b)
        assert isinstance(result, tuple) and len(result) == 2

    def test_same_length_output(self):
        a = _ramp(5)
        b = _ramp(8)
        out_a, out_b = align_sequences(a, b)
        assert len(out_a) == len(out_b)

    def test_default_target_max_len(self):
        a = _ramp(5)
        b = _ramp(8)
        out_a, out_b = align_sequences(a, b)
        assert len(out_a) == 8

    def test_custom_target_len(self):
        a = _ramp(5)
        b = _ramp(8)
        out_a, out_b = align_sequences(a, b, target_len=10)
        assert len(out_a) == 10 and len(out_b) == 10

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            align_sequences(np.zeros((2, 3)), np.zeros(5))

    def test_target_lt_1_raises(self):
        with pytest.raises(ValueError):
            align_sequences(_ramp(3), _ramp(4), target_len=0)

    def test_endpoints_preserved(self):
        a = np.array([0.0, 1.0])
        b = np.array([0.0, 1.0])
        out_a, out_b = align_sequences(a, b, target_len=5)
        assert out_a[0] == pytest.approx(0.0)
        assert out_a[-1] == pytest.approx(1.0)


# ─── kendall_tau_distance ─────────────────────────────────────────────────────

class TestKendallTauDistanceExtra:
    def test_returns_int(self):
        a = np.array([0, 1, 2])
        b = np.array([0, 1, 2])
        assert isinstance(kendall_tau_distance(a, b), int)

    def test_identical_distance_zero(self):
        a = np.array([0, 1, 2, 3])
        assert kendall_tau_distance(a, a) == 0

    def test_reversed_max_distance(self):
        n = 4
        a = np.arange(n)
        b = np.arange(n)[::-1].copy()
        d = kendall_tau_distance(a, b)
        assert d == n * (n - 1) // 2

    def test_different_lengths_raises(self):
        with pytest.raises(ValueError):
            kendall_tau_distance(np.array([0, 1]), np.array([0, 1, 2]))

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            kendall_tau_distance(np.zeros((2, 3)), np.zeros(6))

    def test_nonneg(self):
        a = np.array([2, 0, 1])
        b = np.array([1, 2, 0])
        assert kendall_tau_distance(a, b) >= 0

    def test_single_element(self):
        assert kendall_tau_distance(np.array([0]), np.array([0])) == 0


# ─── longest_increasing ───────────────────────────────────────────────────────

class TestLongestIncreasingExtra:
    def test_returns_int(self):
        assert isinstance(longest_increasing(np.array([1.0, 2.0, 3.0])), int)

    def test_sorted_ascending(self):
        seq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert longest_increasing(seq) == 5

    def test_sorted_descending(self):
        seq = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert longest_increasing(seq) == 1

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            longest_increasing(np.zeros((2, 3)))

    def test_single_element(self):
        assert longest_increasing(np.array([7.0])) == 1

    def test_all_same(self):
        seq = np.array([3.0, 3.0, 3.0])
        assert longest_increasing(seq) == 1

    def test_mixed(self):
        seq = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        lis = longest_increasing(seq)
        assert lis >= 4  # e.g. 1, 4, 5, 9


# ─── segment_by_threshold ─────────────────────────────────────────────────────

class TestSegmentByThresholdExtra:
    def test_returns_list(self):
        seq = np.array([0.3, 0.7, 0.8, 0.2])
        result = segment_by_threshold(seq)
        assert isinstance(result, list)

    def test_no_segments_all_below(self):
        seq = np.zeros(5)
        cfg = SequenceConfig(threshold=0.5)
        result = segment_by_threshold(seq, cfg=cfg)
        assert result == []

    def test_all_above_one_segment(self):
        seq = np.ones(5)
        cfg = SequenceConfig(threshold=0.5)
        result = segment_by_threshold(seq, cfg=cfg)
        assert len(result) == 1
        start, end = result[0]
        assert start == 0 and end == 4

    def test_two_segments(self):
        seq = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
        cfg = SequenceConfig(threshold=0.5)
        result = segment_by_threshold(seq, cfg=cfg)
        assert len(result) == 2

    def test_none_cfg(self):
        seq = np.array([0.3, 0.7, 0.8, 0.2])
        result = segment_by_threshold(seq, cfg=None)
        assert isinstance(result, list)

    def test_elements_are_tuples(self):
        seq = np.array([1.0, 1.0, 0.0, 1.0])
        cfg = SequenceConfig(threshold=0.5)
        for item in segment_by_threshold(seq, cfg=cfg):
            assert isinstance(item, tuple) and len(item) == 2


# ─── batch_rank ───────────────────────────────────────────────────────────────

class TestBatchRankExtra:
    def test_returns_list(self):
        result = batch_rank([np.array([3.0, 1.0, 2.0])])
        assert isinstance(result, list)

    def test_length_matches(self):
        seqs = [_ramp(5), _ramp(8)]
        result = batch_rank(seqs)
        assert len(result) == 2

    def test_each_element_ndarray(self):
        for out in batch_rank([_ramp(5), _ramp(3)]):
            assert isinstance(out, np.ndarray)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            batch_rank([])

    def test_ranks_correct(self):
        seq = np.array([10.0, 20.0, 30.0])
        out = batch_rank([seq])[0]
        assert out[0] < out[1] < out[2]

    def test_lengths_preserved(self):
        seqs = [_ramp(4), _ramp(7)]
        results = batch_rank(seqs)
        assert len(results[0]) == 4
        assert len(results[1]) == 7
