"""
Property-based tests for:
  - puzzle_reconstruction.utils.score_matrix_utils
  - puzzle_reconstruction.utils.shape_match_utils
  - puzzle_reconstruction.utils.score_norm_utils

Verifies mathematical invariants:
- score_matrix_utils:
    zero_diagonal:        diagonal is zero, off-diagonal unchanged
    symmetrize:           result equals its transpose; idempotent
    normalize_rows:       non-zero rows sum to 1.0; non-negative
    threshold_matrix:     entries <= threshold become zero; others unchanged
    top_k_indices:        length <= k; all in valid range; descending order
    matrix_stats:         sparsity in [0,1]; max >= mean >= min (non-zero vals)
    intra_fragment_mask:  symmetric; diagonal blocks True; off-blocks False
    apply_intra_fragment_mask: masked entries are zero; total size preserved
    top_k_per_row:        each row <= k entries; entries sorted descending
- shape_match_utils:
    entries_from_results: len preserved; rank sequential
    summarise_matches:    n_total = len(entries); n_good + n_poor = n_total
    filter_good_matches:  subset; every entry.is_good
    filter_poor_matches:  subset; disjoint from filter_good; union = original
    top_k_match_entries:  len <= k; descending score order
    filter_match_by_score_range: all scores in [lo, hi]
    batch_summarise_matches: same length as input
- score_norm_utils:
    entries_from_scores:  len preserved; method tag correct
    summarise_norm:       n_total = len(entries); min <= max
    filter_by_normalized_range: all normalized_scores in [lo, hi]
    filter_by_original_range:   all original_scores in [lo, hi]
    top_k_norm_entries:   len <= k; descending normalized_score
    batch_summarise_norm: same length as input
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest

from puzzle_reconstruction.utils.score_matrix_utils import (
    ScoreMatrixConfig,
    zero_diagonal,
    symmetrize,
    normalize_rows,
    threshold_matrix,
    top_k_indices,
    matrix_stats,
    intra_fragment_mask,
    apply_intra_fragment_mask,
    top_k_per_row,
)
from puzzle_reconstruction.utils.shape_match_utils import (
    ShapeMatchConfig,
    ShapeMatchEntry,
    make_match_entry,
    entries_from_results,
    summarise_matches,
    filter_good_matches,
    filter_poor_matches,
    top_k_match_entries,
    filter_match_by_score_range,
    batch_summarise_matches,
    match_entry_stats,
)
from puzzle_reconstruction.utils.score_norm_utils import (
    ScoreNormConfig,
    make_norm_entry,
    entries_from_scores,
    summarise_norm,
    filter_by_normalized_range,
    filter_by_original_range,
    top_k_norm_entries,
    norm_entry_stats,
    batch_summarise_norm,
)

RNG = np.random.default_rng(2026)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rand_matrix(n: int = 5, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    return RNG.uniform(lo, hi, size=(n, n)).astype(float)


def _rand_scores(n: int = 10, lo: float = 0.0, hi: float = 1.0) -> List[float]:
    return RNG.uniform(lo, hi, size=n).tolist()


def _rand_entries(n: int = 10) -> List[ShapeMatchEntry]:
    scores = _rand_scores(n)
    return [make_match_entry(idx1=i, idx2=i + 1, score=s, rank=i)
            for i, s in enumerate(scores)]


# ═══════════════════════════════════════════════════════════════════════════════
# score_matrix_utils
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreMatrixConfig:
    def test_default_valid(self):
        cfg = ScoreMatrixConfig()
        assert 0.0 <= cfg.threshold <= 1.0
        assert cfg.top_k >= 1
        assert cfg.eps > 0.0

    def test_custom_valid(self):
        cfg = ScoreMatrixConfig(threshold=0.3, top_k=5, symmetrize=False)
        assert cfg.threshold == pytest.approx(0.3)
        assert cfg.top_k == 5

    @pytest.mark.parametrize("t", [-0.1, 1.1, 2.0])
    def test_invalid_threshold(self, t):
        with pytest.raises(ValueError):
            ScoreMatrixConfig(threshold=t)

    @pytest.mark.parametrize("k", [0, -1])
    def test_invalid_top_k(self, k):
        with pytest.raises(ValueError):
            ScoreMatrixConfig(top_k=k)


class TestZeroDiagonal:
    def test_diagonal_is_zero(self):
        for n in [2, 4, 6]:
            m = _rand_matrix(n)
            result = zero_diagonal(m)
            np.testing.assert_array_equal(np.diag(result), np.zeros(n))

    def test_off_diagonal_unchanged(self):
        for n in [3, 5]:
            m = _rand_matrix(n)
            result = zero_diagonal(m)
            off_mask = ~np.eye(n, dtype=bool)
            np.testing.assert_array_almost_equal(result[off_mask], m[off_mask])

    def test_input_not_mutated(self):
        m = _rand_matrix(4)
        m_copy = m.copy()
        zero_diagonal(m)
        np.testing.assert_array_equal(m, m_copy)

    def test_already_zero_diagonal(self):
        m = _rand_matrix(4)
        np.fill_diagonal(m, 0.0)
        result = zero_diagonal(m)
        np.testing.assert_array_almost_equal(result, m)

    def test_shape_preserved(self):
        for n in [2, 5, 8]:
            m = _rand_matrix(n)
            assert zero_diagonal(m).shape == (n, n)


class TestSymmetrize:
    def test_result_is_symmetric(self):
        for n in [2, 4, 6]:
            m = _rand_matrix(n)
            s = symmetrize(m)
            np.testing.assert_array_almost_equal(s, s.T)

    def test_idempotent(self):
        for n in [3, 5]:
            m = _rand_matrix(n)
            s1 = symmetrize(m)
            s2 = symmetrize(s1)
            np.testing.assert_array_almost_equal(s1, s2)

    def test_already_symmetric_unchanged(self):
        m = _rand_matrix(4)
        s_init = (m + m.T) / 2.0
        result = symmetrize(s_init)
        np.testing.assert_array_almost_equal(result, s_init)

    def test_diagonal_preserved(self):
        m = _rand_matrix(4)
        s = symmetrize(m)
        np.testing.assert_array_almost_equal(np.diag(s), np.diag(m))

    def test_shape_preserved(self):
        for n in [3, 7]:
            assert symmetrize(_rand_matrix(n)).shape == (n, n)


class TestNormalizeRows:
    def test_nonzero_rows_sum_to_one(self):
        for _ in range(20):
            m = _rand_matrix(5, lo=0.1, hi=1.0)
            r = normalize_rows(m)
            row_sums = r.sum(axis=1)
            np.testing.assert_array_almost_equal(row_sums, np.ones(5))

    def test_nonnegative(self):
        for _ in range(20):
            m = _rand_matrix(4, lo=0.0, hi=1.0)
            r = normalize_rows(m)
            assert (r >= -1e-12).all()

    def test_zero_row_stays_zero(self):
        m = np.ones((4, 4), dtype=float)
        m[2, :] = 0.0
        r = normalize_rows(m)
        np.testing.assert_array_almost_equal(r[2, :], np.zeros(4))

    def test_shape_preserved(self):
        for n in [3, 6]:
            m = _rand_matrix(n, lo=0.1, hi=1.0)
            assert normalize_rows(m).shape == (n, n)

    def test_uniform_row_normalized(self):
        m = np.ones((3, 4), dtype=float)
        r = normalize_rows(m)
        expected = np.full((3, 4), 0.25)
        np.testing.assert_array_almost_equal(r, expected)


class TestThresholdMatrix:
    def test_entries_le_threshold_are_zero(self):
        for _ in range(20):
            m = _rand_matrix(5)
            t = float(RNG.uniform(0.2, 0.8))
            result = threshold_matrix(m, t)
            assert (result[m <= t] == 0.0).all()

    def test_entries_above_threshold_unchanged(self):
        for _ in range(20):
            m = _rand_matrix(5)
            t = 0.3
            result = threshold_matrix(m, t)
            mask = m > t
            np.testing.assert_array_almost_equal(result[mask], m[mask])

    def test_zero_threshold_keeps_positives(self):
        m = _rand_matrix(4, lo=0.1, hi=1.0)
        result = threshold_matrix(m, 0.0)
        assert (result > 0.0).all()

    def test_one_threshold_zeros_all(self):
        m = _rand_matrix(4, lo=0.0, hi=0.9)
        result = threshold_matrix(m, 1.0)
        np.testing.assert_array_equal(result, np.zeros_like(m))

    def test_input_not_mutated(self):
        m = _rand_matrix(4)
        m_copy = m.copy()
        threshold_matrix(m, 0.5)
        np.testing.assert_array_equal(m, m_copy)


class TestTopKIndices:
    def test_length_le_k(self):
        for n in [5, 10, 20]:
            row = RNG.uniform(0, 1, size=n)
            for k in [1, 3, 5, n + 5]:
                result = top_k_indices(row, k)
                assert len(result) <= k

    def test_length_exactly_min_k_n(self):
        for n in [5, 10]:
            row = RNG.uniform(0, 1, size=n)
            for k in [1, 3, n + 2]:
                result = top_k_indices(row, k)
                assert len(result) == min(k, n)

    def test_indices_in_valid_range(self):
        for n in [5, 10]:
            row = RNG.uniform(0, 1, size=n)
            result = top_k_indices(row, 3)
            assert (result >= 0).all()
            assert (result < n).all()

    def test_descending_values(self):
        row = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
        result = top_k_indices(row, 5)
        values = row[result]
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    def test_zero_k_returns_empty(self):
        row = RNG.uniform(0, 1, size=5)
        result = top_k_indices(row, 0)
        assert len(result) == 0

    def test_top_1_is_argmax(self):
        row = np.array([0.3, 0.9, 0.1, 0.5, 0.7])
        result = top_k_indices(row, 1)
        assert result[0] == int(np.argmax(row))


class TestMatrixStats:
    def test_sparsity_in_range(self):
        for n in [3, 5]:
            m = _rand_matrix(n)
            stats = matrix_stats(m)
            assert 0.0 <= stats.sparsity <= 1.0

    def test_max_ge_mean_ge_min(self):
        for _ in range(20):
            m = _rand_matrix(5, lo=0.1, hi=1.0)
            s = matrix_stats(m)
            if s.n_nonzero > 0:
                assert s.max_score >= s.mean_score - 1e-12
                assert s.mean_score >= s.min_score - 1e-12

    def test_n_edges_equals_n(self):
        for n in [3, 5, 7]:
            m = _rand_matrix(n)
            assert matrix_stats(m).n_edges == n

    def test_all_zero_matrix(self):
        m = np.zeros((4, 4))
        s = matrix_stats(m)
        assert s.n_nonzero == 0
        assert s.sparsity == pytest.approx(1.0)

    def test_top_pair_in_range(self):
        for n in [4, 6]:
            m = _rand_matrix(n)
            s = matrix_stats(m)
            r, c = s.top_pair
            assert 0 <= r < n
            assert 0 <= c < n

    def test_to_dict_keys(self):
        s = matrix_stats(_rand_matrix(4))
        d = s.to_dict()
        for key in ("n_edges", "n_nonzero", "mean_score", "max_score",
                    "min_score", "sparsity", "top_pair"):
            assert key in d


class TestIntraFragmentMask:
    def test_symmetric(self):
        for counts in [[2, 3], [1, 2, 4], [3, 3, 3]]:
            mask = intra_fragment_mask(counts)
            np.testing.assert_array_equal(mask, mask.T)

    def test_diagonal_blocks_true(self):
        counts = [2, 3]
        mask = intra_fragment_mask(counts)
        # First block [0:2, 0:2]
        assert mask[0, 0] and mask[0, 1] and mask[1, 0] and mask[1, 1]
        # Second block [2:5, 2:5]
        for i in range(2, 5):
            for j in range(2, 5):
                assert mask[i, j]

    def test_off_blocks_false(self):
        counts = [2, 3]
        mask = intra_fragment_mask(counts)
        # Off-block region
        assert not mask[0, 2]
        assert not mask[2, 0]
        assert not mask[1, 3]
        assert not mask[4, 1]

    def test_size_correct(self):
        for counts in [[1, 2, 3], [4, 4], [5]]:
            mask = intra_fragment_mask(counts)
            n = sum(counts)
            assert mask.shape == (n, n)

    def test_single_fragment(self):
        mask = intra_fragment_mask([5])
        assert mask.all()


class TestApplyIntraFragmentMask:
    def test_masked_entries_zero(self):
        counts = [2, 3]
        m = _rand_matrix(5, lo=0.1, hi=1.0)
        result = apply_intra_fragment_mask(m, counts)
        frag_mask = intra_fragment_mask(counts)
        np.testing.assert_array_equal(result[frag_mask], np.zeros(frag_mask.sum()))

    def test_off_block_entries_unchanged(self):
        counts = [2, 3]
        m = _rand_matrix(5, lo=0.1, hi=1.0)
        result = apply_intra_fragment_mask(m, counts)
        frag_mask = intra_fragment_mask(counts)
        off = ~frag_mask
        np.testing.assert_array_almost_equal(result[off], m[off])

    def test_input_not_mutated(self):
        counts = [2, 2]
        m = _rand_matrix(4)
        m_copy = m.copy()
        apply_intra_fragment_mask(m, counts)
        np.testing.assert_array_equal(m, m_copy)


class TestTopKPerRow:
    def test_each_row_le_k(self):
        m = _rand_matrix(5, lo=0.1, hi=1.0)
        result = top_k_per_row(m, k=3)
        for entries in result:
            assert len(entries) <= 3

    def test_n_rows_correct(self):
        for n in [3, 5, 7]:
            m = _rand_matrix(n, lo=0.1, hi=1.0)
            result = top_k_per_row(m, k=2)
            assert len(result) == n

    def test_self_excluded(self):
        m = _rand_matrix(5, lo=0.1, hi=1.0)
        result = top_k_per_row(m, k=5, exclude_self=True)
        for i, entries in enumerate(result):
            assert all(e.idx != i for e in entries)

    def test_descending_order_per_row(self):
        m = _rand_matrix(4, lo=0.1, hi=1.0)
        result = top_k_per_row(m, k=4)
        for entries in result:
            scores = [e.score for e in entries]
            assert scores == sorted(scores, reverse=True)

    def test_all_zero_row_empty(self):
        m = np.zeros((3, 3))
        result = top_k_per_row(m, k=2)
        for entries in result:
            assert entries == []


# ═══════════════════════════════════════════════════════════════════════════════
# shape_match_utils
# ═══════════════════════════════════════════════════════════════════════════════


class TestShapeMatchConfig:
    def test_default_valid(self):
        cfg = ShapeMatchConfig()
        assert cfg.min_score >= 0.0
        assert cfg.max_pairs >= 1
        assert cfg.method in {"hu", "zernike", "combined"}

    @pytest.mark.parametrize("s", [-0.1, -1.0])
    def test_invalid_min_score(self, s):
        with pytest.raises(ValueError):
            ShapeMatchConfig(min_score=s)

    @pytest.mark.parametrize("p", [0, -1])
    def test_invalid_max_pairs(self, p):
        with pytest.raises(ValueError):
            ShapeMatchConfig(max_pairs=p)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            ShapeMatchConfig(method="unknown")


class TestShapeMatchEntry:
    def test_is_good_threshold(self):
        e_good = make_match_entry(0, 1, 0.9)
        e_poor = make_match_entry(0, 1, 0.3)
        e_border = make_match_entry(0, 1, 0.5)
        assert e_good.is_good
        assert not e_poor.is_good
        assert not e_border.is_good  # > 0.5 required

    def test_negative_idx_raises(self):
        with pytest.raises(ValueError):
            ShapeMatchEntry(idx1=-1, idx2=0, score=0.5)

    def test_zero_idx_valid(self):
        e = make_match_entry(0, 0, 0.5)
        assert e.idx1 == 0


class TestEntriesFromResults:
    def test_length_preserved(self):
        results = [(i, i + 1, float(i) / 10) for i in range(10)]
        entries = entries_from_results(results)
        assert len(entries) == 10

    def test_rank_sequential(self):
        results = [(0, 1, 0.8), (1, 2, 0.6), (2, 3, 0.4)]
        entries = entries_from_results(results)
        for i, e in enumerate(entries):
            assert e.rank == i

    def test_empty_input(self):
        entries = entries_from_results([])
        assert entries == []

    def test_scores_preserved(self):
        results = [(0, 1, 0.7), (1, 2, 0.3)]
        entries = entries_from_results(results)
        assert entries[0].score == pytest.approx(0.7)
        assert entries[1].score == pytest.approx(0.3)


class TestSummariseMatches:
    def test_n_total_equals_len(self):
        for n in [0, 1, 5, 10]:
            entries = _rand_entries(n)
            s = summarise_matches(entries)
            assert s.n_total == n

    def test_n_good_plus_n_poor_equals_n_total(self):
        for _ in range(20):
            entries = _rand_entries(10)
            s = summarise_matches(entries)
            assert s.n_good + s.n_poor == s.n_total

    def test_empty_entries(self):
        s = summarise_matches([])
        assert s.n_total == 0
        assert s.n_good == 0
        assert s.mean_score == 0.0

    def test_max_ge_mean_ge_min(self):
        for _ in range(20):
            entries = _rand_entries(10)
            s = summarise_matches(entries)
            if s.n_total > 0:
                assert s.max_score >= s.mean_score - 1e-12
                assert s.mean_score >= s.min_score - 1e-12

    def test_all_good_entries(self):
        entries = [make_match_entry(i, i + 1, 0.9) for i in range(5)]
        s = summarise_matches(entries)
        assert s.n_good == 5
        assert s.n_poor == 0

    def test_all_poor_entries(self):
        entries = [make_match_entry(i, i + 1, 0.1) for i in range(5)]
        s = summarise_matches(entries)
        assert s.n_good == 0
        assert s.n_poor == 5


class TestFilterGoodMatches:
    def test_all_good(self):
        entries = _rand_entries(15)
        good = filter_good_matches(entries)
        assert all(e.is_good for e in good)

    def test_subset_of_original(self):
        entries = _rand_entries(15)
        good = filter_good_matches(entries)
        assert set(id(e) for e in good).issubset(set(id(e) for e in entries))

    def test_empty_input(self):
        assert filter_good_matches([]) == []

    def test_no_good_matches(self):
        entries = [make_match_entry(i, i + 1, 0.1) for i in range(5)]
        assert filter_good_matches(entries) == []


class TestFilterPoorMatches:
    def test_all_poor(self):
        entries = _rand_entries(15)
        poor = filter_poor_matches(entries)
        assert all(not e.is_good for e in poor)

    def test_disjoint_from_good(self):
        entries = _rand_entries(15)
        good_ids = {id(e) for e in filter_good_matches(entries)}
        poor_ids = {id(e) for e in filter_poor_matches(entries)}
        assert good_ids.isdisjoint(poor_ids)

    def test_union_equals_original(self):
        entries = _rand_entries(15)
        good = filter_good_matches(entries)
        poor = filter_poor_matches(entries)
        combined_ids = {id(e) for e in good} | {id(e) for e in poor}
        original_ids = {id(e) for e in entries}
        assert combined_ids == original_ids

    def test_empty_input(self):
        assert filter_poor_matches([]) == []


class TestTopKMatchEntries:
    def test_length_le_k(self):
        entries = _rand_entries(15)
        for k in [1, 3, 5, 20]:
            result = top_k_match_entries(entries, k)
            assert len(result) <= k

    def test_descending_score(self):
        entries = _rand_entries(10)
        result = top_k_match_entries(entries, 10)
        scores = [e.score for e in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_input(self):
        result = top_k_match_entries([], 5)
        assert result == []

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            top_k_match_entries(_rand_entries(5), 0)

    def test_k_larger_than_entries(self):
        entries = _rand_entries(3)
        result = top_k_match_entries(entries, 100)
        assert len(result) == 3


class TestFilterMatchByScoreRange:
    def test_all_in_range(self):
        entries = _rand_entries(15)
        for lo, hi in [(0.0, 0.5), (0.3, 0.7), (0.5, 1.0)]:
            result = filter_match_by_score_range(entries, lo, hi)
            for e in result:
                assert lo <= e.score <= hi

    def test_subset_of_original(self):
        entries = _rand_entries(15)
        result = filter_match_by_score_range(entries, 0.2, 0.8)
        ids = {id(e) for e in result}
        assert ids.issubset({id(e) for e in entries})

    def test_full_range_keeps_all(self):
        entries = [make_match_entry(i, i + 1, float(i) / 10) for i in range(10)]
        result = filter_match_by_score_range(entries, 0.0, 1.0)
        assert len(result) == len(entries)

    def test_impossible_range_empty(self):
        entries = [make_match_entry(i, i + 1, 0.5) for i in range(5)]
        result = filter_match_by_score_range(entries, 0.8, 1.0)
        assert result == []


class TestBatchSummariseMatches:
    def test_same_length(self):
        groups = [_rand_entries(n) for n in [5, 10, 3]]
        result = batch_summarise_matches(groups)
        assert len(result) == 3

    def test_each_matches_individual(self):
        groups = [_rand_entries(n) for n in [5, 8]]
        result = batch_summarise_matches(groups)
        for i, group in enumerate(groups):
            expected = summarise_matches(group)
            assert result[i].n_total == expected.n_total


class TestMatchEntryStats:
    def test_empty_returns_zeros(self):
        stats = match_entry_stats([])
        assert stats["n"] == 0
        assert stats["mean_score"] == 0.0

    def test_n_correct(self):
        entries = _rand_entries(7)
        stats = match_entry_stats(entries)
        assert stats["n"] == 7

    def test_mean_in_range(self):
        entries = [make_match_entry(i, i + 1, float(i) / 10) for i in range(1, 6)]
        stats = match_entry_stats(entries)
        scores = [e.score for e in entries]
        assert stats["mean_score"] == pytest.approx(sum(scores) / len(scores))


# ═══════════════════════════════════════════════════════════════════════════════
# score_norm_utils
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreNormConfig:
    def test_default_valid(self):
        cfg = ScoreNormConfig()
        assert cfg.method in {"minmax", "zscore", "rank", "calibrated"}
        lo, hi = cfg.feature_range
        assert lo < hi

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            ScoreNormConfig(method="unknown")

    @pytest.mark.parametrize("lo, hi", [(1.0, 0.0), (0.5, 0.5)])
    def test_invalid_feature_range(self, lo, hi):
        with pytest.raises(ValueError):
            ScoreNormConfig(feature_range=(lo, hi))

    def test_custom_valid(self):
        cfg = ScoreNormConfig(method="zscore", feature_range=(-1.0, 1.0))
        assert cfg.method == "zscore"


class TestMakeNormEntry:
    def test_delta_property(self):
        e = make_norm_entry(0, original_score=0.3, normalized_score=0.7)
        assert e.delta == pytest.approx(0.4)

    def test_negative_idx_raises(self):
        from puzzle_reconstruction.utils.score_norm_utils import ScoreNormEntry
        with pytest.raises(ValueError):
            ScoreNormEntry(idx=-1, original_score=0.5, normalized_score=0.5)

    def test_method_stored(self):
        e = make_norm_entry(0, 0.5, 0.8, method="zscore")
        assert e.method == "zscore"


class TestEntriesFromScores:
    def test_length_preserved(self):
        orig = _rand_scores(10)
        norm = _rand_scores(10)
        entries = entries_from_scores(orig, norm)
        assert len(entries) == 10

    def test_scores_correct(self):
        orig = [0.1, 0.5, 0.9]
        norm = [0.2, 0.6, 0.8]
        entries = entries_from_scores(orig, norm)
        for i, e in enumerate(entries):
            assert e.original_score == pytest.approx(orig[i])
            assert e.normalized_score == pytest.approx(norm[i])

    def test_empty_input(self):
        entries = entries_from_scores([], [])
        assert entries == []

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            entries_from_scores([0.1, 0.2], [0.3])

    def test_idx_sequential(self):
        orig = _rand_scores(5)
        norm = _rand_scores(5)
        entries = entries_from_scores(orig, norm)
        for i, e in enumerate(entries):
            assert e.idx == i


class TestSummariseNorm:
    def test_n_total_equals_len(self):
        for n in [0, 1, 5, 10]:
            orig = _rand_scores(n)
            norm = _rand_scores(n)
            entries = entries_from_scores(orig, norm)
            s = summarise_norm(entries)
            assert s.n_total == n

    def test_orig_min_le_max(self):
        for _ in range(20):
            orig = _rand_scores(10)
            norm = _rand_scores(10)
            s = summarise_norm(entries_from_scores(orig, norm))
            if s.n_total > 0:
                assert s.original_min <= s.original_max

    def test_norm_min_le_max(self):
        for _ in range(20):
            orig = _rand_scores(10)
            norm = _rand_scores(10)
            s = summarise_norm(entries_from_scores(orig, norm))
            if s.n_total > 0:
                assert s.normalized_min <= s.normalized_max

    def test_empty_entries(self):
        s = summarise_norm([])
        assert s.n_total == 0

    def test_single_entry(self):
        entries = entries_from_scores([0.4], [0.7])
        s = summarise_norm(entries)
        assert s.n_total == 1
        assert s.original_min == pytest.approx(0.4)
        assert s.original_max == pytest.approx(0.4)


class TestFilterByNormalizedRange:
    def test_all_in_range(self):
        orig = _rand_scores(20)
        norm = _rand_scores(20)
        entries = entries_from_scores(orig, norm)
        for lo, hi in [(0.0, 0.5), (0.3, 0.8)]:
            result = filter_by_normalized_range(entries, lo, hi)
            for e in result:
                assert lo <= e.normalized_score <= hi

    def test_subset_of_original(self):
        orig = _rand_scores(20)
        norm = _rand_scores(20)
        entries = entries_from_scores(orig, norm)
        result = filter_by_normalized_range(entries, 0.2, 0.7)
        assert set(id(e) for e in result).issubset(set(id(e) for e in entries))


class TestFilterByOriginalRange:
    def test_all_in_range(self):
        orig = _rand_scores(20)
        norm = _rand_scores(20)
        entries = entries_from_scores(orig, norm)
        result = filter_by_original_range(entries, 0.2, 0.8)
        for e in result:
            assert 0.2 <= e.original_score <= 0.8


class TestTopKNormEntries:
    def test_length_le_k(self):
        orig = _rand_scores(15)
        norm = _rand_scores(15)
        entries = entries_from_scores(orig, norm)
        for k in [1, 5, 20]:
            result = top_k_norm_entries(entries, k)
            assert len(result) <= k

    def test_descending_order(self):
        orig = _rand_scores(10)
        norm = _rand_scores(10)
        entries = entries_from_scores(orig, norm)
        result = top_k_norm_entries(entries, 10)
        scores = [e.normalized_score for e in result]
        assert scores == sorted(scores, reverse=True)

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            top_k_norm_entries([], 0)

    def test_empty_input(self):
        result = top_k_norm_entries([], 5)
        assert result == []


class TestNormEntryStats:
    def test_empty_returns_zeros(self):
        stats = norm_entry_stats([])
        assert stats["n"] == 0
        assert stats["mean_original"] == 0.0

    def test_n_correct(self):
        orig = _rand_scores(8)
        norm = _rand_scores(8)
        stats = norm_entry_stats(entries_from_scores(orig, norm))
        assert stats["n"] == 8

    def test_mean_original_correct(self):
        orig = [0.2, 0.4, 0.6]
        norm = [0.3, 0.5, 0.7]
        entries = entries_from_scores(orig, norm)
        stats = norm_entry_stats(entries)
        assert stats["mean_original"] == pytest.approx(sum(orig) / len(orig))

    def test_mean_delta_correct(self):
        orig = [0.2, 0.4]
        norm = [0.5, 0.6]
        entries = entries_from_scores(orig, norm)
        stats = norm_entry_stats(entries)
        expected_delta = ((0.5 - 0.2) + (0.6 - 0.4)) / 2
        assert stats["mean_delta"] == pytest.approx(expected_delta)


class TestBatchSummariseNorm:
    def test_same_length(self):
        score_lists = [
            (_rand_scores(n), _rand_scores(n))
            for n in [3, 5, 8]
        ]
        result = batch_summarise_norm(score_lists)
        assert len(result) == 3

    def test_each_matches_individual(self):
        score_lists = [(_rand_scores(5), _rand_scores(5)) for _ in range(3)]
        result = batch_summarise_norm(score_lists)
        for i, (orig, norm) in enumerate(score_lists):
            expected = summarise_norm(entries_from_scores(orig, norm))
            assert result[i].n_total == expected.n_total
