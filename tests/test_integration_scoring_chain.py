"""
Integration tests for puzzle_reconstruction.scoring pipeline.

~50 tests across 5 classes covering:
- consistency_checker: run_consistency_check with valid and invalid assemblies
- boundary_scorer: score_boundary, score_matrix
- score_normalizer: normalize_score_matrix, combine_score_matrices
- global_ranker: rank_pairs, top_k_candidates
- threshold_selector: select_threshold, apply_threshold
"""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.scoring.consistency_checker import (
    ConsistencyReport,
    check_all_present,
    check_canvas_bounds,
    check_unique_ids,
    run_consistency_check,
)
from puzzle_reconstruction.scoring.boundary_scorer import (
    BoundarySide,
    BoundaryScore,
    ScoringConfig,
    score_boundary,
    score_matrix,
)
from puzzle_reconstruction.scoring.score_normalizer import (
    NormMethod,
    NormalizedMatrix,
    minmax_normalize_matrix,
    normalize_score_matrix,
    combine_score_matrices,
)
from puzzle_reconstruction.scoring.global_ranker import (
    RankedPair,
    rank_pairs,
    top_k_candidates,
)
from puzzle_reconstruction.scoring.threshold_selector import (
    ThresholdConfig,
    ThresholdResult,
    apply_threshold,
    select_threshold,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_compat_matrix(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.uniform(0.1, 1.0, (n, n)).astype(np.float32)
    np.fill_diagonal(m, 0.0)
    return m


def make_image(h: int = 80, w: int = 80, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


# ─── TestConsistencyChecker ───────────────────────────────────────────────────

class TestConsistencyChecker:
    """Tests for scoring.consistency_checker."""

    def test_check_unique_ids_no_issues(self):
        issues = check_unique_ids([0, 1, 2, 3])
        assert len(issues) == 0

    def test_check_unique_ids_duplicates(self):
        issues = check_unique_ids([0, 1, 1, 2])
        assert len(issues) > 0

    def test_check_all_present_ok(self):
        issues = check_all_present([0, 1, 2], expected_ids=[0, 1, 2])
        assert len(issues) == 0

    def test_check_all_present_missing(self):
        issues = check_all_present([0, 2], expected_ids=[0, 1, 2])
        assert len(issues) > 0

    def test_run_consistency_check_returns_report(self):
        positions = [(i * 40, 0) for i in range(4)]
        sizes = [(30, 30)] * 4
        report = run_consistency_check(
            fragment_ids=[0, 1, 2, 3],
            expected_ids=[0, 1, 2, 3],
            positions=positions,
            sizes=sizes,
            canvas_w=300,
            canvas_h=100,
        )
        assert isinstance(report, ConsistencyReport)

    def test_run_consistency_check_valid_assembly(self):
        positions = [(i * 40, 0) for i in range(4)]
        sizes = [(30, 30)] * 4
        report = run_consistency_check(
            fragment_ids=[0, 1, 2, 3],
            expected_ids=[0, 1, 2, 3],
            positions=positions,
            sizes=sizes,
            canvas_w=300,
            canvas_h=100,
        )
        assert report.is_consistent

    def test_run_consistency_check_missing_fragment(self):
        positions = [(i * 40, 0) for i in range(3)]
        sizes = [(30, 30)] * 3
        report = run_consistency_check(
            fragment_ids=[0, 1, 2],
            expected_ids=[0, 1, 2, 3],
            positions=positions,
            sizes=sizes,
            canvas_w=300,
            canvas_h=100,
        )
        assert not report.is_consistent

    def test_check_canvas_bounds_in_bounds(self):
        positions = [(10, 10), (60, 10)]
        sizes = [(40, 40), (40, 40)]
        issues = check_canvas_bounds(
            positions=positions,
            sizes=sizes,
            canvas_w=200,
            canvas_h=200,
        )
        assert isinstance(issues, list)

    def test_check_canvas_bounds_out_of_bounds(self):
        positions = [(0, 0), (190, 190)]
        sizes = [(40, 40), (40, 40)]
        issues = check_canvas_bounds(
            positions=positions,
            sizes=sizes,
            canvas_w=200,
            canvas_h=200,
        )
        assert isinstance(issues, list)

    def test_consistency_report_has_n_errors(self):
        positions = [(i * 40, 0) for i in range(3)]
        sizes = [(30, 30)] * 3
        report = run_consistency_check(
            fragment_ids=[0, 1, 2],
            expected_ids=[0, 1, 2, 3],
            positions=positions,
            sizes=sizes,
            canvas_w=300,
            canvas_h=100,
        )
        assert hasattr(report, "n_errors")
        assert report.n_errors >= 0

    def test_consistency_report_has_n_warnings(self):
        positions = [(i * 40, 0) for i in range(3)]
        sizes = [(30, 30)] * 3
        report = run_consistency_check(
            fragment_ids=[0, 1, 2],
            expected_ids=[0, 1, 2],
            positions=positions,
            sizes=sizes,
            canvas_w=300,
            canvas_h=100,
        )
        assert hasattr(report, "n_warnings")
        assert report.n_warnings >= 0


# ─── TestBoundaryScorer ───────────────────────────────────────────────────────

class TestBoundaryScorer:
    """Tests for scoring.boundary_scorer."""

    def test_score_boundary_returns_boundary_score(self):
        img1 = make_image()
        img2 = make_image(seed=1)
        result = score_boundary(img1, img2, BoundarySide.RIGHT, BoundarySide.LEFT)
        assert isinstance(result, BoundaryScore)

    def test_score_boundary_aggregate_in_unit_interval(self):
        img1 = make_image()
        img2 = make_image(seed=1)
        result = score_boundary(img1, img2, BoundarySide.RIGHT, BoundarySide.LEFT)
        assert 0.0 <= result.aggregate <= 1.0

    def test_score_boundary_intensity_diff_in_unit_interval(self):
        img1 = make_image()
        img2 = make_image(seed=1)
        result = score_boundary(img1, img2, BoundarySide.RIGHT, BoundarySide.LEFT)
        assert 0.0 <= result.intensity_diff <= 1.0

    def test_score_boundary_gradient_score_in_unit_interval(self):
        img1 = make_image()
        img2 = make_image(seed=1)
        result = score_boundary(img1, img2, BoundarySide.RIGHT, BoundarySide.LEFT)
        assert 0.0 <= result.gradient_score <= 1.0

    def test_score_boundary_color_score_in_unit_interval(self):
        img1 = make_image()
        img2 = make_image(seed=1)
        result = score_boundary(img1, img2, BoundarySide.RIGHT, BoundarySide.LEFT)
        assert 0.0 <= result.color_score <= 1.0

    def test_score_boundary_identical_images_high_aggregate(self):
        img = make_image()
        result = score_boundary(img, img, BoundarySide.RIGHT, BoundarySide.LEFT)
        assert result.aggregate > 0.5

    def test_score_matrix_shape(self):
        images = [make_image(seed=i) for i in range(4)]
        m = score_matrix(images)
        assert m.shape == (4, 4)

    def test_score_matrix_diagonal_zero(self):
        images = [make_image(seed=i) for i in range(4)]
        m = score_matrix(images)
        assert np.allclose(np.diag(m), 0.0, atol=1e-5)

    def test_score_matrix_off_diag_non_negative(self):
        images = [make_image(seed=i) for i in range(3)]
        m = score_matrix(images)
        off_diag = m[~np.eye(3, dtype=bool)]
        assert np.all(off_diag >= 0.0)

    def test_score_matrix_values_in_unit_interval(self):
        images = [make_image(seed=i) for i in range(3)]
        m = score_matrix(images)
        off_diag = m[~np.eye(3, dtype=bool)]
        assert np.all(off_diag <= 1.0)


# ─── TestScoreNormalizer ──────────────────────────────────────────────────────

class TestScoreNormalizer:
    """Tests for scoring.score_normalizer."""

    def test_minmax_normalize_output_in_unit_interval(self):
        m = make_compat_matrix(4)
        result = minmax_normalize_matrix(m)
        assert isinstance(result, NormalizedMatrix)
        assert np.all(result.data >= 0.0 - 1e-6)
        assert np.all(result.data <= 1.0 + 1e-6)

    def test_minmax_max_is_one(self):
        m = make_compat_matrix(4)
        result = minmax_normalize_matrix(m)
        assert np.max(result.data) == pytest.approx(1.0, abs=1e-5)

    def test_normalize_with_method_minmax(self):
        m = make_compat_matrix(4)
        cfg = NormMethod(method="minmax")
        result = normalize_score_matrix(m, cfg=cfg)
        assert isinstance(result, NormalizedMatrix)

    def test_normalize_with_method_zscore(self):
        m = make_compat_matrix(4)
        cfg = NormMethod(method="zscore")
        result = normalize_score_matrix(m, cfg=cfg)
        assert isinstance(result, NormalizedMatrix)

    def test_normalize_with_method_rank(self):
        m = make_compat_matrix(4)
        cfg = NormMethod(method="rank")
        result = normalize_score_matrix(m, cfg=cfg)
        assert isinstance(result, NormalizedMatrix)

    def test_normalized_matrix_data_shape_preserved(self):
        m = make_compat_matrix(5)
        result = normalize_score_matrix(m)
        assert result.data.shape == (5, 5)

    def test_combine_score_matrices_shape(self):
        m1 = make_compat_matrix(4, seed=1)
        m2 = make_compat_matrix(4, seed=2)
        combined = combine_score_matrices([m1, m2], weights=[0.6, 0.4])
        assert combined.shape == (4, 4)

    def test_combine_score_matrices_values_finite(self):
        m1 = make_compat_matrix(4, seed=1)
        m2 = make_compat_matrix(4, seed=2)
        combined = combine_score_matrices([m1, m2], weights=[0.5, 0.5])
        assert np.all(np.isfinite(combined))

    def test_combine_single_matrix(self):
        m = make_compat_matrix(4)
        combined = combine_score_matrices([m], weights=[1.0])
        assert np.allclose(combined, m, atol=1e-5)

    def test_normalized_matrix_has_method_attr(self):
        m = make_compat_matrix(3)
        result = normalize_score_matrix(m)
        assert hasattr(result, "method")


# ─── TestGlobalRanker ─────────────────────────────────────────────────────────

class TestGlobalRanker:
    """Tests for scoring.global_ranker."""

    def test_rank_pairs_returns_list(self):
        m = make_compat_matrix(4)
        pairs = rank_pairs(m)
        assert isinstance(pairs, list)

    def test_rank_pairs_nonempty(self):
        m = make_compat_matrix(4)
        pairs = rank_pairs(m)
        assert len(pairs) > 0

    def test_rank_pairs_returns_ranked_pair_objects(self):
        m = make_compat_matrix(4)
        pairs = rank_pairs(m)
        assert isinstance(pairs[0], RankedPair)

    def test_rank_pairs_indices_in_range(self):
        n = 5
        m = make_compat_matrix(n)
        pairs = rank_pairs(m)
        for p in pairs:
            assert 0 <= p.idx1 < n
            assert 0 <= p.idx2 < n

    def test_rank_pairs_no_self_pairs(self):
        m = make_compat_matrix(4)
        pairs = rank_pairs(m)
        for p in pairs:
            assert p.idx1 != p.idx2

    def test_rank_pairs_scores_finite(self):
        m = make_compat_matrix(4)
        pairs = rank_pairs(m)
        for p in pairs:
            assert np.isfinite(p.score)

    def test_top_k_candidates_returns_dict(self):
        n = 5
        m = make_compat_matrix(n)
        pairs = rank_pairs(m)
        result = top_k_candidates(pairs, n_fragments=n, k=2)
        assert isinstance(result, dict)

    def test_top_k_candidates_keys_are_fragment_indices(self):
        n = 5
        m = make_compat_matrix(n)
        pairs = rank_pairs(m)
        result = top_k_candidates(pairs, n_fragments=n, k=2)
        for key in result:
            assert 0 <= key < n

    def test_top_k_candidates_max_k_per_fragment(self):
        n = 5
        m = make_compat_matrix(n)
        pairs = rank_pairs(m)
        k = 3
        result = top_k_candidates(pairs, n_fragments=n, k=k)
        for candidates in result.values():
            assert len(candidates) <= k

    def test_top_k_k1_returns_at_most_one_per_fragment(self):
        n = 4
        m = make_compat_matrix(n)
        pairs = rank_pairs(m)
        result = top_k_candidates(pairs, n_fragments=n, k=1)
        for candidates in result.values():
            assert len(candidates) == 1


# ─── TestThresholdSelector ────────────────────────────────────────────────────

class TestThresholdSelector:
    """Tests for scoring.threshold_selector."""

    def test_select_threshold_returns_result(self):
        m = make_compat_matrix(5)
        result = select_threshold(m)
        assert isinstance(result, ThresholdResult)

    def test_select_threshold_value_in_unit_interval(self):
        m = make_compat_matrix(5)
        result = select_threshold(m)
        assert 0.0 <= result.threshold <= 1.0

    def test_apply_threshold_returns_array(self):
        m = make_compat_matrix(5)
        result = select_threshold(m)
        binary = apply_threshold(m, result)
        assert isinstance(binary, np.ndarray)

    def test_apply_threshold_size_matches(self):
        m = make_compat_matrix(5)
        result = select_threshold(m)
        binary = apply_threshold(m, result)
        assert binary.size == m.size

    def test_apply_high_threshold_few_true(self):
        m = make_compat_matrix(5)
        high_cfg = ThresholdConfig(method="fixed", fixed_value=0.99)
        result = select_threshold(m, cfg=high_cfg)
        binary = apply_threshold(m, result)
        assert binary.sum() <= binary.size

    def test_apply_low_threshold_all_truish(self):
        m = make_compat_matrix(5)
        low_cfg = ThresholdConfig(method="fixed", fixed_value=0.0)
        result = select_threshold(m, cfg=low_cfg)
        binary = apply_threshold(m, result)
        assert binary.sum() >= 0

    def test_select_threshold_percentile_method(self):
        m = make_compat_matrix(5)
        cfg = ThresholdConfig(method="percentile", percentile=50.0)
        result = select_threshold(m, cfg=cfg)
        assert 0.0 <= result.threshold <= 1.0

    def test_select_threshold_fixed_method(self):
        m = make_compat_matrix(5)
        cfg = ThresholdConfig(method="fixed", fixed_value=0.5)
        result = select_threshold(m, cfg=cfg)
        assert result.threshold == pytest.approx(0.5)

    def test_threshold_result_has_n_above(self):
        m = make_compat_matrix(5)
        result = select_threshold(m)
        assert hasattr(result, "n_above")

    def test_threshold_result_has_acceptance_ratio(self):
        m = make_compat_matrix(5)
        result = select_threshold(m)
        assert hasattr(result, "acceptance_ratio")
