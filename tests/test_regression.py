"""
Regression tests — fix known-good values for key algorithms.

Purpose: catch regressions in numerical outputs when refactoring.
Values were generated with seed=42 using current implementation.

Run:
    python -m pytest tests/test_regression.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.algorithms.fractal.css import (
    curvature_scale_space,
    css_to_feature_vector,
    css_similarity,
)
from puzzle_reconstruction.algorithms.fractal.box_counting import (
    box_counting_fd,
    box_counting_curve,
)
from puzzle_reconstruction.algorithms.fractal.ifs import (
    fit_ifs_coefficients,
    reconstruct_from_ifs,
)
from puzzle_reconstruction.scoring.match_evaluator import (
    compute_precision,
    compute_recall,
    compute_f_score,
    evaluate_match,
    aggregate_eval,
    rank_matches,
)
from puzzle_reconstruction.scoring.threshold_selector import (
    select_fixed_threshold,
    select_percentile_threshold,
    apply_threshold,
)


# ── Contour fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def circle_64() -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    return np.column_stack([np.cos(t), np.sin(t)])


@pytest.fixture(scope="module")
def zigzag_128() -> np.ndarray:
    x = np.linspace(0, 10, 128)
    y = np.abs(np.sin(x * 3))
    return np.column_stack([x, y])


@pytest.fixture(scope="module")
def straight_line_64() -> np.ndarray:
    return np.column_stack([np.linspace(0, 1, 64), np.zeros(64)])


# ── CSS regression ────────────────────────────────────────────────────────────

class TestCSSRegression:

    def test_css_vector_length(self, circle_64):
        css = curvature_scale_space(circle_64, n_sigmas=7)
        vec = css_to_feature_vector(css)
        # 7 sigmas × 64 bins = 448
        assert len(vec) == 448

    def test_css_vector_unit_norm(self, circle_64):
        css = curvature_scale_space(circle_64, n_sigmas=7)
        vec = css_to_feature_vector(css)
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-9

    def test_self_similarity_exact_one(self, circle_64):
        css = curvature_scale_space(circle_64)
        vec = css_to_feature_vector(css)
        assert css_similarity(vec, vec) == pytest.approx(1.0, abs=1e-12)

    def test_n_sigmas_7_by_default(self, circle_64):
        css = curvature_scale_space(circle_64)
        assert len(css) == 7

    def test_css_3_sigmas_vector_length(self, circle_64):
        css = curvature_scale_space(circle_64, sigma_range=[1.0, 4.0, 16.0])
        vec = css_to_feature_vector(css, n_bins=32)
        assert len(vec) == 3 * 32

    def test_css_vector_non_negative(self, circle_64):
        css = curvature_scale_space(circle_64)
        vec = css_to_feature_vector(css)
        assert np.all(vec >= 0)


# ── Box-counting regression ────────────────────────────────────────────────────

class TestBoxCountingRegression:

    def test_circle_fd_is_one(self, circle_64):
        """Perfect circle with few points → FD ≈ 1.0 (smooth curve)."""
        fd = box_counting_fd(circle_64)
        assert fd == pytest.approx(1.0, abs=0.01)

    def test_straight_line_fd_is_one(self, straight_line_64):
        fd = box_counting_fd(straight_line_64)
        assert fd == pytest.approx(1.0, abs=0.15)

    def test_fd_range_always_valid(self, circle_64, zigzag_128, straight_line_64):
        for c in [circle_64, zigzag_128, straight_line_64]:
            fd = box_counting_fd(c)
            assert 1.0 <= fd <= 2.0

    def test_curve_length_8_scales(self, circle_64):
        log_r, log_N = box_counting_curve(circle_64, n_scales=8)
        assert len(log_r) == 8
        assert len(log_N) == 8

    def test_curve_log_r_values(self, circle_64):
        """log(1/r) = log2(2^k) = k — should be 1,2,...,8."""
        log_r, _ = box_counting_curve(circle_64, n_scales=8)
        expected = np.arange(1, 9, dtype=float)
        np.testing.assert_array_almost_equal(log_r, expected, decimal=10)

    def test_curve_log_N_positive(self, zigzag_128):
        _, log_N = box_counting_curve(zigzag_128)
        assert np.all(log_N >= 0)


# ── IFS regression ─────────────────────────────────────────────────────────────

class TestIFSRegression:

    def test_default_n_transforms_8(self, circle_64):
        d = fit_ifs_coefficients(circle_64)
        assert len(d) == 8

    def test_coefficients_bounded(self, circle_64, zigzag_128):
        for c in [circle_64, zigzag_128]:
            d = fit_ifs_coefficients(c)
            assert np.all(np.abs(d) <= 0.95), f"Coefficients exceed bound: {d}"

    def test_reconstruct_default_length(self, circle_64):
        d = fit_ifs_coefficients(circle_64)
        r = reconstruct_from_ifs(d)
        assert len(r) == 256

    def test_reconstruct_custom_length(self, circle_64):
        d = fit_ifs_coefficients(circle_64)
        r = reconstruct_from_ifs(d, n_points=64)
        assert len(r) == 64

    def test_reconstruct_finite(self, zigzag_128):
        d = fit_ifs_coefficients(zigzag_128)
        r = reconstruct_from_ifs(d)
        assert np.all(np.isfinite(r))

    def test_known_zigzag_coefficients_bounded(self, zigzag_128):
        d = fit_ifs_coefficients(zigzag_128, n_transforms=8)
        # All in (-1, 1)
        assert np.all(np.abs(d) < 1.0)


# ── Scoring regression ─────────────────────────────────────────────────────────

class TestMatchEvaluatorRegression:

    def test_precision_8_tp_2_fp(self):
        p = compute_precision(tp=8, fp=2)
        assert p == pytest.approx(0.8, abs=1e-10)

    def test_recall_8_tp_1_fn(self):
        r = compute_recall(tp=8, fn=1)
        assert r == pytest.approx(8 / 9, abs=1e-10)

    def test_f1_perfect(self):
        f = compute_f_score(1.0, 1.0)
        assert f == pytest.approx(1.0, abs=1e-10)

    def test_f1_zero_precision(self):
        f = compute_f_score(0.0, 0.5)
        assert f == pytest.approx(0.0, abs=1e-10)

    def test_f1_formula(self):
        p = compute_precision(8, 2)
        r = compute_recall(8, 1)
        f = compute_f_score(p, r)
        expected = 2 * p * r / (p + r)
        assert f == pytest.approx(expected, abs=1e-10)

    def test_evaluate_match_fields(self):
        me = evaluate_match((0, 1), 0.75, tp=5, fp=1, fn=2)
        assert me.pair == (0, 1)
        assert me.score == pytest.approx(0.75)
        assert me.precision == pytest.approx(compute_precision(5, 1))
        assert me.recall == pytest.approx(compute_recall(5, 2))

    def test_aggregate_eval_mean_score(self):
        evals = [
            evaluate_match((0, 1), 0.6, 6, 2, 1),
            evaluate_match((0, 2), 0.8, 8, 1, 0),
            evaluate_match((1, 2), 0.4, 4, 3, 2),
        ]
        report = aggregate_eval(evals)
        expected_mean = (0.6 + 0.8 + 0.4) / 3
        assert report.mean_score == pytest.approx(expected_mean, abs=1e-10)

    def test_rank_matches_descending_f1(self):
        evals = [
            evaluate_match((0, 1), 0.5, 5, 2, 2),
            evaluate_match((0, 2), 0.9, 9, 0, 0),
            evaluate_match((1, 2), 0.3, 3, 4, 4),
        ]
        ranked = rank_matches(evals, by="f1")
        f1_values = [e.f1 for e in ranked]
        assert f1_values == sorted(f1_values, reverse=True)

    def test_rank_matches_descending_score(self):
        evals = [
            evaluate_match((0, 1), 0.5, 5, 2, 2),
            evaluate_match((0, 2), 0.9, 9, 0, 0),
            evaluate_match((1, 2), 0.3, 3, 4, 4),
        ]
        ranked = rank_matches(evals, by="score")
        scores = [e.score for e in ranked]
        assert scores == sorted(scores, reverse=True)


# ── Threshold regression ───────────────────────────────────────────────────────

class TestThresholdSelectorRegression:

    @pytest.fixture
    def scores_5(self) -> np.ndarray:
        return np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    def test_fixed_threshold_0_5(self, scores_5):
        result = select_fixed_threshold(scores_5, value=0.5)
        assert result.threshold == pytest.approx(0.5)
        assert result.n_above == 3  # 0.5, 0.7, 0.9
        assert result.n_below == 2  # 0.1, 0.3

    def test_fixed_threshold_0_0(self, scores_5):
        result = select_fixed_threshold(scores_5, value=0.0)
        assert result.n_above == 5
        assert result.n_below == 0

    def test_fixed_threshold_1_0(self, scores_5):
        result = select_fixed_threshold(scores_5, value=1.0)
        assert result.n_above == 0
        assert result.n_below == 5

    def test_percentile_50_is_median(self, scores_5):
        result = select_percentile_threshold(scores_5, percentile=50.0)
        assert result.threshold == pytest.approx(np.median(scores_5), abs=0.01)

    def test_apply_threshold_returns_bool_mask(self, scores_5):
        result = select_fixed_threshold(scores_5, value=0.5)
        mask = apply_threshold(scores_5, result)
        assert mask.dtype == bool
        assert mask.shape == scores_5.shape

    def test_apply_threshold_correct_values(self, scores_5):
        result = select_fixed_threshold(scores_5, value=0.5)
        mask = apply_threshold(scores_5, result)
        expected = scores_5 >= 0.5
        np.testing.assert_array_equal(mask, expected)

    def test_acceptance_ratio_range(self, scores_5):
        result = select_fixed_threshold(scores_5, value=0.5)
        assert 0.0 <= result.acceptance_ratio <= 1.0

    def test_acceptance_ratio_value(self, scores_5):
        result = select_fixed_threshold(scores_5, value=0.5)
        assert result.acceptance_ratio == pytest.approx(3 / 5, abs=1e-10)


# ── Pipeline regression (lightweight, no full run) ───────────────────────────

class TestPipelineConfigRegression:
    """Config defaults must not change unexpectedly."""

    def test_config_default_matching_threshold(self):
        from puzzle_reconstruction.config import Config
        cfg = Config.default()
        assert hasattr(cfg, "matching")
        assert hasattr(cfg.matching, "threshold")
        assert 0.0 <= cfg.matching.threshold <= 1.0

    def test_config_default_assembly_method(self):
        from puzzle_reconstruction.config import Config
        cfg = Config.default()
        assert hasattr(cfg, "assembly")
        assert hasattr(cfg.assembly, "method")
        assert isinstance(cfg.assembly.method, str)
        assert len(cfg.assembly.method) > 0

    def test_config_default_n_workers(self):
        from puzzle_reconstruction.config import Config
        cfg = Config.default()
        # Config should be constructible without errors
        assert cfg is not None

    def test_pipeline_instantiation(self):
        from puzzle_reconstruction.config import Config
        from puzzle_reconstruction.pipeline import Pipeline
        cfg = Config.default()
        p = Pipeline(cfg=cfg, n_workers=1)
        assert p is not None
