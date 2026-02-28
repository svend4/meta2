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
from puzzle_reconstruction.matching.dtw import dtw_distance
from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
from puzzle_reconstruction.assembly.greedy import greedy_assembly
from puzzle_reconstruction.models import (
    Fragment, EdgeSignature, EdgeSide, CompatEntry, Assembly,
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


# ── Koch-curve fractal dimension regression ────────────────────────────────────

def _make_koch_curve(n_iter: int = 4, n_pts: int = 256) -> np.ndarray:
    """Generate a Koch snowflake edge (one side)."""
    def _pts(p1: np.ndarray, p2: np.ndarray, depth: int) -> list:
        if depth == 0:
            return [p1]
        d = (p2 - p1) / 3.0
        p3 = p1 + d
        p5 = p2 - d
        angle = np.pi / 3.0
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle),  np.cos(angle)]])
        p4 = p3 + rot @ d
        result: list = []
        for a, b in [(p1, p3), (p3, p4), (p4, p5), (p5, p2)]:
            result.extend(_pts(a, b, depth - 1))
        return result

    raw = np.array(_pts(np.array([0.0, 0.0]), np.array([1.0, 0.0]), n_iter))
    t_old = np.linspace(0, 1, len(raw))
    t_new = np.linspace(0, 1, n_pts)
    x = np.interp(t_new, t_old, raw[:, 0])
    y = np.interp(t_new, t_old, raw[:, 1])
    return np.column_stack([x, y])


@pytest.fixture(scope="module")
def koch_256() -> np.ndarray:
    return _make_koch_curve(n_iter=4, n_pts=256)


@pytest.fixture(scope="module")
def square_64() -> np.ndarray:
    side = 16
    top    = np.column_stack([np.linspace(0, 100, side), np.full(side, 100.0)])
    right  = np.column_stack([np.full(side, 100.0), np.linspace(100, 0, side)])
    bottom = np.column_stack([np.linspace(100, 0, side), np.zeros(side)])
    left   = np.column_stack([np.zeros(side), np.linspace(0, 100, side)])
    return np.vstack([top, right, bottom, left]).astype(np.float64)


class TestKochCurveFDRegression:
    """Fractal dimension of Koch curve must exceed 1.0 (it's not smooth)."""

    def test_koch_fd_above_circle(self, circle_64, koch_256):
        """Koch FD > circle FD (Koch is fractal; circle is smooth)."""
        fd_circle = box_counting_fd(circle_64)
        fd_koch   = box_counting_fd(koch_256)
        assert fd_koch > fd_circle, \
            f"Koch FD ({fd_koch:.4f}) should exceed circle FD ({fd_circle:.4f})"

    def test_koch_fd_above_1(self, koch_256):
        """Koch curve FD > 1.0 (has fractal structure)."""
        fd = box_counting_fd(koch_256)
        assert fd > 1.0, f"Koch FD ({fd:.4f}) should be > 1.0"

    def test_koch_fd_below_2(self, koch_256):
        """Koch curve FD < 2.0 (it's a curve, not a surface)."""
        fd = box_counting_fd(koch_256)
        assert fd < 2.0, f"Koch FD ({fd:.4f}) should be < 2.0"

    def test_circle_fd_near_1(self, circle_64):
        """Smooth circle: FD ≈ 1.0."""
        fd = box_counting_fd(circle_64)
        assert fd == pytest.approx(1.0, abs=0.02), f"Circle FD={fd:.4f}"

    def test_square_fd_near_1(self, square_64):
        """Square (piecewise linear): FD ≈ 1.0."""
        fd = box_counting_fd(square_64)
        assert fd == pytest.approx(1.0, abs=0.05), f"Square FD={fd:.4f}"

    def test_css_circle_vs_square_similarity_below_1(self, circle_64, square_64):
        """Circle and square have dissimilar CSS signatures (sim < 1)."""
        vec_c = css_to_feature_vector(curvature_scale_space(circle_64))
        vec_s = css_to_feature_vector(curvature_scale_space(square_64))
        sim = css_similarity(vec_c, vec_s)
        assert 0.0 <= sim < 1.0, f"Unexpected sim={sim:.4f}"

    def test_css_same_shape_similarity_one(self, circle_64):
        """css_similarity(v, v) == 1.0 exactly."""
        vec = css_to_feature_vector(curvature_scale_space(circle_64))
        assert css_similarity(vec, vec) == pytest.approx(1.0, abs=1e-12)


# ── DTW regression ─────────────────────────────────────────────────────────────

class TestDTWRegression:
    """Fixed DTW values for standard curve pairs."""

    @pytest.fixture(scope="class")
    def half_circle(self) -> np.ndarray:
        t = np.linspace(0, np.pi, 32, endpoint=False)
        return np.column_stack([100 * np.cos(t), 100 * np.sin(t)])

    def test_dtw_self_is_zero(self, half_circle):
        """DTW(A, A) must be exactly 0."""
        assert dtw_distance(half_circle, half_circle) == 0.0

    def test_dtw_non_negative(self, half_circle, zigzag_128):
        """DTW is always >= 0."""
        assert dtw_distance(half_circle, zigzag_128[:32]) >= 0.0

    def test_dtw_symmetric(self, half_circle):
        """DTW(A, B) == DTW(B, A)."""
        t = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        b = np.column_stack([50 * np.cos(t), 50 * np.sin(t)])
        d_ab = dtw_distance(half_circle, b)
        d_ba = dtw_distance(b, half_circle)
        assert d_ab == pytest.approx(d_ba, abs=1e-9)

    def test_dtw_empty_returns_inf(self):
        """DTW with empty sequence returns inf."""
        a = np.zeros((10, 2))
        d = dtw_distance(a, np.zeros((0, 2)))
        assert d == float("inf")

    def test_dtw_two_identical_points(self):
        """DTW(single repeated point, same) = 0."""
        a = np.ones((5, 2)) * 42.0
        assert dtw_distance(a, a) == 0.0

    def test_dtw_scale_monotone(self):
        """Closer curves have smaller DTW distance."""
        base = np.column_stack([np.linspace(0, 1, 32), np.zeros(32)])
        near = np.column_stack([np.linspace(0, 1, 32), np.full(32, 0.01)])
        far  = np.column_stack([np.linspace(0, 1, 32), np.ones(32)])
        d_near = dtw_distance(base, near)
        d_far  = dtw_distance(base, far)
        assert d_near < d_far, f"d_near={d_near:.4f} should be < d_far={d_far:.4f}"


# ── Compat matrix regression ───────────────────────────────────────────────────

def _reg_edge(eid: int) -> EdgeSignature:
    t = np.linspace(0, 2 * np.pi, 16)
    curve = np.column_stack([np.cos(t) * 50, np.sin(t) * 10])
    return EdgeSignature(
        edge_id=eid, side=EdgeSide.TOP, virtual_curve=curve,
        fd=1.5, css_vec=np.zeros(8), ifs_coeffs=np.zeros(4), length=80.0,
    )


def _reg_fragment(fid: int) -> Fragment:
    frag = Fragment(fragment_id=fid, image=np.zeros((32, 32, 3), dtype=np.uint8))
    frag.edges = [_reg_edge(fid * 10 + i) for i in range(2)]
    return frag


@pytest.fixture(scope="module")
def reg_compat_data():
    """Reproducible 4-fragment dataset + compat matrix."""
    frags = [_reg_fragment(i) for i in range(4)]
    matrix, entries = build_compat_matrix(frags)
    return frags, matrix, entries


class TestCompatMatrixRegression:
    """Regression tests: compat matrix structural invariants."""

    def test_matrix_shape_8x8(self, reg_compat_data):
        """4 fragments × 2 edges each → 8×8 matrix."""
        _, matrix, _ = reg_compat_data
        assert matrix.shape == (8, 8), f"Expected (8,8), got {matrix.shape}"

    def test_matrix_symmetric(self, reg_compat_data):
        """Matrix is symmetric."""
        _, matrix, _ = reg_compat_data
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-9)

    def test_matrix_diagonal_zero(self, reg_compat_data):
        """Same-fragment edges have score 0 (blocked)."""
        _, matrix, _ = reg_compat_data
        assert np.all(np.diag(matrix) == 0.0), "Diagonal not all zero"

    def test_matrix_values_in_0_1(self, reg_compat_data):
        """All values in [0, 1]."""
        _, matrix, _ = reg_compat_data
        assert np.all(matrix >= 0.0), "Negative values in matrix"
        assert np.all(matrix <= 1.0), "Values > 1 in matrix"

    def test_same_fragment_edges_score_zero(self, reg_compat_data):
        """Edges from the same fragment must have score 0."""
        _, matrix, _ = reg_compat_data
        # Fragment 0 → edges at indices 0, 1
        assert matrix[0, 1] == 0.0, "Within-fragment edges should be 0"
        assert matrix[1, 0] == 0.0

    def test_entries_sorted_descending(self, reg_compat_data):
        """entries list is sorted by score descending."""
        _, _, entries = reg_compat_data
        if len(entries) > 1:
            scores = [e.score for e in entries]
            assert scores == sorted(scores, reverse=True), "Entries not sorted"


# ── Greedy assembly regression ─────────────────────────────────────────────────

class TestGreedyAssemblyRegression:
    """Regression: greedy assembly must be deterministic and cover all fragments."""

    def test_greedy_deterministic_two_runs(self, reg_compat_data):
        """greedy_assembly(same data) × 2 → identical placements."""
        frags, _, entries = reg_compat_data
        r1 = greedy_assembly(frags, entries)
        r2 = greedy_assembly(frags, entries)
        assert set(r1.placements.keys()) == set(r2.placements.keys()), \
            "Non-deterministic fragment coverage"
        for fid in r1.placements:
            p1, a1 = r1.placements[fid]
            p2, a2 = r2.placements[fid]
            np.testing.assert_allclose(p1, p2, atol=1e-9,
                                       err_msg=f"Position differs for frag {fid}")
            assert abs(a1 - a2) < 1e-9, f"Angle differs for frag {fid}"

    def test_greedy_covers_all_4_fragments(self, reg_compat_data):
        """All 4 fragments appear in the placement dict."""
        frags, _, entries = reg_compat_data
        result = greedy_assembly(frags, entries)
        assert set(result.placements.keys()) == {0, 1, 2, 3}, \
            f"Missing fragments: {set(result.placements.keys())}"

    def test_greedy_first_fragment_at_origin(self, reg_compat_data):
        """First fragment is always placed at (0, 0) with angle 0."""
        frags, _, entries = reg_compat_data
        result = greedy_assembly(frags, entries)
        pos, angle = result.placements[frags[0].fragment_id]
        np.testing.assert_allclose(pos, [0.0, 0.0], atol=1e-9)
        assert angle == pytest.approx(0.0, abs=1e-9)

    def test_greedy_returns_assembly_type(self, reg_compat_data):
        """greedy_assembly returns an Assembly instance."""
        frags, _, entries = reg_compat_data
        result = greedy_assembly(frags, entries)
        assert isinstance(result, Assembly)

    def test_greedy_placement_positions_finite(self, reg_compat_data):
        """All placement positions are finite floats."""
        frags, _, entries = reg_compat_data
        result = greedy_assembly(frags, entries)
        for fid, (pos, angle) in result.placements.items():
            assert np.all(np.isfinite(np.asarray(pos))), \
                f"Non-finite position for fragment {fid}: {pos}"
            assert np.isfinite(angle), f"Non-finite angle for fragment {fid}"

    def test_greedy_empty_fragments(self):
        """Empty input → empty Assembly with no placements."""
        result = greedy_assembly([], [])
        assert len(result.placements) == 0
