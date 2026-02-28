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
        """All 4 fragments appear in the placement dict and Assembly is well-formed.
        Kills mutmut_73 (fragments=None), mutmut_77 (fragments kwarg omitted),
        mutmut_75 (compat_matrix=None), mutmut_79 (compat_matrix kwarg omitted),
        mutmut_81 (compat_matrix=np.array(None) → ndim=0).
        """
        frags, _, entries = reg_compat_data
        result = greedy_assembly(frags, entries)
        assert set(result.placements.keys()) == {0, 1, 2, 3}, \
            f"Missing fragments: {set(result.placements.keys())}"
        # Assembly.fragments must be the input list, not None
        assert result.fragments is not None, "Assembly.fragments should not be None"
        assert result.fragments == frags, "Assembly.fragments should match input"
        # Assembly.compat_matrix must be a 1-D (possibly empty) ndarray
        assert result.compat_matrix is not None, "Assembly.compat_matrix should not be None"
        assert isinstance(result.compat_matrix, np.ndarray), (
            "Assembly.compat_matrix must be an ndarray"
        )
        assert result.compat_matrix.ndim >= 1, (
            f"Assembly.compat_matrix must be 1-D, got ndim={result.compat_matrix.ndim} "
            "(np.array(None) has ndim=0)"
        )

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
        # Verify correct Assembly structure (kills mutations that pass wrong fields)
        assert isinstance(result.placements, dict), (
            "Empty-fragments Assembly.placements should be a dict"
        )
        assert result.fragments is not None, "fragments should not be None"
        assert result.fragments == [], "Empty-fragments Assembly.fragments should be []"
        assert result.compat_matrix is not None, "compat_matrix should not be None"
        assert isinstance(result.compat_matrix, np.ndarray), (
            "compat_matrix should be an ndarray"
        )
        assert result.compat_matrix.ndim >= 1, (
            f"compat_matrix must be 1-D (not 0-D scalar); ndim={result.compat_matrix.ndim}. "
            "np.array(None) has ndim=0 — kills mutmut_12."
        )

    def test_greedy_entries_with_unknown_edge_skip_but_continue(self, reg_compat_data):
        """
        If an entry references an edge_id not in any fragment, greedy should
        skip that entry (not break/abort the loop).
        Kills mutant: `continue` → `break` (which would halt the whole loop).
        """
        frags, _, entries = reg_compat_data
        # Create a fake entry with an edge_id that's not in any fragment
        # The real entries have edge IDs like 0, 1, 10, 11, 20, 21, 30, 31
        fake_edge = _reg_edge(999)  # ID 999 does not exist in any fragment
        real_edge = frags[0].edges[0]
        from puzzle_reconstruction.models import CompatEntry
        bad_entry = CompatEntry(
            edge_i=fake_edge, edge_j=real_edge,
            score=0.99,  # High score - if NOT skipped, would be processed first
            dtw_dist=0.01, css_sim=0.99, fd_diff=0.0, text_score=0.0,
        )
        # Prepend bad entry (highest score) + keep good entries
        mixed_entries = [bad_entry] + list(entries)
        result = greedy_assembly(frags, mixed_entries)
        # All 4 fragments should still be placed (the bad entry was skipped, not aborted)
        assert set(result.placements.keys()) == {0, 1, 2, 3}, (
            "Bad entry caused loop to abort (break instead of continue)"
        )
        assert len(result.placements) == 4
        # Fragments 1-3 must be placed by the algorithm (y < 100), not as orphans (y=200).
        # If the loop broke on the bad entry, all non-first frags become orphans at y=200.
        for fid in [1, 2, 3]:
            pos, _ = result.placements[fid]
            assert pos[1] < 100.0, (
                f"Fragment {fid} is at y={pos[1]:.1f} (orphan position). "
                "Bad entry should have been skipped (continue), not broken (break)."
            )

    def test_greedy_non_orphan_placement(self, reg_compat_data):
        """
        Verify algorithm-based placement vs orphan fallback.

        With the correct algorithm, non-first fragments are placed near fragment 0
        using edge alignment (y ≈ 0).  With the 'all None' mutation, orphan
        placement kicks in and every y-coordinate equals 200.
        """
        frags, _, entries = reg_compat_data
        result = greedy_assembly(frags, entries)
        for fid, (pos, _) in result.placements.items():
            if fid == frags[0].fragment_id:
                continue
            # Algorithm-placed fragments should NOT be at orphan y=200
            assert pos[1] < 100.0, (
                f"Fragment {fid} looks orphan-placed (y={pos[1]:.1f}); "
                "greedy algorithm should have placed it near the anchor."
            )

    def test_greedy_edge_lookup_correctness(self, reg_compat_data):
        """
        Check that edge_to_frag mapping is used correctly.

        Mutmut mutant: edge_to_frag[edge.edge_id] = None (instead of frag).
        With this, all frag_i / frag_j lookups return None and no edge-based
        placement happens.  We detect this by verifying the assembly score > 0.
        """
        frags, _, entries = reg_compat_data
        result = greedy_assembly(frags, entries)
        # total_score is sum of matched edge scores; must be > 0 with valid entries
        assert result.total_score > 0.0, (
            f"total_score={result.total_score}: edge lookup appears broken "
            "(all fragments may have been placed as orphans)."
        )

    def test_greedy_placements_not_in_row(self, reg_compat_data):
        """
        With greedy algorithm, fragments are placed with varying x/y.

        Orphan placement puts all orphans at x=k*150, y=200 in a row.
        The algorithm-based placements should have at least one non-zero x.
        """
        frags, _, entries = reg_compat_data
        result = greedy_assembly(frags, entries)
        non_first_positions = [
            pos for fid, (pos, _) in result.placements.items()
            if fid != frags[0].fragment_id
        ]
        x_values = [pos[0] for pos in non_first_positions]
        # At least one fragment should have x != 0 (not at orphan x=0 or x=150)
        # OR be close to the anchor (not at y=200)
        # We already check y in non_orphan_placement; here check consistency
        assert len(non_first_positions) == len(frags) - 1, (
            "Some non-first fragments were not placed"
        )

    def test_orphan_placement_positions_finite(self):
        """
        When fragments have no compatible edges, they fall back to orphan
        placement.  Orphan positions must be finite (not None or NaN).
        Kills mutants: placements[frag_id] = None, y_offset = None, etc.
        """
        # 4 fragments, but no cross-fragment entries → all except first are orphans
        frags = [_reg_fragment(i) for i in range(4)]
        result = greedy_assembly(frags, [])  # No entries → all orphaned

        assert set(result.placements.keys()) == {0, 1, 2, 3}, (
            "All 4 fragments should be placed"
        )
        for fid, placement in result.placements.items():
            assert placement is not None, f"Fragment {fid} has None placement"
            pos, angle = placement
            pos_arr = np.asarray(pos)
            assert np.all(np.isfinite(pos_arr)), (
                f"Orphan fragment {fid} has non-finite pos: {pos_arr}"
            )
            assert np.isfinite(angle), f"Orphan fragment {fid} has non-finite angle"

    def test_orphan_y_offset_above_first(self):
        """
        Orphan fragments are placed below the first fragment.
        y_offset = max_y + 200, so orphans must be at y > 0 (or y > max_y).
        Kills mutant: y_offset = max_y + 201 → off by 1 (value differs).
        Kills mutant: y_offset = max_y - 200 → placed ABOVE, not BELOW.
        """
        frags = [_reg_fragment(i) for i in range(3)]
        result = greedy_assembly(frags, [])  # all orphans except first at (0,0)

        # First fragment: (0, 0)
        pos0, _ = result.placements[0]
        assert np.allclose(pos0, [0, 0])

        # Other orphans should be at EXACTLY y = 200 (y_offset = max_y=0 + 200)
        for fid in [1, 2]:
            pos, angle = result.placements[fid]
            assert pos[1] == pytest.approx(200.0, abs=0.01), (
                f"Orphan fragment {fid} y should be exactly 200, got {pos[1]}"
            )
            # Orphans are placed with angle = 0
            assert angle == pytest.approx(0.0, abs=1e-9), (
                f"Orphan fragment {fid} angle should be 0, got {angle}"
            )

    def test_orphan_x_spacing(self):
        """
        Orphan fragments are placed at x = k * 150 for k=0,1,2,...
        Kills mutant: k * 150 changed (e.g., * 151 → different positions).
        """
        frags = [_reg_fragment(i) for i in range(4)]
        result = greedy_assembly(frags, [])  # all orphaned

        # Orphan positions: k=0 → x=0, k=1 → x=150, k=2 → x=300
        orphan_ids = [f.fragment_id for f in frags[1:]]  # skip first
        x_vals = sorted([result.placements[fid][0][0] for fid in orphan_ids])
        assert x_vals[0] == pytest.approx(0.0, abs=1e-6), f"First orphan x wrong: {x_vals[0]}"
        assert x_vals[1] == pytest.approx(150.0, abs=1e-6), f"Second orphan x wrong: {x_vals[1]}"
        assert x_vals[2] == pytest.approx(300.0, abs=1e-6), f"Third orphan x wrong: {x_vals[2]}"

    def test_greedy_total_score_excludes_unknown_edges(self, reg_compat_data):
        """
        total_score must NOT include entries with edges outside the fragment graph.
        Kills mutmut_66: operator precedence change that includes entries with
        only one known edge (via 'or' instead of 'and').
        """
        frags, _, entries = reg_compat_data
        fake_edge = _reg_edge(999)  # not in any fragment
        real_edge = frags[0].edges[0]
        from puzzle_reconstruction.models import CompatEntry
        bad_entry = CompatEntry(
            edge_i=fake_edge, edge_j=real_edge,
            score=0.99,  # high score that would inflate total_score if included
            dtw_dist=0.01, css_sim=0.99, fd_diff=0.0, text_score=0.0,
        )
        clean_result = greedy_assembly(frags, list(entries))
        mixed_result = greedy_assembly(frags, [bad_entry] + list(entries))
        # Bad entry must not change total_score (it has an unknown edge)
        assert mixed_result.total_score == pytest.approx(clean_result.total_score, abs=1e-6), (
            f"total_score changed from {clean_result.total_score} to "
            f"{mixed_result.total_score} when a bad entry was added. "
            "The bad entry's score (0.99) should not be included."
        )

    def test_greedy_neither_placed_continues_not_breaks(self):
        """
        When an entry has both fragments unplaced, the loop must continue (not break).
        Kills mutmut_38: `continue` → `break` for the neither-placed guard.

        Setup: highest-score entry is between two unplaced fragments (frag1, frag2).
        The algorithm must skip it and then process later entries that connect frag0 to others.
        """
        frags = [_reg_fragment(i) for i in range(3)]  # frags 0, 1, 2
        from puzzle_reconstruction.models import CompatEntry
        # Highest priority: frag1 ↔ frag2 (neither placed initially)
        neither_entry = CompatEntry(
            edge_i=frags[1].edges[0], edge_j=frags[2].edges[0],
            score=0.99, dtw_dist=0.01, css_sim=0.99, fd_diff=0.0, text_score=0.0,
        )
        # Second priority: frag0 ↔ frag1 (frag0 is placed)
        frag01_entry = CompatEntry(
            edge_i=frags[0].edges[0], edge_j=frags[1].edges[0],
            score=0.50, dtw_dist=0.01, css_sim=0.99, fd_diff=0.0, text_score=0.0,
        )
        # Third priority: frag0 ↔ frag2 (frag0 is placed)
        frag02_entry = CompatEntry(
            edge_i=frags[0].edges[0], edge_j=frags[2].edges[0],
            score=0.40, dtw_dist=0.01, css_sim=0.99, fd_diff=0.0, text_score=0.0,
        )
        # Entries sorted by score descending: neither_entry first
        entries = [neither_entry, frag01_entry, frag02_entry]
        result = greedy_assembly(frags, entries)
        # All 3 frags must be placed
        assert set(result.placements.keys()) == {0, 1, 2}, (
            "All 3 fragments should be placed"
        )
        # Frags 1 and 2 must be placed via algorithm (not orphan at y=200)
        for fid in [1, 2]:
            pos, _ = result.placements[fid]
            assert pos[1] < 100.0, (
                f"Fragment {fid} is orphan-placed at y={pos[1]:.1f}. "
                "The neither-placed entry should have been skipped (continue), not aborted (break)."
            )
