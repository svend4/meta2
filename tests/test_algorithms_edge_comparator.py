"""Tests for puzzle_reconstruction/algorithms/edge_comparator.py"""
import pytest
import numpy as np

from puzzle_reconstruction.models import EdgeSignature, EdgeSide
from puzzle_reconstruction.algorithms.edge_comparator import (
    CompareConfig,
    EdgeCompareResult,
    dtw_distance,
    css_similarity,
    fd_score,
    ifs_similarity,
    compare_edges,
    build_compat_matrix,
    top_k_matches,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_curve(n=10, scale=1.0):
    angles = np.linspace(0, np.pi, n)
    return np.column_stack([np.cos(angles) * scale,
                            np.sin(angles) * scale]).astype(float)


def make_edge_sig(edge_id=0, n_pts=8, fd=1.5, scale=1.0):
    curve = make_curve(n_pts, scale)
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.RIGHT,
        virtual_curve=curve,
        fd=fd,
        css_vec=np.ones(8) / 8.0,
        ifs_coeffs=np.ones(5) * 0.5,
        length=float(n_pts),
    )


# ─── CompareConfig ────────────────────────────────────────────────────────────

class TestCompareConfig:
    def test_defaults(self):
        cfg = CompareConfig()
        assert cfg.w_dtw == pytest.approx(0.4)
        assert cfg.w_css == pytest.approx(0.3)
        assert cfg.w_fd == pytest.approx(0.15)
        assert cfg.w_ifs == pytest.approx(0.15)
        assert cfg.dtw_band == 0
        assert cfg.fd_sigma == pytest.approx(0.5)

    def test_negative_w_dtw_raises(self):
        with pytest.raises(ValueError):
            CompareConfig(w_dtw=-0.1)

    def test_negative_w_css_raises(self):
        with pytest.raises(ValueError):
            CompareConfig(w_css=-1.0)

    def test_negative_w_fd_raises(self):
        with pytest.raises(ValueError):
            CompareConfig(w_fd=-0.5)

    def test_negative_w_ifs_raises(self):
        with pytest.raises(ValueError):
            CompareConfig(w_ifs=-0.01)

    def test_negative_dtw_band_raises(self):
        with pytest.raises(ValueError):
            CompareConfig(dtw_band=-1)

    def test_zero_fd_sigma_raises(self):
        with pytest.raises(ValueError):
            CompareConfig(fd_sigma=0.0)

    def test_negative_fd_sigma_raises(self):
        with pytest.raises(ValueError):
            CompareConfig(fd_sigma=-1.0)

    def test_total_weight_sums(self):
        cfg = CompareConfig()
        assert cfg.total_weight == pytest.approx(1.0)

    def test_custom_weights(self):
        cfg = CompareConfig(w_dtw=1.0, w_css=1.0, w_fd=1.0, w_ifs=1.0)
        assert cfg.total_weight == pytest.approx(4.0)

    def test_zero_weights_allowed(self):
        cfg = CompareConfig(w_dtw=0.0, w_css=0.0, w_fd=0.5, w_ifs=0.5)
        assert cfg.w_dtw == pytest.approx(0.0)


# ─── EdgeCompareResult ────────────────────────────────────────────────────────

class TestEdgeCompareResult:
    def test_basic_creation(self):
        r = EdgeCompareResult(
            edge_id_a=0, edge_id_b=1,
            dtw_dist=2.5, css_sim=0.8, fd_diff=0.1,
            ifs_sim=0.9, score=0.7,
        )
        assert r.score == pytest.approx(0.7)

    def test_negative_dtw_dist_raises(self):
        with pytest.raises(ValueError):
            EdgeCompareResult(edge_id_a=0, edge_id_b=1,
                              dtw_dist=-0.1, css_sim=0.5, fd_diff=0.0,
                              ifs_sim=0.5, score=0.5)

    def test_css_sim_out_of_range_raises(self):
        with pytest.raises(ValueError):
            EdgeCompareResult(edge_id_a=0, edge_id_b=1,
                              dtw_dist=1.0, css_sim=1.5, fd_diff=0.0,
                              ifs_sim=0.5, score=0.5)

    def test_negative_fd_diff_raises(self):
        with pytest.raises(ValueError):
            EdgeCompareResult(edge_id_a=0, edge_id_b=1,
                              dtw_dist=1.0, css_sim=0.5, fd_diff=-0.1,
                              ifs_sim=0.5, score=0.5)

    def test_ifs_sim_out_of_range_raises(self):
        with pytest.raises(ValueError):
            EdgeCompareResult(edge_id_a=0, edge_id_b=1,
                              dtw_dist=1.0, css_sim=0.5, fd_diff=0.0,
                              ifs_sim=-0.1, score=0.5)

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            EdgeCompareResult(edge_id_a=0, edge_id_b=1,
                              dtw_dist=1.0, css_sim=0.5, fd_diff=0.0,
                              ifs_sim=0.5, score=1.5)

    def test_pair_key_sorted(self):
        r = EdgeCompareResult(edge_id_a=5, edge_id_b=2,
                              dtw_dist=0.0, css_sim=1.0, fd_diff=0.0,
                              ifs_sim=1.0, score=1.0)
        assert r.pair_key == (2, 5)

    def test_pair_key_same_order(self):
        r = EdgeCompareResult(edge_id_a=1, edge_id_b=3,
                              dtw_dist=0.0, css_sim=1.0, fd_diff=0.0,
                              ifs_sim=1.0, score=1.0)
        assert r.pair_key == (1, 3)

    def test_is_compatible_above_threshold(self):
        r = EdgeCompareResult(edge_id_a=0, edge_id_b=1,
                              dtw_dist=0.5, css_sim=0.8, fd_diff=0.1,
                              ifs_sim=0.8, score=0.7)
        assert r.is_compatible is True

    def test_is_compatible_below_threshold(self):
        r = EdgeCompareResult(edge_id_a=0, edge_id_b=1,
                              dtw_dist=5.0, css_sim=0.2, fd_diff=0.5,
                              ifs_sim=0.2, score=0.3)
        assert r.is_compatible is False

    def test_score_zero_ok(self):
        r = EdgeCompareResult(edge_id_a=0, edge_id_b=1,
                              dtw_dist=0.0, css_sim=0.0, fd_diff=0.0,
                              ifs_sim=0.0, score=0.0)
        assert r.score == pytest.approx(0.0)


# ─── dtw_distance ─────────────────────────────────────────────────────────────

class TestDtwDistance:
    def test_identical_curves_zero(self):
        a = make_curve(5)
        dist = dtw_distance(a, a)
        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_nonnegative(self):
        a = make_curve(5)
        b = make_curve(5, scale=2.0)
        assert dtw_distance(a, b) >= 0.0

    def test_different_lengths(self):
        a = make_curve(5)
        b = make_curve(10)
        dist = dtw_distance(a, b)
        assert dist >= 0.0

    def test_not_2d_raises(self):
        bad = np.ones((5, 3))
        good = make_curve(5)
        with pytest.raises(ValueError):
            dtw_distance(bad, good)

    def test_b_not_2d_raises(self):
        good = make_curve(5)
        bad = np.ones((5, 3))
        with pytest.raises(ValueError):
            dtw_distance(good, bad)

    def test_negative_band_raises(self):
        a = make_curve(5)
        with pytest.raises(ValueError):
            dtw_distance(a, a, band=-1)

    def test_with_band_constraint(self):
        a = make_curve(8)
        b = make_curve(8, scale=1.5)
        dist = dtw_distance(a, b, band=3)
        assert dist >= 0.0

    def test_scaled_curve_larger_dist(self):
        a = make_curve(8)
        b_close = make_curve(8, scale=1.1)
        b_far = make_curve(8, scale=3.0)
        assert dtw_distance(a, b_far) > dtw_distance(a, b_close)


# ─── css_similarity ───────────────────────────────────────────────────────────

class TestCssSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert css_similarity(v, v) == pytest.approx(1.0)

    def test_range(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        result = css_similarity(v1, v2)
        assert 0.0 <= result <= 1.0

    def test_shape_mismatch_raises(self):
        v1 = np.array([1.0, 2.0])
        v2 = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            css_similarity(v1, v2)

    def test_nonnegative_output(self):
        v1 = np.array([1.0, -1.0, 0.5])
        v2 = np.array([-1.0, 1.0, 0.5])
        result = css_similarity(v1, v2)
        assert result >= 0.0

    def test_zero_vectors_safe(self):
        v1 = np.zeros(5)
        v2 = np.zeros(5)
        # Both zero norms → clips to valid range
        result = css_similarity(v1, v2)
        assert 0.0 <= result <= 1.0


# ─── fd_score ─────────────────────────────────────────────────────────────────

class TestFdScore:
    def test_equal_fd_gives_one(self):
        assert fd_score(1.5, 1.5) == pytest.approx(1.0)

    def test_large_diff_gives_small_score(self):
        score = fd_score(0.0, 5.0, sigma=0.5)
        assert score < 0.01

    def test_range(self):
        score = fd_score(1.5, 2.0, sigma=0.5)
        assert 0.0 < score <= 1.0

    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError):
            fd_score(1.0, 1.0, sigma=0.0)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError):
            fd_score(1.0, 1.0, sigma=-1.0)

    def test_symmetric(self):
        assert fd_score(1.2, 1.8, sigma=0.5) == pytest.approx(
            fd_score(1.8, 1.2, sigma=0.5)
        )

    def test_large_sigma_gives_high_score(self):
        score = fd_score(1.0, 2.0, sigma=100.0)
        assert score > 0.99


# ─── ifs_similarity ───────────────────────────────────────────────────────────

class TestIfsSimilarity:
    def test_identical_coefficients(self):
        c = np.array([0.5, 0.3, 0.2])
        assert ifs_similarity(c, c) == pytest.approx(1.0)

    def test_empty_gives_zero(self):
        assert ifs_similarity([], []) == pytest.approx(0.0)

    def test_range(self):
        a = np.array([1.0, 0.5, 0.0])
        b = np.array([0.0, 0.5, 1.0])
        result = ifs_similarity(a, b)
        assert 0.0 <= result <= 1.0

    def test_different_lengths_truncates(self):
        a = np.array([1.0, 0.5, 0.3, 0.1])
        b = np.array([1.0, 0.5])
        result = ifs_similarity(a, b)
        assert 0.0 <= result <= 1.0

    def test_one_empty(self):
        a = np.array([1.0, 0.5])
        b = np.array([])
        assert ifs_similarity(a, b) == pytest.approx(0.0)

    def test_nonnegative(self):
        a = np.array([1.0, -0.5])
        b = np.array([0.5, 1.0])
        result = ifs_similarity(a, b)
        assert result >= 0.0


# ─── compare_edges ────────────────────────────────────────────────────────────

class TestCompareEdges:
    def test_returns_edge_compare_result(self):
        a = make_edge_sig(0)
        b = make_edge_sig(1)
        r = compare_edges(a, b)
        assert isinstance(r, EdgeCompareResult)

    def test_same_edge_high_score(self):
        a = make_edge_sig(0)
        r = compare_edges(a, a)
        assert r.score > 0.5

    def test_score_in_range(self):
        a = make_edge_sig(0)
        b = make_edge_sig(1, fd=3.0)
        r = compare_edges(a, b)
        assert 0.0 <= r.score <= 1.0

    def test_correct_edge_ids(self):
        a = make_edge_sig(10)
        b = make_edge_sig(20)
        r = compare_edges(a, b)
        assert r.edge_id_a == 10
        assert r.edge_id_b == 20

    def test_custom_config(self):
        a = make_edge_sig(0)
        b = make_edge_sig(1)
        cfg = CompareConfig(w_dtw=1.0, w_css=0.0, w_fd=0.0, w_ifs=0.0)
        r = compare_edges(a, b, cfg=cfg)
        assert isinstance(r, EdgeCompareResult)

    def test_dtw_dist_nonneg(self):
        a = make_edge_sig(0)
        b = make_edge_sig(1, scale=2.0)
        r = compare_edges(a, b)
        assert r.dtw_dist >= 0.0


# ─── build_compat_matrix ──────────────────────────────────────────────────────

class TestBuildCompatMatrix:
    def test_shape(self):
        edges = [make_edge_sig(i) for i in range(3)]
        mat = build_compat_matrix(edges)
        assert mat.shape == (3, 3)

    def test_dtype_float32(self):
        edges = [make_edge_sig(i) for i in range(2)]
        mat = build_compat_matrix(edges)
        assert mat.dtype == np.float32

    def test_diagonal_one(self):
        edges = [make_edge_sig(i) for i in range(3)]
        mat = build_compat_matrix(edges)
        np.testing.assert_array_almost_equal(np.diag(mat), 1.0)

    def test_symmetric(self):
        edges = [make_edge_sig(i) for i in range(3)]
        mat = build_compat_matrix(edges)
        np.testing.assert_allclose(mat, mat.T, atol=1e-6)

    def test_single_edge(self):
        edges = [make_edge_sig(0)]
        mat = build_compat_matrix(edges)
        assert mat.shape == (1, 1)
        assert mat[0, 0] == pytest.approx(1.0)

    def test_values_in_range(self):
        edges = [make_edge_sig(i) for i in range(4)]
        mat = build_compat_matrix(edges)
        assert (mat >= 0.0).all()
        assert (mat <= 1.0).all()


# ─── top_k_matches ────────────────────────────────────────────────────────────

class TestTopKMatches:
    def test_returns_list(self):
        query = make_edge_sig(0)
        candidates = [make_edge_sig(i + 1) for i in range(5)]
        results = top_k_matches(query, candidates, k=3)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_sorted_by_score_desc(self):
        query = make_edge_sig(0)
        candidates = [make_edge_sig(i + 1) for i in range(5)]
        results = top_k_matches(query, candidates, k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_all_edge_compare_results(self):
        query = make_edge_sig(0)
        candidates = [make_edge_sig(i + 1) for i in range(3)]
        results = top_k_matches(query, candidates)
        assert all(isinstance(r, EdgeCompareResult) for r in results)

    def test_k_less_than_one_raises(self):
        query = make_edge_sig(0)
        candidates = [make_edge_sig(1)]
        with pytest.raises(ValueError):
            top_k_matches(query, candidates, k=0)

    def test_empty_candidates_raises(self):
        query = make_edge_sig(0)
        with pytest.raises(ValueError):
            top_k_matches(query, [], k=3)

    def test_k_larger_than_candidates_clipped(self):
        query = make_edge_sig(0)
        candidates = [make_edge_sig(1), make_edge_sig(2)]
        results = top_k_matches(query, candidates, k=10)
        assert len(results) == 2
