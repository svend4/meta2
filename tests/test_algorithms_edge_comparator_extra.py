"""Extra tests for puzzle_reconstruction/algorithms/edge_comparator.py."""
from __future__ import annotations

import numpy as np
import pytest

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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _curve(n=10, scale=1.0):
    t = np.linspace(0, np.pi, n)
    return np.column_stack([np.cos(t) * scale, np.sin(t) * scale]).astype(float)


def _sig(edge_id=0, n=8, fd=1.5, scale=1.0):
    curve = _curve(n, scale)
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.RIGHT,
        virtual_curve=curve,
        fd=fd,
        css_vec=np.ones(8) / 8.0,
        ifs_coeffs=np.ones(5) * 0.5,
        length=float(n),
    )


def _result(score=0.5, dtw=0.0, css=0.5, fd=0.0, ifs=0.5):
    return EdgeCompareResult(
        edge_id_a=0, edge_id_b=1,
        dtw_dist=dtw, css_sim=css, fd_diff=fd,
        ifs_sim=ifs, score=score,
    )


# ─── CompareConfig (extra) ────────────────────────────────────────────────────

class TestCompareConfigExtra:
    def test_large_weights_ok(self):
        cfg = CompareConfig(w_dtw=10.0, w_css=5.0, w_fd=3.0, w_ifs=2.0)
        assert cfg.total_weight == pytest.approx(20.0)

    def test_zero_dtw_band_ok(self):
        cfg = CompareConfig(dtw_band=0)
        assert cfg.dtw_band == 0

    def test_positive_dtw_band_ok(self):
        cfg = CompareConfig(dtw_band=5)
        assert cfg.dtw_band == 5

    def test_large_fd_sigma_ok(self):
        cfg = CompareConfig(fd_sigma=100.0)
        assert cfg.fd_sigma == pytest.approx(100.0)

    def test_all_zero_except_one(self):
        cfg = CompareConfig(w_dtw=1.0, w_css=0.0, w_fd=0.0, w_ifs=0.0)
        assert cfg.total_weight == pytest.approx(1.0)

    def test_independent_instances(self):
        c1 = CompareConfig(w_dtw=0.4)
        c2 = CompareConfig(w_dtw=0.8)
        assert c1.w_dtw != c2.w_dtw


# ─── EdgeCompareResult (extra) ────────────────────────────────────────────────

class TestEdgeCompareResultExtra:
    def test_pair_key_same_ids(self):
        r = _result()
        assert r.pair_key == (0, 1)

    def test_score_one_ok(self):
        r = _result(score=1.0, css=1.0, ifs=1.0)
        assert r.score == pytest.approx(1.0)

    def test_dtw_zero_ok(self):
        r = _result(dtw=0.0)
        assert r.dtw_dist == pytest.approx(0.0)

    def test_css_sim_zero_ok(self):
        r = _result(css=0.0)
        assert r.css_sim == pytest.approx(0.0)

    def test_ifs_sim_one_ok(self):
        r = _result(ifs=1.0)
        assert r.ifs_sim == pytest.approx(1.0)

    def test_large_dtw_dist_ok(self):
        r = EdgeCompareResult(
            edge_id_a=0, edge_id_b=1,
            dtw_dist=1000.0, css_sim=0.0, fd_diff=0.0,
            ifs_sim=0.0, score=0.0,
        )
        assert r.dtw_dist == pytest.approx(1000.0)

    def test_pair_key_higher_first_when_reversed(self):
        r = EdgeCompareResult(
            edge_id_a=7, edge_id_b=2,
            dtw_dist=0.0, css_sim=1.0, fd_diff=0.0,
            ifs_sim=1.0, score=1.0,
        )
        assert r.pair_key == (2, 7)


# ─── dtw_distance (extra) ─────────────────────────────────────────────────────

class TestDtwDistanceExtra:
    def test_symmetric(self):
        a = _curve(8, scale=1.0)
        b = _curve(8, scale=2.0)
        assert dtw_distance(a, b) == pytest.approx(dtw_distance(b, a), rel=1e-6)

    def test_same_length_curves(self):
        a = _curve(10)
        b = _curve(10, scale=1.5)
        dist = dtw_distance(a, b)
        assert dist >= 0.0

    def test_large_band_no_raises(self):
        a = _curve(8)
        dist = dtw_distance(a, a, band=100)
        assert dist == pytest.approx(0.0, abs=1e-9)

    def test_scaled_further_is_larger(self):
        a = _curve(10)
        b1 = _curve(10, scale=1.1)
        b2 = _curve(10, scale=5.0)
        assert dtw_distance(a, b2) > dtw_distance(a, b1)

    def test_single_point_each(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[3.0, 4.0]])
        dist = dtw_distance(a, b)
        assert dist == pytest.approx(5.0, abs=1e-9)


# ─── css_similarity (extra) ───────────────────────────────────────────────────

class TestCssSimilarityExtra:
    def test_proportional_vectors_give_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert css_similarity(v, v * 2.0) == pytest.approx(1.0, abs=1e-6)

    def test_single_element(self):
        v = np.array([1.0])
        result = css_similarity(v, v)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_negative_elements_ok(self):
        v1 = np.array([-1.0, 2.0])
        v2 = np.array([-1.0, 2.0])
        assert css_similarity(v1, v2) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors_zero(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        result = css_similarity(v1, v2)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_zero(self):
        v = np.array([1.0, 1.0])
        result = css_similarity(v, -v)
        assert result == pytest.approx(0.0, abs=1e-6)


# ─── fd_score (extra) ─────────────────────────────────────────────────────────

class TestFdScoreExtra:
    def test_small_diff_gives_high_score(self):
        score = fd_score(1.0, 1.01, sigma=0.5)
        assert score > 0.99

    def test_large_diff_gives_low_score(self):
        score = fd_score(0.0, 10.0, sigma=0.5)
        assert score < 0.01

    def test_result_always_positive(self):
        for a, b in [(0.0, 0.0), (1.0, 2.0), (5.0, 3.0)]:
            assert fd_score(a, b) > 0.0

    def test_large_sigma_close_to_one(self):
        score = fd_score(0.0, 100.0, sigma=1e6)
        assert score > 0.99

    def test_small_sigma_drops_fast(self):
        s1 = fd_score(0.0, 1.0, sigma=0.1)
        s2 = fd_score(0.0, 1.0, sigma=1.0)
        assert s1 < s2


# ─── ifs_similarity (extra) ───────────────────────────────────────────────────

class TestIfsSimilarityExtra:
    def test_single_element_same(self):
        assert ifs_similarity([1.0], [1.0]) == pytest.approx(1.0, abs=1e-6)

    def test_larger_b_truncates_ok(self):
        a = [1.0, 0.5]
        b = [1.0, 0.5, 0.3]
        result = ifs_similarity(a, b)
        assert 0.0 <= result <= 1.0

    def test_all_ones_same(self):
        a = np.ones(6)
        result = ifs_similarity(a, a)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_nonneg_for_opposite(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert ifs_similarity(a, b) >= 0.0

    def test_large_arrays(self):
        a = np.random.rand(20)
        b = np.random.rand(20)
        result = ifs_similarity(a, b)
        assert 0.0 <= result <= 1.0


# ─── compare_edges (extra) ────────────────────────────────────────────────────

class TestCompareEdgesExtra:
    def test_dtw_only_config(self):
        a = _sig(0)
        b = _sig(1)
        cfg = CompareConfig(w_dtw=1.0, w_css=0.0, w_fd=0.0, w_ifs=0.0)
        r = compare_edges(a, b, cfg=cfg)
        assert 0.0 <= r.score <= 1.0

    def test_css_only_config(self):
        a = _sig(0)
        b = _sig(1)
        cfg = CompareConfig(w_dtw=0.0, w_css=1.0, w_fd=0.0, w_ifs=0.0)
        r = compare_edges(a, b, cfg=cfg)
        assert 0.0 <= r.score <= 1.0

    def test_fd_diff_nonneg(self):
        a = _sig(0, fd=1.0)
        b = _sig(1, fd=2.0)
        r = compare_edges(a, b)
        assert r.fd_diff >= 0.0

    def test_same_edge_css_sim_one(self):
        a = _sig(0)
        r = compare_edges(a, a)
        assert r.css_sim == pytest.approx(1.0, abs=1e-6)

    def test_same_edge_ifs_sim_one(self):
        a = _sig(0)
        r = compare_edges(a, a)
        assert r.ifs_sim == pytest.approx(1.0, abs=1e-6)

    def test_score_finite(self):
        a = _sig(0)
        b = _sig(1, fd=10.0)
        r = compare_edges(a, b)
        assert np.isfinite(r.score)


# ─── build_compat_matrix (extra) ──────────────────────────────────────────────

class TestBuildCompatMatrixExtra:
    def test_4_edges_shape(self):
        edges = [_sig(i) for i in range(4)]
        mat = build_compat_matrix(edges)
        assert mat.shape == (4, 4)

    def test_off_diagonal_in_0_1(self):
        edges = [_sig(i) for i in range(3)]
        mat = build_compat_matrix(edges)
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert 0.0 <= mat[i, j] <= 1.0 + 1e-6

    def test_diagonal_values_one(self):
        edges = [_sig(i) for i in range(2)]
        mat = build_compat_matrix(edges)
        assert mat[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert mat[1, 1] == pytest.approx(1.0, abs=1e-5)

    def test_different_fd_values(self):
        edges = [_sig(i, fd=float(i + 1)) for i in range(3)]
        mat = build_compat_matrix(edges)
        assert mat.shape == (3, 3)

    def test_large_matrix_no_error(self):
        edges = [_sig(i) for i in range(6)]
        mat = build_compat_matrix(edges)
        assert mat.shape == (6, 6)


# ─── top_k_matches (extra) ────────────────────────────────────────────────────

class TestTopKMatchesExtra:
    def test_k_equals_1(self):
        q = _sig(0)
        cands = [_sig(i + 1) for i in range(5)]
        result = top_k_matches(q, cands, k=1)
        assert len(result) == 1

    def test_top_result_has_highest_score(self):
        q = _sig(0)
        cands = [_sig(i + 1) for i in range(5)]
        result = top_k_matches(q, cands, k=5)
        for r in result[1:]:
            assert result[0].score >= r.score

    def test_query_matches_itself_in_top(self):
        q = _sig(0)
        cands = [_sig(i + 1, scale=float(i + 2)) for i in range(4)]
        # If query is also in candidates list (same params)
        cands_with_self = [_sig(0)] + cands
        result = top_k_matches(q, cands_with_self, k=1)
        assert len(result) == 1

    def test_all_same_candidates(self):
        q = _sig(0)
        cands = [_sig(i + 1) for i in range(3)]
        result = top_k_matches(q, cands, k=3)
        assert len(result) == 3

    def test_scores_decreasing(self):
        q = _sig(0)
        cands = [_sig(i + 1, fd=float(i)) for i in range(5)]
        result = top_k_matches(q, cands, k=5)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)
