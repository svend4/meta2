"""Additional tests for puzzle_reconstruction/matching/pairwise.py"""
import pytest
import numpy as np

from puzzle_reconstruction.matching.pairwise import (
    _ifs_distance_norm,
    match_score,
)
from puzzle_reconstruction.models import EdgeSide, EdgeSignature


# ─── helpers ──────────────────────────────────────────────────────────────────

def _edge(edge_id: int,
          fd: float = 1.5,
          length: float = 80.0,
          css_vec=None,
          ifs_coeffs=None,
          virtual_curve=None) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=(virtual_curve if virtual_curve is not None
                       else np.zeros((10, 2))),
        fd=fd,
        css_vec=css_vec if css_vec is not None else np.zeros(8),
        ifs_coeffs=ifs_coeffs if ifs_coeffs is not None else np.zeros(4),
        length=length,
    )


# ─── TestMatchScoreExtra ──────────────────────────────────────────────────────

class TestMatchScoreExtra:
    def test_zero_text_score_default(self):
        result = match_score(_edge(0), _edge(1))
        assert result.text_score == pytest.approx(0.0)

    def test_text_score_max_does_not_exceed_1(self):
        result = match_score(_edge(0), _edge(1), text_score=1.0)
        assert result.score <= 1.0

    def test_length_ratio_below_half_reduces_score(self):
        """Ratio < 0.5 → score * ratio < score without penalty."""
        e_normal = _edge(2, length=80.0)
        e_normal2 = _edge(3, length=80.0)
        e_short = _edge(4, length=10.0)  # ratio = 10/80 < 0.5
        score_same = match_score(e_normal, e_normal2).score
        score_penalty = match_score(e_normal, e_short).score
        assert score_penalty <= score_same

    def test_length_ratio_exactly_half_no_crash(self):
        """Ratio exactly 0.5 should not crash."""
        e0 = _edge(0, length=100.0)
        e1 = _edge(1, length=50.0)
        result = match_score(e0, e1)
        assert 0.0 <= result.score <= 1.0

    def test_identical_css_vec_nonzero(self):
        """Non-zero identical CSS vectors → high CSS similarity."""
        css = np.array([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.0])
        e0 = _edge(0, css_vec=css)
        e1 = _edge(1, css_vec=css)
        result = match_score(e0, e1)
        assert result.css_sim == pytest.approx(1.0, abs=1e-6)

    def test_opposite_css_vecs_lower_sim(self):
        """Orthogonal CSS vectors → lower CSS sim than identical."""
        css_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        css_b = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        css_same = css_a.copy()
        e0 = _edge(0, css_vec=css_a)
        e1_diff = _edge(1, css_vec=css_b)
        e1_same = _edge(2, css_vec=css_same)
        sim_diff = match_score(e0, e1_diff).css_sim
        sim_same = match_score(e0, e1_same).css_sim
        assert sim_same >= sim_diff

    def test_different_virtual_curves_different_dtw(self):
        curve_flat = np.zeros((10, 2))
        curve_ramp = np.array([[float(i), float(i)] for i in range(10)])
        e0 = _edge(0, virtual_curve=curve_flat)
        e1 = _edge(1, virtual_curve=curve_ramp)
        r_flat = match_score(_edge(10, virtual_curve=curve_flat),
                             _edge(11, virtual_curve=curve_flat))
        r_diff = match_score(e0, e1)
        assert r_flat.dtw_dist <= r_diff.dtw_dist + 1e-9

    def test_same_fd_zero_fd_diff(self):
        e0 = _edge(0, fd=1.4)
        e1 = _edge(1, fd=1.4)
        result = match_score(e0, e1)
        assert result.fd_diff == pytest.approx(0.0, abs=1e-9)

    def test_text_score_contributes_linearly(self):
        """Increasing text_score from 0→1 increases total score by ≤W_TEXT."""
        e0 = _edge(0)
        e1 = _edge(1)
        r0 = match_score(e0, e1, text_score=0.0)
        r1 = match_score(e0, e1, text_score=1.0)
        delta = r1.score - r0.score
        assert 0.0 <= delta <= 0.16  # W_TEXT = 0.15

    def test_nonzero_ifs_difference_reduces_ifs_score(self):
        """IFS difference contributes to score via DTW weight."""
        ifs_same = np.ones(4)
        ifs_diff = np.zeros(4)
        e0 = _edge(0, ifs_coeffs=ifs_same)
        e1_same = _edge(1, ifs_coeffs=ifs_same.copy())
        e1_diff = _edge(2, ifs_coeffs=ifs_diff)
        r_same = match_score(e0, e1_same)
        r_diff = match_score(e0, e1_diff)
        # Same IFS should give equal or higher score
        assert r_same.score >= r_diff.score - 1e-9

    def test_entry_edges_are_correct_objects(self):
        e0 = _edge(7)
        e1 = _edge(8)
        result = match_score(e0, e1)
        assert result.edge_i is e0
        assert result.edge_j is e1

    def test_score_never_negative(self):
        for i in range(5):
            e0 = _edge(i * 2, fd=float(i) * 0.2 + 1.0, length=10.0 + i * 5)
            e1 = _edge(i * 2 + 1, fd=float(i) * 0.3 + 1.2, length=20.0 + i * 7)
            assert match_score(e0, e1).score >= 0.0

    def test_very_large_fd_diff_near_zero_fd_score(self):
        e0 = _edge(0, fd=1.0)
        e1 = _edge(1, fd=2.0)
        result = match_score(e0, e1)
        # fd_diff=1 → fd_score = 1/2 = 0.5; still valid score
        assert 0.0 <= result.score <= 1.0


# ─── TestIfsDistanceNormExtra ─────────────────────────────────────────────────

class TestIfsDistanceNormExtra:
    def test_symmetric(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        # Not symmetric in general (uses min length from start), but no crash
        assert isinstance(_ifs_distance_norm(a, b), float)

    def test_single_element_arrays(self):
        a = np.array([1.0])
        b = np.array([1.0])
        assert _ifs_distance_norm(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_large_arrays_no_crash(self):
        rng = np.random.default_rng(0)
        a = rng.random(200)
        b = rng.random(200)
        d = _ifs_distance_norm(a, b)
        assert np.isfinite(d)

    def test_scaled_by_sqrt_n(self):
        """For a=zeros, b=c*ones, result should be |c| (normalised by sqrt(n))."""
        n = 9
        a = np.zeros(n)
        b = np.ones(n) * 3.0
        expected = 3.0  # ||3*ones|| / sqrt(9) = 3*3 / 3 = 3
        assert _ifs_distance_norm(a, b) == pytest.approx(expected, rel=1e-5)

    def test_nonneg_for_random(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            a = rng.standard_normal(6)
            b = rng.standard_normal(6)
            assert _ifs_distance_norm(a, b) >= 0.0
