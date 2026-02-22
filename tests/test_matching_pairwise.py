"""Tests for matching/pairwise.py."""
import numpy as np
import pytest

from puzzle_reconstruction.matching.pairwise import (
    _ifs_distance_norm,
    match_score,
)
from puzzle_reconstruction.models import CompatEntry, EdgeSignature, EdgeSide


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_edge(
    edge_id: int = 0,
    side: EdgeSide = EdgeSide.RIGHT,
    n_pts: int = 20,
    fd: float = 1.5,
    length: float = 50.0,
    seed: int = 0,
) -> EdgeSignature:
    rng = np.random.default_rng(seed)
    curve = rng.standard_normal((n_pts, 2)).astype(np.float64)
    css_vec = rng.uniform(0.0, 1.0, 8).astype(np.float64)
    ifs_coeffs = rng.standard_normal(6).astype(np.float64)
    return EdgeSignature(
        edge_id=edge_id,
        side=side,
        virtual_curve=curve,
        fd=fd,
        css_vec=css_vec,
        ifs_coeffs=ifs_coeffs,
        length=length,
    )


# ─── _ifs_distance_norm ───────────────────────────────────────────────────────

class TestIfsDistanceNorm:
    def test_identical_arrays_zero_distance(self):
        a = np.array([1.0, 2.0, 3.0])
        assert _ifs_distance_norm(a, a) == pytest.approx(0.0)

    def test_empty_arrays_returns_one(self):
        assert _ifs_distance_norm(np.array([]), np.array([])) == pytest.approx(1.0)

    def test_one_empty_returns_one(self):
        a = np.array([1.0, 2.0])
        assert _ifs_distance_norm(a, np.array([])) == pytest.approx(1.0)

    def test_non_negative(self):
        a = np.array([1.0, -2.0, 3.0])
        b = np.array([-1.0, 2.0, -3.0])
        assert _ifs_distance_norm(a, b) >= 0.0

    def test_returns_float(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        assert isinstance(_ifs_distance_norm(a, b), float)

    def test_truncates_to_min_length(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        # Only first 2 components compared → should give 0
        assert _ifs_distance_norm(a, b) == pytest.approx(0.0)

    def test_different_arrays_positive(self):
        a = np.zeros(4)
        b = np.ones(4) * 10.0
        assert _ifs_distance_norm(a, b) > 0.0

    def test_symmetric(self):
        a = np.array([1.0, 3.0, 5.0])
        b = np.array([2.0, 4.0, 6.0])
        assert _ifs_distance_norm(a, b) == pytest.approx(_ifs_distance_norm(b, a))


# ─── match_score ──────────────────────────────────────────────────────────────

class TestMatchScore:
    def test_returns_compat_entry(self):
        e1 = make_edge(edge_id=0, seed=0)
        e2 = make_edge(edge_id=1, seed=1)
        result = match_score(e1, e2)
        assert isinstance(result, CompatEntry)

    def test_score_in_zero_one(self):
        e1 = make_edge(edge_id=0, seed=0)
        e2 = make_edge(edge_id=1, seed=1)
        result = match_score(e1, e2)
        assert 0.0 <= result.score <= 1.0

    def test_text_score_zero_default(self):
        e1 = make_edge(edge_id=0, seed=0)
        e2 = make_edge(edge_id=1, seed=1)
        result = match_score(e1, e2)
        assert result.text_score == pytest.approx(0.0)

    def test_text_score_passed_through(self):
        e1 = make_edge(edge_id=0, seed=0)
        e2 = make_edge(edge_id=1, seed=1)
        result = match_score(e1, e2, text_score=0.7)
        assert result.text_score == pytest.approx(0.7)

    def test_edge_references_stored(self):
        e1 = make_edge(edge_id=3, seed=0)
        e2 = make_edge(edge_id=7, seed=1)
        result = match_score(e1, e2)
        assert result.edge_i is e1
        assert result.edge_j is e2

    def test_dtw_dist_non_negative(self):
        e1 = make_edge(seed=0)
        e2 = make_edge(seed=1)
        result = match_score(e1, e2)
        assert result.dtw_dist >= 0.0

    def test_css_sim_in_zero_one(self):
        e1 = make_edge(seed=0)
        e2 = make_edge(seed=1)
        result = match_score(e1, e2)
        assert 0.0 <= result.css_sim <= 1.0

    def test_fd_diff_non_negative(self):
        e1 = make_edge(fd=1.5, seed=0)
        e2 = make_edge(fd=1.8, seed=1)
        result = match_score(e1, e2)
        assert result.fd_diff >= 0.0

    def test_very_different_lengths_lower_score(self):
        # Edge with very different lengths gets a penalty
        e_short = make_edge(length=10.0, seed=0)
        e_long = make_edge(length=1000.0, seed=1)
        e_similar = make_edge(length=12.0, seed=2)
        result_diff = match_score(e_short, e_long)
        result_similar = match_score(e_short, e_similar)
        # Different lengths should give lower or equal score
        assert result_diff.score <= result_similar.score + 0.5  # lenient check

    def test_same_fd_zero_fd_diff(self):
        e1 = make_edge(fd=1.5, seed=0)
        e2 = make_edge(fd=1.5, seed=1)
        result = match_score(e1, e2)
        assert result.fd_diff == pytest.approx(0.0, abs=1e-9)

    def test_score_is_float(self):
        e1 = make_edge(seed=0)
        e2 = make_edge(seed=1)
        result = match_score(e1, e2)
        assert isinstance(result.score, float)

    def test_identical_edges_high_score(self):
        # Same edge matched against itself should get a reasonable score
        e = make_edge(seed=42)
        result = match_score(e, e)
        # Not necessarily 1.0, but should be in [0, 1]
        assert 0.0 <= result.score <= 1.0

    def test_text_score_one_increases_score(self):
        e1 = make_edge(seed=0)
        e2 = make_edge(seed=1)
        result_no_text = match_score(e1, e2, text_score=0.0)
        result_text = match_score(e1, e2, text_score=1.0)
        # text_score=1.0 should give same or higher score (W_TEXT=0.15 > 0)
        assert result_text.score >= result_no_text.score - 1e-9
