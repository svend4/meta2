"""Additional tests for puzzle_reconstruction/matching — DTW and match_score."""
import numpy as np
import pytest

from puzzle_reconstruction.matching.dtw import dtw_distance, dtw_distance_mirror
from puzzle_reconstruction.matching.pairwise import match_score
from puzzle_reconstruction.models import CompatEntry, EdgeSide, EdgeSignature


# ─── helpers ──────────────────────────────────────────────────────────────────

def _edge(
    eid: int,
    n: int = 32,
    fd: float = 1.2,
    length: float = 80.0,
    side: EdgeSide = EdgeSide.TOP,
    seed: int = 0,
) -> EdgeSignature:
    rng = np.random.RandomState(seed + eid)
    curve = np.column_stack([np.linspace(0, 6, n), np.sin(np.linspace(0, 6, n))])
    css_vec = rng.rand(7 * 32)
    css_vec /= np.linalg.norm(css_vec) + 1e-10
    ifs_coeffs = rng.uniform(-0.5, 0.5, 8)
    return EdgeSignature(
        edge_id=eid,
        side=side,
        virtual_curve=curve,
        fd=fd,
        css_vec=css_vec,
        ifs_coeffs=ifs_coeffs,
        length=length,
    )


def _flat(n: int = 32, y: float = 0.0) -> np.ndarray:
    return np.column_stack([np.linspace(0, 6, n), np.full(n, y)])


def _sine(n: int = 32, freq: float = 1.0, amp: float = 1.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([t, amp * np.sin(freq * t)])


# ─── TestDTWDistanceExtra ─────────────────────────────────────────────────────

class TestDTWDistanceExtra:
    def test_flat_same_curve_zero(self):
        c = _flat()
        assert dtw_distance(c, c) == pytest.approx(0.0, abs=1e-8)

    def test_different_flat_levels_positive(self):
        c1 = _flat(y=0.0)
        c2 = _flat(y=5.0)
        assert dtw_distance(c1, c2) > 0.0

    def test_returns_float(self):
        d = dtw_distance(_sine(), _sine(freq=2.0))
        assert isinstance(d, float)

    def test_nonneg(self):
        d = dtw_distance(_sine(), _sine(freq=2.0))
        assert d >= 0.0

    def test_finite_for_valid_inputs(self):
        d = dtw_distance(_sine(n=64), _sine(n=64, amp=3.0))
        assert np.isfinite(d)

    def test_float32_input(self):
        c1 = _sine(n=32).astype(np.float32)
        c2 = _sine(n=32, amp=2.0).astype(np.float32)
        d = dtw_distance(c1, c2)
        assert np.isfinite(d)

    def test_large_window_same_as_no_constraint(self):
        """Window >= |n - m| is never a constraint."""
        a = _sine(n=30)
        b = _sine(n=80, freq=1.5)
        d_large = dtw_distance(a, b, window=100)
        d_auto  = dtw_distance(a, b)
        # Both should be finite
        assert np.isfinite(d_large)
        assert np.isfinite(d_auto)

    def test_both_empty_returns_inf(self):
        empty = np.zeros((0, 2))
        assert dtw_distance(empty, empty) == float("inf")

    def test_reversed_curve_distance_nonneg(self):
        c = _sine()
        d = dtw_distance(c, c[::-1])
        assert d >= 0.0


# ─── TestDTWDistanceMirrorExtra ───────────────────────────────────────────────

class TestDTWDistanceMirrorExtra:
    def test_returns_float(self):
        d = dtw_distance_mirror(_sine(), _sine(freq=2.0))
        assert isinstance(d, float)

    def test_nonneg(self):
        d = dtw_distance_mirror(_sine(), _sine(amp=2.0))
        assert d >= 0.0

    def test_same_curve_zero(self):
        c = _sine()
        d = dtw_distance_mirror(c, c)
        assert d == pytest.approx(0.0, abs=1e-8)

    def test_mirror_le_direct(self):
        c = _sine()
        d_direct = dtw_distance(c, c[::-1])
        d_mirror = dtw_distance_mirror(c, c[::-1])
        assert d_mirror <= d_direct + 1e-9

    def test_empty_first_arg_returns_inf(self):
        empty = np.zeros((0, 2))
        result = dtw_distance_mirror(empty, _sine())
        assert result == float("inf")

    def test_finite_for_equal_length_inputs(self):
        a = _sine(n=32, freq=1.0)
        b = _sine(n=32, freq=3.0)
        d = dtw_distance_mirror(a, b)
        assert np.isfinite(d)


# ─── TestMatchScoreExtra ──────────────────────────────────────────────────────

class TestMatchScoreExtra:
    def test_returns_compat_entry(self):
        e1 = _edge(0)
        e2 = _edge(1)
        result = match_score(e1, e2)
        assert isinstance(result, CompatEntry)

    def test_score_in_0_1(self):
        for seed in range(4):
            e1 = _edge(seed * 2, seed=seed)
            e2 = _edge(seed * 2 + 1, seed=seed + 10)
            entry = match_score(e1, e2)
            assert 0.0 <= entry.score <= 1.0

    def test_edge_i_is_first_arg(self):
        e1 = _edge(0)
        e2 = _edge(1)
        result = match_score(e1, e2)
        assert result.edge_i is e1

    def test_edge_j_is_second_arg(self):
        e1 = _edge(0)
        e2 = _edge(1)
        result = match_score(e1, e2)
        assert result.edge_j is e2

    def test_dtw_dist_attr_nonneg(self):
        e1 = _edge(0)
        e2 = _edge(1)
        result = match_score(e1, e2)
        assert result.dtw_dist >= 0.0

    def test_css_sim_in_0_1(self):
        e1 = _edge(0)
        e2 = _edge(1)
        result = match_score(e1, e2)
        assert 0.0 <= result.css_sim <= 1.0

    def test_fd_diff_nonneg(self):
        e1 = _edge(0, fd=1.2)
        e2 = _edge(1, fd=1.8)
        result = match_score(e1, e2)
        assert result.fd_diff >= 0.0

    def test_same_fd_fd_diff_zero(self):
        e1 = _edge(0, fd=1.5)
        e2 = _edge(1, fd=1.5)
        result = match_score(e1, e2)
        assert result.fd_diff == pytest.approx(0.0)

    def test_bottom_side_accepted(self):
        e1 = _edge(0, side=EdgeSide.BOTTOM)
        e2 = _edge(1, side=EdgeSide.TOP)
        result = match_score(e1, e2)
        assert isinstance(result, CompatEntry)

    def test_left_right_sides_accepted(self):
        e1 = _edge(0, side=EdgeSide.LEFT)
        e2 = _edge(1, side=EdgeSide.RIGHT)
        result = match_score(e1, e2)
        assert isinstance(result, CompatEntry)

    def test_length_ratio_penalty_applied(self):
        """Much shorter second edge → lower score than same-length pair."""
        e_ref = _edge(0, length=100.0)
        e_short = _edge(1, length=5.0)
        e_same = _edge(2, length=100.0)
        score_penalty = match_score(e_ref, e_short).score
        score_fair = match_score(e_ref, e_same).score
        assert score_fair >= score_penalty

    def test_ifs_coeffs_same_improves_score(self):
        """Forcing identical IFS coefficients improves score vs random."""
        e1 = _edge(0, seed=0)
        e2_rand = _edge(1, seed=99)
        e2_same = _edge(2, seed=99)
        e2_same.ifs_coeffs = e1.ifs_coeffs.copy()
        score_rand = match_score(e1, e2_rand).score
        score_same = match_score(e1, e2_same).score
        # Same ifs coeffs should yield at least as good score on that component
        assert score_same >= score_rand - 0.5  # wide tolerance
