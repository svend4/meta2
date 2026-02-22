"""Расширенные тесты для puzzle_reconstruction/matching/pairwise.py."""
import numpy as np
import pytest

from puzzle_reconstruction.matching.pairwise import (
    _ifs_distance_norm,
    match_score,
)
from puzzle_reconstruction.models import (
    CompatEntry,
    EdgeSide,
    EdgeSignature,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _edge(edge_id: int,
          fd: float = 1.5,
          length: float = 80.0,
          css_vec=None,
          ifs_coeffs=None,
          virtual_curve=None,
          ) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=virtual_curve if virtual_curve is not None else np.zeros((10, 2)),
        fd=fd,
        css_vec=css_vec if css_vec is not None else np.zeros(8),
        ifs_coeffs=ifs_coeffs if ifs_coeffs is not None else np.zeros(4),
        length=length,
    )


# ─── TestMatchScore ───────────────────────────────────────────────────────────

class TestMatchScore:
    def test_returns_compat_entry(self):
        result = match_score(_edge(0), _edge(1))
        assert isinstance(result, CompatEntry)

    def test_score_in_0_to_1(self):
        result = match_score(_edge(0), _edge(1))
        assert 0.0 <= result.score <= 1.0

    def test_edges_stored_in_entry(self):
        e0 = _edge(0)
        e1 = _edge(1)
        result = match_score(e0, e1)
        assert result.edge_i is e0
        assert result.edge_j is e1

    def test_dtw_dist_nonneg(self):
        result = match_score(_edge(0), _edge(1))
        assert result.dtw_dist >= 0.0

    def test_css_sim_in_range(self):
        result = match_score(_edge(0), _edge(1))
        assert 0.0 <= result.css_sim <= 1.0

    def test_fd_diff_nonneg(self):
        result = match_score(_edge(0), _edge(1))
        assert result.fd_diff >= 0.0

    def test_text_score_zero_default(self):
        result = match_score(_edge(0), _edge(1))
        assert result.text_score == pytest.approx(0.0)

    def test_text_score_passed_through(self):
        result = match_score(_edge(0), _edge(1), text_score=0.7)
        assert result.text_score == pytest.approx(0.7)

    def test_score_is_float(self):
        result = match_score(_edge(0), _edge(1))
        assert isinstance(result.score, float)

    def test_same_fd_higher_score_than_different(self):
        e0 = _edge(0, fd=1.5)
        e1_same = _edge(1, fd=1.5)
        e2_diff = _edge(2, fd=1.9)
        assert match_score(e0, e1_same).score >= match_score(e0, e2_diff).score

    def test_same_length_positive_score(self):
        e0 = _edge(0, length=80.0)
        e1 = _edge(1, length=80.0)
        assert match_score(e0, e1).score > 0.0

    def test_very_different_length_lower_score(self):
        e0 = _edge(0, length=80.0)
        e1_short = _edge(1, length=5.0)  # ratio < 0.5 → penalty
        e1_same = _edge(2, length=80.0)
        score_short = match_score(e0, e1_short).score
        score_same = match_score(e0, e1_same).score
        assert score_same >= score_short

    def test_score_clipped_to_1(self):
        result = match_score(_edge(0), _edge(1), text_score=1.0)
        assert result.score <= 1.0

    def test_score_clipped_to_0(self):
        result = match_score(_edge(0), _edge(1))
        assert result.score >= 0.0

    def test_high_text_score_increases_total(self):
        e0 = _edge(0)
        e1 = _edge(1)
        r0 = match_score(e0, e1, text_score=0.0)
        r1 = match_score(e0, e1, text_score=1.0)
        assert r1.score >= r0.score

    def test_fd_diff_matches_abs_fd_difference(self):
        e0 = _edge(0, fd=1.3)
        e1 = _edge(1, fd=1.7)
        result = match_score(e0, e1)
        assert result.fd_diff == pytest.approx(abs(1.3 - 1.7), abs=1e-6)

    def test_deterministic(self):
        e0 = _edge(0)
        e1 = _edge(1)
        r1 = match_score(e0, e1)
        r2 = match_score(e0, e1)
        assert r1.score == r2.score

    def test_dtw_dist_stored(self):
        result = match_score(_edge(0), _edge(1))
        assert isinstance(result.dtw_dist, float)

    def test_fd_diff_stored(self):
        e0 = _edge(0, fd=1.2)
        e1 = _edge(1, fd=1.8)
        result = match_score(e0, e1)
        assert result.fd_diff == pytest.approx(0.6, abs=1e-6)

    def test_compat_entry_fields_complete(self):
        result = match_score(_edge(0), _edge(1))
        assert hasattr(result, 'score')
        assert hasattr(result, 'dtw_dist')
        assert hasattr(result, 'css_sim')
        assert hasattr(result, 'fd_diff')
        assert hasattr(result, 'text_score')


# ─── TestIfsDistanceNorm ──────────────────────────────────────────────────────

class TestIfsDistanceNorm:
    def test_returns_float(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert isinstance(_ifs_distance_norm(a, b), float)

    def test_identical_arrays_near_zero(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        assert _ifs_distance_norm(a, a.copy()) == pytest.approx(0.0, abs=1e-9)

    def test_nonneg(self):
        a = np.random.randn(5)
        b = np.random.randn(5)
        assert _ifs_distance_norm(a, b) >= 0.0

    def test_both_empty_returns_1(self):
        assert _ifs_distance_norm(np.array([]), np.array([])) == pytest.approx(1.0)

    def test_one_empty_returns_1(self):
        assert _ifs_distance_norm(np.array([1.0, 2.0]), np.array([])) == pytest.approx(1.0)

    def test_different_lengths_uses_shorter(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.0, 2.0])
        result = _ifs_distance_norm(a, b)
        assert result >= 0.0

    def test_larger_diff_larger_distance(self):
        a = np.zeros(4)
        b_near = np.ones(4) * 0.1
        b_far = np.ones(4) * 10.0
        assert _ifs_distance_norm(a, b_far) > _ifs_distance_norm(a, b_near)

    def test_float_type(self):
        assert isinstance(_ifs_distance_norm(np.ones(3), np.ones(3)), float)
