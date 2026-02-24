"""Extra tests for puzzle_reconstruction/matching/boundary_matcher.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.matching.boundary_matcher import (
    BoundaryMatch,
    extract_boundary_points,
    hausdorff_distance,
    chamfer_distance,
    frechet_approx,
    score_boundary_pair,
    match_boundary_pair,
    batch_match_boundaries,
)


# ─── helpers ────────────────────────────────────────────────────────────────

def _square(x0=0, y0=0, size=100):
    pts = [
        (x0, y0), (x0 + size, y0),
        (x0 + size, y0 + size), (x0, y0 + size),
    ]
    return np.array(pts, dtype=np.float32)


def _circle(cx=50.0, cy=50.0, r=30.0, n=64):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack(
        [cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1
    ).astype(np.float32)


def _line(n=20):
    return np.column_stack(
        [np.linspace(0, 100, n), np.zeros(n)]
    ).astype(np.float64)


# ─── BoundaryMatch (extra) ───────────────────────────────────────────────────

class TestBoundaryMatchExtra:
    def test_total_score_in_range(self):
        bm = BoundaryMatch(0, 1, 0, 2, 0.7, 0.8, 0.6, 0.7)
        assert 0.0 <= bm.total_score <= 1.0

    def test_params_default_empty(self):
        bm = BoundaryMatch(0, 1, 0, 2, 0.5, 0.5, 0.5, 0.5)
        assert bm.params == {}

    def test_params_stored(self):
        bm = BoundaryMatch(0, 1, 0, 2, 0.5, 0.5, 0.5, 0.5, params={"k": 3})
        assert bm.params["k"] == 3

    def test_idx1_idx2_stored(self):
        bm = BoundaryMatch(10, 20, 0, 1, 0.5, 0.5, 0.5, 0.5)
        assert bm.idx1 == 10
        assert bm.idx2 == 20

    def test_side1_side2_stored(self):
        bm = BoundaryMatch(0, 1, 3, 1, 0.5, 0.5, 0.5, 0.5)
        assert bm.side1 == 3
        assert bm.side2 == 1


# ─── extract_boundary_points (extra) ─────────────────────────────────────────

class TestExtractBoundaryPointsExtra:
    def test_n_points_default_positive(self):
        cnt = _square()
        pts = extract_boundary_points(cnt, side=0)
        assert pts.shape[0] > 0

    def test_float32_input_accepted(self):
        cnt = _square().astype(np.float32)
        pts = extract_boundary_points(cnt, side=1, n_points=10)
        assert pts.shape == (10, 2)

    def test_float64_input_accepted(self):
        cnt = _square().astype(np.float64)
        pts = extract_boundary_points(cnt, side=2, n_points=10)
        assert pts.shape == (10, 2)

    def test_n1_2_format_accepted(self):
        cnt = _square().reshape(-1, 1, 2)
        pts = extract_boundary_points(cnt, side=0, n_points=5)
        assert pts.shape == (5, 2)

    def test_large_contour(self):
        cnt = _circle(n=200)
        pts = extract_boundary_points(cnt, side=0, n_points=30)
        assert pts.shape == (30, 2)

    def test_side_0_top_pts(self):
        cnt = _square(0, 0, 100)
        pts = extract_boundary_points(cnt, side=0, n_points=4)
        assert pts[:, 1].max() <= 50.0

    def test_side_2_bottom_pts(self):
        cnt = _square(0, 0, 100)
        pts = extract_boundary_points(cnt, side=2, n_points=4)
        assert pts[:, 1].min() >= 50.0


# ─── hausdorff_distance (extra) ──────────────────────────────────────────────

class TestHausdorffDistanceExtra:
    def test_returns_float(self):
        pts = _line(5)
        assert isinstance(hausdorff_distance(pts, pts), float)

    def test_identical_zero(self):
        pts = _line(10)
        assert hausdorff_distance(pts, pts) == pytest.approx(0.0)

    def test_nonneg(self):
        pts1 = _line(8)
        pts2 = _line(8) + np.array([[5.0, 0.0]])
        assert hausdorff_distance(pts1, pts2) >= 0.0

    def test_known_single_point(self):
        p1 = np.array([[0.0, 0.0]])
        p2 = np.array([[3.0, 4.0]])
        assert hausdorff_distance(p1, p2) == pytest.approx(5.0)

    def test_symmetric(self):
        pts1 = _line(10)
        pts2 = _line(10) + np.array([[3.0, 3.0]])
        assert hausdorff_distance(pts1, pts2) == pytest.approx(
            hausdorff_distance(pts2, pts1), abs=1e-6
        )

    def test_both_empty_zero(self):
        emp = np.empty((0, 2), dtype=np.float64)
        assert hausdorff_distance(emp, emp) == pytest.approx(0.0)


# ─── chamfer_distance (extra) ────────────────────────────────────────────────

class TestChamferDistanceExtra:
    def test_returns_float(self):
        pts = _line(5)
        assert isinstance(chamfer_distance(pts, pts), float)

    def test_identical_zero(self):
        pts = _line(10)
        assert chamfer_distance(pts, pts) == pytest.approx(0.0)

    def test_nonneg(self):
        pts1 = _line(8)
        pts2 = _line(8) + np.array([[5.0, 0.0]])
        assert chamfer_distance(pts1, pts2) >= 0.0

    def test_symmetric(self):
        pts1 = _line(10)
        pts2 = _line(10) + np.array([[2.0, 3.0]])
        assert chamfer_distance(pts1, pts2) == pytest.approx(
            chamfer_distance(pts2, pts1), abs=1e-5
        )

    def test_le_hausdorff(self):
        rng = np.random.default_rng(42)
        pts1 = rng.random((20, 2)) * 50
        pts2 = rng.random((20, 2)) * 50
        assert chamfer_distance(pts1, pts2) <= hausdorff_distance(pts1, pts2) + 1e-6

    def test_single_point_known(self):
        p1 = np.array([[0.0, 0.0]])
        p2 = np.array([[3.0, 4.0]])
        assert chamfer_distance(p1, p2) == pytest.approx(5.0)


# ─── frechet_approx (extra) ──────────────────────────────────────────────────

class TestFrechetApproxExtra:
    def test_returns_float(self):
        pts = _line(5)
        assert isinstance(frechet_approx(pts, pts), float)

    def test_identical_zero(self):
        pts = _line(10)
        assert frechet_approx(pts, pts) == pytest.approx(0.0)

    def test_nonneg(self):
        pts1 = _line(8)
        pts2 = _line(8) + np.array([[2.0, 0.0]])
        assert frechet_approx(pts1, pts2) >= 0.0

    def test_monotone_with_offset(self):
        base = _line(10)
        near = base + np.array([[1.0, 0.0]])
        far = base + np.array([[20.0, 0.0]])
        assert frechet_approx(base, near) < frechet_approx(base, far)

    def test_one_empty_zero(self):
        pts = _line(5)
        emp = np.empty((0, 2), dtype=np.float64)
        assert frechet_approx(emp, pts) == pytest.approx(0.0)


# ─── score_boundary_pair (extra) ─────────────────────────────────────────────

class TestScoreBoundaryPairExtra:
    def test_returns_4_values(self):
        pts = _line(10)
        result = score_boundary_pair(pts, pts)
        assert len(result) == 4

    def test_identical_all_ones(self):
        pts = _line(10)
        h, c, f, total = score_boundary_pair(pts, pts)
        assert h == pytest.approx(1.0)
        assert total == pytest.approx(1.0)

    def test_all_values_in_range(self):
        rng = np.random.default_rng(0)
        pts1 = rng.random((15, 2)) * 50
        pts2 = rng.random((15, 2)) * 50
        for s in score_boundary_pair(pts1, pts2):
            assert 0.0 <= s <= 1.0

    def test_custom_weights_sum_one(self):
        pts = _line(10)
        h, c, f, total = score_boundary_pair(pts, pts, weights=(0.5, 0.3, 0.2))
        assert total == pytest.approx(1.0)

    def test_large_distance_lower_score(self):
        pts1 = _line(10)
        pts_close = pts1 + np.array([[5.0, 0.0]])
        pts_far = pts1 + np.array([[100.0, 0.0]])
        _, _, _, total_close = score_boundary_pair(pts1, pts_close, max_dist=200.0)
        _, _, _, total_far = score_boundary_pair(pts1, pts_far, max_dist=200.0)
        assert total_close >= total_far


# ─── match_boundary_pair (extra) ─────────────────────────────────────────────

class TestMatchBoundaryPairExtra:
    def test_returns_boundary_match(self):
        cnt = _square()
        bm = match_boundary_pair(cnt, cnt)
        assert isinstance(bm, BoundaryMatch)

    def test_identical_high_score(self):
        cnt = _circle()
        bm = match_boundary_pair(cnt, cnt, side1=0, side2=0)
        assert bm.total_score > 0.8

    def test_score_in_range(self):
        cnt1 = _square(0, 0, 50)
        cnt2 = _square(200, 200, 50)
        bm = match_boundary_pair(cnt1, cnt2)
        assert 0.0 <= bm.total_score <= 1.0

    def test_n_points_in_params(self):
        cnt = _square()
        bm = match_boundary_pair(cnt, cnt, n_points=25)
        assert bm.params["n_points"] == 25

    def test_idx_stored(self):
        cnt = _square()
        bm = match_boundary_pair(cnt, cnt, idx1=5, idx2=7)
        assert bm.idx1 == 5
        assert bm.idx2 == 7

    def test_sides_stored(self):
        cnt = _square()
        bm = match_boundary_pair(cnt, cnt, side1=0, side2=2)
        assert bm.side1 == 0
        assert bm.side2 == 2


# ─── batch_match_boundaries (extra) ──────────────────────────────────────────

class TestBatchMatchBoundariesExtra:
    def test_empty_pairs_empty_result(self):
        cnts = [_square()]
        assert batch_match_boundaries(cnts, []) == []

    def test_length_matches(self):
        cnts = [_square(i * 10, 0, 50) for i in range(4)]
        pairs = [(0, 1), (2, 3)]
        result = batch_match_boundaries(cnts, pairs)
        assert len(result) == 2

    def test_all_are_boundary_match(self):
        cnts = [_circle() for _ in range(3)]
        pairs = [(0, 1), (1, 2)]
        result = batch_match_boundaries(cnts, pairs)
        for r in result:
            assert isinstance(r, BoundaryMatch)

    def test_scores_in_range(self):
        cnts = [_square(i * 20, 0, 40) for i in range(3)]
        pairs = [(0, 1), (0, 2), (1, 2)]
        result = batch_match_boundaries(cnts, pairs)
        for bm in result:
            assert 0.0 <= bm.total_score <= 1.0

    def test_side_pairs_applied(self):
        cnts = [_square(), _square()]
        result = batch_match_boundaries(cnts, [(0, 1)], side_pairs=[(0, 2)])
        assert result[0].side1 == 0
        assert result[0].side2 == 2

    def test_kwargs_n_points_forwarded(self):
        cnts = [_square(), _square()]
        result = batch_match_boundaries(cnts, [(0, 1)], n_points=20)
        assert result[0].params["n_points"] == 20
