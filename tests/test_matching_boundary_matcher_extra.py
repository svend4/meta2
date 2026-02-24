"""Extra tests for puzzle_reconstruction/matching/boundary_matcher.py"""
import pytest
import numpy as np

from puzzle_reconstruction.matching.boundary_matcher import (
    BoundaryMatch,
    batch_match_boundaries,
    chamfer_distance,
    extract_boundary_points,
    frechet_approx,
    hausdorff_distance,
    match_boundary_pair,
    score_boundary_pair,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _square(size=100):
    pts = []
    for i in range(size):
        pts.append([i, 0])
    for i in range(size):
        pts.append([size, i])
    for i in range(size, -1, -1):
        pts.append([i, size])
    for i in range(size, -1, -1):
        pts.append([0, i])
    return np.array(pts, dtype=np.float64)


def _line(n=40, horizontal=True):
    if horizontal:
        return np.column_stack([np.linspace(0, 100, n),
                                np.zeros(n)]).astype(np.float64)
    return np.column_stack([np.zeros(n),
                             np.linspace(0, 100, n)]).astype(np.float64)


# ─── TestExtractBoundaryPointsExtra ──────────────────────────────────────────

class TestExtractBoundaryPointsExtra:
    def test_each_side_returns_correct_n(self):
        c = _square()
        for side in range(4):
            pts = extract_boundary_points(c, side=side, n_points=25)
            assert pts.shape == (25, 2)

    def test_large_n_points(self):
        c = _square()
        pts = extract_boundary_points(c, side=0, n_points=200)
        assert pts.shape == (200, 2)

    def test_n_points_1(self):
        c = _square()
        pts = extract_boundary_points(c, side=0, n_points=1)
        assert pts.shape == (1, 2)

    def test_2d_contour_accepted(self):
        c = _square()
        pts = extract_boundary_points(c, side=2, n_points=10)
        assert pts.shape == (10, 2)

    def test_output_is_float64(self):
        c = _square()
        pts = extract_boundary_points(c, side=1, n_points=15)
        assert pts.dtype == np.float64

    def test_side_2_exists(self):
        c = _square()
        pts = extract_boundary_points(c, side=2, n_points=10)
        assert pts.ndim == 2

    def test_side_3_exists(self):
        c = _square()
        pts = extract_boundary_points(c, side=3, n_points=10)
        assert pts.shape[0] == 10


# ─── TestHausdorffDistanceExtra ───────────────────────────────────────────────

class TestHausdorffDistanceExtra:
    def test_multiple_identical_points(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert hausdorff_distance(pts, pts) == pytest.approx(0.0)

    def test_3_4_5_triangle(self):
        pts1 = np.array([[0.0, 0.0]])
        pts2 = np.array([[3.0, 4.0]])
        d = hausdorff_distance(pts1, pts2)
        assert abs(d - 5.0) < 1e-9

    def test_returns_float(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        d = hausdorff_distance(pts, pts)
        assert isinstance(d, float)

    def test_shifted_by_one(self):
        pts1 = np.array([[0.0, 0.0], [1.0, 0.0]])
        pts2 = pts1 + np.array([[1.0, 0.0]])
        d = hausdorff_distance(pts1, pts2)
        assert d == pytest.approx(1.0)

    def test_large_sets_nonneg(self):
        rng = np.random.default_rng(42)
        pts1 = rng.random((30, 2)) * 100
        pts2 = rng.random((30, 2)) * 100
        assert hausdorff_distance(pts1, pts2) >= 0.0


# ─── TestChamferDistanceExtra ─────────────────────────────────────────────────

class TestChamferDistanceExtra:
    def test_3_4_5_known(self):
        pts1 = np.array([[0.0, 0.0]])
        pts2 = np.array([[3.0, 4.0]])
        d = chamfer_distance(pts1, pts2)
        assert abs(d - 5.0) < 1e-9

    def test_large_sets(self):
        rng = np.random.default_rng(7)
        pts1 = rng.random((50, 2)) * 100
        pts2 = rng.random((50, 2)) * 100
        d = chamfer_distance(pts1, pts2)
        assert d >= 0.0

    def test_chamfer_leq_hausdorff_large(self):
        rng = np.random.default_rng(11)
        pts1 = rng.random((20, 2)) * 50
        pts2 = rng.random((20, 2)) * 50
        c = chamfer_distance(pts1, pts2)
        h = hausdorff_distance(pts1, pts2)
        assert c <= h + 1e-9

    def test_returns_float(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        assert isinstance(chamfer_distance(pts, pts), float)

    def test_single_shifted_point(self):
        pts1 = np.array([[0.0, 0.0]])
        pts2 = np.array([[1.0, 0.0]])
        d = chamfer_distance(pts1, pts2)
        assert abs(d - 1.0) < 1e-9


# ─── TestFrechetApproxExtra ───────────────────────────────────────────────────

class TestFrechetApproxExtra:
    def test_identical_two_points(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        d = frechet_approx(pts, pts)
        assert d == pytest.approx(0.0)

    def test_returns_float(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        d = frechet_approx(pts, pts)
        assert isinstance(d, float)

    def test_horizontal_vs_vertical_line(self):
        h = _line(10, horizontal=True)
        v = _line(10, horizontal=False)
        d = frechet_approx(h, v)
        assert d >= 0.0

    def test_larger_sets(self):
        rng = np.random.default_rng(5)
        pts1 = rng.random((40, 2))
        pts2 = rng.random((40, 2))
        d = frechet_approx(pts1, pts2)
        assert d >= 0.0

    def test_nonneg_many_seeds(self):
        for s in range(5):
            rng = np.random.default_rng(s)
            pts1 = rng.random((8, 2))
            pts2 = rng.random((8, 2))
            assert frechet_approx(pts1, pts2) >= 0.0


# ─── TestScoreBoundaryPairExtra ───────────────────────────────────────────────

class TestScoreBoundaryPairExtra:
    def test_all_zeros_all_ones(self):
        pts = np.zeros((10, 2))
        h, c, f, total = score_boundary_pair(pts, pts)
        assert abs(h - 1.0) < 1e-6
        assert abs(c - 1.0) < 1e-6

    def test_various_seeds_in_range(self):
        rng = np.random.default_rng(42)
        for _ in range(5):
            pts1 = rng.random((12, 2)) * 50
            pts2 = rng.random((12, 2)) * 50
            h, c, f, total = score_boundary_pair(pts1, pts2)
            for v in (h, c, f, total):
                assert 0.0 <= v <= 1.0

    def test_unequal_sizes(self):
        pts1 = np.zeros((5, 2))
        pts2 = np.zeros((15, 2))
        h, c, f, total = score_boundary_pair(pts1, pts2)
        assert 0.0 <= total <= 1.0

    def test_zero_weight_component_ignored(self):
        pts1 = np.zeros((5, 2))
        pts2 = np.ones((5, 2)) * 100
        _, _, _, total = score_boundary_pair(pts1, pts2, weights=(0.0, 1.0, 0.0))
        assert 0.0 <= total <= 1.0


# ─── TestMatchBoundaryPairExtra ───────────────────────────────────────────────

class TestMatchBoundaryPairExtra:
    def test_idx1_idx2_stored(self):
        c = _square()
        result = match_boundary_pair(c, c, idx1=7, idx2=9)
        assert result.idx1 == 7
        assert result.idx2 == 9

    def test_custom_n_points_stored(self):
        c = _square()
        result = match_boundary_pair(c, c, n_points=15)
        assert result.params["n_points"] == 15

    def test_score_in_range_various_sides(self):
        c = _square()
        for s1, s2 in [(0, 2), (1, 3), (2, 0), (3, 1)]:
            r = match_boundary_pair(c, c, side1=s1, side2=s2)
            assert 0.0 <= r.total_score <= 1.0

    def test_max_dist_stored(self):
        c = _square()
        result = match_boundary_pair(c, c, max_dist=75.0)
        assert result.params["max_dist"] == pytest.approx(75.0)

    def test_returns_boundary_match_type(self):
        c = _square()
        result = match_boundary_pair(c, c)
        assert isinstance(result, BoundaryMatch)


# ─── TestBatchMatchBoundariesExtra ───────────────────────────────────────────

class TestBatchMatchBoundariesExtra:
    def test_five_pairs(self):
        contours = [_square() for _ in range(6)]
        pairs = [(i, i + 1) for i in range(5)]
        results = batch_match_boundaries(contours, pairs=pairs)
        assert len(results) == 5

    def test_all_boundary_match_type(self):
        contours = [_square(), _square(), _square()]
        results = batch_match_boundaries(contours, pairs=[(0, 1), (1, 2)])
        for r in results:
            assert isinstance(r, BoundaryMatch)

    def test_custom_side_pairs_all_stored(self):
        contours = [_square(), _square(), _square()]
        pairs = [(0, 1), (0, 2)]
        sp = [(1, 3), (2, 0)]
        results = batch_match_boundaries(contours, pairs=pairs, side_pairs=sp)
        assert results[0].side1 == 1
        assert results[1].side1 == 2

    def test_scores_all_in_range(self):
        contours = [_square() for _ in range(4)]
        pairs = [(0, 1), (1, 2), (2, 3)]
        for r in batch_match_boundaries(contours, pairs=pairs):
            assert 0.0 <= r.total_score <= 1.0

    def test_indices_preserved(self):
        contours = [_square() for _ in range(4)]
        results = batch_match_boundaries(contours, pairs=[(3, 1)])
        assert results[0].idx1 == 3
        assert results[0].idx2 == 1


# ─── TestBoundaryMatchExtra ───────────────────────────────────────────────────

class TestBoundaryMatchExtra:
    def test_field_values_stored(self):
        bm = BoundaryMatch(idx1=2, idx2=5, side1=1, side2=3,
                           hausdorff=0.6, chamfer=0.7, frechet=0.65,
                           total_score=0.65)
        assert bm.hausdorff == pytest.approx(0.6)
        assert bm.chamfer == pytest.approx(0.7)
        assert bm.frechet == pytest.approx(0.65)

    def test_side_stored(self):
        bm = BoundaryMatch(idx1=0, idx2=1, side1=2, side2=0,
                           hausdorff=0.5, chamfer=0.5, frechet=0.5,
                           total_score=0.5)
        assert bm.side1 == 2
        assert bm.side2 == 0

    def test_total_score_in_range(self):
        bm = BoundaryMatch(idx1=0, idx2=1, side1=0, side2=2,
                           hausdorff=0.8, chamfer=0.9, frechet=0.85,
                           total_score=0.85)
        assert 0.0 <= bm.total_score <= 1.0

    def test_params_key_stored(self):
        bm = BoundaryMatch(idx1=0, idx2=1, side1=0, side2=2,
                           hausdorff=0.8, chamfer=0.9, frechet=0.85,
                           total_score=0.85, params={"n_points": 20})
        assert bm.params["n_points"] == 20

    def test_repr_contains_total(self):
        bm = BoundaryMatch(idx1=0, idx2=1, side1=0, side2=2,
                           hausdorff=0.8, chamfer=0.9, frechet=0.85,
                           total_score=0.85)
        assert "0.850" in repr(bm)
