"""Tests for puzzle_reconstruction/matching/boundary_matcher.py"""
import pytest
import numpy as np

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


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_square_contour(size=100):
    """Square contour (N, 2)."""
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


def make_line_contour(n=40, horizontal=True):
    """Simple horizontal or vertical line contour."""
    if horizontal:
        pts = np.column_stack([np.linspace(0, 100, n), np.zeros(n)])
    else:
        pts = np.column_stack([np.zeros(n), np.linspace(0, 100, n)])
    return pts.astype(np.float64)


# ─── extract_boundary_points ─────────────────────────────────────────────────

class TestExtractBoundaryPoints:
    def test_invalid_side_raises(self):
        contour = make_square_contour()
        with pytest.raises(ValueError):
            extract_boundary_points(contour, side=4)

    def test_invalid_side_negative_raises(self):
        contour = make_square_contour()
        with pytest.raises(ValueError):
            extract_boundary_points(contour, side=-1)

    def test_empty_contour_raises(self):
        with pytest.raises(ValueError):
            extract_boundary_points(np.array([]).reshape(0, 2), side=0)

    def test_output_shape(self):
        contour = make_square_contour()
        pts = extract_boundary_points(contour, side=0, n_points=50)
        assert pts.shape == (50, 2)

    def test_output_shape_custom_n(self):
        contour = make_square_contour()
        pts = extract_boundary_points(contour, side=1, n_points=30)
        assert pts.shape == (30, 2)

    def test_all_four_sides(self):
        contour = make_square_contour()
        for side in range(4):
            pts = extract_boundary_points(contour, side=side, n_points=20)
            assert pts.shape == (20, 2)

    def test_float64_output(self):
        contour = make_square_contour()
        pts = extract_boundary_points(contour, side=0, n_points=10)
        assert pts.dtype == np.float64

    def test_handles_3d_contour(self):
        """Contour in (N, 1, 2) format."""
        contour = make_square_contour().reshape(-1, 1, 2)
        pts = extract_boundary_points(contour, side=0, n_points=20)
        assert pts.shape == (20, 2)

    def test_single_point_contour(self):
        contour = np.array([[5.0, 5.0]])
        pts = extract_boundary_points(contour, side=0, n_points=10)
        assert pts.shape == (10, 2)

    def test_top_side_has_low_y(self):
        """Side=0 (top) should return points near min y."""
        contour = make_square_contour(size=100)
        pts = extract_boundary_points(contour, side=0, n_points=20)
        # Top points have small y values
        assert pts[:, 1].mean() < 50


# ─── hausdorff_distance ──────────────────────────────────────────────────────

class TestHausdorffDistance:
    def test_identical_sets(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        d = hausdorff_distance(pts, pts)
        assert d == 0.0

    def test_empty_returns_zero(self):
        pts = np.array([[0.0, 0.0]])
        d = hausdorff_distance(np.array([]), pts)
        assert d == 0.0

    def test_known_distance(self):
        pts1 = np.array([[0.0, 0.0]])
        pts2 = np.array([[3.0, 4.0]])
        d = hausdorff_distance(pts1, pts2)
        assert abs(d - 5.0) < 1e-9

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        pts1 = rng.random((10, 2))
        pts2 = rng.random((15, 2))
        d12 = hausdorff_distance(pts1, pts2)
        d21 = hausdorff_distance(pts2, pts1)
        assert abs(d12 - d21) < 1e-9

    def test_non_negative(self):
        rng = np.random.default_rng(0)
        pts1 = rng.random((5, 2))
        pts2 = rng.random((8, 2))
        assert hausdorff_distance(pts1, pts2) >= 0.0

    def test_shifted_sets(self):
        pts1 = np.array([[0.0, 0.0], [1.0, 0.0]])
        pts2 = pts1 + np.array([[10.0, 0.0]])
        d = hausdorff_distance(pts1, pts2)
        # Hausdorff = max(directed) = 10
        assert abs(d - 10.0) < 1e-9


# ─── chamfer_distance ────────────────────────────────────────────────────────

class TestChamferDistance:
    def test_identical_sets(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        d = chamfer_distance(pts, pts)
        assert d == 0.0

    def test_empty_returns_zero(self):
        pts = np.array([[0.0, 0.0]])
        d = chamfer_distance(np.array([]), pts)
        assert d == 0.0

    def test_symmetric(self):
        rng = np.random.default_rng(7)
        pts1 = rng.random((10, 2))
        pts2 = rng.random((12, 2))
        d12 = chamfer_distance(pts1, pts2)
        d21 = chamfer_distance(pts2, pts1)
        assert abs(d12 - d21) < 1e-9

    def test_non_negative(self):
        rng = np.random.default_rng(1)
        pts1 = rng.random((5, 2)) * 100
        pts2 = rng.random((5, 2)) * 100
        assert chamfer_distance(pts1, pts2) >= 0.0

    def test_chamfer_leq_hausdorff(self):
        """Chamfer is always <= Hausdorff (mean vs max)."""
        rng = np.random.default_rng(99)
        pts1 = rng.random((8, 2)) * 50
        pts2 = rng.random((8, 2)) * 50
        c = chamfer_distance(pts1, pts2)
        h = hausdorff_distance(pts1, pts2)
        assert c <= h + 1e-9

    def test_known_value(self):
        pts1 = np.array([[0.0, 0.0]])
        pts2 = np.array([[3.0, 4.0]])
        d = chamfer_distance(pts1, pts2)
        assert abs(d - 5.0) < 1e-9


# ─── frechet_approx ──────────────────────────────────────────────────────────

class TestFrechetApprox:
    def test_identical_sets(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        d = frechet_approx(pts, pts)
        assert d == 0.0

    def test_empty_returns_zero(self):
        d = frechet_approx(np.array([]), np.array([[0.0, 0.0]]))
        assert d == 0.0

    def test_non_negative(self):
        rng = np.random.default_rng(3)
        pts1 = rng.random((6, 2))
        pts2 = rng.random((6, 2))
        assert frechet_approx(pts1, pts2) >= 0.0

    def test_handles_large_sets(self):
        """Should handle >MAX_PTS (30) without error."""
        rng = np.random.default_rng(5)
        pts1 = rng.random((50, 2))
        pts2 = rng.random((50, 2))
        d = frechet_approx(pts1, pts2)
        assert d >= 0.0

    def test_two_point_curves(self):
        pts1 = np.array([[0.0, 0.0], [1.0, 0.0]])
        pts2 = np.array([[0.0, 1.0], [1.0, 1.0]])
        d = frechet_approx(pts1, pts2)
        assert d == 1.0


# ─── score_boundary_pair ─────────────────────────────────────────────────────

class TestScoreBoundaryPair:
    def test_returns_four_floats(self):
        pts = np.zeros((10, 2))
        result = score_boundary_pair(pts, pts)
        assert len(result) == 4
        h, c, f, total = result
        assert all(isinstance(v, float) for v in [h, c, f, total])

    def test_identical_pts_gives_high_score(self):
        pts = np.tile(np.array([[0.0, 0.0]]), (10, 1))
        h, c, f, total = score_boundary_pair(pts, pts)
        # Distance=0 → exp(0)=1.0
        assert abs(h - 1.0) < 1e-6
        assert abs(c - 1.0) < 1e-6
        assert abs(f - 1.0) < 1e-6

    def test_all_scores_in_range(self):
        rng = np.random.default_rng(42)
        pts1 = rng.random((20, 2)) * 100
        pts2 = rng.random((20, 2)) * 100
        h, c, f, total = score_boundary_pair(pts1, pts2)
        for v in [h, c, f, total]:
            assert 0.0 <= v <= 1.0

    def test_custom_weights(self):
        pts1 = np.zeros((10, 2))
        pts2 = np.zeros((10, 2))
        h, c, f, total = score_boundary_pair(pts1, pts2, weights=(1.0, 0.0, 0.0))
        # Only hausdorff contributes: all distances=0 → exp(-0)=1.0
        assert abs(total - 1.0) < 1e-6

    def test_large_max_dist_gives_higher_scores(self):
        pts1 = np.zeros((5, 2))
        pts2 = np.ones((5, 2)) * 10  # 10 distance
        _, _, _, total_small = score_boundary_pair(pts1, pts2, max_dist=1.0)
        _, _, _, total_large = score_boundary_pair(pts1, pts2, max_dist=1000.0)
        assert total_large > total_small


# ─── match_boundary_pair ─────────────────────────────────────────────────────

class TestMatchBoundaryPair:
    def test_returns_boundary_match(self):
        contour = make_square_contour()
        result = match_boundary_pair(contour, contour)
        assert isinstance(result, BoundaryMatch)

    def test_default_sides(self):
        contour = make_square_contour()
        result = match_boundary_pair(contour, contour, idx1=3, idx2=4)
        assert result.idx1 == 3
        assert result.idx2 == 4
        assert result.side1 == 2
        assert result.side2 == 0

    def test_total_score_in_range(self):
        contour = make_square_contour()
        result = match_boundary_pair(contour, contour)
        assert 0.0 <= result.total_score <= 1.0

    def test_identical_contours_high_score(self):
        contour = make_square_contour()
        result = match_boundary_pair(contour, contour, side1=0, side2=0)
        assert result.total_score > 0.5

    def test_params_stored(self):
        contour = make_square_contour()
        result = match_boundary_pair(contour, contour, n_points=30, max_dist=50.0)
        assert result.params["n_points"] == 30
        assert result.params["max_dist"] == 50.0

    def test_custom_side_assignment(self):
        contour = make_square_contour()
        result = match_boundary_pair(contour, contour, side1=1, side2=3)
        assert result.side1 == 1
        assert result.side2 == 3


# ─── batch_match_boundaries ──────────────────────────────────────────────────

class TestBatchMatchBoundaries:
    def test_empty_pairs(self):
        contours = [make_square_contour()]
        result = batch_match_boundaries(contours, pairs=[])
        assert result == []

    def test_single_pair(self):
        c1 = make_square_contour()
        c2 = make_square_contour()
        result = batch_match_boundaries([c1, c2], pairs=[(0, 1)])
        assert len(result) == 1
        assert isinstance(result[0], BoundaryMatch)

    def test_multiple_pairs(self):
        contours = [make_square_contour() for _ in range(4)]
        pairs = [(0, 1), (1, 2), (2, 3)]
        result = batch_match_boundaries(contours, pairs=pairs)
        assert len(result) == 3

    def test_default_side_pairs(self):
        """Without side_pairs, defaults to (2, 0) for all."""
        contours = [make_square_contour(), make_square_contour()]
        result = batch_match_boundaries(contours, pairs=[(0, 1)])
        assert result[0].side1 == 2
        assert result[0].side2 == 0

    def test_custom_side_pairs(self):
        contours = [make_square_contour(), make_square_contour()]
        result = batch_match_boundaries(contours, pairs=[(0, 1)],
                                        side_pairs=[(1, 3)])
        assert result[0].side1 == 1
        assert result[0].side2 == 3

    def test_indices_in_results(self):
        contours = [make_square_contour(), make_square_contour(),
                    make_square_contour()]
        result = batch_match_boundaries(contours, pairs=[(0, 2)])
        assert result[0].idx1 == 0
        assert result[0].idx2 == 2

    def test_all_scores_in_range(self):
        contours = [make_square_contour() for _ in range(3)]
        pairs = [(0, 1), (0, 2), (1, 2)]
        results = batch_match_boundaries(contours, pairs=pairs)
        for r in results:
            assert 0.0 <= r.total_score <= 1.0


# ─── BoundaryMatch dataclass ─────────────────────────────────────────────────

class TestBoundaryMatch:
    def test_repr(self):
        bm = BoundaryMatch(
            idx1=1, idx2=2, side1=0, side2=2,
            hausdorff=0.8, chamfer=0.7, frechet=0.75, total_score=0.75,
        )
        r = repr(bm)
        assert "BoundaryMatch" in r
        assert "idx1=1" in r
        assert "idx2=2" in r
        assert "total=0.750" in r

    def test_default_params(self):
        bm = BoundaryMatch(
            idx1=0, idx2=1, side1=2, side2=0,
            hausdorff=0.5, chamfer=0.5, frechet=0.5, total_score=0.5,
        )
        assert bm.params == {}
