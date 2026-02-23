"""Extra tests for puzzle_reconstruction.matching.shape_matcher."""
import pytest
import numpy as np

from puzzle_reconstruction.matching.shape_matcher import (
    ShapeMatchResult,
    hu_moments,
    hu_distance,
    zernike_approx,
    match_shapes,
    find_best_shape_match,
    batch_match_shapes,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_square(size=50, offset=0):
    return np.array([
        [offset, offset],
        [offset + size, offset],
        [offset + size, offset + size],
        [offset, offset + size],
        [offset, offset],
    ], dtype=np.float64)


def make_circle(n=32, radius=20, cx=50, cy=50):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)
    return np.stack([x, y], axis=1).astype(np.float64)


def make_triangle():
    return np.array([[10, 10], [60, 10], [35, 60]], dtype=np.float64)


# ─── TestShapeMatchResultExtra ──────────────────────────────────────────────

class TestShapeMatchResultExtra:
    def test_defaults(self):
        r = ShapeMatchResult(idx1=0, idx2=0, hu_dist=0.0, iou=1.0,
                             chamfer=0.0, score=1.0)
        assert r.params == {}

    def test_all_fields(self):
        r = ShapeMatchResult(idx1=3, idx2=5, hu_dist=1.5, iou=0.6,
                             chamfer=2.5, score=0.4, params={"k": 1})
        assert r.idx1 == 3
        assert r.idx2 == 5
        assert r.hu_dist == pytest.approx(1.5)
        assert r.iou == pytest.approx(0.6)
        assert r.chamfer == pytest.approx(2.5)
        assert r.score == pytest.approx(0.4)
        assert r.params["k"] == 1

    def test_zero_score(self):
        r = ShapeMatchResult(0, 0, 10.0, 0.0, 100.0, 0.0)
        assert r.score == pytest.approx(0.0)

    def test_perfect_score(self):
        r = ShapeMatchResult(0, 0, 0.0, 1.0, 0.0, 1.0)
        assert r.score == pytest.approx(1.0)


# ─── TestHuMomentsExtra ────────────────────────────────────────────────────

class TestHuMomentsExtra:
    def test_circle_nonzero(self):
        hm = hu_moments(make_circle())
        assert not np.all(hm == 0)

    def test_triangle_nonzero(self):
        hm = hu_moments(make_triangle())
        assert not np.all(hm == 0)

    def test_dtype_float64(self):
        hm = hu_moments(make_square())
        assert hm.dtype == np.float64

    def test_scaled_square_similar(self):
        h1 = hu_moments(make_square(size=50))
        h2 = hu_moments(make_square(size=100))
        # Hu moments are scale-invariant for similar shapes
        dist = np.linalg.norm(h1 - h2)
        assert dist < 1.0

    def test_different_shapes_different_moments(self):
        h_sq = hu_moments(make_square())
        h_ci = hu_moments(make_circle())
        assert not np.allclose(h_sq, h_ci, atol=1e-3)

    def test_large_n_points(self):
        hm = hu_moments(make_circle(n=256))
        assert hm.shape == (7,)

    def test_offset_square_nonzero(self):
        h1 = hu_moments(make_square(size=40, offset=0))
        h2 = hu_moments(make_square(size=40, offset=100))
        assert not np.all(h1 == 0)
        assert not np.all(h2 == 0)


# ─── TestHuDistanceExtra ────────────────────────────────────────────────────

class TestHuDistanceExtra:
    def test_same_shape_zero(self):
        c = make_square()
        hm = hu_moments(c)
        assert hu_distance(hm, hm) == pytest.approx(0.0)

    def test_different_shapes_positive(self):
        h1 = hu_moments(make_square())
        h2 = hu_moments(make_circle())
        assert hu_distance(h1, h2) > 0.0

    def test_triangle_norms(self):
        hm = np.array([3.0, 4.0])
        assert hu_distance(hm, np.zeros(2)) == pytest.approx(5.0)

    def test_symmetry(self):
        h1 = hu_moments(make_square())
        h2 = hu_moments(make_circle())
        assert hu_distance(h1, h2) == pytest.approx(hu_distance(h2, h1))

    def test_all_zeros(self):
        z = np.zeros(7)
        assert hu_distance(z, z) == pytest.approx(0.0)


# ─── TestZernikeApproxExtra ─────────────────────────────────────────────────

class TestZernikeApproxExtra:
    def test_circle(self):
        z = zernike_approx(make_circle())
        assert z.shape == (8,)
        assert np.all(z >= 0.0)

    def test_triangle(self):
        z = zernike_approx(make_triangle())
        assert z.shape == (8,)

    def test_n_radii_4(self):
        z = zernike_approx(make_square(), n_radii=4)
        assert z.shape == (4,)

    def test_n_radii_32(self):
        z = zernike_approx(make_circle(n=64), n_radii=32)
        assert z.shape == (32,)

    def test_sum_one(self):
        z = zernike_approx(make_circle(n=128), n_radii=8)
        assert abs(z.sum() - 1.0) < 1e-6

    def test_different_shapes_different_features(self):
        z1 = zernike_approx(make_square())
        z2 = zernike_approx(make_circle())
        assert not np.allclose(z1, z2, atol=1e-3)


# ─── TestMatchShapesExtra ───────────────────────────────────────────────────

class TestMatchShapesExtra:
    def test_square_vs_square(self):
        r = match_shapes(make_square(), make_square())
        assert r.score > 0.5

    def test_square_vs_circle(self):
        r = match_shapes(make_square(), make_circle())
        assert 0.0 <= r.score <= 1.0

    def test_triangle_vs_triangle(self):
        r = match_shapes(make_triangle(), make_triangle())
        assert r.score > 0.5

    def test_custom_idx(self):
        r = match_shapes(make_square(), make_circle(), idx1=10, idx2=20)
        assert r.idx1 == 10
        assert r.idx2 == 20

    def test_weights_hu_only(self):
        r = match_shapes(make_square(), make_circle(),
                         weights=(1.0, 0.0, 0.0))
        assert 0.0 <= r.score <= 1.0

    def test_weights_iou_only(self):
        r = match_shapes(make_square(), make_circle(),
                         weights=(0.0, 1.0, 0.0))
        assert 0.0 <= r.score <= 1.0

    def test_weights_chamfer_only(self):
        r = match_shapes(make_square(), make_circle(),
                         weights=(0.0, 0.0, 1.0))
        assert 0.0 <= r.score <= 1.0

    def test_scaled_square_similar_score(self):
        r = match_shapes(make_square(50), make_square(100))
        assert r.score > 0.3


# ─── TestFindBestShapeMatchExtra ────────────────────────────────────────────

class TestFindBestShapeMatchExtra:
    def test_two_candidates(self):
        query = make_square()
        cands = [make_circle(), make_square()]
        result = find_best_shape_match(query, cands)
        assert result is not None
        assert result.idx2 == 1  # square should match best

    def test_three_candidates(self):
        query = make_triangle()
        cands = [make_square(), make_circle(), make_triangle()]
        result = find_best_shape_match(query, cands)
        assert result is not None
        assert result.idx2 == 2

    def test_query_idx(self):
        result = find_best_shape_match(make_square(), [make_circle()],
                                       query_idx=99)
        assert result.idx1 == 99

    def test_score_in_0_1(self):
        result = find_best_shape_match(make_square(),
                                       [make_circle(), make_triangle()])
        assert 0.0 <= result.score <= 1.0

    def test_single_identical(self):
        sq = make_square(30, 5)
        result = find_best_shape_match(sq, [sq.copy()])
        assert result.score > 0.5


# ─── TestBatchMatchShapesExtra ──────────────────────────────────────────────

class TestBatchMatchShapesExtra:
    def test_2x3(self):
        c1s = [make_square(), make_circle()]
        c2s = [make_triangle(), make_square(), make_circle()]
        result = batch_match_shapes(c1s, c2s)
        assert result.shape == (2, 3)

    def test_diagonal_high(self):
        shapes = [make_square(), make_circle(), make_triangle()]
        result = batch_match_shapes(shapes, shapes)
        for i in range(3):
            assert float(result[i, i]) > 0.3

    def test_symmetric(self):
        shapes = [make_square(), make_circle()]
        result = batch_match_shapes(shapes, shapes)
        assert float(result[0, 1]) == pytest.approx(float(result[1, 0]),
                                                      abs=0.05)

    def test_single_pair(self):
        result = batch_match_shapes([make_square()], [make_circle()])
        assert result.shape == (1, 1)

    def test_all_in_range(self):
        c1s = [make_square(), make_triangle()]
        c2s = [make_circle(), make_square()]
        result = batch_match_shapes(c1s, c2s)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_dtype(self):
        result = batch_match_shapes([make_square()], [make_square()])
        assert result.dtype == np.float32
