"""Extra tests for puzzle_reconstruction.matching.shape_matcher."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.matching.shape_matcher import (
    ShapeMatchResult,
    batch_match_shapes,
    find_best_shape_match,
    hu_distance,
    hu_moments,
    match_shapes,
    zernike_approx,
)


def _square(size=20.0, offset=(5.0, 5.0)):
    x0, y0 = offset
    return np.array([
        [x0,        y0],
        [x0 + size, y0],
        [x0 + size, y0 + size],
        [x0,        y0 + size],
    ], dtype=np.float64)


def _triangle(size=20.0):
    return np.array([[0.0, 0.0], [size, 0.0], [size / 2, size]], dtype=np.float64)


def _circle(radius=20.0, n=64):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cx, cy = radius + 5.0, radius + 5.0
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    return np.stack([xs, ys], axis=1)


def _two_points():
    return np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)


# ─── ShapeMatchResult extras ──────────────────────────────────────────────────

class TestShapeMatchResultExtra:
    def test_score_zero(self):
        r = ShapeMatchResult(idx1=0, idx2=1, hu_dist=5.0, iou=0.0, chamfer=10.0, score=0.0)
        assert r.score == pytest.approx(0.0)

    def test_score_one(self):
        r = ShapeMatchResult(idx1=0, idx2=1, hu_dist=0.0, iou=1.0, chamfer=0.0, score=1.0)
        assert r.score == pytest.approx(1.0)

    def test_params_empty_by_default(self):
        r = ShapeMatchResult(0, 1, 1.0, 0.5, 2.0, 0.6)
        assert r.params == {}

    def test_params_multiple_keys(self):
        r = ShapeMatchResult(0, 1, 1.0, 0.5, 2.0, 0.6,
                              params={"w_hu": 0.5, "w_iou": 0.3, "w_ch": 0.2})
        assert len(r.params) == 3

    def test_repr_is_string(self):
        r = ShapeMatchResult(0, 1, 1.0, 0.5, 2.0, 0.6)
        assert isinstance(repr(r), str)

    def test_negative_idx_allowed(self):
        r = ShapeMatchResult(idx1=-1, idx2=-2, hu_dist=0.0, iou=0.0, chamfer=0.0, score=0.5)
        assert r.idx1 == -1


# ─── hu_moments extras ────────────────────────────────────────────────────────

class TestHuMomentsExtra:
    def test_large_contour(self):
        c = _circle(radius=100, n=200)
        hm = hu_moments(c)
        assert hm.shape == (7,)
        assert hm.dtype == np.float64

    def test_small_contour_3_pts(self):
        c = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float64)
        hm = hu_moments(c)
        assert hm.shape == (7,)

    def test_float32_input(self):
        c = _square().astype(np.float32)
        hm = hu_moments(c)
        assert hm.shape == (7,)

    def test_triangle_differs_from_circle(self):
        hm_tr = hu_moments(_triangle())
        hm_ci = hu_moments(_circle())
        assert not np.allclose(hm_tr, hm_ci)

    def test_square_differs_from_triangle(self):
        hm_sq = hu_moments(_square())
        hm_tr = hu_moments(_triangle())
        assert not np.allclose(hm_sq, hm_tr)

    def test_single_point_returns_zeros(self):
        c = np.array([[5.0, 5.0]], dtype=np.float64)
        hm = hu_moments(c)
        np.testing.assert_array_equal(hm, np.zeros(7))

    def test_cv2_format_3d(self):
        c = _circle().reshape(-1, 1, 2).astype(np.float32)
        hm = hu_moments(c)
        assert hm.shape == (7,)

    def test_same_contour_same_moments(self):
        sq = _square(size=20, offset=(5, 5))
        hm1 = hu_moments(sq)
        hm2 = hu_moments(sq.copy())
        np.testing.assert_array_almost_equal(hm1, hm2, decimal=10)


# ─── hu_distance extras ───────────────────────────────────────────────────────

class TestHuDistanceExtra:
    def test_returns_float(self):
        hm = hu_moments(_square())
        assert isinstance(hu_distance(hm, hm), float)

    def test_zero_vectors(self):
        assert hu_distance(np.zeros(7), np.zeros(7)) == pytest.approx(0.0)

    def test_commutative(self):
        hm1 = hu_moments(_square())
        hm2 = hu_moments(_circle())
        assert hu_distance(hm1, hm2) == pytest.approx(hu_distance(hm2, hm1))

    def test_large_different_vectors(self):
        a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert hu_distance(a, b) > 0.0

    def test_all_ones_vs_all_zeros(self):
        a = np.ones(7)
        b = np.zeros(7)
        assert hu_distance(a, b) >= 0.0


# ─── zernike_approx extras ────────────────────────────────────────────────────

class TestZernikeApproxExtra:
    def test_square_n_radii_4(self):
        r = zernike_approx(_square(), n_radii=4)
        assert r.shape == (4,)
        assert r.dtype == np.float64

    def test_square_n_radii_16(self):
        r = zernike_approx(_square(), n_radii=16)
        assert r.shape == (16,)

    def test_non_negative_values(self):
        r = zernike_approx(_triangle(), n_radii=8)
        assert np.all(r >= 0.0)

    def test_circle_sums_to_one(self):
        r = zernike_approx(_circle(radius=30, n=200), n_radii=8)
        assert r.sum() == pytest.approx(1.0, abs=0.05)

    def test_triangle_nonneg_values(self):
        r = zernike_approx(_triangle(), n_radii=8)
        assert np.all(r >= 0.0)
        assert r.shape == (8,)

    def test_n_radii_2(self):
        r = zernike_approx(_square(), n_radii=2)
        assert r.shape == (2,)

    def test_large_n_radii(self):
        r = zernike_approx(_circle(n=100), n_radii=32)
        assert r.shape == (32,)
        assert np.all(r >= 0.0)


# ─── match_shapes extras ──────────────────────────────────────────────────────

class TestMatchShapesExtra:
    def test_identical_circle_high_score(self):
        c = _circle()
        r = match_shapes(c, c.copy(), canvas_size=(100, 100))
        assert r.score >= 0.8

    def test_identical_triangle_high_score(self):
        c = _triangle()
        r = match_shapes(c, c.copy(), canvas_size=(60, 60))
        assert r.score >= 0.8

    def test_hu_dist_zero_same_shape(self):
        c = _square()
        r = match_shapes(c, c.copy())
        assert r.hu_dist == pytest.approx(0.0, abs=1e-6)

    def test_chamfer_zero_same_contour(self):
        c = _square()
        r = match_shapes(c, c.copy())
        assert r.chamfer == pytest.approx(0.0, abs=1e-4)

    def test_iou_stored_in_range(self):
        r = match_shapes(_square(), _triangle(), canvas_size=(60, 60))
        assert 0.0 <= r.iou <= 1.0

    def test_custom_weights_1_0_0(self):
        r = match_shapes(_square(), _triangle(), weights=(1.0, 0.0, 0.0))
        assert isinstance(r, ShapeMatchResult)
        assert 0.0 <= r.score <= 1.0

    def test_custom_weights_0_1_0(self):
        r = match_shapes(_square(), _triangle(),
                          weights=(0.0, 1.0, 0.0), canvas_size=(60, 60))
        assert isinstance(r, ShapeMatchResult)
        assert 0.0 <= r.score <= 1.0

    def test_params_weights_keys(self):
        r = match_shapes(_square(), _triangle(), weights=(0.5, 0.3, 0.2))
        assert "w_hu" in r.params

    def test_large_circle_vs_small_circle(self):
        c1 = _circle(radius=10, n=32)
        c2 = _circle(radius=40, n=64)
        r = match_shapes(c1, c2, canvas_size=(100, 100))
        assert isinstance(r, ShapeMatchResult)


# ─── find_best_shape_match extras ─────────────────────────────────────────────

class TestFindBestShapeMatchExtra:
    def test_none_for_empty_candidates(self):
        assert find_best_shape_match(_square(), []) is None

    def test_single_candidate_idx2_is_0(self):
        r = find_best_shape_match(_square(), [_triangle()])
        assert r is not None
        assert r.idx2 == 0

    def test_three_candidates_best_returned(self):
        query = _square(size=20, offset=(2, 2))
        candidates = [_triangle(), query.copy(), _circle()]
        result = find_best_shape_match(query, candidates, canvas_size=(60, 60))
        assert result.idx2 == 1

    def test_query_idx_stored(self):
        r = find_best_shape_match(_square(), [_triangle()], query_idx=99)
        assert r.idx1 == 99

    def test_score_in_range(self):
        r = find_best_shape_match(_square(), [_triangle(), _circle()])
        assert 0.0 <= r.score <= 1.0

    def test_two_identical_candidates(self):
        c = _square()
        result = find_best_shape_match(c, [c.copy(), c.copy()], canvas_size=(50, 50))
        assert result is not None
        assert result.score >= 0.9


# ─── batch_match_shapes extras ────────────────────────────────────────────────

class TestBatchMatchShapesExtra:
    def test_1x1_triangle_vs_triangle(self):
        result = batch_match_shapes([_triangle()], [_triangle()], canvas_size=(60, 60))
        assert result.shape == (1, 1)
        assert float(result[0, 0]) >= 0.9

    def test_2x2_values_in_range(self):
        c1 = [_square(), _triangle()]
        c2 = [_circle(), _square()]
        result = batch_match_shapes(c1, c2, canvas_size=(80, 80))
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_dtype_float32(self):
        result = batch_match_shapes([_square()], [_circle()])
        assert result.dtype == np.float32

    def test_3x1_shape(self):
        result = batch_match_shapes([_square(), _triangle(), _circle()], [_square()])
        assert result.shape == (3, 1)

    def test_1x3_shape(self):
        result = batch_match_shapes([_square()], [_square(), _triangle(), _circle()])
        assert result.shape == (1, 3)

    def test_same_shape_diagonal_high(self):
        shapes = [_square(), _triangle(), _circle()]
        result = batch_match_shapes(shapes, shapes, canvas_size=(100, 100))
        for i in range(3):
            assert float(result[i, i]) >= 0.8
