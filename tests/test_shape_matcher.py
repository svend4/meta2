"""Tests for puzzle_reconstruction.matching.shape_matcher."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _square(size: float = 20.0, offset: tuple = (5.0, 5.0)) -> np.ndarray:
    x0, y0 = offset
    return np.array([
        [x0,        y0],
        [x0 + size, y0],
        [x0 + size, y0 + size],
        [x0,        y0 + size],
    ], dtype=np.float64)


def _triangle(size: float = 20.0) -> np.ndarray:
    return np.array([[0.0, 0.0], [size, 0.0], [size / 2, size]], dtype=np.float64)


def _circle(radius: float = 20.0, n: int = 64) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cx, cy = radius + 5.0, radius + 5.0
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    return np.stack([xs, ys], axis=1)


def _two_points() -> np.ndarray:
    return np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)


# ─── ShapeMatchResult ────────────────────────────────────────────────────────

class TestShapeMatchResult:
    def test_fields_stored(self):
        r = ShapeMatchResult(idx1=0, idx2=1, hu_dist=2.5, iou=0.7, chamfer=3.0, score=0.8)
        assert r.idx1 == 0
        assert r.idx2 == 1
        assert r.hu_dist == pytest.approx(2.5)
        assert r.iou == pytest.approx(0.7)
        assert r.chamfer == pytest.approx(3.0)
        assert r.score == pytest.approx(0.8)
        assert r.params == {}

    def test_params_stored(self):
        r = ShapeMatchResult(0, 1, 1.0, 0.5, 2.0, 0.6, params={"w_hu": 0.5})
        assert r.params["w_hu"] == pytest.approx(0.5)


# ─── hu_moments ──────────────────────────────────────────────────────────────

class TestHuMoments:
    def test_returns_array_shape_7(self):
        result = hu_moments(_square())
        assert isinstance(result, np.ndarray)
        assert result.shape == (7,)

    def test_dtype_float64(self):
        assert hu_moments(_square()).dtype == np.float64

    def test_less_than_3_points_returns_zeros(self):
        result = hu_moments(_two_points())
        np.testing.assert_array_equal(result, np.zeros(7))

    def test_empty_contour_returns_zeros(self):
        result = hu_moments(np.empty((0, 2), dtype=np.float64))
        np.testing.assert_array_equal(result, np.zeros(7))

    def test_cv2_format_accepted(self):
        c = _square().reshape(-1, 1, 2)
        result = hu_moments(c)
        assert result.shape == (7,)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            hu_moments(np.ones((5, 3)))

    def test_square_and_circle_differ(self):
        hm_sq = hu_moments(_square())
        hm_ci = hu_moments(_circle())
        # Hu moments are shape descriptors — square and circle should differ
        assert not np.allclose(hm_sq, hm_ci)

    def test_same_shape_consistent(self):
        c = _square()
        hm1 = hu_moments(c)
        hm2 = hu_moments(c.copy())
        np.testing.assert_array_almost_equal(hm1, hm2)


# ─── hu_distance ─────────────────────────────────────────────────────────────

class TestHuDistance:
    def test_same_vector_returns_zero(self):
        hm = hu_moments(_square())
        assert hu_distance(hm, hm) == pytest.approx(0.0)

    def test_different_shapes_positive(self):
        hm_sq = hu_moments(_square())
        hm_tr = hu_moments(_triangle())
        assert hu_distance(hm_sq, hm_tr) > 0.0

    def test_non_negative(self):
        hm1 = hu_moments(_square())
        hm2 = hu_moments(_circle())
        assert hu_distance(hm1, hm2) >= 0.0

    def test_symmetric(self):
        hm1 = hu_moments(_square())
        hm2 = hu_moments(_triangle())
        assert hu_distance(hm1, hm2) == pytest.approx(hu_distance(hm2, hm1))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            hu_distance(np.zeros(7), np.zeros(5))

    def test_zero_vectors_distance_zero(self):
        assert hu_distance(np.zeros(7), np.zeros(7)) == pytest.approx(0.0)


# ─── zernike_approx ──────────────────────────────────────────────────────────

class TestZernikeApprox:
    def test_returns_array(self):
        result = zernike_approx(_square())
        assert isinstance(result, np.ndarray)

    def test_shape_n_radii(self):
        for n in [4, 8, 16]:
            result = zernike_approx(_square(), n_radii=n)
            assert result.shape == (n,)

    def test_dtype_float64(self):
        assert zernike_approx(_square()).dtype == np.float64

    def test_n_radii_less_than_2_raises(self):
        with pytest.raises(ValueError):
            zernike_approx(_square(), n_radii=1)

    def test_empty_contour_returns_zeros(self):
        result = zernike_approx(np.empty((0, 2), dtype=np.float64))
        np.testing.assert_array_equal(result, np.zeros(8))

    def test_less_than_3_points_returns_zeros(self):
        result = zernike_approx(_two_points())
        np.testing.assert_array_equal(result, np.zeros(8))

    def test_values_non_negative(self):
        result = zernike_approx(_square())
        assert np.all(result >= 0.0)

    def test_sums_to_approx_one(self):
        result = zernike_approx(_circle(radius=30, n=200))
        assert result.sum() == pytest.approx(1.0, abs=0.01)

    def test_degenerate_same_point_returns_zeros(self):
        c = np.ones((10, 2), dtype=np.float64)
        result = zernike_approx(c)
        np.testing.assert_array_equal(result, np.zeros(8))


# ─── match_shapes ────────────────────────────────────────────────────────────

class TestMatchShapes:
    def test_returns_shape_match_result(self):
        r = match_shapes(_square(), _square())
        assert isinstance(r, ShapeMatchResult)

    def test_same_contour_score_near_one(self):
        c = _square(size=20, offset=(2, 2))
        r = match_shapes(c, c.copy(), canvas_size=(50, 50))
        assert r.score >= 0.9

    def test_score_in_unit_interval(self):
        r = match_shapes(_square(), _circle(), canvas_size=(80, 80))
        assert 0.0 <= r.score <= 1.0

    def test_different_shapes_score_less_than_same(self):
        c = _square(size=20, offset=(2, 2))
        r_same = match_shapes(c, c.copy(), canvas_size=(50, 50))
        r_diff = match_shapes(_square(), _triangle(), canvas_size=(50, 50))
        assert r_same.score >= r_diff.score

    def test_indices_stored(self):
        r = match_shapes(_square(), _circle(), idx1=3, idx2=7)
        assert r.idx1 == 3
        assert r.idx2 == 7

    def test_hu_dist_non_negative(self):
        r = match_shapes(_square(), _triangle())
        assert r.hu_dist >= 0.0

    def test_iou_in_unit_interval(self):
        r = match_shapes(_square(), _triangle(), canvas_size=(50, 50))
        assert 0.0 <= r.iou <= 1.0

    def test_chamfer_non_negative(self):
        r = match_shapes(_square(), _circle())
        assert r.chamfer >= 0.0

    def test_zero_weights_raises(self):
        with pytest.raises(ValueError):
            match_shapes(_square(), _circle(), weights=(0.0, 0.0, 0.0))

    def test_params_contain_weights(self):
        r = match_shapes(_square(), _triangle(), weights=(0.5, 0.3, 0.2))
        assert "w_hu" in r.params
        assert r.params["w_hu"] == pytest.approx(0.5)


# ─── find_best_shape_match ───────────────────────────────────────────────────

class TestFindBestShapeMatch:
    def test_empty_candidates_returns_none(self):
        result = find_best_shape_match(_square(), [])
        assert result is None

    def test_returns_shape_match_result(self):
        result = find_best_shape_match(_square(), [_triangle(), _circle()])
        assert isinstance(result, ShapeMatchResult)

    def test_single_candidate_returned(self):
        result = find_best_shape_match(_square(), [_triangle()])
        assert result is not None
        assert result.idx2 == 0

    def test_best_score_selected(self):
        query = _square(size=20, offset=(2, 2))
        candidates = [_triangle(), query.copy(), _circle()]
        result = find_best_shape_match(query, candidates, canvas_size=(60, 60))
        # The identical contour (index 1) should have highest score
        assert result.idx2 == 1

    def test_query_idx_stored(self):
        result = find_best_shape_match(_square(), [_triangle()], query_idx=5)
        assert result.idx1 == 5

    def test_score_in_unit_interval(self):
        result = find_best_shape_match(_square(), [_triangle(), _circle()])
        assert 0.0 <= result.score <= 1.0


# ─── batch_match_shapes ──────────────────────────────────────────────────────

class TestBatchMatchShapes:
    def test_empty_both_returns_empty_matrix(self):
        result = batch_match_shapes([], [])
        assert result.shape == (0, 0)

    def test_empty_first_returns_empty_rows(self):
        result = batch_match_shapes([], [_square()])
        assert result.shape == (0, 1)

    def test_empty_second_returns_empty_cols(self):
        result = batch_match_shapes([_square()], [])
        assert result.shape == (1, 0)

    def test_shape_n_by_m(self):
        c1 = [_square(), _triangle()]
        c2 = [_circle(), _square(), _triangle()]
        result = batch_match_shapes(c1, c2)
        assert result.shape == (2, 3)

    def test_dtype_float32(self):
        result = batch_match_shapes([_square()], [_triangle()])
        assert result.dtype == np.float32

    def test_values_in_unit_interval(self):
        c1 = [_square(), _triangle()]
        c2 = [_circle(), _square()]
        result = batch_match_shapes(c1, c2, canvas_size=(80, 80))
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_1x1_same_contour_near_one(self):
        c = _square(size=20, offset=(2, 2))
        result = batch_match_shapes([c], [c.copy()], canvas_size=(50, 50))
        assert float(result[0, 0]) >= 0.9
