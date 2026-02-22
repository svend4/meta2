"""Тесты для puzzle_reconstruction/matching/shape_matcher.py."""
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


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_square(size=50, offset=0):
    """Square contour (N,2) in pixel coordinates."""
    pts = np.array([
        [offset, offset],
        [offset + size, offset],
        [offset + size, offset + size],
        [offset, offset + size],
        [offset, offset],
    ], dtype=np.float64)
    return pts


def make_circle(n=32, radius=20, cx=50, cy=50):
    """Circle contour approximated by n points."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)
    return np.stack([x, y], axis=1).astype(np.float64)


def make_triangle():
    return np.array([[10, 10], [60, 10], [35, 60]], dtype=np.float64)


# ─── ShapeMatchResult ─────────────────────────────────────────────────────────

class TestShapeMatchResult:
    def test_creation(self):
        r = ShapeMatchResult(idx1=0, idx2=1, hu_dist=0.5, iou=0.8, chamfer=3.0, score=0.7)
        assert r.idx1 == 0
        assert r.idx2 == 1
        assert r.hu_dist == pytest.approx(0.5)
        assert r.iou == pytest.approx(0.8)
        assert r.chamfer == pytest.approx(3.0)
        assert r.score == pytest.approx(0.7)
        assert r.params == {}

    def test_creation_with_params(self):
        r = ShapeMatchResult(0, 1, 0.0, 1.0, 0.0, 1.0, params={"w_hu": 0.5})
        assert r.params["w_hu"] == 0.5


# ─── hu_moments ───────────────────────────────────────────────────────────────

class TestHuMoments:
    def test_returns_shape_7(self):
        c = make_square()
        hm = hu_moments(c)
        assert hm.shape == (7,)
        assert hm.dtype == np.float64

    def test_less_than_3_points_returns_zeros(self):
        c = np.array([[0, 0], [1, 1]], dtype=np.float64)
        hm = hu_moments(c)
        assert hm.shape == (7,)
        np.testing.assert_array_equal(hm, 0.0)

    def test_single_point_returns_zeros(self):
        c = np.array([[5, 5]], dtype=np.float64)
        hm = hu_moments(c)
        np.testing.assert_array_equal(hm, 0.0)

    def test_wrong_shape_raises(self):
        c = np.ones((5, 3))  # wrong: should be (N,2)
        with pytest.raises(ValueError):
            hu_moments(c)

    def test_1d_raises(self):
        c = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            hu_moments(c)

    def test_square_nonzero(self):
        c = make_square(size=50, offset=5)
        hm = hu_moments(c)
        # For a non-degenerate shape, at least first moment is nonzero
        assert not np.all(hm == 0)

    def test_accepts_n1_2_shape(self):
        c = make_circle().reshape(-1, 1, 2)
        hm = hu_moments(c)
        assert hm.shape == (7,)

    def test_identical_shapes_same_moments(self):
        c1 = make_square(size=40, offset=10)
        c2 = make_square(size=40, offset=10)
        hm1 = hu_moments(c1)
        hm2 = hu_moments(c2)
        np.testing.assert_allclose(hm1, hm2, atol=1e-10)


# ─── hu_distance ──────────────────────────────────────────────────────────────

class TestHuDistance:
    def test_identical_vectors_zero(self):
        hm = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        assert hu_distance(hm, hm) == pytest.approx(0.0)

    def test_different_lengths_raises(self):
        hm1 = np.array([1.0, 2.0, 3.0])
        hm2 = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="length"):
            hu_distance(hm1, hm2)

    def test_nonnegative(self):
        hm1 = np.array([1.0, -2.0, 3.0, 0.0, 0.5, -1.0, 2.0])
        hm2 = np.zeros(7)
        assert hu_distance(hm1, hm2) >= 0.0

    def test_symmetric(self):
        hm1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        hm2 = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        assert hu_distance(hm1, hm2) == pytest.approx(hu_distance(hm2, hm1))

    def test_l2_norm(self):
        hm1 = np.array([3.0, 4.0])
        hm2 = np.zeros(2)
        assert hu_distance(hm1, hm2) == pytest.approx(5.0)


# ─── zernike_approx ───────────────────────────────────────────────────────────

class TestZernikeApprox:
    def test_default_shape(self):
        c = make_circle()
        z = zernike_approx(c)
        assert z.shape == (8,)
        assert z.dtype == np.float64

    def test_custom_n_radii(self):
        c = make_square()
        z = zernike_approx(c, n_radii=16)
        assert z.shape == (16,)

    def test_n_radii_less_than_2_raises(self):
        c = make_square()
        with pytest.raises(ValueError, match="n_radii"):
            zernike_approx(c, n_radii=1)

    def test_n_radii_0_raises(self):
        c = make_square()
        with pytest.raises(ValueError):
            zernike_approx(c, n_radii=0)

    def test_less_than_3_points_returns_zeros(self):
        c = np.array([[0, 0], [1, 1]], dtype=np.float64)
        z = zernike_approx(c, n_radii=8)
        np.testing.assert_array_equal(z, 0.0)

    def test_sums_to_approximately_1(self):
        c = make_circle(n=64)
        z = zernike_approx(c, n_radii=8)
        # Histogram normalized: sum should be ~1
        assert abs(z.sum() - 1.0) < 1e-6

    def test_nonnegative(self):
        c = make_square()
        z = zernike_approx(c)
        assert np.all(z >= 0.0)


# ─── match_shapes ─────────────────────────────────────────────────────────────

class TestMatchShapes:
    def test_returns_shape_match_result(self):
        c1 = make_square()
        c2 = make_square()
        result = match_shapes(c1, c2)
        assert isinstance(result, ShapeMatchResult)

    def test_score_in_0_1(self):
        c1 = make_square()
        c2 = make_circle()
        result = match_shapes(c1, c2)
        assert 0.0 <= result.score <= 1.0

    def test_identical_shapes_high_score(self):
        c = make_square(size=40, offset=5)
        result = match_shapes(c, c.copy())
        assert result.score > 0.5

    def test_idx_stored(self):
        c1 = make_square()
        c2 = make_circle()
        result = match_shapes(c1, c2, idx1=3, idx2=7)
        assert result.idx1 == 3
        assert result.idx2 == 7

    def test_zero_weight_sum_raises(self):
        c1 = make_square()
        c2 = make_circle()
        with pytest.raises(ValueError, match="weight"):
            match_shapes(c1, c2, weights=(0.0, 0.0, 0.0))

    def test_hu_dist_nonnegative(self):
        c1 = make_square()
        c2 = make_circle()
        result = match_shapes(c1, c2)
        assert result.hu_dist >= 0.0

    def test_chamfer_nonnegative(self):
        c1 = make_square()
        c2 = make_circle()
        result = match_shapes(c1, c2)
        assert result.chamfer >= 0.0

    def test_iou_in_0_1(self):
        c1 = make_square()
        c2 = make_circle()
        result = match_shapes(c1, c2)
        assert 0.0 <= result.iou <= 1.0

    def test_params_stored(self):
        c1 = make_square()
        c2 = make_circle()
        result = match_shapes(c1, c2)
        assert "w_hu" in result.params
        assert "max_chamfer" in result.params

    def test_wrong_shape_raises(self):
        c1 = np.ones((5, 3))
        c2 = make_square()
        with pytest.raises(ValueError):
            match_shapes(c1, c2)

    def test_custom_weights(self):
        c1 = make_square()
        c2 = make_square()
        result = match_shapes(c1, c2, weights=(1.0, 0.0, 0.0))
        assert 0.0 <= result.score <= 1.0


# ─── find_best_shape_match ────────────────────────────────────────────────────

class TestFindBestShapeMatch:
    def test_empty_candidates_returns_none(self):
        query = make_square()
        result = find_best_shape_match(query, [])
        assert result is None

    def test_single_candidate(self):
        query = make_square()
        cand = make_circle()
        result = find_best_shape_match(query, [cand])
        assert isinstance(result, ShapeMatchResult)

    def test_returns_best_among_candidates(self):
        query = make_square(size=40, offset=5)
        # One candidate identical to query, one very different
        c_same = make_square(size=40, offset=5)
        c_diff = make_circle(radius=5, cx=200, cy=200)
        result = find_best_shape_match(query, [c_diff, c_same])
        # Best match should be the identical square (index 1)
        assert result.idx2 == 1

    def test_query_idx_stored(self):
        query = make_square()
        cand = make_circle()
        result = find_best_shape_match(query, [cand], query_idx=5)
        assert result.idx1 == 5

    def test_score_in_0_1(self):
        query = make_square()
        candidates = [make_circle(), make_triangle()]
        result = find_best_shape_match(query, candidates)
        assert 0.0 <= result.score <= 1.0

    def test_multiple_candidates(self):
        query = make_square()
        candidates = [make_circle(), make_triangle(), make_square()]
        result = find_best_shape_match(query, candidates)
        assert result is not None
        assert isinstance(result, ShapeMatchResult)


# ─── batch_match_shapes ───────────────────────────────────────────────────────

class TestBatchMatchShapes:
    def test_returns_float32(self):
        c1s = [make_square(), make_circle()]
        c2s = [make_triangle(), make_square()]
        result = batch_match_shapes(c1s, c2s)
        assert result.dtype == np.float32

    def test_shape_n_m(self):
        c1s = [make_square(), make_circle(), make_triangle()]
        c2s = [make_square(), make_circle()]
        result = batch_match_shapes(c1s, c2s)
        assert result.shape == (3, 2)

    def test_empty_contours1(self):
        c2s = [make_square()]
        result = batch_match_shapes([], c2s)
        assert result.shape[0] == 0

    def test_empty_contours2(self):
        c1s = [make_square()]
        result = batch_match_shapes(c1s, [])
        assert result.shape[1] == 0

    def test_both_empty(self):
        result = batch_match_shapes([], [])
        assert result.shape == (0, 0)

    def test_values_in_0_1(self):
        c1s = [make_square(), make_circle()]
        c2s = [make_square(), make_triangle()]
        result = batch_match_shapes(c1s, c2s)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_1x1_case(self):
        c1s = [make_square()]
        c2s = [make_square()]
        result = batch_match_shapes(c1s, c2s)
        assert result.shape == (1, 1)
        assert 0.0 <= float(result[0, 0]) <= 1.0
