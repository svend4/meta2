"""
Integration tests for puzzle_reconstruction.utils modules.

~55 tests across 5 classes covering:
- geometry: polygon_area, poly_iou, normalize_contour, rotate_points, resample_curve
- distance_utils: euclidean_distance, cosine_similarity, hausdorff_distance, pairwise_distances
- array_utils: pad_to_shape, crop_center, normalize_array, flatten_images
- mask_utils: apply_mask, erode_mask, dilate_mask, combine_masks, invert_mask
- integration: chained utility pipelines
"""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.geometry import (
    bbox_from_points,
    normalize_contour,
    point_in_polygon,
    poly_iou,
    polygon_area,
    polygon_centroid,
    resample_curve,
    rotate_points,
    rotation_matrix_2d,
)
from puzzle_reconstruction.utils.distance_utils import (
    chamfer_distance,
    cosine_similarity,
    euclidean_distance,
    hausdorff_distance,
    pairwise_distances,
)
from puzzle_reconstruction.utils.array_utils import (
    crop_center,
    flatten_images,
    normalize_array,
    pad_to_shape,
    stack_arrays,
)
from puzzle_reconstruction.utils.mask_utils import (
    apply_mask,
    combine_masks,
    dilate_mask,
    erode_mask,
    invert_mask,
    mask_from_contour,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def square_polygon() -> np.ndarray:
    return np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)


def triangle_polygon() -> np.ndarray:
    return np.array([[0, 0], [10, 0], [5, 10]], dtype=np.float64)


def make_mask(h: int = 60, w: int = 60, fill: bool = True) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    if fill:
        m[10:50, 10:50] = 255
    return m


def make_image(h: int = 60, w: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


# ─── TestGeometry ─────────────────────────────────────────────────────────────

class TestGeometry:
    """Tests for puzzle_reconstruction.utils.geometry functions."""

    def test_polygon_area_square(self):
        area = polygon_area(square_polygon())
        assert area == pytest.approx(100.0, abs=1e-5)

    def test_polygon_area_triangle(self):
        area = polygon_area(triangle_polygon())
        assert area == pytest.approx(50.0, abs=1e-5)

    def test_polygon_area_non_negative(self):
        area = polygon_area(square_polygon())
        assert area >= 0.0

    def test_poly_iou_identical(self):
        sq = square_polygon()
        iou = poly_iou(sq, sq.copy())
        assert iou == pytest.approx(1.0, abs=0.05)

    def test_poly_iou_no_overlap(self):
        sq1 = square_polygon()
        sq2 = sq1 + np.array([100, 100])
        iou = poly_iou(sq1, sq2)
        assert iou == pytest.approx(0.0, abs=0.05)

    def test_poly_iou_in_unit_interval(self):
        sq = square_polygon()
        shifted = sq + np.array([5, 0])
        iou = poly_iou(sq, shifted)
        assert 0.0 <= iou <= 1.0

    def test_normalize_contour_returns_ndarray(self):
        pts = square_polygon()
        result = normalize_contour(pts)
        assert isinstance(result, np.ndarray)

    def test_normalize_contour_scaled(self):
        pts = square_polygon()
        result = normalize_contour(pts, target_scale=1.0)
        assert result.shape == pts.shape

    def test_rotate_points_shape_preserved(self):
        pts = square_polygon()
        rotated = rotate_points(pts, angle=45.0)
        assert rotated.shape == pts.shape

    def test_rotate_2pi_is_identity(self):
        pts = square_polygon()
        rotated = rotate_points(pts, angle=2 * np.pi)
        assert np.allclose(rotated, pts, atol=1e-5)

    def test_resample_curve_output_count(self):
        pts = square_polygon()
        resampled = resample_curve(pts, n=8)
        assert resampled.shape[0] == 8

    def test_resample_curve_2d(self):
        pts = square_polygon()
        resampled = resample_curve(pts, n=5)
        assert resampled.shape[1] == 2

    def test_rotation_matrix_shape(self):
        R = rotation_matrix_2d(45.0)
        assert R.shape == (2, 2)

    def test_rotation_matrix_orthogonal(self):
        R = rotation_matrix_2d(30.0)
        assert np.allclose(R @ R.T, np.eye(2), atol=1e-6)


# ─── TestDistanceUtils ────────────────────────────────────────────────────────

class TestDistanceUtils:
    """Tests for puzzle_reconstruction.utils.distance_utils functions."""

    def test_euclidean_distance_zero(self):
        a = np.array([1.0, 2.0, 3.0])
        assert euclidean_distance(a, a) == pytest.approx(0.0)

    def test_euclidean_distance_known(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert euclidean_distance(a, b) == pytest.approx(5.0)

    def test_euclidean_distance_non_negative(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(10)
        b = rng.standard_normal(10)
        assert euclidean_distance(a, b) >= 0.0

    def test_cosine_similarity_identical(self):
        a = np.array([1.0, 0.0, 1.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-5)

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_cosine_similarity_in_range(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal(20)
        b = rng.standard_normal(20)
        sim = cosine_similarity(a, b)
        assert -1.0 - 1e-5 <= sim <= 1.0 + 1e-5

    def test_hausdorff_distance_zero_identical(self):
        pts = square_polygon()
        assert hausdorff_distance(pts, pts) == pytest.approx(0.0, abs=1e-5)

    def test_hausdorff_distance_non_negative(self):
        pts_a = square_polygon()
        pts_b = triangle_polygon()
        assert hausdorff_distance(pts_a, pts_b) >= 0.0

    def test_chamfer_distance_zero_identical(self):
        pts = square_polygon()
        assert chamfer_distance(pts, pts) == pytest.approx(0.0, abs=1e-5)

    def test_pairwise_distances_shape(self):
        X = np.random.default_rng(0).standard_normal((5, 3))
        D = pairwise_distances(X)
        assert D.shape == (5, 5)

    def test_pairwise_distances_diagonal_zero(self):
        X = np.random.default_rng(0).standard_normal((4, 3))
        D = pairwise_distances(X)
        assert np.allclose(np.diag(D), 0.0, atol=1e-5)

    def test_pairwise_distances_symmetric(self):
        X = np.random.default_rng(0).standard_normal((4, 3))
        D = pairwise_distances(X)
        assert np.allclose(D, D.T, atol=1e-5)


# ─── TestArrayUtils ───────────────────────────────────────────────────────────

class TestArrayUtils:
    """Tests for puzzle_reconstruction.utils.array_utils functions."""

    def test_pad_to_shape_output_shape(self):
        arr = np.zeros((3, 4))
        padded = pad_to_shape(arr, (6, 8))
        assert padded.shape == (6, 8)

    def test_pad_to_shape_original_preserved(self):
        arr = np.ones((3, 4))
        padded = pad_to_shape(arr, (6, 8))
        assert np.all(padded[:3, :4] == 1.0)

    def test_pad_to_shape_padding_value(self):
        arr = np.ones((2, 2))
        padded = pad_to_shape(arr, (4, 4), value=99.0)
        assert padded[3, 3] == pytest.approx(99.0)

    def test_crop_center_output_shape(self):
        arr = np.ones((60, 60, 3), dtype=np.uint8)
        cropped = crop_center(arr, (40, 40))
        assert cropped.shape == (40, 40, 3)

    def test_crop_center_smaller_than_input(self):
        arr = np.arange(100).reshape(10, 10)
        cropped = crop_center(arr, (6, 6))
        assert cropped.shape == (6, 6)

    def test_normalize_array_range(self):
        arr = np.array([0.0, 5.0, 10.0])
        result = normalize_array(arr, low=0.0, high=1.0)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_normalize_array_custom_range(self):
        arr = np.array([0.0, 5.0, 10.0])
        result = normalize_array(arr, low=-1.0, high=1.0)
        assert result.min() == pytest.approx(-1.0)
        assert result.max() == pytest.approx(1.0)

    def test_flatten_images_shape(self):
        imgs = [np.ones((10, 10, 3), dtype=np.uint8) for _ in range(4)]
        flat = flatten_images(imgs)
        assert flat.shape == (4, 300)

    def test_flatten_images_count(self):
        imgs = [np.zeros((8, 8), dtype=np.float32) for _ in range(3)]
        flat = flatten_images(imgs)
        assert flat.shape[0] == 3

    def test_stack_arrays_returns_ndarray(self):
        arrs = [np.ones((5, 3)) for _ in range(3)]
        stacked = stack_arrays(arrs)
        assert isinstance(stacked, np.ndarray)
        assert stacked.shape[0] == 3


# ─── TestMaskUtils ────────────────────────────────────────────────────────────

class TestMaskUtils:
    """Tests for puzzle_reconstruction.utils.mask_utils functions."""

    def test_apply_mask_output_shape(self):
        img = make_image()
        mask = make_mask()
        result = apply_mask(img, mask)
        assert result.shape == img.shape

    def test_apply_mask_zeros_outside(self):
        img = np.ones((60, 60, 3), dtype=np.uint8) * 200
        mask = np.zeros((60, 60), dtype=np.uint8)
        result = apply_mask(img, mask, fill=0)
        assert np.all(result == 0)

    def test_erode_mask_returns_ndarray(self):
        mask = make_mask()
        result = erode_mask(mask)
        assert isinstance(result, np.ndarray)

    def test_erode_mask_reduces_area(self):
        mask = make_mask()
        eroded = erode_mask(mask, ksize=3, iterations=2)
        assert eroded.sum() <= mask.sum()

    def test_dilate_mask_increases_area(self):
        mask = make_mask()
        dilated = dilate_mask(mask, ksize=3)
        assert dilated.sum() >= mask.sum()

    def test_invert_mask_flips_values(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 3:7] = 255
        inv = invert_mask(mask)
        assert inv[0, 0] != 0  # was 0, now non-zero
        assert inv[5, 5] == 0  # was 255, now 0

    def test_combine_masks_and(self):
        m1 = np.zeros((10, 10), dtype=np.uint8)
        m1[2:8, 2:8] = 255
        m2 = np.zeros((10, 10), dtype=np.uint8)
        m2[4:10, 4:10] = 255
        combined = combine_masks(m1, m2, mode="and")
        assert combined.sum() <= m1.sum()

    def test_combine_masks_or(self):
        m1 = np.zeros((10, 10), dtype=np.uint8)
        m1[0:5, 0:5] = 255
        m2 = np.zeros((10, 10), dtype=np.uint8)
        m2[5:10, 5:10] = 255
        combined = combine_masks(m1, m2, mode="or")
        assert combined.sum() >= m1.sum()

    def test_mask_from_contour_returns_ndarray(self):
        contour = np.array([[5, 5], [55, 5], [55, 55], [5, 55]], dtype=np.int32)
        mask = mask_from_contour(contour, h=60, w=60)
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (60, 60)


# ─── TestUtilsIntegration ─────────────────────────────────────────────────────

class TestUtilsIntegration:
    """Integration tests: chained utility pipelines."""

    def test_geometry_then_distance(self):
        sq = square_polygon()
        norm = normalize_contour(sq)
        dist = hausdorff_distance(sq, norm)
        assert np.isfinite(dist)

    def test_rotate_zero_then_iou(self):
        sq = square_polygon()
        rotated = rotate_points(sq, angle=0.0)
        iou = poly_iou(sq, rotated)
        assert iou == pytest.approx(1.0, abs=0.1)

    def test_mask_pipeline_erode_dilate(self):
        mask = make_mask()
        eroded = erode_mask(mask, ksize=3)
        dilated = dilate_mask(eroded, ksize=3)
        # Dilating after eroding should not exceed original
        assert dilated.sum() <= mask.sum() * 1.1

    def test_array_utils_pipeline(self):
        imgs = [make_image(seed=i) for i in range(4)]
        flat = flatten_images(imgs)
        normalized = normalize_array(flat.astype(np.float32), low=0.0, high=1.0)
        assert normalized.min() == pytest.approx(0.0, abs=1e-5)
        assert normalized.max() == pytest.approx(1.0, abs=1e-5)

    def test_resample_and_distance(self):
        sq = square_polygon()
        resampled = resample_curve(sq, n=20)
        dist = chamfer_distance(sq, resampled)
        assert np.isfinite(dist)
        assert dist >= 0.0

    def test_pairwise_then_normalize(self):
        X = np.random.default_rng(0).standard_normal((5, 3))
        D = pairwise_distances(X)
        normed = normalize_array(D, low=0.0, high=1.0)
        assert normed.min() == pytest.approx(0.0, abs=1e-5)
        assert normed.max() == pytest.approx(1.0, abs=1e-5)

    def test_mask_and_apply(self):
        img = make_image()
        mask = make_mask()
        result = apply_mask(img, mask)
        # Pixels outside mask should be fill value
        assert result[0, 0, 0] == 0

    def test_pad_then_crop_roundtrip(self):
        arr = np.ones((20, 20))
        padded = pad_to_shape(arr, (40, 40))
        cropped = crop_center(padded, (20, 20))
        assert cropped.shape == (20, 20)
