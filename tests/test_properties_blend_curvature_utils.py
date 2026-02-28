"""
Property-based tests for:
  1. puzzle_reconstruction.utils.blend_utils
  2. puzzle_reconstruction.utils.curvature_utils

Verifies mathematical invariants:

blend_utils:
- BlendConfig:           feather_px >=0; gamma >0; invalid raises
- alpha_blend:           same shape as inputs; output uint8; alpha=0 → dst;
                         alpha=1 → src; shape mismatch raises; α ∉[0,1] raises
- weighted_blend:        same shape; output uint8; equal weights = arithmetic mean;
                         single image = identity; mismatched shapes raise
- feather_mask:          shape = (h, w); values ∈ [0, 1]; center > edges;
                         feather_px=0 → all ones
- horizontal_blend:      same height as inputs; width = sum of widths;
                         dtype uint8
- vertical_blend:        same width as inputs; height = sum of heights;
                         dtype uint8
- batch_blend:           same count as pairs; each same shape as inputs;
                         raises on unknown mode

curvature_utils:
- CurvatureConfig:       corner_threshold >0; min_distance >=1; invalid raises
- compute_curvature:     length = N; < 3 points raises; straight line ≈ 0;
                         float64
- compute_total_curvature: ≥ 0; straight line ≈ 0
- find_inflection_points: indices in [0, N-1]; monotone increasing;
                          straight line → 0 inflections
- compute_turning_angle: straight line ≈ 0; N<2 raises; returns float
- smooth_curvature:      same length; constant → identity; sigma≤0 raises;
                         2d raises
- corner_score:          ∈ [0, 1]; straight line → all zeros; length = N
- find_corners:          indices in [0, N-1]; monotone increasing
- batch_curvature:       same count as input; each length = N; empty raises
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

from puzzle_reconstruction.utils.blend_utils import (
    BlendConfig,
    alpha_blend,
    weighted_blend,
    feather_mask,
    horizontal_blend,
    vertical_blend,
    batch_blend,
)
from puzzle_reconstruction.utils.curvature_utils import (
    CurvatureConfig,
    compute_curvature,
    compute_total_curvature,
    find_inflection_points,
    compute_turning_angle,
    smooth_curvature,
    corner_score,
    find_corners,
    batch_curvature,
)

RNG = np.random.default_rng(55)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _img(h: int = 20, w: int = 20, channels: int = 1) -> np.ndarray:
    if channels == 1:
        return RNG.integers(0, 256, size=(h, w), dtype=np.uint8)
    return RNG.integers(0, 256, size=(h, w, channels), dtype=np.uint8)


def _circle_curve(n: int = 30, r: float = 10.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _line_curve(n: int = 20) -> np.ndarray:
    x = np.linspace(0.0, 10.0, n)
    y = np.linspace(0.0, 10.0, n)
    return np.stack([x, y], axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# BlendConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestBlendConfig:

    def test_default_valid(self):
        cfg = BlendConfig()
        assert cfg.feather_px >= 0
        assert cfg.gamma > 0

    def test_raises_negative_feather(self):
        with pytest.raises(ValueError):
            BlendConfig(feather_px=-1)

    def test_raises_zero_gamma(self):
        with pytest.raises(ValueError):
            BlendConfig(gamma=0.0)

    def test_raises_negative_gamma(self):
        with pytest.raises(ValueError):
            BlendConfig(gamma=-0.5)

    def test_feather_zero_valid(self):
        cfg = BlendConfig(feather_px=0)
        assert cfg.feather_px == 0


# ═══════════════════════════════════════════════════════════════════════════════
# alpha_blend
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlphaBlend:

    def test_same_shape(self):
        src = _img(20, 20)
        dst = _img(20, 20)
        out = alpha_blend(src, dst, alpha=0.5)
        assert out.shape == (20, 20)

    def test_dtype_uint8(self):
        src = _img(10, 10)
        dst = _img(10, 10)
        out = alpha_blend(src, dst, alpha=0.3)
        assert out.dtype == np.uint8

    def test_alpha_zero_returns_dst(self):
        src = np.full((10, 10), 200, dtype=np.uint8)
        dst = np.full((10, 10), 100, dtype=np.uint8)
        out = alpha_blend(src, dst, alpha=0.0)
        assert np.array_equal(out, dst)

    def test_alpha_one_returns_src(self):
        src = np.full((10, 10), 200, dtype=np.uint8)
        dst = np.full((10, 10), 100, dtype=np.uint8)
        out = alpha_blend(src, dst, alpha=1.0)
        assert np.array_equal(out, src)

    def test_alpha_half_in_middle(self):
        src = np.full((10, 10), 200, dtype=np.uint8)
        dst = np.full((10, 10), 100, dtype=np.uint8)
        out = alpha_blend(src, dst, alpha=0.5)
        assert np.all(out == 150)

    def test_shape_mismatch_raises(self):
        src = _img(10, 10)
        dst = _img(20, 20)
        with pytest.raises(ValueError):
            alpha_blend(src, dst, alpha=0.5)

    def test_alpha_out_of_range_raises(self):
        src = _img(10, 10)
        dst = _img(10, 10)
        with pytest.raises(ValueError):
            alpha_blend(src, dst, alpha=1.5)
        with pytest.raises(ValueError):
            alpha_blend(src, dst, alpha=-0.1)

    def test_3d_image(self):
        src = _img(10, 10, channels=3)
        dst = _img(10, 10, channels=3)
        out = alpha_blend(src, dst, alpha=0.5)
        assert out.shape == (10, 10, 3)

    def test_output_in_valid_range(self):
        src = _img(15, 15)
        dst = _img(15, 15)
        out = alpha_blend(src, dst, alpha=0.7)
        assert int(out.min()) >= 0
        assert int(out.max()) <= 255


# ═══════════════════════════════════════════════════════════════════════════════
# weighted_blend
# ═══════════════════════════════════════════════════════════════════════════════

class TestWeightedBlend:

    def test_same_shape(self):
        images = [_img(10, 10) for _ in range(3)]
        out = weighted_blend(images)
        assert out.shape == (10, 10)

    def test_dtype_uint8(self):
        images = [_img(10, 10) for _ in range(2)]
        out = weighted_blend(images)
        assert out.dtype == np.uint8

    def test_single_image_identity(self):
        img = np.full((10, 10), 128, dtype=np.uint8)
        out = weighted_blend([img])
        assert np.array_equal(out, img)

    def test_equal_weights_is_mean(self):
        img1 = np.full((10, 10), 100, dtype=np.uint8)
        img2 = np.full((10, 10), 200, dtype=np.uint8)
        out = weighted_blend([img1, img2])
        assert np.all(out == 150)

    def test_output_in_valid_range(self):
        images = [_img(10, 10) for _ in range(4)]
        out = weighted_blend(images)
        assert int(out.min()) >= 0
        assert int(out.max()) <= 255

    def test_raises_empty(self):
        with pytest.raises(ValueError):
            weighted_blend([])

    def test_raises_mismatched_shapes(self):
        with pytest.raises(ValueError):
            weighted_blend([_img(10, 10), _img(20, 20)])

    def test_raises_wrong_weight_count(self):
        with pytest.raises(ValueError):
            weighted_blend([_img(5, 5), _img(5, 5)], weights=[1.0, 2.0, 3.0])


# ═══════════════════════════════════════════════════════════════════════════════
# feather_mask
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatherMask:

    def test_shape(self):
        mask = feather_mask(20, 30, feather_px=5)
        assert mask.shape == (20, 30)

    def test_values_in_zero_one(self):
        mask = feather_mask(20, 20, feather_px=5)
        assert float(mask.min()) >= 0.0 - 1e-8
        assert float(mask.max()) <= 1.0 + 1e-8

    def test_zero_feather_all_ones(self):
        mask = feather_mask(20, 20, feather_px=0)
        assert np.allclose(mask, 1.0, atol=1e-8)

    def test_center_ge_edges(self):
        mask = feather_mask(40, 40, feather_px=10)
        center = float(mask[20, 20])
        corners = [float(mask[0, 0]), float(mask[0, -1]),
                   float(mask[-1, 0]), float(mask[-1, -1])]
        for c in corners:
            assert center >= c - 1e-8

    def test_dtype_float(self):
        mask = feather_mask(10, 10, feather_px=2)
        assert mask.dtype in (np.float32, np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# horizontal_blend / vertical_blend
# ═══════════════════════════════════════════════════════════════════════════════

class TestHorizontalBlend:

    def test_same_height(self):
        left = _img(20, 15)
        right = _img(20, 25)
        out = horizontal_blend(left, right, overlap=0)
        assert out.shape[0] == 20

    def test_width_is_sum_no_overlap(self):
        left = _img(20, 15)
        right = _img(20, 25)
        out = horizontal_blend(left, right, overlap=0)
        assert out.shape[1] == 40

    def test_width_with_overlap(self):
        left = _img(20, 15)
        right = _img(20, 25)
        out = horizontal_blend(left, right, overlap=5)
        assert out.shape[1] == 35  # 15 + 25 - 5

    def test_dtype_uint8(self):
        left = _img(10, 10)
        right = _img(10, 10)
        out = horizontal_blend(left, right, overlap=0)
        assert out.dtype == np.uint8

    def test_raises_height_mismatch(self):
        left = _img(10, 10)
        right = _img(20, 10)
        with pytest.raises(ValueError):
            horizontal_blend(left, right, overlap=0)

    def test_raises_negative_overlap(self):
        left = _img(10, 10)
        right = _img(10, 10)
        with pytest.raises(ValueError):
            horizontal_blend(left, right, overlap=-1)


class TestVerticalBlend:

    def test_same_width(self):
        top = _img(15, 20)
        bottom = _img(25, 20)
        out = vertical_blend(top, bottom, overlap=0)
        assert out.shape[1] == 20

    def test_height_is_sum_no_overlap(self):
        top = _img(15, 20)
        bottom = _img(25, 20)
        out = vertical_blend(top, bottom, overlap=0)
        assert out.shape[0] == 40

    def test_height_with_overlap(self):
        top = _img(15, 20)
        bottom = _img(25, 20)
        out = vertical_blend(top, bottom, overlap=5)
        assert out.shape[0] == 35  # 15 + 25 - 5

    def test_dtype_uint8(self):
        top = _img(10, 10)
        bottom = _img(10, 10)
        out = vertical_blend(top, bottom, overlap=0)
        assert out.dtype == np.uint8

    def test_raises_width_mismatch(self):
        top = _img(10, 10)
        bottom = _img(10, 20)
        with pytest.raises(ValueError):
            vertical_blend(top, bottom, overlap=0)


# ═══════════════════════════════════════════════════════════════════════════════
# batch_blend
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchBlend:

    def test_same_count(self):
        pairs = [(_img(10, 10), _img(10, 10)) for _ in range(3)]
        out = batch_blend(pairs, alpha=0.5)
        assert len(out) == 3

    def test_each_same_shape(self):
        pairs = [(_img(10, 10), _img(10, 10))]
        out = batch_blend(pairs, alpha=0.5)
        assert out[0].shape == (10, 10)

    def test_dtype_uint8(self):
        pairs = [(_img(8, 8), _img(8, 8))]
        out = batch_blend(pairs, alpha=0.3)
        assert out[0].dtype == np.uint8

    def test_empty_list(self):
        out = batch_blend([], alpha=0.5)
        assert out == []


# ═══════════════════════════════════════════════════════════════════════════════
# CurvatureConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestCurvatureConfig:

    def test_default_valid(self):
        cfg = CurvatureConfig()
        assert cfg.corner_threshold > 0
        assert cfg.min_distance >= 1

    def test_raises_zero_corner_threshold(self):
        with pytest.raises(ValueError):
            CurvatureConfig(corner_threshold=0.0)

    def test_raises_negative_corner_threshold(self):
        with pytest.raises(ValueError):
            CurvatureConfig(corner_threshold=-0.1)

    def test_raises_zero_min_distance(self):
        with pytest.raises(ValueError):
            CurvatureConfig(min_distance=0)


# ═══════════════════════════════════════════════════════════════════════════════
# compute_curvature
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeCurvature:

    def test_length_equals_n(self):
        curve = _circle_curve(30)
        kappa = compute_curvature(curve)
        assert len(kappa) == 30

    def test_dtype_float64(self):
        curve = _circle_curve(20)
        kappa = compute_curvature(curve)
        assert kappa.dtype == np.float64

    def test_raises_less_than_3_points(self):
        with pytest.raises(ValueError):
            compute_curvature(np.array([[0.0, 0.0], [1.0, 1.0]]))

    def test_raises_wrong_shape(self):
        with pytest.raises(ValueError):
            compute_curvature(np.ones((10, 3)))

    def test_straight_line_near_zero(self):
        curve = _line_curve(20)
        cfg = CurvatureConfig(smooth_sigma=0.0)
        kappa = compute_curvature(curve, cfg)
        # Straight line has zero curvature (up to numerical precision)
        assert np.all(np.abs(kappa) < 1e-8)

    def test_circle_curvature_constant(self):
        r = 5.0
        curve = _circle_curve(100, r=r)
        cfg = CurvatureConfig(smooth_sigma=1.0)
        kappa = compute_curvature(curve, cfg)
        # For a circle of radius r, |κ| ≈ 1/r
        # With discrete approx and smoothing, check all values are similar sign
        # and roughly constant magnitude
        # Use inner points to avoid boundary effects
        inner = np.abs(kappa[10:-10])
        assert float(inner.std()) / (float(inner.mean()) + 1e-10) < 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# compute_total_curvature
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeTotalCurvature:

    def test_non_negative(self):
        curve = _circle_curve(30)
        tc = compute_total_curvature(curve)
        assert tc >= 0.0

    def test_straight_line_near_zero(self):
        curve = _line_curve(20)
        cfg = CurvatureConfig(smooth_sigma=0.0)
        tc = compute_total_curvature(curve, cfg)
        assert tc < 1e-7

    def test_raises_less_than_3_points(self):
        with pytest.raises(ValueError):
            compute_total_curvature(np.array([[0.0, 0.0], [1.0, 1.0]]))

    def test_curved_path_greater_than_straight(self):
        straight = _line_curve(30)
        circle = _circle_curve(30)
        tc_straight = compute_total_curvature(straight, CurvatureConfig(smooth_sigma=0.0))
        tc_circle = compute_total_curvature(circle)
        assert tc_circle > tc_straight


# ═══════════════════════════════════════════════════════════════════════════════
# find_inflection_points
# ═══════════════════════════════════════════════════════════════════════════════

class TestFindInflectionPoints:

    def test_indices_in_range(self):
        curve = _circle_curve(30)
        inf_pts = find_inflection_points(curve)
        if len(inf_pts) > 0:
            assert int(inf_pts.min()) >= 0
            assert int(inf_pts.max()) < 30

    def test_monotone_increasing(self):
        curve = _circle_curve(50)
        inf_pts = find_inflection_points(curve)
        for i in range(len(inf_pts) - 1):
            assert inf_pts[i] < inf_pts[i + 1]

    def test_straight_line_no_inflections(self):
        curve = _line_curve(20)
        cfg = CurvatureConfig(smooth_sigma=0.0)
        inf_pts = find_inflection_points(curve, cfg)
        assert len(inf_pts) == 0

    def test_dtype_int64(self):
        curve = _circle_curve(30)
        inf_pts = find_inflection_points(curve)
        assert inf_pts.dtype == np.int64


# ═══════════════════════════════════════════════════════════════════════════════
# compute_turning_angle
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeTurningAngle:

    def test_returns_float(self):
        curve = _circle_curve(20)
        angle = compute_turning_angle(curve)
        assert isinstance(angle, float)

    def test_straight_line_near_zero(self):
        curve = _line_curve(20)
        angle = compute_turning_angle(curve)
        assert abs(angle) < 1e-8

    def test_raises_less_than_2_points(self):
        with pytest.raises(ValueError):
            compute_turning_angle(np.array([[0.0, 0.0]]))

    def test_raises_wrong_shape(self):
        with pytest.raises(ValueError):
            compute_turning_angle(np.ones((10, 3)))

    def test_circle_turning_angle(self):
        # A full circle should give turning angle ≈ ±2π
        # Use many points for accuracy
        curve = _circle_curve(200)
        angle = compute_turning_angle(curve)
        # Total turning angle ≈ 2π (positive or negative)
        assert abs(abs(angle) - 2 * math.pi) < 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# smooth_curvature
# ═══════════════════════════════════════════════════════════════════════════════

class TestSmoothCurvature:

    def test_same_length(self):
        kappa = np.random.default_rng(1).uniform(-1.0, 1.0, 30)
        out = smooth_curvature(kappa, sigma=1.0)
        assert len(out) == 30

    def test_constant_preserved(self):
        kappa = np.full(20, 3.0)
        out = smooth_curvature(kappa, sigma=1.0)
        assert np.allclose(out, 3.0, atol=1e-8)

    def test_raises_zero_sigma(self):
        with pytest.raises(ValueError):
            smooth_curvature(np.ones(10), sigma=0.0)

    def test_raises_2d_input(self):
        with pytest.raises(ValueError):
            smooth_curvature(np.ones((5, 5)), sigma=1.0)

    def test_dtype_float64(self):
        kappa = np.ones(15)
        out = smooth_curvature(kappa)
        assert out.dtype == np.float64


# ═══════════════════════════════════════════════════════════════════════════════
# corner_score
# ═══════════════════════════════════════════════════════════════════════════════

class TestCornerScore:

    def test_in_zero_one(self):
        curve = _circle_curve(30)
        scores = corner_score(curve)
        assert float(scores.min()) >= 0.0 - 1e-8
        assert float(scores.max()) <= 1.0 + 1e-8

    def test_straight_line_all_zeros(self):
        curve = _line_curve(20)
        cfg = CurvatureConfig(smooth_sigma=0.0)
        scores = corner_score(curve, cfg)
        assert np.all(scores < 1e-8)

    def test_length_equals_n(self):
        curve = _circle_curve(25)
        scores = corner_score(curve)
        assert len(scores) == 25

    def test_max_is_one(self):
        curve = _circle_curve(30)
        scores = corner_score(curve)
        if float(scores.max()) > 1e-10:
            assert abs(float(scores.max()) - 1.0) < 1e-8


# ═══════════════════════════════════════════════════════════════════════════════
# find_corners
# ═══════════════════════════════════════════════════════════════════════════════

class TestFindCorners:

    def test_indices_in_range(self):
        curve = _circle_curve(40)
        corners = find_corners(curve)
        if len(corners) > 0:
            assert int(corners.min()) >= 0
            assert int(corners.max()) < 40

    def test_monotone_increasing(self):
        curve = _circle_curve(50)
        corners = find_corners(curve)
        for i in range(len(corners) - 1):
            assert corners[i] < corners[i + 1]

    def test_straight_line_no_corners(self):
        curve = _line_curve(20)
        cfg = CurvatureConfig(smooth_sigma=0.0, corner_threshold=0.01)
        corners = find_corners(curve, cfg)
        assert len(corners) == 0

    def test_dtype_int64(self):
        curve = _circle_curve(30)
        corners = find_corners(curve)
        assert corners.dtype == np.int64

    def test_min_distance_respected(self):
        curve = _circle_curve(50)
        cfg = CurvatureConfig(min_distance=5, corner_threshold=0.01)
        corners = find_corners(curve, cfg)
        for i in range(len(corners) - 1):
            assert int(corners[i + 1]) - int(corners[i]) >= 5


# ═══════════════════════════════════════════════════════════════════════════════
# batch_curvature
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchCurvature:

    def test_same_count(self):
        curves = [_circle_curve(20) for _ in range(4)]
        results = batch_curvature(curves)
        assert len(results) == 4

    def test_each_correct_length(self):
        curves = [_circle_curve(20), _circle_curve(30)]
        results = batch_curvature(curves)
        assert len(results[0]) == 20
        assert len(results[1]) == 30

    def test_raises_empty(self):
        with pytest.raises(ValueError):
            batch_curvature([])

    def test_each_dtype_float64(self):
        curves = [_circle_curve(20)]
        results = batch_curvature(curves)
        assert results[0].dtype == np.float64
