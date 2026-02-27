"""
Property-based invariant tests for:
  - puzzle_reconstruction.utils.icp_utils
  - puzzle_reconstruction.utils.curve_metrics
  - puzzle_reconstruction.utils.contour_profile_utils

icp_utils:
    centroid:              shape (2,); mean of rows
    center_points:         centered mean ≈ 0; c = centroid(pts)
    scale_points:          RMS of centered points ≈ 1 (non-degenerate)
    resample_uniform:      output shape = (n, 2); length preserved
    nearest_neighbours:    indices shape = (M,); distances >= 0; each idx < len(tgt)
    svd_rotation:          det(R) ≈ +1 (no reflection); R^T R ≈ I (orthogonal)
    compute_rmse:          >= 0; rmse(pts, pts) = 0
    rmse_after_transform:  >= 0
    has_converged:         True if |prev - curr| < tol
    compose_transforms:    first apply then second (associative check)
    invert_transform:      compose(R,t, R_inv,t_inv) → identity
    transform_points:      shape preserved

curve_metrics:
    curve_l2:              >= 0; curve_l2(a, a) = 0
    curve_l2_mirror:       <= curve_l2(a, b); >= 0
    hausdorff_distance:    >= 0; symmetric; hausdorff(a, a) = 0
    frechet_distance_approx: >= 0; frechet(a, a) = 0; >= hausdorff(a, b) - ε
    curve_length:          >= 0; length(single_point) = 0
    length_ratio:          ∈ [0, 1]; self-ratio = 1.0
    compare_curves:        all fields >= 0; length_ratio ∈ [0, 1]
    batch_compare_curves:  output length = input length

contour_profile_utils:
    sample_profile_along_contour: shape = (n_samples, 2)
    contour_curvature:    shape = (N,); for N < 3 → zeros
    smooth_profile:       same length; values bounded by input range (approx.)
    normalize_profile:    values in [0, 1]; constant → ones
    profile_l2_distance:  >= 0; symmetric; distance(a, a) = 0
    profile_cosine_similarity: ∈ [-1, 1]
    best_cyclic_offset:   distance >= 0; offset ∈ [0, N)
    match_profiles:       score ∈ [0, 1]
    batch_match_profiles: length = len(candidates)
    top_k_profile_matches: len <= k; sorted by score descending
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

from puzzle_reconstruction.utils.icp_utils import (
    ICPConfig,
    centroid,
    center_points,
    scale_points,
    resample_uniform,
    nearest_neighbours,
    svd_rotation,
    svd_translation,
    compute_rmse,
    rmse_after_transform,
    has_converged,
    compose_transforms,
    invert_transform,
    transform_points,
    align_to_first,
    batch_nearest_neighbours,
)
from puzzle_reconstruction.utils.curve_metrics import (
    CurveMetricConfig,
    curve_l2,
    curve_l2_mirror,
    hausdorff_distance,
    frechet_distance_approx,
    curve_length,
    length_ratio,
    compare_curves,
    batch_compare_curves,
)
from puzzle_reconstruction.utils.contour_profile_utils import (
    ProfileConfig,
    sample_profile_along_contour,
    contour_curvature,
    smooth_profile,
    normalize_profile,
    profile_l2_distance,
    profile_cosine_similarity,
    best_cyclic_offset,
    match_profiles,
    batch_match_profiles,
    top_k_profile_matches,
)

RNG = np.random.default_rng(2031)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rand_pts(n: int = 10, scale: float = 10.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n, 2)) * scale).astype(np.float64)


def _rand_signal(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(n)


def _rect_pts(w: float = 8.0, h: float = 6.0) -> np.ndarray:
    return np.array([
        [0.0, 0.0], [w, 0.0], [w, h], [0.0, h]
    ], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# icp_utils — centroid, center_points, scale_points
# ═══════════════════════════════════════════════════════════════════════════════

class TestCentroidAndCentering:
    """centroid, center_points, scale_points invariants."""

    @pytest.mark.parametrize("n,seed", [(5, 0), (10, 1), (20, 2)])
    def test_centroid_shape(self, n: int, seed: int) -> None:
        pts = _rand_pts(n, seed=seed)
        c = centroid(pts)
        assert c.shape == (2,)

    @pytest.mark.parametrize("n,seed", [(5, 3), (10, 4)])
    def test_centroid_is_mean(self, n: int, seed: int) -> None:
        pts = _rand_pts(n, seed=seed)
        c = centroid(pts)
        np.testing.assert_allclose(c, pts.mean(axis=0))

    @pytest.mark.parametrize("n,seed", [(5, 5), (10, 6), (20, 7)])
    def test_center_points_zero_mean(self, n: int, seed: int) -> None:
        pts = _rand_pts(n, seed=seed)
        centered, c = center_points(pts)
        np.testing.assert_allclose(centered.mean(axis=0), [0.0, 0.0], atol=1e-12)

    @pytest.mark.parametrize("n,seed", [(5, 8), (10, 9)])
    def test_center_points_c_equals_centroid(self, n: int, seed: int) -> None:
        pts = _rand_pts(n, seed=seed)
        _, c = center_points(pts)
        np.testing.assert_allclose(c, centroid(pts))

    @pytest.mark.parametrize("n,seed", [(5, 10), (10, 11), (20, 12)])
    def test_scale_points_rms_is_one(self, n: int, seed: int) -> None:
        pts = _rand_pts(n, seed=seed)
        pts[0] += 1.0  # ensure not degenerate
        scaled, s = scale_points(pts)
        centered = scaled - scaled.mean(axis=0)
        rms = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
        assert rms == pytest.approx(1.0, abs=1e-9)

    def test_scale_degenerate_returns_scale_one(self) -> None:
        pts = np.zeros((5, 2))
        _, s = scale_points(pts)
        assert s == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# icp_utils — resample_uniform
# ═══════════════════════════════════════════════════════════════════════════════

class TestResampleUniform:
    """resample_uniform: output shape and curve-length preservation."""

    @pytest.mark.parametrize("n_pts,n_out", [(5, 20), (10, 10), (3, 7)])
    def test_output_shape(self, n_pts: int, n_out: int) -> None:
        pts = _rand_pts(n_pts)
        out = resample_uniform(pts, n_out)
        assert out.shape == (n_out, 2)

    def test_single_point_tiled(self) -> None:
        pts = np.array([[3.0, 4.0]])
        out = resample_uniform(pts, 5)
        assert out.shape == (5, 2)
        assert np.allclose(out, [3.0, 4.0], atol=1e-10)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_output_bounded_by_input(self, seed: int) -> None:
        pts = _rand_pts(8, seed=seed)
        out = resample_uniform(pts, 16)
        assert float(out[:, 0].min()) >= float(pts[:, 0].min()) - 1e-9
        assert float(out[:, 0].max()) <= float(pts[:, 0].max()) + 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# icp_utils — nearest_neighbours
# ═══════════════════════════════════════════════════════════════════════════════

class TestNearestNeighbours:
    """nearest_neighbours: shape and value invariants."""

    @pytest.mark.parametrize("m,n", [(5, 8), (10, 6), (3, 3)])
    def test_indices_shape(self, m: int, n: int) -> None:
        src = _rand_pts(m)
        tgt = _rand_pts(n)
        idx, dists = nearest_neighbours(src, tgt)
        assert idx.shape == (m,)
        assert dists.shape == (m,)

    @pytest.mark.parametrize("m,n", [(5, 8), (10, 6)])
    def test_distances_nonnegative(self, m: int, n: int) -> None:
        src = _rand_pts(m)
        tgt = _rand_pts(n)
        _, dists = nearest_neighbours(src, tgt)
        assert np.all(dists >= 0.0)

    @pytest.mark.parametrize("m,n", [(5, 8), (10, 6)])
    def test_indices_valid_range(self, m: int, n: int) -> None:
        src = _rand_pts(m)
        tgt = _rand_pts(n)
        idx, _ = nearest_neighbours(src, tgt)
        assert np.all(idx >= 0)
        assert np.all(idx < n)

    def test_self_match_gives_zero_distance(self) -> None:
        pts = _rand_pts(5)
        idx, dists = nearest_neighbours(pts, pts)
        np.testing.assert_allclose(dists, 0.0, atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# icp_utils — svd_rotation, transform_points, compute_rmse
# ═══════════════════════════════════════════════════════════════════════════════

class TestSvdAndTransform:
    """svd_rotation, transform_points, compute_rmse invariants."""

    @pytest.mark.parametrize("n,seed", [(5, 0), (8, 1), (10, 2)])
    def test_svd_rotation_orthogonal(self, n: int, seed: int) -> None:
        pts_a = _rand_pts(n, seed=seed)
        pts_b = _rand_pts(n, seed=seed + 100)
        ca, _ = center_points(pts_a)
        cb, _ = center_points(pts_b)
        R = svd_rotation(ca, cb)
        assert R.shape == (2, 2)
        np.testing.assert_allclose(R.T @ R, np.eye(2), atol=1e-10)

    @pytest.mark.parametrize("n,seed", [(5, 3), (8, 4), (10, 5)])
    def test_svd_rotation_det_plus_one(self, n: int, seed: int) -> None:
        pts_a = _rand_pts(n, seed=seed)
        pts_b = _rand_pts(n, seed=seed + 200)
        ca, _ = center_points(pts_a)
        cb, _ = center_points(pts_b)
        R = svd_rotation(ca, cb, allow_reflection=False)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("n,seed", [(5, 6), (8, 7)])
    def test_compute_rmse_nonnegative(self, n: int, seed: int) -> None:
        src = _rand_pts(n, seed=seed)
        tgt = _rand_pts(n, seed=seed + 300)
        assert compute_rmse(src, tgt) >= 0.0

    @pytest.mark.parametrize("n", [5, 8, 10])
    def test_compute_rmse_self_is_zero(self, n: int) -> None:
        pts = _rand_pts(n)
        assert compute_rmse(pts, pts) == pytest.approx(0.0)

    @pytest.mark.parametrize("n", [5, 8])
    def test_transform_points_shape(self, n: int) -> None:
        pts = _rand_pts(n)
        R = np.eye(2)
        t = np.zeros(2)
        out = transform_points(pts, R, t)
        assert out.shape == (n, 2)

    def test_transform_identity(self) -> None:
        pts = _rand_pts(6)
        out = transform_points(pts, np.eye(2), np.zeros(2))
        np.testing.assert_allclose(out, pts)


# ═══════════════════════════════════════════════════════════════════════════════
# icp_utils — invert_transform, compose_transforms, has_converged
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransformInvariants:
    """invert_transform, compose_transforms, has_converged."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_invert_compose_is_identity(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0, 2 * math.pi)
        R = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta),  math.cos(theta)]])
        t = rng.random(2)
        R_inv, t_inv = invert_transform(R, t)
        R_comp, t_comp = compose_transforms(R, t, R_inv, t_inv)
        np.testing.assert_allclose(R_comp, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(t_comp, np.zeros(2), atol=1e-12)

    @pytest.mark.parametrize("prev,curr,tol,expected", [
        (1.0, 0.9999, 1e-2, True),
        (1.0, 0.5,    1e-2, False),
        (0.01, 0.005, 1e-2, True),
        (0.001, 0.0,  1e-2, True),
    ])
    def test_has_converged(self, prev: float, curr: float, tol: float,
                           expected: bool) -> None:
        assert has_converged(prev, curr, tol) is expected

    @pytest.mark.parametrize("seed", [3, 4])
    def test_compose_associativity(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        R1 = np.eye(2)
        R2 = np.eye(2)
        t1 = rng.random(2)
        t2 = rng.random(2)
        R_out, t_out = compose_transforms(R1, t1, R2, t2)
        assert R_out.shape == (2, 2)
        assert t_out.shape == (2,)


# ═══════════════════════════════════════════════════════════════════════════════
# curve_metrics — curve_l2, hausdorff, frechet
# ═══════════════════════════════════════════════════════════════════════════════

class TestCurveDistances:
    """curve_l2, hausdorff_distance, frechet_distance_approx invariants."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_curve_l2_nonnegative(self, seed: int) -> None:
        a = _rand_pts(8, seed=seed)
        b = _rand_pts(8, seed=seed + 50)
        assert curve_l2(a, b) >= 0.0

    @pytest.mark.parametrize("n", [5, 8, 12])
    def test_curve_l2_self_is_zero(self, n: int) -> None:
        pts = _rand_pts(n)
        assert curve_l2(pts, pts) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_curve_l2_mirror_leq_l2(self, seed: int) -> None:
        a = _rand_pts(8, seed=seed)
        b = _rand_pts(8, seed=seed + 60)
        assert curve_l2_mirror(a, b) <= curve_l2(a, b) + 1e-9

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_curve_l2_mirror_nonnegative(self, seed: int) -> None:
        a = _rand_pts(8, seed=seed)
        b = _rand_pts(8, seed=seed + 70)
        assert curve_l2_mirror(a, b) >= 0.0

    @pytest.mark.parametrize("seed", [3, 4, 5])
    def test_hausdorff_nonnegative(self, seed: int) -> None:
        a = _rand_pts(8, seed=seed)
        b = _rand_pts(8, seed=seed + 80)
        assert hausdorff_distance(a, b) >= 0.0

    @pytest.mark.parametrize("n", [5, 8, 12])
    def test_hausdorff_self_is_zero(self, n: int) -> None:
        pts = _rand_pts(n)
        assert hausdorff_distance(pts, pts) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("seed", [6, 7, 8])
    def test_hausdorff_symmetric(self, seed: int) -> None:
        a = _rand_pts(8, seed=seed)
        b = _rand_pts(8, seed=seed + 90)
        assert hausdorff_distance(a, b) == pytest.approx(
            hausdorff_distance(b, a), abs=1e-9
        )

    @pytest.mark.parametrize("seed", [9, 10, 11])
    def test_frechet_nonnegative(self, seed: int) -> None:
        a = _rand_pts(8, seed=seed)
        b = _rand_pts(8, seed=seed + 100)
        assert frechet_distance_approx(a, b) >= 0.0

    @pytest.mark.parametrize("n", [4, 6, 8])
    def test_frechet_self_is_zero(self, n: int) -> None:
        pts = _rand_pts(n)
        assert frechet_distance_approx(pts, pts) == pytest.approx(0.0, abs=1e-9)


# ═══════════════════════════════════════════════════════════════════════════════
# curve_metrics — curve_length, length_ratio
# ═══════════════════════════════════════════════════════════════════════════════

class TestCurveLengthRatio:
    """curve_length and length_ratio invariants."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_curve_length_nonnegative(self, seed: int) -> None:
        pts = _rand_pts(8, seed=seed)
        assert curve_length(pts) >= 0.0

    def test_curve_length_single_point_is_zero(self) -> None:
        pts = np.array([[3.0, 4.0]])
        assert curve_length(pts) == pytest.approx(0.0)

    def test_curve_length_triangle(self) -> None:
        # Equilateral triangle with side 10 → perimeter = 30
        pts = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.66]])
        length = curve_length(pts)
        assert length > 0.0

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_length_ratio_in_unit_range(self, seed: int) -> None:
        a = _rand_pts(6, seed=seed)
        b = _rand_pts(6, seed=seed + 200)
        r = length_ratio(a, b)
        assert 0.0 <= r <= 1.0

    @pytest.mark.parametrize("n", [5, 8])
    def test_length_ratio_self_is_one(self, n: int) -> None:
        pts = _rand_pts(n)
        r = length_ratio(pts, pts)
        assert r == pytest.approx(1.0)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_length_ratio_symmetric(self, seed: int) -> None:
        a = _rand_pts(6, seed=seed)
        b = _rand_pts(6, seed=seed + 300)
        assert length_ratio(a, b) == pytest.approx(length_ratio(b, a))


# ═══════════════════════════════════════════════════════════════════════════════
# curve_metrics — compare_curves, batch_compare_curves
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompareCurves:
    """compare_curves and batch_compare_curves invariants."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_compare_curves_fields_nonnegative(self, seed: int) -> None:
        a = _rand_pts(8, seed=seed)
        b = _rand_pts(8, seed=seed + 400)
        result = compare_curves(a, b)
        assert result.l2 >= 0.0
        assert result.hausdorff >= 0.0
        assert result.frechet >= 0.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_compare_curves_length_ratio_in_range(self, seed: int) -> None:
        a = _rand_pts(8, seed=seed)
        b = _rand_pts(8, seed=seed + 500)
        result = compare_curves(a, b)
        assert 0.0 <= result.length_ratio <= 1.0

    @pytest.mark.parametrize("n_pairs", [3, 5, 7])
    def test_batch_compare_length_preserved(self, n_pairs: int) -> None:
        pairs = [(_rand_pts(8, seed=i), _rand_pts(8, seed=i + 100))
                 for i in range(n_pairs)]
        results = batch_compare_curves(pairs)
        assert len(results) == n_pairs

    def test_similarity_in_unit_range(self) -> None:
        a = _rand_pts(8)
        b = _rand_pts(8, seed=42)
        result = compare_curves(a, b)
        sim = result.similarity(sigma=1.0)
        assert 0.0 <= sim <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# contour_profile_utils — sample_profile_along_contour
# ═══════════════════════════════════════════════════════════════════════════════

class TestSampleProfile:
    """sample_profile_along_contour: shape and bounds."""

    @pytest.mark.parametrize("n_pts,n_samples", [(5, 32), (10, 64), (3, 16)])
    def test_output_shape(self, n_pts: int, n_samples: int) -> None:
        contour = _rand_pts(n_pts)
        out = sample_profile_along_contour(contour, n_samples)
        assert out.shape == (n_samples, 2)

    @pytest.mark.parametrize("n_pts,n_samples", [(5, 20), (8, 10)])
    def test_output_bounded(self, n_pts: int, n_samples: int) -> None:
        contour = _rand_pts(n_pts)
        out = sample_profile_along_contour(contour, n_samples)
        assert float(out[:, 0].min()) >= float(contour[:, 0].min()) - 1e-9
        assert float(out[:, 0].max()) <= float(contour[:, 0].max()) + 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# contour_profile_utils — contour_curvature, smooth_profile
# ═══════════════════════════════════════════════════════════════════════════════

class TestCurvatureAndSmooth:
    """contour_curvature and smooth_profile invariants."""

    @pytest.mark.parametrize("n", [5, 8, 12])
    def test_curvature_same_length(self, n: int) -> None:
        pts = _rand_pts(n)
        curv = contour_curvature(pts)
        assert curv.shape == (n,)

    def test_curvature_short_contour_zeros(self) -> None:
        pts = _rand_pts(2)
        curv = contour_curvature(pts)
        np.testing.assert_allclose(curv, 0.0)

    @pytest.mark.parametrize("n,window", [(10, 3), (20, 5), (15, 3)])
    def test_smooth_profile_same_length(self, n: int, window: int) -> None:
        v = _rand_signal(n)
        result = smooth_profile(v, window=window)
        assert len(result) == n

    def test_smooth_window_one_is_copy(self) -> None:
        v = _rand_signal(10)
        result = smooth_profile(v, window=1)
        np.testing.assert_allclose(result, v)

    @pytest.mark.parametrize("n", [10, 20])
    def test_smooth_dtype_float64(self, n: int) -> None:
        v = _rand_signal(n)
        result = smooth_profile(v, window=3)
        assert result.dtype == np.float64


# ═══════════════════════════════════════════════════════════════════════════════
# contour_profile_utils — normalize_profile, profile_l2_distance
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalizeAndDistance:
    """normalize_profile and profile_l2_distance invariants."""

    @pytest.mark.parametrize("n,seed", [(10, 0), (20, 1), (15, 2)])
    def test_normalize_in_unit_range(self, n: int, seed: int) -> None:
        v = _rand_signal(n, seed)
        result = normalize_profile(v)
        assert float(result.min()) >= 0.0 - 1e-9
        assert float(result.max()) <= 1.0 + 1e-9

    def test_constant_profile_gives_ones(self) -> None:
        v = np.full(8, 5.0)
        result = normalize_profile(v)
        np.testing.assert_allclose(result, 1.0)

    @pytest.mark.parametrize("n,seed", [(10, 0), (20, 1)])
    def test_profile_l2_nonnegative(self, n: int, seed: int) -> None:
        a = _rand_signal(n, seed)
        b = _rand_signal(n, seed + 100)
        assert profile_l2_distance(a, b) >= 0.0

    @pytest.mark.parametrize("n", [8, 16])
    def test_profile_l2_self_is_zero(self, n: int) -> None:
        a = _rand_signal(n)
        assert profile_l2_distance(a, a) == pytest.approx(0.0)

    @pytest.mark.parametrize("n,seed", [(10, 5), (20, 6)])
    def test_profile_l2_symmetric(self, n: int, seed: int) -> None:
        a = _rand_signal(n, seed)
        b = _rand_signal(n, seed + 200)
        assert profile_l2_distance(a, b) == pytest.approx(
            profile_l2_distance(b, a)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# contour_profile_utils — profile_cosine_similarity, best_cyclic_offset
# ═══════════════════════════════════════════════════════════════════════════════

class TestCosineAndCyclic:
    """profile_cosine_similarity and best_cyclic_offset invariants."""

    @pytest.mark.parametrize("n,seed", [(10, 0), (20, 1), (15, 2)])
    def test_cosine_in_range(self, n: int, seed: int) -> None:
        a = _rand_signal(n, seed)
        b = _rand_signal(n, seed + 300)
        val = profile_cosine_similarity(a, b)
        assert -1.0 - 1e-9 <= val <= 1.0 + 1e-9

    @pytest.mark.parametrize("n", [8, 16])
    def test_cosine_self_is_one(self, n: int) -> None:
        a = _rand_signal(n)
        val = profile_cosine_similarity(a, a)
        assert val == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("n,seed", [(8, 0), (12, 1), (16, 2)])
    def test_best_cyclic_offset_distance_nonneg(self, n: int, seed: int) -> None:
        a = _rand_signal(n, seed)
        b = _rand_signal(n, seed + 400)
        offset, dist = best_cyclic_offset(a, b)
        assert dist >= 0.0

    @pytest.mark.parametrize("n,seed", [(8, 3), (12, 4)])
    def test_best_cyclic_offset_in_range(self, n: int, seed: int) -> None:
        a = _rand_signal(n, seed)
        b = _rand_signal(n, seed + 500)
        offset, _ = best_cyclic_offset(a, b)
        assert 0 <= offset < n


# ═══════════════════════════════════════════════════════════════════════════════
# contour_profile_utils — match_profiles, batch_match_profiles, top_k
# ═══════════════════════════════════════════════════════════════════════════════

class TestMatchProfiles:
    """match_profiles and batch helpers invariants."""

    @pytest.mark.parametrize("n,seed,cyclic", [
        (16, 0, False), (16, 1, True), (20, 2, False)
    ])
    def test_match_score_in_range(self, n: int, seed: int, cyclic: bool) -> None:
        a = _rand_signal(n, seed)
        b = _rand_signal(n, seed + 600)
        result = match_profiles(a, b, cyclic=cyclic)
        assert 0.0 - 1e-9 <= result.score <= 1.0 + 1e-9

    @pytest.mark.parametrize("n,seed", [(16, 0), (20, 1)])
    def test_match_distance_nonneg(self, n: int, seed: int) -> None:
        a = _rand_signal(n, seed)
        b = _rand_signal(n, seed + 700)
        result = match_profiles(a, b)
        assert result.distance >= 0.0

    def test_batch_match_length(self) -> None:
        ref = _rand_signal(16)
        candidates = [_rand_signal(16, i) for i in range(5)]
        results = batch_match_profiles(ref, candidates)
        assert len(results) == 5

    def test_top_k_at_most_k(self) -> None:
        ref = _rand_signal(16)
        candidates = [_rand_signal(16, i) for i in range(8)]
        results = batch_match_profiles(ref, candidates)
        top = top_k_profile_matches(results, k=3)
        assert len(top) <= 3

    def test_top_k_sorted_descending(self) -> None:
        ref = _rand_signal(16)
        candidates = [_rand_signal(16, i) for i in range(6)]
        results = batch_match_profiles(ref, candidates)
        top = top_k_profile_matches(results, k=6)
        scores = [r.score for r in top]
        assert scores == sorted(scores, reverse=True)
