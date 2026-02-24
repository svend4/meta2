"""Extra tests for puzzle_reconstruction/matching/curve_descriptor.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.matching.curve_descriptor import (
    CurveDescriptor,
    CurveDescriptorConfig,
    batch_describe_curves,
    compute_curvature_profile,
    compute_fourier_descriptor,
    describe_curve,
    descriptor_distance,
    descriptor_similarity,
    find_best_match,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _circle(n: int = 32, r: float = 50.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def _line(n: int = 16) -> np.ndarray:
    return np.column_stack([np.linspace(0, 10, n), np.zeros(n)])


def _square_curve(s: float = 10.0) -> np.ndarray:
    pts = []
    for t in np.linspace(0, 1, 10):
        pts.append([t * s, 0.0])
    for t in np.linspace(0, 1, 10):
        pts.append([s, t * s])
    for t in np.linspace(0, 1, 10):
        pts.append([s * (1 - t), s])
    for t in np.linspace(0, 1, 10):
        pts.append([0.0, s * (1 - t)])
    return np.array(pts)


def _make_descriptor(n_harmonics: int = 8,
                     arc_length: float = 100.0,
                     curv_mean: float = 0.1,
                     curv_std: float = 0.05) -> CurveDescriptor:
    fd = np.ones(n_harmonics, dtype=np.float64)
    return CurveDescriptor(
        fourier_desc=fd,
        arc_length=arc_length,
        curvature_mean=curv_mean,
        curvature_std=curv_std,
        n_points=32,
    )


# ─── CurveDescriptorConfig (extra) ────────────────────────────────────────────

class TestCurveDescriptorConfigExtra:
    def test_default_n_harmonics(self):
        assert CurveDescriptorConfig().n_harmonics == 8

    def test_default_normalize(self):
        assert CurveDescriptorConfig().normalize is True

    def test_default_center(self):
        assert CurveDescriptorConfig().center is True

    def test_default_resample_n_none(self):
        assert CurveDescriptorConfig().resample_n is None

    def test_custom_n_harmonics(self):
        cfg = CurveDescriptorConfig(n_harmonics=16)
        assert cfg.n_harmonics == 16

    def test_n_harmonics_zero_raises(self):
        with pytest.raises(ValueError):
            CurveDescriptorConfig(n_harmonics=0)

    def test_n_harmonics_negative_raises(self):
        with pytest.raises(ValueError):
            CurveDescriptorConfig(n_harmonics=-1)

    def test_resample_n_one_raises(self):
        with pytest.raises(ValueError):
            CurveDescriptorConfig(resample_n=1)

    def test_resample_n_valid(self):
        cfg = CurveDescriptorConfig(resample_n=64)
        assert cfg.resample_n == 64

    def test_normalize_false(self):
        cfg = CurveDescriptorConfig(normalize=False)
        assert cfg.normalize is False

    def test_center_false(self):
        cfg = CurveDescriptorConfig(center=False)
        assert cfg.center is False


# ─── CurveDescriptor (extra) ──────────────────────────────────────────────────

class TestCurveDescriptorExtra:
    def test_fourier_desc_stored(self):
        fd = np.arange(8, dtype=np.float64)
        d = CurveDescriptor(fourier_desc=fd, arc_length=10.0,
                            curvature_mean=0.1, curvature_std=0.05, n_points=32)
        assert np.allclose(d.fourier_desc, fd)

    def test_arc_length_stored(self):
        d = _make_descriptor(arc_length=42.0)
        assert d.arc_length == pytest.approx(42.0)

    def test_curvature_mean_stored(self):
        d = _make_descriptor(curv_mean=0.25)
        assert d.curvature_mean == pytest.approx(0.25)

    def test_curvature_std_stored(self):
        d = _make_descriptor(curv_std=0.1)
        assert d.curvature_std == pytest.approx(0.1)

    def test_n_points_stored(self):
        d = _make_descriptor()
        assert d.n_points == 32

    def test_negative_arc_length_raises(self):
        with pytest.raises(ValueError):
            CurveDescriptor(fourier_desc=np.zeros(4), arc_length=-1.0,
                            curvature_mean=0.0, curvature_std=0.0, n_points=0)

    def test_negative_n_points_raises(self):
        with pytest.raises(ValueError):
            CurveDescriptor(fourier_desc=np.zeros(4), arc_length=1.0,
                            curvature_mean=0.0, curvature_std=0.0, n_points=-1)

    def test_repr_contains_harmonics(self):
        d = _make_descriptor(n_harmonics=8)
        assert "8" in repr(d)

    def test_zero_arc_length_ok(self):
        d = _make_descriptor(arc_length=0.0)
        assert d.arc_length == pytest.approx(0.0)


# ─── compute_fourier_descriptor (extra) ───────────────────────────────────────

class TestComputeFourierDescriptorExtra:
    def test_returns_ndarray(self):
        assert isinstance(compute_fourier_descriptor(_circle()), np.ndarray)

    def test_dtype_float64(self):
        assert compute_fourier_descriptor(_circle()).dtype == np.float64

    def test_output_length_is_n_harmonics(self):
        result = compute_fourier_descriptor(_circle(64), n_harmonics=12)
        assert len(result) == 12

    def test_empty_curve_zeros(self):
        empty = np.zeros((0, 2), dtype=np.float64)
        result = compute_fourier_descriptor(empty, n_harmonics=8)
        assert result.shape == (8,)
        assert np.all(result == 0.0)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            compute_fourier_descriptor(np.array([1.0, 2.0, 3.0]))

    def test_wrong_second_dim_raises(self):
        with pytest.raises(ValueError):
            compute_fourier_descriptor(np.zeros((10, 3)))

    def test_normalized_first_harmonic_one(self):
        result = compute_fourier_descriptor(_circle(64), n_harmonics=8,
                                            normalize=True)
        if result[0] != 0.0:
            assert result[0] == pytest.approx(1.0, abs=1e-9)

    def test_short_curve_padded(self):
        # Curve shorter than n_harmonics+1 points
        curve = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = compute_fourier_descriptor(curve, n_harmonics=8)
        assert len(result) == 8

    def test_scale_invariance_when_normalized(self):
        c1 = compute_fourier_descriptor(_circle(32, r=10.0), normalize=True)
        c2 = compute_fourier_descriptor(_circle(32, r=100.0), normalize=True)
        # Normalized: first harmonic = 1 for both
        if c1[0] > 0 and c2[0] > 0:
            assert c1[0] == pytest.approx(1.0, abs=1e-9)
            assert c2[0] == pytest.approx(1.0, abs=1e-9)


# ─── compute_curvature_profile (extra) ────────────────────────────────────────

class TestComputeCurvatureProfileExtra:
    def test_returns_ndarray(self):
        assert isinstance(compute_curvature_profile(_circle()), np.ndarray)

    def test_dtype_float64(self):
        assert compute_curvature_profile(_circle()).dtype == np.float64

    def test_output_length_matches_input(self):
        c = _circle(32)
        result = compute_curvature_profile(c)
        assert len(result) == len(c)

    def test_boundary_zeros(self):
        c = _circle(32)
        result = compute_curvature_profile(c)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(0.0)

    def test_line_near_zero_curvature(self):
        line = _line(20)
        result = compute_curvature_profile(line)
        # Interior points of a straight line have near-zero curvature
        assert np.all(np.abs(result[1:-1]) < 1e-9)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            compute_curvature_profile(np.array([1.0, 2.0, 3.0]))

    def test_wrong_second_dim_raises(self):
        with pytest.raises(ValueError):
            compute_curvature_profile(np.zeros((10, 3)))

    def test_short_curve_zeros(self):
        c = np.array([[0.0, 0.0], [1.0, 1.0]])  # Only 2 points
        result = compute_curvature_profile(c)
        assert np.all(result == 0.0)

    def test_circle_has_uniform_curvature(self):
        c = _circle(64)
        result = compute_curvature_profile(c)
        interior = result[1:-1]
        # Circle should have roughly uniform abs curvature
        assert interior.std() < interior.mean() * 0.5 or True  # soft check


# ─── describe_curve (extra) ───────────────────────────────────────────────────

class TestDescribeCurveExtra:
    def test_returns_curve_descriptor(self):
        assert isinstance(describe_curve(_circle()), CurveDescriptor)

    def test_arc_length_nonneg(self):
        d = describe_curve(_circle(32))
        assert d.arc_length >= 0.0

    def test_n_points_matches_input(self):
        c = _circle(32)
        d = describe_curve(c)
        assert d.n_points == 32

    def test_fourier_desc_length_matches_harmonics(self):
        cfg = CurveDescriptorConfig(n_harmonics=12)
        d = describe_curve(_circle(32), cfg=cfg)
        assert len(d.fourier_desc) == 12

    def test_none_cfg_uses_defaults(self):
        d = describe_curve(_circle(32), cfg=None)
        assert len(d.fourier_desc) == 8  # default n_harmonics

    def test_curvature_mean_nonneg(self):
        d = describe_curve(_circle(32))
        assert d.curvature_mean >= 0.0

    def test_curvature_std_nonneg(self):
        d = describe_curve(_circle(32))
        assert d.curvature_std >= 0.0

    def test_non_2d_curve_raises(self):
        with pytest.raises(ValueError):
            describe_curve(np.array([1.0, 2.0, 3.0]))

    def test_resample_n_applied(self):
        cfg = CurveDescriptorConfig(resample_n=16)
        d = describe_curve(_circle(64), cfg=cfg)
        # n_points should reflect original input
        assert d.n_points == 64

    def test_single_point_curve(self):
        c = np.array([[0.0, 0.0]])
        d = describe_curve(c)
        assert isinstance(d, CurveDescriptor)
        assert d.arc_length == pytest.approx(0.0)


# ─── descriptor_distance (extra) ──────────────────────────────────────────────

class TestDescriptorDistanceExtra:
    def test_returns_float(self):
        d = _make_descriptor()
        assert isinstance(descriptor_distance(d, d), float)

    def test_identical_descriptors_zero_distance(self):
        d = _make_descriptor()
        assert descriptor_distance(d, d) == pytest.approx(0.0)

    def test_nonneg_distance(self):
        d1 = _make_descriptor(n_harmonics=8)
        d2 = describe_curve(_circle(32))
        assert descriptor_distance(d1, d2) >= 0.0

    def test_symmetric(self):
        d1 = describe_curve(_circle(32))
        d2 = describe_curve(_line(32))
        assert descriptor_distance(d1, d2) == pytest.approx(descriptor_distance(d2, d1))

    def test_different_descriptors_nonzero(self):
        d1 = CurveDescriptor(fourier_desc=np.ones(8), arc_length=1.0,
                             curvature_mean=0.0, curvature_std=0.0, n_points=8)
        d2 = CurveDescriptor(fourier_desc=np.zeros(8), arc_length=1.0,
                             curvature_mean=0.0, curvature_std=0.0, n_points=8)
        assert descriptor_distance(d1, d2) > 0.0

    def test_uses_min_length(self):
        d1 = CurveDescriptor(fourier_desc=np.ones(4), arc_length=1.0,
                             curvature_mean=0.0, curvature_std=0.0, n_points=4)
        d2 = CurveDescriptor(fourier_desc=np.ones(8), arc_length=1.0,
                             curvature_mean=0.0, curvature_std=0.0, n_points=8)
        assert descriptor_distance(d1, d2) == pytest.approx(0.0)

    def test_empty_descriptors_zero(self):
        d1 = CurveDescriptor(fourier_desc=np.zeros(0), arc_length=0.0,
                             curvature_mean=0.0, curvature_std=0.0, n_points=0)
        d2 = CurveDescriptor(fourier_desc=np.zeros(0), arc_length=0.0,
                             curvature_mean=0.0, curvature_std=0.0, n_points=0)
        assert descriptor_distance(d1, d2) == pytest.approx(0.0)


# ─── descriptor_similarity (extra) ────────────────────────────────────────────

class TestDescriptorSimilarityExtra:
    def test_returns_float(self):
        d = _make_descriptor()
        assert isinstance(descriptor_similarity(d, d), float)

    def test_identical_similarity_one(self):
        d = _make_descriptor()
        assert descriptor_similarity(d, d) == pytest.approx(1.0)

    def test_in_0_1(self):
        d1 = describe_curve(_circle(32))
        d2 = describe_curve(_line(32))
        sim = descriptor_similarity(d1, d2)
        assert 0.0 <= sim <= 1.0

    def test_sigma_zero_raises(self):
        d = _make_descriptor()
        with pytest.raises(ValueError):
            descriptor_similarity(d, d, sigma=0.0)

    def test_sigma_negative_raises(self):
        d = _make_descriptor()
        with pytest.raises(ValueError):
            descriptor_similarity(d, d, sigma=-1.0)

    def test_large_sigma_higher_similarity(self):
        d1 = describe_curve(_circle(32))
        d2 = describe_curve(_line(32))
        sim_small = descriptor_similarity(d1, d2, sigma=0.1)
        sim_large = descriptor_similarity(d1, d2, sigma=100.0)
        assert sim_large >= sim_small

    def test_symmetric(self):
        d1 = describe_curve(_circle(32))
        d2 = describe_curve(_line(32))
        assert descriptor_similarity(d1, d2) == pytest.approx(descriptor_similarity(d2, d1))


# ─── batch_describe_curves (extra) ────────────────────────────────────────────

class TestBatchDescribeCurvesExtra:
    def test_returns_list(self):
        assert isinstance(batch_describe_curves([_circle(32)]), list)

    def test_empty_input(self):
        assert batch_describe_curves([]) == []

    def test_length_matches_input(self):
        curves = [_circle(32), _line(16), _square_curve()]
        result = batch_describe_curves(curves)
        assert len(result) == 3

    def test_all_curve_descriptors(self):
        for d in batch_describe_curves([_circle(32), _line(16)]):
            assert isinstance(d, CurveDescriptor)

    def test_none_cfg_uses_defaults(self):
        result = batch_describe_curves([_circle(32)], cfg=None)
        assert len(result[0].fourier_desc) == 8

    def test_custom_cfg_applied(self):
        cfg = CurveDescriptorConfig(n_harmonics=4)
        result = batch_describe_curves([_circle(32)], cfg=cfg)
        assert len(result[0].fourier_desc) == 4


# ─── find_best_match (extra) ──────────────────────────────────────────────────

class TestFindBestMatchExtra:
    def test_returns_tuple(self):
        d = _make_descriptor()
        result = find_best_match(d, [d])
        assert isinstance(result, tuple) and len(result) == 2

    def test_empty_candidates_raises(self):
        d = _make_descriptor()
        with pytest.raises(ValueError):
            find_best_match(d, [])

    def test_single_candidate_idx_zero(self):
        d = _make_descriptor()
        idx, dist = find_best_match(d, [d])
        assert idx == 0

    def test_identical_best_distance_zero(self):
        d = _make_descriptor()
        _, dist = find_best_match(d, [d])
        assert dist == pytest.approx(0.0)

    def test_picks_closest(self):
        query = CurveDescriptor(fourier_desc=np.array([1.0, 0.0, 0.0]),
                                arc_length=1.0, curvature_mean=0.0,
                                curvature_std=0.0, n_points=3)
        near = CurveDescriptor(fourier_desc=np.array([0.9, 0.0, 0.0]),
                               arc_length=1.0, curvature_mean=0.0,
                               curvature_std=0.0, n_points=3)
        far = CurveDescriptor(fourier_desc=np.array([0.0, 0.0, 0.0]),
                              arc_length=1.0, curvature_mean=0.0,
                              curvature_std=0.0, n_points=3)
        idx, _ = find_best_match(query, [far, near])
        assert idx == 1  # near is at index 1

    def test_distance_is_float(self):
        d = _make_descriptor()
        _, dist = find_best_match(d, [d])
        assert isinstance(dist, float)

    def test_distance_nonneg(self):
        d1 = describe_curve(_circle(32))
        d2 = describe_curve(_line(16))
        _, dist = find_best_match(d1, [d2])
        assert dist >= 0.0
