"""Tests for matching/curve_descriptor.py."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_line(n=20):
    """Straight horizontal line: y=0."""
    x = np.linspace(0.0, 10.0, n)
    y = np.zeros(n)
    return np.column_stack([x, y])


def make_circle(n=32, r=5.0):
    """Circle of radius r."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.column_stack([x, y])


def make_l_shape():
    """L-shaped curve with a sharp corner."""
    pts = [(0, 0), (0, 5), (0, 10), (5, 10), (10, 10)]
    return np.array(pts, dtype=np.float64)


def make_empty_curve():
    return np.zeros((0, 2), dtype=np.float64)


# ─── CurveDescriptorConfig ────────────────────────────────────────────────────

class TestCurveDescriptorConfig:
    def test_defaults(self):
        cfg = CurveDescriptorConfig()
        assert cfg.n_harmonics == 8
        assert cfg.normalize is True
        assert cfg.center is True
        assert cfg.resample_n is None

    def test_n_harmonics_zero_raises(self):
        with pytest.raises(ValueError, match="n_harmonics"):
            CurveDescriptorConfig(n_harmonics=0)

    def test_n_harmonics_negative_raises(self):
        with pytest.raises(ValueError, match="n_harmonics"):
            CurveDescriptorConfig(n_harmonics=-1)

    def test_n_harmonics_one_valid(self):
        cfg = CurveDescriptorConfig(n_harmonics=1)
        assert cfg.n_harmonics == 1

    def test_resample_n_one_raises(self):
        with pytest.raises(ValueError, match="resample_n"):
            CurveDescriptorConfig(resample_n=1)

    def test_resample_n_two_valid(self):
        cfg = CurveDescriptorConfig(resample_n=2)
        assert cfg.resample_n == 2

    def test_resample_n_none_valid(self):
        cfg = CurveDescriptorConfig(resample_n=None)
        assert cfg.resample_n is None


# ─── CurveDescriptor ──────────────────────────────────────────────────────────

class TestCurveDescriptor:
    def _make(self, n_harmonics=4, arc=5.0, n_points=10):
        fd = np.ones(n_harmonics, dtype=np.float64)
        return CurveDescriptor(
            fourier_desc=fd,
            arc_length=arc,
            curvature_mean=0.1,
            curvature_std=0.05,
            n_points=n_points,
        )

    def test_basic_creation(self):
        d = self._make()
        assert d.arc_length == pytest.approx(5.0)
        assert d.n_points == 10

    def test_arc_length_negative_raises(self):
        with pytest.raises(ValueError, match="arc_length"):
            CurveDescriptor(
                fourier_desc=np.zeros(4),
                arc_length=-1.0,
                curvature_mean=0.0,
                curvature_std=0.0,
                n_points=5,
            )

    def test_n_points_negative_raises(self):
        with pytest.raises(ValueError, match="n_points"):
            CurveDescriptor(
                fourier_desc=np.zeros(4),
                arc_length=1.0,
                curvature_mean=0.0,
                curvature_std=0.0,
                n_points=-1,
            )

    def test_arc_length_zero_valid(self):
        d = CurveDescriptor(
            fourier_desc=np.zeros(4),
            arc_length=0.0,
            curvature_mean=0.0,
            curvature_std=0.0,
            n_points=1,
        )
        assert d.arc_length == pytest.approx(0.0)

    def test_repr_contains_n_harmonics(self):
        d = self._make(n_harmonics=6)
        assert "6" in repr(d)

    def test_repr_contains_arc_length(self):
        d = self._make(arc=3.14)
        assert "3.14" in repr(d)

    def test_repr_contains_n_points(self):
        d = self._make(n_points=20)
        assert "20" in repr(d)


# ─── compute_fourier_descriptor ───────────────────────────────────────────────

class TestComputeFourierDescriptor:
    def test_wrong_shape_1d_raises(self):
        with pytest.raises(ValueError, match="shape"):
            compute_fourier_descriptor(np.zeros(10))

    def test_wrong_shape_3_columns_raises(self):
        with pytest.raises(ValueError, match="shape"):
            compute_fourier_descriptor(np.zeros((10, 3)))

    def test_empty_curve_returns_zeros(self):
        result = compute_fourier_descriptor(make_empty_curve(), n_harmonics=8)
        np.testing.assert_array_equal(result, np.zeros(8))

    def test_returns_n_harmonics_values(self):
        result = compute_fourier_descriptor(make_circle(), n_harmonics=6)
        assert result.shape == (6,)

    def test_returns_float64(self):
        result = compute_fourier_descriptor(make_line(), n_harmonics=4)
        assert result.dtype == np.float64

    def test_normalize_true_first_component_is_one(self):
        curve = make_circle(n=64)
        result = compute_fourier_descriptor(curve, n_harmonics=8, normalize=True)
        # With normalize=True and enough signal, desc[0] = 1.0
        if result[0] > 1e-9:
            assert result[0] == pytest.approx(1.0, abs=1e-6)

    def test_normalize_false_arbitrary_magnitudes(self):
        # Upscaled circle should give larger amplitudes
        small = compute_fourier_descriptor(make_circle(r=1.0), n_harmonics=4, normalize=False)
        large = compute_fourier_descriptor(make_circle(r=100.0), n_harmonics=4, normalize=False)
        assert large[0] > small[0]

    def test_identical_curves_same_descriptor(self):
        curve = make_circle()
        d1 = compute_fourier_descriptor(curve, n_harmonics=8)
        d2 = compute_fourier_descriptor(curve, n_harmonics=8)
        np.testing.assert_array_equal(d1, d2)

    def test_non_negative_amplitudes(self):
        result = compute_fourier_descriptor(make_circle(), n_harmonics=8)
        assert np.all(result >= 0.0)

    def test_short_curve_padded_to_n_harmonics(self):
        # 3-point curve (only 2 harmonics available before DC)
        curve = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        result = compute_fourier_descriptor(curve, n_harmonics=8)
        assert result.shape == (8,)


# ─── compute_curvature_profile ────────────────────────────────────────────────

class TestComputeCurvatureProfile:
    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            compute_curvature_profile(np.zeros(10))

    def test_returns_n_values(self):
        curve = make_line(n=15)
        result = compute_curvature_profile(curve)
        assert result.shape == (15,)

    def test_returns_float64(self):
        result = compute_curvature_profile(make_line())
        assert result.dtype == np.float64

    def test_less_than_3_points_returns_zeros(self):
        curve = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = compute_curvature_profile(curve)
        np.testing.assert_array_equal(result, np.zeros(2))

    def test_one_point_returns_zeros(self):
        curve = np.array([[0.0, 0.0]])
        result = compute_curvature_profile(curve)
        np.testing.assert_array_equal(result, np.zeros(1))

    def test_straight_line_near_zero_curvature(self):
        result = compute_curvature_profile(make_line(n=20))
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_l_shape_has_nonzero_curvature_at_corner(self):
        curve = make_l_shape()
        result = compute_curvature_profile(curve)
        # corner is at index 2 (point (0,10))
        assert abs(result[2]) > 1e-6

    def test_endpoints_are_zero(self):
        result = compute_curvature_profile(make_circle(n=20))
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(0.0)


# ─── describe_curve ───────────────────────────────────────────────────────────

class TestDescribeCurve:
    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            describe_curve(np.zeros(10))

    def test_returns_curve_descriptor(self):
        result = describe_curve(make_circle())
        assert isinstance(result, CurveDescriptor)

    def test_n_points_matches_input(self):
        curve = make_line(n=25)
        result = describe_curve(curve)
        assert result.n_points == 25

    def test_arc_length_positive_for_non_degenerate(self):
        result = describe_curve(make_circle(n=64, r=10.0))
        assert result.arc_length > 0.0

    def test_fourier_desc_length_matches_n_harmonics(self):
        cfg = CurveDescriptorConfig(n_harmonics=5)
        result = describe_curve(make_circle(), cfg=cfg)
        assert len(result.fourier_desc) == 5

    def test_default_config_n_harmonics_8(self):
        result = describe_curve(make_circle())
        assert len(result.fourier_desc) == 8

    def test_resample_n_changes_processing(self):
        cfg = CurveDescriptorConfig(resample_n=10)
        # Should not raise even with more points than resample_n
        result = describe_curve(make_circle(n=64), cfg=cfg)
        assert isinstance(result, CurveDescriptor)

    def test_curvature_mean_non_negative(self):
        result = describe_curve(make_circle())
        assert result.curvature_mean >= 0.0

    def test_curvature_std_non_negative(self):
        result = describe_curve(make_circle())
        assert result.curvature_std >= 0.0


# ─── descriptor_distance ──────────────────────────────────────────────────────

class TestDescriptorDistance:
    def _desc(self, values):
        return CurveDescriptor(
            fourier_desc=np.array(values, dtype=np.float64),
            arc_length=1.0,
            curvature_mean=0.0,
            curvature_std=0.0,
            n_points=5,
        )

    def test_identical_returns_zero(self):
        d = self._desc([1.0, 2.0, 3.0])
        assert descriptor_distance(d, d) == pytest.approx(0.0)

    def test_non_negative(self):
        d1 = self._desc([0.0, 1.0, 2.0])
        d2 = self._desc([3.0, 4.0, 5.0])
        assert descriptor_distance(d1, d2) >= 0.0

    def test_symmetric(self):
        d1 = self._desc([1.0, 0.0])
        d2 = self._desc([0.0, 1.0])
        assert descriptor_distance(d1, d2) == pytest.approx(descriptor_distance(d2, d1))

    def test_returns_float(self):
        d = self._desc([1.0, 2.0])
        assert isinstance(descriptor_distance(d, d), float)

    def test_different_descriptors_positive(self):
        d1 = self._desc([0.0, 0.0, 0.0])
        d2 = self._desc([1.0, 1.0, 1.0])
        assert descriptor_distance(d1, d2) > 0.0

    def test_known_value(self):
        d1 = self._desc([0.0, 0.0])
        d2 = self._desc([3.0, 4.0])
        assert descriptor_distance(d1, d2) == pytest.approx(5.0)

    def test_truncated_to_min_length(self):
        d1 = self._desc([1.0, 2.0, 3.0])
        d2 = self._desc([1.0, 2.0])
        # Only first 2 components compared
        assert descriptor_distance(d1, d2) == pytest.approx(0.0)


# ─── descriptor_similarity ────────────────────────────────────────────────────

class TestDescriptorSimilarity:
    def _desc(self, values):
        return CurveDescriptor(
            fourier_desc=np.array(values, dtype=np.float64),
            arc_length=1.0,
            curvature_mean=0.0,
            curvature_std=0.0,
            n_points=5,
        )

    def test_sigma_zero_raises(self):
        d = self._desc([1.0, 2.0])
        with pytest.raises(ValueError, match="sigma"):
            descriptor_similarity(d, d, sigma=0.0)

    def test_sigma_negative_raises(self):
        d = self._desc([1.0, 2.0])
        with pytest.raises(ValueError, match="sigma"):
            descriptor_similarity(d, d, sigma=-1.0)

    def test_identical_returns_one(self):
        d = self._desc([1.0, 2.0, 3.0])
        sim = descriptor_similarity(d, d)
        assert sim == pytest.approx(1.0)

    def test_range_zero_to_one(self):
        d1 = self._desc([0.0, 0.0])
        d2 = self._desc([100.0, 100.0])
        sim = descriptor_similarity(d1, d2, sigma=1.0)
        assert 0.0 <= sim <= 1.0

    def test_larger_sigma_higher_similarity(self):
        d1 = self._desc([0.0])
        d2 = self._desc([2.0])
        sim_small = descriptor_similarity(d1, d2, sigma=0.5)
        sim_large = descriptor_similarity(d1, d2, sigma=5.0)
        assert sim_large > sim_small


# ─── batch_describe_curves ────────────────────────────────────────────────────

class TestBatchDescribeCurves:
    def test_empty_returns_empty(self):
        assert batch_describe_curves([]) == []

    def test_length_preserved(self):
        curves = [make_line(), make_circle(), make_line(n=10)]
        result = batch_describe_curves(curves)
        assert len(result) == 3

    def test_all_curve_descriptors(self):
        result = batch_describe_curves([make_line(), make_circle()])
        for d in result:
            assert isinstance(d, CurveDescriptor)

    def test_custom_config_applied(self):
        cfg = CurveDescriptorConfig(n_harmonics=3)
        result = batch_describe_curves([make_circle()], cfg=cfg)
        assert len(result[0].fourier_desc) == 3


# ─── find_best_match ──────────────────────────────────────────────────────────

class TestFindBestMatch:
    def _desc(self, values):
        return CurveDescriptor(
            fourier_desc=np.array(values, dtype=np.float64),
            arc_length=1.0,
            curvature_mean=0.0,
            curvature_std=0.0,
            n_points=5,
        )

    def test_empty_candidates_raises(self):
        query = self._desc([1.0, 2.0])
        with pytest.raises(ValueError, match="empty"):
            find_best_match(query, [])

    def test_returns_tuple_of_two(self):
        query = self._desc([1.0, 2.0])
        result = find_best_match(query, [query])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_index_in_range(self):
        query = self._desc([1.0])
        cands = [self._desc([10.0]), self._desc([1.0]), self._desc([5.0])]
        idx, _ = find_best_match(query, cands)
        assert 0 <= idx < len(cands)

    def test_identical_candidate_has_zero_distance(self):
        query = self._desc([1.0, 2.0, 3.0])
        _, dist = find_best_match(query, [query])
        assert dist == pytest.approx(0.0)

    def test_closest_is_selected(self):
        query = self._desc([0.0])
        near = self._desc([1.0])
        far = self._desc([100.0])
        idx, _ = find_best_match(query, [far, near])
        assert idx == 1  # near is at index 1

    def test_distance_is_non_negative(self):
        query = self._desc([1.0])
        cands = [self._desc([2.0]), self._desc([3.0])]
        _, dist = find_best_match(query, cands)
        assert dist >= 0.0
