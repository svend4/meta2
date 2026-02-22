"""Тесты для puzzle_reconstruction/utils/edge_profiler.py."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.edge_profiler import (
    ProfileConfig,
    EdgeProfile,
    compute_brightness_profile,
    compute_gradient_profile,
    compute_diff_profile,
    normalize_profile,
    aggregate_profiles,
    compare_profiles,
    batch_profile_edges,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_strip(h=8, w=32, fill=128.0):
    """Равномерная полоса края."""
    return np.full((h, w), fill, dtype=np.float64)


def make_ramp_strip(h=8, w=32):
    """Полоса с линейным градиентом по оси 1."""
    col = np.linspace(0, 255, w)
    return np.tile(col, (h, 1))


def make_sine_strip(h=8, w=32):
    """Полоса с синусоидальным профилем."""
    col = np.sin(np.linspace(0, 4 * np.pi, w)) * 100 + 128
    return np.tile(col, (h, 1))


# ─── ProfileConfig ────────────────────────────────────────────────────────────

class TestProfileConfig:
    def test_defaults(self):
        cfg = ProfileConfig()
        assert cfg.n_samples == 32
        assert cfg.profile_type == "brightness"
        assert cfg.normalize is True
        assert cfg.strip_width == 4

    def test_n_samples_1_raises(self):
        with pytest.raises(ValueError, match="n_samples"):
            ProfileConfig(n_samples=1)

    def test_n_samples_0_raises(self):
        with pytest.raises(ValueError, match="n_samples"):
            ProfileConfig(n_samples=0)

    def test_n_samples_2_valid(self):
        cfg = ProfileConfig(n_samples=2)
        assert cfg.n_samples == 2

    def test_invalid_profile_type_raises(self):
        with pytest.raises(ValueError, match="profile_type"):
            ProfileConfig(profile_type="sobel")

    def test_valid_profile_types(self):
        for t in ("brightness", "gradient", "diff"):
            cfg = ProfileConfig(profile_type=t)
            assert cfg.profile_type == t

    def test_strip_width_zero_raises(self):
        with pytest.raises(ValueError, match="strip_width"):
            ProfileConfig(strip_width=0)

    def test_strip_width_negative_raises(self):
        with pytest.raises(ValueError, match="strip_width"):
            ProfileConfig(strip_width=-1)

    def test_strip_width_1_valid(self):
        cfg = ProfileConfig(strip_width=1)
        assert cfg.strip_width == 1


# ─── EdgeProfile ──────────────────────────────────────────────────────────────

class TestEdgeProfile:
    def test_creation(self):
        ep = EdgeProfile(
            profile=np.zeros(16),
            edge_id=3,
            profile_type="brightness",
            n_samples=16,
        )
        assert ep.edge_id == 3
        assert ep.n_samples == 16

    def test_negative_edge_id_raises(self):
        with pytest.raises(ValueError, match="edge_id"):
            EdgeProfile(
                profile=np.zeros(16),
                edge_id=-1,
                profile_type="brightness",
                n_samples=16,
            )

    def test_mismatched_profile_length_raises(self):
        with pytest.raises(ValueError, match="profile length"):
            EdgeProfile(
                profile=np.zeros(10),  # length 10
                edge_id=0,
                profile_type="brightness",
                n_samples=16,          # mismatch
            )

    def test_default_params_empty(self):
        ep = EdgeProfile(
            profile=np.zeros(8),
            edge_id=0,
            profile_type="diff",
            n_samples=8,
        )
        assert ep.params == {}

    def test_zero_edge_id_valid(self):
        ep = EdgeProfile(
            profile=np.zeros(8),
            edge_id=0,
            profile_type="gradient",
            n_samples=8,
        )
        assert ep.edge_id == 0


# ─── compute_brightness_profile ───────────────────────────────────────────────

class TestComputeBrightnessProfile:
    def test_returns_float64(self):
        strip = make_strip()
        result = compute_brightness_profile(strip, n_samples=16)
        assert result.dtype == np.float64

    def test_length_equals_n_samples(self):
        strip = make_strip()
        result = compute_brightness_profile(strip, n_samples=20)
        assert len(result) == 20

    def test_uniform_strip_constant_profile(self):
        strip = make_strip(fill=100.0)
        result = compute_brightness_profile(strip, n_samples=16)
        np.testing.assert_allclose(result, 100.0, atol=1e-9)

    def test_ramp_profile_nondecreasing(self):
        strip = make_ramp_strip()
        result = compute_brightness_profile(strip, n_samples=16)
        assert (np.diff(result) >= -1e-9).all()

    def test_1d_strip_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            compute_brightness_profile(np.zeros(16), n_samples=8)

    def test_n_samples_1_raises(self):
        with pytest.raises(ValueError, match="n_samples"):
            compute_brightness_profile(make_strip(), n_samples=1)

    def test_empty_strip_returns_zeros(self):
        strip = np.zeros((0, 0), dtype=np.float64)
        result = compute_brightness_profile(strip, n_samples=8)
        np.testing.assert_array_equal(result, 0.0)

    def test_axis_0(self):
        strip = make_ramp_strip(h=32, w=8)
        result = compute_brightness_profile(strip, n_samples=8, axis=0)
        assert len(result) == 8


# ─── compute_gradient_profile ─────────────────────────────────────────────────

class TestComputeGradientProfile:
    def test_returns_float64(self):
        strip = make_strip()
        result = compute_gradient_profile(strip, n_samples=16)
        assert result.dtype == np.float64

    def test_length_equals_n_samples(self):
        strip = make_strip()
        result = compute_gradient_profile(strip, n_samples=12)
        assert len(result) == 12

    def test_nonneg_values(self):
        strip = make_ramp_strip()
        result = compute_gradient_profile(strip, n_samples=16)
        assert (result >= -1e-9).all()

    def test_uniform_strip_near_zero_gradient(self):
        strip = make_strip(fill=128.0)
        result = compute_gradient_profile(strip, n_samples=16)
        np.testing.assert_allclose(result, 0.0, atol=1e-9)

    def test_1d_strip_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            compute_gradient_profile(np.zeros(16), n_samples=8)

    def test_n_samples_1_raises(self):
        with pytest.raises(ValueError, match="n_samples"):
            compute_gradient_profile(make_strip(), n_samples=1)

    def test_empty_strip_returns_zeros(self):
        strip = np.zeros((0, 0), dtype=np.float64)
        result = compute_gradient_profile(strip, n_samples=8)
        np.testing.assert_array_equal(result, 0.0)

    def test_ramp_has_nonzero_gradient(self):
        strip = make_ramp_strip()
        result = compute_gradient_profile(strip, n_samples=16)
        # A linear ramp has constant non-zero gradient
        assert result.sum() > 0.0


# ─── compute_diff_profile ─────────────────────────────────────────────────────

class TestComputeDiffProfile:
    def test_returns_float64(self):
        strip = make_strip()
        result = compute_diff_profile(strip, n_samples=16)
        assert result.dtype == np.float64

    def test_length_equals_n_samples(self):
        strip = make_strip()
        result = compute_diff_profile(strip, n_samples=10)
        assert len(result) == 10

    def test_nonneg_values(self):
        strip = make_ramp_strip()
        result = compute_diff_profile(strip, n_samples=16)
        assert (result >= -1e-9).all()

    def test_uniform_strip_near_zero_diff(self):
        strip = make_strip(fill=128.0)
        result = compute_diff_profile(strip, n_samples=16)
        np.testing.assert_allclose(result, 0.0, atol=1e-9)

    def test_1d_strip_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            compute_diff_profile(np.zeros(16), n_samples=8)

    def test_n_samples_1_raises(self):
        with pytest.raises(ValueError, match="n_samples"):
            compute_diff_profile(make_strip(), n_samples=1)

    def test_single_column_returns_zeros(self):
        strip = np.ones((8, 1), dtype=np.float64)
        result = compute_diff_profile(strip, n_samples=8)
        np.testing.assert_array_equal(result, 0.0)

    def test_ramp_has_nonzero_diff(self):
        strip = make_ramp_strip()
        result = compute_diff_profile(strip, n_samples=16)
        assert result.sum() > 0.0


# ─── normalize_profile ────────────────────────────────────────────────────────

class TestNormalizeProfile:
    def test_returns_float64(self):
        p = np.array([1.0, 2.0, 3.0])
        result = normalize_profile(p)
        assert result.dtype == np.float64

    def test_in_0_1(self):
        p = np.array([0.0, 50.0, 100.0])
        result = normalize_profile(p)
        assert result.min() >= 0.0 - 1e-9
        assert result.max() <= 1.0 + 1e-9

    def test_min_is_0(self):
        p = np.array([10.0, 50.0, 100.0])
        result = normalize_profile(p)
        assert result.min() == pytest.approx(0.0)

    def test_max_is_1(self):
        p = np.array([10.0, 50.0, 100.0])
        result = normalize_profile(p)
        assert result.max() == pytest.approx(1.0)

    def test_uniform_returns_zeros(self):
        p = np.full(8, 42.0)
        result = normalize_profile(p)
        np.testing.assert_array_equal(result, 0.0)

    def test_2d_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            normalize_profile(np.zeros((4, 4)))

    def test_empty_returns_empty(self):
        p = np.array([], dtype=np.float64)
        result = normalize_profile(p)
        assert len(result) == 0

    def test_same_length(self):
        p = np.linspace(0, 10, 20)
        result = normalize_profile(p)
        assert len(result) == 20


# ─── aggregate_profiles ───────────────────────────────────────────────────────

class TestAggregateProfiles:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            aggregate_profiles([])

    def test_single_profile(self):
        p = np.linspace(0, 1, 16)
        result = aggregate_profiles([p])
        np.testing.assert_allclose(result, p, atol=1e-9)

    def test_equal_weights_is_mean(self):
        p1 = np.zeros(8)
        p2 = np.ones(8)
        result = aggregate_profiles([p1, p2])
        np.testing.assert_allclose(result, 0.5, atol=1e-9)

    def test_custom_weights(self):
        p1 = np.zeros(4)
        p2 = np.full(4, 2.0)
        result = aggregate_profiles([p1, p2], weights=[1.0, 3.0])
        np.testing.assert_allclose(result, 1.5, atol=1e-9)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="length"):
            aggregate_profiles([np.zeros(8), np.zeros(16)])

    def test_zero_weight_sum_raises(self):
        with pytest.raises(ValueError, match="Sum of weights"):
            aggregate_profiles([np.zeros(4)], weights=[0.0])

    def test_returns_float64(self):
        result = aggregate_profiles([np.ones(8)])
        assert result.dtype == np.float64

    def test_length_preserved(self):
        result = aggregate_profiles([np.zeros(12)])
        assert len(result) == 12

    def test_three_profiles_equal_weights(self):
        p1 = np.zeros(4)
        p2 = np.full(4, 3.0)
        p3 = np.full(4, 6.0)
        result = aggregate_profiles([p1, p2, p3])
        np.testing.assert_allclose(result, 3.0, atol=1e-9)


# ─── compare_profiles ─────────────────────────────────────────────────────────

class TestCompareProfiles:
    def test_identical_returns_1(self):
        p = np.linspace(0, 1, 16)
        result = compare_profiles(p, p)
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_in_0_1(self):
        p1 = np.random.default_rng(0).random(16)
        p2 = np.random.default_rng(1).random(16)
        result = compare_profiles(p1, p2)
        assert 0.0 <= result <= 1.0

    def test_different_lengths_raises(self):
        p1 = np.zeros(8)
        p2 = np.zeros(16)
        with pytest.raises(ValueError, match="length"):
            compare_profiles(p1, p2)

    def test_zero_profiles_return_value(self):
        p1 = np.zeros(8)
        p2 = np.zeros(8)
        result = compare_profiles(p1, p2)
        assert result == pytest.approx(1.0)

    def test_opposite_profiles(self):
        p1 = np.array([0.0] * 8)
        p2 = np.array([10.0] * 8)
        result = compare_profiles(p1, p2)
        assert 0.0 <= result <= 1.0

    def test_returns_float(self):
        p = np.linspace(0, 1, 8)
        result = compare_profiles(p, p)
        assert isinstance(result, float)

    def test_similar_profiles_high_similarity(self):
        p = np.sin(np.linspace(0, np.pi, 32))
        p_noisy = p + np.random.default_rng(0).normal(0, 0.01, 32)
        result = compare_profiles(p, p_noisy)
        assert result > 0.9


# ─── batch_profile_edges ──────────────────────────────────────────────────────

class TestBatchProfileEdges:
    def test_empty_strips_returns_empty(self):
        result = batch_profile_edges([])
        assert result == []

    def test_length_matches(self):
        strips = [make_strip(h=4, w=16) for _ in range(5)]
        result = batch_profile_edges(strips)
        assert len(result) == 5

    def test_all_edge_profiles(self):
        strips = [make_strip(h=4, w=16)]
        result = batch_profile_edges(strips)
        assert isinstance(result[0], EdgeProfile)

    def test_default_edge_ids(self):
        strips = [make_strip(h=4, w=16), make_strip(h=4, w=16)]
        result = batch_profile_edges(strips)
        assert result[0].edge_id == 0
        assert result[1].edge_id == 1

    def test_custom_edge_ids(self):
        strips = [make_strip(h=4, w=16), make_strip(h=4, w=16)]
        result = batch_profile_edges(strips, edge_ids=[10, 20])
        assert result[0].edge_id == 10
        assert result[1].edge_id == 20

    def test_profile_type_stored(self):
        cfg = ProfileConfig(profile_type="gradient", n_samples=16)
        strips = [make_strip(h=4, w=16)]
        result = batch_profile_edges(strips, cfg=cfg)
        assert result[0].profile_type == "gradient"

    def test_n_samples_correct(self):
        cfg = ProfileConfig(n_samples=24)
        strips = [make_strip(h=4, w=24)]
        result = batch_profile_edges(strips, cfg=cfg)
        assert result[0].n_samples == 24
        assert len(result[0].profile) == 24

    def test_normalize_applies(self):
        cfg = ProfileConfig(normalize=True, n_samples=16)
        strip = make_ramp_strip(h=4, w=16)
        result = batch_profile_edges([strip], cfg=cfg)
        p = result[0].profile
        assert p.max() <= 1.0 + 1e-9
        assert p.min() >= 0.0 - 1e-9

    def test_diff_type(self):
        cfg = ProfileConfig(profile_type="diff", n_samples=16)
        strips = [make_ramp_strip(h=4, w=16)]
        result = batch_profile_edges(strips, cfg=cfg)
        assert result[0].profile_type == "diff"

    def test_brightness_type(self):
        cfg = ProfileConfig(profile_type="brightness", n_samples=16)
        strips = [make_strip(h=4, w=16, fill=100.0)]
        result = batch_profile_edges(strips, cfg=cfg)
        assert result[0].profile_type == "brightness"

    def test_strip_width_in_params(self):
        cfg = ProfileConfig(strip_width=6, n_samples=16)
        strips = [make_strip(h=4, w=16)]
        result = batch_profile_edges(strips, cfg=cfg)
        assert result[0].params.get("strip_width") == 6
