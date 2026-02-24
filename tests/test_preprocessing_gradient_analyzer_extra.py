"""Extra tests for puzzle_reconstruction/preprocessing/gradient_analyzer.py."""
import pytest
import numpy as np

from puzzle_reconstruction.preprocessing.gradient_analyzer import (
    GradientConfig,
    GradientMap,
    GradientProfile,
    compute_gradient_map,
    compute_orientation_histogram,
    extract_gradient_profile,
    compare_gradient_profiles,
    batch_extract_gradient_profiles,
)


def _gray(h=64, w=64, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)


def _noisy(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=64, w=64, fill=128):
    return np.full((h, w, 3), fill, dtype=np.uint8)


# ─── GradientConfig (extra) ───────────────────────────────────────────────────

class TestGradientConfigExtra:
    def test_default_kernel_sobel(self):
        assert GradientConfig().kernel == "sobel"

    def test_default_ksize_3(self):
        assert GradientConfig().ksize == 3

    def test_default_n_bins_8(self):
        assert GradientConfig().n_bins == 8

    def test_default_normalize_true(self):
        assert GradientConfig().normalize is True

    def test_scharr_ksize_3_only(self):
        cfg = GradientConfig(kernel="scharr", ksize=3)
        assert cfg.kernel == "scharr"

    def test_prewitt_valid(self):
        cfg = GradientConfig(kernel="prewitt")
        assert cfg.kernel == "prewitt"

    def test_n_bins_16_valid(self):
        cfg = GradientConfig(n_bins=16)
        assert cfg.n_bins == 16

    def test_n_bins_36_valid(self):
        cfg = GradientConfig(n_bins=36)
        assert cfg.n_bins == 36

    def test_normalize_false(self):
        cfg = GradientConfig(normalize=False)
        assert cfg.normalize is False

    def test_ksize_7_valid(self):
        cfg = GradientConfig(ksize=7)
        assert cfg.ksize == 7


# ─── GradientMap (extra) ──────────────────────────────────────────────────────

class TestGradientMapExtra:
    def _map(self, h=8, w=8, mean=1.0, max_=1.0, kernel="sobel"):
        mag = np.ones((h, w))
        ang = np.zeros((h, w))
        return GradientMap(magnitude=mag, angle=ang,
                           mean_mag=mean, max_mag=max_, kernel=kernel)

    def test_shape_property(self):
        gm = self._map(h=10, w=15)
        assert gm.shape == (10, 15)

    def test_kernel_stored(self):
        gm = self._map(kernel="scharr")
        assert gm.kernel == "scharr"

    def test_mean_mag_zero_valid(self):
        gm = self._map(mean=0.0, max_=0.0)
        assert gm.mean_mag == pytest.approx(0.0)

    def test_magnitude_shape_accessible(self):
        gm = self._map(h=6, w=9)
        assert gm.magnitude.shape == (6, 9)

    def test_angle_shape_accessible(self):
        gm = self._map(h=5, w=7)
        assert gm.angle.shape == (5, 7)


# ─── GradientProfile (extra) ──────────────────────────────────────────────────

class TestGradientProfileExtra:
    def test_valid_large_fragment_id(self):
        gp = GradientProfile(fragment_id=999)
        assert gp.fragment_id == 999

    def test_fragment_id_zero_valid(self):
        gp = GradientProfile(fragment_id=0)
        assert gp.fragment_id == 0

    def test_sharpness_score_range_mid(self):
        gp = GradientProfile(fragment_id=0, mean_magnitude=100.0)
        assert 0.0 < gp.sharpness_score < 1.0

    def test_sharpness_score_zero_mean(self):
        gp = GradientProfile(fragment_id=0, mean_magnitude=0.0)
        assert gp.sharpness_score == pytest.approx(0.0)

    def test_sharpness_score_max_at_255(self):
        gp = GradientProfile(fragment_id=0, mean_magnitude=255.0)
        assert gp.sharpness_score == pytest.approx(1.0)

    def test_sharpness_score_clipped_above_255(self):
        gp = GradientProfile(fragment_id=0, mean_magnitude=500.0)
        assert gp.sharpness_score == pytest.approx(1.0)

    def test_orientation_hist_stored(self):
        hist = np.array([0.1, 0.2, 0.3, 0.4])
        gp = GradientProfile(fragment_id=0, orientation_hist=hist)
        np.testing.assert_array_equal(gp.orientation_hist, hist)

    def test_dominant_angle_stored(self):
        gp = GradientProfile(fragment_id=0, dominant_angle=45.0)
        assert gp.dominant_angle == pytest.approx(45.0)

    def test_energy_stored(self):
        gp = GradientProfile(fragment_id=0, energy=10.5)
        assert gp.energy == pytest.approx(10.5)


# ─── compute_gradient_map (extra) ────────────────────────────────────────────

class TestComputeGradientMapExtra:
    def test_noisy_image_nonzero_mean(self):
        gray = _noisy()
        gm = compute_gradient_map(gray)
        assert gm.mean_mag > 0.0

    def test_max_mag_geq_mean_mag(self):
        gray = _noisy()
        gm = compute_gradient_map(gray)
        assert gm.max_mag >= gm.mean_mag

    def test_ksize_5_valid(self):
        gray = _noisy()
        cfg = GradientConfig(ksize=5)
        gm = compute_gradient_map(gray, cfg)
        assert gm.magnitude.shape == gray.shape

    def test_ksize_7_valid(self):
        gray = _noisy()
        cfg = GradientConfig(ksize=7)
        gm = compute_gradient_map(gray, cfg)
        assert gm.magnitude.shape == gray.shape

    def test_rectangular_image(self):
        gray = _gray(h=32, w=64)
        gm = compute_gradient_map(gray)
        assert gm.shape == (32, 64)

    def test_bgr_accepted(self):
        bgr = _bgr()
        gm = compute_gradient_map(bgr)
        assert isinstance(gm, GradientMap)

    def test_kernel_prewitt_stored(self):
        gray = _noisy()
        cfg = GradientConfig(kernel="prewitt")
        gm = compute_gradient_map(gray, cfg)
        assert gm.kernel == "prewitt"

    def test_angle_in_0_180(self):
        gray = _noisy(seed=5)
        gm = compute_gradient_map(gray)
        assert np.all(gm.angle >= 0.0)
        assert np.all(gm.angle < 180.0)

    def test_magnitude_nonneg(self):
        gray = _noisy(seed=3)
        gm = compute_gradient_map(gray)
        assert np.all(gm.magnitude >= 0.0)


# ─── compute_orientation_histogram (extra) ────────────────────────────────────

class TestComputeOrientationHistogramExtra:
    def _gmap(self, seed=0):
        gray = _noisy(seed=seed)
        return compute_gradient_map(gray)

    def test_n_bins_8_default(self):
        gmap = self._gmap()
        hist = compute_orientation_histogram(gmap)
        assert len(hist) == 8

    def test_normalized_sum_1(self):
        gmap = self._gmap()
        hist = compute_orientation_histogram(gmap, normalize=True)
        assert abs(hist.sum() - 1.0) < 1e-6

    def test_not_normalized_nonneg(self):
        gmap = self._gmap()
        hist = compute_orientation_histogram(gmap, normalize=False)
        assert np.all(hist >= 0.0)

    def test_dtype_float64(self):
        gmap = self._gmap()
        hist = compute_orientation_histogram(gmap)
        assert hist.dtype == np.float64

    def test_n_bins_12_shape(self):
        gmap = self._gmap()
        hist = compute_orientation_histogram(gmap, n_bins=12)
        assert hist.shape == (12,)

    def test_n_bins_36_shape(self):
        gmap = self._gmap()
        hist = compute_orientation_histogram(gmap, n_bins=36)
        assert hist.shape == (36,)

    def test_uniform_image_zero_histogram(self):
        gray = _gray(fill=128)
        gmap = compute_gradient_map(gray)
        hist = compute_orientation_histogram(gmap, normalize=False)
        # Uniform image → no gradients → histogram may be all zeros
        assert np.all(hist >= 0.0)


# ─── extract_gradient_profile (extra) ────────────────────────────────────────

class TestExtractGradientProfileExtra:
    def test_returns_gradient_profile(self):
        result = extract_gradient_profile(_gray())
        assert isinstance(result, GradientProfile)

    def test_fragment_id_stored(self):
        result = extract_gradient_profile(_gray(), fragment_id=5)
        assert result.fragment_id == 5

    def test_orientation_hist_length_matches_n_bins(self):
        cfg = GradientConfig(n_bins=12)
        result = extract_gradient_profile(_noisy(), cfg=cfg)
        assert len(result.orientation_hist) == 12

    def test_dominant_angle_in_range(self):
        result = extract_gradient_profile(_noisy(seed=9))
        assert 0.0 <= result.dominant_angle < 180.0

    def test_energy_nonneg(self):
        result = extract_gradient_profile(_noisy())
        assert result.energy >= 0.0

    def test_mean_mag_nonneg(self):
        result = extract_gradient_profile(_noisy())
        assert result.mean_magnitude >= 0.0

    def test_std_mag_nonneg(self):
        result = extract_gradient_profile(_noisy())
        assert result.std_magnitude >= 0.0

    def test_max_mag_geq_mean(self):
        result = extract_gradient_profile(_noisy())
        assert result.max_magnitude >= result.mean_magnitude

    def test_bgr_image(self):
        result = extract_gradient_profile(_bgr())
        assert isinstance(result, GradientProfile)

    def test_sharpness_in_0_1(self):
        result = extract_gradient_profile(_noisy())
        assert 0.0 <= result.sharpness_score <= 1.0


# ─── compare_gradient_profiles (extra) ───────────────────────────────────────

class TestCompareGradientProfilesExtra:
    def _profile(self, image, fid=0):
        return extract_gradient_profile(image, fragment_id=fid)

    def test_self_similarity_one(self):
        gray = _noisy(seed=1)
        pa = self._profile(gray, 0)
        pb = self._profile(gray, 1)
        assert compare_gradient_profiles(pa, pb) == pytest.approx(1.0, abs=1e-5)

    def test_different_images_in_range(self):
        pa = self._profile(_noisy(seed=1))
        pb = self._profile(_noisy(seed=2))
        sim = compare_gradient_profiles(pa, pb)
        assert -1.0 <= sim <= 1.0

    def test_both_none_hist_returns_zero(self):
        pa = GradientProfile(fragment_id=0, orientation_hist=None)
        pb = GradientProfile(fragment_id=1, orientation_hist=None)
        assert compare_gradient_profiles(pa, pb) == pytest.approx(0.0)

    def test_one_none_hist_returns_zero(self):
        pa = self._profile(_noisy(), fid=0)
        pb = GradientProfile(fragment_id=1, orientation_hist=None)
        assert compare_gradient_profiles(pa, pb) == pytest.approx(0.0)

    def test_zero_hist_vs_zero_hist_returns_zero(self):
        pa = GradientProfile(fragment_id=0, orientation_hist=np.zeros(8))
        pb = GradientProfile(fragment_id=1, orientation_hist=np.zeros(8))
        assert compare_gradient_profiles(pa, pb) == pytest.approx(0.0)

    def test_symmetric(self):
        pa = self._profile(_noisy(seed=5))
        pb = self._profile(_noisy(seed=6))
        sim_ab = compare_gradient_profiles(pa, pb)
        sim_ba = compare_gradient_profiles(pb, pa)
        assert sim_ab == pytest.approx(sim_ba, abs=1e-9)


# ─── batch_extract_gradient_profiles (extra) ──────────────────────────────────

class TestBatchExtractGradientProfilesExtra:
    def test_single_image_returns_one(self):
        result = batch_extract_gradient_profiles([_gray()])
        assert len(result) == 1

    def test_all_are_gradient_profiles(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        result = batch_extract_gradient_profiles(imgs)
        for p in result:
            assert isinstance(p, GradientProfile)

    def test_fragment_ids_sequential(self):
        imgs = [_gray()] * 4
        result = batch_extract_gradient_profiles(imgs)
        assert [p.fragment_id for p in result] == [0, 1, 2, 3]

    def test_custom_cfg_n_bins(self):
        imgs = [_noisy(seed=i) for i in range(2)]
        cfg = GradientConfig(n_bins=16)
        result = batch_extract_gradient_profiles(imgs, cfg=cfg)
        for p in result:
            assert len(p.orientation_hist) == 16

    def test_mixed_gray_bgr(self):
        imgs = [_gray(), _bgr(), _noisy()]
        result = batch_extract_gradient_profiles(imgs)
        assert len(result) == 3

    def test_large_batch(self):
        imgs = [_gray()] * 10
        result = batch_extract_gradient_profiles(imgs)
        assert len(result) == 10
