"""Тесты для puzzle_reconstruction/preprocessing/gradient_analyzer.py."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_gray(h=64, w=64, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)


def make_noisy(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def make_bgr(h=64, w=64, fill=128):
    return np.full((h, w, 3), fill, dtype=np.uint8)


# ─── GradientConfig ───────────────────────────────────────────────────────────

class TestGradientConfig:
    def test_defaults(self):
        cfg = GradientConfig()
        assert cfg.kernel == "sobel"
        assert cfg.ksize == 3
        assert cfg.n_bins == 8
        assert cfg.normalize is True

    def test_invalid_kernel_raises(self):
        with pytest.raises(ValueError, match="kernel"):
            GradientConfig(kernel="laplacian")

    def test_invalid_ksize_raises(self):
        with pytest.raises(ValueError, match="ksize"):
            GradientConfig(ksize=4)

    def test_ksize_1_raises(self):
        with pytest.raises(ValueError):
            GradientConfig(ksize=1)

    def test_valid_kernels(self):
        for k in ("sobel", "scharr", "prewitt"):
            cfg = GradientConfig(kernel=k)
            assert cfg.kernel == k

    def test_valid_ksizes(self):
        for ks in (3, 5, 7):
            cfg = GradientConfig(ksize=ks)
            assert cfg.ksize == ks

    def test_n_bins_less_than_2_raises(self):
        with pytest.raises(ValueError, match="n_bins"):
            GradientConfig(n_bins=1)

    def test_n_bins_2_valid(self):
        cfg = GradientConfig(n_bins=2)
        assert cfg.n_bins == 2


# ─── GradientMap ──────────────────────────────────────────────────────────────

class TestGradientMap:
    def test_creation(self):
        mag = np.ones((10, 10))
        ang = np.zeros((10, 10))
        gm = GradientMap(magnitude=mag, angle=ang, mean_mag=1.0, max_mag=1.0, kernel="sobel")
        assert gm.mean_mag == pytest.approx(1.0)
        assert gm.max_mag == pytest.approx(1.0)
        assert gm.kernel == "sobel"

    def test_negative_mean_mag_raises(self):
        mag = np.ones((5, 5))
        ang = np.zeros((5, 5))
        with pytest.raises(ValueError, match="mean_mag"):
            GradientMap(magnitude=mag, angle=ang, mean_mag=-1.0, max_mag=1.0, kernel="sobel")

    def test_negative_max_mag_raises(self):
        mag = np.ones((5, 5))
        ang = np.zeros((5, 5))
        with pytest.raises(ValueError, match="max_mag"):
            GradientMap(magnitude=mag, angle=ang, mean_mag=1.0, max_mag=-1.0, kernel="sobel")

    def test_shape_property(self):
        mag = np.ones((8, 12))
        ang = np.zeros((8, 12))
        gm = GradientMap(magnitude=mag, angle=ang, mean_mag=1.0, max_mag=1.0, kernel="sobel")
        assert gm.shape == (8, 12)


# ─── GradientProfile ──────────────────────────────────────────────────────────

class TestGradientProfile:
    def test_creation_defaults(self):
        gp = GradientProfile(fragment_id=0)
        assert gp.fragment_id == 0
        assert gp.mean_magnitude == 0.0
        assert gp.std_magnitude == 0.0
        assert gp.max_magnitude == 0.0
        assert gp.energy == 0.0
        assert gp.dominant_angle == 0.0
        assert gp.orientation_hist is None

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError, match="fragment_id"):
            GradientProfile(fragment_id=-1)

    def test_negative_mean_magnitude_raises(self):
        with pytest.raises(ValueError, match="mean_magnitude"):
            GradientProfile(fragment_id=0, mean_magnitude=-1.0)

    def test_negative_std_magnitude_raises(self):
        with pytest.raises(ValueError, match="std_magnitude"):
            GradientProfile(fragment_id=0, std_magnitude=-1.0)

    def test_negative_max_magnitude_raises(self):
        with pytest.raises(ValueError, match="max_magnitude"):
            GradientProfile(fragment_id=0, max_magnitude=-0.1)

    def test_negative_energy_raises(self):
        with pytest.raises(ValueError, match="energy"):
            GradientProfile(fragment_id=0, energy=-1.0)

    def test_sharpness_score_zero(self):
        gp = GradientProfile(fragment_id=0, mean_magnitude=0.0)
        assert gp.sharpness_score == pytest.approx(0.0)

    def test_sharpness_score_max(self):
        gp = GradientProfile(fragment_id=0, mean_magnitude=255.0)
        assert gp.sharpness_score == pytest.approx(1.0)

    def test_sharpness_score_clipped(self):
        gp = GradientProfile(fragment_id=0, mean_magnitude=1000.0)
        assert gp.sharpness_score == pytest.approx(1.0)

    def test_sharpness_score_mid(self):
        gp = GradientProfile(fragment_id=0, mean_magnitude=127.5)
        assert 0.0 < gp.sharpness_score < 1.0


# ─── compute_gradient_map ─────────────────────────────────────────────────────

class TestComputeGradientMap:
    def test_returns_gradient_map(self):
        gray = make_gray()
        result = compute_gradient_map(gray)
        assert isinstance(result, GradientMap)

    def test_magnitude_shape(self):
        gray = make_gray(32, 48)
        result = compute_gradient_map(gray)
        assert result.magnitude.shape == (32, 48)

    def test_angle_shape(self):
        gray = make_gray(32, 48)
        result = compute_gradient_map(gray)
        assert result.angle.shape == (32, 48)

    def test_magnitude_nonnegative(self):
        gray = make_noisy()
        result = compute_gradient_map(gray)
        assert np.all(result.magnitude >= 0)

    def test_angle_range(self):
        gray = make_noisy()
        result = compute_gradient_map(gray)
        assert np.all(result.angle >= 0.0)
        assert np.all(result.angle < 180.0)

    def test_mean_mag_nonnegative(self):
        gray = make_gray()
        result = compute_gradient_map(gray)
        assert result.mean_mag >= 0.0

    def test_max_mag_nonneg(self):
        gray = make_gray()
        result = compute_gradient_map(gray)
        assert result.max_mag >= 0.0

    def test_accepts_3d_bgr(self):
        bgr = make_bgr()
        result = compute_gradient_map(bgr)
        assert isinstance(result, GradientMap)

    def test_ndim_1_raises(self):
        with pytest.raises(ValueError):
            compute_gradient_map(np.array([1, 2, 3], dtype=np.uint8))

    def test_kernel_sobel(self):
        gray = make_noisy()
        result = compute_gradient_map(gray, GradientConfig(kernel="sobel"))
        assert result.kernel == "sobel"

    def test_kernel_scharr(self):
        gray = make_noisy()
        result = compute_gradient_map(gray, GradientConfig(kernel="scharr"))
        assert result.kernel == "scharr"

    def test_kernel_prewitt(self):
        gray = make_noisy()
        result = compute_gradient_map(gray, GradientConfig(kernel="prewitt"))
        assert result.kernel == "prewitt"

    def test_uniform_image_zero_gradients(self):
        gray = make_gray(fill=128)
        result = compute_gradient_map(gray)
        assert result.mean_mag == pytest.approx(0.0, abs=1e-6)

    def test_ksize_5(self):
        gray = make_noisy()
        result = compute_gradient_map(gray, GradientConfig(ksize=5))
        assert result.magnitude.shape == gray.shape


# ─── compute_orientation_histogram ────────────────────────────────────────────

class TestComputeOrientationHistogram:
    def _make_gmap(self, seed=0):
        gray = make_noisy(seed=seed)
        return compute_gradient_map(gray)

    def test_returns_1d_array(self):
        gmap = self._make_gmap()
        hist = compute_orientation_histogram(gmap)
        assert hist.ndim == 1

    def test_shape_matches_n_bins(self):
        gmap = self._make_gmap()
        hist = compute_orientation_histogram(gmap, n_bins=16)
        assert len(hist) == 16

    def test_dtype_float64(self):
        gmap = self._make_gmap()
        hist = compute_orientation_histogram(gmap)
        assert hist.dtype == np.float64

    def test_n_bins_less_than_2_raises(self):
        gmap = self._make_gmap()
        with pytest.raises(ValueError, match="n_bins"):
            compute_orientation_histogram(gmap, n_bins=1)

    def test_normalized_sums_to_1(self):
        gmap = self._make_gmap()
        hist = compute_orientation_histogram(gmap, normalize=True)
        assert abs(hist.sum() - 1.0) < 1e-6

    def test_not_normalized_nonnegative(self):
        gmap = self._make_gmap()
        hist = compute_orientation_histogram(gmap, normalize=False)
        assert np.all(hist >= 0.0)


# ─── extract_gradient_profile ─────────────────────────────────────────────────

class TestExtractGradientProfile:
    def test_returns_gradient_profile(self):
        gray = make_gray()
        result = extract_gradient_profile(gray)
        assert isinstance(result, GradientProfile)

    def test_fragment_id_stored(self):
        gray = make_gray()
        result = extract_gradient_profile(gray, fragment_id=7)
        assert result.fragment_id == 7

    def test_negative_fragment_id_raises(self):
        gray = make_gray()
        with pytest.raises(ValueError):
            extract_gradient_profile(gray, fragment_id=-1)

    def test_mean_magnitude_nonnegative(self):
        gray = make_noisy()
        result = extract_gradient_profile(gray)
        assert result.mean_magnitude >= 0.0

    def test_std_magnitude_nonnegative(self):
        gray = make_noisy()
        result = extract_gradient_profile(gray)
        assert result.std_magnitude >= 0.0

    def test_max_magnitude_nonnegative(self):
        gray = make_noisy()
        result = extract_gradient_profile(gray)
        assert result.max_magnitude >= 0.0

    def test_energy_nonnegative(self):
        gray = make_noisy()
        result = extract_gradient_profile(gray)
        assert result.energy >= 0.0

    def test_orientation_hist_not_none(self):
        gray = make_noisy()
        result = extract_gradient_profile(gray)
        assert result.orientation_hist is not None

    def test_hist_shape_matches_n_bins(self):
        gray = make_noisy()
        cfg = GradientConfig(n_bins=16)
        result = extract_gradient_profile(gray, cfg=cfg)
        assert len(result.orientation_hist) == 16

    def test_dominant_angle_in_range(self):
        gray = make_noisy()
        result = extract_gradient_profile(gray)
        assert 0.0 <= result.dominant_angle < 180.0


# ─── compare_gradient_profiles ────────────────────────────────────────────────

class TestCompareGradientProfiles:
    def _profile(self, image, fid=0):
        return extract_gradient_profile(image, fragment_id=fid)

    def test_same_image_high_similarity(self):
        gray = make_noisy()
        pa = self._profile(gray, 0)
        pb = self._profile(gray, 1)
        sim = compare_gradient_profiles(pa, pb)
        assert sim > 0.99

    def test_result_in_neg1_1(self):
        pa = self._profile(make_noisy(seed=0))
        pb = self._profile(make_noisy(seed=1))
        sim = compare_gradient_profiles(pa, pb)
        assert -1.0 <= sim <= 1.0

    def test_none_hist_returns_0(self):
        pa = GradientProfile(fragment_id=0, orientation_hist=None)
        pb = GradientProfile(fragment_id=1, orientation_hist=None)
        assert compare_gradient_profiles(pa, pb) == pytest.approx(0.0)

    def test_one_none_hist_returns_0(self):
        gray = make_noisy()
        pa = self._profile(gray)
        pb = GradientProfile(fragment_id=1, orientation_hist=None)
        assert compare_gradient_profiles(pa, pb) == pytest.approx(0.0)

    def test_zero_histogram_returns_0(self):
        pa = GradientProfile(fragment_id=0, orientation_hist=np.zeros(8))
        pb = GradientProfile(fragment_id=1, orientation_hist=np.zeros(8))
        assert compare_gradient_profiles(pa, pb) == pytest.approx(0.0)


# ─── batch_extract_gradient_profiles ─────────────────────────────────────────

class TestBatchExtractGradientProfiles:
    def test_empty_returns_empty(self):
        result = batch_extract_gradient_profiles([])
        assert result == []

    def test_length_matches_input(self):
        images = [make_gray()] * 4
        result = batch_extract_gradient_profiles(images)
        assert len(result) == 4

    def test_fragment_ids_sequential(self):
        images = [make_gray()] * 3
        result = batch_extract_gradient_profiles(images)
        assert [p.fragment_id for p in result] == [0, 1, 2]

    def test_returns_list_of_gradient_profiles(self):
        images = [make_noisy(seed=i) for i in range(3)]
        result = batch_extract_gradient_profiles(images)
        for p in result:
            assert isinstance(p, GradientProfile)

    def test_custom_config(self):
        images = [make_gray()] * 2
        cfg = GradientConfig(kernel="scharr", n_bins=12)
        result = batch_extract_gradient_profiles(images, cfg=cfg)
        for p in result:
            assert len(p.orientation_hist) == 12
