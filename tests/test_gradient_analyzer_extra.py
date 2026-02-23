"""Extra tests for puzzle_reconstruction/preprocessing/gradient_analyzer.py"""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, seed=0):
    return np.random.default_rng(seed).integers(30, 220, (h, w), dtype=np.uint8)


def _bgr(h=32, w=32, seed=0):
    return np.random.default_rng(seed).integers(30, 220, (h, w, 3), dtype=np.uint8)


def _zero_gmap():
    mag = np.zeros((8, 8))
    ang = np.zeros((8, 8))
    return GradientMap(magnitude=mag, angle=ang, mean_mag=0.0, max_mag=0.0, kernel="sobel")


def _gmap(seed=0):
    return compute_gradient_map(_gray(seed=seed))


# ─── TestGradientConfigExtra ──────────────────────────────────────────────────

class TestGradientConfigExtra:
    def test_ksize_7_valid(self):
        cfg = GradientConfig(ksize=7)
        assert cfg.ksize == 7

    def test_ksize_5_valid(self):
        cfg = GradientConfig(ksize=5)
        assert cfg.ksize == 5

    def test_n_bins_2_valid(self):
        cfg = GradientConfig(n_bins=2)
        assert cfg.n_bins == 2

    def test_n_bins_16(self):
        cfg = GradientConfig(n_bins=16)
        assert cfg.n_bins == 16

    def test_normalize_false(self):
        cfg = GradientConfig(normalize=False)
        assert cfg.normalize is False


# ─── TestGradientMapExtra ─────────────────────────────────────────────────────

class TestGradientMapExtra:
    def test_shape_non_square(self):
        gm = GradientMap(
            magnitude=np.zeros((10, 20)),
            angle=np.zeros((10, 20)),
            mean_mag=0.0, max_mag=0.0, kernel="prewitt",
        )
        assert gm.shape == (10, 20)

    def test_kernel_prewitt_stored(self):
        gm = GradientMap(
            magnitude=np.ones((4, 4)),
            angle=np.zeros((4, 4)),
            mean_mag=1.0, max_mag=1.0, kernel="prewitt",
        )
        assert gm.kernel == "prewitt"

    def test_mean_max_zero_valid(self):
        gm = _zero_gmap()
        assert gm.mean_mag == pytest.approx(0.0)
        assert gm.max_mag == pytest.approx(0.0)

    def test_large_magnitude_values(self):
        mag = np.full((5, 5), 500.0)
        gm = GradientMap(magnitude=mag, angle=np.zeros((5, 5)),
                         mean_mag=500.0, max_mag=500.0, kernel="scharr")
        assert gm.mean_mag == pytest.approx(500.0)


# ─── TestGradientProfileExtra ─────────────────────────────────────────────────

class TestGradientProfileExtra:
    def test_std_magnitude_zero_valid(self):
        p = GradientProfile(fragment_id=0, std_magnitude=0.0)
        assert p.std_magnitude == pytest.approx(0.0)

    def test_energy_zero_valid(self):
        p = GradientProfile(fragment_id=0, energy=0.0)
        assert p.energy == pytest.approx(0.0)

    def test_max_magnitude_zero_valid(self):
        p = GradientProfile(fragment_id=0, max_magnitude=0.0)
        assert p.max_magnitude == pytest.approx(0.0)

    def test_sharpness_255_mean_clips_to_one(self):
        p = GradientProfile(fragment_id=0, mean_magnitude=255.0)
        assert p.sharpness_score == pytest.approx(1.0)

    def test_sharpness_127_mean(self):
        p = GradientProfile(fragment_id=0, mean_magnitude=127.5)
        assert p.sharpness_score == pytest.approx(0.5, abs=1e-3)

    def test_dominant_angle_default_zero(self):
        p = GradientProfile(fragment_id=0)
        assert p.dominant_angle == pytest.approx(0.0)


# ─── TestComputeGradientMapExtra ──────────────────────────────────────────────

class TestComputeGradientMapExtra:
    def test_non_square_gray(self):
        gm = compute_gradient_map(_gray(h=16, w=48))
        assert gm.shape == (16, 48)

    def test_ksize_5_sobel(self):
        gm = compute_gradient_map(_gray(), GradientConfig(kernel="sobel", ksize=5))
        assert gm.shape == (32, 32)
        assert gm.kernel == "sobel"

    def test_ksize_7_sobel(self):
        gm = compute_gradient_map(_gray(), GradientConfig(kernel="sobel", ksize=7))
        assert gm.shape == (32, 32)

    def test_prewitt_angle_range(self):
        gm = compute_gradient_map(_gray(), GradientConfig(kernel="prewitt"))
        assert float(gm.angle.min()) >= 0.0
        assert float(gm.angle.max()) < 181.0

    def test_bgr_non_square(self):
        gm = compute_gradient_map(_bgr(h=20, w=40))
        assert gm.shape == (20, 40)

    def test_magnitude_nonneg_scharr(self):
        gm = compute_gradient_map(_gray(), GradientConfig(kernel="scharr"))
        assert float(gm.magnitude.min()) >= 0.0


# ─── TestComputeOrientationHistogramExtra ────────────────────────────────────

class TestComputeOrientationHistogramExtra:
    def test_2_bins(self):
        hist = compute_orientation_histogram(_gmap(), n_bins=2, normalize=True)
        assert hist.shape == (2,)
        assert abs(hist.sum() - 1.0) < 1e-9

    def test_32_bins(self):
        hist = compute_orientation_histogram(_gmap(), n_bins=32)
        assert hist.shape == (32,)

    def test_zero_gmap_normalized_all_zero(self):
        gm = _zero_gmap()
        hist = compute_orientation_histogram(gm, n_bins=4, normalize=True)
        assert np.all(hist == 0.0)

    def test_non_normalized_sum_gt_zero(self):
        hist = compute_orientation_histogram(_gmap(), n_bins=8, normalize=False)
        assert hist.sum() > 0.0

    def test_dtype_float64(self):
        hist = compute_orientation_histogram(_gmap(), n_bins=8)
        assert hist.dtype == np.float64


# ─── TestExtractGradientProfileExtra ─────────────────────────────────────────

class TestExtractGradientProfileExtra:
    def test_zero_fragment_id(self):
        p = extract_gradient_profile(_gray(), fragment_id=0)
        assert p.fragment_id == 0

    def test_large_fragment_id(self):
        p = extract_gradient_profile(_gray(), fragment_id=999)
        assert p.fragment_id == 999

    def test_n_bins_4_hist_shape(self):
        cfg = GradientConfig(n_bins=4)
        p = extract_gradient_profile(_gray(), cfg=cfg)
        assert p.orientation_hist.shape == (4,)

    def test_std_magnitude_nonneg(self):
        p = extract_gradient_profile(_gray())
        assert p.std_magnitude >= 0.0

    def test_energy_nonneg(self):
        p = extract_gradient_profile(_gray())
        assert p.energy >= 0.0

    def test_dominant_angle_in_180_range(self):
        p = extract_gradient_profile(_gray())
        assert 0.0 <= p.dominant_angle < 180.0

    def test_prewitt_kernel(self):
        cfg = GradientConfig(kernel="prewitt")
        p = extract_gradient_profile(_gray(), cfg=cfg)
        assert p.mean_magnitude >= 0.0


# ─── TestCompareGradientProfilesExtra ────────────────────────────────────────

class TestCompareGradientProfilesExtra:
    def test_zero_histogram_vs_real_returns_0(self):
        p_real = extract_gradient_profile(_gray())
        p_zero = GradientProfile(
            fragment_id=0,
            orientation_hist=np.zeros(8),
        )
        assert compare_gradient_profiles(p_real, p_zero) == pytest.approx(0.0)

    def test_symmetric(self):
        p1 = extract_gradient_profile(_gray(seed=1), fragment_id=0)
        p2 = extract_gradient_profile(_gray(seed=2), fragment_id=1)
        s12 = compare_gradient_profiles(p1, p2)
        s21 = compare_gradient_profiles(p2, p1)
        assert s12 == pytest.approx(s21, abs=1e-9)

    def test_result_float(self):
        p = extract_gradient_profile(_gray(), fragment_id=0)
        result = compare_gradient_profiles(p, p)
        assert isinstance(result, float)

    def test_same_profile_near_1(self):
        p = extract_gradient_profile(_gray(seed=5), fragment_id=0)
        score = compare_gradient_profiles(p, p)
        assert score == pytest.approx(1.0, abs=1e-9)


# ─── TestBatchExtractGradientProfilesExtra ───────────────────────────────────

class TestBatchExtractGradientProfilesExtra:
    def test_bgr_batch(self):
        images = [_bgr(seed=i) for i in range(3)]
        result = batch_extract_gradient_profiles(images)
        assert len(result) == 3

    def test_fragment_ids_sequential(self):
        images = [_gray(seed=i) for i in range(6)]
        result = batch_extract_gradient_profiles(images)
        ids = [p.fragment_id for p in result]
        assert ids == list(range(6))

    def test_all_orientation_hists_normalized(self):
        images = [_gray(seed=i) for i in range(4)]
        result = batch_extract_gradient_profiles(images)
        for p in result:
            assert abs(float(p.orientation_hist.sum()) - 1.0) < 1e-9

    def test_prewitt_cfg(self):
        cfg = GradientConfig(kernel="prewitt")
        images = [_gray(seed=0), _gray(seed=1)]
        result = batch_extract_gradient_profiles(images, cfg)
        assert len(result) == 2
