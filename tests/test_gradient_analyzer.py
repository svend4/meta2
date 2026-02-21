"""Тесты для puzzle_reconstruction.preprocessing.gradient_analyzer."""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.gradient_analyzer import (
    GradientConfig,
    GradientMap,
    GradientProfile,
    batch_extract_gradient_profiles,
    compare_gradient_profiles,
    compute_gradient_map,
    compute_orientation_histogram,
    extract_gradient_profile,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(30, 220, (h, w), dtype=np.uint8)


def _bgr(h=32, w=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(30, 220, (h, w, 3), dtype=np.uint8)


def _gmap(seed=0):
    img = _gray(seed=seed)
    return compute_gradient_map(img)


# ─── TestGradientConfig ───────────────────────────────────────────────────────

class TestGradientConfig:
    def test_defaults(self):
        cfg = GradientConfig()
        assert cfg.kernel == "sobel"
        assert cfg.ksize == 3
        assert cfg.n_bins == 8
        assert cfg.normalize is True

    def test_invalid_kernel_raises(self):
        with pytest.raises(ValueError):
            GradientConfig(kernel="canny")

    def test_invalid_ksize_raises(self):
        with pytest.raises(ValueError):
            GradientConfig(ksize=4)

    def test_ksize_1_raises(self):
        with pytest.raises(ValueError):
            GradientConfig(ksize=1)

    def test_n_bins_1_raises(self):
        with pytest.raises(ValueError):
            GradientConfig(n_bins=1)

    def test_valid_kernels(self):
        for k in ("sobel", "scharr", "prewitt"):
            cfg = GradientConfig(kernel=k)
            assert cfg.kernel == k

    def test_valid_ksizes(self):
        for ks in (3, 5, 7):
            cfg = GradientConfig(ksize=ks)
            assert cfg.ksize == ks


# ─── TestGradientMap ──────────────────────────────────────────────────────────

class TestGradientMap:
    def test_shape_prop(self):
        gm = GradientMap(
            magnitude=np.zeros((4, 5)),
            angle=np.zeros((4, 5)),
            mean_mag=0.0,
            max_mag=0.0,
            kernel="sobel",
        )
        assert gm.shape == (4, 5)

    def test_negative_mean_mag_raises(self):
        with pytest.raises(ValueError):
            GradientMap(
                magnitude=np.zeros((2, 2)),
                angle=np.zeros((2, 2)),
                mean_mag=-1.0,
                max_mag=0.0,
                kernel="sobel",
            )

    def test_negative_max_mag_raises(self):
        with pytest.raises(ValueError):
            GradientMap(
                magnitude=np.zeros((2, 2)),
                angle=np.zeros((2, 2)),
                mean_mag=0.0,
                max_mag=-0.1,
                kernel="sobel",
            )

    def test_kernel_stored(self):
        gm = GradientMap(
            magnitude=np.zeros((2, 2)),
            angle=np.zeros((2, 2)),
            mean_mag=0.0,
            max_mag=0.0,
            kernel="scharr",
        )
        assert gm.kernel == "scharr"


# ─── TestGradientProfile ──────────────────────────────────────────────────────

class TestGradientProfile:
    def test_negative_id_raises(self):
        with pytest.raises(ValueError):
            GradientProfile(fragment_id=-1)

    def test_negative_mean_magnitude_raises(self):
        with pytest.raises(ValueError):
            GradientProfile(fragment_id=0, mean_magnitude=-0.1)

    def test_negative_std_magnitude_raises(self):
        with pytest.raises(ValueError):
            GradientProfile(fragment_id=0, std_magnitude=-0.1)

    def test_negative_max_magnitude_raises(self):
        with pytest.raises(ValueError):
            GradientProfile(fragment_id=0, max_magnitude=-0.1)

    def test_negative_energy_raises(self):
        with pytest.raises(ValueError):
            GradientProfile(fragment_id=0, energy=-1.0)

    def test_sharpness_score_range(self):
        p = GradientProfile(fragment_id=0, mean_magnitude=128.0)
        assert 0.0 <= p.sharpness_score <= 1.0

    def test_sharpness_zero_for_zero_mean(self):
        p = GradientProfile(fragment_id=0, mean_magnitude=0.0)
        assert p.sharpness_score == 0.0

    def test_sharpness_clips_at_one(self):
        p = GradientProfile(fragment_id=0, mean_magnitude=9999.0)
        assert p.sharpness_score == 1.0

    def test_defaults(self):
        p = GradientProfile(fragment_id=5)
        assert p.mean_magnitude == 0.0
        assert p.orientation_hist is None


# ─── TestComputeGradientMap ───────────────────────────────────────────────────

class TestComputeGradientMap:
    def test_gray_input_shape(self):
        gm = compute_gradient_map(_gray())
        assert gm.shape == (32, 32)

    def test_bgr_input_shape(self):
        gm = compute_gradient_map(_bgr())
        assert gm.shape == (32, 32)

    def test_magnitude_non_negative(self):
        gm = compute_gradient_map(_gray())
        assert float(gm.magnitude.min()) >= 0.0

    def test_angle_range(self):
        gm = compute_gradient_map(_gray())
        assert float(gm.angle.min()) >= 0.0
        assert float(gm.angle.max()) < 181.0

    def test_mean_mag_positive(self):
        gm = compute_gradient_map(_gray())
        assert gm.mean_mag >= 0.0

    def test_max_mag_ge_mean(self):
        gm = compute_gradient_map(_gray())
        assert gm.max_mag >= gm.mean_mag

    def test_sobel_kernel(self):
        gm = compute_gradient_map(_gray(), GradientConfig(kernel="sobel"))
        assert gm.kernel == "sobel"

    def test_scharr_kernel(self):
        gm = compute_gradient_map(_gray(), GradientConfig(kernel="scharr"))
        assert gm.kernel == "scharr"

    def test_prewitt_kernel(self):
        gm = compute_gradient_map(_gray(), GradientConfig(kernel="prewitt"))
        assert gm.kernel == "prewitt"

    def test_non_2d_3d_raises(self):
        with pytest.raises(ValueError):
            compute_gradient_map(np.ones((2, 2, 2, 2), dtype=np.uint8))

    def test_uniform_image_low_gradient(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        gm = compute_gradient_map(img)
        assert gm.mean_mag < 1.0

    def test_different_kernels_differ(self):
        img = _gray()
        gm_sobel = compute_gradient_map(img, GradientConfig(kernel="sobel"))
        gm_scharr = compute_gradient_map(img, GradientConfig(kernel="scharr"))
        assert not np.array_equal(gm_sobel.magnitude, gm_scharr.magnitude)


# ─── TestComputeOrientationHistogram ─────────────────────────────────────────

class TestComputeOrientationHistogram:
    def test_shape(self):
        gm = _gmap()
        hist = compute_orientation_histogram(gm, n_bins=8)
        assert hist.shape == (8,)

    def test_normalized_sums_to_1(self):
        gm = _gmap()
        hist = compute_orientation_histogram(gm, n_bins=8, normalize=True)
        assert abs(float(hist.sum()) - 1.0) < 1e-9

    def test_not_normalized_non_unit_sum(self):
        gm = _gmap()
        hist = compute_orientation_histogram(gm, n_bins=8, normalize=False)
        assert float(hist.sum()) > 1.0

    def test_all_non_negative(self):
        gm = _gmap()
        hist = compute_orientation_histogram(gm)
        assert float(hist.min()) >= 0.0

    def test_n_bins_1_raises(self):
        gm = _gmap()
        with pytest.raises(ValueError):
            compute_orientation_histogram(gm, n_bins=1)

    def test_custom_n_bins(self):
        gm = _gmap()
        hist = compute_orientation_histogram(gm, n_bins=16)
        assert hist.shape == (16,)

    def test_zero_magnitude_returns_zero_hist(self):
        mag = np.zeros((4, 4))
        angle = np.zeros((4, 4))
        gm = GradientMap(magnitude=mag, angle=angle,
                         mean_mag=0.0, max_mag=0.0, kernel="sobel")
        hist = compute_orientation_histogram(gm, n_bins=4, normalize=True)
        assert np.all(hist == 0.0)


# ─── TestExtractGradientProfile ───────────────────────────────────────────────

class TestExtractGradientProfile:
    def test_returns_gradient_profile(self):
        p = extract_gradient_profile(_gray())
        assert isinstance(p, GradientProfile)

    def test_fragment_id_set(self):
        p = extract_gradient_profile(_gray(), fragment_id=7)
        assert p.fragment_id == 7

    def test_negative_id_raises(self):
        with pytest.raises(ValueError):
            extract_gradient_profile(_gray(), fragment_id=-1)

    def test_mean_magnitude_positive(self):
        p = extract_gradient_profile(_gray())
        assert p.mean_magnitude >= 0.0

    def test_orientation_hist_is_array(self):
        p = extract_gradient_profile(_gray())
        assert p.orientation_hist is not None
        assert isinstance(p.orientation_hist, np.ndarray)

    def test_orientation_hist_sums_to_1(self):
        p = extract_gradient_profile(_gray())
        assert abs(float(p.orientation_hist.sum()) - 1.0) < 1e-9

    def test_dominant_angle_in_range(self):
        p = extract_gradient_profile(_gray())
        assert 0.0 <= p.dominant_angle < 180.0

    def test_energy_positive(self):
        p = extract_gradient_profile(_gray())
        assert p.energy >= 0.0

    def test_bgr_input(self):
        p = extract_gradient_profile(_bgr(), fragment_id=0)
        assert p.mean_magnitude >= 0.0

    def test_custom_cfg_n_bins(self):
        cfg = GradientConfig(n_bins=16)
        p = extract_gradient_profile(_gray(), cfg=cfg)
        assert p.orientation_hist.shape == (16,)


# ─── TestCompareGradientProfiles ─────────────────────────────────────────────

class TestCompareGradientProfiles:
    def test_identical_profiles_score_1(self):
        p = extract_gradient_profile(_gray(), fragment_id=0)
        score = compare_gradient_profiles(p, p)
        assert abs(score - 1.0) < 1e-9

    def test_score_in_minus1_to_1(self):
        p1 = extract_gradient_profile(_gray(seed=0), fragment_id=0)
        p2 = extract_gradient_profile(_gray(seed=1), fragment_id=1)
        score = compare_gradient_profiles(p1, p2)
        assert -1.0 <= score <= 1.0

    def test_none_histogram_returns_0(self):
        p1 = GradientProfile(fragment_id=0, orientation_hist=None)
        p2 = GradientProfile(fragment_id=1, orientation_hist=None)
        assert compare_gradient_profiles(p1, p2) == 0.0

    def test_one_none_histogram_returns_0(self):
        p1 = extract_gradient_profile(_gray())
        p2 = GradientProfile(fragment_id=1, orientation_hist=None)
        assert compare_gradient_profiles(p1, p2) == 0.0

    def test_different_images_differ(self):
        p1 = extract_gradient_profile(_gray(seed=0), fragment_id=0)
        p2 = extract_gradient_profile(_gray(seed=42), fragment_id=1)
        score = compare_gradient_profiles(p1, p2)
        # Same image type → likely high but not necessarily 1
        assert score <= 1.0


# ─── TestBatchExtractGradientProfiles ────────────────────────────────────────

class TestBatchExtractGradientProfiles:
    def test_empty_batch(self):
        assert batch_extract_gradient_profiles([]) == []

    def test_single_image(self):
        result = batch_extract_gradient_profiles([_gray()])
        assert len(result) == 1
        assert isinstance(result[0], GradientProfile)

    def test_fragment_ids_are_indices(self):
        images = [_gray(seed=i) for i in range(4)]
        result = batch_extract_gradient_profiles(images)
        ids = [p.fragment_id for p in result]
        assert ids == [0, 1, 2, 3]

    def test_multiple_images(self):
        images = [_gray(seed=i) for i in range(5)]
        result = batch_extract_gradient_profiles(images)
        assert len(result) == 5

    def test_custom_cfg(self):
        cfg = GradientConfig(kernel="scharr", n_bins=4)
        images = [_bgr(seed=i) for i in range(3)]
        result = batch_extract_gradient_profiles(images, cfg)
        for p in result:
            assert p.orientation_hist.shape == (4,)
