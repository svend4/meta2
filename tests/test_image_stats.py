"""Тесты для puzzle_reconstruction/utils/image_stats.py."""
import numpy as np
import pytest

from puzzle_reconstruction.utils.image_stats import (
    ImageStats,
    compute_entropy,
    compute_sharpness,
    compute_histogram_stats,
    compute_gradient_stats,
    compute_image_stats,
    compare_images,
    batch_stats,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _noisy(h=64, w=64, seed=7):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=64, w=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    img[:, :, 1] = 100
    img[:, :, 2] = 50
    return img


def _sharp(h=64, w=64):
    img = np.zeros((h, w), dtype=np.uint8)
    img[::2, :] = 255
    return img


def _blurry():
    import cv2
    return cv2.GaussianBlur(_noisy().astype(np.uint8), (15, 15), 5)


# ─── ImageStats ──────────────────────────────────────────────────────────────

class TestImageStats:
    def test_fields_exist(self):
        s = ImageStats(
            mean=100.0, std=20.0, entropy=5.0, contrast=20.0,
            sharpness=50.0, histogram=np.zeros(256, dtype=np.float32),
            percentiles={50: 100.0}, n_pixels=64 * 64,
        )
        assert s.mean == pytest.approx(100.0)
        assert s.std == pytest.approx(20.0)
        assert s.entropy == pytest.approx(5.0)
        assert s.contrast == pytest.approx(20.0)
        assert s.sharpness == pytest.approx(50.0)
        assert s.n_pixels == 64 * 64

    def test_extra_default_empty(self):
        s = ImageStats(
            mean=0, std=0, entropy=0, contrast=0, sharpness=0,
            histogram=np.zeros(256, dtype=np.float32),
            percentiles={}, n_pixels=1,
        )
        assert isinstance(s.extra, dict)
        assert len(s.extra) == 0

    def test_extra_accepts_values(self):
        s = ImageStats(
            mean=0, std=0, entropy=0, contrast=0, sharpness=0,
            histogram=np.zeros(256, dtype=np.float32),
            percentiles={}, n_pixels=1, extra={"grad_mean": 3.5},
        )
        assert s.extra["grad_mean"] == pytest.approx(3.5)

    def test_repr_contains_class_name(self):
        s = ImageStats(
            mean=128.0, std=10.0, entropy=4.5, contrast=10.0,
            sharpness=30.0, histogram=np.zeros(256, dtype=np.float32),
            percentiles={}, n_pixels=100,
        )
        assert "ImageStats" in repr(s)

    def test_repr_contains_mean(self):
        s = ImageStats(
            mean=128.0, std=10.0, entropy=4.5, contrast=10.0,
            sharpness=30.0, histogram=np.zeros(256, dtype=np.float32),
            percentiles={}, n_pixels=100,
        )
        assert "128" in repr(s)

    def test_n_pixels_via_compute(self):
        s = compute_image_stats(_gray(32, 48))
        assert s.n_pixels == 32 * 48

    def test_percentile_keys(self):
        s = compute_image_stats(_gray())
        assert set(s.percentiles.keys()) == {5, 25, 50, 75, 95}

    def test_histogram_sum(self):
        s = compute_image_stats(_noisy())
        assert s.histogram.sum() == pytest.approx(1.0, abs=1e-5)

    def test_histogram_dtype(self):
        s = compute_image_stats(_noisy())
        assert s.histogram.dtype == np.float32

    def test_histogram_nonneg(self):
        s = compute_image_stats(_noisy())
        assert np.all(s.histogram >= 0)


# ─── compute_entropy ─────────────────────────────────────────────────────────

class TestComputeEntropy:
    def test_returns_float(self):
        assert isinstance(compute_entropy(_noisy()), float)

    def test_range(self):
        e = compute_entropy(_noisy())
        assert 0.0 <= e <= 8.0

    def test_constant_image_zero(self):
        assert compute_entropy(_gray(val=100)) == pytest.approx(0.0, abs=1e-6)

    def test_all_values_present_high_entropy(self):
        # 16×16 = 256 pixels, one per gray level → maximal entropy
        img = np.arange(256, dtype=np.uint8).reshape(16, 16)
        assert compute_entropy(img) > 7.0

    def test_bgr_input(self):
        e = compute_entropy(_bgr())
        assert 0.0 <= e <= 8.0

    def test_small_image(self):
        img = np.array([[0, 255], [128, 64]], dtype=np.uint8)
        e = compute_entropy(img)
        assert 0.0 <= e <= 8.0

    def test_noisy_higher_than_constant(self):
        assert compute_entropy(_noisy()) > compute_entropy(_gray(val=100))

    def test_noisy_higher_than_text(self):
        # Noisy image has wider histogram → higher entropy
        e_noise = compute_entropy(_noisy())
        e_text = compute_entropy(_gray(val=200))
        assert e_noise > e_text


# ─── compute_sharpness ───────────────────────────────────────────────────────

class TestComputeSharpness:
    def test_returns_float(self):
        assert isinstance(compute_sharpness(_noisy()), float)

    def test_nonneg(self):
        assert compute_sharpness(_noisy()) >= 0.0

    def test_constant_zero(self):
        assert compute_sharpness(_gray()) == pytest.approx(0.0, abs=1e-6)

    def test_sharp_greater_than_blurry(self):
        s_sharp = compute_sharpness(_sharp())
        s_blurry = compute_sharpness(_blurry())
        assert s_sharp > s_blurry

    def test_bgr_input(self):
        assert compute_sharpness(_bgr()) >= 0.0

    def test_noisy_positive(self):
        assert compute_sharpness(_noisy()) > 0.0

    def test_edge_image_high(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[:, 32:] = 255
        s = compute_sharpness(img)
        assert s > 0.0


# ─── compute_histogram_stats ─────────────────────────────────────────────────

class TestComputeHistogramStats:
    def test_returns_dict(self):
        assert isinstance(compute_histogram_stats(_noisy()), dict)

    def test_keys(self):
        d = compute_histogram_stats(_noisy())
        assert set(d.keys()) == {"mean", "std", "skewness", "kurtosis"}

    def test_all_floats(self):
        for v in compute_histogram_stats(_noisy()).values():
            assert isinstance(v, float)

    def test_mean_in_range(self):
        d = compute_histogram_stats(_noisy())
        assert 0.0 <= d["mean"] <= 255.0

    def test_std_nonneg(self):
        assert compute_histogram_stats(_noisy())["std"] >= 0.0

    def test_constant_image_std_zero(self):
        d = compute_histogram_stats(_gray(val=100))
        assert d["std"] == pytest.approx(0.0, abs=1e-6)

    def test_constant_image_skew_zero(self):
        d = compute_histogram_stats(_gray(val=100))
        assert d["skewness"] == pytest.approx(0.0, abs=1e-6)

    def test_bgr_input(self):
        d = compute_histogram_stats(_bgr())
        assert 0.0 <= d["mean"] <= 255.0

    def test_uniform_gray_image_mean(self):
        img = np.arange(256, dtype=np.uint8).reshape(16, 16)
        d = compute_histogram_stats(img)
        assert d["mean"] == pytest.approx(127.5, abs=1.0)


# ─── compute_gradient_stats ──────────────────────────────────────────────────

class TestComputeGradientStats:
    def test_returns_dict(self):
        assert isinstance(compute_gradient_stats(_noisy()), dict)

    def test_keys(self):
        d = compute_gradient_stats(_noisy())
        assert set(d.keys()) == {"grad_mean", "grad_std", "grad_max", "grad_energy"}

    def test_all_floats(self):
        for v in compute_gradient_stats(_noisy()).values():
            assert isinstance(v, float)

    def test_constant_image_zero_mean(self):
        d = compute_gradient_stats(_gray())
        assert d["grad_mean"] == pytest.approx(0.0, abs=1e-3)

    def test_constant_image_zero_max(self):
        d = compute_gradient_stats(_gray())
        assert d["grad_max"] == pytest.approx(0.0, abs=1e-3)

    def test_grad_max_geq_mean(self):
        d = compute_gradient_stats(_noisy())
        assert d["grad_max"] >= d["grad_mean"]

    def test_all_nonneg(self):
        d = compute_gradient_stats(_noisy())
        for v in d.values():
            assert v >= 0.0

    def test_bgr_input(self):
        d = compute_gradient_stats(_bgr())
        assert d["grad_mean"] >= 0.0

    def test_edge_image_high_gradient(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[:, 32:] = 255
        d = compute_gradient_stats(img)
        assert d["grad_max"] > 0.0


# ─── compute_image_stats ─────────────────────────────────────────────────────

class TestComputeImageStats:
    def test_returns_imagestats(self):
        assert isinstance(compute_image_stats(_noisy()), ImageStats)

    def test_n_pixels_correct(self):
        s = compute_image_stats(_gray(20, 30))
        assert s.n_pixels == 600

    def test_mean_correct(self):
        img = _gray(val=100)
        s = compute_image_stats(img)
        assert s.mean == pytest.approx(100.0, abs=1.0)

    def test_std_zero_for_constant(self):
        s = compute_image_stats(_gray(val=50))
        assert s.std == pytest.approx(0.0, abs=1e-6)

    def test_contrast_equals_std(self):
        s = compute_image_stats(_noisy())
        assert s.contrast == pytest.approx(s.std, abs=1e-6)

    def test_histogram_length_default(self):
        s = compute_image_stats(_noisy())
        assert len(s.histogram) == 256

    def test_histogram_length_custom(self):
        s = compute_image_stats(_noisy(), hist_bins=128)
        assert len(s.histogram) == 128

    def test_percentile_values_ordered(self):
        s = compute_image_stats(_noisy())
        assert s.percentiles[5] <= s.percentiles[25] <= s.percentiles[50]
        assert s.percentiles[50] <= s.percentiles[75] <= s.percentiles[95]

    def test_extra_has_gradient_when_requested(self):
        s = compute_image_stats(_noisy(), include_gradient=True)
        assert "grad_mean" in s.extra

    def test_no_gradient_in_extra(self):
        s = compute_image_stats(_noisy(), include_gradient=False)
        assert "grad_mean" not in s.extra

    def test_bgr_input(self):
        s = compute_image_stats(_bgr())
        assert s.n_pixels == 64 * 64
        assert 0.0 <= s.mean <= 255.0

    def test_sharpness_positive_noisy(self):
        assert compute_image_stats(_noisy()).sharpness > 0.0

    def test_entropy_in_range(self):
        s = compute_image_stats(_noisy())
        assert 0.0 <= s.entropy <= 8.0

    def test_extra_has_skewness_kurtosis(self):
        s = compute_image_stats(_noisy())
        assert "skewness" in s.extra
        assert "kurtosis" in s.extra


# ─── compare_images ──────────────────────────────────────────────────────────

class TestCompareImages:
    def test_returns_dict(self):
        d = compare_images(_noisy(), _noisy(seed=99))
        assert isinstance(d, dict)

    def test_keys(self):
        d = compare_images(_noisy(), _noisy(seed=99))
        assert set(d.keys()) == {
            "mean_diff", "std_ratio", "entropy_diff",
            "sharpness_ratio", "hist_corr", "hist_bhatt",
        }

    def test_same_image_mean_diff_zero(self):
        img = _noisy()
        assert compare_images(img, img)["mean_diff"] == pytest.approx(0.0, abs=1e-3)

    def test_same_image_std_ratio_one(self):
        img = _noisy()
        assert compare_images(img, img)["std_ratio"] == pytest.approx(1.0, abs=1e-3)

    def test_same_image_entropy_diff_zero(self):
        img = _noisy()
        assert compare_images(img, img)["entropy_diff"] == pytest.approx(0.0, abs=1e-3)

    def test_same_image_hist_corr_one(self):
        img = _noisy()
        assert compare_images(img, img)["hist_corr"] == pytest.approx(1.0, abs=1e-3)

    def test_same_image_bhatt_zero(self):
        img = _noisy()
        assert compare_images(img, img)["hist_bhatt"] == pytest.approx(0.0, abs=1e-3)

    def test_different_means(self):
        d = compare_images(_gray(val=10), _gray(val=200))
        assert abs(d["mean_diff"]) > 100

    def test_all_floats(self):
        d = compare_images(_noisy(), _noisy(seed=13))
        for v in d.values():
            assert isinstance(v, float)

    def test_bgr_same_image(self):
        img = _bgr()
        d = compare_images(img, img)
        assert d["hist_corr"] == pytest.approx(1.0, abs=1e-3)

    def test_sharpness_ratio_same(self):
        img = _noisy()
        d = compare_images(img, img)
        assert d["sharpness_ratio"] == pytest.approx(1.0, abs=1e-3)


# ─── batch_stats ─────────────────────────────────────────────────────────────

class TestBatchStats:
    def test_returns_list(self):
        result = batch_stats([_noisy(), _gray(), _bgr()])
        assert isinstance(result, list)
        assert len(result) == 3

    def test_each_is_imagestats(self):
        for s in batch_stats([_noisy(), _noisy()]):
            assert isinstance(s, ImageStats)

    def test_empty_list(self):
        assert batch_stats([]) == []

    def test_hist_bins_forwarded(self):
        results = batch_stats([_noisy()], hist_bins=64)
        assert len(results[0].histogram) == 64

    def test_gradient_flag_forwarded(self):
        results = batch_stats([_noisy()], include_gradient=True)
        assert "grad_mean" in results[0].extra

    def test_n_pixels_correct(self):
        results = batch_stats([_gray(10, 20)])
        assert results[0].n_pixels == 200

    def test_multiple_shapes(self):
        imgs = [_noisy(20, 30), _noisy(40, 50)]
        results = batch_stats(imgs)
        assert results[0].n_pixels == 600
        assert results[1].n_pixels == 2000

    def test_single_image(self):
        results = batch_stats([_noisy()])
        assert len(results) == 1
        assert isinstance(results[0], ImageStats)
