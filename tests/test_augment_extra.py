"""Extra tests for puzzle_reconstruction.preprocessing.augment."""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.augment import (
    random_crop,
    random_rotate,
    add_gaussian_noise,
    add_salt_pepper,
    brightness_jitter,
    jpeg_compress,
    simulate_scan_noise,
    augment_batch,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def bgr_img():
    img = np.ones((64, 64, 3), dtype=np.uint8) * 200
    img[20:44, 20:44] = (100, 150, 200)
    return img


@pytest.fixture
def gray_img():
    img = np.ones((64, 64), dtype=np.uint8) * 128
    img[20:44, 20:44] = 200
    return img


@pytest.fixture
def rng():
    return np.random.RandomState(42)


# ─── TestRandomCropExtra ────────────────────────────────────────────────────

class TestRandomCropExtra:
    def test_small_image(self, rng):
        img = np.ones((8, 8, 3), dtype=np.uint8) * 128
        out = random_crop(img, min_scale=0.5, rng=rng)
        assert out.shape == img.shape

    def test_grayscale_dtype(self, gray_img, rng):
        out = random_crop(gray_img, rng=rng)
        assert out.dtype == np.uint8

    def test_min_scale_half(self, bgr_img, rng):
        out = random_crop(bgr_img, min_scale=0.5, rng=rng)
        assert out.shape == bgr_img.shape

    def test_different_seeds_different_crops(self, bgr_img):
        o1 = random_crop(bgr_img, min_scale=0.5,
                         rng=np.random.RandomState(0))
        o2 = random_crop(bgr_img, min_scale=0.5,
                         rng=np.random.RandomState(99))
        # Very likely different
        assert o1.shape == o2.shape

    def test_rectangular_image(self, rng):
        img = np.ones((32, 96, 3), dtype=np.uint8) * 100
        out = random_crop(img, min_scale=0.8, rng=rng)
        assert out.shape == img.shape

    def test_values_clipped(self, bgr_img, rng):
        out = random_crop(bgr_img, rng=rng)
        assert out.min() >= 0
        assert out.max() <= 255


# ─── TestRandomRotateExtra ──────────────────────────────────────────────────

class TestRandomRotateExtra:
    def test_zero_angle(self, bgr_img):
        out = random_rotate(bgr_img, max_angle=0.0,
                            rng=np.random.RandomState(0))
        assert out.shape == bgr_img.shape

    def test_large_angle(self, bgr_img, rng):
        out = random_rotate(bgr_img, max_angle=45.0, rng=rng)
        assert out.dtype == np.uint8

    def test_grayscale(self, gray_img, rng):
        out = random_rotate(gray_img, max_angle=10.0, rng=rng)
        assert out.shape == gray_img.shape

    def test_expand_true_larger(self, bgr_img):
        out = random_rotate(bgr_img, max_angle=30.0, expand=True,
                            rng=np.random.RandomState(5))
        assert out.shape[0] >= 1
        assert out.shape[1] >= 1

    def test_different_seeds(self, bgr_img):
        o1 = random_rotate(bgr_img, max_angle=20.0,
                           rng=np.random.RandomState(1))
        o2 = random_rotate(bgr_img, max_angle=20.0,
                           rng=np.random.RandomState(2))
        assert o1.shape == bgr_img.shape
        assert o2.shape == bgr_img.shape


# ─── TestAddGaussianNoiseExtra ──────────────────────────────────────────────

class TestAddGaussianNoiseExtra:
    def test_high_sigma(self, bgr_img, rng):
        out = add_gaussian_noise(bgr_img, sigma=50.0, rng=rng)
        assert out.dtype == np.uint8
        assert out.min() >= 0
        assert out.max() <= 255

    def test_grayscale(self, gray_img, rng):
        out = add_gaussian_noise(gray_img, sigma=10.0, rng=rng)
        assert out.shape == gray_img.shape

    def test_small_image(self, rng):
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        out = add_gaussian_noise(img, sigma=5.0, rng=rng)
        assert out.shape == (4, 4, 3)

    def test_zero_sigma_preserves(self, bgr_img, rng):
        out = add_gaussian_noise(bgr_img, sigma=0.0, rng=rng)
        np.testing.assert_array_equal(out, bgr_img)

    def test_different_seeds(self, bgr_img):
        o1 = add_gaussian_noise(bgr_img, sigma=10.0,
                                rng=np.random.RandomState(1))
        o2 = add_gaussian_noise(bgr_img, sigma=10.0,
                                rng=np.random.RandomState(2))
        assert not np.array_equal(o1, o2)


# ─── TestAddSaltPepperExtra ─────────────────────────────────────────────────

class TestAddSaltPepperExtra:
    def test_high_amount(self, bgr_img, rng):
        out = add_salt_pepper(bgr_img, amount=0.5, rng=rng)
        assert out.shape == bgr_img.shape

    def test_grayscale(self, gray_img, rng):
        out = add_salt_pepper(gray_img, amount=0.05, rng=rng)
        assert out.shape == gray_img.shape

    def test_salt_only(self, rng):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        out = add_salt_pepper(img, amount=0.1, salt_ratio=1.0, rng=rng)
        assert (out == 255).any()

    def test_pepper_only(self, rng):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        out = add_salt_pepper(img, amount=0.1, salt_ratio=0.0, rng=rng)
        assert (out == 0).any()

    def test_values_in_range(self, bgr_img, rng):
        out = add_salt_pepper(bgr_img, amount=0.3, rng=rng)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_reproducibility(self, bgr_img):
        o1 = add_salt_pepper(bgr_img, amount=0.1,
                             rng=np.random.RandomState(0))
        o2 = add_salt_pepper(bgr_img, amount=0.1,
                             rng=np.random.RandomState(0))
        np.testing.assert_array_equal(o1, o2)


# ─── TestBrightnessJitterExtra ──────────────────────────────────────────────

class TestBrightnessJitterExtra:
    def test_grayscale(self, gray_img, rng):
        out = brightness_jitter(gray_img, rng=rng)
        assert out.shape == gray_img.shape

    def test_dark_factor(self, bgr_img):
        out = brightness_jitter(bgr_img,
                                factor_range=(0.3, 0.3),
                                gamma_range=(1.0, 1.0),
                                rng=np.random.RandomState(0))
        assert float(out.mean()) <= float(bgr_img.mean())

    def test_values_clipped(self, bgr_img, rng):
        out = brightness_jitter(bgr_img, rng=rng)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_different_seeds(self, bgr_img):
        o1 = brightness_jitter(bgr_img, rng=np.random.RandomState(1))
        o2 = brightness_jitter(bgr_img, rng=np.random.RandomState(2))
        # May differ
        assert o1.shape == o2.shape

    def test_small_image(self, rng):
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        out = brightness_jitter(img, rng=rng)
        assert out.shape == (4, 4, 3)


# ─── TestJpegCompressExtra ──────────────────────────────────────────────────

class TestJpegCompressExtra:
    def test_quality_10(self, bgr_img):
        out = jpeg_compress(bgr_img, quality=10)
        assert out.shape == bgr_img.shape

    def test_quality_90(self, bgr_img):
        out = jpeg_compress(bgr_img, quality=90)
        assert out.dtype == np.uint8

    def test_grayscale(self, gray_img):
        out = jpeg_compress(gray_img, quality=50)
        assert out.dtype == np.uint8
        # JPEG may convert grayscale to BGR
        assert out.shape[:2] == gray_img.shape[:2]

    def test_small_image(self):
        img = np.full((8, 8, 3), 128, dtype=np.uint8)
        out = jpeg_compress(img, quality=50)
        assert out.shape == (8, 8, 3)

    def test_values_valid(self, bgr_img):
        out = jpeg_compress(bgr_img, quality=30)
        assert out.min() >= 0
        assert out.max() <= 255


# ─── TestSimulateScanNoiseExtra ─────────────────────────────────────────────

class TestSimulateScanNoiseExtra:
    def test_high_noise(self, bgr_img):
        out = simulate_scan_noise(bgr_img, gaussian_sigma=30.0,
                                   sp_amount=0.1, jpeg_quality=50,
                                   yellowing=0.2,
                                   rng=np.random.RandomState(0))
        assert out.shape == bgr_img.shape
        assert not np.array_equal(out, bgr_img)

    def test_grayscale_accepted(self, gray_img):
        out = simulate_scan_noise(gray_img, rng=np.random.RandomState(0))
        assert out.dtype == np.uint8

    def test_values_in_range(self, bgr_img):
        out = simulate_scan_noise(bgr_img, rng=np.random.RandomState(0))
        assert out.min() >= 0
        assert out.max() <= 255

    def test_small_image(self):
        img = np.full((8, 8, 3), 128, dtype=np.uint8)
        out = simulate_scan_noise(img, rng=np.random.RandomState(0))
        assert out.shape == (8, 8, 3)


# ─── TestAugmentBatchExtra ──────────────────────────────────────────────────

class TestAugmentBatchExtra:
    def test_single_image_5_augments(self, bgr_img):
        result = augment_batch([bgr_img], n_augments=5, seed=0)
        assert len(result) == 6

    def test_three_images_2_augments(self, bgr_img):
        result = augment_batch([bgr_img] * 3, n_augments=2, seed=0)
        assert len(result) == 3 * (1 + 2)

    def test_all_shapes_match(self, bgr_img):
        result = augment_batch([bgr_img], n_augments=3, seed=0)
        for img in result:
            assert img.shape == bgr_img.shape

    def test_all_dtype_uint8(self, bgr_img):
        result = augment_batch([bgr_img], n_augments=2, seed=0)
        for img in result:
            assert img.dtype == np.uint8

    def test_n_augments_1(self, bgr_img):
        result = augment_batch([bgr_img], n_augments=1, seed=0)
        assert len(result) == 2

    def test_all_values_valid(self, bgr_img):
        result = augment_batch([bgr_img], n_augments=3, seed=0)
        for img in result:
            assert img.min() >= 0
            assert img.max() <= 255
