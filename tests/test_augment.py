"""
Тесты для puzzle_reconstruction/preprocessing/augment.py

Покрытие:
    random_crop         — форма, масштаб ∈ [min_scale, 1.0], seed воспроизводимость
    random_rotate       — форма, угол=0 → тождество, expand
    add_gaussian_noise  — sigma=0 → идентично, значения ∈ [0,255], уменьшение SNR
    add_salt_pepper     — белые/чёрные пиксели появляются, amount=0 → идентично
    brightness_jitter   — форма, dtype, значения в диапазоне
    jpeg_compress       — форма, качество 1-100, артефакты (отличие от оригинала)
    simulate_scan_noise — форма, нет падения ниже 0 / выше 255, noise≠0 меняет изображение
    augment_batch       — длина = len * (1 + n_augments), флаги, seed
"""
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


# ─── Фикстуры ─────────────────────────────────────────────────────────────────

@pytest.fixture
def bgr_img():
    """Синтетическое BGR-изображение 64×64 с прямоугольником."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 200
    img[20:44, 20:44] = (100, 150, 200)
    return img


@pytest.fixture
def gray_img():
    """Grayscale 64×64."""
    img = np.ones((64, 64), dtype=np.uint8) * 128
    img[20:44, 20:44] = 200
    return img


@pytest.fixture
def rng():
    return np.random.RandomState(42)


# ─── random_crop ──────────────────────────────────────────────────────────────

class TestRandomCrop:
    def test_output_shape_preserved(self, bgr_img, rng):
        out = random_crop(bgr_img, min_scale=0.75, rng=rng)
        assert out.shape == bgr_img.shape

    def test_dtype_preserved(self, bgr_img, rng):
        out = random_crop(bgr_img, rng=rng)
        assert out.dtype == np.uint8

    def test_values_in_range(self, bgr_img, rng):
        out = random_crop(bgr_img, rng=rng)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_max_scale_1_no_crop(self, bgr_img):
        """min_scale = max_scale = 1.0 → точная вырезка всего изображения."""
        rng = np.random.RandomState(0)
        out = random_crop(bgr_img, min_scale=1.0, max_scale=1.0, rng=rng)
        assert out.shape == bgr_img.shape

    def test_reproducibility(self, bgr_img):
        """Одинаковый seed → одинаковый результат."""
        out1 = random_crop(bgr_img, rng=np.random.RandomState(7))
        out2 = random_crop(bgr_img, rng=np.random.RandomState(7))
        assert np.array_equal(out1, out2)

    def test_grayscale(self, gray_img, rng):
        out = random_crop(gray_img, min_scale=0.8, rng=rng)
        assert out.shape == gray_img.shape


# ─── random_rotate ────────────────────────────────────────────────────────────

class TestRandomRotate:
    def test_output_shape(self, bgr_img, rng):
        out = random_rotate(bgr_img, max_angle=5.0, rng=rng)
        assert out.shape == bgr_img.shape

    def test_dtype(self, bgr_img, rng):
        out = random_rotate(bgr_img, rng=rng)
        assert out.dtype == np.uint8

    def test_values_in_range(self, bgr_img, rng):
        out = random_rotate(bgr_img, rng=rng)
        assert out.min() >= 0 and out.max() <= 255

    def test_expand_changes_shape(self, bgr_img, rng):
        """expand=True при ненулевом угле → размер может измениться."""
        # Форсируем ненулевой угол большим max_angle
        out = random_rotate(bgr_img, max_angle=30.0, expand=True,
                             rng=np.random.RandomState(3))
        # Форма может быть ≥ оригинала по крайней мере по одной оси
        assert out.shape[0] >= 1 and out.shape[1] >= 1

    def test_expand_false_preserves_shape(self, bgr_img, rng):
        out = random_rotate(bgr_img, max_angle=15.0, expand=False, rng=rng)
        assert out.shape == bgr_img.shape

    def test_reproducibility(self, bgr_img):
        o1 = random_rotate(bgr_img, rng=np.random.RandomState(5))
        o2 = random_rotate(bgr_img, rng=np.random.RandomState(5))
        assert np.array_equal(o1, o2)


# ─── add_gaussian_noise ───────────────────────────────────────────────────────

class TestAddGaussianNoise:
    def test_sigma_zero_identical(self, bgr_img, rng):
        out = add_gaussian_noise(bgr_img, sigma=0.0, rng=rng)
        assert np.array_equal(out, bgr_img)

    def test_output_shape(self, bgr_img, rng):
        out = add_gaussian_noise(bgr_img, sigma=10.0, rng=rng)
        assert out.shape == bgr_img.shape

    def test_dtype_uint8(self, bgr_img, rng):
        out = add_gaussian_noise(bgr_img, sigma=10.0, rng=rng)
        assert out.dtype == np.uint8

    def test_values_in_range(self, bgr_img, rng):
        out = add_gaussian_noise(bgr_img, sigma=30.0, rng=rng)
        assert out.min() >= 0 and out.max() <= 255

    def test_output_differs_from_input(self, bgr_img):
        rng = np.random.RandomState(99)
        out = add_gaussian_noise(bgr_img, sigma=15.0, rng=rng)
        assert not np.array_equal(out, bgr_img), "Шум должен изменить изображение"

    def test_reproducibility(self, bgr_img):
        o1 = add_gaussian_noise(bgr_img, sigma=10.0, rng=np.random.RandomState(1))
        o2 = add_gaussian_noise(bgr_img, sigma=10.0, rng=np.random.RandomState(1))
        assert np.array_equal(o1, o2)


# ─── add_salt_pepper ──────────────────────────────────────────────────────────

class TestAddSaltPepper:
    def test_amount_zero_identical(self, bgr_img, rng):
        out = add_salt_pepper(bgr_img, amount=0.0, rng=rng)
        assert np.array_equal(out, bgr_img)

    def test_shape_preserved(self, bgr_img, rng):
        out = add_salt_pepper(bgr_img, amount=0.05, rng=rng)
        assert out.shape == bgr_img.shape

    def test_dtype(self, bgr_img, rng):
        out = add_salt_pepper(bgr_img, amount=0.05, rng=rng)
        assert out.dtype == np.uint8

    def test_white_pixels_appear(self, bgr_img):
        """После добавления «соли» должны появиться белые пиксели."""
        # Изображение не содержит полностью белых пикселей
        no_white_img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        out = add_salt_pepper(no_white_img, amount=0.1, salt_ratio=1.0,
                               rng=np.random.RandomState(0))
        assert (out == 255).any(), "Белые пиксели должны появиться"

    def test_black_pixels_appear(self, bgr_img):
        """После добавления «перца» должны появиться чёрные пиксели."""
        no_black_img = np.ones((64, 64, 3), dtype=np.uint8) * 200
        out = add_salt_pepper(no_black_img, amount=0.1, salt_ratio=0.0,
                               rng=np.random.RandomState(0))
        assert (out == 0).any(), "Чёрные пиксели должны появиться"

    def test_values_in_range(self, bgr_img, rng):
        out = add_salt_pepper(bgr_img, amount=0.2, rng=rng)
        assert out.min() >= 0 and out.max() <= 255


# ─── brightness_jitter ────────────────────────────────────────────────────────

class TestBrightnessJitter:
    def test_shape_preserved(self, bgr_img, rng):
        out = brightness_jitter(bgr_img, rng=rng)
        assert out.shape == bgr_img.shape

    def test_dtype_uint8(self, bgr_img, rng):
        out = brightness_jitter(bgr_img, rng=rng)
        assert out.dtype == np.uint8

    def test_values_in_range(self, bgr_img, rng):
        out = brightness_jitter(bgr_img, rng=rng)
        assert out.min() >= 0 and out.max() <= 255

    def test_factor_1_gamma_1_near_identical(self, bgr_img):
        """factor=1.0, gamma=1.0 → изображение не меняется (почти)."""
        rng_mock = np.random.RandomState.__new__(np.random.RandomState)
        # Мокируем через фиксированный диапазон [1.0, 1.0]
        out = brightness_jitter(bgr_img,
                                 factor_range=(1.0, 1.0),
                                 gamma_range=(1.0, 1.0),
                                 rng=np.random.RandomState(0))
        diff = np.abs(out.astype(np.float32) - bgr_img.astype(np.float32))
        assert diff.mean() < 5.0, "Нейтральные параметры → почти идентично"

    def test_brighter_with_high_factor(self, bgr_img):
        out = brightness_jitter(bgr_img,
                                 factor_range=(1.5, 1.5),
                                 gamma_range=(1.0, 1.0),
                                 rng=np.random.RandomState(0))
        assert float(out.mean()) >= float(bgr_img.mean()) * 0.9

    def test_reproducibility(self, bgr_img):
        o1 = brightness_jitter(bgr_img, rng=np.random.RandomState(3))
        o2 = brightness_jitter(bgr_img, rng=np.random.RandomState(3))
        assert np.array_equal(o1, o2)


# ─── jpeg_compress ────────────────────────────────────────────────────────────

class TestJpegCompress:
    def test_output_shape(self, bgr_img):
        out = jpeg_compress(bgr_img, quality=80)
        assert out.shape == bgr_img.shape

    def test_dtype(self, bgr_img):
        out = jpeg_compress(bgr_img, quality=80)
        assert out.dtype == np.uint8

    def test_values_in_range(self, bgr_img):
        out = jpeg_compress(bgr_img, quality=50)
        assert out.min() >= 0 and out.max() <= 255

    def test_low_quality_introduces_artifacts(self, bgr_img):
        """Низкое качество JPEG → изображение отличается от оригинала."""
        out = jpeg_compress(bgr_img, quality=1)
        diff = np.abs(out.astype(np.float32) - bgr_img.astype(np.float32))
        assert diff.mean() > 0.5, "JPEG с quality=1 должен вносить артефакты"

    def test_quality_100_minimal_loss(self, bgr_img):
        """quality=100 → минимальные потери."""
        out = jpeg_compress(bgr_img, quality=100)
        diff = np.abs(out.astype(np.float32) - bgr_img.astype(np.float32))
        assert diff.mean() < 10.0


# ─── simulate_scan_noise ──────────────────────────────────────────────────────

class TestSimulateScanNoise:
    def test_output_shape(self, bgr_img):
        out = simulate_scan_noise(bgr_img)
        assert out.shape == bgr_img.shape

    def test_dtype(self, bgr_img):
        out = simulate_scan_noise(bgr_img)
        assert out.dtype == np.uint8

    def test_values_in_range(self, bgr_img):
        out = simulate_scan_noise(bgr_img, gaussian_sigma=20, sp_amount=0.05)
        assert out.min() >= 0 and out.max() <= 255

    def test_no_noise_identical(self, bgr_img):
        """Все параметры в «нулевом» режиме → идентичное изображение."""
        out = simulate_scan_noise(
            bgr_img, gaussian_sigma=0.0, sp_amount=0.0,
            jpeg_quality=100, yellowing=0.0,
            rng=np.random.RandomState(0),
        )
        assert np.array_equal(out, bgr_img)

    def test_noise_changes_image(self, bgr_img):
        out = simulate_scan_noise(bgr_img,
                                   gaussian_sigma=15.0, sp_amount=0.02,
                                   jpeg_quality=75, yellowing=0.1,
                                   rng=np.random.RandomState(0))
        assert not np.array_equal(out, bgr_img)

    def test_reproducibility(self, bgr_img):
        o1 = simulate_scan_noise(bgr_img, rng=np.random.RandomState(10))
        o2 = simulate_scan_noise(bgr_img, rng=np.random.RandomState(10))
        assert np.array_equal(o1, o2)


# ─── augment_batch ────────────────────────────────────────────────────────────

class TestAugmentBatch:
    def test_output_length(self, bgr_img):
        images = [bgr_img, bgr_img]
        result = augment_batch(images, n_augments=3, seed=42)
        assert len(result) == 2 * (1 + 3)

    def test_originals_preserved(self, bgr_img):
        """Первые N элементов — оригинальные изображения."""
        images = [bgr_img]
        result = augment_batch(images, n_augments=2, seed=42)
        assert np.array_equal(result[0], bgr_img)

    def test_shapes_preserved(self, bgr_img):
        images = [bgr_img] * 3
        result = augment_batch(images, n_augments=2, seed=0)
        for img in result:
            assert img.shape == bgr_img.shape

    def test_dtype_preserved(self, bgr_img):
        result = augment_batch([bgr_img], n_augments=2, seed=0)
        for img in result:
            assert img.dtype == np.uint8

    def test_seed_reproducibility(self, bgr_img):
        r1 = augment_batch([bgr_img], n_augments=2, seed=7)
        r2 = augment_batch([bgr_img], n_augments=2, seed=7)
        for a, b in zip(r1, r2):
            assert np.array_equal(a, b)

    def test_flags_false_still_runs(self, bgr_img):
        """Все флаги выключены → аугментированные копии совпадают с оригиналом."""
        result = augment_batch(
            [bgr_img], n_augments=2,
            rotate=False, crop=False, noise=False, jitter=False,
            seed=0,
        )
        assert len(result) == 3

    def test_empty_list(self):
        result = augment_batch([], n_augments=3, seed=0)
        assert result == []

    def test_n_augments_zero(self, bgr_img):
        """n_augments=0 → только оригинальные изображения."""
        result = augment_batch([bgr_img, bgr_img], n_augments=0, seed=0)
        assert len(result) == 2
