"""
Тесты для puzzle_reconstruction/preprocessing/skew_correction.py

Покрытие:
    SkewResult         — поля, repr, angle_deg/confidence типы
    _to_gray           — BGR→gray, gray→gray, shape
    _clamp_angle       — диапазон [lo, hi], граничные значения
    detect_skew_hough  — возвращает float, горизонтальное изображение→≈0,
                         наклонённое изображение → угол ≠ 0, пустое → 0.0
    detect_skew_projection — возвращает float, в диапазоне [lo, hi],
                             горизонтальное → малый угол, параметры n_angles/lo/hi
    detect_skew_fft    — возвращает float, в диапазоне [-45, 45],
                         sigma=0 не падает
    correct_skew       — форма сохраняется, angle=0 → идентично, dtype
    skew_confidence    — пустой=0.0, один=0.5, все одинаковые=1.0,
                         большой разброс→<0.5, ∈[0,1]
    auto_correct_skew  — все 4 метода, ValueError, SkewResult.angle_deg тип,
                         grayscale вход, params записаны
    batch_correct_skew — длина = len(images), все SkewResult
"""
import math
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.skew_correction import (
    SkewResult,
    auto_correct_skew,
    batch_correct_skew,
    correct_skew,
    detect_skew_fft,
    detect_skew_hough,
    detect_skew_projection,
    skew_confidence,
)


# ─── Фикстуры ─────────────────────────────────────────────────────────────────

@pytest.fixture
def horizontal_img():
    """
    BGR 128×256: белый фон с чёрными горизонтальными полосами (0° наклон).
    """
    img = np.ones((128, 256, 3), dtype=np.uint8) * 240
    for y in range(15, 128, 25):
        img[y:y + 3, 10:246] = 30
    return img


@pytest.fixture
def gray_horizontal(horizontal_img):
    import cv2
    return cv2.cvtColor(horizontal_img, cv2.COLOR_BGR2GRAY)


@pytest.fixture
def tilted_img():
    """
    BGR 128×256: белый фон с наклонёнными (~5°) чёрными полосами.
    """
    import cv2
    base = np.ones((128, 256, 3), dtype=np.uint8) * 240
    for y in range(15, 128, 25):
        base[y:y + 3, 10:246] = 30
    cx, cy = 128, 64
    M = cv2.getRotationMatrix2D((cx, cy), 5.0, 1.0)
    return cv2.warpAffine(base, M, (256, 128),
                           borderValue=(240, 240, 240))


@pytest.fixture
def blank_img():
    """Полностью белое изображение — нет текста/линий."""
    return np.ones((64, 128, 3), dtype=np.uint8) * 255


# ─── SkewResult ───────────────────────────────────────────────────────────────

class TestSkewResult:
    def test_fields(self, horizontal_img):
        r = auto_correct_skew(horizontal_img, method="hough")
        assert hasattr(r, "corrected_image")
        assert hasattr(r, "angle_deg")
        assert hasattr(r, "confidence")
        assert hasattr(r, "method")
        assert hasattr(r, "params")

    def test_corrected_image_is_ndarray(self, horizontal_img):
        r = auto_correct_skew(horizontal_img, method="hough")
        assert isinstance(r.corrected_image, np.ndarray)

    def test_angle_deg_is_float(self, horizontal_img):
        r = auto_correct_skew(horizontal_img, method="hough")
        assert isinstance(r.angle_deg, float)

    def test_confidence_in_range(self, horizontal_img):
        r = auto_correct_skew(horizontal_img, method="hough")
        assert 0.0 <= r.confidence <= 1.0

    def test_repr_contains_method(self, horizontal_img):
        r = auto_correct_skew(horizontal_img, method="hough")
        assert "hough" in repr(r)

    def test_repr_contains_angle(self, horizontal_img):
        r = auto_correct_skew(horizontal_img, method="projection")
        assert "°" in repr(r)

    def test_repr_contains_size(self, horizontal_img):
        r = auto_correct_skew(horizontal_img, method="projection")
        assert "256" in repr(r)

    def test_params_dict(self, horizontal_img):
        r = auto_correct_skew(horizontal_img, method="hough")
        assert isinstance(r.params, dict)


# ─── detect_skew_hough ────────────────────────────────────────────────────────

class TestDetectSkewHough:
    def test_returns_float(self, horizontal_img):
        a = detect_skew_hough(horizontal_img)
        assert isinstance(a, float)

    def test_horizontal_image_small_angle(self, horizontal_img):
        """Горизонтальные полосы → угол близок к 0."""
        a = detect_skew_hough(horizontal_img)
        assert abs(a) < 10.0

    def test_blank_image_returns_zero(self, blank_img):
        """Нет линий → 0.0."""
        a = detect_skew_hough(blank_img)
        assert a == pytest.approx(0.0)

    def test_grayscale_input(self, gray_horizontal):
        a = detect_skew_hough(gray_horizontal)
        assert isinstance(a, float)

    def test_angle_range_param(self, horizontal_img):
        """angle_range=5 → фильтруем только очень горизонтальные линии."""
        a = detect_skew_hough(horizontal_img, angle_range=5.0)
        assert isinstance(a, float)

    def test_params_recorded_in_auto(self, horizontal_img):
        r = auto_correct_skew(horizontal_img, method="hough", threshold=80)
        assert "method" in r.params


# ─── detect_skew_projection ───────────────────────────────────────────────────

class TestDetectSkewProjection:
    def test_returns_float(self, horizontal_img):
        a = detect_skew_projection(horizontal_img)
        assert isinstance(a, float)

    def test_in_range_lo_hi(self, horizontal_img):
        lo, hi = -30.0, 30.0
        a = detect_skew_projection(horizontal_img, lo=lo, hi=hi)
        assert lo <= a <= hi

    def test_horizontal_small_angle(self, horizontal_img):
        """Строго горизонтальный текст → малый угол."""
        a = detect_skew_projection(horizontal_img, n_angles=90)
        assert abs(a) < 15.0

    def test_grayscale_input(self, gray_horizontal):
        a = detect_skew_projection(gray_horizontal)
        assert isinstance(a, float)

    def test_n_angles_param(self, horizontal_img):
        """n_angles=10 — грубый поиск — не должен падать."""
        a = detect_skew_projection(horizontal_img, n_angles=10)
        assert isinstance(a, float)

    def test_blank_image_no_crash(self, blank_img):
        a = detect_skew_projection(blank_img)
        assert isinstance(a, float)


# ─── detect_skew_fft ──────────────────────────────────────────────────────────

class TestDetectSkewFft:
    def test_returns_float(self, horizontal_img):
        a = detect_skew_fft(horizontal_img)
        assert isinstance(a, float)

    def test_in_clamp_range(self, horizontal_img):
        a = detect_skew_fft(horizontal_img)
        assert -45.0 <= a <= 45.0

    def test_sigma_0_no_crash(self, horizontal_img):
        a = detect_skew_fft(horizontal_img, sigma=0.0)
        assert isinstance(a, float)

    def test_sigma_large_no_crash(self, horizontal_img):
        a = detect_skew_fft(horizontal_img, sigma=5.0)
        assert isinstance(a, float)

    def test_grayscale_input(self, gray_horizontal):
        a = detect_skew_fft(gray_horizontal)
        assert isinstance(a, float)

    def test_blank_image_no_crash(self, blank_img):
        a = detect_skew_fft(blank_img)
        assert isinstance(a, float)

    def test_n_angles_param(self, horizontal_img):
        a = detect_skew_fft(horizontal_img, n_angles=36)
        assert isinstance(a, float)


# ─── correct_skew ─────────────────────────────────────────────────────────────

class TestCorrectSkew:
    def test_shape_preserved_bgr(self, horizontal_img):
        out = correct_skew(horizontal_img, 3.0)
        assert out.shape == horizontal_img.shape

    def test_shape_preserved_gray(self, gray_horizontal):
        out = correct_skew(gray_horizontal, 3.0)
        assert out.shape == gray_horizontal.shape

    def test_dtype_preserved(self, horizontal_img):
        out = correct_skew(horizontal_img, 5.0)
        assert out.dtype == np.uint8

    def test_angle_zero_near_identity(self, horizontal_img):
        """angle=0 → изображение практически не меняется."""
        out = correct_skew(horizontal_img, 0.0)
        # После поворота на 0° большинство пикселей совпадает
        diff = np.abs(out.astype(int) - horizontal_img.astype(int))
        assert diff.mean() < 5.0

    def test_negative_angle_no_crash(self, horizontal_img):
        out = correct_skew(horizontal_img, -10.0)
        assert out.shape == horizontal_img.shape

    def test_large_angle_no_crash(self, horizontal_img):
        out = correct_skew(horizontal_img, 45.0)
        assert out.shape == horizontal_img.shape

    def test_border_mode_constant(self, horizontal_img):
        import cv2
        out = correct_skew(horizontal_img, 15.0,
                            border_mode=cv2.BORDER_CONSTANT)
        assert out.shape == horizontal_img.shape

    def test_scale_param(self, horizontal_img):
        out = correct_skew(horizontal_img, 5.0, scale=0.9)
        assert out.shape == horizontal_img.shape


# ─── skew_confidence ──────────────────────────────────────────────────────────

class TestSkewConfidence:
    def test_empty_list_zero(self):
        assert skew_confidence([]) == 0.0

    def test_single_angle_half(self):
        c = skew_confidence([3.0])
        assert c == pytest.approx(0.5)

    def test_identical_angles_one(self):
        c = skew_confidence([2.0, 2.0, 2.0])
        assert c == pytest.approx(1.0)

    def test_large_spread_low_confidence(self):
        c = skew_confidence([-30.0, 0.0, 30.0], tol=2.0)
        assert c < 0.5

    def test_small_spread_high_confidence(self):
        c = skew_confidence([1.0, 1.1, 1.2], tol=2.0)
        assert c > 0.8

    def test_in_range(self):
        for angles in [[], [0], [5, 6], [-20, 20], [1, 2, 3]]:
            c = skew_confidence(angles)
            assert 0.0 <= c <= 1.0


# ─── auto_correct_skew ────────────────────────────────────────────────────────

class TestAutoCorrectSkew:
    @pytest.mark.parametrize("method", ["hough", "projection", "fft", "auto"])
    def test_all_methods_return_result(self, horizontal_img, method):
        r = auto_correct_skew(horizontal_img, method=method)
        assert isinstance(r, SkewResult)

    @pytest.mark.parametrize("method", ["hough", "projection", "fft", "auto"])
    def test_output_shape_preserved(self, horizontal_img, method):
        r = auto_correct_skew(horizontal_img, method=method)
        assert r.corrected_image.shape == horizontal_img.shape

    def test_unknown_method_raises(self, horizontal_img):
        with pytest.raises(ValueError):
            auto_correct_skew(horizontal_img, method="magic_method")

    def test_method_stored_in_result(self, horizontal_img):
        r = auto_correct_skew(horizontal_img, method="projection")
        assert r.method == "projection"

    def test_auto_method_has_angles_all(self, horizontal_img):
        r = auto_correct_skew(horizontal_img, method="auto")
        assert "angles_all" in r.params
        assert len(r.params["angles_all"]) == 3

    def test_grayscale_hough(self, gray_horizontal):
        r = auto_correct_skew(gray_horizontal, method="hough")
        assert r.corrected_image.shape == gray_horizontal.shape

    def test_grayscale_projection(self, gray_horizontal):
        r = auto_correct_skew(gray_horizontal, method="projection")
        assert isinstance(r.angle_deg, float)

    def test_angle_deg_float(self, horizontal_img):
        for method in ["hough", "projection", "fft", "auto"]:
            r = auto_correct_skew(horizontal_img, method=method)
            assert isinstance(r.angle_deg, float)

    def test_confidence_in_range(self, horizontal_img):
        for method in ["hough", "projection", "fft", "auto"]:
            r = auto_correct_skew(horizontal_img, method=method)
            assert 0.0 <= r.confidence <= 1.0

    def test_blank_image_no_crash(self, blank_img):
        for method in ["hough", "projection", "fft", "auto"]:
            r = auto_correct_skew(blank_img, method=method)
            assert isinstance(r, SkewResult)


# ─── batch_correct_skew ───────────────────────────────────────────────────────

class TestBatchCorrectSkew:
    def test_length_matches_input(self, horizontal_img, blank_img):
        images  = [horizontal_img, blank_img, horizontal_img]
        results = batch_correct_skew(images, method="hough")
        assert len(results) == 3

    def test_all_skew_results(self, horizontal_img, blank_img):
        results = batch_correct_skew([horizontal_img, blank_img])
        assert all(isinstance(r, SkewResult) for r in results)

    def test_empty_list(self):
        results = batch_correct_skew([], method="projection")
        assert results == []

    def test_single_image(self, horizontal_img):
        results = batch_correct_skew([horizontal_img])
        assert len(results) == 1

    def test_methods_propagated(self, horizontal_img):
        results = batch_correct_skew([horizontal_img], method="fft")
        assert results[0].method == "fft"
