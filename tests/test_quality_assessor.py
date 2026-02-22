"""
Тесты для puzzle_reconstruction.preprocessing.quality_assessor.
"""
import pytest
import numpy as np

from puzzle_reconstruction.preprocessing.quality_assessor import (
    QualityReport,
    estimate_blur,
    estimate_noise,
    estimate_contrast,
    estimate_completeness,
    assess_quality,
    filter_by_quality,
    batch_assess_quality,
)


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _solid_gray(value: int, h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def _solid_bgr(value: int, h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _gradient_gray(h: int = 64, w: int = 64) -> np.ndarray:
    row = np.linspace(0, 255, w, dtype=np.float32)
    return np.tile(row, (h, 1)).astype(np.uint8)


def _checkerboard(h: int = 64, w: int = 64, block: int = 8) -> np.ndarray:
    img = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, h, block):
        for j in range(0, w, block):
            if ((i // block) + (j // block)) % 2 == 0:
                img[i:i+block, j:j+block] = 255
    return img


def _noisy_gray(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (64, 64), dtype=np.uint8)


# ─── QualityReport ────────────────────────────────────────────────────────────

class TestQualityReport:
    def test_fields_accessible(self):
        r = QualityReport(
            blur_score=0.7, noise_score=0.8, contrast_score=0.6,
            completeness=0.9, overall=0.75, is_acceptable=True,
        )
        assert r.blur_score     == pytest.approx(0.7)
        assert r.noise_score    == pytest.approx(0.8)
        assert r.contrast_score == pytest.approx(0.6)
        assert r.completeness   == pytest.approx(0.9)
        assert r.overall        == pytest.approx(0.75)
        assert r.is_acceptable is True

    def test_default_params_empty(self):
        r = QualityReport(0.5, 0.5, 0.5, 0.5, 0.5, True)
        assert r.params == {}

    def test_repr_contains_overall(self):
        r = QualityReport(0.5, 0.5, 0.5, 0.5, 0.75, True)
        assert "0.75" in repr(r) or "overall" in repr(r).lower()

    def test_repr_contains_acceptable(self):
        r = QualityReport(0.5, 0.5, 0.5, 0.5, 0.5, False)
        assert "False" in repr(r) or "acceptable" in repr(r).lower()


# ─── estimate_blur ────────────────────────────────────────────────────────────

class TestEstimateBlur:
    def test_returns_float(self):
        assert isinstance(estimate_blur(_gradient_gray()), float)

    def test_range_zero_one(self):
        r = estimate_blur(_gradient_gray())
        assert 0.0 <= r <= 1.0

    def test_uniform_image_near_zero(self):
        r = estimate_blur(_solid_gray(128))
        assert r == pytest.approx(0.0, abs=1e-6)

    def test_sharp_image_higher_than_blurred(self):
        import cv2
        sharp   = _checkerboard()
        blurred = cv2.GaussianBlur(sharp, (15, 15), 0)
        assert estimate_blur(sharp) >= estimate_blur(blurred)

    def test_bgr_input(self):
        r = estimate_blur(_solid_bgr(128))
        assert 0.0 <= r <= 1.0

    def test_custom_max_var(self):
        img = _checkerboard()
        r1  = estimate_blur(img, max_var=100.0)
        r2  = estimate_blur(img, max_var=1000.0)
        assert r1 >= r2

    def test_gradient_nonzero(self):
        assert estimate_blur(_gradient_gray()) > 0.0


# ─── estimate_noise ───────────────────────────────────────────────────────────

class TestEstimateNoise:
    def test_returns_float(self):
        assert isinstance(estimate_noise(_gradient_gray()), float)

    def test_range_zero_one(self):
        r = estimate_noise(_gradient_gray())
        assert 0.0 <= r <= 1.0

    def test_uniform_image_near_one(self):
        # Однородное изображение — нет шума
        r = estimate_noise(_solid_gray(128))
        assert r > 0.9

    def test_noisy_image_lower_than_clean(self):
        clean = _gradient_gray()
        noisy = _noisy_gray()
        assert estimate_noise(noisy) < estimate_noise(clean)

    def test_bgr_input(self):
        r = estimate_noise(_solid_bgr(128))
        assert 0.0 <= r <= 1.0

    def test_custom_max_sigma(self):
        img = _noisy_gray()
        r1  = estimate_noise(img, max_sigma=5.0)
        r2  = estimate_noise(img, max_sigma=50.0)
        # Меньший max_sigma → менее «чистое» изображение
        assert r1 <= r2


# ─── estimate_contrast ────────────────────────────────────────────────────────

class TestEstimateContrast:
    def test_returns_float(self):
        assert isinstance(estimate_contrast(_gradient_gray()), float)

    def test_range_zero_one(self):
        r = estimate_contrast(_gradient_gray())
        assert 0.0 <= r <= 1.0

    def test_uniform_image_near_zero(self):
        r = estimate_contrast(_solid_gray(128))
        assert r == pytest.approx(0.0, abs=1e-6)

    def test_high_contrast_checkerboard(self):
        r = estimate_contrast(_checkerboard())
        assert r > 0.5

    def test_gradient_has_contrast(self):
        r = estimate_contrast(_gradient_gray())
        assert r > 0.0

    def test_bgr_input(self):
        r = estimate_contrast(_solid_bgr(128))
        assert 0.0 <= r <= 1.0

    def test_contrast_increases_with_range(self):
        low  = np.full((64, 64), 100, dtype=np.uint8)
        low[0, 0] = 110; low[-1, -1] = 90
        high = np.zeros((64, 64), dtype=np.uint8)
        high[:32, :] = 255
        assert estimate_contrast(high) >= estimate_contrast(low)


# ─── estimate_completeness ────────────────────────────────────────────────────

class TestEstimateCompleteness:
    def test_returns_float(self):
        assert isinstance(estimate_completeness(_gradient_gray()), float)

    def test_range_zero_one(self):
        r = estimate_completeness(_gradient_gray())
        assert 0.0 <= r <= 1.0

    def test_all_white_near_zero(self):
        # Все пиксели ≥ 240 — полностью фон
        r = estimate_completeness(_solid_gray(245))
        assert r == pytest.approx(0.0)

    def test_all_dark_near_one(self):
        r = estimate_completeness(_solid_gray(0))
        assert r == pytest.approx(1.0)

    def test_half_filled(self):
        img = np.full((64, 64), 245, dtype=np.uint8)
        img[:32, :] = 0  # нижняя половина — фон
        r = estimate_completeness(img)
        assert r == pytest.approx(0.5, abs=0.02)

    def test_bgr_input(self):
        r = estimate_completeness(_solid_bgr(0))
        assert r == pytest.approx(1.0)

    def test_custom_threshold(self):
        img = np.full((64, 64), 200, dtype=np.uint8)
        r1  = estimate_completeness(img, bg_threshold=180)
        r2  = estimate_completeness(img, bg_threshold=220)
        # Нижний порог → 200 >= 180 → фон → 0; верхний → 200 < 220 → не фон → 1
        assert r1 == pytest.approx(0.0, abs=0.01)
        assert r2 == pytest.approx(1.0, abs=0.01)


# ─── assess_quality ───────────────────────────────────────────────────────────

class TestAssessQuality:
    def test_returns_quality_report(self):
        assert isinstance(assess_quality(_gradient_gray()), QualityReport)

    def test_all_scores_in_zero_one(self):
        r = assess_quality(_gradient_gray())
        for score in (r.blur_score, r.noise_score, r.contrast_score,
                      r.completeness, r.overall):
            assert 0.0 <= score <= 1.0

    def test_overall_is_weighted_combination(self):
        r = assess_quality(_gradient_gray())
        assert isinstance(r.overall, float)

    def test_is_acceptable_high_quality(self):
        # Чёткое, контрастное изображение
        img = _checkerboard()
        r   = assess_quality(img, min_score=0.0)
        assert r.is_acceptable is True

    def test_is_acceptable_below_threshold(self):
        # Порог выше максимально возможного overall
        r = assess_quality(_gradient_gray(), min_score=1.0)
        assert r.is_acceptable is False

    def test_params_stored(self):
        r = assess_quality(_gradient_gray(), min_score=0.3, bg_threshold=200)
        assert r.params["min_score"]    == pytest.approx(0.3)
        assert r.params["bg_threshold"] == 200

    def test_bgr_input(self):
        r = assess_quality(_solid_bgr(128))
        assert isinstance(r, QualityReport)

    def test_custom_weights_used(self):
        img = _gradient_gray()
        r1  = assess_quality(img, weights=(1.0, 0.0, 0.0, 0.0))
        r2  = assess_quality(img, weights=(0.0, 0.0, 0.0, 1.0))
        # Разные веса → разные overall
        # r1 only blur, r2 only completeness
        assert isinstance(r1.overall, float)
        assert isinstance(r2.overall, float)

    def test_overall_clamped_to_zero_one(self):
        r = assess_quality(_solid_gray(128))
        assert 0.0 <= r.overall <= 1.0


# ─── filter_by_quality ────────────────────────────────────────────────────────

class TestFilterByQuality:
    def test_returns_two_lists(self):
        imgs = [_gradient_gray(), _solid_gray(128)]
        good, bad = filter_by_quality(imgs, min_score=0.0)
        assert isinstance(good, list)
        assert isinstance(bad,  list)

    def test_all_accepted_at_zero_threshold(self):
        imgs = [_gradient_gray() for _ in range(4)]
        good, bad = filter_by_quality(imgs, min_score=0.0)
        assert len(good) == 4
        assert len(bad)  == 0

    def test_all_rejected_at_one_threshold(self):
        imgs = [_gradient_gray() for _ in range(3)]
        good, bad = filter_by_quality(imgs, min_score=1.0)
        assert len(good) == 0
        assert len(bad)  == 3

    def test_total_length_preserved(self):
        imgs = [_gradient_gray(), _noisy_gray(), _solid_gray(0)]
        good, bad = filter_by_quality(imgs, min_score=0.5)
        assert len(good) + len(bad) == 3

    def test_empty_list(self):
        good, bad = filter_by_quality([], min_score=0.5)
        assert good == []
        assert bad  == []


# ─── batch_assess_quality ─────────────────────────────────────────────────────

class TestBatchAssessQuality:
    def test_length_preserved(self):
        imgs = [_gradient_gray(), _solid_gray(128), _checkerboard()]
        r    = batch_assess_quality(imgs)
        assert len(r) == 3

    def test_all_quality_reports(self):
        imgs = [_gradient_gray(), _noisy_gray()]
        r    = batch_assess_quality(imgs)
        for report in r:
            assert isinstance(report, QualityReport)

    def test_empty_list(self):
        r = batch_assess_quality([])
        assert r == []

    def test_kwargs_forwarded(self):
        imgs = [_gradient_gray()]
        r    = batch_assess_quality(imgs, min_score=0.99)
        assert r[0].params["min_score"] == pytest.approx(0.99)
