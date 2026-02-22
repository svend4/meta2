"""Extra tests for puzzle_reconstruction.preprocessing.quality_assessor."""
import numpy as np
import pytest
import cv2

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _solid(val, h=64, w=64):
    return np.full((h, w), val, dtype=np.uint8)


def _solid_bgr(val, h=64, w=64):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _gradient(h=64, w=64):
    row = np.linspace(0, 255, w, dtype=np.float32)
    return np.tile(row, (h, 1)).astype(np.uint8)


def _checker(h=64, w=64, block=8):
    img = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, h, block):
        for j in range(0, w, block):
            if ((i // block) + (j // block)) % 2 == 0:
                img[i:i+block, j:j+block] = 255
    return img


def _noisy(seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (64, 64), dtype=np.uint8)


# ─── TestQualityReportExtra ───────────────────────────────────────────────────

class TestQualityReportExtra:
    def test_blur_score_stored(self):
        r = QualityReport(0.3, 0.5, 0.7, 0.9, 0.6, True)
        assert r.blur_score == pytest.approx(0.3)

    def test_noise_score_stored(self):
        r = QualityReport(0.3, 0.5, 0.7, 0.9, 0.6, True)
        assert r.noise_score == pytest.approx(0.5)

    def test_contrast_score_stored(self):
        r = QualityReport(0.3, 0.5, 0.7, 0.9, 0.6, True)
        assert r.contrast_score == pytest.approx(0.7)

    def test_completeness_stored(self):
        r = QualityReport(0.3, 0.5, 0.7, 0.9, 0.6, True)
        assert r.completeness == pytest.approx(0.9)

    def test_overall_stored(self):
        r = QualityReport(0.3, 0.5, 0.7, 0.9, 0.6, True)
        assert r.overall == pytest.approx(0.6)

    def test_is_acceptable_true(self):
        r = QualityReport(0.5, 0.5, 0.5, 0.5, 0.5, True)
        assert r.is_acceptable is True

    def test_is_acceptable_false(self):
        r = QualityReport(0.5, 0.5, 0.5, 0.5, 0.5, False)
        assert r.is_acceptable is False

    def test_params_default_empty(self):
        r = QualityReport(0.5, 0.5, 0.5, 0.5, 0.5, True)
        assert r.params == {}

    def test_repr_is_string(self):
        r = QualityReport(0.5, 0.5, 0.5, 0.5, 0.75, True)
        assert isinstance(repr(r), str)


# ─── TestEstimateBlurExtra ────────────────────────────────────────────────────

class TestEstimateBlurExtra:
    def test_returns_float(self):
        assert isinstance(estimate_blur(_gradient()), float)

    def test_range_0_1(self):
        assert 0.0 <= estimate_blur(_gradient()) <= 1.0

    def test_uniform_zero(self):
        assert estimate_blur(_solid(128)) == pytest.approx(0.0, abs=1e-6)

    def test_gradient_positive(self):
        assert estimate_blur(_gradient()) > 0.0

    def test_checker_positive(self):
        assert estimate_blur(_checker()) > 0.0

    def test_sharp_gt_blurred(self):
        sharp = _checker()
        blurred = cv2.GaussianBlur(sharp, (15, 15), 0)
        assert estimate_blur(sharp) >= estimate_blur(blurred)

    def test_bgr_input(self):
        assert 0.0 <= estimate_blur(_solid_bgr(128)) <= 1.0

    def test_noisy_positive(self):
        assert estimate_blur(_noisy()) > 0.0

    def test_custom_max_var_high(self):
        img = _checker()
        r = estimate_blur(img, max_var=10000.0)
        assert 0.0 <= r <= 1.0

    def test_custom_max_var_low(self):
        img = _checker()
        r = estimate_blur(img, max_var=1.0)
        assert 0.0 <= r <= 1.0


# ─── TestEstimateNoiseExtra ───────────────────────────────────────────────────

class TestEstimateNoiseExtra:
    def test_returns_float(self):
        assert isinstance(estimate_noise(_gradient()), float)

    def test_range_0_1(self):
        assert 0.0 <= estimate_noise(_gradient()) <= 1.0

    def test_uniform_near_one(self):
        assert estimate_noise(_solid(128)) > 0.9

    def test_noisy_lower_than_clean(self):
        assert estimate_noise(_noisy()) < estimate_noise(_gradient())

    def test_bgr_input(self):
        assert 0.0 <= estimate_noise(_solid_bgr(128)) <= 1.0

    def test_custom_max_sigma_5(self):
        r = estimate_noise(_noisy(), max_sigma=5.0)
        assert 0.0 <= r <= 1.0

    def test_custom_max_sigma_50(self):
        r = estimate_noise(_noisy(), max_sigma=50.0)
        assert 0.0 <= r <= 1.0

    def test_checker_score(self):
        r = estimate_noise(_checker())
        assert 0.0 <= r <= 1.0


# ─── TestEstimateContrastExtra ────────────────────────────────────────────────

class TestEstimateContrastExtra:
    def test_returns_float(self):
        assert isinstance(estimate_contrast(_gradient()), float)

    def test_range_0_1(self):
        assert 0.0 <= estimate_contrast(_gradient()) <= 1.0

    def test_uniform_zero(self):
        assert estimate_contrast(_solid(128)) == pytest.approx(0.0, abs=1e-6)

    def test_checker_high(self):
        assert estimate_contrast(_checker()) > 0.5

    def test_gradient_positive(self):
        assert estimate_contrast(_gradient()) > 0.0

    def test_bgr_input(self):
        assert 0.0 <= estimate_contrast(_solid_bgr(128)) <= 1.0

    def test_high_range_beats_low(self):
        low = np.full((64, 64), 100, dtype=np.uint8)
        low[0, 0] = 110
        high = np.zeros((64, 64), dtype=np.uint8)
        high[:32, :] = 255
        assert estimate_contrast(high) >= estimate_contrast(low)


# ─── TestEstimateCompletenessExtra ────────────────────────────────────────────

class TestEstimateCompletenessExtra:
    def test_returns_float(self):
        assert isinstance(estimate_completeness(_gradient()), float)

    def test_range_0_1(self):
        assert 0.0 <= estimate_completeness(_gradient()) <= 1.0

    def test_all_white_zero(self):
        assert estimate_completeness(_solid(245)) == pytest.approx(0.0)

    def test_all_dark_one(self):
        assert estimate_completeness(_solid(0)) == pytest.approx(1.0)

    def test_half_filled(self):
        img = np.full((64, 64), 245, dtype=np.uint8)
        img[:32, :] = 0
        assert estimate_completeness(img) == pytest.approx(0.5, abs=0.02)

    def test_bgr_input(self):
        assert estimate_completeness(_solid_bgr(0)) == pytest.approx(1.0)

    def test_custom_threshold_low(self):
        img = np.full((64, 64), 200, dtype=np.uint8)
        r = estimate_completeness(img, bg_threshold=180)
        assert r == pytest.approx(0.0, abs=0.01)

    def test_custom_threshold_high(self):
        img = np.full((64, 64), 200, dtype=np.uint8)
        r = estimate_completeness(img, bg_threshold=220)
        assert r == pytest.approx(1.0, abs=0.01)


# ─── TestAssessQualityExtra ───────────────────────────────────────────────────

class TestAssessQualityExtra:
    def test_returns_quality_report(self):
        assert isinstance(assess_quality(_gradient()), QualityReport)

    def test_all_scores_in_range(self):
        r = assess_quality(_gradient())
        for s in (r.blur_score, r.noise_score, r.contrast_score,
                  r.completeness, r.overall):
            assert 0.0 <= s <= 1.0

    def test_acceptable_at_zero_threshold(self):
        r = assess_quality(_checker(), min_score=0.0)
        assert r.is_acceptable is True

    def test_not_acceptable_at_one_threshold(self):
        r = assess_quality(_gradient(), min_score=1.0)
        assert r.is_acceptable is False

    def test_params_stored(self):
        r = assess_quality(_gradient(), min_score=0.4, bg_threshold=210)
        assert r.params["min_score"] == pytest.approx(0.4)
        assert r.params["bg_threshold"] == 210

    def test_bgr_input(self):
        r = assess_quality(_solid_bgr(128))
        assert isinstance(r, QualityReport)

    def test_overall_clamped(self):
        r = assess_quality(_solid(128))
        assert 0.0 <= r.overall <= 1.0

    def test_weights_1000(self):
        r = assess_quality(_gradient(), weights=(1.0, 0.0, 0.0, 0.0))
        assert isinstance(r.overall, float)

    def test_weights_0001(self):
        r = assess_quality(_gradient(), weights=(0.0, 0.0, 0.0, 1.0))
        assert isinstance(r.overall, float)


# ─── TestFilterByQualityExtra ─────────────────────────────────────────────────

class TestFilterByQualityExtra:
    def test_returns_two_lists(self):
        good, bad = filter_by_quality([_gradient()], min_score=0.0)
        assert isinstance(good, list) and isinstance(bad, list)

    def test_all_accepted_zero(self):
        imgs = [_gradient() for _ in range(3)]
        good, bad = filter_by_quality(imgs, min_score=0.0)
        assert len(good) == 3 and len(bad) == 0

    def test_all_rejected_one(self):
        imgs = [_gradient() for _ in range(2)]
        good, bad = filter_by_quality(imgs, min_score=1.0)
        assert len(good) == 0 and len(bad) == 2

    def test_total_preserved(self):
        imgs = [_gradient(), _noisy(), _solid(0)]
        good, bad = filter_by_quality(imgs, min_score=0.5)
        assert len(good) + len(bad) == 3

    def test_empty_input(self):
        good, bad = filter_by_quality([], min_score=0.5)
        assert good == [] and bad == []

    def test_good_are_ndarray(self):
        imgs = [_gradient()]
        good, _ = filter_by_quality(imgs, min_score=0.0)
        for g in good:
            assert isinstance(g, np.ndarray)


# ─── TestBatchAssessQualityExtra ──────────────────────────────────────────────

class TestBatchAssessQualityExtra:
    def test_length_preserved(self):
        imgs = [_gradient(), _solid(128), _checker()]
        assert len(batch_assess_quality(imgs)) == 3

    def test_all_quality_reports(self):
        for r in batch_assess_quality([_gradient(), _noisy()]):
            assert isinstance(r, QualityReport)

    def test_empty_input(self):
        assert batch_assess_quality([]) == []

    def test_kwargs_forwarded(self):
        r = batch_assess_quality([_gradient()], min_score=0.99)
        assert r[0].params["min_score"] == pytest.approx(0.99)

    def test_single_image(self):
        r = batch_assess_quality([_checker()])
        assert len(r) == 1
        assert isinstance(r[0], QualityReport)
