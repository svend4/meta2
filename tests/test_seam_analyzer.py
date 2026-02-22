"""
Тесты для puzzle_reconstruction.verification.seam_analyzer.
"""
import pytest
import numpy as np

from puzzle_reconstruction.verification.seam_analyzer import (
    SeamAnalysis,
    extract_seam_profiles,
    brightness_continuity,
    gradient_continuity,
    texture_continuity,
    analyze_seam,
    score_seam_quality,
    batch_analyze_seams,
)


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _solid_gray(value: int, h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def _solid_bgr(value: int, h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _gradient_gray(h: int = 64, w: int = 64) -> np.ndarray:
    row = np.linspace(0, 255, w, dtype=np.float32)
    return np.tile(row, (h, 1)).astype(np.uint8)


def _ramp_vertical(h: int = 64, w: int = 64) -> np.ndarray:
    col = np.linspace(0, 255, h, dtype=np.float32)
    return np.tile(col, (w, 1)).T.astype(np.uint8)


def _noisy_gray(seed: int = 7, h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


# ─── SeamAnalysis ─────────────────────────────────────────────────────────────

class TestSeamAnalysis:
    def test_fields_accessible(self):
        sa = SeamAnalysis(
            idx1=0, idx2=1, side1=2, side2=0,
            brightness_score=0.8, gradient_score=0.7,
            texture_score=0.6, quality_score=0.7,
            profile_length=64,
        )
        assert sa.idx1 == 0
        assert sa.idx2 == 1
        assert sa.side1 == 2
        assert sa.side2 == 0
        assert sa.brightness_score == pytest.approx(0.8)
        assert sa.gradient_score   == pytest.approx(0.7)
        assert sa.texture_score    == pytest.approx(0.6)
        assert sa.quality_score    == pytest.approx(0.7)
        assert sa.profile_length   == 64

    def test_default_params_empty_dict(self):
        sa = SeamAnalysis(0, 1, 2, 0, 0.5, 0.5, 0.5, 0.5, 32)
        assert sa.params == {}

    def test_repr_contains_quality(self):
        sa = SeamAnalysis(0, 1, 2, 0, 0.9, 0.9, 0.9, 0.9, 64)
        assert "quality" in repr(sa).lower() or "0.9" in repr(sa)


# ─── extract_seam_profiles ────────────────────────────────────────────────────

class TestExtractSeamProfiles:
    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_returns_two_float64_arrays(self, side):
        img = _solid_gray(128)
        p1, p2 = extract_seam_profiles(img, img, side1=side, side2=side)
        assert p1.dtype == np.float64
        assert p2.dtype == np.float64

    def test_profiles_same_length(self):
        img1 = _solid_gray(100, h=64, w=80)
        img2 = _solid_gray(200, h=50, w=60)
        p1, p2 = extract_seam_profiles(img1, img2, side1=2, side2=0)
        assert len(p1) == len(p2)

    def test_side0_top_profile_length_equals_width(self):
        img = _solid_gray(128, h=64, w=80)
        p, _ = extract_seam_profiles(img, img, side1=0, side2=0)
        assert len(p) == 80

    def test_side1_right_profile_length_equals_height(self):
        img = _solid_gray(128, h=64, w=80)
        p, _ = extract_seam_profiles(img, img, side1=1, side2=1)
        assert len(p) == 64

    def test_side2_bottom_values(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[-8:, :] = 200
        p, _ = extract_seam_profiles(img, img, side1=2, side2=0, border_px=8)
        assert np.all(p == 200.0)

    def test_side3_left_values(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[:, :6] = 100
        p, _ = extract_seam_profiles(img, img, side1=3, side2=1, border_px=6)
        assert np.all(p == 100.0)

    def test_invalid_side1_raises(self):
        img = _solid_gray(100)
        with pytest.raises(ValueError):
            extract_seam_profiles(img, img, side1=5, side2=0)

    def test_invalid_side2_raises(self):
        img = _solid_gray(100)
        with pytest.raises(ValueError):
            extract_seam_profiles(img, img, side1=0, side2=-1)

    def test_bgr_converted(self):
        img = _solid_bgr(150)
        p1, p2 = extract_seam_profiles(img, img, side1=2, side2=0)
        assert len(p1) == 64
        assert np.all(p1 > 0)

    def test_different_width_truncated(self):
        img1 = _solid_gray(100, h=32, w=80)
        img2 = _solid_gray(100, h=32, w=60)
        p1, p2 = extract_seam_profiles(img1, img2, side1=2, side2=0)
        assert len(p1) == 60 and len(p2) == 60


# ─── brightness_continuity ────────────────────────────────────────────────────

class TestBrightnessContinuity:
    def test_identical_profiles_one(self):
        p = np.random.rand(50).astype(np.float64) * 200
        assert brightness_continuity(p, p) == pytest.approx(1.0)

    def test_max_diff_low(self):
        p1 = np.zeros(50, dtype=np.float64)
        p2 = np.full(50, 255.0)
        r  = brightness_continuity(p1, p2)
        assert 0.0 <= r < 0.1

    def test_half_diff(self):
        p1 = np.zeros(50, dtype=np.float64)
        p2 = np.full(50, 127.5)
        r  = brightness_continuity(p1, p2)
        assert pytest.approx(r, abs=0.01) == 0.5

    def test_empty_returns_zero(self):
        p = np.empty(0, dtype=np.float64)
        assert brightness_continuity(p, p) == pytest.approx(0.0)

    def test_result_in_zero_one(self):
        p1 = np.random.rand(30).astype(np.float64) * 255
        p2 = np.random.rand(30).astype(np.float64) * 255
        r  = brightness_continuity(p1, p2)
        assert 0.0 <= r <= 1.0

    def test_custom_max_diff(self):
        p1 = np.zeros(10, dtype=np.float64)
        p2 = np.full(10, 50.0)
        # max_diff=100 → score = 1 - 50/100 = 0.5
        r  = brightness_continuity(p1, p2, max_diff=100.0)
        assert r == pytest.approx(0.5, abs=0.01)


# ─── gradient_continuity ──────────────────────────────────────────────────────

class TestGradientContinuity:
    def test_identical_returns_one(self):
        p = np.linspace(0, 200, 50).astype(np.float64)
        assert gradient_continuity(p, p) == pytest.approx(1.0, abs=1e-5)

    def test_both_flat_returns_one(self):
        p1 = np.full(20, 100.0, dtype=np.float64)
        p2 = np.full(20,  50.0, dtype=np.float64)
        assert gradient_continuity(p1, p2) == pytest.approx(1.0)

    def test_one_flat_returns_half(self):
        p1 = np.full(20, 100.0, dtype=np.float64)
        p2 = np.random.rand(20).astype(np.float64) * 200
        # p2 must have non-zero std
        p2 += np.arange(20, dtype=np.float64)
        r = gradient_continuity(p1, p2)
        assert r == pytest.approx(0.5)

    def test_result_in_zero_one(self):
        p1 = np.random.rand(40).astype(np.float64) * 200
        p2 = np.random.rand(40).astype(np.float64) * 200
        r  = gradient_continuity(p1, p2)
        assert 0.0 <= r <= 1.0

    def test_short_array_returns_zero(self):
        p = np.array([5.0], dtype=np.float64)
        assert gradient_continuity(p, p) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        p = np.empty(0, dtype=np.float64)
        assert gradient_continuity(p, p) == pytest.approx(0.0)

    def test_inverted_gradient_low_score(self):
        p1 = np.linspace(0, 200, 50).astype(np.float64)
        p2 = np.linspace(200, 0, 50).astype(np.float64)
        r  = gradient_continuity(p1, p2)
        assert r < 0.3


# ─── texture_continuity ───────────────────────────────────────────────────────

class TestTextureContinuity:
    def test_identical_returns_one(self):
        p = np.random.rand(50).astype(np.float64) * 200
        assert texture_continuity(p, p) == pytest.approx(1.0)

    def test_both_uniform_returns_one(self):
        p1 = np.full(20, 100.0, dtype=np.float64)
        p2 = np.full(20,  50.0, dtype=np.float64)
        assert texture_continuity(p1, p2) == pytest.approx(1.0)

    def test_one_uniform_low_score(self):
        p1 = np.full(20, 100.0, dtype=np.float64)
        p2 = np.random.rand(20).astype(np.float64) * 200
        # Force non-zero std in p2
        p2[0] = 0.0; p2[-1] = 200.0
        r = texture_continuity(p1, p2)
        assert r == pytest.approx(0.0, abs=1e-6)

    def test_result_in_zero_one(self):
        p1 = np.random.rand(50).astype(np.float64) * 200
        p2 = np.random.rand(50).astype(np.float64) * 10
        r  = texture_continuity(p1, p2)
        assert 0.0 <= r <= 1.0

    def test_empty_returns_zero(self):
        p = np.empty(0, dtype=np.float64)
        assert texture_continuity(p, p) == pytest.approx(0.0)

    def test_similar_variance_high_score(self):
        rng = np.random.default_rng(42)
        p1  = rng.normal(100, 10, 50).astype(np.float64)
        p2  = rng.normal(100, 11, 50).astype(np.float64)
        r   = texture_continuity(p1, p2)
        assert r > 0.8


# ─── analyze_seam ─────────────────────────────────────────────────────────────

class TestAnalyzeSeam:
    def test_returns_seam_analysis(self):
        img = _solid_gray(128)
        sa  = analyze_seam(img, img)
        assert isinstance(sa, SeamAnalysis)

    def test_indices_stored(self):
        img = _solid_gray(100)
        sa  = analyze_seam(img, img, idx1=5, idx2=9)
        assert sa.idx1 == 5
        assert sa.idx2 == 9

    def test_sides_stored(self):
        img = _solid_gray(100)
        sa  = analyze_seam(img, img, side1=1, side2=3)
        assert sa.side1 == 1
        assert sa.side2 == 3

    def test_identical_images_high_quality(self):
        img = _gradient_gray()
        sa  = analyze_seam(img, img, border_px=8)
        assert sa.quality_score > 0.8

    def test_quality_in_zero_one(self):
        img1 = _noisy_gray(seed=1)
        img2 = _noisy_gray(seed=99)
        sa   = analyze_seam(img1, img2)
        assert 0.0 <= sa.quality_score <= 1.0

    def test_border_px_stored_in_params(self):
        img = _solid_gray(100)
        sa  = analyze_seam(img, img, border_px=12)
        assert sa.params["border_px"] == 12

    def test_weights_stored_in_params(self):
        img = _solid_gray(100)
        w   = (1.0, 0.0, 0.0)
        sa  = analyze_seam(img, img, weights=w)
        assert sa.params["weights"] == w

    def test_profile_length_positive(self):
        img = _solid_gray(128, h=64, w=80)
        sa  = analyze_seam(img, img, side1=2, side2=0)
        assert sa.profile_length > 0

    def test_bgr_images(self):
        img = _solid_bgr(150)
        sa  = analyze_seam(img, img)
        assert isinstance(sa, SeamAnalysis)

    def test_uniform_image_brightness_one(self):
        img = _solid_gray(200)
        sa  = analyze_seam(img, img, side1=2, side2=0)
        assert sa.brightness_score == pytest.approx(1.0)


# ─── score_seam_quality ───────────────────────────────────────────────────────

class TestScoreSeamQuality:
    def test_returns_float(self):
        img = _solid_gray(100)
        sa  = analyze_seam(img, img)
        r   = score_seam_quality(sa)
        assert isinstance(r, float)

    def test_result_equals_quality_score(self):
        img = _gradient_gray()
        sa  = analyze_seam(img, img)
        assert score_seam_quality(sa) == pytest.approx(sa.quality_score)

    def test_clamped_to_zero_one(self):
        # Патологический случай
        sa = SeamAnalysis(0, 1, 2, 0, 1.0, 1.0, 1.0, 1.1, 64)
        r  = score_seam_quality(sa)
        assert 0.0 <= r <= 1.0


# ─── batch_analyze_seams ──────────────────────────────────────────────────────

class TestBatchAnalyzeSeams:
    def test_empty_pairs(self):
        imgs   = [_solid_gray(100)]
        result = batch_analyze_seams(imgs, [])
        assert result == []

    def test_length_matches_pairs(self):
        imgs   = [_solid_gray(i * 40) for i in range(4)]
        pairs  = [(0, 1), (1, 2), (2, 3)]
        result = batch_analyze_seams(imgs, pairs)
        assert len(result) == 3

    def test_all_results_seam_analysis(self):
        imgs   = [_gradient_gray() for _ in range(3)]
        pairs  = [(0, 1), (1, 2)]
        result = batch_analyze_seams(imgs, pairs)
        for r in result:
            assert isinstance(r, SeamAnalysis)

    def test_indices_correct(self):
        imgs  = [_solid_gray(50 * i) for i in range(4)]
        pairs = [(0, 3)]
        result = batch_analyze_seams(imgs, pairs)
        assert result[0].idx1 == 0
        assert result[0].idx2 == 3

    def test_custom_side_pairs(self):
        imgs       = [_gradient_gray(), _gradient_gray()]
        pairs      = [(0, 1)]
        side_pairs = [(1, 3)]
        result = batch_analyze_seams(imgs, pairs, side_pairs=side_pairs)
        assert result[0].side1 == 1
        assert result[0].side2 == 3

    def test_default_side_pair_2_0(self):
        imgs  = [_solid_gray(128), _solid_gray(128)]
        pairs = [(0, 1)]
        result = batch_analyze_seams(imgs, pairs)
        assert result[0].side1 == 2
        assert result[0].side2 == 0

    def test_border_px_forwarded(self):
        imgs  = [_solid_gray(100), _solid_gray(100)]
        pairs = [(0, 1)]
        result = batch_analyze_seams(imgs, pairs, border_px=15)
        assert result[0].params["border_px"] == 15

    def test_weights_forwarded(self):
        imgs  = [_solid_gray(100), _solid_gray(100)]
        pairs = [(0, 1)]
        w     = (2.0, 1.0, 0.0)
        result = batch_analyze_seams(imgs, pairs, weights=w)
        assert result[0].params["weights"] == w
