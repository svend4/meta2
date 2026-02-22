"""
Тесты для puzzle_reconstruction.matching.patch_matcher.
"""
import pytest
import numpy as np

from puzzle_reconstruction.matching.patch_matcher import (
    PatchMatch,
    extract_edge_strip,
    ncc_score,
    ssd_score,
    ssim_score,
    match_edge_strips,
    match_patch_pair,
    batch_patch_match,
)


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _solid_gray(value: int, h: int = 64, w: int = 64) -> np.ndarray:
    """Grayscale изображение одного цвета."""
    return np.full((h, w), value, dtype=np.uint8)


def _solid_bgr(value: int, h: int = 64, w: int = 64) -> np.ndarray:
    """BGR-изображение одного цвета."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _gradient_img(h: int = 64, w: int = 64) -> np.ndarray:
    """Grayscale-градиент 0..255 по горизонтали."""
    row = np.linspace(0, 255, w, dtype=np.float32)
    return np.tile(row, (h, 1)).astype(np.uint8)


def _noisy_img(seed: int = 42, h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


# ─── PatchMatch ───────────────────────────────────────────────────────────────

class TestPatchMatch:
    def test_fields(self):
        pm = PatchMatch(idx1=0, idx2=1, side1=2, side2=0,
                        ncc=0.9, ssd=0.8, ssim=0.85, total_score=0.85)
        assert pm.idx1 == 0
        assert pm.idx2 == 1
        assert pm.side1 == 2
        assert pm.side2 == 0
        assert pm.ncc   == pytest.approx(0.9)
        assert pm.ssd   == pytest.approx(0.8)
        assert pm.ssim  == pytest.approx(0.85)

    def test_default_params(self):
        pm = PatchMatch(0, 1, 0, 2, 0.5, 0.5, 0.5, 0.5)
        assert pm.params == {}

    def test_repr_contains_indices(self):
        pm = PatchMatch(3, 7, 2, 0, 0.5, 0.5, 0.5, 0.5)
        r = repr(pm)
        assert "3" in r and "7" in r


# ─── extract_edge_strip ───────────────────────────────────────────────────────

class TestExtractEdgeStrip:
    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_dtype_float32(self, side):
        img = _solid_gray(128)
        strip = extract_edge_strip(img, side=side, border_px=10)
        assert strip.dtype == np.float32

    def test_side0_top_shape(self):
        img = _solid_gray(100, h=64, w=80)
        strip = extract_edge_strip(img, side=0, border_px=8)
        assert strip.shape == (8, 80)

    def test_side2_bottom_shape(self):
        img = _solid_gray(100, h=64, w=80)
        strip = extract_edge_strip(img, side=2, border_px=8)
        assert strip.shape == (8, 80)

    def test_side1_right_shape(self):
        img = _solid_gray(100, h=64, w=80)
        strip = extract_edge_strip(img, side=1, border_px=6)
        assert strip.shape == (64, 6)

    def test_side3_left_shape(self):
        img = _solid_gray(100, h=64, w=80)
        strip = extract_edge_strip(img, side=3, border_px=6)
        assert strip.shape == (64, 6)

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError):
            extract_edge_strip(_solid_gray(0), side=4)

    def test_bgr_input_converted(self):
        img = _solid_bgr(200, h=32, w=32)
        strip = extract_edge_strip(img, side=0, border_px=5)
        assert strip.shape == (5, 32)

    def test_strip_values_from_top(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[:10, :] = 200
        strip = extract_edge_strip(img, side=0, border_px=10)
        assert np.all(strip == 200.0)

    def test_strip_values_from_bottom(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[-5:, :] = 100
        strip = extract_edge_strip(img, side=2, border_px=5)
        assert np.all(strip == 100.0)


# ─── ncc_score ────────────────────────────────────────────────────────────────

class TestNccScore:
    def test_identical_strips_near_one(self):
        s = np.random.rand(10, 20).astype(np.float32) * 200
        assert ncc_score(s, s) == pytest.approx(1.0, abs=1e-5)

    def test_inverted_strips_near_minus_one(self):
        s  = np.random.rand(10, 20).astype(np.float32) * 200
        si = 255.0 - s
        assert ncc_score(s, si) == pytest.approx(-1.0, abs=0.05)

    def test_flat_strip1_returns_zero(self):
        s1 = np.full((5, 10), 128.0, dtype=np.float32)
        s2 = np.random.rand(5, 10).astype(np.float32) * 200
        assert ncc_score(s1, s2) == pytest.approx(0.0)

    def test_flat_strip2_returns_zero(self):
        s1 = np.random.rand(5, 10).astype(np.float32) * 200
        s2 = np.full((5, 10), 50.0, dtype=np.float32)
        assert ncc_score(s1, s2) == pytest.approx(0.0)

    def test_result_in_range(self):
        s1 = np.random.rand(8, 12).astype(np.float32) * 255
        s2 = np.random.rand(8, 12).astype(np.float32) * 255
        r  = ncc_score(s1, s2)
        assert -1.0 <= r <= 1.0

    def test_empty_strip_returns_zero(self):
        s = np.empty((0, 5), dtype=np.float32)
        assert ncc_score(s, s) == pytest.approx(0.0)


# ─── ssd_score ────────────────────────────────────────────────────────────────

class TestSsdScore:
    def test_identical_strips_score_one(self):
        s = np.random.rand(10, 10).astype(np.float32) * 200
        assert ssd_score(s, s) == pytest.approx(1.0)

    def test_max_diff_score_low(self):
        s1 = np.zeros((10, 10), dtype=np.float32)
        s2 = np.full((10, 10), 255.0, dtype=np.float32)
        r  = ssd_score(s1, s2)
        assert 0.0 < r < 1.0

    def test_score_in_range(self):
        s1 = np.random.rand(8, 8).astype(np.float32) * 255
        s2 = np.random.rand(8, 8).astype(np.float32) * 255
        r  = ssd_score(s1, s2)
        assert 0.0 < r <= 1.0

    def test_empty_strip_returns_zero(self):
        s = np.empty((0, 5), dtype=np.float32)
        assert ssd_score(s, s) == pytest.approx(0.0)

    def test_monotone_decreasing_with_diff(self):
        base = np.full((10, 10), 100.0, dtype=np.float32)
        r1 = ssd_score(base, base + 10)
        r2 = ssd_score(base, base + 50)
        assert r1 > r2


# ─── ssim_score ───────────────────────────────────────────────────────────────

class TestSsimScore:
    def test_identical_returns_one(self):
        s = np.random.rand(8, 8).astype(np.float32) * 200
        assert ssim_score(s, s) == pytest.approx(1.0, abs=1e-4)

    def test_result_in_zero_one(self):
        s1 = np.random.rand(8, 8).astype(np.float32) * 255
        s2 = np.random.rand(8, 8).astype(np.float32) * 255
        r  = ssim_score(s1, s2)
        assert 0.0 <= r <= 1.0

    def test_empty_returns_zero(self):
        s = np.empty((0, 5), dtype=np.float32)
        assert ssim_score(s, s) == pytest.approx(0.0)

    def test_similar_strips_higher_than_different(self):
        base  = np.random.rand(10, 10).astype(np.float32) * 200
        close = base + np.random.rand(10, 10).astype(np.float32) * 5
        far   = np.random.rand(10, 10).astype(np.float32) * 200
        assert ssim_score(base, close) >= ssim_score(base, far) - 0.1


# ─── match_edge_strips ────────────────────────────────────────────────────────

class TestMatchEdgeStrips:
    def _identical_strips(self, h=5, w=20):
        s = np.random.rand(h, w).astype(np.float32) * 200
        return s, s.copy()

    def test_returns_4_tuple(self):
        s1, s2 = self._identical_strips()
        result = match_edge_strips(s1, s2)
        assert len(result) == 4

    def test_identical_total_near_one(self):
        s1, s2 = self._identical_strips()
        ncc, ssd, ssim, total = match_edge_strips(s1, s2)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_total_in_zero_one(self):
        s1 = np.random.rand(5, 20).astype(np.float32) * 255
        s2 = np.random.rand(5, 20).astype(np.float32) * 255
        _, _, _, total = match_edge_strips(s1, s2)
        assert 0.0 <= total <= 1.0

    def test_custom_weights_sum_to_one(self):
        s1, s2 = self._identical_strips()
        ncc, ssd, ssim, total = match_edge_strips(s1, s2, weights=(2.0, 1.0, 1.0))
        assert total == pytest.approx(1.0, abs=0.01)

    def test_different_size_strips_auto_resize(self):
        s1 = np.random.rand(5, 20).astype(np.float32) * 200
        s2 = np.random.rand(8, 30).astype(np.float32) * 200
        # Не должно падать
        ncc, ssd, ssim, total = match_edge_strips(s1, s2)
        assert 0.0 <= total <= 1.0


# ─── match_patch_pair ─────────────────────────────────────────────────────────

class TestMatchPatchPair:
    def test_returns_patch_match(self):
        img1 = _solid_gray(200)
        img2 = _solid_gray(200)
        pm = match_patch_pair(img1, img2, idx1=0, idx2=1)
        assert isinstance(pm, PatchMatch)

    def test_indices_stored(self):
        img1 = _solid_gray(100)
        img2 = _solid_gray(150)
        pm = match_patch_pair(img1, img2, idx1=3, idx2=7)
        assert pm.idx1 == 3
        assert pm.idx2 == 7

    def test_sides_stored(self):
        img1 = _solid_gray(100)
        img2 = _solid_gray(100)
        pm = match_patch_pair(img1, img2, side1=1, side2=3)
        assert pm.side1 == 1
        assert pm.side2 == 3

    def test_identical_images_high_score(self):
        img = _gradient_img()
        pm  = match_patch_pair(img, img.copy(), side1=2, side2=0, border_px=8)
        assert pm.total_score > 0.8

    def test_border_px_stored_in_params(self):
        img1 = _solid_gray(100)
        img2 = _solid_gray(100)
        pm = match_patch_pair(img1, img2, border_px=15)
        assert pm.params["border_px"] == 15

    def test_weights_stored_in_params(self):
        img1 = _solid_gray(100)
        img2 = _solid_gray(100)
        pm = match_patch_pair(img1, img2, weights=(1.0, 0.0, 0.0))
        assert pm.params["weights"] == (1.0, 0.0, 0.0)

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides(self, side):
        img1 = _gradient_img()
        img2 = _gradient_img()
        pm = match_patch_pair(img1, img2, side1=side, side2=side, border_px=5)
        assert isinstance(pm.total_score, float)

    def test_bgr_images(self):
        img1 = _solid_bgr(150)
        img2 = _solid_bgr(150)
        pm = match_patch_pair(img1, img2)
        assert isinstance(pm, PatchMatch)


# ─── batch_patch_match ────────────────────────────────────────────────────────

class TestBatchPatchMatch:
    def test_empty_pairs(self):
        result = batch_patch_match([_solid_gray(100)], [])
        assert result == []

    def test_length_matches_pairs(self):
        imgs  = [_solid_gray(i * 30) for i in range(4)]
        pairs = [(0, 1), (1, 2), (2, 3)]
        result = batch_patch_match(imgs, pairs)
        assert len(result) == 3

    def test_all_results_patch_match(self):
        imgs   = [_gradient_img() for _ in range(3)]
        pairs  = [(0, 1), (1, 2)]
        result = batch_patch_match(imgs, pairs)
        for r in result:
            assert isinstance(r, PatchMatch)

    def test_indices_correct(self):
        imgs  = [_solid_gray(i * 50) for i in range(4)]
        pairs = [(0, 3)]
        result = batch_patch_match(imgs, pairs)
        assert result[0].idx1 == 0
        assert result[0].idx2 == 3

    def test_custom_side_pairs(self):
        imgs       = [_gradient_img(), _gradient_img()]
        pairs      = [(0, 1)]
        side_pairs = [(1, 3)]
        result = batch_patch_match(imgs, pairs, side_pairs=side_pairs)
        assert result[0].side1 == 1
        assert result[0].side2 == 3

    def test_default_side_pair_is_2_0(self):
        imgs  = [_solid_gray(100), _solid_gray(100)]
        pairs = [(0, 1)]
        result = batch_patch_match(imgs, pairs)
        assert result[0].side1 == 2
        assert result[0].side2 == 0

    def test_kwargs_forwarded(self):
        imgs  = [_gradient_img(), _gradient_img()]
        pairs = [(0, 1)]
        result = batch_patch_match(imgs, pairs, border_px=12)
        assert result[0].params["border_px"] == 12
