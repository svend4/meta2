"""Тесты для puzzle_reconstruction/matching/patch_matcher.py."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_gray(h=64, w=64, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)


def make_noisy(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def make_bgr(h=64, w=64, fill=128):
    return np.full((h, w, 3), fill, dtype=np.uint8)


# ─── PatchMatch ───────────────────────────────────────────────────────────────

class TestPatchMatch:
    def test_creation(self):
        pm = PatchMatch(idx1=0, idx2=1, side1=2, side2=0,
                        ncc=0.8, ssd=0.9, ssim=0.85, total_score=0.85)
        assert pm.idx1 == 0
        assert pm.idx2 == 1
        assert pm.side1 == 2
        assert pm.side2 == 0
        assert pm.ncc == pytest.approx(0.8)
        assert pm.ssd == pytest.approx(0.9)
        assert pm.ssim == pytest.approx(0.85)
        assert pm.total_score == pytest.approx(0.85)
        assert pm.params == {}

    def test_params_stored(self):
        pm = PatchMatch(idx1=0, idx2=1, side1=0, side2=2,
                        ncc=0.5, ssd=0.5, ssim=0.5, total_score=0.5,
                        params={"border_px": 10})
        assert pm.params["border_px"] == 10

    def test_negative_ncc_valid(self):
        pm = PatchMatch(idx1=0, idx2=1, side1=0, side2=2,
                        ncc=-0.5, ssd=0.5, ssim=0.5, total_score=0.5)
        assert pm.ncc == pytest.approx(-0.5)


# ─── extract_edge_strip ───────────────────────────────────────────────────────

class TestExtractEdgeStrip:
    def test_side_0_top(self):
        img = make_gray(h=64, w=80)
        strip = extract_edge_strip(img, side=0, border_px=10)
        assert strip.shape == (10, 80)

    def test_side_1_right(self):
        img = make_gray(h=64, w=80)
        strip = extract_edge_strip(img, side=1, border_px=5)
        assert strip.shape == (64, 5)

    def test_side_2_bottom(self):
        img = make_gray(h=64, w=80)
        strip = extract_edge_strip(img, side=2, border_px=8)
        assert strip.shape == (8, 80)

    def test_side_3_left(self):
        img = make_gray(h=64, w=80)
        strip = extract_edge_strip(img, side=3, border_px=6)
        assert strip.shape == (64, 6)

    def test_returns_float32(self):
        img = make_gray()
        strip = extract_edge_strip(img, side=0)
        assert strip.dtype == np.float32

    def test_invalid_side_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            extract_edge_strip(img, side=4)

    def test_invalid_side_negative_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            extract_edge_strip(img, side=-1)

    def test_accepts_bgr(self):
        img = make_bgr(h=64, w=80)
        strip = extract_edge_strip(img, side=0, border_px=5)
        assert strip.shape == (5, 80)  # grayscale conversion

    def test_pixel_values_from_image(self):
        img = np.full((32, 32), 200, dtype=np.uint8)
        strip = extract_edge_strip(img, side=0, border_px=4)
        assert float(strip.mean()) == pytest.approx(200.0)


# ─── ncc_score ────────────────────────────────────────────────────────────────

class TestNccScore:
    def test_identical_strips_returns_1(self):
        rng = np.random.default_rng(0)
        s = rng.uniform(0, 255, (10, 32)).astype(np.float32)
        score = ncc_score(s, s)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_returns_float(self):
        s1 = make_gray().astype(np.float32)
        s2 = make_noisy().astype(np.float32)
        score = ncc_score(s1[:5], s2[:5])
        assert isinstance(score, float)

    def test_result_in_neg1_1(self):
        s1 = make_noisy(seed=0).astype(np.float32)
        s2 = make_noisy(seed=1).astype(np.float32)
        score = ncc_score(s1, s2)
        assert -1.0 <= score <= 1.0

    def test_uniform_strip_returns_0(self):
        s1 = make_gray().astype(np.float32)
        s2 = make_noisy().astype(np.float32)
        # Uniform strip has std < 1e-6 → returns 0
        score = ncc_score(s1, s2)
        assert score == pytest.approx(0.0)

    def test_empty_strip_returns_0(self):
        s = np.array([], dtype=np.float32).reshape(0, 0)
        score = ncc_score(s, s)
        assert score == pytest.approx(0.0)


# ─── ssd_score ────────────────────────────────────────────────────────────────

class TestSsdScore:
    def test_identical_strips_returns_1(self):
        s = make_noisy().astype(np.float32)
        score = ssd_score(s, s)
        assert score == pytest.approx(1.0)

    def test_result_in_0_1(self):
        s1 = make_noisy(seed=0).astype(np.float32)
        s2 = make_noisy(seed=1).astype(np.float32)
        score = ssd_score(s1, s2)
        assert 0.0 < score <= 1.0

    def test_empty_strip_returns_0(self):
        s = np.array([], dtype=np.float32)
        score = ssd_score(s, s)
        assert score == pytest.approx(0.0)

    def test_returns_float(self):
        s1 = make_noisy(seed=2).astype(np.float32)
        s2 = make_noisy(seed=3).astype(np.float32)
        score = ssd_score(s1, s2)
        assert isinstance(score, float)

    def test_large_diff_low_score(self):
        s1 = np.zeros((10, 10), dtype=np.float32)
        s2 = np.full((10, 10), 255.0, dtype=np.float32)
        score = ssd_score(s1, s2)
        assert score <= 0.5


# ─── ssim_score ───────────────────────────────────────────────────────────────

class TestSsimScore:
    def test_identical_returns_1(self):
        s = make_noisy().astype(np.float32)
        score = ssim_score(s, s)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_result_in_0_1(self):
        s1 = make_noisy(seed=0).astype(np.float32)
        s2 = make_noisy(seed=1).astype(np.float32)
        score = ssim_score(s1, s2)
        assert 0.0 <= score <= 1.0

    def test_empty_returns_0(self):
        s = np.array([], dtype=np.float32)
        score = ssim_score(s, s)
        assert score == pytest.approx(0.0)

    def test_returns_float(self):
        s1 = make_noisy(seed=4).astype(np.float32)
        s2 = make_noisy(seed=5).astype(np.float32)
        score = ssim_score(s1, s2)
        assert isinstance(score, float)


# ─── match_edge_strips ────────────────────────────────────────────────────────

class TestMatchEdgeStrips:
    def test_returns_4_tuple(self):
        s1 = make_noisy(seed=0).astype(np.float32)
        s2 = make_noisy(seed=1).astype(np.float32)
        result = match_edge_strips(s1, s2)
        assert len(result) == 4

    def test_identical_strips_high_total(self):
        s = make_noisy().astype(np.float32)
        ncc, ssd, ssim, total = match_edge_strips(s, s)
        assert total > 0.8

    def test_total_in_0_1(self):
        s1 = make_noisy(seed=0).astype(np.float32)
        s2 = make_noisy(seed=1).astype(np.float32)
        _, _, _, total = match_edge_strips(s1, s2)
        assert 0.0 <= total <= 1.0

    def test_different_shapes_clipped(self):
        s1 = make_noisy(h=10, w=32, seed=0).astype(np.float32)
        s2 = make_noisy(h=8, w=40, seed=1).astype(np.float32)
        # Should not raise
        result = match_edge_strips(s1, s2)
        assert len(result) == 4

    def test_custom_weights_normalized(self):
        s1 = make_noisy(seed=0).astype(np.float32)
        s2 = make_noisy(seed=1).astype(np.float32)
        _, _, _, total = match_edge_strips(s1, s2, weights=(2.0, 1.0, 1.0))
        assert 0.0 <= total <= 1.0


# ─── match_patch_pair ─────────────────────────────────────────────────────────

class TestMatchPatchPair:
    def test_returns_patch_match(self):
        img1 = make_noisy()
        img2 = make_noisy(seed=1)
        pm = match_patch_pair(img1, img2)
        assert isinstance(pm, PatchMatch)

    def test_idx1_idx2_stored(self):
        img1 = make_noisy()
        img2 = make_noisy(seed=1)
        pm = match_patch_pair(img1, img2, idx1=3, idx2=7)
        assert pm.idx1 == 3
        assert pm.idx2 == 7

    def test_side1_side2_stored(self):
        img1 = make_noisy()
        img2 = make_noisy(seed=1)
        pm = match_patch_pair(img1, img2, side1=1, side2=3)
        assert pm.side1 == 1
        assert pm.side2 == 3

    def test_total_score_in_0_1(self):
        img1 = make_noisy()
        img2 = make_noisy(seed=1)
        pm = match_patch_pair(img1, img2)
        assert 0.0 <= pm.total_score <= 1.0

    def test_ssd_in_0_1(self):
        img1 = make_noisy()
        img2 = make_noisy(seed=1)
        pm = match_patch_pair(img1, img2)
        assert 0.0 <= pm.ssd <= 1.0

    def test_ssim_in_0_1(self):
        img1 = make_noisy()
        img2 = make_noisy(seed=1)
        pm = match_patch_pair(img1, img2)
        assert 0.0 <= pm.ssim <= 1.0

    def test_params_contain_border_px(self):
        img1 = make_noisy()
        img2 = make_noisy(seed=1)
        pm = match_patch_pair(img1, img2, border_px=5)
        assert pm.params["border_px"] == 5

    def test_identical_images_high_score(self):
        img = make_noisy()
        pm = match_patch_pair(img, img, side1=2, side2=2)
        assert pm.total_score > 0.8

    def test_accepts_bgr(self):
        img1 = make_bgr()
        img2 = make_bgr(fill=100)
        pm = match_patch_pair(img1, img2)
        assert isinstance(pm, PatchMatch)

    def test_all_4_sides(self):
        img1 = make_noisy()
        img2 = make_noisy(seed=1)
        for s in range(4):
            pm = match_patch_pair(img1, img2, side1=s, side2=(s+2) % 4)
            assert isinstance(pm, PatchMatch)


# ─── batch_patch_match ────────────────────────────────────────────────────────

class TestBatchPatchMatch:
    def test_empty_pairs_returns_empty(self):
        images = [make_noisy(seed=i) for i in range(3)]
        result = batch_patch_match(images, [])
        assert result == []

    def test_length_matches_pairs(self):
        images = [make_noisy(seed=i) for i in range(3)]
        pairs = [(0, 1), (1, 2), (0, 2)]
        result = batch_patch_match(images, pairs)
        assert len(result) == 3

    def test_returns_list_of_patch_matches(self):
        images = [make_noisy(seed=i) for i in range(3)]
        pairs = [(0, 1), (1, 2)]
        result = batch_patch_match(images, pairs)
        for pm in result:
            assert isinstance(pm, PatchMatch)

    def test_default_side_pairs(self):
        images = [make_noisy(seed=i) for i in range(2)]
        result = batch_patch_match(images, [(0, 1)])
        # Default side_pairs=(2, 0)
        assert result[0].side1 == 2
        assert result[0].side2 == 0

    def test_custom_side_pairs(self):
        images = [make_noisy(seed=i) for i in range(2)]
        result = batch_patch_match(images, [(0, 1)], side_pairs=[(1, 3)])
        assert result[0].side1 == 1
        assert result[0].side2 == 3

    def test_indices_stored_correctly(self):
        images = [make_noisy(seed=i) for i in range(3)]
        pairs = [(0, 2)]
        result = batch_patch_match(images, pairs)
        assert result[0].idx1 == 0
        assert result[0].idx2 == 2
