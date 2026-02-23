"""Extra tests for puzzle_reconstruction.matching.patch_matcher (v2)."""
import numpy as np
import pytest

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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _solid(val, h=64, w=64):
    return np.full((h, w), val, dtype=np.uint8)


def _gradient(h=64, w=64):
    row = np.linspace(0, 255, w, dtype=np.float32)
    return np.tile(row, (h, 1)).astype(np.uint8)


def _noisy(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=64, w=64, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _fstrip(h=8, w=32, seed=0):
    return np.random.default_rng(seed).random((h, w)).astype(np.float32) * 200.0


# ─── TestPatchMatchFieldsExtra ────────────────────────────────────────────────

class TestPatchMatchFieldsExtra:
    def test_idx_fields(self):
        pm = PatchMatch(idx1=4, idx2=8, side1=0, side2=2,
                        ncc=0.6, ssd=0.7, ssim=0.8, total_score=0.7)
        assert pm.idx1 == 4
        assert pm.idx2 == 8

    def test_side_fields(self):
        pm = PatchMatch(0, 1, 2, 3, 0.5, 0.5, 0.5, 0.5)
        assert pm.side1 == 2
        assert pm.side2 == 3

    def test_all_score_fields_stored(self):
        pm = PatchMatch(0, 1, 0, 2, 0.1, 0.2, 0.3, 0.4)
        assert pm.ncc == pytest.approx(0.1)
        assert pm.ssd == pytest.approx(0.2)
        assert pm.ssim == pytest.approx(0.3)
        assert pm.total_score == pytest.approx(0.4)

    def test_params_empty_by_default(self):
        pm = PatchMatch(0, 1, 0, 2, 0.5, 0.5, 0.5, 0.5)
        assert pm.params == {}

    def test_repr_contains_indices(self):
        pm = PatchMatch(3, 7, 2, 0, 0.5, 0.5, 0.5, 0.5)
        r = repr(pm)
        assert "3" in r and "7" in r


# ─── TestExtractEdgeStripShapeExtra ──────────────────────────────────────────

class TestExtractEdgeStripShapeExtra:
    @pytest.mark.parametrize("side,expected_h,expected_w", [
        (0, 5, 80),   # top
        (2, 5, 80),   # bottom
        (1, 64, 5),   # right
        (3, 64, 5),   # left
    ])
    def test_shapes(self, side, expected_h, expected_w):
        img = _solid(100, h=64, w=80)
        strip = extract_edge_strip(img, side=side, border_px=5)
        assert strip.shape == (expected_h, expected_w)

    def test_dtype_float32_bgr(self):
        img = _bgr(64, 80, val=150)
        strip = extract_edge_strip(img, side=0, border_px=6)
        assert strip.dtype == np.float32

    def test_top_values_from_gradient(self):
        img = _gradient(64, 64)
        strip_top = extract_edge_strip(img, side=0, border_px=4)
        # Top strip: first 4 rows of gradient
        expected = np.tile(np.linspace(0, 255, 64, dtype=np.float32), (4, 1))
        np.testing.assert_allclose(strip_top, expected, atol=1.0)

    def test_side_4_raises(self):
        with pytest.raises(ValueError):
            extract_edge_strip(_solid(0), side=4)

    def test_side_minus1_raises(self):
        with pytest.raises(ValueError):
            extract_edge_strip(_solid(0), side=-1)

    def test_bgr_strip_is_2d(self):
        img = _bgr()
        strip = extract_edge_strip(img, side=1, border_px=3)
        assert strip.ndim == 2


# ─── TestNccScoreRangeExtra ───────────────────────────────────────────────────

class TestNccScoreRangeExtra:
    def test_identical_is_one(self):
        s = _fstrip(seed=10)
        assert ncc_score(s, s) == pytest.approx(1.0, abs=1e-5)

    def test_empty_is_zero(self):
        s = np.empty((0, 5), dtype=np.float32)
        assert ncc_score(s, s) == pytest.approx(0.0)

    def test_flat_vs_varying_is_zero(self):
        s1 = np.full((8, 32), 100.0, dtype=np.float32)
        s2 = _fstrip(seed=0)
        assert ncc_score(s1, s2) == pytest.approx(0.0)

    def test_range_any_pair(self):
        for seed in range(5):
            s1 = _fstrip(seed=seed)
            s2 = _fstrip(seed=seed + 10)
            assert -1.0 <= ncc_score(s1, s2) <= 1.0


# ─── TestSsdScoreRangeExtra ───────────────────────────────────────────────────

class TestSsdScoreRangeExtra:
    def test_identical_one(self):
        s = _fstrip(seed=20)
        assert ssd_score(s, s) == pytest.approx(1.0)

    def test_empty_zero(self):
        s = np.empty((0, 5), dtype=np.float32)
        assert ssd_score(s, s) == pytest.approx(0.0)

    def test_large_diff_low(self):
        s1 = np.zeros((8, 32), dtype=np.float32)
        s2 = np.full((8, 32), 255.0, dtype=np.float32)
        assert ssd_score(s1, s2) <= 0.5

    def test_in_range_random(self):
        for seed in range(5):
            s1 = _fstrip(seed=seed)
            s2 = _fstrip(seed=seed + 20)
            assert 0.0 <= ssd_score(s1, s2) <= 1.0


# ─── TestSsimScoreRangeExtra ──────────────────────────────────────────────────

class TestSsimScoreRangeExtra:
    def test_identical_one(self):
        s = _fstrip(seed=30)
        assert ssim_score(s, s) == pytest.approx(1.0, abs=1e-4)

    def test_empty_zero(self):
        s = np.empty((0, 5), dtype=np.float32)
        assert ssim_score(s, s) == pytest.approx(0.0)

    def test_in_range_random(self):
        for seed in range(5):
            s1 = _fstrip(seed=seed)
            s2 = _fstrip(seed=seed + 30)
            assert 0.0 <= ssim_score(s1, s2) <= 1.0

    def test_float_type(self):
        s1 = _fstrip(seed=0)
        s2 = _fstrip(seed=1)
        assert isinstance(ssim_score(s1, s2), float)


# ─── TestMatchEdgeStripsParamsExtra ──────────────────────────────────────────

class TestMatchEdgeStripsParamsExtra:
    def test_default_weights_identical(self):
        s = _fstrip(seed=0)
        _, _, _, total = match_edge_strips(s, s)
        assert total == pytest.approx(1.0, abs=0.02)

    def test_ncc_weight_only(self):
        s = _fstrip(seed=0)
        ncc, _, _, total = match_edge_strips(s, s, weights=(1.0, 0.0, 0.0))
        assert total == pytest.approx(ncc, abs=0.01)

    def test_ssd_weight_only(self):
        s = _fstrip(seed=0)
        _, ssd, _, total = match_edge_strips(s, s, weights=(0.0, 1.0, 0.0))
        assert total == pytest.approx(ssd, abs=0.01)

    def test_returns_4_values(self):
        s1 = _fstrip(seed=0)
        s2 = _fstrip(seed=1)
        result = match_edge_strips(s1, s2)
        assert len(result) == 4

    def test_different_size_no_crash(self):
        s1 = _fstrip(h=5, w=20)
        s2 = _fstrip(h=8, w=40)
        r = match_edge_strips(s1, s2)
        assert 0.0 <= r[3] <= 1.0


# ─── TestMatchPatchPairParamsExtra ────────────────────────────────────────────

class TestMatchPatchPairParamsExtra:
    def test_gradient_high_score(self):
        img = _gradient()
        pm = match_patch_pair(img, img.copy(), side1=2, side2=0, border_px=8)
        assert pm.total_score > 0.8

    def test_border_px_15_stored(self):
        img1 = _solid(100)
        img2 = _solid(100)
        pm = match_patch_pair(img1, img2, border_px=15)
        assert pm.params["border_px"] == 15

    def test_weights_stored(self):
        img1 = _solid(100)
        img2 = _solid(100)
        pm = match_patch_pair(img1, img2, weights=(1.0, 0.0, 0.0))
        assert pm.params["weights"] == (1.0, 0.0, 0.0)

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides_return_float_score(self, side):
        img1 = _gradient()
        img2 = _gradient()
        pm = match_patch_pair(img1, img2, side1=side, side2=side, border_px=5)
        assert isinstance(pm.total_score, float)

    def test_bgr_images(self):
        img1 = _bgr(64, 64, 150)
        img2 = _bgr(64, 64, 150)
        pm = match_patch_pair(img1, img2)
        assert isinstance(pm, PatchMatch)

    def test_idx_stored(self):
        img1 = _solid(100)
        img2 = _solid(150)
        pm = match_patch_pair(img1, img2, idx1=3, idx2=7)
        assert pm.idx1 == 3
        assert pm.idx2 == 7


# ─── TestBatchPatchMatchParamsExtra ──────────────────────────────────────────

class TestBatchPatchMatchParamsExtra:
    def test_empty_pairs(self):
        result = batch_patch_match([_solid(100)], [])
        assert result == []

    def test_length_matches(self):
        imgs = [_solid(i * 30) for i in range(4)]
        pairs = [(0, 1), (1, 2), (2, 3)]
        result = batch_patch_match(imgs, pairs)
        assert len(result) == 3

    def test_all_results_patch_match(self):
        imgs = [_gradient() for _ in range(3)]
        for r in batch_patch_match(imgs, [(0, 1), (1, 2)]):
            assert isinstance(r, PatchMatch)

    def test_indices_correct(self):
        imgs = [_solid(i * 50) for i in range(4)]
        result = batch_patch_match(imgs, [(0, 3)])
        assert result[0].idx1 == 0
        assert result[0].idx2 == 3

    def test_custom_side_pairs(self):
        imgs = [_gradient(), _gradient()]
        result = batch_patch_match(imgs, [(0, 1)], side_pairs=[(1, 3)])
        assert result[0].side1 == 1
        assert result[0].side2 == 3

    def test_default_side_2_0(self):
        imgs = [_solid(100), _solid(100)]
        result = batch_patch_match(imgs, [(0, 1)])
        assert result[0].side1 == 2
        assert result[0].side2 == 0

    def test_kwargs_forwarded_to_match(self):
        imgs = [_gradient(), _gradient()]
        result = batch_patch_match(imgs, [(0, 1)], border_px=12)
        assert result[0].params["border_px"] == 12

    def test_scores_in_range(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        for pm in batch_patch_match(imgs, [(0, 1), (0, 2), (1, 2)]):
            assert 0.0 <= pm.total_score <= 1.0
