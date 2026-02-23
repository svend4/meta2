"""Extra tests for puzzle_reconstruction.matching.patch_matcher."""
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

def _gray(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _noisy(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=64, w=64, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _fstrip(h=8, w=32, seed=0):
    return np.random.default_rng(seed).random((h, w)).astype(np.float32) * 200.0


# ─── TestPatchMatchExtra ─────────────────────────────────────────────────────

class TestPatchMatchExtra:
    def test_total_score_stored(self):
        pm = PatchMatch(0, 1, 0, 2, 0.7, 0.8, 0.9, 0.85)
        assert pm.total_score == pytest.approx(0.85)

    def test_ncc_can_be_negative(self):
        pm = PatchMatch(0, 1, 0, 2, -0.3, 0.5, 0.5, 0.4)
        assert pm.ncc < 0.0

    def test_params_custom_key(self):
        pm = PatchMatch(0, 1, 0, 2, 0.5, 0.5, 0.5, 0.5,
                        params={"weights": (1.0, 0.0, 0.0)})
        assert pm.params["weights"][0] == pytest.approx(1.0)

    def test_side_values_3_1(self):
        pm = PatchMatch(0, 1, 3, 1, 0.5, 0.5, 0.5, 0.5)
        assert pm.side1 == 3
        assert pm.side2 == 1

    def test_large_indices(self):
        pm = PatchMatch(999, 1000, 0, 2, 0.5, 0.5, 0.5, 0.5)
        assert pm.idx1 == 999
        assert pm.idx2 == 1000

    def test_zero_scores(self):
        pm = PatchMatch(0, 1, 0, 2, 0.0, 0.0, 0.0, 0.0)
        assert pm.total_score == pytest.approx(0.0)

    def test_one_scores(self):
        pm = PatchMatch(0, 1, 0, 2, 1.0, 1.0, 1.0, 1.0)
        assert pm.total_score == pytest.approx(1.0)


# ─── TestExtractEdgeStripExtra ────────────────────────────────────────────────

class TestExtractEdgeStripExtra:
    def test_border_px_1(self):
        img = _gray(32, 32)
        strip = extract_edge_strip(img, side=0, border_px=1)
        assert strip.shape == (1, 32)

    def test_right_strip_values(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[:, -6:] = 77
        strip = extract_edge_strip(img, side=1, border_px=6)
        assert float(strip.mean()) == pytest.approx(77.0)

    def test_left_strip_values(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[:, :4] = 55
        strip = extract_edge_strip(img, side=3, border_px=4)
        assert float(strip.mean()) == pytest.approx(55.0)

    def test_bgr_converted_to_2d(self):
        img = _bgr(64, 80, val=200)
        strip = extract_edge_strip(img, side=2, border_px=5)
        assert strip.ndim == 2

    def test_float32_all_sides(self):
        img = _gray()
        for s in range(4):
            strip = extract_edge_strip(img, side=s, border_px=4)
            assert strip.dtype == np.float32

    def test_side_5_raises(self):
        with pytest.raises(ValueError):
            extract_edge_strip(_gray(), side=5)

    def test_side_negative_raises(self):
        with pytest.raises(ValueError):
            extract_edge_strip(_gray(), side=-2)

    def test_bottom_strip_values(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[-8:, :] = 123
        strip = extract_edge_strip(img, side=2, border_px=8)
        assert float(strip.mean()) == pytest.approx(123.0)


# ─── TestNccScoreExtra ────────────────────────────────────────────────────────

class TestNccScoreExtra:
    def test_inverted_near_minus_one(self):
        s = _fstrip(seed=0)
        si = 255.0 - s
        assert ncc_score(s, si) == pytest.approx(-1.0, abs=0.05)

    def test_different_in_range(self):
        s1 = _fstrip(seed=0)
        s2 = _fstrip(seed=1)
        assert -1.0 <= ncc_score(s1, s2) <= 1.0

    def test_returns_float(self):
        s1 = _fstrip(seed=2)
        s2 = _fstrip(seed=3)
        assert isinstance(ncc_score(s1, s2), float)

    def test_shift_invariant_near_one(self):
        s = _fstrip(seed=5)
        s2 = s + 50.0
        assert ncc_score(s, s2) == pytest.approx(1.0, abs=1e-4)

    def test_empty_returns_zero(self):
        s = np.empty((0, 5), dtype=np.float32)
        assert ncc_score(s, s) == pytest.approx(0.0)


# ─── TestSsdScoreExtra ────────────────────────────────────────────────────────

class TestSsdScoreExtra:
    def test_identical_is_one(self):
        s = _fstrip(seed=0)
        assert ssd_score(s, s) == pytest.approx(1.0)

    def test_monotone_with_bigger_diff(self):
        base = np.full((8, 32), 100.0, dtype=np.float32)
        r1 = ssd_score(base, base + 10)
        r2 = ssd_score(base, base + 100)
        assert r1 > r2

    def test_in_range(self):
        s1 = _fstrip(seed=10)
        s2 = _fstrip(seed=11)
        assert 0.0 <= ssd_score(s1, s2) <= 1.0

    def test_returns_float(self):
        s1 = _fstrip(seed=4)
        s2 = _fstrip(seed=5)
        assert isinstance(ssd_score(s1, s2), float)

    def test_zero_vs_max_low(self):
        s1 = np.zeros((8, 32), dtype=np.float32)
        s2 = np.full((8, 32), 255.0, dtype=np.float32)
        assert ssd_score(s1, s2) <= 0.5


# ─── TestSsimScoreExtra ───────────────────────────────────────────────────────

class TestSsimScoreExtra:
    def test_identical_near_one(self):
        s = _fstrip(seed=7)
        assert ssim_score(s, s) == pytest.approx(1.0, abs=1e-4)

    def test_result_in_0_1(self):
        s1 = _fstrip(seed=8)
        s2 = _fstrip(seed=9)
        assert 0.0 <= ssim_score(s1, s2) <= 1.0

    def test_returns_float(self):
        s1 = _fstrip(seed=12)
        s2 = _fstrip(seed=13)
        assert isinstance(ssim_score(s1, s2), float)

    def test_empty_returns_zero(self):
        s = np.empty((0, 5), dtype=np.float32)
        assert ssim_score(s, s) == pytest.approx(0.0)


# ─── TestMatchEdgeStripsExtra ────────────────────────────────────────────────

class TestMatchEdgeStripsExtra:
    def test_identical_total_high(self):
        s = _fstrip(seed=0)
        _, _, _, total = match_edge_strips(s, s)
        assert total > 0.8

    def test_total_in_range(self):
        s1 = _fstrip(seed=0)
        s2 = _fstrip(seed=1)
        _, _, _, total = match_edge_strips(s1, s2)
        assert 0.0 <= total <= 1.0

    def test_ncc_component_in_range(self):
        s1 = _fstrip(seed=2)
        s2 = _fstrip(seed=3)
        ncc, _, _, _ = match_edge_strips(s1, s2)
        assert -1.0 <= ncc <= 1.0

    def test_ssd_component_in_range(self):
        s1 = _fstrip(seed=4)
        s2 = _fstrip(seed=5)
        _, ssd, _, _ = match_edge_strips(s1, s2)
        assert 0.0 <= ssd <= 1.0

    def test_ssim_component_in_range(self):
        s1 = _fstrip(seed=6)
        s2 = _fstrip(seed=7)
        _, _, ssim, _ = match_edge_strips(s1, s2)
        assert 0.0 <= ssim <= 1.0

    def test_mismatched_size_no_crash(self):
        s1 = _fstrip(h=5, w=20)
        s2 = _fstrip(h=10, w=40)
        ncc, ssd, ssim, total = match_edge_strips(s1, s2)
        assert 0.0 <= total <= 1.0

    def test_custom_weights_total_in_range(self):
        s1 = _fstrip(seed=0)
        s2 = _fstrip(seed=1)
        _, _, _, total = match_edge_strips(s1, s2, weights=(2.0, 1.0, 1.0))
        assert 0.0 <= total <= 1.0


# ─── TestMatchPatchPairExtra ──────────────────────────────────────────────────

class TestMatchPatchPairExtra:
    def test_ncc_in_minus1_1(self):
        img1 = _noisy(seed=0)
        img2 = _noisy(seed=1)
        pm = match_patch_pair(img1, img2)
        assert -1.0 <= pm.ncc <= 1.0

    def test_ssim_in_0_1(self):
        img1 = _noisy(seed=2)
        img2 = _noisy(seed=3)
        pm = match_patch_pair(img1, img2)
        assert 0.0 <= pm.ssim <= 1.0

    def test_border_px_in_params(self):
        pm = match_patch_pair(_noisy(seed=4), _noisy(seed=5))
        assert "border_px" in pm.params

    def test_identical_high_total(self):
        img = _noisy(seed=6)
        pm = match_patch_pair(img, img, side1=2, side2=2)
        assert pm.total_score > 0.8

    def test_sides_0_2(self):
        pm = match_patch_pair(_noisy(seed=0), _noisy(seed=1), side1=0, side2=2)
        assert pm.side1 == 0 and pm.side2 == 2

    def test_sides_1_3(self):
        pm = match_patch_pair(_noisy(seed=0), _noisy(seed=1), side1=1, side2=3)
        assert pm.side1 == 1 and pm.side2 == 3

    def test_bgr_no_crash(self):
        pm = match_patch_pair(_bgr(64, 64, 100), _bgr(64, 64, 150))
        assert isinstance(pm, PatchMatch)

    def test_weights_in_params(self):
        img = _noisy(seed=0)
        pm = match_patch_pair(img, img, weights=(0.5, 0.3, 0.2))
        assert pm.params.get("weights") == (0.5, 0.3, 0.2)

    def test_ssd_in_0_1(self):
        pm = match_patch_pair(_noisy(seed=7), _noisy(seed=8))
        assert 0.0 <= pm.ssd <= 1.0


# ─── TestBatchPatchMatchExtra ────────────────────────────────────────────────

class TestBatchPatchMatchExtra:
    def test_four_pairs(self):
        imgs = [_noisy(seed=i) for i in range(5)]
        pairs = [(i, i + 1) for i in range(4)]
        result = batch_patch_match(imgs, pairs)
        assert len(result) == 4

    def test_all_patch_match_instances(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        for pm in batch_patch_match(imgs, [(0, 1), (1, 2)]):
            assert isinstance(pm, PatchMatch)

    def test_scores_in_range(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        for pm in batch_patch_match(imgs, [(0, 1), (0, 2), (1, 2)]):
            assert 0.0 <= pm.total_score <= 1.0

    def test_multiple_side_pairs(self):
        imgs = [_noisy(seed=i) for i in range(4)]
        result = batch_patch_match(imgs, [(0, 1), (2, 3)],
                                   side_pairs=[(0, 2), (1, 3)])
        assert result[0].side1 == 0
        assert result[1].side1 == 1

    def test_border_px_forwarded(self):
        imgs = [_noisy(seed=i) for i in range(2)]
        result = batch_patch_match(imgs, [(0, 1)], border_px=7)
        assert result[0].params["border_px"] == 7

    def test_idx_stored_correctly(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        result = batch_patch_match(imgs, [(0, 2)])
        assert result[0].idx1 == 0
        assert result[0].idx2 == 2
