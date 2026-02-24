"""Extra tests for puzzle_reconstruction/algorithms/patch_matcher.py"""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.patch_matcher import (
    PatchConfig,
    PatchMatch,
    batch_patch_match,
    extract_patch,
    find_matches,
    match_patch_in_image,
    ncc_score,
    sad_score,
    ssd_score,
    top_matches,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _img(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _rgb(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


# ─── TestPatchConfigExtra ─────────────────────────────────────────────────────

class TestPatchConfigExtra:
    def test_patch_size_3_valid(self):
        cfg = PatchConfig(patch_size=3)
        assert cfg.patch_size == 3

    def test_patch_size_5(self):
        cfg = PatchConfig(patch_size=5)
        assert cfg.patch_size == 5

    def test_stride_1_valid(self):
        cfg = PatchConfig(stride=1)
        assert cfg.stride == 1

    def test_max_matches_100(self):
        cfg = PatchConfig(max_matches=100)
        assert cfg.max_matches == 100

    def test_method_ncc_valid(self):
        cfg = PatchConfig(method="ncc")
        assert cfg.method == "ncc"

    def test_method_ssd_valid(self):
        cfg = PatchConfig(method="ssd")
        assert cfg.method == "ssd"

    def test_method_sad_valid(self):
        cfg = PatchConfig(method="sad")
        assert cfg.method == "sad"

    def test_patch_size_negative_raises(self):
        with pytest.raises(ValueError):
            PatchConfig(patch_size=-1)


# ─── TestPatchMatchExtra ──────────────────────────────────────────────────────

class TestPatchMatchExtra:
    def test_zero_coords_valid(self):
        pm = PatchMatch(row1=0, col1=0, row2=0, col2=0, score=0.0)
        assert pm.row1 == 0

    def test_large_coords_valid(self):
        pm = PatchMatch(row1=1000, col1=1000, row2=999, col2=999, score=0.5)
        assert pm.row1 == 1000

    def test_method_ssd_stored(self):
        pm = PatchMatch(row1=0, col1=0, row2=0, col2=0, score=5.0, method="ssd")
        assert pm.method == "ssd"

    def test_method_sad_stored(self):
        pm = PatchMatch(row1=0, col1=0, row2=0, col2=0, score=3.0, method="sad")
        assert pm.method == "sad"

    def test_src_pos_matches_row_col(self):
        pm = PatchMatch(row1=5, col1=8, row2=0, col2=0, score=0.0)
        assert pm.src_pos == (5, 8)

    def test_dst_pos_matches_row2_col2(self):
        pm = PatchMatch(row1=0, col1=0, row2=3, col2=9, score=0.0)
        assert pm.dst_pos == (3, 9)

    def test_negative_row2_raises(self):
        with pytest.raises(ValueError):
            PatchMatch(row1=0, col1=0, row2=-1, col2=0, score=0.0)


# ─── TestExtractPatchExtra ────────────────────────────────────────────────────

class TestExtractPatchExtra:
    def test_patch_size_5_at_center(self):
        img = _img(32, 32)
        p = extract_patch(img, 10, 10, 5)
        assert p.shape == (5, 5)

    def test_patch_size_9(self):
        img = _img(32, 32)
        p = extract_patch(img, 0, 0, 9)
        assert p.shape == (9, 9)

    def test_patch_size_3_various_positions(self):
        img = _img(32, 32)
        for r, c in ((0, 0), (0, 29), (29, 0), (15, 15)):
            p = extract_patch(img, r, c, 3)
            assert p.shape == (3, 3)

    def test_patch_values_correct_3x3(self):
        img = np.zeros((20, 20), dtype=np.uint8)
        img[5:8, 5:8] = 100
        p = extract_patch(img, 5, 5, 3)
        assert np.all(p == 100)

    def test_float32_type(self):
        img = _img(32, 32)
        p = extract_patch(img, 10, 10, 7)
        assert p.dtype == np.float32

    def test_rgb_returns_2d(self):
        img = _rgb(32, 32)
        p = extract_patch(img, 0, 0, 5)
        assert p.ndim == 2


# ─── TestNccScoreExtra ────────────────────────────────────────────────────────

class TestNccScoreExtra:
    def test_five_random_pairs_in_range(self):
        for s in range(5):
            p1 = _img(7, 7, seed=s).astype(np.float32)
            p2 = _img(7, 7, seed=s + 10).astype(np.float32)
            score = ncc_score(p1, p2)
            assert -1.0 <= score <= 1.0

    def test_near_identical_high_score(self):
        p = _img(7, 7, seed=5).astype(np.float32) + 1.0
        noise = p + np.random.default_rng(1).random((7, 7)).astype(np.float32) * 0.01
        score = ncc_score(p, noise)
        assert score > 0.9

    def test_3x3_patches_valid(self):
        p1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        p2 = p1 * 2
        score = ncc_score(p1, p2)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_opposite_patches_score_minus_1(self):
        p1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        p2 = -p1
        score = ncc_score(p1, p2)
        assert score == pytest.approx(-1.0, abs=1e-5)


# ─── TestSsdScoreExtra ────────────────────────────────────────────────────────

class TestSsdScoreExtra:
    def test_five_pairs_nonneg(self):
        for s in range(5):
            p1 = _img(5, 5, seed=s).astype(np.float32)
            p2 = _img(5, 5, seed=s + 5).astype(np.float32)
            assert ssd_score(p1, p2) >= 0.0

    def test_scaled_patch(self):
        p1 = np.ones((5, 5), dtype=np.float32)
        p2 = np.full((5, 5), 2.0, dtype=np.float32)
        # (1-2)^2 * 25 = 25
        assert ssd_score(p1, p2) == pytest.approx(25.0)

    def test_mismatched_shapes_raises(self):
        with pytest.raises(ValueError):
            ssd_score(np.ones((5, 5)), np.ones((6, 6)))

    def test_symmetric(self):
        p1 = _img(5, 5, seed=1).astype(np.float32)
        p2 = _img(5, 5, seed=2).astype(np.float32)
        assert ssd_score(p1, p2) == pytest.approx(ssd_score(p2, p1))


# ─── TestSadScoreExtra ────────────────────────────────────────────────────────

class TestSadScoreExtra:
    def test_five_pairs_nonneg(self):
        for s in range(5):
            p1 = _img(5, 5, seed=s).astype(np.float32)
            p2 = _img(5, 5, seed=s + 5).astype(np.float32)
            assert sad_score(p1, p2) >= 0.0

    def test_known_value_all_ones(self):
        p1 = np.zeros((4, 4), dtype=np.float32)
        p2 = np.ones((4, 4), dtype=np.float32)
        assert sad_score(p1, p2) == pytest.approx(16.0)

    def test_symmetric(self):
        p1 = _img(5, 5, seed=3).astype(np.float32)
        p2 = _img(5, 5, seed=4).astype(np.float32)
        assert sad_score(p1, p2) == pytest.approx(sad_score(p2, p1))

    def test_large_patch(self):
        p1 = _img(17, 17, seed=0).astype(np.float32)
        p2 = _img(17, 17, seed=1).astype(np.float32)
        assert sad_score(p1, p2) >= 0.0


# ─── TestMatchPatchInImageExtra ───────────────────────────────────────────────

class TestMatchPatchInImageExtra:
    def test_stride_1_vs_4_same_bounds(self):
        tmpl = _img(5, 5)
        img = _img(32, 32)
        r1, c1, _ = match_patch_in_image(tmpl, img, method="ncc", stride=1)
        r4, c4, _ = match_patch_in_image(tmpl, img, method="ncc", stride=4)
        assert 0 <= r1 <= 32 - 5
        assert 0 <= r4 <= 32 - 5

    def test_ssd_best_match_zero_ssd(self):
        img = _img(32, 32)
        r0, c0, ps = 4, 6, 5
        tmpl = img[r0:r0 + ps, c0:c0 + ps]
        _, _, score = match_patch_in_image(tmpl, img, method="ssd", stride=1)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_sad_method_bounds(self):
        tmpl = _img(5, 5)
        img = _img(32, 32)
        r, c, s = match_patch_in_image(tmpl, img, method="sad", stride=2)
        assert s >= 0.0

    def test_various_template_sizes(self):
        img = _img(64, 64)
        for ps in (3, 5, 7):
            tmpl = _img(ps, ps)
            r, c, _ = match_patch_in_image(tmpl, img, method="ncc", stride=4)
            assert 0 <= r <= 64 - ps
            assert 0 <= c <= 64 - ps


# ─── TestFindMatchesExtra ─────────────────────────────────────────────────────

class TestFindMatchesExtra:
    def test_ssd_method(self):
        img1 = _img(32, 32)
        img2 = _img(32, 32, seed=1)
        result = find_matches(img1, img2, PatchConfig(patch_size=9, stride=8, method="ssd"))
        assert isinstance(result, list)

    def test_sad_method(self):
        img1 = _img(32, 32)
        img2 = _img(32, 32, seed=2)
        result = find_matches(img1, img2, PatchConfig(patch_size=9, stride=8, method="sad"))
        assert isinstance(result, list)

    def test_rgb_images_no_crash(self):
        img1 = _rgb(32, 32)
        img2 = _rgb(32, 32, seed=3)
        result = find_matches(img1, img2, PatchConfig(patch_size=9, stride=8))
        assert isinstance(result, list)

    def test_results_are_patch_matches(self):
        img1 = _img(32, 32)
        img2 = _img(32, 32, seed=4)
        result = find_matches(img1, img2, PatchConfig(patch_size=9, stride=8))
        for m in result:
            assert isinstance(m, PatchMatch)

    def test_large_stride_fewer_matches(self):
        img1 = _img(64, 64)
        img2 = _img(64, 64, seed=5)
        cfg_small = PatchConfig(patch_size=9, stride=4, max_matches=200)
        cfg_large = PatchConfig(patch_size=9, stride=16, max_matches=200)
        result_small = find_matches(img1, img2, cfg_small)
        result_large = find_matches(img1, img2, cfg_large)
        assert len(result_large) <= len(result_small)


# ─── TestTopMatchesExtra ──────────────────────────────────────────────────────

class TestTopMatchesExtra:
    def _make_matches(self, scores, method="ncc"):
        return [
            PatchMatch(row1=0, col1=i, row2=0, col2=i, score=s, method=method)
            for i, s in enumerate(scores)
        ]

    def test_sad_sorted_ascending(self):
        matches = self._make_matches([10.0, 5.0, 20.0, 1.0], method="sad")
        result = top_matches(matches, k=4, method="sad")
        scores = [m.score for m in result]
        assert scores == sorted(scores)

    def test_all_same_scores(self):
        matches = self._make_matches([0.5, 0.5, 0.5])
        result = top_matches(matches, k=3)
        assert len(result) == 3

    def test_k_1_returns_single(self):
        matches = self._make_matches([0.3, 0.9, 0.1])
        result = top_matches(matches, k=1)
        assert len(result) == 1

    def test_ncc_top_is_highest(self):
        matches = self._make_matches([0.3, 0.9, 0.1, 0.7])
        result = top_matches(matches, k=1)
        assert result[0].score == pytest.approx(0.9)

    def test_ssd_top_is_lowest(self):
        matches = self._make_matches([30.0, 5.0, 15.0], method="ssd")
        result = top_matches(matches, k=1, method="ssd")
        assert result[0].score == pytest.approx(5.0)


# ─── TestBatchPatchMatchExtra ─────────────────────────────────────────────────

class TestBatchPatchMatchExtra:
    def test_three_pairs(self):
        pairs = [((_img(32, 32, s), _img(32, 32, s + 1))) for s in range(3)]
        result = batch_patch_match(pairs, PatchConfig(patch_size=9, stride=8))
        assert len(result) == 3

    def test_rgb_pairs(self):
        pairs = [(_rgb(32, 32, s), _rgb(32, 32, s + 1)) for s in range(2)]
        result = batch_patch_match(pairs, PatchConfig(patch_size=9, stride=8))
        assert len(result) == 2

    def test_inner_lists_are_lists(self):
        pairs = [(_img(32, 32), _img(32, 32, 1)), (_img(32, 32, 2), _img(32, 32, 3))]
        result = batch_patch_match(pairs, PatchConfig(patch_size=9, stride=8))
        for inner in result:
            assert isinstance(inner, list)

    def test_ssd_cfg(self):
        pairs = [(_img(32, 32), _img(32, 32, 1))]
        result = batch_patch_match(pairs, PatchConfig(patch_size=9, stride=8, method="ssd"))
        assert isinstance(result, list)
