"""Тесты для puzzle_reconstruction.algorithms.patch_matcher."""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.patch_matcher import (
    PatchConfig,
    PatchMatch,
    extract_patch,
    ncc_score,
    ssd_score,
    sad_score,
    match_patch_in_image,
    find_matches,
    top_matches,
    batch_patch_match,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _img(h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _rgb(h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ─── TestPatchConfig ──────────────────────────────────────────────────────────

class TestPatchConfig:
    def test_defaults(self):
        cfg = PatchConfig()
        assert cfg.patch_size == 17
        assert cfg.stride == 4
        assert cfg.method == "ncc"
        assert cfg.max_matches == 50

    def test_patch_size_below_3_raises(self):
        with pytest.raises(ValueError):
            PatchConfig(patch_size=2)

    def test_patch_size_even_raises(self):
        with pytest.raises(ValueError):
            PatchConfig(patch_size=4)

    def test_stride_zero_raises(self):
        with pytest.raises(ValueError):
            PatchConfig(stride=0)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            PatchConfig(method="cosine")

    def test_max_matches_zero_raises(self):
        with pytest.raises(ValueError):
            PatchConfig(max_matches=0)

    def test_valid_ssd(self):
        cfg = PatchConfig(patch_size=5, stride=2, method="ssd")
        assert cfg.method == "ssd"

    def test_valid_sad(self):
        cfg = PatchConfig(patch_size=7, method="sad")
        assert cfg.method == "sad"


# ─── TestPatchMatch ───────────────────────────────────────────────────────────

class TestPatchMatch:
    def test_basic_creation(self):
        pm = PatchMatch(row1=0, col1=0, row2=5, col2=5, score=0.9)
        assert pm.score == pytest.approx(0.9)

    def test_src_pos(self):
        pm = PatchMatch(row1=3, col1=7, row2=0, col2=0, score=0.0)
        assert pm.src_pos == (3, 7)

    def test_dst_pos(self):
        pm = PatchMatch(row1=0, col1=0, row2=4, col2=6, score=0.0)
        assert pm.dst_pos == (4, 6)

    def test_negative_row1_raises(self):
        with pytest.raises(ValueError):
            PatchMatch(row1=-1, col1=0, row2=0, col2=0, score=0.0)

    def test_negative_col2_raises(self):
        with pytest.raises(ValueError):
            PatchMatch(row1=0, col1=0, row2=0, col2=-1, score=0.0)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            PatchMatch(row1=0, col1=0, row2=0, col2=0, score=0.0, method="ncc2")

    def test_default_method_ncc(self):
        pm = PatchMatch(row1=0, col1=0, row2=0, col2=0, score=0.0)
        assert pm.method == "ncc"


# ─── TestExtractPatch ─────────────────────────────────────────────────────────

class TestExtractPatch:
    def test_shape(self):
        img = _img(32, 32)
        p = extract_patch(img, 0, 0, 7)
        assert p.shape == (7, 7)

    def test_dtype_float32(self):
        img = _img(32, 32)
        p = extract_patch(img, 0, 0, 5)
        assert p.dtype == np.float32

    def test_even_patch_size_raises(self):
        img = _img(32, 32)
        with pytest.raises(ValueError):
            extract_patch(img, 0, 0, 4)

    def test_patch_size_below_3_raises(self):
        img = _img(32, 32)
        with pytest.raises(ValueError):
            extract_patch(img, 0, 0, 1)

    def test_oob_row_raises(self):
        img = _img(10, 10)
        with pytest.raises(ValueError):
            extract_patch(img, 8, 0, 5)  # row+5=13 > 10

    def test_oob_col_raises(self):
        img = _img(10, 10)
        with pytest.raises(ValueError):
            extract_patch(img, 0, 8, 5)

    def test_rgb_image(self):
        img = _rgb(32, 32)
        p = extract_patch(img, 0, 0, 5)
        assert p.shape == (5, 5)

    def test_correct_values(self):
        img = np.zeros((20, 20), dtype=np.uint8)
        img[2:5, 3:6] = 200
        p = extract_patch(img, 2, 3, 3)
        assert np.all(p == 200)


# ─── TestNccScore ─────────────────────────────────────────────────────────────

class TestNccScore:
    def test_identical_patches_score_one(self):
        p = _img(7, 7).astype(np.float32) + 1.0
        assert ncc_score(p, p) == pytest.approx(1.0, abs=1e-6)

    def test_range(self):
        p1 = _img(7, 7).astype(np.float32)
        p2 = _img(7, 7, seed=1).astype(np.float32)
        s = ncc_score(p1, p2)
        assert -1.0 <= s <= 1.0

    def test_shape_mismatch_raises(self):
        p1 = np.ones((5, 5), dtype=np.float32)
        p2 = np.ones((7, 7), dtype=np.float32)
        with pytest.raises(ValueError):
            ncc_score(p1, p2)

    def test_constant_patch_returns_zero(self):
        p1 = np.ones((5, 5), dtype=np.float32)
        p2 = _img(5, 5).astype(np.float32)
        assert ncc_score(p1, p2) == pytest.approx(0.0, abs=1e-6)

    def test_returns_float(self):
        p = _img(5, 5).astype(np.float32)
        assert isinstance(ncc_score(p, p), float)


# ─── TestSsdScore ─────────────────────────────────────────────────────────────

class TestSsdScore:
    def test_identical_patches_zero(self):
        p = _img(5, 5).astype(np.float32)
        assert ssd_score(p, p) == pytest.approx(0.0, abs=1e-6)

    def test_nonnegative(self):
        p1 = _img(5, 5).astype(np.float32)
        p2 = _img(5, 5, seed=1).astype(np.float32)
        assert ssd_score(p1, p2) >= 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            ssd_score(np.ones((3, 3)), np.ones((5, 5)))

    def test_known_value(self):
        p1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        p2 = np.array([[2.0, 3.0], [4.0, 5.0]])
        assert ssd_score(p1, p2) == pytest.approx(4.0)

    def test_returns_float(self):
        p = _img(5, 5).astype(np.float32)
        assert isinstance(ssd_score(p, p), float)


# ─── TestSadScore ─────────────────────────────────────────────────────────────

class TestSadScore:
    def test_identical_patches_zero(self):
        p = _img(5, 5).astype(np.float32)
        assert sad_score(p, p) == pytest.approx(0.0, abs=1e-6)

    def test_nonnegative(self):
        p1 = _img(5, 5).astype(np.float32)
        p2 = _img(5, 5, seed=2).astype(np.float32)
        assert sad_score(p1, p2) >= 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            sad_score(np.ones((3, 3)), np.ones((4, 4)))

    def test_known_value(self):
        p1 = np.array([[0.0, 1.0], [2.0, 3.0]])
        p2 = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert sad_score(p1, p2) == pytest.approx(4.0)

    def test_returns_float(self):
        p = _img(5, 5).astype(np.float32)
        assert isinstance(sad_score(p, p), float)


# ─── TestMatchPatchInImage ────────────────────────────────────────────────────

class TestMatchPatchInImage:
    def test_returns_tuple(self):
        tmpl = _img(5, 5)
        img = _img(32, 32)
        result = match_patch_in_image(tmpl, img, method="ncc")
        assert len(result) == 3

    def test_row_col_within_bounds(self):
        tmpl = _img(5, 5)
        img = _img(32, 32)
        r, c, _ = match_patch_in_image(tmpl, img, method="ncc")
        assert 0 <= r <= 32 - 5
        assert 0 <= c <= 32 - 5

    def test_exact_match_ncc(self):
        img = _img(32, 32)
        r0, c0, ps = 5, 8, 7
        tmpl = img[r0:r0 + ps, c0:c0 + ps]
        br, bc, score = match_patch_in_image(tmpl, img, method="ncc", stride=1)
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_ssd_method(self):
        tmpl = _img(5, 5)
        img = _img(32, 32)
        r, c, s = match_patch_in_image(tmpl, img, method="ssd")
        assert s >= 0.0

    def test_sad_method(self):
        tmpl = _img(5, 5)
        img = _img(32, 32)
        r, c, s = match_patch_in_image(tmpl, img, method="sad")
        assert s >= 0.0

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            match_patch_in_image(_img(5, 5), _img(32, 32), method="bad")

    def test_stride_zero_raises(self):
        with pytest.raises(ValueError):
            match_patch_in_image(_img(5, 5), _img(32, 32), stride=0)

    def test_template_larger_than_image_raises(self):
        with pytest.raises(ValueError):
            match_patch_in_image(_img(40, 40), _img(32, 32))


# ─── TestFindMatches ──────────────────────────────────────────────────────────

class TestFindMatches:
    def test_returns_list(self):
        img1 = _img(32, 32)
        img2 = _img(32, 32, seed=1)
        result = find_matches(img1, img2, PatchConfig(patch_size=9, stride=8))
        assert isinstance(result, list)

    def test_all_patch_matches(self):
        img1 = _img(32, 32)
        img2 = _img(32, 32, seed=1)
        result = find_matches(img1, img2, PatchConfig(patch_size=9, stride=8))
        assert all(isinstance(m, PatchMatch) for m in result)

    def test_default_config(self):
        img1 = _img(64, 64)
        img2 = _img(64, 64, seed=2)
        result = find_matches(img1, img2)
        assert isinstance(result, list)

    def test_rgb_images(self):
        img1 = _rgb(32, 32)
        img2 = _rgb(32, 32, seed=3)
        result = find_matches(img1, img2, PatchConfig(patch_size=9, stride=8))
        assert isinstance(result, list)

    def test_method_stored_in_matches(self):
        img1 = _img(32, 32)
        img2 = _img(32, 32, seed=1)
        cfg = PatchConfig(patch_size=9, stride=8, method="ssd")
        result = find_matches(img1, img2, cfg)
        if result:
            assert all(m.method == "ssd" for m in result)


# ─── TestTopMatches ───────────────────────────────────────────────────────────

class TestTopMatches:
    def _make_matches(self, scores, method="ncc"):
        return [
            PatchMatch(row1=0, col1=i, row2=0, col2=i, score=s, method=method)
            for i, s in enumerate(scores)
        ]

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            top_matches([], k=0)

    def test_empty_returns_empty(self):
        assert top_matches([], k=5) == []

    def test_ncc_sorted_descending(self):
        matches = self._make_matches([0.3, 0.9, 0.1, 0.7])
        result = top_matches(matches, k=4, method="ncc")
        scores = [m.score for m in result]
        assert scores == sorted(scores, reverse=True)

    def test_ssd_sorted_ascending(self):
        matches = self._make_matches([30.0, 5.0, 15.0], method="ssd")
        result = top_matches(matches, k=3, method="ssd")
        scores = [m.score for m in result]
        assert scores == sorted(scores)

    def test_k_limits_result(self):
        matches = self._make_matches([0.1, 0.5, 0.9, 0.3])
        result = top_matches(matches, k=2)
        assert len(result) == 2

    def test_k_larger_than_list(self):
        matches = self._make_matches([0.5, 0.8])
        result = top_matches(matches, k=10)
        assert len(result) == 2


# ─── TestBatchPatchMatch ──────────────────────────────────────────────────────

class TestBatchPatchMatch:
    def test_returns_list(self):
        pairs = [(_img(32, 32), _img(32, 32, seed=1))]
        result = batch_patch_match(pairs, PatchConfig(patch_size=9, stride=8))
        assert isinstance(result, list)

    def test_length_matches_pairs(self):
        pairs = [
            (_img(32, 32), _img(32, 32, seed=1)),
            (_img(32, 32, seed=2), _img(32, 32, seed=3)),
        ]
        result = batch_patch_match(pairs, PatchConfig(patch_size=9, stride=8))
        assert len(result) == 2

    def test_empty_pairs(self):
        result = batch_patch_match([])
        assert result == []

    def test_each_inner_list(self):
        pairs = [(_img(32, 32), _img(32, 32, seed=1))]
        result = batch_patch_match(pairs, PatchConfig(patch_size=9, stride=8))
        assert isinstance(result[0], list)
