"""Additional tests for puzzle_reconstruction.preprocessing.background_remover."""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.background_remover import (
    BackgroundRemovalResult,
    remove_background_thresh,
    remove_background_edges,
    remove_background_grabcut,
    auto_remove_background,
    batch_remove_background,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _white(h=64, w=64):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _black(h=64, w=64):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _fg_on_white(h=64, w=64, fg_color=0):
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    img[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = fg_color
    return img


def _fg_on_black(h=64, w=64, fg_val=200):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = fg_val
    return img


def _rand_bgr(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


# ─── TestBackgroundRemovalResultExtra ────────────────────────────────────────

class TestBackgroundRemovalResultExtra:
    def test_params_stored(self):
        fg = np.zeros((8, 8, 3), dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        res = BackgroundRemovalResult(foreground=fg, mask=mask,
                                      method="thresh",
                                      params={"bg_thresh": 240})
        assert res.params["bg_thresh"] == 240

    def test_foreground_dtype(self):
        fg = np.zeros((8, 8, 3), dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        res = BackgroundRemovalResult(foreground=fg, mask=mask, method="thresh")
        assert res.foreground.dtype == np.uint8

    def test_mask_dtype_uint8(self):
        fg = np.zeros((8, 8, 3), dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        res = BackgroundRemovalResult(foreground=fg, mask=mask, method="edges")
        assert res.mask.dtype == np.uint8

    def test_method_grabcut_stored(self):
        fg = np.zeros((8, 8, 3), dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        res = BackgroundRemovalResult(foreground=fg, mask=mask, method="grabcut")
        assert res.method == "grabcut"


# ─── TestRemoveBackgroundThreshExtra ─────────────────────────────────────────

class TestRemoveBackgroundThreshExtra:
    def test_black_bg_with_invert(self):
        img = _fg_on_black()
        res = remove_background_thresh(img, bg_thresh=10, invert=True)
        assert res.method == "thresh"
        assert res.foreground.shape == img.shape

    def test_bg_fill_0_black(self):
        img = _fg_on_white()
        res = remove_background_thresh(img, bg_thresh=240, bg_fill=0)
        # Corner pixels (background) should be 0
        assert int(res.foreground[0, 0, 0]) == 0

    def test_non_square_image(self):
        img = _fg_on_white(h=32, w=80)
        res = remove_background_thresh(img)
        assert res.foreground.shape == (32, 80, 3)
        assert res.mask.shape == (32, 80)

    def test_mask_values_binary(self):
        img = _fg_on_white()
        res = remove_background_thresh(img, bg_thresh=240)
        unique = np.unique(res.mask)
        assert set(unique).issubset({0, 255})

    def test_high_bg_thresh_all_fg(self):
        # bg_thresh=255: nothing qualifies as white-enough background
        img = _fg_on_white()
        res = remove_background_thresh(img, bg_thresh=255)
        assert res.foreground.shape == img.shape

    def test_foreground_values_in_range(self):
        img = _fg_on_white()
        res = remove_background_thresh(img, bg_thresh=240)
        assert res.foreground.min() >= 0
        assert res.foreground.max() <= 255


# ─── TestRemoveBackgroundEdgesExtra ───────────────────────────────────────────

class TestRemoveBackgroundEdgesExtra:
    def test_black_image_no_crash(self):
        res = remove_background_edges(_black())
        assert res.method == "edges"
        assert res.mask.shape == (64, 64)

    def test_high_thresholds(self):
        img = _fg_on_white()
        res = remove_background_edges(img, low_thresh=100, high_thresh=200)
        assert res.foreground.shape == img.shape

    def test_dilate_ksize_5(self):
        img = _fg_on_white()
        res = remove_background_edges(img, dilate_ksize=5)
        assert res.mask.shape == (64, 64)

    def test_non_square_image(self):
        img = _fg_on_white(h=32, w=96)
        res = remove_background_edges(img)
        assert res.mask.shape == (32, 96)
        assert res.foreground.shape == (32, 96, 3)

    def test_foreground_dtype_uint8(self):
        res = remove_background_edges(_fg_on_white())
        assert res.foreground.dtype == np.uint8

    def test_low_thresh_0(self):
        img = _fg_on_white()
        res = remove_background_edges(img, low_thresh=0, high_thresh=50)
        assert res.method == "edges"


# ─── TestRemoveBackgroundGrabcutExtra ────────────────────────────────────────

class TestRemoveBackgroundGrabcutExtra:
    def test_n_iter_1(self):
        img = _fg_on_white()
        res = remove_background_grabcut(img, n_iter=1)
        assert res.method == "grabcut"
        assert res.mask.shape == (64, 64)

    def test_n_iter_5(self):
        img = _fg_on_white()
        res = remove_background_grabcut(img, n_iter=5)
        assert isinstance(res, BackgroundRemovalResult)

    def test_large_margin(self):
        img = _fg_on_white()
        res = remove_background_grabcut(img, margin=15)
        assert res.foreground.shape == img.shape

    def test_non_square_image(self):
        img = _fg_on_white(h=48, w=80)
        res = remove_background_grabcut(img)
        assert res.mask.shape == (48, 80)
        assert res.foreground.shape == (48, 80, 3)

    def test_foreground_dtype_uint8(self):
        res = remove_background_grabcut(_fg_on_white())
        assert res.foreground.dtype == np.uint8


# ─── TestAutoRemoveBackgroundExtra ───────────────────────────────────────────

class TestAutoRemoveBackgroundExtra:
    def test_thresh_result_has_valid_mask(self):
        img = _fg_on_white()
        res = auto_remove_background(img, method="thresh", bg_thresh=240)
        assert res.mask.dtype == np.uint8
        assert res.mask.shape == img.shape[:2]

    def test_edges_result_foreground_shape(self):
        img = _fg_on_white(h=32, w=80)
        res = auto_remove_background(img, method="edges")
        assert res.foreground.shape == (32, 80, 3)

    def test_grabcut_result_method_name(self):
        img = _fg_on_white()
        res = auto_remove_background(img, method="grabcut")
        assert res.method == "grabcut"

    def test_thresh_kwargs_forwarded(self):
        img = _fg_on_white()
        res = auto_remove_background(img, method="thresh", bg_fill=42)
        assert res.params.get("bg_fill") == 42

    def test_edges_kwargs_forwarded(self):
        img = _fg_on_white()
        res = auto_remove_background(img, method="edges", dilate_ksize=5)
        assert res.params.get("dilate_ksize") == 5


# ─── TestBatchRemoveBackgroundExtra ──────────────────────────────────────────

class TestBatchRemoveBackgroundExtra:
    def test_grabcut_batch_method_stored(self):
        imgs = [_fg_on_white() for _ in range(2)]
        result = batch_remove_background(imgs, method="grabcut")
        for r in result:
            assert r.method == "grabcut"

    def test_all_foregrounds_uint8(self):
        imgs = [_fg_on_white() for _ in range(3)]
        result = batch_remove_background(imgs, method="thresh")
        for r in result:
            assert r.foreground.dtype == np.uint8

    def test_mixed_sizes(self):
        imgs = [_fg_on_white(32, 64), _fg_on_white(64, 32)]
        result = batch_remove_background(imgs, method="thresh")
        assert result[0].foreground.shape == (32, 64, 3)
        assert result[1].foreground.shape == (64, 32, 3)

    def test_edges_batch_length(self):
        imgs = [_fg_on_white() for _ in range(5)]
        result = batch_remove_background(imgs, method="edges")
        assert len(result) == 5

    def test_mask_shapes_match_inputs(self):
        imgs = [_fg_on_white(40, 60), _fg_on_white(60, 40)]
        result = batch_remove_background(imgs, method="thresh")
        assert result[0].mask.shape == (40, 60)
        assert result[1].mask.shape == (60, 40)
