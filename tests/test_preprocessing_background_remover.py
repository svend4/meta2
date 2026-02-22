"""Расширенные тесты для puzzle_reconstruction/preprocessing/background_remover.py."""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.background_remover import (
    BackgroundRemovalResult,
    auto_remove_background,
    batch_remove_background,
    remove_background_edges,
    remove_background_grabcut,
    remove_background_thresh,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _bgr(h: int = 64, w: int = 64) -> np.ndarray:
    """White BGR image."""
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _gray(h: int = 64, w: int = 64) -> np.ndarray:
    """White grayscale image."""
    return np.full((h, w), 255, dtype=np.uint8)


def _dark_bgr(h: int = 64, w: int = 64) -> np.ndarray:
    """Dark BGR image."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _mixed_bgr(h: int = 64, w: int = 64) -> np.ndarray:
    """BGR with white background and dark object in center."""
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    img[h//4: 3*h//4, w//4: 3*w//4] = 50
    return img


# ─── TestBackgroundRemovalResult ──────────────────────────────────────────────

class TestBackgroundRemovalResult:
    def test_stores_foreground(self):
        fg = _bgr()
        mask = np.ones((64, 64), dtype=np.uint8) * 255
        r = BackgroundRemovalResult(foreground=fg, mask=mask, method="thresh")
        assert r.foreground is fg

    def test_stores_mask(self):
        fg = _bgr()
        mask = np.ones((64, 64), dtype=np.uint8) * 255
        r = BackgroundRemovalResult(foreground=fg, mask=mask, method="thresh")
        assert r.mask is mask

    def test_stores_method(self):
        r = BackgroundRemovalResult(foreground=_bgr(),
                                     mask=np.zeros((64, 64), dtype=np.uint8),
                                     method="edges")
        assert r.method == "edges"

    def test_default_params_empty_dict(self):
        r = BackgroundRemovalResult(foreground=_bgr(),
                                     mask=np.zeros((64, 64), dtype=np.uint8),
                                     method="thresh")
        assert isinstance(r.params, dict)

    def test_params_stored(self):
        r = BackgroundRemovalResult(foreground=_bgr(),
                                     mask=np.zeros((64, 64), dtype=np.uint8),
                                     method="thresh",
                                     params={"bg_thresh": 200})
        assert r.params["bg_thresh"] == 200

    def test_repr_contains_method(self):
        r = BackgroundRemovalResult(foreground=_bgr(),
                                     mask=np.zeros((64, 64), dtype=np.uint8),
                                     method="grabcut")
        assert "grabcut" in repr(r)

    def test_repr_contains_shape(self):
        r = BackgroundRemovalResult(foreground=_bgr(32, 48),
                                     mask=np.zeros((32, 48), dtype=np.uint8),
                                     method="thresh")
        text = repr(r)
        assert "32" in text or "48" in text

    def test_repr_contains_coverage(self):
        r = BackgroundRemovalResult(foreground=_bgr(),
                                     mask=np.zeros((64, 64), dtype=np.uint8),
                                     method="thresh")
        assert "fg_coverage" in repr(r)

    def test_repr_is_string(self):
        r = BackgroundRemovalResult(foreground=_bgr(),
                                     mask=np.zeros((64, 64), dtype=np.uint8),
                                     method="thresh")
        assert isinstance(repr(r), str)


# ─── TestRemoveBackgroundThresh ───────────────────────────────────────────────

class TestRemoveBackgroundThresh:
    def test_returns_result(self):
        result = remove_background_thresh(_bgr())
        assert isinstance(result, BackgroundRemovalResult)

    def test_method_is_thresh(self):
        result = remove_background_thresh(_bgr())
        assert result.method == "thresh"

    def test_mask_dtype_uint8(self):
        result = remove_background_thresh(_bgr())
        assert result.mask.dtype == np.uint8

    def test_mask_binary_values(self):
        result = remove_background_thresh(_bgr())
        unique = set(np.unique(result.mask))
        assert unique.issubset({0, 255})

    def test_foreground_same_shape(self):
        img = _bgr(32, 48)
        result = remove_background_thresh(img)
        assert result.foreground.shape == img.shape

    def test_mask_same_shape(self):
        img = _bgr(32, 48)
        result = remove_background_thresh(img)
        assert result.mask.shape == (32, 48)

    def test_params_stored(self):
        result = remove_background_thresh(_bgr(), bg_thresh=200)
        assert result.params["bg_thresh"] == 200

    def test_white_image_mostly_background(self):
        # White image → most pixels above threshold → background
        result = remove_background_thresh(_bgr(), bg_thresh=200)
        # Most mask values should be 0 (background)
        fg_ratio = float(result.mask.mean()) / 255.0
        assert fg_ratio < 0.5

    def test_dark_image_mostly_foreground(self):
        # Dark image → all pixels below threshold → foreground
        result = remove_background_thresh(_dark_bgr(), bg_thresh=200)
        fg_ratio = float(result.mask.mean()) / 255.0
        assert fg_ratio > 0.5

    def test_invert_flag(self):
        img = _dark_bgr()
        normal = remove_background_thresh(img, bg_thresh=128, invert=False)
        inverted = remove_background_thresh(img, bg_thresh=128, invert=True)
        # Inverted result should differ from normal
        assert not np.array_equal(normal.mask, inverted.mask)

    def test_bg_fill_applied(self):
        img = _bgr()
        result = remove_background_thresh(img, bg_thresh=200, bg_fill=128)
        # Background pixels replaced with bg_fill=128
        bg_pixels = result.foreground[result.mask == 0]
        if len(bg_pixels) > 0:
            assert np.all(bg_pixels == 128)

    def test_grayscale_input(self):
        result = remove_background_thresh(_gray())
        assert isinstance(result, BackgroundRemovalResult)
        assert result.mask.dtype == np.uint8

    def test_mixed_image(self):
        result = remove_background_thresh(_mixed_bgr(), bg_thresh=200)
        assert isinstance(result, BackgroundRemovalResult)


# ─── TestRemoveBackgroundEdges ────────────────────────────────────────────────

class TestRemoveBackgroundEdges:
    def test_returns_result(self):
        result = remove_background_edges(_bgr())
        assert isinstance(result, BackgroundRemovalResult)

    def test_method_is_edges(self):
        result = remove_background_edges(_bgr())
        assert result.method == "edges"

    def test_mask_dtype_uint8(self):
        result = remove_background_edges(_bgr())
        assert result.mask.dtype == np.uint8

    def test_mask_binary_values(self):
        result = remove_background_edges(_bgr())
        unique = set(np.unique(result.mask))
        assert unique.issubset({0, 255})

    def test_foreground_same_shape(self):
        img = _bgr(48, 64)
        result = remove_background_edges(img)
        assert result.foreground.shape == img.shape

    def test_mask_same_2d_shape(self):
        img = _bgr(48, 64)
        result = remove_background_edges(img)
        assert result.mask.shape == (48, 64)

    def test_params_stored(self):
        result = remove_background_edges(_bgr(), low_thresh=30, high_thresh=100)
        assert result.params["low_thresh"] == 30
        assert result.params["high_thresh"] == 100

    def test_grayscale_input(self):
        result = remove_background_edges(_gray())
        assert isinstance(result, BackgroundRemovalResult)

    def test_dilate_ksize_1(self):
        result = remove_background_edges(_bgr(), dilate_ksize=1)
        assert result.method == "edges"

    def test_bg_fill_applied(self):
        img = _bgr()
        result = remove_background_edges(img, bg_fill=100)
        bg_pixels = result.foreground[result.mask == 0]
        if len(bg_pixels) > 0:
            assert np.all(bg_pixels == 100)


# ─── TestRemoveBackgroundGrabcut ──────────────────────────────────────────────

class TestRemoveBackgroundGrabcut:
    def test_returns_result(self):
        result = remove_background_grabcut(_bgr())
        assert isinstance(result, BackgroundRemovalResult)

    def test_method_is_grabcut(self):
        result = remove_background_grabcut(_bgr())
        assert result.method == "grabcut"

    def test_mask_dtype_uint8(self):
        result = remove_background_grabcut(_bgr())
        assert result.mask.dtype == np.uint8

    def test_mask_binary_values(self):
        result = remove_background_grabcut(_bgr())
        unique = set(np.unique(result.mask))
        assert unique.issubset({0, 255})

    def test_foreground_same_shape_bgr(self):
        img = _bgr(64, 64)
        result = remove_background_grabcut(img)
        assert result.foreground.shape == img.shape

    def test_mask_2d_shape(self):
        img = _bgr(64, 64)
        result = remove_background_grabcut(img)
        assert result.mask.shape == (64, 64)

    def test_params_stored(self):
        result = remove_background_grabcut(_bgr(), margin=5, n_iter=3)
        assert result.params["margin"] == 5
        assert result.params["n_iter"] == 3

    def test_grayscale_input(self):
        result = remove_background_grabcut(_gray())
        assert isinstance(result, BackgroundRemovalResult)

    def test_large_margin(self):
        img = _bgr(64, 64)
        result = remove_background_grabcut(img, margin=25)
        assert result.method == "grabcut"

    def test_one_iteration(self):
        result = remove_background_grabcut(_bgr(), n_iter=1)
        assert isinstance(result, BackgroundRemovalResult)


# ─── TestAutoRemoveBackground ─────────────────────────────────────────────────

class TestAutoRemoveBackground:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            auto_remove_background(_bgr(), method="otsu")

    def test_thresh_method(self):
        result = auto_remove_background(_bgr(), method="thresh")
        assert result.method == "thresh"

    def test_edges_method(self):
        result = auto_remove_background(_bgr(), method="edges")
        assert result.method == "edges"

    def test_grabcut_method(self):
        result = auto_remove_background(_bgr(), method="grabcut")
        assert result.method == "grabcut"

    def test_default_method_is_thresh(self):
        result = auto_remove_background(_bgr())
        assert result.method == "thresh"

    def test_kwargs_passed(self):
        result = auto_remove_background(_bgr(), method="thresh", bg_thresh=100)
        assert result.params["bg_thresh"] == 100

    def test_returns_result_instance(self):
        result = auto_remove_background(_bgr())
        assert isinstance(result, BackgroundRemovalResult)

    def test_none_method_raises(self):
        with pytest.raises((ValueError, TypeError)):
            auto_remove_background(_bgr(), method="none")


# ─── TestBatchRemoveBackground ────────────────────────────────────────────────

class TestBatchRemoveBackground:
    def test_returns_list(self):
        result = batch_remove_background([_bgr(), _bgr()])
        assert isinstance(result, list)

    def test_same_length(self):
        images = [_bgr(), _bgr(32, 32), _bgr(48, 48)]
        result = batch_remove_background(images)
        assert len(result) == 3

    def test_each_is_result(self):
        for r in batch_remove_background([_bgr(), _bgr()]):
            assert isinstance(r, BackgroundRemovalResult)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            batch_remove_background([_bgr()], method="unknown")

    def test_thresh_method(self):
        results = batch_remove_background([_bgr()], method="thresh")
        assert results[0].method == "thresh"

    def test_edges_method(self):
        results = batch_remove_background([_bgr()], method="edges")
        assert results[0].method == "edges"

    def test_grabcut_method(self):
        results = batch_remove_background([_bgr()], method="grabcut")
        assert results[0].method == "grabcut"

    def test_single_image(self):
        result = batch_remove_background([_bgr()])
        assert len(result) == 1

    def test_kwargs_passed(self):
        results = batch_remove_background([_bgr()], method="thresh", bg_thresh=150)
        assert results[0].params["bg_thresh"] == 150

    def test_empty_list(self):
        result = batch_remove_background([])
        assert result == []

    def test_mixed_sizes(self):
        imgs = [_bgr(32, 32), _bgr(64, 48), _bgr(16, 16)]
        results = batch_remove_background(imgs)
        assert len(results) == 3
        for i, r in enumerate(results):
            assert r.foreground.shape == imgs[i].shape
