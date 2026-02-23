"""Extra tests for puzzle_reconstruction.preprocessing.background_remover."""
from __future__ import annotations

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

def _bgr(h=64, w=64):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _dark(h=64, w=64):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _gray(h=64, w=64):
    return np.full((h, w), 255, dtype=np.uint8)


def _mixed(h=64, w=64):
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    img[h//4: 3*h//4, w//4: 3*w//4] = 50
    return img


# ─── TestBackgroundRemovalResultExtra ─────────────────────────────────────────

class TestBackgroundRemovalResultExtra:
    def test_foreground_array(self):
        fg = _bgr()
        mask = np.zeros((64, 64), dtype=np.uint8)
        r = BackgroundRemovalResult(foreground=fg, mask=mask, method="thresh")
        assert r.foreground.shape == (64, 64, 3)

    def test_mask_2d(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        r = BackgroundRemovalResult(foreground=_bgr(), mask=mask, method="thresh")
        assert r.mask.ndim == 2

    def test_method_string(self):
        r = BackgroundRemovalResult(foreground=_bgr(),
                                     mask=np.zeros((64, 64), dtype=np.uint8),
                                     method="edges")
        assert isinstance(r.method, str)

    def test_params_stored_custom(self):
        r = BackgroundRemovalResult(foreground=_bgr(),
                                     mask=np.zeros((64, 64), dtype=np.uint8),
                                     method="thresh",
                                     params={"a": 1, "b": 2})
        assert r.params["a"] == 1 and r.params["b"] == 2

    def test_repr_is_nonempty_string(self):
        r = BackgroundRemovalResult(foreground=_bgr(),
                                     mask=np.zeros((64, 64), dtype=np.uint8),
                                     method="thresh")
        assert len(repr(r)) > 0


# ─── TestRemoveBackgroundThreshExtra ──────────────────────────────────────────

class TestRemoveBackgroundThreshExtra:
    def test_mask_values_binary(self):
        result = remove_background_thresh(_mixed())
        unique = set(np.unique(result.mask))
        assert unique.issubset({0, 255})

    def test_low_threshold_more_foreground(self):
        img = _mixed()
        r_low = remove_background_thresh(img, bg_thresh=250)
        r_high = remove_background_thresh(img, bg_thresh=100)
        fg_low = float(r_low.mask.sum())
        fg_high = float(r_high.mask.sum())
        assert fg_low >= fg_high

    def test_non_square_image(self):
        result = remove_background_thresh(_bgr(32, 48))
        assert result.foreground.shape == (32, 48, 3)
        assert result.mask.shape == (32, 48)

    def test_dark_image_all_foreground(self):
        result = remove_background_thresh(_dark(), bg_thresh=200)
        fg_ratio = float(result.mask.mean()) / 255.0
        assert fg_ratio > 0.5

    def test_method_always_thresh(self):
        result = remove_background_thresh(_bgr())
        assert result.method == "thresh"


# ─── TestRemoveBackgroundEdgesExtra ───────────────────────────────────────────

class TestRemoveBackgroundEdgesExtra:
    def test_mask_shape_matches_input(self):
        result = remove_background_edges(_bgr(32, 48))
        assert result.mask.shape == (32, 48)

    def test_foreground_shape_matches_input(self):
        img = _bgr(32, 48)
        result = remove_background_edges(img)
        assert result.foreground.shape == img.shape

    def test_method_always_edges(self):
        result = remove_background_edges(_bgr())
        assert result.method == "edges"

    def test_mask_dtype_uint8(self):
        result = remove_background_edges(_mixed())
        assert result.mask.dtype == np.uint8

    def test_mixed_image_processes(self):
        result = remove_background_edges(_mixed())
        assert isinstance(result, BackgroundRemovalResult)


# ─── TestRemoveBackgroundGrabcutExtra ─────────────────────────────────────────

class TestRemoveBackgroundGrabcutExtra:
    def test_foreground_shape_preserved(self):
        img = _bgr(48, 48)
        result = remove_background_grabcut(img)
        assert result.foreground.shape == img.shape

    def test_mask_shape_2d(self):
        result = remove_background_grabcut(_bgr())
        assert result.mask.ndim == 2

    def test_method_always_grabcut(self):
        result = remove_background_grabcut(_bgr())
        assert result.method == "grabcut"

    def test_n_iter_2_ok(self):
        result = remove_background_grabcut(_bgr(), n_iter=2)
        assert isinstance(result, BackgroundRemovalResult)

    def test_mixed_image_processes(self):
        result = remove_background_grabcut(_mixed())
        assert isinstance(result, BackgroundRemovalResult)


# ─── TestAutoRemoveBackgroundExtra ────────────────────────────────────────────

class TestAutoRemoveBackgroundExtra:
    def test_returns_background_removal_result(self):
        result = auto_remove_background(_bgr())
        assert isinstance(result, BackgroundRemovalResult)

    def test_all_valid_methods(self):
        for method in ("thresh", "edges", "grabcut"):
            result = auto_remove_background(_bgr(), method=method)
            assert result.method == method

    def test_mask_dtype_uint8(self):
        result = auto_remove_background(_mixed())
        assert result.mask.dtype == np.uint8

    def test_foreground_same_shape_as_input(self):
        img = _bgr(48, 32)
        result = auto_remove_background(img)
        assert result.foreground.shape == img.shape

    def test_gray_input_thresh(self):
        result = auto_remove_background(_gray(), method="thresh")
        assert isinstance(result, BackgroundRemovalResult)


# ─── TestBatchRemoveBackgroundExtra ───────────────────────────────────────────

class TestBatchRemoveBackgroundExtra:
    def test_two_images(self):
        result = batch_remove_background([_bgr(), _mixed()])
        assert len(result) == 2

    def test_all_results_correct_type(self):
        results = batch_remove_background([_bgr(), _bgr()])
        for r in results:
            assert isinstance(r, BackgroundRemovalResult)

    def test_method_edges_forwarded(self):
        results = batch_remove_background([_bgr()], method="edges")
        assert results[0].method == "edges"

    def test_single_image_batch(self):
        results = batch_remove_background([_mixed()])
        assert len(results) == 1

    def test_masks_binary(self):
        results = batch_remove_background([_bgr(), _mixed()])
        for r in results:
            unique = set(np.unique(r.mask))
            assert unique.issubset({0, 255})

    def test_shapes_preserved(self):
        imgs = [_bgr(32, 32), _bgr(64, 48)]
        results = batch_remove_background(imgs)
        for i, r in enumerate(results):
            assert r.foreground.shape == imgs[i].shape
