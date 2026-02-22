"""
Тесты для puzzle_reconstruction.preprocessing.background_remover.
"""
import pytest
import numpy as np

from puzzle_reconstruction.preprocessing.background_remover import (
    BackgroundRemovalResult,
    remove_background_thresh,
    remove_background_edges,
    remove_background_grabcut,
    auto_remove_background,
    batch_remove_background,
)


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _white_img(h=64, w=64):
    """Белое BGR-изображение."""
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _black_img(h=64, w=64):
    """Чёрное BGR-изображение."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _gray_img(h=64, w=64):
    """Серое grayscale-изображение (128)."""
    return np.full((h, w), 128, dtype=np.uint8)


def _fg_on_white(h=64, w=64, fg_color=0):
    """Белый фон с тёмным прямоугольником в центре."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    img[16:48, 16:48] = fg_color
    return img


def _fg_on_black(h=64, w=64, fg_value=200):
    """Чёрный фон со светлым прямоугольником в центре."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[16:48, 16:48] = fg_value
    return img


# ─── BackgroundRemovalResult ──────────────────────────────────────────────────

class TestBackgroundRemovalResult:
    def test_fields_accessible(self):
        fg   = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        res  = BackgroundRemovalResult(foreground=fg, mask=mask, method="thresh")
        assert res.method == "thresh"
        assert res.foreground.shape == (10, 10, 3)
        assert res.mask.shape == (10, 10)

    def test_default_params_empty_dict(self):
        res = BackgroundRemovalResult(
            foreground=np.zeros((4, 4, 3), dtype=np.uint8),
            mask=np.zeros((4, 4), dtype=np.uint8),
            method="edges",
        )
        assert res.params == {}

    def test_repr_contains_method(self):
        fg   = np.zeros((8, 8, 3), dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        res  = BackgroundRemovalResult(foreground=fg, mask=mask, method="grabcut")
        assert "grabcut" in repr(res)


# ─── remove_background_thresh ─────────────────────────────────────────────────

class TestRemoveBackgroundThresh:
    def test_method_name(self):
        res = remove_background_thresh(_white_img())
        assert res.method == "thresh"

    def test_white_bg_mask_mostly_zero(self):
        img = _fg_on_white()
        res = remove_background_thresh(img, bg_thresh=240)
        # Белый фон → маска 0 (фон удалён)
        corners = [res.mask[0, 0], res.mask[0, -1],
                   res.mask[-1, 0], res.mask[-1, -1]]
        assert all(c == 0 for c in corners)

    def test_center_foreground_preserved(self):
        img = _fg_on_white()
        res = remove_background_thresh(img, bg_thresh=240)
        # Центральный тёмный прямоугольник — маска 255
        assert res.mask[32, 32] == 255

    def test_invert_dark_background(self):
        img = _fg_on_black()
        res = remove_background_thresh(img, bg_thresh=10, invert=True)
        # Тёмный фон → маска 0
        assert res.mask[0, 0] == 0
        # Светлый центр → маска 255
        assert res.mask[32, 32] == 255

    def test_foreground_shape_matches_input(self):
        img = _white_img(32, 48)
        res = remove_background_thresh(img)
        assert res.foreground.shape == img.shape

    def test_mask_shape_matches_input(self):
        img = _white_img(32, 48)
        res = remove_background_thresh(img)
        assert res.mask.shape == (32, 48)

    def test_mask_dtype_uint8(self):
        res = remove_background_thresh(_white_img())
        assert res.mask.dtype == np.uint8

    def test_bg_fill_applied(self):
        img = _fg_on_white()
        res = remove_background_thresh(img, bg_thresh=240, bg_fill=127)
        # Угол (фон) должен иметь значение bg_fill=127
        assert int(res.foreground[0, 0, 0]) == 127

    def test_params_stored(self):
        res = remove_background_thresh(_white_img(), bg_thresh=200, invert=True)
        assert res.params["bg_thresh"] == 200
        assert res.params["invert"] is True

    def test_grayscale_input(self):
        img = np.full((32, 32), 250, dtype=np.uint8)
        img[8:24, 8:24] = 50
        res = remove_background_thresh(img, bg_thresh=200)
        assert res.foreground.shape == (32, 32)
        assert res.mask.shape == (32, 32)


# ─── remove_background_edges ──────────────────────────────────────────────────

class TestRemoveBackgroundEdges:
    def test_method_name(self):
        res = remove_background_edges(_fg_on_white())
        assert res.method == "edges"

    def test_mask_shape(self):
        img = _fg_on_white(64, 80)
        res = remove_background_edges(img)
        assert res.mask.shape == (64, 80)

    def test_foreground_shape(self):
        img = _fg_on_white(64, 80)
        res = remove_background_edges(img)
        assert res.foreground.shape == img.shape

    def test_mask_dtype(self):
        res = remove_background_edges(_fg_on_white())
        assert res.mask.dtype == np.uint8

    def test_mask_binary_values(self):
        res = remove_background_edges(_fg_on_white())
        unique = np.unique(res.mask)
        assert set(unique).issubset({0, 255})

    def test_params_stored(self):
        res = remove_background_edges(
            _fg_on_white(), low_thresh=30, high_thresh=120, dilate_ksize=3)
        assert res.params["low_thresh"] == 30
        assert res.params["high_thresh"] == 120
        assert res.params["dilate_ksize"] == 3

    def test_dilate_ksize_1_no_dilation(self):
        # dilate_ksize=1 → ветка пропуска дилатации, не должно падать
        res = remove_background_edges(_fg_on_white(), dilate_ksize=1)
        assert res.mask.shape == (64, 64)

    def test_grayscale_input(self):
        img = np.full((32, 32), 255, dtype=np.uint8)
        img[8:24, 8:24] = 0
        res = remove_background_edges(img)
        assert res.foreground.shape == (32, 32)


# ─── remove_background_grabcut ────────────────────────────────────────────────

class TestRemoveBackgroundGrabcut:
    def test_method_name(self):
        res = remove_background_grabcut(_fg_on_white())
        assert res.method == "grabcut"

    def test_mask_shape(self):
        img = _fg_on_white(64, 80)
        res = remove_background_grabcut(img)
        assert res.mask.shape == (64, 80)

    def test_foreground_shape(self):
        img = _fg_on_white(64, 80)
        res = remove_background_grabcut(img)
        assert res.foreground.shape == img.shape

    def test_mask_dtype(self):
        res = remove_background_grabcut(_fg_on_white())
        assert res.mask.dtype == np.uint8

    def test_params_stored(self):
        res = remove_background_grabcut(_fg_on_white(), margin=5, n_iter=3)
        assert res.params["margin"] == 5
        assert res.params["n_iter"] == 3

    def test_grayscale_input(self):
        img = np.full((32, 32), 200, dtype=np.uint8)
        img[8:24, 8:24] = 50
        res = remove_background_grabcut(img)
        assert res.foreground.shape == (32, 32)

    def test_small_margin(self):
        res = remove_background_grabcut(_fg_on_white(), margin=2)
        assert res.mask.shape == (64, 64)


# ─── auto_remove_background ───────────────────────────────────────────────────

class TestAutoRemoveBackground:
    @pytest.mark.parametrize("method", ["thresh", "edges", "grabcut"])
    def test_known_methods(self, method):
        img = _fg_on_white()
        res = auto_remove_background(img, method=method)
        assert res.method == method
        assert isinstance(res, BackgroundRemovalResult)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            auto_remove_background(_white_img(), method="foobar")

    def test_kwargs_forwarded_thresh(self):
        res = auto_remove_background(_fg_on_white(), method="thresh", bg_thresh=200)
        assert res.params["bg_thresh"] == 200

    def test_kwargs_forwarded_edges(self):
        res = auto_remove_background(
            _fg_on_white(), method="edges", low_thresh=40)
        assert res.params["low_thresh"] == 40


# ─── batch_remove_background ──────────────────────────────────────────────────

class TestBatchRemoveBackground:
    def test_empty_list(self):
        result = batch_remove_background([], method="thresh")
        assert result == []

    def test_length_preserved(self):
        imgs = [_fg_on_white() for _ in range(4)]
        result = batch_remove_background(imgs, method="thresh")
        assert len(result) == 4

    def test_all_results_are_instances(self):
        imgs = [_fg_on_white(), _fg_on_white(32, 32)]
        result = batch_remove_background(imgs, method="thresh")
        for r in result:
            assert isinstance(r, BackgroundRemovalResult)

    def test_unknown_method_raises_upfront(self):
        imgs = [_fg_on_white()]
        with pytest.raises(ValueError, match="Unknown"):
            batch_remove_background(imgs, method="magic")

    def test_shapes_match_inputs(self):
        imgs = [_white_img(32, 64), _white_img(48, 48)]
        result = batch_remove_background(imgs, method="thresh")
        assert result[0].foreground.shape == (32, 64, 3)
        assert result[1].foreground.shape == (48, 48, 3)

    @pytest.mark.parametrize("method", ["thresh", "edges"])
    def test_method_applied_to_all(self, method):
        imgs = [_fg_on_white() for _ in range(3)]
        result = batch_remove_background(imgs, method=method)
        assert all(r.method == method for r in result)

    def test_kwargs_forwarded(self):
        imgs = [_fg_on_white()]
        result = batch_remove_background(imgs, method="thresh", bg_thresh=200)
        assert result[0].params["bg_thresh"] == 200
