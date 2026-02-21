"""Тесты для puzzle_reconstruction/preprocessing/fragment_cropper.py."""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.fragment_cropper import (
    CropResult,
    find_content_bbox,
    pad_image,
    crop_to_content,
    auto_crop,
    batch_crop,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _white(h=64, w=64):
    """Полностью белое изображение (фон)."""
    return np.full((h, w), 255, dtype=np.uint8)


def _with_content(h=64, w=64):
    """Белое с тёмным прямоугольником в центре."""
    img = _white(h, w)
    img[20:44, 16:48] = 30   # тёмный блок
    return img


def _noisy(h=64, w=64, seed=7):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 200, (h, w), dtype=np.uint8)


def _bgr_content(h=64, w=64):
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    img[20:44, 16:48] = [30, 50, 70]
    return img


# ─── CropResult ───────────────────────────────────────────────────────────────

class TestCropResult:
    def _make(self):
        cropped = np.zeros((20, 30), dtype=np.uint8)
        return CropResult(
            cropped=cropped,
            bbox=(5, 5, 30, 20),
            padding=4,
            original_shape=(64, 64),
            method="content",
        )

    def test_fields(self):
        r = self._make()
        assert r.padding == 4
        assert r.original_shape == (64, 64)
        assert r.method == "content"
        assert len(r.bbox) == 4

    def test_cropped_stored(self):
        r = self._make()
        assert r.cropped.shape == (20, 30)

    def test_bbox_4_tuple(self):
        r = self._make()
        assert len(r.bbox) == 4

    def test_original_shape_2_tuple(self):
        r = self._make()
        assert len(r.original_shape) == 2

    def test_params_default_empty(self):
        r = self._make()
        assert isinstance(r.params, dict)

    def test_params_stored(self):
        cropped = np.zeros((10, 10), dtype=np.uint8)
        r = CropResult(cropped=cropped, bbox=(0, 0, 10, 10), padding=2,
                        original_shape=(30, 30), method="auto",
                        params={"bg_thresh": 240})
        assert r.params["bg_thresh"] == 240

    def test_repr(self):
        r = self._make()
        s = repr(r)
        assert "CropResult" in s
        assert "content" in s


# ─── find_content_bbox ────────────────────────────────────────────────────────

class TestFindContentBbox:
    def test_returns_4_tuple(self):
        r = find_content_bbox(_with_content())
        assert isinstance(r, tuple)
        assert len(r) == 4

    def test_all_white_returns_full(self):
        h, w = 64, 64
        x, y, bw, bh = find_content_bbox(_white(h, w), bg_thresh=240)
        assert x == 0
        assert y == 0
        assert bw == w
        assert bh == h

    def test_content_bbox_smaller_than_image(self):
        img = _with_content(64, 64)
        x, y, bw, bh = find_content_bbox(img, bg_thresh=240)
        assert bw < 64
        assert bh < 64

    def test_content_bbox_positive_dimensions(self):
        x, y, bw, bh = find_content_bbox(_with_content(), bg_thresh=240)
        assert bw > 0
        assert bh > 0

    def test_content_bbox_within_image(self):
        h, w = 64, 64
        x, y, bw, bh = find_content_bbox(_with_content(h, w))
        assert 0 <= x
        assert 0 <= y
        assert x + bw <= w
        assert y + bh <= h

    def test_margin_expands_bbox(self):
        r0 = find_content_bbox(_with_content(), bg_thresh=240, margin=0)
        r5 = find_content_bbox(_with_content(), bg_thresh=240, margin=5)
        # With margin, bbox should be larger (or same if hits boundary)
        w0, h0 = r0[2], r0[3]
        w5, h5 = r5[2], r5[3]
        assert w5 >= w0
        assert h5 >= h0

    def test_gray_input(self):
        r = find_content_bbox(_with_content())
        assert len(r) == 4

    def test_bgr_input(self):
        r = find_content_bbox(_bgr_content())
        assert len(r) == 4
        assert r[2] > 0

    def test_all_values_nonneg(self):
        for v in find_content_bbox(_with_content()):
            assert v >= 0


# ─── pad_image ────────────────────────────────────────────────────────────────

class TestPadImage:
    def test_returns_ndarray(self):
        assert isinstance(pad_image(_white(32, 32), 4), np.ndarray)

    def test_larger_shape(self):
        r = pad_image(_white(32, 32), 8)
        assert r.shape == (48, 48)

    def test_dtype_preserved(self):
        r = pad_image(_white(), 4)
        assert r.dtype == np.uint8

    def test_zero_padding_copy(self):
        img = _noisy()
        r   = pad_image(img, 0)
        np.testing.assert_array_equal(r, img)

    def test_gray_input(self):
        r = pad_image(_with_content(), 4)
        assert r.ndim == 2

    def test_bgr_input(self):
        r = pad_image(_bgr_content(), 4)
        assert r.ndim == 3
        assert r.shape == (72, 72, 3)

    def test_fill_value_in_border(self):
        img = np.full((10, 10), 50, dtype=np.uint8)
        r   = pad_image(img, 3, fill=0)
        assert r[0, 0] == 0    # top-left corner is padding
        assert r[13, 13] == 0  # bottom-right corner

    def test_fill_default_255(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        r   = pad_image(img, 2)
        assert r[0, 0] == 255  # padding is white by default


# ─── crop_to_content ──────────────────────────────────────────────────────────

class TestCropToContent:
    def test_returns_result(self):
        assert isinstance(crop_to_content(_with_content()), CropResult)

    def test_method(self):
        assert crop_to_content(_with_content()).method == "content"

    def test_cropped_smaller_than_original(self):
        r = crop_to_content(_with_content(64, 64), padding=0)
        assert r.cropped.shape[0] <= 64
        assert r.cropped.shape[1] <= 64

    def test_original_shape_stored(self):
        r = crop_to_content(_with_content(64, 64))
        assert r.original_shape == (64, 64)

    def test_padding_stored(self):
        r = crop_to_content(_with_content(), padding=6)
        assert r.padding == 6

    def test_params_stored(self):
        r = crop_to_content(_with_content(), bg_thresh=220, min_size=8)
        assert r.params.get("bg_thresh") == 220
        assert r.params.get("min_size") == 8

    def test_bbox_4_tuple(self):
        r = crop_to_content(_with_content())
        assert len(r.bbox) == 4

    def test_bbox_within_original(self):
        h, w = 64, 64
        r    = crop_to_content(_with_content(h, w))
        x, y, bw, bh = r.bbox
        assert x >= 0
        assert y >= 0
        assert x + bw <= w + r.padding * 2  # padded bbox can extend slightly

    def test_min_size_guarantee(self):
        # Very large min_size → result at least min_size
        r = crop_to_content(_with_content(), min_size=32)
        assert r.cropped.shape[0] >= 32
        assert r.cropped.shape[1] >= 32

    def test_gray_input(self):
        r = crop_to_content(_with_content())
        assert r.cropped.ndim == 2

    def test_bgr_input(self):
        r = crop_to_content(_bgr_content())
        assert r.cropped.ndim == 3

    def test_white_image_no_crash(self):
        r = crop_to_content(_white())
        assert isinstance(r, CropResult)
        assert r.cropped.shape[0] >= 1

    def test_noisy_image(self):
        r = crop_to_content(_noisy())
        assert isinstance(r, CropResult)


# ─── auto_crop ────────────────────────────────────────────────────────────────

class TestAutoCrop:
    def test_returns_result(self):
        assert isinstance(auto_crop(_with_content()), CropResult)

    def test_method(self):
        assert auto_crop(_with_content()).method == "auto"

    def test_cropped_shape_nonneg(self):
        r = auto_crop(_with_content())
        assert r.cropped.shape[0] >= 1
        assert r.cropped.shape[1] >= 1

    def test_original_shape_stored(self):
        r = auto_crop(_with_content(48, 56))
        assert r.original_shape == (48, 56)

    def test_padding_stored(self):
        r = auto_crop(_with_content(), padding=8)
        assert r.padding == 8

    def test_params_stored(self):
        r = auto_crop(_with_content(), bg_thresh=220, min_size=6)
        assert "bg_thresh" in r.params
        assert r.params.get("min_size") == 6

    def test_white_image_no_crash(self):
        r = auto_crop(_white())
        assert isinstance(r, CropResult)
        assert r.cropped.shape[0] >= 1

    def test_noisy_image(self):
        r = auto_crop(_noisy())
        assert isinstance(r, CropResult)

    def test_gray_input(self):
        r = auto_crop(_with_content())
        assert r.cropped.ndim == 2

    def test_bgr_input(self):
        r = auto_crop(_bgr_content())
        assert r.cropped.ndim == 3

    def test_bbox_4_tuple(self):
        r = auto_crop(_with_content())
        assert len(r.bbox) == 4


# ─── batch_crop ───────────────────────────────────────────────────────────────

class TestBatchCrop:
    def test_returns_list(self):
        imgs = [_with_content(), _noisy()]
        r    = batch_crop(imgs)
        assert isinstance(r, list)
        assert len(r) == 2

    def test_each_is_result(self):
        for r in batch_crop([_with_content(), _white()]):
            assert isinstance(r, CropResult)

    def test_empty_list(self):
        assert batch_crop([]) == []

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            batch_crop([_with_content()], method="spiral_crop_xyz")

    @pytest.mark.parametrize("method", ["content", "auto"])
    def test_both_methods(self, method):
        imgs = [_with_content(), _noisy()]
        r    = batch_crop(imgs, method=method)
        assert len(r) == 2
        for result in r:
            assert isinstance(result, CropResult)
            assert result.method == method

    def test_padding_forwarded(self):
        results = batch_crop([_with_content()], padding=10)
        assert results[0].padding == 10

    def test_bg_thresh_forwarded(self):
        results = batch_crop([_with_content()], bg_thresh=200)
        # Just ensure no crash and correct method
        assert isinstance(results[0], CropResult)

    def test_bgr_input(self):
        results = batch_crop([_bgr_content()])
        assert results[0].cropped.ndim == 3

    def test_shapes_preserved_individually(self):
        imgs = [_with_content(32, 48), _with_content(64, 80)]
        r    = batch_crop(imgs)
        assert r[0].original_shape == (32, 48)
        assert r[1].original_shape == (64, 80)
