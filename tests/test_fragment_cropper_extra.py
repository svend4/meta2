"""Extra tests for puzzle_reconstruction.preprocessing.fragment_cropper."""
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _white(h=64, w=64):
    return np.full((h, w), 255, dtype=np.uint8)


def _content(h=64, w=64):
    img = _white(h, w)
    img[20:44, 16:48] = 30
    return img


def _noisy(h=64, w=64, seed=7):
    return np.random.default_rng(seed).integers(0, 200, (h, w), dtype=np.uint8)


def _bgr_content(h=64, w=64):
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    img[20:44, 16:48] = [30, 50, 70]
    return img


# ─── TestCropResultExtra ─────────────────────────────────────────────────────

class TestCropResultExtra:
    def _make(self, **kw):
        defaults = dict(cropped=np.zeros((20, 30), dtype=np.uint8),
                        bbox=(5, 5, 30, 20), padding=4,
                        original_shape=(64, 64), method="content")
        defaults.update(kw)
        return CropResult(**defaults)

    def test_fields(self):
        r = self._make()
        assert r.padding == 4
        assert r.method == "content"

    def test_cropped_shape(self):
        assert self._make().cropped.shape == (20, 30)

    def test_bbox_len_4(self):
        assert len(self._make().bbox) == 4

    def test_original_shape_len_2(self):
        assert len(self._make().original_shape) == 2

    def test_params_default_empty(self):
        assert isinstance(self._make().params, dict)

    def test_params_custom(self):
        r = self._make(params={"bg_thresh": 240})
        assert r.params["bg_thresh"] == 240

    def test_repr_contains_method(self):
        assert "content" in repr(self._make())

    def test_method_auto(self):
        r = self._make(method="auto")
        assert r.method == "auto"


# ─── TestFindContentBboxExtra ────────────────────────────────────────────────

class TestFindContentBboxExtra:
    def test_returns_4_tuple(self):
        r = find_content_bbox(_content())
        assert isinstance(r, tuple) and len(r) == 4

    def test_all_white_full_bbox(self):
        x, y, bw, bh = find_content_bbox(_white(64, 64), bg_thresh=240)
        assert x == 0 and y == 0 and bw == 64 and bh == 64

    def test_content_bbox_smaller(self):
        x, y, bw, bh = find_content_bbox(_content(), bg_thresh=240)
        assert bw < 64 and bh < 64

    def test_positive_dimensions(self):
        _, _, bw, bh = find_content_bbox(_content(), bg_thresh=240)
        assert bw > 0 and bh > 0

    def test_within_image(self):
        x, y, bw, bh = find_content_bbox(_content(64, 64))
        assert x >= 0 and y >= 0 and x + bw <= 64 and y + bh <= 64

    def test_margin_expands(self):
        r0 = find_content_bbox(_content(), bg_thresh=240, margin=0)
        r5 = find_content_bbox(_content(), bg_thresh=240, margin=5)
        assert r5[2] >= r0[2] and r5[3] >= r0[3]

    def test_bgr_input(self):
        r = find_content_bbox(_bgr_content())
        assert len(r) == 4 and r[2] > 0

    def test_nonneg_values(self):
        for v in find_content_bbox(_content()):
            assert v >= 0


# ─── TestPadImageExtra ───────────────────────────────────────────────────────

class TestPadImageExtra:
    def test_returns_ndarray(self):
        assert isinstance(pad_image(_white(32, 32), 4), np.ndarray)

    def test_larger_shape(self):
        r = pad_image(_white(32, 32), 8)
        assert r.shape == (48, 48)

    def test_dtype_preserved(self):
        assert pad_image(_white(), 4).dtype == np.uint8

    def test_zero_padding(self):
        img = _noisy()
        np.testing.assert_array_equal(pad_image(img, 0), img)

    def test_bgr_input(self):
        r = pad_image(_bgr_content(), 4)
        assert r.ndim == 3 and r.shape == (72, 72, 3)

    def test_fill_value(self):
        r = pad_image(np.full((10, 10), 50, dtype=np.uint8), 3, fill=0)
        assert r[0, 0] == 0

    def test_fill_default_255(self):
        r = pad_image(np.zeros((10, 10), dtype=np.uint8), 2)
        assert r[0, 0] == 255

    def test_gray_ndim(self):
        assert pad_image(_content(), 4).ndim == 2


# ─── TestCropToContentExtra ──────────────────────────────────────────────────

class TestCropToContentExtra:
    def test_returns_result(self):
        assert isinstance(crop_to_content(_content()), CropResult)

    def test_method_content(self):
        assert crop_to_content(_content()).method == "content"

    def test_cropped_le_original(self):
        r = crop_to_content(_content(64, 64), padding=0)
        assert r.cropped.shape[0] <= 64 and r.cropped.shape[1] <= 64

    def test_original_shape(self):
        r = crop_to_content(_content(64, 64))
        assert r.original_shape == (64, 64)

    def test_padding_stored(self):
        assert crop_to_content(_content(), padding=10).padding == 10

    def test_params_stored(self):
        r = crop_to_content(_content(), bg_thresh=220, min_size=8)
        assert r.params.get("bg_thresh") == 220

    def test_bbox_4_tuple(self):
        assert len(crop_to_content(_content()).bbox) == 4

    def test_min_size(self):
        r = crop_to_content(_content(), min_size=32)
        assert r.cropped.shape[0] >= 32 and r.cropped.shape[1] >= 32

    def test_gray_input(self):
        assert crop_to_content(_content()).cropped.ndim == 2

    def test_bgr_input(self):
        assert crop_to_content(_bgr_content()).cropped.ndim == 3

    def test_white_no_crash(self):
        r = crop_to_content(_white())
        assert isinstance(r, CropResult)

    def test_noisy_no_crash(self):
        assert isinstance(crop_to_content(_noisy()), CropResult)


# ─── TestAutoCropExtra ───────────────────────────────────────────────────────

class TestAutoCropExtra:
    def test_returns_result(self):
        assert isinstance(auto_crop(_content()), CropResult)

    def test_method_auto(self):
        assert auto_crop(_content()).method == "auto"

    def test_cropped_positive(self):
        r = auto_crop(_content())
        assert r.cropped.shape[0] >= 1 and r.cropped.shape[1] >= 1

    def test_original_shape(self):
        r = auto_crop(_content(48, 56))
        assert r.original_shape == (48, 56)

    def test_padding_stored(self):
        assert auto_crop(_content(), padding=8).padding == 8

    def test_params_stored(self):
        r = auto_crop(_content(), bg_thresh=220, min_size=6)
        assert "bg_thresh" in r.params

    def test_white_no_crash(self):
        assert isinstance(auto_crop(_white()), CropResult)

    def test_noisy_no_crash(self):
        assert isinstance(auto_crop(_noisy()), CropResult)

    def test_gray_ndim(self):
        assert auto_crop(_content()).cropped.ndim == 2

    def test_bgr_ndim(self):
        assert auto_crop(_bgr_content()).cropped.ndim == 3

    def test_bbox_4_tuple(self):
        assert len(auto_crop(_content()).bbox) == 4


# ─── TestBatchCropExtra ──────────────────────────────────────────────────────

class TestBatchCropExtra:
    def test_returns_list(self):
        assert isinstance(batch_crop([_content(), _noisy()]), list)

    def test_correct_length(self):
        assert len(batch_crop([_content(), _white(), _noisy()])) == 3

    def test_each_crop_result(self):
        for r in batch_crop([_content(), _white()]):
            assert isinstance(r, CropResult)

    def test_empty_list(self):
        assert batch_crop([]) == []

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            batch_crop([_content()], method="unknown_xyz")

    @pytest.mark.parametrize("method", ["content", "auto"])
    def test_both_methods(self, method):
        results = batch_crop([_content()], method=method)
        assert results[0].method == method

    def test_padding_forwarded(self):
        assert batch_crop([_content()], padding=12)[0].padding == 12

    def test_bgr_input(self):
        assert batch_crop([_bgr_content()])[0].cropped.ndim == 3

    def test_shapes_preserved(self):
        imgs = [_content(32, 48), _content(64, 80)]
        results = batch_crop(imgs)
        assert results[0].original_shape == (32, 48)
        assert results[1].original_shape == (64, 80)

    def test_single_image(self):
        results = batch_crop([_content()])
        assert len(results) == 1
        assert isinstance(results[0], CropResult)
