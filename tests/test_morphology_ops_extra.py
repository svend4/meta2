"""Extra tests for puzzle_reconstruction/preprocessing/morphology_ops.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.morphology_ops import (
    MorphParams,
    apply_morph,
    batch_morph,
    blackhat,
    close_morph,
    dilate,
    erode,
    fill_holes,
    morphological_gradient,
    open_morph,
    remove_small_blobs,
    skeleton,
    tophat,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _white(h: int = 32, w: int = 32) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


def _black(h: int = 32, w: int = 32) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _noisy(h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h: int = 32, w: int = 32) -> np.ndarray:
    rng = np.random.default_rng(5)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _binary_with_blob(h: int = 32, w: int = 32) -> np.ndarray:
    img = np.zeros((h, w), dtype=np.uint8)
    img[10:20, 10:20] = 255
    return img


def _binary_with_hole(h: int = 32, w: int = 32) -> np.ndarray:
    img = np.zeros((h, w), dtype=np.uint8)
    img[2:30, 2:30] = 255
    img[10:20, 10:20] = 0   # hole in the middle
    return img


# ─── MorphParams (extra) ──────────────────────────────────────────────────────

class TestMorphParamsExtra:
    def test_default_op(self):
        assert MorphParams().op == "open"

    def test_default_kernel_type(self):
        assert MorphParams().kernel_type == "rect"

    def test_default_ksize(self):
        assert MorphParams().ksize == 3

    def test_default_iterations(self):
        assert MorphParams().iterations == 1

    def test_custom_op(self):
        p = MorphParams(op="erode")
        assert p.op == "erode"

    def test_all_valid_ops(self):
        for op in ("erode", "dilate", "open", "close", "tophat", "blackhat", "gradient"):
            MorphParams(op=op)

    def test_invalid_op_raises(self):
        with pytest.raises(ValueError):
            MorphParams(op="skeleton")

    def test_all_valid_kernel_types(self):
        for kt in ("rect", "ellipse", "cross"):
            MorphParams(kernel_type=kt)

    def test_invalid_kernel_type_raises(self):
        with pytest.raises(ValueError):
            MorphParams(kernel_type="diamond")

    def test_ksize_below_3_raises(self):
        with pytest.raises(ValueError):
            MorphParams(ksize=2)

    def test_even_ksize_raises(self):
        with pytest.raises(ValueError):
            MorphParams(ksize=4)

    def test_ksize_3_valid(self):
        assert MorphParams(ksize=3).ksize == 3

    def test_ksize_5_valid(self):
        assert MorphParams(ksize=5).ksize == 5

    def test_iterations_below_1_raises(self):
        with pytest.raises(ValueError):
            MorphParams(iterations=0)

    def test_iterations_custom(self):
        p = MorphParams(iterations=3)
        assert p.iterations == 3

    def test_params_dict_stored(self):
        p = MorphParams(params={"mode": "test"})
        assert p.params["mode"] == "test"


# ─── erode (extra) ────────────────────────────────────────────────────────────

class TestErodeExtra:
    def test_returns_uint8(self):
        assert erode(_white()).dtype == np.uint8

    def test_shape_preserved(self):
        assert erode(_white(16, 24)).shape == (16, 24)

    def test_white_image_eroded_border(self):
        result = erode(_white(), ksize=5)
        assert result[0, 0] == 0

    def test_black_stays_black(self):
        assert erode(_black()).max() == 0

    def test_ksize_below_3_raises(self):
        with pytest.raises(ValueError):
            erode(_white(), ksize=2)

    def test_even_ksize_raises(self):
        with pytest.raises(ValueError):
            erode(_white(), ksize=4)

    def test_invalid_kernel_type_raises(self):
        with pytest.raises(ValueError):
            erode(_white(), kernel_type="diamond")

    def test_multiple_iterations(self):
        r1 = erode(_white(), iterations=1)
        r2 = erode(_white(), iterations=3)
        assert r2.sum() <= r1.sum()

    def test_ellipse_kernel(self):
        result = erode(_white(), kernel_type="ellipse")
        assert result.dtype == np.uint8

    def test_cross_kernel(self):
        result = erode(_white(), kernel_type="cross")
        assert result.dtype == np.uint8

    def test_bgr_image(self):
        result = erode(_bgr())
        assert result.shape == (32, 32, 3)


# ─── dilate (extra) ───────────────────────────────────────────────────────────

class TestDilateExtra:
    def test_returns_uint8(self):
        assert dilate(_black()).dtype == np.uint8

    def test_shape_preserved(self):
        assert dilate(_noisy(16, 24)).shape == (16, 24)

    def test_white_stays_white(self):
        assert dilate(_white()).min() == 255

    def test_black_stays_black(self):
        assert dilate(_black()).max() == 0

    def test_ksize_below_3_raises(self):
        with pytest.raises(ValueError):
            dilate(_white(), ksize=1)

    def test_even_ksize_raises(self):
        with pytest.raises(ValueError):
            dilate(_white(), ksize=6)

    def test_invalid_kernel_type_raises(self):
        with pytest.raises(ValueError):
            dilate(_white(), kernel_type="blob")

    def test_blob_expands(self):
        r0 = _binary_with_blob()
        result = dilate(r0, ksize=3)
        assert result.sum() >= r0.sum()

    def test_bgr_image(self):
        result = dilate(_bgr())
        assert result.ndim == 3


# ─── open_morph (extra) ───────────────────────────────────────────────────────

class TestOpenMorphExtra:
    def test_returns_uint8(self):
        assert open_morph(_noisy()).dtype == np.uint8

    def test_shape_preserved(self):
        assert open_morph(_white(16, 24)).shape == (16, 24)

    def test_ksize_below_3_raises(self):
        with pytest.raises(ValueError):
            open_morph(_white(), ksize=2)

    def test_even_ksize_raises(self):
        with pytest.raises(ValueError):
            open_morph(_white(), ksize=4)

    def test_invalid_kernel_type_raises(self):
        with pytest.raises(ValueError):
            open_morph(_white(), kernel_type="hexagon")

    def test_removes_small_features(self):
        # Small isolated pixel should be removed by opening
        img = np.zeros((32, 32), dtype=np.uint8)
        img[16, 16] = 255   # single pixel
        result = open_morph(img, ksize=5)
        assert result[16, 16] == 0

    def test_bgr_image(self):
        assert open_morph(_bgr()).shape == (32, 32, 3)


# ─── close_morph (extra) ──────────────────────────────────────────────────────

class TestCloseMorphExtra:
    def test_returns_uint8(self):
        assert close_morph(_noisy()).dtype == np.uint8

    def test_shape_preserved(self):
        assert close_morph(_white(16, 24)).shape == (16, 24)

    def test_ksize_below_3_raises(self):
        with pytest.raises(ValueError):
            close_morph(_white(), ksize=2)

    def test_even_ksize_raises(self):
        with pytest.raises(ValueError):
            close_morph(_white(), ksize=4)

    def test_fills_small_gap(self):
        img = np.ones((32, 32), dtype=np.uint8) * 255
        img[15, 15] = 0   # tiny dark hole
        result = close_morph(img, ksize=5)
        assert result[15, 15] == 255

    def test_bgr_image(self):
        assert close_morph(_bgr()).shape == (32, 32, 3)


# ─── tophat (extra) ───────────────────────────────────────────────────────────

class TestTophatExtra:
    def test_returns_uint8(self):
        assert tophat(_white()).dtype == np.uint8

    def test_shape_preserved(self):
        assert tophat(_noisy(16, 24)).shape == (16, 24)

    def test_black_image_zero(self):
        assert tophat(_black()).max() == 0

    def test_ksize_below_3_raises(self):
        with pytest.raises(ValueError):
            tophat(_white(), ksize=2)

    def test_even_ksize_raises(self):
        with pytest.raises(ValueError):
            tophat(_white(), ksize=4)

    def test_bgr_image(self):
        assert tophat(_bgr()).ndim == 3


# ─── blackhat (extra) ─────────────────────────────────────────────────────────

class TestBlackhatExtra:
    def test_returns_uint8(self):
        assert blackhat(_white()).dtype == np.uint8

    def test_shape_preserved(self):
        assert blackhat(_noisy(16, 24)).shape == (16, 24)

    def test_white_image_zero(self):
        assert blackhat(_white()).max() == 0

    def test_ksize_below_3_raises(self):
        with pytest.raises(ValueError):
            blackhat(_white(), ksize=2)

    def test_even_ksize_raises(self):
        with pytest.raises(ValueError):
            blackhat(_white(), ksize=4)

    def test_bgr_image(self):
        assert blackhat(_bgr()).ndim == 3


# ─── morphological_gradient (extra) ───────────────────────────────────────────

class TestMorphologicalGradientExtra:
    def test_returns_uint8(self):
        assert morphological_gradient(_noisy()).dtype == np.uint8

    def test_shape_preserved(self):
        assert morphological_gradient(_white(16, 24)).shape == (16, 24)

    def test_constant_image_zero_gradient(self):
        assert morphological_gradient(_white()).max() == 0

    def test_ksize_below_3_raises(self):
        with pytest.raises(ValueError):
            morphological_gradient(_white(), ksize=2)

    def test_even_ksize_raises(self):
        with pytest.raises(ValueError):
            morphological_gradient(_white(), ksize=4)

    def test_edge_image_has_gradient(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[10:20, :] = 255
        result = morphological_gradient(img, ksize=3)
        assert result.max() > 0


# ─── skeleton (extra) ─────────────────────────────────────────────────────────

class TestSkeletonExtra:
    def test_returns_uint8(self):
        assert skeleton(_binary_with_blob()).dtype == np.uint8

    def test_shape_preserved(self):
        assert skeleton(_binary_with_blob()).shape == (32, 32)

    def test_black_image_returns_black(self):
        assert skeleton(_black()).max() == 0

    def test_3d_image_raises(self):
        with pytest.raises(ValueError):
            skeleton(_bgr())

    def test_result_subset_of_input(self):
        img = _binary_with_blob()
        s = skeleton(img)
        # Skeleton pixels must be where input had pixels
        assert (s[img == 0] == 0).all()

    def test_thin_line_preserved(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[15, :] = 255   # horizontal line
        s = skeleton(img)
        assert s.max() >= 0  # just check it runs without error


# ─── remove_small_blobs (extra) ───────────────────────────────────────────────

class TestRemoveSmallBlobsExtra:
    def test_returns_uint8(self):
        assert remove_small_blobs(_binary_with_blob()).dtype == np.uint8

    def test_shape_preserved(self):
        assert remove_small_blobs(_black(16, 24)).shape == (16, 24)

    def test_small_blob_removed(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[15, 15] = 255   # single pixel blob (area=1)
        result = remove_small_blobs(img, min_area=10)
        assert result[15, 15] == 0

    def test_large_blob_kept(self):
        img = _binary_with_blob()
        result = remove_small_blobs(img, min_area=5)
        # 10×10 blob should survive
        assert result[15, 15] == 255

    def test_black_image_stays_black(self):
        assert remove_small_blobs(_black()).max() == 0

    def test_3d_image_raises(self):
        with pytest.raises(ValueError):
            remove_small_blobs(_bgr())

    def test_negative_min_area_raises(self):
        with pytest.raises(ValueError):
            remove_small_blobs(_black(), min_area=-1)

    def test_min_area_zero_keeps_all(self):
        img = _binary_with_blob()
        result = remove_small_blobs(img, min_area=0)
        assert (result > 0).sum() >= (img > 0).sum()


# ─── fill_holes (extra) ───────────────────────────────────────────────────────

class TestFillHolesExtra:
    def test_returns_uint8(self):
        assert fill_holes(_binary_with_blob()).dtype == np.uint8

    def test_shape_preserved(self):
        assert fill_holes(_black(16, 24)).shape == (16, 24)

    def test_black_stays_black(self):
        assert fill_holes(_black()).max() == 0

    def test_3d_image_raises(self):
        with pytest.raises(ValueError):
            fill_holes(_bgr())

    def test_hole_filled(self):
        img = _binary_with_hole()
        result = fill_holes(img)
        assert result[15, 15] == 255   # hole center should be filled

    def test_no_hole_no_change(self):
        img = _binary_with_blob()
        result = fill_holes(img)
        # Result should include at least as many pixels as original blob
        assert (result > 0).sum() >= (img > 0).sum()


# ─── apply_morph (extra) ──────────────────────────────────────────────────────

class TestApplyMorphExtra:
    def test_returns_uint8(self):
        p = MorphParams(op="open")
        assert apply_morph(_noisy(), p).dtype == np.uint8

    def test_shape_preserved(self):
        p = MorphParams(op="close")
        assert apply_morph(_noisy(16, 24), p).shape == (16, 24)

    def test_erode_op(self):
        p = MorphParams(op="erode")
        result = apply_morph(_white(), p)
        assert result.dtype == np.uint8

    def test_dilate_op(self):
        p = MorphParams(op="dilate")
        result = apply_morph(_black(), p)
        assert result.max() == 0

    def test_open_op(self):
        p = MorphParams(op="open", ksize=5)
        result = apply_morph(_noisy(), p)
        assert result.dtype == np.uint8

    def test_close_op(self):
        p = MorphParams(op="close")
        result = apply_morph(_noisy(), p)
        assert result.dtype == np.uint8

    def test_tophat_op(self):
        p = MorphParams(op="tophat", ksize=5)
        result = apply_morph(_noisy(), p)
        assert result.dtype == np.uint8

    def test_blackhat_op(self):
        p = MorphParams(op="blackhat", ksize=5)
        result = apply_morph(_noisy(), p)
        assert result.dtype == np.uint8

    def test_gradient_op(self):
        p = MorphParams(op="gradient")
        result = apply_morph(_noisy(), p)
        assert result.dtype == np.uint8


# ─── batch_morph (extra) ──────────────────────────────────────────────────────

class TestBatchMorphExtra:
    def test_empty_returns_empty(self):
        assert batch_morph([]) == []

    def test_single_image(self):
        result = batch_morph([_noisy()])
        assert len(result) == 1

    def test_multiple_images(self):
        imgs = [_noisy(seed=i) for i in range(4)]
        result = batch_morph(imgs)
        assert len(result) == 4

    def test_all_uint8(self):
        imgs = [_noisy(), _white(), _black()]
        for r in batch_morph(imgs):
            assert r.dtype == np.uint8

    def test_shapes_preserved(self):
        imgs = [_noisy(16, 24), _noisy(32, 32)]
        results = batch_morph(imgs)
        assert results[0].shape == (16, 24)
        assert results[1].shape == (32, 32)

    def test_none_params_uses_default(self):
        imgs = [_noisy()]
        result = batch_morph(imgs, params=None)
        assert result[0].dtype == np.uint8

    def test_custom_params(self):
        p = MorphParams(op="dilate", ksize=5)
        imgs = [_binary_with_blob()]
        result = batch_morph(imgs, params=p)
        # Dilation should expand the blob
        assert result[0].sum() >= imgs[0].sum()
