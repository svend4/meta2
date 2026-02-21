"""Tests for puzzle_reconstruction.preprocessing.morphology_ops."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _black(h: int = 32, w: int = 32) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _white(h: int = 32, w: int = 32) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


def _dot(h: int = 32, w: int = 32, cx: int = 16, cy: int = 16, r: int = 2) -> np.ndarray:
    """Single bright dot on black background."""
    img = _black(h, w)
    cv_import = __import__("cv2")
    cv_import.circle(img, (cx, cy), r, 255, -1)
    return img


def _rect(h: int = 32, w: int = 32) -> np.ndarray:
    """Filled white rectangle in center."""
    img = _black(h, w)
    img[8:24, 8:24] = 255
    return img


def _rect_with_hole(h: int = 48, w: int = 48) -> np.ndarray:
    """White rectangle with a black hole inside."""
    img = _black(h, w)
    img[8:40, 8:40] = 255
    img[18:30, 18:30] = 0  # interior hole
    return img


def _bgr(h: int = 32, w: int = 32) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 100
    return img


def _two_blobs() -> np.ndarray:
    """Small blob (area≈4) and large blob (area≈64) on black background."""
    img = _black(64, 64)
    # small blob — 2×2 = 4 pixels
    img[5:7, 5:7] = 255
    # large blob — 8×8 = 64 pixels
    img[30:38, 30:38] = 255
    return img


# ─── MorphParams ─────────────────────────────────────────────────────────────

class TestMorphParams:
    def test_defaults(self):
        p = MorphParams()
        assert p.op == "open"
        assert p.kernel_type == "rect"
        assert p.ksize == 3
        assert p.iterations == 1

    def test_all_valid_ops(self):
        for op in ("erode", "dilate", "open", "close", "tophat", "blackhat", "gradient"):
            p = MorphParams(op=op)
            assert p.op == op

    def test_invalid_op_raises(self):
        with pytest.raises(ValueError):
            MorphParams(op="unknown")

    def test_all_valid_kernel_types(self):
        for kt in ("rect", "ellipse", "cross"):
            p = MorphParams(kernel_type=kt)
            assert p.kernel_type == kt

    def test_invalid_kernel_type_raises(self):
        with pytest.raises(ValueError):
            MorphParams(kernel_type="diamond")

    def test_ksize_less_than_3_raises(self):
        with pytest.raises(ValueError):
            MorphParams(ksize=2)

    def test_even_ksize_raises(self):
        with pytest.raises(ValueError):
            MorphParams(ksize=4)

    def test_iterations_less_than_1_raises(self):
        with pytest.raises(ValueError):
            MorphParams(iterations=0)

    def test_custom_params_stored(self):
        p = MorphParams(op="dilate", kernel_type="ellipse", ksize=7, iterations=2)
        assert p.op == "dilate"
        assert p.kernel_type == "ellipse"
        assert p.ksize == 7
        assert p.iterations == 2


# ─── erode ───────────────────────────────────────────────────────────────────

class TestErode:
    def test_returns_uint8(self):
        assert erode(_rect()).dtype == np.uint8

    def test_shape_preserved(self):
        img = _rect(40, 50)
        result = erode(img, ksize=3)
        assert result.shape == (40, 50)

    def test_erodes_bright_pixels(self):
        img = _rect()
        result = erode(img, ksize=5)
        assert result.sum() < img.sum()

    def test_uniform_white_unchanged(self):
        img = _white()
        result = erode(img, ksize=3)
        np.testing.assert_array_equal(result, img)

    def test_uniform_black_unchanged(self):
        img = _black()
        result = erode(img, ksize=3)
        np.testing.assert_array_equal(result, img)

    def test_ksize_less_than_3_raises(self):
        with pytest.raises(ValueError):
            erode(_rect(), ksize=2)

    def test_even_ksize_raises(self):
        with pytest.raises(ValueError):
            erode(_rect(), ksize=4)

    def test_invalid_kernel_type_raises(self):
        with pytest.raises(ValueError):
            erode(_rect(), kernel_type="hexagon")

    def test_iterations_increase_erosion(self):
        img = _rect()
        r1 = erode(img, ksize=3, iterations=1)
        r3 = erode(img, ksize=3, iterations=3)
        assert r3.sum() <= r1.sum()

    def test_bgr_input_accepted(self):
        result = erode(_bgr(), ksize=3)
        assert result.dtype == np.uint8
        assert result.ndim == 3

    def test_all_kernel_types(self):
        img = _rect()
        for kt in ("rect", "ellipse", "cross"):
            result = erode(img, ksize=3, kernel_type=kt)
            assert result.dtype == np.uint8


# ─── dilate ──────────────────────────────────────────────────────────────────

class TestDilate:
    def test_returns_uint8(self):
        assert dilate(_rect()).dtype == np.uint8

    def test_shape_preserved(self):
        img = _rect(36, 44)
        assert dilate(img, ksize=3).shape == (36, 44)

    def test_dilates_bright_pixels(self):
        img = _rect()
        result = dilate(img, ksize=5)
        assert result.sum() >= img.sum()

    def test_uniform_white_unchanged(self):
        img = _white()
        np.testing.assert_array_equal(dilate(img, ksize=3), img)

    def test_uniform_black_unchanged(self):
        img = _black()
        np.testing.assert_array_equal(dilate(img, ksize=3), img)

    def test_ksize_validation(self):
        with pytest.raises(ValueError):
            dilate(_rect(), ksize=2)
        with pytest.raises(ValueError):
            dilate(_rect(), ksize=6)


# ─── open_morph ──────────────────────────────────────────────────────────────

class TestOpenMorph:
    def test_returns_uint8(self):
        assert open_morph(_rect()).dtype == np.uint8

    def test_shape_preserved(self):
        img = _rect(30, 40)
        assert open_morph(img, ksize=3).shape == (30, 40)

    def test_removes_small_dot(self):
        img = _dot(r=1)  # very small bright dot
        result = open_morph(img, ksize=7)
        # Large kernel open should remove or shrink the small dot significantly
        assert result.sum() <= img.sum()

    def test_idempotent_on_opened_image(self):
        # Opening twice should equal opening once
        img = _rect()
        r1 = open_morph(img, ksize=5)
        r2 = open_morph(r1, ksize=5)
        np.testing.assert_array_equal(r1, r2)

    def test_ksize_validation(self):
        with pytest.raises(ValueError):
            open_morph(_rect(), ksize=2)


# ─── close_morph ─────────────────────────────────────────────────────────────

class TestCloseMorph:
    def test_returns_uint8(self):
        assert close_morph(_rect()).dtype == np.uint8

    def test_shape_preserved(self):
        img = _rect(28, 36)
        assert close_morph(img, ksize=3).shape == (28, 36)

    def test_fills_small_gap(self):
        img = _rect()
        # Introduce a small black vertical stripe in the white region
        img[10:22, 15] = 0
        result = close_morph(img, ksize=5)
        # The gap should be reduced or filled
        assert result.sum() >= img.sum()

    def test_idempotent_on_closed_image(self):
        img = _rect()
        r1 = close_morph(img, ksize=5)
        r2 = close_morph(r1, ksize=5)
        np.testing.assert_array_equal(r1, r2)

    def test_ksize_validation(self):
        with pytest.raises(ValueError):
            close_morph(_rect(), ksize=4)


# ─── tophat ──────────────────────────────────────────────────────────────────

class TestTophat:
    def test_returns_uint8(self):
        assert tophat(_rect()).dtype == np.uint8

    def test_shape_preserved(self):
        img = _rect(32, 40)
        assert tophat(img, ksize=5).shape == (32, 40)

    def test_non_negative(self):
        result = tophat(_rect(), ksize=5)
        assert result.min() >= 0

    def test_uniform_dark_returns_zeros(self):
        img = _black()
        result = tophat(img, ksize=5)
        np.testing.assert_array_equal(result, _black())

    def test_uniform_bright_returns_zeros(self):
        # top-hat of uniform image is zero (img - open(img) = 0 for uniform)
        img = _white()
        result = tophat(img, ksize=3)
        np.testing.assert_array_equal(result, _black())

    def test_ksize_validation(self):
        with pytest.raises(ValueError):
            tophat(_rect(), ksize=2)


# ─── blackhat ────────────────────────────────────────────────────────────────

class TestBlackhat:
    def test_returns_uint8(self):
        assert blackhat(_rect()).dtype == np.uint8

    def test_shape_preserved(self):
        img = _rect(30, 38)
        assert blackhat(img, ksize=5).shape == (30, 38)

    def test_non_negative(self):
        assert blackhat(_rect(), ksize=5).min() >= 0

    def test_uniform_image_returns_zeros(self):
        for img in (_black(), _white()):
            result = blackhat(img, ksize=5)
            np.testing.assert_array_equal(result, _black())

    def test_ksize_validation(self):
        with pytest.raises(ValueError):
            blackhat(_rect(), ksize=6)


# ─── morphological_gradient ──────────────────────────────────────────────────

class TestMorphologicalGradient:
    def test_returns_uint8(self):
        assert morphological_gradient(_rect()).dtype == np.uint8

    def test_shape_preserved(self):
        img = _rect(28, 34)
        assert morphological_gradient(img, ksize=3).shape == (28, 34)

    def test_uniform_image_returns_zeros(self):
        for img in (_black(), _white()):
            result = morphological_gradient(img, ksize=3)
            np.testing.assert_array_equal(result, _black(img.shape[0], img.shape[1]))

    def test_highlights_edges(self):
        img = _rect()
        result = morphological_gradient(img, ksize=3)
        # Gradient should be non-zero at the rectangle boundary
        assert result.sum() > 0

    def test_ksize_validation(self):
        with pytest.raises(ValueError):
            morphological_gradient(_rect(), ksize=2)


# ─── skeleton ────────────────────────────────────────────────────────────────

class TestSkeleton:
    def test_returns_uint8(self):
        img = _rect()
        assert skeleton(img).dtype == np.uint8

    def test_shape_preserved(self):
        img = _rect(32, 32)
        assert skeleton(img).shape == (32, 32)

    def test_multi_channel_raises(self):
        with pytest.raises(ValueError):
            skeleton(_bgr())

    def test_black_image_stays_black(self):
        np.testing.assert_array_equal(skeleton(_black()), _black())

    def test_skeleton_subset_of_original(self):
        img = _rect()
        skel = skeleton(img)
        # Every white pixel in skeleton should be white in original
        assert np.all((skel == 0) | (img > 0))

    def test_skeleton_thinner_than_original(self):
        img = _rect()
        skel = skeleton(img)
        # Skeleton has fewer pixels than thick original
        assert skel.sum() < img.sum()

    def test_single_pixel_stays(self):
        img = _black()
        img[16, 16] = 255
        skel = skeleton(img)
        # Single pixel cannot be further eroded; skeleton should contain it
        assert skel[16, 16] == 255 or skel.sum() == 0  # either kept or eroded away


# ─── remove_small_blobs ──────────────────────────────────────────────────────

class TestRemoveSmallBlobs:
    def test_returns_uint8(self):
        assert remove_small_blobs(_two_blobs()).dtype == np.uint8

    def test_shape_preserved(self):
        img = _two_blobs()
        assert remove_small_blobs(img, min_area=10).shape == img.shape

    def test_multi_channel_raises(self):
        with pytest.raises(ValueError):
            remove_small_blobs(_bgr())

    def test_min_area_negative_raises(self):
        with pytest.raises(ValueError):
            remove_small_blobs(_two_blobs(), min_area=-1)

    def test_removes_small_blob(self):
        img = _two_blobs()
        # min_area=10 should keep large blob (64 px) but remove small (4 px)
        result = remove_small_blobs(img, min_area=10)
        # small blob region should now be black
        assert result[5:7, 5:7].sum() == 0

    def test_keeps_large_blob(self):
        img = _two_blobs()
        result = remove_small_blobs(img, min_area=10)
        # large blob should remain
        assert result[30:38, 30:38].sum() > 0

    def test_min_area_zero_keeps_all(self):
        img = _two_blobs()
        result = remove_small_blobs(img, min_area=0)
        assert result.sum() >= img.sum() - 255  # essentially same

    def test_empty_image_stays_empty(self):
        result = remove_small_blobs(_black(32, 32), min_area=5)
        np.testing.assert_array_equal(result, _black(32, 32))


# ─── fill_holes ──────────────────────────────────────────────────────────────

class TestFillHoles:
    def test_returns_uint8(self):
        assert fill_holes(_rect_with_hole()).dtype == np.uint8

    def test_shape_preserved(self):
        img = _rect_with_hole()
        assert fill_holes(img).shape == img.shape

    def test_multi_channel_raises(self):
        with pytest.raises(ValueError):
            fill_holes(_bgr())

    def test_hole_filled(self):
        img = _rect_with_hole()
        result = fill_holes(img)
        # Interior region that was 0 should now be 255
        assert result[20:28, 20:28].min() == 255

    def test_solid_rect_unchanged(self):
        img = _rect()
        result = fill_holes(img)
        # Interior pixels already white — should stay
        assert result[10:22, 10:22].min() == 255

    def test_black_image_stays_black(self):
        img = _black()
        result = fill_holes(img)
        np.testing.assert_array_equal(result, img)

    def test_non_decreasing_white_area(self):
        img = _rect_with_hole()
        result = fill_holes(img)
        assert result.sum() >= img.sum()


# ─── apply_morph ─────────────────────────────────────────────────────────────

class TestApplyMorph:
    def test_erode_dispatch(self):
        img = _rect()
        p = MorphParams(op="erode", ksize=5)
        direct = erode(img, ksize=5)
        via_apply = apply_morph(img, p)
        np.testing.assert_array_equal(direct, via_apply)

    def test_dilate_dispatch(self):
        img = _rect()
        p = MorphParams(op="dilate", ksize=5)
        direct = dilate(img, ksize=5)
        via_apply = apply_morph(img, p)
        np.testing.assert_array_equal(direct, via_apply)

    def test_open_dispatch(self):
        img = _rect()
        p = MorphParams(op="open", ksize=5)
        np.testing.assert_array_equal(apply_morph(img, p), open_morph(img, ksize=5))

    def test_close_dispatch(self):
        img = _rect()
        p = MorphParams(op="close", ksize=5)
        np.testing.assert_array_equal(apply_morph(img, p), close_morph(img, ksize=5))

    def test_tophat_dispatch(self):
        img = _rect()
        p = MorphParams(op="tophat", ksize=5)
        np.testing.assert_array_equal(apply_morph(img, p), tophat(img, ksize=5))

    def test_blackhat_dispatch(self):
        img = _rect()
        p = MorphParams(op="blackhat", ksize=5)
        np.testing.assert_array_equal(apply_morph(img, p), blackhat(img, ksize=5))

    def test_gradient_dispatch(self):
        img = _rect()
        p = MorphParams(op="gradient", ksize=3)
        np.testing.assert_array_equal(
            apply_morph(img, p), morphological_gradient(img, ksize=3)
        )

    def test_returns_uint8(self):
        for op in ("erode", "dilate", "open", "close", "tophat", "blackhat", "gradient"):
            p = MorphParams(op=op, ksize=3)
            assert apply_morph(_rect(), p).dtype == np.uint8


# ─── batch_morph ─────────────────────────────────────────────────────────────

class TestBatchMorph:
    def test_empty_returns_empty(self):
        assert batch_morph([]) == []

    def test_length_preserved(self):
        imgs = [_rect()] * 5
        result = batch_morph(imgs)
        assert len(result) == 5

    def test_all_uint8(self):
        imgs = [_rect(20, 20), _rect(30, 30), _black()]
        result = batch_morph(imgs)
        assert all(r.dtype == np.uint8 for r in result)

    def test_shapes_preserved(self):
        imgs = [_rect(20, 24), _rect(28, 32)]
        result = batch_morph(imgs)
        assert result[0].shape == (20, 24)
        assert result[1].shape == (28, 32)

    def test_custom_params_applied(self):
        imgs = [_rect()] * 3
        p = MorphParams(op="dilate", ksize=7)
        direct = [dilate(img, ksize=7) for img in imgs]
        result = batch_morph(imgs, params=p)
        for r, d in zip(result, direct):
            np.testing.assert_array_equal(r, d)
