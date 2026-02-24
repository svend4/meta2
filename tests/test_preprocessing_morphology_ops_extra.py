"""Extra tests for puzzle_reconstruction/preprocessing/morphology_ops.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.morphology_ops import (
    MorphParams,
    erode,
    dilate,
    open_morph,
    close_morph,
    tophat,
    blackhat,
    morphological_gradient,
    skeleton,
    remove_small_blobs,
    fill_holes,
    apply_morph,
    batch_morph,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=50, w=50, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=50, w=50, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _binary(h=50, w=50):
    """Binary image with a filled rectangle."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[10:40, 10:40] = 255
    return img


# ─── MorphParams ──────────────────────────────────────────────────────────────

class TestMorphParamsExtra:
    def test_defaults(self):
        p = MorphParams()
        assert p.op == "open"
        assert p.kernel_type == "rect"
        assert p.ksize == 3
        assert p.iterations == 1

    def test_valid_ops(self):
        for op in ("erode", "dilate", "open", "close", "tophat", "blackhat", "gradient"):
            MorphParams(op=op)

    def test_valid_kernels(self):
        for kt in ("rect", "ellipse", "cross"):
            MorphParams(kernel_type=kt)

    def test_invalid_op_raises(self):
        with pytest.raises(ValueError):
            MorphParams(op="bad")

    def test_invalid_kernel_raises(self):
        with pytest.raises(ValueError):
            MorphParams(kernel_type="bad")

    def test_small_ksize_raises(self):
        with pytest.raises(ValueError):
            MorphParams(ksize=1)

    def test_even_ksize_raises(self):
        with pytest.raises(ValueError):
            MorphParams(ksize=4)

    def test_zero_iterations_raises(self):
        with pytest.raises(ValueError):
            MorphParams(iterations=0)


# ─── erode / dilate ───────────────────────────────────────────────────────────

class TestErodeExtra:
    def test_shape_preserved(self):
        out = erode(_gray())
        assert out.shape == (50, 50)

    def test_dtype_uint8(self):
        assert erode(_gray()).dtype == np.uint8

    def test_bgr(self):
        out = erode(_bgr())
        assert out.shape == (50, 50, 3)

    def test_bad_ksize_raises(self):
        with pytest.raises(ValueError):
            erode(_gray(), ksize=2)


class TestDilateExtra:
    def test_shape_preserved(self):
        out = dilate(_gray())
        assert out.shape == (50, 50)

    def test_dtype_uint8(self):
        assert dilate(_gray()).dtype == np.uint8

    def test_bad_kernel_type_raises(self):
        with pytest.raises(ValueError):
            dilate(_gray(), kernel_type="bad")


# ─── open / close ─────────────────────────────────────────────────────────────

class TestOpenMorphExtra:
    def test_shape_preserved(self):
        out = open_morph(_binary())
        assert out.shape == (50, 50)

    def test_dtype_uint8(self):
        assert open_morph(_binary()).dtype == np.uint8


class TestCloseMorphExtra:
    def test_shape_preserved(self):
        out = close_morph(_binary())
        assert out.shape == (50, 50)

    def test_dtype_uint8(self):
        assert close_morph(_binary()).dtype == np.uint8


# ─── tophat / blackhat ────────────────────────────────────────────────────────

class TestTophatExtra:
    def test_shape_preserved(self):
        out = tophat(_gray())
        assert out.shape == (50, 50)

    def test_dtype_uint8(self):
        assert tophat(_gray()).dtype == np.uint8


class TestBlackhatExtra:
    def test_shape_preserved(self):
        out = blackhat(_gray())
        assert out.shape == (50, 50)

    def test_dtype_uint8(self):
        assert blackhat(_gray()).dtype == np.uint8


# ─── morphological_gradient ──────────────────────────────────────────────────

class TestMorphologicalGradientExtra:
    def test_shape_preserved(self):
        out = morphological_gradient(_binary())
        assert out.shape == (50, 50)

    def test_uniform_is_zero(self):
        out = morphological_gradient(_gray())
        assert out.max() == 0


# ─── skeleton ─────────────────────────────────────────────────────────────────

class TestSkeletonExtra:
    def test_binary(self):
        out = skeleton(_binary())
        assert out.shape == (50, 50)
        assert out.dtype == np.uint8

    def test_3d_raises(self):
        with pytest.raises(ValueError):
            skeleton(_bgr())

    def test_all_zeros(self):
        out = skeleton(np.zeros((20, 20), dtype=np.uint8))
        assert out.max() == 0


# ─── remove_small_blobs ──────────────────────────────────────────────────────

class TestRemoveSmallBlobsExtra:
    def test_shape_preserved(self):
        out = remove_small_blobs(_binary())
        assert out.shape == (50, 50)

    def test_3d_raises(self):
        with pytest.raises(ValueError):
            remove_small_blobs(_bgr())

    def test_negative_area_raises(self):
        with pytest.raises(ValueError):
            remove_small_blobs(_binary(), min_area=-1)

    def test_large_area_removes_all(self):
        out = remove_small_blobs(_binary(), min_area=999999)
        assert out.max() == 0


# ─── fill_holes ───────────────────────────────────────────────────────────────

class TestFillHolesExtra:
    def test_shape_preserved(self):
        out = fill_holes(_binary())
        assert out.shape == (50, 50)

    def test_3d_raises(self):
        with pytest.raises(ValueError):
            fill_holes(_bgr())

    def test_no_holes(self):
        img = _binary()
        out = fill_holes(img)
        # No holes in solid rectangle
        assert np.array_equal(out, img)


# ─── apply_morph ──────────────────────────────────────────────────────────────

class TestApplyMorphExtra:
    def test_all_ops(self):
        for op in ("erode", "dilate", "open", "close", "tophat", "blackhat", "gradient"):
            p = MorphParams(op=op)
            out = apply_morph(_gray(), p)
            assert out.shape == (50, 50)

    def test_with_ellipse(self):
        p = MorphParams(op="erode", kernel_type="ellipse", ksize=5)
        out = apply_morph(_gray(), p)
        assert out.dtype == np.uint8


# ─── batch_morph ──────────────────────────────────────────────────────────────

class TestBatchMorphExtra:
    def test_empty(self):
        assert batch_morph([]) == []

    def test_length(self):
        results = batch_morph([_gray(), _gray()])
        assert len(results) == 2

    def test_default_params(self):
        results = batch_morph([_gray()])
        assert results[0].dtype == np.uint8
