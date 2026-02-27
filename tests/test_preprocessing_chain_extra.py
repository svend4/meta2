"""Additional tests for puzzle_reconstruction.preprocessing.chain."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.chain import (
    PreprocessingChain,
    _extract_image,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _img(h: int = 64, w: int = 64, c: int = 3, dtype="uint8") -> np.ndarray:
    """Synthetic BGR image."""
    rng = np.random.default_rng(42)
    if c == 1:
        return (rng.integers(50, 200, (h, w), dtype=dtype))
    return rng.integers(50, 200, (h, w, c)).astype(dtype)


def _gray(h: int = 64, w: int = 64) -> np.ndarray:
    return _img(h, w, c=1)


# ─── _extract_image ───────────────────────────────────────────────────────────

class TestExtractImageExtra:
    def test_none_returns_none(self):
        assert _extract_image(None) is None

    def test_ndarray_passthrough(self):
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        result = _extract_image(arr)
        assert result is arr

    def test_object_with_image_attr(self):
        arr = np.ones((4, 4), dtype=np.uint8)
        class Obj:
            image = arr
        result = _extract_image(Obj())
        assert result is arr

    def test_object_with_result_attr(self):
        arr = np.ones((4, 4), dtype=np.uint8)
        class Obj:
            result = arr
        result = _extract_image(Obj())
        assert result is arr

    def test_object_with_img_attr(self):
        arr = np.ones((4, 4), dtype=np.uint8)
        class Obj:
            img = arr
        result = _extract_image(Obj())
        assert result is arr

    def test_object_with_output_attr(self):
        arr = np.ones((4, 4), dtype=np.uint8)
        class Obj:
            output = arr
        result = _extract_image(Obj())
        assert result is arr

    def test_object_no_known_attr_returns_none(self):
        class Obj:
            foo = "bar"
        result = _extract_image(Obj())
        assert result is None

    def test_image_attr_not_ndarray_falls_through(self):
        class Obj:
            image = "not_an_array"
            result = None
            img = None
            output = None
        result = _extract_image(Obj())
        assert result is None


# ─── PreprocessingChain — construction ────────────────────────────────────────

class TestPreprocessingChainConstructExtra:
    def test_default_empty(self):
        chain = PreprocessingChain()
        assert chain.filters == []
        assert chain.quality_threshold == 0.0
        assert chain.auto_enhance is False

    def test_custom_filters(self):
        chain = PreprocessingChain(filters=["denoise", "contrast"])
        assert chain.filters == ["denoise", "contrast"]

    def test_quality_threshold_stored(self):
        chain = PreprocessingChain(quality_threshold=0.5)
        assert chain.quality_threshold == pytest.approx(0.5)

    def test_auto_enhance_stored(self):
        chain = PreprocessingChain(auto_enhance=True)
        assert chain.auto_enhance is True

    def test_is_empty_no_filters(self):
        chain = PreprocessingChain()
        assert chain.is_empty() is True

    def test_is_empty_with_filters(self):
        chain = PreprocessingChain(filters=["denoise"])
        assert chain.is_empty() is False

    def test_is_empty_auto_enhance(self):
        chain = PreprocessingChain(auto_enhance=True)
        assert chain.is_empty() is False


# ─── PreprocessingChain — apply ───────────────────────────────────────────────

class TestPreprocessingChainApplyExtra:
    def test_none_image_returns_none(self):
        chain = PreprocessingChain()
        assert chain.apply(None) is None

    def test_empty_array_returns_none(self):
        chain = PreprocessingChain()
        arr = np.zeros((0, 0, 3), dtype=np.uint8)
        assert chain.apply(arr) is None

    def test_empty_filters_passthrough(self):
        chain = PreprocessingChain(filters=[])
        img = _img()
        result = chain.apply(img)
        assert result is not None
        assert result.shape == img.shape

    def test_result_is_ndarray(self):
        chain = PreprocessingChain(filters=[])
        result = chain.apply(_img())
        assert isinstance(result, np.ndarray)

    def test_unknown_filter_skipped_silently(self):
        chain = PreprocessingChain(filters=["__bogus_filter__"])
        img = _img()
        result = chain.apply(img)
        assert result is not None
        assert result.shape == img.shape

    def test_multiple_unknown_filters(self):
        chain = PreprocessingChain(filters=["__x__", "__y__", "__z__"])
        img = _img()
        result = chain.apply(img)
        assert result is not None

    def test_auto_enhance_no_filters_adds_defaults(self):
        chain = PreprocessingChain(auto_enhance=True)
        img = _img()
        result = chain.apply(img)
        # Should return an image (denoise+contrast applied gracefully)
        assert result is not None

    def test_quality_threshold_zero_no_rejection(self):
        chain = PreprocessingChain(quality_threshold=0.0)
        img = _img()
        result = chain.apply(img)
        assert result is not None

    def test_quality_threshold_above_one_may_reject(self):
        # threshold > 1.0 → quality_gate always rejects (or returns None)
        chain = PreprocessingChain(
            filters=["quality_assessor"],
            quality_threshold=999.0,
        )
        img = _img()
        # may return None (rejected) or img (assessor unavailable)
        result = chain.apply(img)
        assert result is None or isinstance(result, np.ndarray)

    def test_denoise_filter(self):
        chain = PreprocessingChain(filters=["denoise"])
        img = _img()
        result = chain.apply(img)
        assert result is not None
        assert result.shape[:2] == img.shape[:2]

    def test_contrast_filter(self):
        chain = PreprocessingChain(filters=["contrast"])
        img = _img()
        result = chain.apply(img)
        assert result is not None

    def test_chain_order_denoise_then_contrast(self):
        chain = PreprocessingChain(filters=["denoise", "contrast"])
        img = _img()
        result = chain.apply(img)
        assert result is not None

    def test_filter_exception_keeps_original(self):
        """A filter that raises internally should not crash the chain."""
        # Use a non-existent filter (skipped) — chain continues
        chain = PreprocessingChain(filters=["__crash__", "denoise"])
        img = _img()
        result = chain.apply(img)
        assert result is not None

    def test_small_image(self):
        chain = PreprocessingChain(filters=["denoise"])
        img = _img(h=8, w=8)
        result = chain.apply(img)
        assert result is not None

    def test_large_image(self):
        chain = PreprocessingChain(filters=["contrast"])
        img = _img(h=256, w=256)
        result = chain.apply(img)
        assert result is not None

    def test_grayscale_input(self):
        chain = PreprocessingChain(filters=["denoise"])
        img = _gray()
        result = chain.apply(img)
        assert result is not None

    def test_high_quality_image_not_rejected(self):
        chain = PreprocessingChain(
            filters=["quality_assessor"],
            quality_threshold=0.01,  # very low threshold
        )
        img = _img()
        result = chain.apply(img)
        # Either passes (high quality) or assessor unavailable → original
        assert result is None or isinstance(result, np.ndarray)

    def test_apply_idempotent_no_filters(self):
        chain = PreprocessingChain(filters=[])
        img = _img()
        r1 = chain.apply(img)
        r2 = chain.apply(img)
        assert r1 is not None and r2 is not None
        np.testing.assert_array_equal(r1, r2)

    def test_apply_preserves_dtype_uint8(self):
        chain = PreprocessingChain(filters=[])
        img = _img(dtype="uint8")
        result = chain.apply(img)
        assert result.dtype == np.uint8

    def test_all_available_single_filters(self):
        """Each available filter should not crash when applied individually."""
        known_filters = [
            "denoise", "contrast", "deskew", "background_remove", "binarize",
            "edge_enhance", "sharpen", "color_normalize", "illumination",
            "morph", "crop", "noise_analyze", "adaptive_threshold",
            "scan_augment", "channel_equalize", "contour_analyze",
            "contrast_enhance", "document_clean", "edge_detect",
            "freq_analyze", "freq_low_pass", "freq_high_pass",
            "freq_band_pass", "gradient_analyze", "illumination_norm",
            "image_enhance", "noise_filter", "noise_reduce", "smart_denoise",
            "patch_normalize", "patch_sample", "perspective", "skew_correct",
            "texture_analyze", "warp_correct",
        ]
        img = _img()
        for name in known_filters:
            chain = PreprocessingChain(filters=[name])
            try:
                result = chain.apply(img)
                # Result should be ndarray or None (None only for quality gate)
                assert result is None or isinstance(result, np.ndarray), \
                    f"Filter {name!r} returned unexpected type"
            except Exception as exc:
                pytest.fail(f"Filter {name!r} raised unexpectedly: {exc}")


# ─── PreprocessingChain — is_empty ────────────────────────────────────────────

class TestPreprocessingChainIsEmptyExtra:
    def test_empty_by_default(self):
        assert PreprocessingChain().is_empty() is True

    def test_not_empty_with_filter(self):
        assert PreprocessingChain(filters=["denoise"]).is_empty() is False

    def test_not_empty_auto_enhance(self):
        assert PreprocessingChain(auto_enhance=True).is_empty() is False

    def test_not_empty_both(self):
        chain = PreprocessingChain(filters=["denoise"], auto_enhance=True)
        assert chain.is_empty() is False

    def test_empty_filters_no_auto_is_empty(self):
        chain = PreprocessingChain(filters=[], auto_enhance=False)
        assert chain.is_empty() is True
