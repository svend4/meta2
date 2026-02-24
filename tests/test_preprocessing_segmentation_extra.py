"""Extra tests for puzzle_reconstruction/preprocessing/segmentation.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.segmentation import (
    segment_fragment,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bgr_with_fragment(h=100, w=100):
    """White background with dark rectangle simulating a fragment."""
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    img[20:80, 20:80] = 50
    return img


def _gray_with_fragment(h=100, w=100):
    """Grayscale version."""
    img = np.full((h, w), 240, dtype=np.uint8)
    img[20:80, 20:80] = 50
    return img


# ─── segment_fragment ─────────────────────────────────────────────────────────

class TestSegmentFragmentExtra:
    def test_otsu_bgr(self):
        img = _bgr_with_fragment()
        mask = segment_fragment(img, method="otsu")
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})

    def test_otsu_gray(self):
        img = _gray_with_fragment()
        mask = segment_fragment(img, method="otsu")
        assert mask.ndim == 2

    def test_adaptive(self):
        img = _bgr_with_fragment()
        mask = segment_fragment(img, method="adaptive")
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            segment_fragment(_bgr_with_fragment(), method="bad")

    def test_has_foreground(self):
        img = _bgr_with_fragment()
        mask = segment_fragment(img, method="otsu")
        # Should have some foreground pixels
        assert mask.max() == 255

    def test_morph_kernel(self):
        img = _bgr_with_fragment()
        mask = segment_fragment(img, method="otsu", morph_kernel=5)
        assert mask.dtype == np.uint8
