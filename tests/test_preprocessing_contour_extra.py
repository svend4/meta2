"""Extra tests for puzzle_reconstruction/preprocessing/contour.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.contour import (
    extract_contour,
    rdp_simplify,
    split_contour_to_edges,
    resample_curve,
    normalize_contour,
)
from puzzle_reconstruction.models import EdgeSide


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rect_mask(h=100, w=100, margin=10):
    """Create a rectangular white mask with margin."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[margin:h - margin, margin:w - margin] = 255
    return mask


def _circle_mask(h=100, w=100, radius=30):
    """Create a circular mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask[dist <= radius] = 255
    return mask


# ─── extract_contour ────────────────────────────────────────────────────────

class TestExtractContourExtra:
    def test_rectangle(self):
        mask = _rect_mask()
        c = extract_contour(mask)
        assert c.ndim == 2 and c.shape[1] == 2
        assert len(c) > 4

    def test_circle(self):
        mask = _circle_mask()
        c = extract_contour(mask)
        assert c.ndim == 2 and c.shape[1] == 2

    def test_empty_mask_raises(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        with pytest.raises(ValueError):
            extract_contour(mask)

    def test_dtype_float32(self):
        mask = _rect_mask()
        c = extract_contour(mask)
        assert c.dtype == np.float32


# ─── rdp_simplify ───────────────────────────────────────────────────────────

class TestRdpSimplifyExtra:
    def test_reduces_points(self):
        mask = _circle_mask(200, 200, 80)
        c = extract_contour(mask)
        simplified = rdp_simplify(c, epsilon_ratio=0.01)
        assert len(simplified) < len(c)

    def test_small_epsilon_preserves(self):
        mask = _rect_mask()
        c = extract_contour(mask)
        simplified = rdp_simplify(c, epsilon_ratio=0.001)
        assert len(simplified) <= len(c)

    def test_output_shape(self):
        mask = _rect_mask()
        c = extract_contour(mask)
        simplified = rdp_simplify(c)
        assert simplified.ndim == 2 and simplified.shape[1] == 2

    def test_dtype_float32(self):
        mask = _rect_mask()
        c = extract_contour(mask)
        simplified = rdp_simplify(c)
        assert simplified.dtype == np.float32


# ─── split_contour_to_edges ─────────────────────────────────────────────────

class TestSplitContourToEdgesExtra:
    def test_four_sides(self):
        mask = _rect_mask()
        c = extract_contour(mask)
        edges = split_contour_to_edges(c, n_sides=4)
        assert len(edges) == 4

    def test_edge_labels(self):
        mask = _rect_mask()
        c = extract_contour(mask)
        edges = split_contour_to_edges(c, n_sides=4)
        labels = {e[1] for e in edges}
        # Should have some edge side labels
        assert len(labels) >= 1
        for lbl in labels:
            assert isinstance(lbl, EdgeSide)

    def test_all_points_covered(self):
        mask = _rect_mask()
        c = extract_contour(mask)
        edges = split_contour_to_edges(c, n_sides=4)
        total_pts = sum(len(e[0]) for e in edges)
        # Each edge includes endpoints, so total >= original
        assert total_pts >= len(c)


# ─── resample_curve ─────────────────────────────────────────────────────────

class TestResampleCurveExtra:
    def test_output_length(self):
        curve = np.array([[0, 0], [10, 0], [10, 10]], dtype=np.float32)
        out = resample_curve(curve, n_points=50)
        assert len(out) == 50

    def test_default_256(self):
        curve = np.array([[0, 0], [100, 0], [100, 100]], dtype=np.float32)
        out = resample_curve(curve)
        assert len(out) == 256

    def test_single_point(self):
        curve = np.array([[5, 5]], dtype=np.float32)
        out = resample_curve(curve, n_points=10)
        assert len(out) == 10
        # All points should be the same
        assert np.allclose(out, [5, 5])

    def test_two_identical_points(self):
        curve = np.array([[5, 5], [5, 5]], dtype=np.float32)
        out = resample_curve(curve, n_points=10)
        assert len(out) == 10


# ─── normalize_contour ──────────────────────────────────────────────────────

class TestNormalizeContourExtra:
    def test_centered(self):
        contour = np.array([[0, 0], [10, 0], [10, 10], [0, 10]],
                           dtype=np.float32)
        norm, centroid, scale = normalize_contour(contour)
        assert np.allclose(norm.mean(axis=0), [0, 0], atol=1e-5)

    def test_scale_factor(self):
        contour = np.array([[0, 0], [10, 0], [10, 10], [0, 10]],
                           dtype=np.float32)
        norm, centroid, scale = normalize_contour(contour)
        assert scale > 0

    def test_single_point(self):
        contour = np.array([[5, 5]], dtype=np.float32)
        norm, centroid, scale = normalize_contour(contour)
        assert scale == 1.0  # Degenerate case

    def test_centroid_correct(self):
        contour = np.array([[0, 0], [10, 0], [10, 10], [0, 10]],
                           dtype=np.float32)
        _, centroid, _ = normalize_contour(contour)
        assert np.allclose(centroid, [5, 5])
