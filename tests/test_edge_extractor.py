"""Tests for puzzle_reconstruction.algorithms.edge_extractor."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.algorithms.edge_extractor import (
    EdgeSegment,
    FragmentEdges,
    batch_extract_edges,
    compute_edge_length,
    detect_boundary,
    extract_edge_points,
    extract_fragment_edges,
    simplify_edge,
    split_edge_by_side,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _white_square(h: int = 32, w: int = 32) -> np.ndarray:
    img = np.zeros((h, w), dtype=np.uint8)
    img[4:h - 4, 4:w - 4] = 255
    return img


def _pts(n: int = 10) -> np.ndarray:
    return np.column_stack([
        np.linspace(0, 10, n),
        np.linspace(0, 10, n),
    ]).astype(np.float32)


# ─── EdgeSegment ─────────────────────────────────────────────────────────────

class TestEdgeSegment:
    def test_valid_side_stored(self):
        seg = EdgeSegment(points=_pts(), side="top", length=5.0)
        assert seg.side == "top"

    def test_default_side_unknown(self):
        seg = EdgeSegment(points=_pts())
        assert seg.side == "unknown"

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError):
            EdgeSegment(points=_pts(), side="diagonal")

    def test_negative_length_raises(self):
        with pytest.raises(ValueError):
            EdgeSegment(points=_pts(), length=-0.1)

    def test_len(self):
        seg = EdgeSegment(points=_pts(7))
        assert len(seg) == 7

    def test_all_valid_sides(self):
        for side in ("top", "bottom", "left", "right", "unknown"):
            seg = EdgeSegment(points=_pts(), side=side)
            assert seg.side == side

    def test_zero_length_allowed(self):
        seg = EdgeSegment(points=_pts(), length=0.0)
        assert seg.length == pytest.approx(0.0)

    def test_params_stored(self):
        seg = EdgeSegment(points=_pts(), params={"epsilon": 1.0})
        assert seg.params["epsilon"] == pytest.approx(1.0)


# ─── FragmentEdges ────────────────────────────────────────────────────────────

class TestFragmentEdges:
    def test_len(self):
        segs = [EdgeSegment(points=_pts(), side="top")]
        fe = FragmentEdges(segments=segs, n_segments=1)
        assert len(fe) == 1

    def test_negative_n_segments_raises(self):
        with pytest.raises(ValueError):
            FragmentEdges(segments=[], n_segments=-1)

    def test_zero_segments_ok(self):
        fe = FragmentEdges(segments=[], n_segments=0)
        assert len(fe) == 0

    def test_params_stored(self):
        fe = FragmentEdges(segments=[], n_segments=0, params={"k": 42})
        assert fe.params["k"] == 42


# ─── detect_boundary ─────────────────────────────────────────────────────────

class TestDetectBoundary:
    def test_returns_uint8(self):
        result = detect_boundary(_white_square())
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        img = _white_square(24, 32)
        result = detect_boundary(img)
        assert result.shape == (24, 32)

    def test_values_only_0_or_255(self):
        result = detect_boundary(_white_square())
        assert set(np.unique(result)).issubset({0, 255})

    def test_black_image_no_boundary(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        result = detect_boundary(img)
        assert np.all(result == 0)

    def test_invalid_threshold_above_raises(self):
        with pytest.raises(ValueError):
            detect_boundary(_white_square(), threshold=256)

    def test_invalid_threshold_below_raises(self):
        with pytest.raises(ValueError):
            detect_boundary(_white_square(), threshold=-1)

    def test_color_image_accepted(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[4:28, 4:28] = 200
        result = detect_boundary(img)
        assert result.shape == (32, 32)

    def test_boundary_pixels_detected(self):
        result = detect_boundary(_white_square())
        assert np.any(result > 0)


# ─── extract_edge_points ─────────────────────────────────────────────────────

class TestExtractEdgePoints:
    def test_returns_float32(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10, 10] = 255
        pts = extract_edge_points(mask)
        assert pts.dtype == np.float32

    def test_shape_n_2(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[5, 5] = 255
        mask[6, 7] = 255
        pts = extract_edge_points(mask)
        assert pts.ndim == 2
        assert pts.shape[1] == 2

    def test_empty_mask_returns_zero_rows(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        pts = extract_edge_points(mask)
        assert pts.shape == (0, 2)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            extract_edge_points(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_count_correct(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[5, 5] = 255
        mask[10, 15] = 255
        pts = extract_edge_points(mask)
        assert len(pts) == 2

    def test_xy_order(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10, 20] = 255  # row=10 → y=10, col=20 → x=20
        pts = extract_edge_points(mask)
        assert pts[0, 0] == pytest.approx(20.0)  # x
        assert pts[0, 1] == pytest.approx(10.0)  # y


# ─── split_edge_by_side ──────────────────────────────────────────────────────

class TestSplitEdgeBySide:
    def test_returns_four_sides(self):
        pts = np.array([[16.0, 0.0]], dtype=np.float32)
        result = split_edge_by_side(pts, (32, 32))
        assert set(result.keys()) == {"top", "bottom", "left", "right"}

    def test_top_point(self):
        pts = np.array([[16.0, 0.0]], dtype=np.float32)
        result = split_edge_by_side(pts, (32, 32))
        assert len(result["top"]) == 1

    def test_bottom_point(self):
        pts = np.array([[16.0, 31.0]], dtype=np.float32)
        result = split_edge_by_side(pts, (32, 32))
        assert len(result["bottom"]) == 1

    def test_left_point(self):
        pts = np.array([[0.0, 16.0]], dtype=np.float32)
        result = split_edge_by_side(pts, (32, 32))
        assert len(result["left"]) == 1

    def test_right_point(self):
        pts = np.array([[31.0, 16.0]], dtype=np.float32)
        result = split_edge_by_side(pts, (32, 32))
        assert len(result["right"]) == 1

    def test_invalid_shape_zero_h_raises(self):
        pts = np.array([[1.0, 2.0]], dtype=np.float32)
        with pytest.raises(ValueError):
            split_edge_by_side(pts, (0, 32))

    def test_invalid_shape_zero_w_raises(self):
        pts = np.array([[1.0, 2.0]], dtype=np.float32)
        with pytest.raises(ValueError):
            split_edge_by_side(pts, (32, 0))

    def test_non_2d_points_raises(self):
        with pytest.raises(ValueError):
            split_edge_by_side(np.zeros((4,), dtype=np.float32), (32, 32))

    def test_all_arrays_float32(self):
        pts = np.array([[0.0, 0.0], [31.0, 31.0]], dtype=np.float32)
        result = split_edge_by_side(pts, (32, 32))
        for arr in result.values():
            assert arr.dtype == np.float32


# ─── compute_edge_length ─────────────────────────────────────────────────────

class TestComputeEdgeLength:
    def test_zero_for_empty(self):
        pts = np.zeros((0, 2), dtype=np.float32)
        assert compute_edge_length(pts) == pytest.approx(0.0)

    def test_zero_for_single_point(self):
        pts = np.array([[5.0, 5.0]], dtype=np.float32)
        assert compute_edge_length(pts) == pytest.approx(0.0)

    def test_known_length(self):
        pts = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        assert compute_edge_length(pts) == pytest.approx(5.0)

    def test_cumulative_length(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        assert compute_edge_length(pts) == pytest.approx(2.0)

    def test_nonnegative(self):
        assert compute_edge_length(_pts(5)) >= 0.0

    def test_returns_float(self):
        assert isinstance(compute_edge_length(_pts(3)), float)


# ─── simplify_edge ───────────────────────────────────────────────────────────

class TestSimplifyEdge:
    def test_returns_float32(self):
        result = simplify_edge(_pts(10))
        assert result.dtype == np.float32

    def test_shape_n_2(self):
        result = simplify_edge(_pts(10))
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_simplified_le_original(self):
        pts = _pts(30)
        result = simplify_edge(pts, epsilon=0.5)
        assert len(result) <= len(pts)

    def test_two_points_unchanged(self):
        pts = np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float32)
        result = simplify_edge(pts)
        assert result.shape == (2, 2)

    def test_negative_epsilon_raises(self):
        with pytest.raises(ValueError):
            simplify_edge(_pts(), epsilon=-1.0)

    def test_zero_epsilon_returns_all(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 0.0]], dtype=np.float32)
        result = simplify_edge(pts, epsilon=0.0)
        assert len(result) >= 2


# ─── extract_fragment_edges ──────────────────────────────────────────────────

class TestExtractFragmentEdges:
    def test_returns_fragment_edges(self):
        result = extract_fragment_edges(_white_square())
        assert isinstance(result, FragmentEdges)

    def test_four_segments(self):
        result = extract_fragment_edges(_white_square())
        assert result.n_segments == 4

    def test_segments_have_valid_sides(self):
        result = extract_fragment_edges(_white_square())
        sides = {seg.side for seg in result.segments}
        assert sides == {"top", "bottom", "left", "right"}

    def test_lengths_nonnegative(self):
        result = extract_fragment_edges(_white_square())
        for seg in result.segments:
            assert seg.length >= 0.0

    def test_color_image_accepted(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[4:28, 4:28] = 200
        result = extract_fragment_edges(img)
        assert isinstance(result, FragmentEdges)

    def test_params_stored(self):
        result = extract_fragment_edges(_white_square(), threshold=10, epsilon=1.0)
        assert result.params.get("threshold") == 10
        assert result.params.get("epsilon") == pytest.approx(1.0)


# ─── batch_extract_edges ─────────────────────────────────────────────────────

class TestBatchExtractEdges:
    def test_returns_list(self):
        result = batch_extract_edges([_white_square()])
        assert isinstance(result, list)

    def test_length_matches(self):
        result = batch_extract_edges([_white_square(), _white_square(16, 16)])
        assert len(result) == 2

    def test_empty_input_returns_empty(self):
        assert batch_extract_edges([]) == []

    def test_all_fragment_edges(self):
        result = batch_extract_edges([_white_square()])
        assert all(isinstance(r, FragmentEdges) for r in result)
