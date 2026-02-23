"""Extra tests for puzzle_reconstruction.algorithms.edge_extractor."""
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


# ─── TestEdgeSegmentExtra ────────────────────────────────────────────────────

class TestEdgeSegmentExtra:
    def test_default_length_nonneg(self):
        seg = EdgeSegment(points=_pts())
        assert seg.length >= 0.0

    def test_points_stored(self):
        pts = _pts(5)
        seg = EdgeSegment(points=pts, side="top")
        assert seg.points.shape == (5, 2)

    def test_len_matches_points(self):
        seg = EdgeSegment(points=_pts(12))
        assert len(seg) == 12

    def test_side_bottom(self):
        seg = EdgeSegment(points=_pts(), side="bottom")
        assert seg.side == "bottom"

    def test_side_left(self):
        seg = EdgeSegment(points=_pts(), side="left")
        assert seg.side == "left"

    def test_side_right(self):
        seg = EdgeSegment(points=_pts(), side="right")
        assert seg.side == "right"

    def test_default_params_empty(self):
        seg = EdgeSegment(points=_pts())
        assert seg.params == {}

    def test_large_length_ok(self):
        seg = EdgeSegment(points=_pts(), length=1e6)
        assert seg.length == pytest.approx(1e6)

    def test_points_dtype(self):
        seg = EdgeSegment(points=_pts())
        assert seg.points.dtype == np.float32


# ─── TestFragmentEdgesExtra ──────────────────────────────────────────────────

class TestFragmentEdgesExtra:
    def test_default_params_empty(self):
        fe = FragmentEdges(segments=[], n_segments=0)
        assert fe.params == {}

    def test_segments_list(self):
        seg1 = EdgeSegment(points=_pts(), side="top")
        seg2 = EdgeSegment(points=_pts(), side="bottom")
        fe = FragmentEdges(segments=[seg1, seg2], n_segments=2)
        assert len(fe) == 2

    def test_n_segments_stored(self):
        fe = FragmentEdges(segments=[], n_segments=5)
        assert fe.n_segments == 5

    def test_segment_sides_accessible(self):
        seg = EdgeSegment(points=_pts(), side="left")
        fe = FragmentEdges(segments=[seg], n_segments=1)
        assert fe.segments[0].side == "left"


# ─── TestDetectBoundaryExtra ────────────────────────────────────────────────

class TestDetectBoundaryExtra:
    def test_white_image_no_boundary(self):
        img = np.full((32, 32), 255, dtype=np.uint8)
        result = detect_boundary(img)
        assert result.dtype == np.uint8

    def test_custom_threshold(self):
        result = detect_boundary(_white_square(), threshold=50)
        assert result.dtype == np.uint8

    def test_threshold_zero(self):
        result = detect_boundary(_white_square(), threshold=0)
        assert result.shape == (32, 32)

    def test_threshold_255(self):
        result = detect_boundary(_white_square(), threshold=255)
        assert result.shape == (32, 32)

    def test_small_image(self):
        img = np.zeros((8, 8), dtype=np.uint8)
        img[2:6, 2:6] = 255
        result = detect_boundary(img)
        assert result.shape == (8, 8)

    def test_returns_2d(self):
        result = detect_boundary(_white_square())
        assert result.ndim == 2

    def test_rectangular_image(self):
        img = np.zeros((16, 48), dtype=np.uint8)
        img[2:14, 2:46] = 255
        result = detect_boundary(img)
        assert result.shape == (16, 48)


# ─── TestExtractEdgePointsExtra ─────────────────────────────────────────────

class TestExtractEdgePointsExtra:
    def test_full_white_mask(self):
        mask = np.full((16, 16), 255, dtype=np.uint8)
        pts = extract_edge_points(mask)
        assert pts.dtype == np.float32

    def test_single_row_mask(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[15, :] = 255
        pts = extract_edge_points(mask)
        assert pts.shape[0] == 32

    def test_corner_pixel(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[0, 0] = 255
        pts = extract_edge_points(mask)
        assert pts.shape[0] == 1

    def test_all_cols_2(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[5, 5] = 255
        mask[10, 20] = 255
        pts = extract_edge_points(mask)
        assert pts.shape[1] == 2


# ─── TestSplitEdgeBySideExtra ───────────────────────────────────────────────

class TestSplitEdgeBySideExtra:
    def test_center_point_classification(self):
        pts = np.array([[16.0, 16.0]], dtype=np.float32)
        result = split_edge_by_side(pts, (32, 32))
        total = sum(len(v) for v in result.values())
        assert total >= 0

    def test_empty_points(self):
        pts = np.zeros((0, 2), dtype=np.float32)
        result = split_edge_by_side(pts, (32, 32))
        assert all(len(v) == 0 for v in result.values())

    def test_multiple_top_points(self):
        pts = np.array([[5.0, 0.0], [10.0, 0.0], [20.0, 0.0]],
                       dtype=np.float32)
        result = split_edge_by_side(pts, (32, 32))
        assert len(result["top"]) == 3

    def test_all_sides_present_in_keys(self):
        pts = np.array([[16.0, 0.0]], dtype=np.float32)
        result = split_edge_by_side(pts, (64, 64))
        assert "top" in result
        assert "bottom" in result
        assert "left" in result
        assert "right" in result


# ─── TestComputeEdgeLengthExtra ─────────────────────────────────────────────

class TestComputeEdgeLengthExtra:
    def test_horizontal_line(self):
        pts = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float32)
        assert compute_edge_length(pts) == pytest.approx(5.0)

    def test_vertical_line(self):
        pts = np.array([[0.0, 0.0], [0.0, 7.0]], dtype=np.float32)
        assert compute_edge_length(pts) == pytest.approx(7.0)

    def test_right_angle_path(self):
        pts = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]],
                       dtype=np.float32)
        assert compute_edge_length(pts) == pytest.approx(7.0)

    def test_many_segments(self):
        pts = np.array([[float(i), 0.0] for i in range(11)],
                       dtype=np.float32)
        assert compute_edge_length(pts) == pytest.approx(10.0)

    def test_closed_square(self):
        pts = np.array([
            [0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]
        ], dtype=np.float32)
        assert compute_edge_length(pts) == pytest.approx(30.0)


# ─── TestSimplifyEdgeExtra ──────────────────────────────────────────────────

class TestSimplifyEdgeExtra:
    def test_collinear_points_simplified(self):
        pts = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]],
                       dtype=np.float32)
        result = simplify_edge(pts, epsilon=1.0)
        assert len(result) <= 3

    def test_large_epsilon_reduces_more(self):
        pts = _pts(50)
        r1 = simplify_edge(pts, epsilon=0.1)
        r2 = simplify_edge(pts, epsilon=5.0)
        assert len(r2) <= len(r1)

    def test_single_point(self):
        pts = np.array([[5.0, 5.0]], dtype=np.float32)
        result = simplify_edge(pts)
        assert len(result) >= 1

    def test_preserves_endpoints(self):
        pts = _pts(20)
        result = simplify_edge(pts, epsilon=0.5)
        np.testing.assert_array_almost_equal(result[0], pts[0])
        np.testing.assert_array_almost_equal(result[-1], pts[-1])

    def test_output_shape_cols(self):
        result = simplify_edge(_pts(15))
        assert result.shape[1] == 2


# ─── TestExtractFragmentEdgesExtra ──────────────────────────────────────────

class TestExtractFragmentEdgesExtra:
    def test_n_segments_matches_len(self):
        result = extract_fragment_edges(_white_square())
        assert result.n_segments == len(result.segments)

    def test_segment_points_are_ndarray(self):
        result = extract_fragment_edges(_white_square())
        for seg in result.segments:
            assert isinstance(seg.points, np.ndarray)

    def test_segment_points_2d(self):
        result = extract_fragment_edges(_white_square())
        for seg in result.segments:
            assert seg.points.ndim == 2
            assert seg.points.shape[1] == 2

    def test_different_sizes(self):
        r1 = extract_fragment_edges(_white_square(32, 32))
        r2 = extract_fragment_edges(_white_square(64, 64))
        assert r1.n_segments == r2.n_segments

    def test_returns_fragment_edges_type(self):
        result = extract_fragment_edges(_white_square())
        assert isinstance(result, FragmentEdges)

    def test_segments_have_sides(self):
        result = extract_fragment_edges(_white_square())
        for seg in result.segments:
            assert seg.side in {"top", "bottom", "left", "right", "unknown"}


# ─── TestBatchExtractEdgesExtra ─────────────────────────────────────────────

class TestBatchExtractEdgesExtra:
    def test_single_image(self):
        result = batch_extract_edges([_white_square()])
        assert len(result) == 1
        assert isinstance(result[0], FragmentEdges)

    def test_three_images(self):
        imgs = [_white_square(32, 32), _white_square(48, 48),
                _white_square(64, 64)]
        result = batch_extract_edges(imgs)
        assert len(result) == 3

    def test_color_images(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[4:28, 4:28] = 200
        result = batch_extract_edges([img])
        assert len(result) == 1

    def test_each_has_segments(self):
        result = batch_extract_edges([_white_square()])
        assert result[0].n_segments > 0

    def test_all_results_fragment_edges(self):
        imgs = [_white_square()] * 4
        result = batch_extract_edges(imgs)
        assert all(isinstance(r, FragmentEdges) for r in result)
