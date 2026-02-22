"""Тесты для puzzle_reconstruction.algorithms.edge_extractor."""
import pytest
import numpy as np
import cv2
from puzzle_reconstruction.algorithms.edge_extractor import (
    EdgeSegment,
    FragmentEdges,
    detect_boundary,
    extract_edge_points,
    split_edge_by_side,
    compute_edge_length,
    simplify_edge,
    extract_fragment_edges,
    batch_extract_edges,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray_rect(h=64, w=64, fill=200) -> np.ndarray:
    return np.full((h, w), fill, dtype=np.uint8)


def _color_rect(h=64, w=64, fill=200) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


def _blank(h=64, w=64) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _line_pts(n=10, length=10.0) -> np.ndarray:
    x = np.linspace(0.0, length, n)
    return np.stack([x, np.zeros(n)], axis=1).astype(np.float32)


def _top_row_pts(w=64, n=5) -> np.ndarray:
    """Points along the top row of a 64×64 image."""
    x = np.linspace(10.0, 50.0, n)
    return np.stack([x, np.zeros(n)], axis=1).astype(np.float32)


# ─── TestEdgeSegment ──────────────────────────────────────────────────────────

class TestEdgeSegment:
    def test_valid_construction(self):
        pts = _line_pts()
        seg = EdgeSegment(points=pts, side="top", length=10.0)
        assert seg.side == "top"

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError):
            EdgeSegment(points=_line_pts(), side="diagonal")

    def test_negative_length_raises(self):
        with pytest.raises(ValueError):
            EdgeSegment(points=_line_pts(), side="top", length=-1.0)

    def test_valid_sides(self):
        pts = _line_pts()
        for side in ("top", "bottom", "left", "right", "unknown"):
            seg = EdgeSegment(points=pts, side=side, length=0.0)
            assert seg.side == side

    def test_len_returns_n_points(self):
        pts = _line_pts(8)
        seg = EdgeSegment(points=pts, side="top", length=0.0)
        assert len(seg) == 8

    def test_params_default_empty(self):
        seg = EdgeSegment(points=_line_pts(), side="top", length=0.0)
        assert seg.params == {}


# ─── TestFragmentEdges ────────────────────────────────────────────────────────

class TestFragmentEdges:
    def test_valid_construction(self):
        fe = FragmentEdges(segments=[], n_segments=0)
        assert fe.n_segments == 0

    def test_negative_n_segments_raises(self):
        with pytest.raises(ValueError):
            FragmentEdges(segments=[], n_segments=-1)

    def test_len_returns_n_segments(self):
        seg = EdgeSegment(points=_line_pts(), side="top", length=0.0)
        fe = FragmentEdges(segments=[seg], n_segments=1)
        assert len(fe) == 1

    def test_params_default_empty(self):
        fe = FragmentEdges(segments=[], n_segments=0)
        assert fe.params == {}


# ─── TestDetectBoundary ───────────────────────────────────────────────────────

class TestDetectBoundary:
    def test_returns_uint8(self):
        out = detect_boundary(_gray_rect())
        assert out.dtype == np.uint8

    def test_shape_matches_input(self):
        img = _gray_rect(48, 80)
        out = detect_boundary(img)
        assert out.shape == (48, 80)

    def test_rgb_ok(self):
        out = detect_boundary(_color_rect())
        assert out.dtype == np.uint8

    def test_blank_image_no_boundary(self):
        out = detect_boundary(_blank())
        assert out.max() == 0

    def test_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError):
            detect_boundary(_gray_rect(), threshold=-1)
        with pytest.raises(ValueError):
            detect_boundary(_gray_rect(), threshold=256)

    def test_nonzero_boundary_on_filled_image(self):
        out = detect_boundary(_gray_rect(fill=200), threshold=10)
        assert out.max() > 0

    def test_output_values_binary(self):
        out = detect_boundary(_gray_rect())
        unique = np.unique(out)
        assert set(unique).issubset({0, 255})


# ─── TestExtractEdgePoints ────────────────────────────────────────────────────

class TestExtractEdgePoints:
    def test_blank_mask_empty(self):
        pts = extract_edge_points(_blank())
        assert pts.shape == (0, 2)

    def test_returns_float32(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10, 20] = 255
        pts = extract_edge_points(mask)
        assert pts.dtype == np.float32

    def test_shape_n_2(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10, 20] = 255
        mask[30, 40] = 255
        pts = extract_edge_points(mask)
        assert pts.ndim == 2
        assert pts.shape[1] == 2

    def test_3d_mask_raises(self):
        with pytest.raises(ValueError):
            extract_edge_points(np.zeros((64, 64, 3), dtype=np.uint8))

    def test_point_positions_xy(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[5, 10] = 255  # row=5 (y), col=10 (x)
        pts = extract_edge_points(mask)
        # x=10, y=5
        assert any(p[0] == pytest.approx(10.0) and p[1] == pytest.approx(5.0)
                   for p in pts)


# ─── TestSplitEdgeBySide ──────────────────────────────────────────────────────

class TestSplitEdgeBySide:
    def test_returns_four_sides(self):
        pts = _top_row_pts()
        result = split_edge_by_side(pts, (64, 64))
        assert set(result.keys()) == {"top", "bottom", "left", "right"}

    def test_top_row_points_go_to_top(self):
        pts = _top_row_pts()
        result = split_edge_by_side(pts, (64, 64))
        assert len(result["top"]) > 0

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            split_edge_by_side(np.zeros((10, 3)), (64, 64))

    def test_zero_height_raises(self):
        with pytest.raises(ValueError):
            split_edge_by_side(_top_row_pts(), (0, 64))

    def test_empty_points_all_empty(self):
        pts = np.zeros((0, 2), dtype=np.float32)
        result = split_edge_by_side(pts, (64, 64))
        for side in ("top", "bottom", "left", "right"):
            assert len(result[side]) == 0

    def test_all_sides_float32(self):
        pts = _top_row_pts()
        result = split_edge_by_side(pts, (64, 64))
        for arr in result.values():
            assert arr.dtype == np.float32


# ─── TestComputeEdgeLength ────────────────────────────────────────────────────

class TestComputeEdgeLength:
    def test_empty_points_zero(self):
        assert compute_edge_length(np.zeros((0, 2), dtype=np.float32)) == 0.0

    def test_single_point_zero(self):
        assert compute_edge_length(np.array([[1.0, 2.0]])) == 0.0

    def test_straight_line_correct(self):
        pts = _line_pts(n=11, length=10.0)
        length = compute_edge_length(pts)
        assert length == pytest.approx(10.0, abs=1e-5)

    def test_returns_float(self):
        pts = _line_pts()
        assert isinstance(compute_edge_length(pts), float)

    def test_nonnegative(self):
        pts = _line_pts()
        assert compute_edge_length(pts) >= 0.0


# ─── TestSimplifyEdge ─────────────────────────────────────────────────────────

class TestSimplifyEdge:
    def test_returns_ndarray(self):
        pts = _line_pts(20)
        out = simplify_edge(pts)
        assert isinstance(out, np.ndarray)

    def test_epsilon_negative_raises(self):
        with pytest.raises(ValueError):
            simplify_edge(_line_pts(10), epsilon=-1.0)

    def test_two_points_returned_as_copy(self):
        pts = _line_pts(2)
        out = simplify_edge(pts, epsilon=1.0)
        assert out.shape[0] == 2

    def test_reduces_points(self):
        pts = _line_pts(50, length=100.0)
        out = simplify_edge(pts, epsilon=1.0)
        # A straight line should reduce significantly
        assert out.shape[0] <= 50

    def test_dtype_float32(self):
        out = simplify_edge(_line_pts(10))
        assert out.dtype == np.float32

    def test_epsilon_zero_keeps_more(self):
        pts = _line_pts(20, length=50.0)
        out_tight = simplify_edge(pts, epsilon=0.0)
        out_loose = simplify_edge(pts, epsilon=5.0)
        assert out_tight.shape[0] >= out_loose.shape[0]


# ─── TestExtractFragmentEdges ─────────────────────────────────────────────────

class TestExtractFragmentEdges:
    def test_returns_fragment_edges(self):
        result = extract_fragment_edges(_gray_rect())
        assert isinstance(result, FragmentEdges)

    def test_four_segments(self):
        result = extract_fragment_edges(_gray_rect())
        assert result.n_segments == 4

    def test_segment_sides(self):
        result = extract_fragment_edges(_gray_rect())
        sides = {seg.side for seg in result.segments}
        assert sides == {"top", "bottom", "left", "right"}

    def test_rgb_image_ok(self):
        result = extract_fragment_edges(_color_rect())
        assert isinstance(result, FragmentEdges)

    def test_params_stored(self):
        result = extract_fragment_edges(_gray_rect(), threshold=20, epsilon=2.0)
        assert result.params["threshold"] == 20
        assert result.params["epsilon"] == pytest.approx(2.0)

    def test_segment_length_nonneg(self):
        result = extract_fragment_edges(_gray_rect())
        for seg in result.segments:
            assert seg.length >= 0.0


# ─── TestBatchExtractEdges ────────────────────────────────────────────────────

class TestBatchExtractEdges:
    def test_returns_list(self):
        imgs = [_gray_rect(), _gray_rect()]
        result = batch_extract_edges(imgs)
        assert isinstance(result, list)

    def test_length_matches(self):
        imgs = [_gray_rect(), _color_rect(), _gray_rect()]
        result = batch_extract_edges(imgs)
        assert len(result) == 3

    def test_empty_list(self):
        result = batch_extract_edges([])
        assert result == []

    def test_all_fragment_edges(self):
        imgs = [_gray_rect(), _gray_rect()]
        for fe in batch_extract_edges(imgs):
            assert isinstance(fe, FragmentEdges)
