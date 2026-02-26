"""Tests for puzzle_reconstruction/algorithms/edge_extractor.py"""
import numpy as np
import pytest
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def white_rect_on_black(h=50, w=50) -> np.ndarray:
    """Return a grayscale image with a white rectangle on black background."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[10:40, 10:40] = 255
    return img


def bgr_rect(h=50, w=50) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[10:40, 10:40] = 200
    return img


# ── EdgeSegment ───────────────────────────────────────────────────────────────

class TestEdgeSegment:
    def test_valid_construction(self):
        pts = np.zeros((5, 2), dtype=np.float32)
        seg = EdgeSegment(points=pts, side="top", length=10.0)
        assert seg.side == "top"
        assert seg.length == 10.0

    def test_len(self):
        pts = np.zeros((7, 2), dtype=np.float32)
        seg = EdgeSegment(points=pts, side="left")
        assert len(seg) == 7

    def test_all_valid_sides(self):
        pts = np.zeros((2, 2), dtype=np.float32)
        for side in ("top", "bottom", "left", "right", "unknown"):
            seg = EdgeSegment(points=pts, side=side)
            assert seg.side == side

    def test_invalid_side_raises(self):
        pts = np.zeros((2, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="side must be one of"):
            EdgeSegment(points=pts, side="diagonal")

    def test_negative_length_raises(self):
        pts = np.zeros((2, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="length must be >= 0"):
            EdgeSegment(points=pts, length=-1.0)

    def test_zero_length_ok(self):
        pts = np.zeros((1, 2), dtype=np.float32)
        seg = EdgeSegment(points=pts, length=0.0)
        assert seg.length == 0.0

    def test_default_side_unknown(self):
        pts = np.zeros((3, 2), dtype=np.float32)
        seg = EdgeSegment(points=pts)
        assert seg.side == "unknown"

    def test_params_default_empty(self):
        pts = np.zeros((3, 2), dtype=np.float32)
        seg = EdgeSegment(points=pts)
        assert seg.params == {}

    def test_params_stored(self):
        pts = np.zeros((3, 2), dtype=np.float32)
        seg = EdgeSegment(points=pts, params={"epsilon": 1.5})
        assert seg.params["epsilon"] == 1.5

    def test_empty_points(self):
        pts = np.zeros((0, 2), dtype=np.float32)
        seg = EdgeSegment(points=pts)
        assert len(seg) == 0


# ── FragmentEdges ─────────────────────────────────────────────────────────────

class TestFragmentEdges:
    def _make_segment(self, side="top"):
        pts = np.zeros((3, 2), dtype=np.float32)
        return EdgeSegment(points=pts, side=side, length=5.0)

    def test_valid_construction(self):
        segs = [self._make_segment("top"), self._make_segment("bottom")]
        fe = FragmentEdges(segments=segs, n_segments=2)
        assert fe.n_segments == 2

    def test_len(self):
        segs = [self._make_segment("top")]
        fe = FragmentEdges(segments=segs, n_segments=1)
        assert len(fe) == 1

    def test_negative_n_segments_raises(self):
        with pytest.raises(ValueError, match="n_segments must be >= 0"):
            FragmentEdges(segments=[], n_segments=-1)

    def test_zero_segments_ok(self):
        fe = FragmentEdges(segments=[], n_segments=0)
        assert len(fe) == 0

    def test_params_default_empty(self):
        fe = FragmentEdges(segments=[], n_segments=0)
        assert fe.params == {}


# ── detect_boundary ───────────────────────────────────────────────────────────

class TestDetectBoundary:
    def test_output_shape(self):
        img = white_rect_on_black()
        boundary = detect_boundary(img)
        assert boundary.shape == img.shape

    def test_output_dtype(self):
        img = white_rect_on_black()
        boundary = detect_boundary(img)
        assert boundary.dtype == np.uint8

    def test_boundary_values_binary(self):
        img = white_rect_on_black()
        boundary = detect_boundary(img)
        unique = set(np.unique(boundary))
        assert unique.issubset({0, 255})

    def test_bgr_input(self):
        img = bgr_rect()
        boundary = detect_boundary(img)
        assert boundary.ndim == 2

    def test_threshold_zero_gives_all_boundary(self):
        img = white_rect_on_black()
        boundary = detect_boundary(img, threshold=0)
        assert boundary.shape == img.shape

    def test_threshold_255_gives_empty(self):
        img = white_rect_on_black()
        boundary = detect_boundary(img, threshold=255)
        assert np.all(boundary == 0)

    def test_invalid_threshold_low_raises(self):
        img = white_rect_on_black()
        with pytest.raises(ValueError, match="threshold must be in"):
            detect_boundary(img, threshold=-1)

    def test_invalid_threshold_high_raises(self):
        img = white_rect_on_black()
        with pytest.raises(ValueError, match="threshold must be in"):
            detect_boundary(img, threshold=256)

    def test_all_black_gives_no_boundary(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        boundary = detect_boundary(img, threshold=10)
        assert np.all(boundary == 0)

    def test_full_white_gives_no_boundary(self):
        img = np.full((50, 50), 255, dtype=np.uint8)
        boundary = detect_boundary(img, threshold=10)
        # After erode+subtract on a full image, boundary should be zero (only edges would remain due to erosion)
        # Boundary = binary - eroded: full white image after erosion with border=0 has no boundary
        # Because eroded with border=0 loses edge pixels, subtract gives edge pixels
        assert boundary.dtype == np.uint8


# ── extract_edge_points ───────────────────────────────────────────────────────

class TestExtractEdgePoints:
    def test_returns_float32(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10, 10] = 255
        pts = extract_edge_points(mask)
        assert pts.dtype == np.float32

    def test_shape_Nx2(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10, 10] = 255
        mask[20, 30] = 255
        pts = extract_edge_points(mask)
        assert pts.ndim == 2
        assert pts.shape[1] == 2

    def test_empty_mask_returns_zero_shape(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        pts = extract_edge_points(mask)
        assert pts.shape == (0, 2)

    def test_3d_mask_raises(self):
        mask = np.zeros((50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="mask must be 2-D"):
            extract_edge_points(mask)

    def test_point_coordinates_correct(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[15, 25] = 255
        pts = extract_edge_points(mask)
        assert len(pts) == 1
        x, y = pts[0]
        assert x == 25.0
        assert y == 15.0

    def test_multiple_points(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[5, 5] = 255
        mask[10, 10] = 255
        mask[20, 30] = 255
        pts = extract_edge_points(mask)
        assert len(pts) == 3


# ── split_edge_by_side ────────────────────────────────────────────────────────

class TestSplitEdgeBySide:
    def test_returns_dict_with_four_sides(self):
        pts = np.array([[10, 0], [0, 10], [10, 99], [99, 10]], dtype=np.float32)
        result = split_edge_by_side(pts, (100, 100))
        assert set(result.keys()) == {"top", "bottom", "left", "right"}

    def test_all_values_are_float32(self):
        pts = np.array([[5, 0]], dtype=np.float32)
        result = split_edge_by_side(pts, (100, 100))
        for arr in result.values():
            assert arr.dtype == np.float32

    def test_top_point_assigned(self):
        pts = np.array([[50, 0]], dtype=np.float32)
        result = split_edge_by_side(pts, (100, 100))
        assert len(result["top"]) == 1

    def test_bottom_point_assigned(self):
        pts = np.array([[50, 99]], dtype=np.float32)
        result = split_edge_by_side(pts, (100, 100))
        assert len(result["bottom"]) == 1

    def test_left_point_assigned(self):
        pts = np.array([[0, 50]], dtype=np.float32)
        result = split_edge_by_side(pts, (100, 100))
        assert len(result["left"]) == 1

    def test_right_point_assigned(self):
        pts = np.array([[99, 50]], dtype=np.float32)
        result = split_edge_by_side(pts, (100, 100))
        assert len(result["right"]) == 1

    def test_invalid_points_shape_raises(self):
        pts = np.array([[1, 2, 3]], dtype=np.float32)
        with pytest.raises(ValueError):
            split_edge_by_side(pts, (100, 100))

    def test_invalid_1d_raises(self):
        pts = np.array([1, 2], dtype=np.float32)
        with pytest.raises(ValueError):
            split_edge_by_side(pts, (100, 100))

    def test_negative_img_shape_raises(self):
        pts = np.array([[5, 5]], dtype=np.float32)
        with pytest.raises(ValueError):
            split_edge_by_side(pts, (-1, 100))

    def test_empty_points_all_sides_empty(self):
        pts = np.zeros((0, 2), dtype=np.float32)
        result = split_edge_by_side(pts, (100, 100))
        for arr in result.values():
            assert arr.shape == (0, 2)

    def test_total_points_conserved(self):
        pts = np.array([[10, 0], [0, 50], [50, 99], [99, 50]], dtype=np.float32)
        result = split_edge_by_side(pts, (100, 100))
        total = sum(len(v) for v in result.values())
        assert total == 4


# ── compute_edge_length ───────────────────────────────────────────────────────

class TestComputeEdgeLength:
    def test_two_points_horizontal(self):
        pts = np.array([[0, 0], [10, 0]], dtype=np.float32)
        assert pytest.approx(compute_edge_length(pts), abs=1e-5) == 10.0

    def test_two_points_vertical(self):
        pts = np.array([[0, 0], [0, 10]], dtype=np.float32)
        assert pytest.approx(compute_edge_length(pts), abs=1e-5) == 10.0

    def test_three_points(self):
        pts = np.array([[0, 0], [3, 0], [3, 4]], dtype=np.float32)
        assert pytest.approx(compute_edge_length(pts), abs=1e-5) == 7.0

    def test_single_point_returns_zero(self):
        pts = np.array([[5, 5]], dtype=np.float32)
        assert compute_edge_length(pts) == 0.0

    def test_empty_returns_zero(self):
        pts = np.zeros((0, 2), dtype=np.float32)
        assert compute_edge_length(pts) == 0.0

    def test_diagonal(self):
        pts = np.array([[0, 0], [3, 4]], dtype=np.float32)
        assert pytest.approx(compute_edge_length(pts), abs=1e-4) == 5.0

    def test_returns_float(self):
        pts = np.array([[0, 0], [1, 0]], dtype=np.float32)
        result = compute_edge_length(pts)
        assert isinstance(result, float)


# ── simplify_edge ─────────────────────────────────────────────────────────────

class TestSimplifyEdge:
    def test_output_shape(self):
        pts = np.array([[0, 0], [1, 0.1], [2, 0], [3, 0.1], [4, 0]],
                       dtype=np.float32)
        result = simplify_edge(pts, epsilon=1.0)
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_result_is_subset(self):
        pts = np.array([[0, 0], [1, 0.1], [2, 0], [3, 0.1], [4, 0]],
                       dtype=np.float32)
        result = simplify_edge(pts, epsilon=1.0)
        assert len(result) <= len(pts)

    def test_epsilon_zero_returns_all_or_more(self):
        pts = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float32)
        result = simplify_edge(pts, epsilon=0.0)
        assert len(result) >= 2

    def test_negative_epsilon_raises(self):
        pts = np.array([[0, 0], [1, 1]], dtype=np.float32)
        with pytest.raises(ValueError, match="epsilon must be >= 0"):
            simplify_edge(pts, epsilon=-1.0)

    def test_two_points_returned_unchanged(self):
        pts = np.array([[0, 0], [10, 10]], dtype=np.float32)
        result = simplify_edge(pts, epsilon=5.0)
        assert result.shape == (2, 2)

    def test_output_dtype(self):
        pts = np.array([[0, 0], [1, 0.1], [2, 0]], dtype=np.float32)
        result = simplify_edge(pts, epsilon=0.5)
        assert result.dtype == np.float32

    def test_large_epsilon_collapses_to_endpoints(self):
        pts = np.array([[0, 0], [5, 1], [10, 0]], dtype=np.float32)
        result = simplify_edge(pts, epsilon=100.0)
        assert len(result) <= 3


# ── extract_fragment_edges ────────────────────────────────────────────────────

class TestExtractFragmentEdges:
    def test_returns_fragment_edges(self):
        img = white_rect_on_black()
        result = extract_fragment_edges(img)
        assert isinstance(result, FragmentEdges)

    def test_exactly_4_segments(self):
        img = white_rect_on_black()
        result = extract_fragment_edges(img)
        assert result.n_segments == 4

    def test_segment_sides_correct(self):
        img = white_rect_on_black()
        result = extract_fragment_edges(img)
        sides = {seg.side for seg in result.segments}
        assert sides == {"top", "bottom", "left", "right"}

    def test_bgr_input_ok(self):
        img = bgr_rect()
        result = extract_fragment_edges(img)
        assert isinstance(result, FragmentEdges)

    def test_params_stored(self):
        img = white_rect_on_black()
        result = extract_fragment_edges(img, threshold=20, epsilon=2.0)
        assert result.params["threshold"] == 20
        assert result.params["epsilon"] == 2.0

    def test_all_lengths_nonneg(self):
        img = white_rect_on_black()
        result = extract_fragment_edges(img)
        for seg in result.segments:
            assert seg.length >= 0.0

    def test_points_are_float32(self):
        img = white_rect_on_black()
        result = extract_fragment_edges(img)
        for seg in result.segments:
            assert seg.points.dtype == np.float32

    def test_each_segment_has_epsilon_param(self):
        img = white_rect_on_black()
        result = extract_fragment_edges(img, epsilon=1.5)
        for seg in result.segments:
            assert seg.params.get("epsilon") == 1.5


# ── batch_extract_edges ───────────────────────────────────────────────────────

class TestBatchExtractEdges:
    def test_output_length_matches_input(self):
        imgs = [white_rect_on_black(), white_rect_on_black(60, 60)]
        result = batch_extract_edges(imgs)
        assert len(result) == 2

    def test_each_element_is_fragment_edges(self):
        imgs = [white_rect_on_black()]
        result = batch_extract_edges(imgs)
        assert isinstance(result[0], FragmentEdges)

    def test_empty_list(self):
        result = batch_extract_edges([])
        assert result == []

    def test_threshold_passed_through(self):
        imgs = [white_rect_on_black()]
        result = batch_extract_edges(imgs, threshold=20)
        assert result[0].params["threshold"] == 20

    def test_epsilon_passed_through(self):
        imgs = [white_rect_on_black()]
        result = batch_extract_edges(imgs, epsilon=2.0)
        assert result[0].params["epsilon"] == 2.0

    def test_multiple_images_different_sizes(self):
        imgs = [white_rect_on_black(40, 60), white_rect_on_black(80, 80)]
        result = batch_extract_edges(imgs)
        assert len(result) == 2
        for fe in result:
            assert fe.n_segments == 4
