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

def _gray(h: int = 64, w: int = 64, fill: int = 200) -> np.ndarray:
    """Solid gray image with a bright rectangle."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[4:h - 4, 4:w - 4] = fill
    return img


def _bgr(h: int = 64, w: int = 64) -> np.ndarray:
    """Solid BGR image with a bright rectangle."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[4:h - 4, 4:w - 4] = (200, 210, 220)
    return img


def _pts(n: int = 10) -> np.ndarray:
    """Horizontal line of n points."""
    xs = np.arange(n, dtype=np.float32)
    ys = np.zeros(n, dtype=np.float32)
    return np.stack([xs, ys], axis=1)


# ─── EdgeSegment ─────────────────────────────────────────────────────────────

class TestEdgeSegment:
    def test_fields_stored(self):
        pts = _pts(5)
        seg = EdgeSegment(points=pts, side="top", length=4.0)
        assert seg.side == "top"
        assert seg.length == pytest.approx(4.0)
        assert seg.points.shape == (5, 2)

    def test_all_valid_sides(self):
        pts = _pts(3)
        for side in ("top", "bottom", "left", "right", "unknown"):
            seg = EdgeSegment(points=pts, side=side, length=0.0)
            assert seg.side == side

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError):
            EdgeSegment(points=_pts(3), side="diagonal", length=0.0)

    def test_negative_length_raises(self):
        with pytest.raises(ValueError):
            EdgeSegment(points=_pts(3), side="top", length=-1.0)

    def test_len_returns_point_count(self):
        seg = EdgeSegment(points=_pts(7), side="left", length=6.0)
        assert len(seg) == 7

    def test_default_side_is_unknown(self):
        seg = EdgeSegment(points=_pts(2), length=1.0)
        assert seg.side == "unknown"

    def test_params_default_empty(self):
        seg = EdgeSegment(points=_pts(2), length=0.0)
        assert seg.params == {}

    def test_params_stored(self):
        seg = EdgeSegment(points=_pts(2), length=0.0, params={"epsilon": 1.5})
        assert seg.params["epsilon"] == pytest.approx(1.5)


# ─── FragmentEdges ────────────────────────────────────────────────────────────

class TestFragmentEdges:
    def _make_fe(self, n: int = 4) -> FragmentEdges:
        segs = [EdgeSegment(points=_pts(5), side="top", length=4.0)
                for _ in range(n)]
        return FragmentEdges(segments=segs, n_segments=n)

    def test_fields_stored(self):
        fe = self._make_fe(4)
        assert fe.n_segments == 4
        assert len(fe.segments) == 4

    def test_len_returns_n_segments(self):
        fe = self._make_fe(3)
        assert len(fe) == 3

    def test_negative_n_segments_raises(self):
        with pytest.raises(ValueError):
            FragmentEdges(segments=[], n_segments=-1)

    def test_zero_segments_ok(self):
        fe = FragmentEdges(segments=[], n_segments=0)
        assert len(fe) == 0

    def test_params_default_empty(self):
        fe = self._make_fe(2)
        assert fe.params == {}

    def test_params_stored(self):
        segs = [EdgeSegment(points=_pts(2), length=0.0)]
        fe = FragmentEdges(segments=segs, n_segments=1, params={"threshold": 10})
        assert fe.params["threshold"] == 10


# ─── detect_boundary ─────────────────────────────────────────────────────────

class TestDetectBoundary:
    def test_returns_uint8(self):
        result = detect_boundary(_gray())
        assert result.dtype == np.uint8

    def test_output_shape_matches_input(self):
        img = _gray(48, 64)
        result = detect_boundary(img)
        assert result.shape == (48, 64)

    def test_bgr_input_accepted(self):
        result = detect_boundary(_bgr())
        assert result.dtype == np.uint8
        assert result.ndim == 2

    def test_boundary_pixels_nonzero(self):
        result = detect_boundary(_gray())
        assert np.any(result > 0)

    def test_threshold_zero_accepted(self):
        result = detect_boundary(_gray(), threshold=0)
        assert result.dtype == np.uint8

    def test_threshold_255_accepted(self):
        result = detect_boundary(_gray(), threshold=255)
        assert result.dtype == np.uint8

    def test_invalid_threshold_negative_raises(self):
        with pytest.raises(ValueError):
            detect_boundary(_gray(), threshold=-1)

    def test_invalid_threshold_above_255_raises(self):
        with pytest.raises(ValueError):
            detect_boundary(_gray(), threshold=256)

    def test_black_image_no_boundary(self):
        black = np.zeros((32, 32), dtype=np.uint8)
        result = detect_boundary(black, threshold=0)
        assert np.all(result == 0)

    def test_output_values_binary(self):
        result = detect_boundary(_gray())
        unique = np.unique(result)
        assert all(v in (0, 255) for v in unique)


# ─── extract_edge_points ──────────────────────────────────────────────────────

class TestExtractEdgePoints:
    def test_returns_float32(self):
        mask = detect_boundary(_gray())
        pts = extract_edge_points(mask)
        assert pts.dtype == np.float32

    def test_shape_is_n_by_2(self):
        mask = detect_boundary(_gray())
        pts = extract_edge_points(mask)
        assert pts.ndim == 2
        assert pts.shape[1] == 2

    def test_black_mask_returns_empty(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        pts = extract_edge_points(mask)
        assert pts.shape == (0, 2)

    def test_nonzero_pixels_yield_points(self):
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[8, 8] = 255
        mask[4, 4] = 255
        pts = extract_edge_points(mask)
        assert len(pts) == 2

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            extract_edge_points(np.zeros((4, 4, 1), dtype=np.uint8))

    def test_columns_are_xy(self):
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[3, 5] = 255   # y=3, x=5
        pts = extract_edge_points(mask)
        assert len(pts) == 1
        assert pts[0, 0] == pytest.approx(5.0)   # x
        assert pts[0, 1] == pytest.approx(3.0)   # y


# ─── split_edge_by_side ───────────────────────────────────────────────────────

class TestSplitEdgeBySide:
    def test_returns_all_four_sides(self):
        pts = _pts(8)
        sides = split_edge_by_side(pts, (64, 64))
        assert set(sides.keys()) == {"top", "bottom", "left", "right"}

    def test_top_edge_assigned_to_top(self):
        # Points along y=0 (top edge)
        pts = np.array([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]],
                       dtype=np.float32)
        sides = split_edge_by_side(pts, (64, 64))
        assert len(sides["top"]) == 3

    def test_bottom_edge_assigned_to_bottom(self):
        pts = np.array([[10.0, 63.0], [20.0, 63.0]], dtype=np.float32)
        sides = split_edge_by_side(pts, (64, 64))
        assert len(sides["bottom"]) == 2

    def test_left_edge_assigned_to_left(self):
        pts = np.array([[0.0, 20.0], [0.0, 30.0]], dtype=np.float32)
        sides = split_edge_by_side(pts, (64, 64))
        assert len(sides["left"]) == 2

    def test_right_edge_assigned_to_right(self):
        pts = np.array([[63.0, 20.0], [63.0, 40.0]], dtype=np.float32)
        sides = split_edge_by_side(pts, (64, 64))
        assert len(sides["right"]) == 2

    def test_all_points_distributed(self):
        mask = detect_boundary(_gray())
        pts = extract_edge_points(mask)
        h, w = 64, 64
        sides = split_edge_by_side(pts, (h, w))
        total = sum(len(v) for v in sides.values())
        assert total == len(pts)

    def test_each_side_is_float32(self):
        pts = _pts(4)
        sides = split_edge_by_side(pts, (32, 32))
        for arr in sides.values():
            assert arr.dtype == np.float32

    def test_each_side_is_n_by_2(self):
        pts = _pts(8)
        sides = split_edge_by_side(pts, (64, 64))
        for arr in sides.values():
            assert arr.ndim == 2
            assert arr.shape[1] == 2

    def test_wrong_points_shape_raises(self):
        bad = np.zeros((5, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            split_edge_by_side(bad, (64, 64))

    def test_zero_height_raises(self):
        pts = _pts(4)
        with pytest.raises(ValueError):
            split_edge_by_side(pts, (0, 64))

    def test_zero_width_raises(self):
        pts = _pts(4)
        with pytest.raises(ValueError):
            split_edge_by_side(pts, (64, 0))


# ─── compute_edge_length ──────────────────────────────────────────────────────

class TestComputeEdgeLength:
    def test_unit_steps_give_correct_length(self):
        pts = _pts(6)     # (0,0),(1,0),...,(5,0) → length=5
        assert compute_edge_length(pts) == pytest.approx(5.0)

    def test_single_point_returns_zero(self):
        assert compute_edge_length(_pts(1)) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        pts = np.zeros((0, 2), dtype=np.float32)
        assert compute_edge_length(pts) == pytest.approx(0.0)

    def test_diagonal_steps(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        expected = float(np.sqrt(2))
        assert compute_edge_length(pts) == pytest.approx(expected, rel=1e-5)

    def test_returns_float(self):
        result = compute_edge_length(_pts(5))
        assert isinstance(result, float)

    def test_longer_path_larger_length(self):
        short = _pts(5)
        long_ = _pts(20)
        assert compute_edge_length(long_) > compute_edge_length(short)

    def test_two_points_exact(self):
        pts = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        assert compute_edge_length(pts) == pytest.approx(5.0)


# ─── simplify_edge ────────────────────────────────────────────────────────────

class TestSimplifyEdge:
    def test_returns_float32(self):
        pts = _pts(20)
        result = simplify_edge(pts, epsilon=1.0)
        assert result.dtype == np.float32

    def test_shape_is_m_by_2(self):
        pts = _pts(20)
        result = simplify_edge(pts)
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_collinear_points_reduced(self):
        pts = _pts(50)     # all on y=0
        result = simplify_edge(pts, epsilon=0.5)
        assert len(result) <= len(pts)

    def test_epsilon_zero_preserves_all(self):
        pts = _pts(10)
        result = simplify_edge(pts, epsilon=0.0)
        assert len(result) == len(pts)

    def test_two_points_unchanged(self):
        pts = _pts(2)
        result = simplify_edge(pts, epsilon=5.0)
        assert len(result) == 2

    def test_negative_epsilon_raises(self):
        with pytest.raises(ValueError):
            simplify_edge(_pts(5), epsilon=-1.0)

    def test_large_epsilon_aggressive_simplification(self):
        pts = _pts(100)
        strict = simplify_edge(pts, epsilon=0.1)
        loose = simplify_edge(pts, epsilon=10.0)
        assert len(loose) <= len(strict)


# ─── extract_fragment_edges ───────────────────────────────────────────────────

class TestExtractFragmentEdges:
    def test_returns_fragment_edges(self):
        result = extract_fragment_edges(_gray())
        assert isinstance(result, FragmentEdges)

    def test_has_four_segments(self):
        result = extract_fragment_edges(_gray())
        assert result.n_segments == 4

    def test_segment_sides_are_correct(self):
        result = extract_fragment_edges(_gray())
        sides = {seg.side for seg in result.segments}
        assert sides == {"top", "bottom", "left", "right"}

    def test_each_segment_is_edge_segment(self):
        result = extract_fragment_edges(_gray())
        for seg in result.segments:
            assert isinstance(seg, EdgeSegment)

    def test_each_segment_points_is_n_by_2(self):
        result = extract_fragment_edges(_gray())
        for seg in result.segments:
            assert seg.points.ndim == 2
            assert seg.points.shape[1] == 2

    def test_length_nonnegative(self):
        result = extract_fragment_edges(_gray())
        for seg in result.segments:
            assert seg.length >= 0.0

    def test_params_stored(self):
        result = extract_fragment_edges(_gray(), threshold=15, epsilon=2.0)
        assert result.params["threshold"] == 15
        assert result.params["epsilon"] == pytest.approx(2.0)

    def test_bgr_image_accepted(self):
        result = extract_fragment_edges(_bgr())
        assert isinstance(result, FragmentEdges)
        assert result.n_segments == 4

    def test_high_epsilon_fewer_points(self):
        r_low = extract_fragment_edges(_gray(), epsilon=0.5)
        r_high = extract_fragment_edges(_gray(), epsilon=10.0)
        total_low = sum(len(seg) for seg in r_low.segments)
        total_high = sum(len(seg) for seg in r_high.segments)
        assert total_high <= total_low


# ─── batch_extract_edges ──────────────────────────────────────────────────────

class TestBatchExtractEdges:
    def test_returns_list(self):
        result = batch_extract_edges([_gray(), _bgr()])
        assert isinstance(result, list)

    def test_length_matches_input(self):
        imgs = [_gray(), _bgr(), _gray(32, 32)]
        result = batch_extract_edges(imgs)
        assert len(result) == 3

    def test_empty_input_returns_empty(self):
        result = batch_extract_edges([])
        assert result == []

    def test_each_result_is_fragment_edges(self):
        result = batch_extract_edges([_gray(), _bgr()])
        for fe in result:
            assert isinstance(fe, FragmentEdges)

    def test_all_have_four_segments(self):
        result = batch_extract_edges([_gray(), _bgr()])
        for fe in result:
            assert fe.n_segments == 4

    def test_params_propagated(self):
        result = batch_extract_edges([_gray()], threshold=20, epsilon=3.0)
        assert result[0].params["threshold"] == 20
        assert result[0].params["epsilon"] == pytest.approx(3.0)
