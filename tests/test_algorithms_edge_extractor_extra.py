"""Extra tests for puzzle_reconstruction.algorithms.edge_extractor."""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, fill=200):
    return np.full((h, w), fill, dtype=np.uint8)


def _bgr(h=64, w=64, fill=200):
    return np.full((h, w, 3), fill, dtype=np.uint8)


def _blank(h=64, w=64):
    return np.zeros((h, w), dtype=np.uint8)


def _pts(n=10, length=10.0):
    x = np.linspace(0.0, length, n)
    return np.stack([x, np.zeros(n)], axis=1).astype(np.float32)


def _diagonal_pts(n=10, length=10.0):
    t = np.linspace(0, length, n)
    return np.stack([t, t], axis=1).astype(np.float32)


# ─── EdgeSegment extras ───────────────────────────────────────────────────────

class TestEdgeSegmentExtra:
    def test_repr_is_string(self):
        seg = EdgeSegment(points=_pts(), side="top", length=10.0)
        assert isinstance(repr(seg), str)

    def test_all_valid_sides(self):
        for side in ("top", "bottom", "left", "right", "unknown"):
            seg = EdgeSegment(points=_pts(), side=side, length=0.0)
            assert seg.side == side

    def test_large_point_count(self):
        pts = np.zeros((500, 2), dtype=np.float32)
        seg = EdgeSegment(points=pts, side="top", length=100.0)
        assert len(seg) == 500

    def test_params_stored(self):
        seg = EdgeSegment(points=_pts(), side="top", length=5.0,
                          params={"epsilon": 1.0})
        assert seg.params["epsilon"] == pytest.approx(1.0)

    def test_zero_length_valid(self):
        seg = EdgeSegment(points=_pts(2), side="top", length=0.0)
        assert seg.length == pytest.approx(0.0)

    def test_len_matches_points(self):
        pts = _pts(7)
        seg = EdgeSegment(points=pts, side="bottom", length=10.0)
        assert len(seg) == 7

    def test_length_stored(self):
        seg = EdgeSegment(points=_pts(), side="left", length=77.7)
        assert seg.length == pytest.approx(77.7)


# ─── FragmentEdges extras ─────────────────────────────────────────────────────

class TestFragmentEdgesExtra:
    def test_repr_is_string(self):
        fe = FragmentEdges(segments=[], n_segments=0)
        assert isinstance(repr(fe), str)

    def test_four_segments_stored(self):
        segs = [EdgeSegment(points=_pts(), side=s, length=0.0)
                for s in ("top", "bottom", "left", "right")]
        fe = FragmentEdges(segments=segs, n_segments=4)
        assert fe.n_segments == 4
        assert len(fe) == 4

    def test_segments_list_stored(self):
        seg = EdgeSegment(points=_pts(), side="top", length=10.0)
        fe = FragmentEdges(segments=[seg], n_segments=1)
        assert fe.segments[0].side == "top"

    def test_params_stored(self):
        fe = FragmentEdges(segments=[], n_segments=0,
                           params={"threshold": 10})
        assert fe.params["threshold"] == 10

    def test_zero_segments_valid(self):
        fe = FragmentEdges(segments=[], n_segments=0)
        assert len(fe) == 0


# ─── detect_boundary extras ───────────────────────────────────────────────────

class TestDetectBoundaryExtra:
    def test_large_image_runs(self):
        out = detect_boundary(_gray(h=256, w=512))
        assert out.shape == (256, 512)

    def test_small_image_runs(self):
        out = detect_boundary(_gray(h=8, w=8, fill=150))
        assert out.dtype == np.uint8

    def test_non_square_shape(self):
        out = detect_boundary(_gray(h=32, w=80))
        assert out.shape == (32, 80)

    def test_threshold_0_all_nonzero_or_zero(self):
        out = detect_boundary(_gray(fill=200), threshold=0)
        # Should not crash
        assert out.dtype == np.uint8

    def test_threshold_254_valid(self):
        out = detect_boundary(_gray(fill=200), threshold=254)
        assert out.dtype == np.uint8

    def test_output_binary(self):
        out = detect_boundary(_gray(fill=150))
        unique = set(np.unique(out))
        assert unique.issubset({0, 255})

    def test_rgb_output_2d(self):
        out = detect_boundary(_bgr(fill=180))
        assert out.ndim == 2

    def test_blank_no_boundary(self):
        out = detect_boundary(_blank())
        assert out.max() == 0


# ─── extract_edge_points extras ───────────────────────────────────────────────

class TestExtractEdgePointsExtra:
    def test_multiple_pixels_correct_count(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[5, 5] = 255
        mask[10, 15] = 255
        mask[20, 25] = 255
        pts = extract_edge_points(mask)
        assert len(pts) == 3

    def test_non_square_mask(self):
        mask = np.zeros((16, 48), dtype=np.uint8)
        mask[8, 24] = 255
        pts = extract_edge_points(mask)
        assert pts.shape == (1, 2)

    def test_full_row_multiple_points(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[0, :] = 255
        pts = extract_edge_points(mask)
        assert len(pts) == 32

    def test_xy_convention(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[3, 7] = 255  # row=3 (y), col=7 (x)
        pts = extract_edge_points(mask)
        assert len(pts) == 1
        # x=col=7, y=row=3
        assert pts[0, 0] == pytest.approx(7.0)
        assert pts[0, 1] == pytest.approx(3.0)

    def test_dtype_float32(self):
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[8, 8] = 255
        pts = extract_edge_points(mask)
        assert pts.dtype == np.float32


# ─── split_edge_by_side extras ────────────────────────────────────────────────

class TestSplitEdgeBySideExtra:
    def test_bottom_row_goes_to_bottom(self):
        """Points at y=63 (bottom of 64×64) should go to 'bottom'."""
        x = np.linspace(10, 50, 5, dtype=np.float32)
        pts = np.stack([x, np.full(5, 63.0, dtype=np.float32)], axis=1)
        result = split_edge_by_side(pts, (64, 64))
        assert len(result["bottom"]) > 0

    def test_left_column_goes_to_left(self):
        """Points at x=0 should go to 'left'."""
        y = np.linspace(10, 50, 5, dtype=np.float32)
        pts = np.stack([np.zeros(5, dtype=np.float32), y], axis=1)
        result = split_edge_by_side(pts, (64, 64))
        assert len(result["left"]) > 0

    def test_right_column_goes_to_right(self):
        """Points at x=63 should go to 'right'."""
        y = np.linspace(10, 50, 5, dtype=np.float32)
        pts = np.stack([np.full(5, 63.0, dtype=np.float32), y], axis=1)
        result = split_edge_by_side(pts, (64, 64))
        assert len(result["right"]) > 0

    def test_returns_four_keys(self):
        result = split_edge_by_side(_pts(), (64, 64))
        assert set(result.keys()) == {"top", "bottom", "left", "right"}

    def test_all_values_float32(self):
        result = split_edge_by_side(_pts(), (64, 64))
        for arr in result.values():
            assert arr.dtype == np.float32

    def test_zero_width_raises(self):
        with pytest.raises(ValueError):
            split_edge_by_side(_pts(), (64, 0))

    def test_total_points_conserved(self):
        """Union of all side points ≤ original (some may be shared borders)."""
        x = np.array([0.0, 63.0, 10.0, 20.0], dtype=np.float32)
        y = np.array([0.0, 63.0, 0.0, 63.0], dtype=np.float32)
        pts = np.stack([x, y], axis=1)
        result = split_edge_by_side(pts, (64, 64))
        total = sum(len(v) for v in result.values())
        assert total >= len(pts)  # corners may be counted in multiple sides


# ─── compute_edge_length extras ───────────────────────────────────────────────

class TestComputeEdgeLengthExtra:
    def test_diagonal_line_length(self):
        """Diagonal (1,1)→(4,4): length=3√2."""
        pts = np.array([[1.0, 1.0], [4.0, 4.0]], dtype=np.float32)
        length = compute_edge_length(pts)
        assert length == pytest.approx(3.0 * np.sqrt(2), abs=1e-4)

    def test_multiple_collinear_points(self):
        pts = _pts(n=101, length=100.0)
        assert compute_edge_length(pts) == pytest.approx(100.0, abs=1e-3)

    def test_two_points(self):
        pts = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        assert compute_edge_length(pts) == pytest.approx(5.0, abs=1e-5)

    def test_closed_loop_no_crash(self):
        pts = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0., 0.]], dtype=np.float32)
        length = compute_edge_length(pts)
        assert length == pytest.approx(4.0, abs=1e-5)

    def test_large_point_set(self):
        pts = _pts(n=1000, length=100.0)
        length = compute_edge_length(pts)
        assert length == pytest.approx(100.0, abs=0.5)


# ─── simplify_edge extras ─────────────────────────────────────────────────────

class TestSimplifyEdgeExtra:
    def test_large_epsilon_keeps_fewer(self):
        pts = _pts(n=50, length=50.0)
        loose = simplify_edge(pts, epsilon=10.0)
        tight = simplify_edge(pts, epsilon=0.01)
        assert loose.shape[0] <= tight.shape[0]

    def test_non_straight_curve_reduces(self):
        t = np.linspace(0, 2 * np.pi, 100)
        pts = np.stack([t, np.sin(t)], axis=1).astype(np.float32)
        out = simplify_edge(pts, epsilon=0.5)
        assert out.shape[0] < 100

    def test_output_ndim_2(self):
        out = simplify_edge(_pts(20))
        assert out.ndim == 2

    def test_output_cols_2(self):
        out = simplify_edge(_pts(10))
        assert out.shape[1] == 2

    def test_single_point_returns_single(self):
        pts = np.array([[1.0, 2.0]], dtype=np.float32)
        out = simplify_edge(pts, epsilon=1.0)
        assert out.shape[0] >= 1

    def test_diagonal_line_simplified(self):
        pts = _diagonal_pts(n=50, length=50.0)
        out = simplify_edge(pts, epsilon=1.0)
        assert out.shape[0] <= 50

    def test_output_dtype_float32(self):
        out = simplify_edge(_pts(15), epsilon=0.5)
        assert out.dtype == np.float32


# ─── extract_fragment_edges extras ───────────────────────────────────────────

class TestExtractFragmentEdgesExtra:
    def test_non_square_image(self):
        result = extract_fragment_edges(_gray(h=48, w=96))
        assert isinstance(result, FragmentEdges)

    def test_n_segments_4(self):
        result = extract_fragment_edges(_gray())
        assert result.n_segments == 4

    def test_segment_sides_set(self):
        result = extract_fragment_edges(_gray())
        sides = {seg.side for seg in result.segments}
        assert sides == {"top", "bottom", "left", "right"}

    def test_params_threshold_stored(self):
        result = extract_fragment_edges(_gray(), threshold=15)
        assert result.params["threshold"] == 15

    def test_params_epsilon_stored(self):
        result = extract_fragment_edges(_gray(), epsilon=3.0)
        assert result.params["epsilon"] == pytest.approx(3.0)

    def test_segment_points_float32(self):
        result = extract_fragment_edges(_gray())
        for seg in result.segments:
            assert seg.points.dtype == np.float32

    def test_bgr_image_ok(self):
        result = extract_fragment_edges(_bgr())
        assert isinstance(result, FragmentEdges)

    def test_all_segment_lengths_nonneg(self):
        result = extract_fragment_edges(_gray())
        for seg in result.segments:
            assert seg.length >= 0.0


# ─── batch_extract_edges extras ───────────────────────────────────────────────

class TestBatchExtractEdgesExtra:
    def test_five_images(self):
        imgs = [_gray() if i % 2 == 0 else _bgr() for i in range(5)]
        result = batch_extract_edges(imgs)
        assert len(result) == 5

    def test_single_image(self):
        result = batch_extract_edges([_gray()])
        assert len(result) == 1
        assert isinstance(result[0], FragmentEdges)

    def test_all_fragment_edges_type(self):
        imgs = [_gray(), _bgr(), _gray()]
        for fe in batch_extract_edges(imgs):
            assert isinstance(fe, FragmentEdges)

    def test_n_segments_4_for_each(self):
        imgs = [_gray() for _ in range(3)]
        for fe in batch_extract_edges(imgs):
            assert fe.n_segments == 4

    def test_non_square_images(self):
        imgs = [_gray(h=32, w=64), _gray(h=64, w=32)]
        result = batch_extract_edges(imgs)
        assert len(result) == 2
