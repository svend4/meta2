"""Extra tests for puzzle_reconstruction.algorithms.contour_tracker (v2)."""
import pytest
import numpy as np
import cv2
from puzzle_reconstruction.algorithms.contour_tracker import (
    ContourInfo,
    TrackState,
    find_contours,
    contour_to_array,
    compute_contour_info,
    filter_contours,
    match_contours,
    track_contour,
    batch_find_contours,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _blank_mask(h=64, w=64) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _circle_mask(h=64, w=64, r=20) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (h // 2, w // 2), r, 255, -1)
    return mask


def _rect_contour() -> np.ndarray:
    return np.array([
        [10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0],
    ], dtype=np.float32)


def _make_info(cx=0.0, cy=0.0) -> ContourInfo:
    return ContourInfo(
        contour=_rect_contour(), area=100.0, perimeter=40.0,
        bbox=(10, 10, 40, 40), centroid=(cx, cy),
    )


def _make_state(track_id=0) -> TrackState:
    return TrackState(track_id=track_id, info=_make_info())


# ─── TestContourInfoExtra ────────────────────────────────────────────────────

class TestContourInfoV2Extra:
    def test_area_stored(self):
        info = _make_info()
        assert info.area == pytest.approx(100.0)

    def test_perimeter_stored(self):
        info = _make_info()
        assert info.perimeter == pytest.approx(40.0)

    def test_bbox_tuple(self):
        info = _make_info()
        assert len(info.bbox) == 4

    def test_centroid_tuple(self):
        info = _make_info(5.0, 10.0)
        assert info.centroid == pytest.approx((5.0, 10.0))

    def test_contour_ndarray(self):
        info = _make_info()
        assert isinstance(info.contour, np.ndarray)
        assert info.contour.dtype == np.float32

    def test_zero_area_ok(self):
        info = ContourInfo(contour=_rect_contour(), area=0.0, perimeter=0.0,
                           bbox=(0, 0, 1, 1), centroid=(0.0, 0.0))
        assert info.area == pytest.approx(0.0)

    def test_large_values(self):
        info = ContourInfo(contour=_rect_contour(), area=1e9, perimeter=1e5,
                           bbox=(0, 0, 1e4, 1e4), centroid=(5000.0, 5000.0))
        assert info.area == pytest.approx(1e9)


# ─── TestTrackStateExtra ─────────────────────────────────────────────────────

class TestTrackStateV2Extra:
    def test_age_stored(self):
        ts = TrackState(track_id=0, info=_make_info(), age=5)
        assert ts.age == 5

    def test_lost_stored(self):
        ts = TrackState(track_id=0, info=_make_info(), lost=3)
        assert ts.lost == 3

    def test_info_accessible(self):
        ts = _make_state(track_id=7)
        assert ts.info.area == pytest.approx(100.0)

    def test_default_age_zero(self):
        ts = _make_state()
        assert ts.age == 0

    def test_default_lost_zero(self):
        ts = _make_state()
        assert ts.lost == 0


# ─── TestFindContoursExtra ───────────────────────────────────────────────────

class TestFindContoursV2Extra:
    def test_circle_points_nonneg(self):
        result = find_contours(_circle_mask())
        for cnt in result:
            assert np.all(cnt >= 0)

    def test_circle_points_within_bounds(self):
        h, w = 64, 64
        result = find_contours(_circle_mask(h, w))
        for cnt in result:
            assert np.all(cnt[:, 0] <= w)
            assert np.all(cnt[:, 1] <= h)

    def test_two_circles(self):
        mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.circle(mask, (32, 32), 15, 255, -1)
        cv2.circle(mask, (96, 96), 15, 255, -1)
        result = find_contours(mask, mode="external")
        assert len(result) == 2

    def test_small_mask_8x8(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        result = find_contours(mask)
        assert len(result) >= 1

    def test_returns_list(self):
        result = find_contours(_circle_mask())
        assert isinstance(result, list)

    def test_mode_all_returns_more_or_equal(self):
        ext = find_contours(_circle_mask(), mode="external")
        all_ = find_contours(_circle_mask(), mode="all")
        assert len(all_) >= len(ext)


# ─── TestContourToArrayExtra ────────────────────────────────────────────────

class TestContourToArrayV2Extra:
    def test_preserves_values(self):
        cnt = _rect_contour()
        out = contour_to_array(cnt)
        np.testing.assert_array_almost_equal(out, cnt)

    def test_3d_opencv_format(self):
        cnt = np.array([[[5, 10]], [[15, 20]]], dtype=np.int32)
        out = contour_to_array(cnt)
        assert out.shape == (2, 2)
        assert out.dtype == np.float32

    def test_single_point(self):
        cnt = np.array([[[7, 9]]], dtype=np.int32)
        out = contour_to_array(cnt)
        assert out.shape == (1, 2)


# ─── TestComputeContourInfoExtra ─────────────────────────────────────────────

class TestComputeContourInfoV2Extra:
    def test_triangle(self):
        tri = np.array([[0, 0], [10, 0], [5, 10]], dtype=np.float32)
        info = compute_contour_info(tri)
        assert info.area > 0

    def test_large_contour(self):
        n = 100
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        circle = np.column_stack([np.cos(t) * 50 + 50,
                                   np.sin(t) * 50 + 50]).astype(np.float32)
        info = compute_contour_info(circle)
        assert info.area > 0
        assert info.perimeter > 0

    def test_bbox_nonneg(self):
        info = compute_contour_info(_rect_contour())
        assert all(v >= 0 for v in info.bbox)

    def test_centroid_within_bbox(self):
        info = compute_contour_info(_rect_contour())
        bx, by, bw, bh = info.bbox
        cx, cy = info.centroid
        assert bx <= cx <= bx + bw
        assert by <= cy <= by + bh


# ─── TestFilterContoursExtra ─────────────────────────────────────────────────

class TestFilterContoursV2Extra:
    def test_max_area_small(self):
        contours = find_contours(_circle_mask())
        result = filter_contours(contours, min_area=0.0, max_area=1.0)
        assert len(result) == 0

    def test_min_perimeter_huge(self):
        contours = find_contours(_circle_mask())
        result = filter_contours(contours, min_area=0.0, min_perimeter=1e8)
        assert len(result) == 0

    def test_no_filter_keeps_all(self):
        contours = find_contours(_circle_mask())
        result = filter_contours(contours, min_area=0.0)
        assert len(result) == len(contours)

    def test_empty_list_ok(self):
        result = filter_contours([], min_area=0.0)
        assert result == []


# ─── TestMatchContoursExtra ──────────────────────────────────────────────────

class TestMatchContoursV2Extra:
    def test_multiple_matches(self):
        prev = [_make_info(10, 10), _make_info(50, 50)]
        curr = [_make_info(11, 10), _make_info(51, 50)]
        matches = match_contours(prev, curr, max_dist=10.0)
        assert len(matches) == 2

    def test_one_close_one_far(self):
        prev = [_make_info(0, 0), _make_info(100, 100)]
        curr = [_make_info(1, 0)]
        matches = match_contours(prev, curr, max_dist=5.0)
        assert len(matches) == 1
        assert matches[0] == (0, 0)

    def test_exact_overlap_match(self):
        info = [_make_info(20.0, 20.0)]
        matches = match_contours(info, info, max_dist=1.0)
        assert len(matches) == 1

    def test_returns_list_of_pairs(self):
        prev = [_make_info(5.0, 5.0)]
        curr = [_make_info(6.0, 6.0)]
        matches = match_contours(prev, curr, max_dist=10.0)
        for m in matches:
            assert len(m) == 2


# ─── TestTrackContourExtra ──────────────────────────────────────────────────

class TestTrackContourV2Extra:
    def test_consecutive_lost(self):
        state = _make_state()
        s1 = track_contour(state, None)
        s2 = track_contour(s1, None)
        s3 = track_contour(s2, None)
        assert s3.lost == 3
        assert s3.age == 3

    def test_lost_reset_after_match(self):
        state = TrackState(track_id=0, info=_make_info(), lost=5, age=5)
        updated = track_contour(state, _make_info(1.0, 1.0))
        assert updated.lost == 0
        assert updated.age == 6

    def test_info_updated(self):
        state = _make_state()
        new_info = _make_info(99.0, 99.0)
        updated = track_contour(state, new_info)
        assert updated.info.centroid == pytest.approx((99.0, 99.0))

    def test_info_preserved_on_none(self):
        state = _make_state()
        updated = track_contour(state, None)
        assert updated.info.centroid == pytest.approx((0.0, 0.0))

    def test_track_id_preserved(self):
        state = _make_state(track_id=42)
        updated = track_contour(state, _make_info())
        assert updated.track_id == 42


# ─── TestBatchFindContoursExtra ──────────────────────────────────────────────

class TestBatchFindContoursV2Extra:
    def test_mixed(self):
        masks = [_circle_mask(), _blank_mask(), _circle_mask(32, 32, 10)]
        result = batch_find_contours(masks)
        assert len(result) == 3
        assert len(result[0]) >= 1
        assert len(result[1]) == 0
        assert len(result[2]) >= 1

    def test_all_blank(self):
        result = batch_find_contours([_blank_mask()] * 3)
        for r in result:
            assert r == []

    def test_single_circle(self):
        result = batch_find_contours([_circle_mask()])
        assert len(result) == 1
        assert len(result[0]) >= 1

    def test_all_results_are_lists(self):
        result = batch_find_contours([_circle_mask(), _blank_mask()])
        for r in result:
            assert isinstance(r, list)

    def test_contour_dtype(self):
        result = batch_find_contours([_circle_mask()])
        for cnt in result[0]:
            assert cnt.dtype == np.float32
