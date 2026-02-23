"""Extra tests for puzzle_reconstruction.algorithms.contour_tracker."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.algorithms.contour_tracker import (
    ContourInfo,
    TrackState,
    batch_find_contours,
    compute_contour_info,
    contour_to_array,
    filter_contours,
    find_contours,
    match_contours,
    track_contour,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _square_mask(h: int = 64, w: int = 64, margin: int = 10) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[margin:h - margin, margin:w - margin] = 255
    return mask


def _two_squares_mask() -> np.ndarray:
    mask = np.zeros((64, 128), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    mask[10:30, 90:110] = 255
    return mask


def _rect_contour() -> np.ndarray:
    return np.array([
        [10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]
    ], dtype=np.float32)


def _make_info(cx: float = 20.0, cy: float = 20.0,
               area: float = 400.0) -> ContourInfo:
    cnt = _rect_contour()
    return ContourInfo(
        contour=cnt,
        area=area,
        perimeter=80.0,
        bbox=(10, 10, 20, 20),
        centroid=(cx, cy),
    )


# ─── TestContourInfoExtra ────────────────────────────────────────────────────

class TestContourInfoExtra:
    def test_perimeter_stored(self):
        info = _make_info()
        assert info.perimeter == pytest.approx(80.0)

    def test_bbox_stored(self):
        info = _make_info()
        assert info.bbox == (10, 10, 20, 20)

    def test_centroid_stored(self):
        info = _make_info(15.0, 25.0)
        assert info.centroid == pytest.approx((15.0, 25.0))

    def test_contour_shape(self):
        info = _make_info()
        assert info.contour.shape == (4, 2)

    def test_contour_dtype(self):
        info = _make_info()
        assert info.contour.dtype == np.float32

    def test_zero_area_ok(self):
        info = ContourInfo(
            contour=_rect_contour(), area=0.0, perimeter=10.0,
            bbox=(0, 0, 1, 1), centroid=(0.0, 0.0),
        )
        assert info.area == pytest.approx(0.0)

    def test_zero_perimeter_ok(self):
        info = ContourInfo(
            contour=_rect_contour(), area=10.0, perimeter=0.0,
            bbox=(0, 0, 1, 1), centroid=(0.0, 0.0),
        )
        assert info.perimeter == pytest.approx(0.0)

    def test_large_area_ok(self):
        info = ContourInfo(
            contour=_rect_contour(), area=1e8, perimeter=4e4,
            bbox=(0, 0, 1e4, 1e4), centroid=(5000.0, 5000.0),
        )
        assert info.area == pytest.approx(1e8)

    def test_params_stored(self):
        info = ContourInfo(
            contour=_rect_contour(), area=10.0, perimeter=10.0,
            bbox=(0, 0, 5, 5), centroid=(2.5, 2.5), params={"k": 42},
        )
        assert info.params["k"] == 42


# ─── TestTrackStateExtra ─────────────────────────────────────────────────────

class TestTrackStateExtra:
    def test_default_values(self):
        ts = TrackState(track_id=0, info=_make_info())
        assert ts.age == 0
        assert ts.lost == 0

    def test_info_stored(self):
        info = _make_info(5.0, 5.0)
        ts = TrackState(track_id=1, info=info)
        assert ts.info.centroid == pytest.approx((5.0, 5.0))

    def test_track_id_stored(self):
        ts = TrackState(track_id=99, info=_make_info())
        assert ts.track_id == 99

    def test_age_stored(self):
        ts = TrackState(track_id=0, info=_make_info(), age=10)
        assert ts.age == 10

    def test_lost_stored(self):
        ts = TrackState(track_id=0, info=_make_info(), lost=3)
        assert ts.lost == 3

    def test_zero_track_id_ok(self):
        ts = TrackState(track_id=0, info=_make_info())
        assert ts.track_id == 0

    def test_large_track_id(self):
        ts = TrackState(track_id=99999, info=_make_info())
        assert ts.track_id == 99999


# ─── TestFindContoursExtra ───────────────────────────────────────────────────

class TestFindContoursExtra:
    def test_full_white_mask(self):
        mask = np.full((64, 64), 255, dtype=np.uint8)
        result = find_contours(mask)
        assert isinstance(result, list)

    def test_single_pixel_contour(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[16, 16] = 255
        result = find_contours(mask)
        assert isinstance(result, list)

    def test_small_mask(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        result = find_contours(mask)
        assert len(result) >= 1

    def test_large_margin(self):
        result = find_contours(_square_mask(128, 128, 30))
        assert len(result) >= 1

    def test_mode_external(self):
        result = find_contours(_square_mask(), mode="external")
        assert isinstance(result, list)

    def test_contour_points_nonneg(self):
        result = find_contours(_square_mask())
        for cnt in result:
            assert np.all(cnt >= 0)


# ─── TestContourToArrayExtra ────────────────────────────────────────────────

class TestContourToArrayExtra:
    def test_int_to_float32(self):
        cnt = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int32)
        result = contour_to_array(cnt)
        assert result.dtype == np.float32

    def test_opencv_format_3d(self):
        cnt = np.array([[[5, 10]], [[15, 20]], [[25, 30]]], dtype=np.int32)
        result = contour_to_array(cnt)
        assert result.shape == (3, 2)
        assert result[0, 0] == pytest.approx(5.0)
        assert result[0, 1] == pytest.approx(10.0)

    def test_single_point(self):
        cnt = np.array([[[7, 9]]], dtype=np.int32)
        result = contour_to_array(cnt)
        assert result.shape == (1, 2)

    def test_values_preserved(self):
        cnt = _rect_contour()
        result = contour_to_array(cnt)
        np.testing.assert_array_almost_equal(result, cnt)


# ─── TestComputeContourInfoExtra ─────────────────────────────────────────────

class TestComputeContourInfoExtra:
    def test_rect_area_positive(self):
        result = compute_contour_info(_rect_contour())
        assert result.area > 0

    def test_rect_perimeter_positive(self):
        result = compute_contour_info(_rect_contour())
        assert result.perimeter > 0

    def test_centroid_within_bbox(self):
        result = compute_contour_info(_rect_contour())
        bx, by, bw, bh = result.bbox
        cx, cy = result.centroid
        assert bx <= cx <= bx + bw
        assert by <= cy <= by + bh

    def test_len_equals_contour_len(self):
        result = compute_contour_info(_rect_contour())
        assert len(result) == 4

    def test_triangle(self):
        tri = np.array([[0, 0], [10, 0], [5, 10]], dtype=np.float32)
        result = compute_contour_info(tri)
        assert result.area > 0
        assert result.perimeter > 0

    def test_large_contour(self):
        n = 100
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        circle = np.column_stack([np.cos(t) * 50 + 50,
                                   np.sin(t) * 50 + 50]).astype(np.float32)
        result = compute_contour_info(circle)
        assert result.area > 0
        assert len(result) == n


# ─── TestFilterContoursExtra ─────────────────────────────────────────────────

class TestFilterContoursExtra:
    def test_max_area_filter(self):
        cnts = find_contours(_square_mask())
        result = filter_contours(cnts, min_area=0.0, max_area=1.0)
        assert len(result) == 0

    def test_min_perimeter_filter(self):
        cnts = find_contours(_square_mask())
        result = filter_contours(cnts, min_area=0.0, min_perimeter=1e6)
        assert len(result) == 0

    def test_zero_min_area_keeps_all(self):
        cnts = find_contours(_square_mask())
        result = filter_contours(cnts, min_area=0.0)
        assert len(result) == len(cnts)

    def test_returns_contour_arrays(self):
        cnts = find_contours(_square_mask())
        result = filter_contours(cnts, min_area=0.0)
        for c in result:
            assert isinstance(c, np.ndarray)

    def test_moderate_area_threshold(self):
        cnts = find_contours(_two_squares_mask())
        result = filter_contours(cnts, min_area=50.0)
        assert len(result) <= len(cnts)


# ─── TestMatchContoursExtra ──────────────────────────────────────────────────

class TestMatchContoursExtra:
    def test_multiple_matches(self):
        info1 = [_make_info(10, 10), _make_info(50, 50)]
        info2 = [_make_info(11, 10), _make_info(51, 50)]
        result = match_contours(info1, info2, max_dist=10.0)
        assert len(result) == 2

    def test_one_to_one(self):
        info1 = [_make_info(0, 0), _make_info(100, 100)]
        info2 = [_make_info(1, 0)]
        result = match_contours(info1, info2, max_dist=10.0)
        assert len(result) <= 1

    def test_negative_max_dist_raises(self):
        with pytest.raises(ValueError):
            match_contours([_make_info()], [_make_info()], max_dist=-5.0)

    def test_both_empty(self):
        result = match_contours([], [])
        assert result == []

    def test_returns_tuples_of_ints(self):
        info1 = [_make_info(20, 20)]
        info2 = [_make_info(21, 20)]
        result = match_contours(info1, info2, max_dist=10.0)
        if result:
            assert isinstance(result[0][0], (int, np.integer))
            assert isinstance(result[0][1], (int, np.integer))

    def test_exact_overlap(self):
        info = [_make_info(20.0, 20.0)]
        result = match_contours(info, info, max_dist=1.0)
        assert len(result) == 1
        assert result[0] == (0, 0)


# ─── TestTrackContourExtra ───────────────────────────────────────────────────

class TestTrackContourExtra:
    def test_age_increments(self):
        state = TrackState(track_id=0, info=_make_info(), age=0)
        updated = track_contour(state, _make_info())
        assert updated.age == 1

    def test_lost_resets_on_match(self):
        state = TrackState(track_id=0, info=_make_info(), age=3, lost=5)
        updated = track_contour(state, _make_info())
        assert updated.lost == 0

    def test_multiple_lost_increments(self):
        state = TrackState(track_id=0, info=_make_info(), age=0, lost=0)
        s1 = track_contour(state, None)
        s2 = track_contour(s1, None)
        assert s2.lost == 2
        assert s2.age == 2

    def test_info_updated_to_new(self):
        state = TrackState(track_id=0, info=_make_info(10, 10), age=0)
        new_info = _make_info(30, 30)
        updated = track_contour(state, new_info)
        assert updated.info.centroid == pytest.approx((30.0, 30.0))

    def test_info_preserved_on_none(self):
        orig_info = _make_info(15, 15)
        state = TrackState(track_id=0, info=orig_info, age=0)
        updated = track_contour(state, None)
        assert updated.info.centroid == pytest.approx((15.0, 15.0))

    def test_returns_track_state(self):
        state = TrackState(track_id=0, info=_make_info())
        assert isinstance(track_contour(state, _make_info()), TrackState)


# ─── TestBatchFindContoursExtra ──────────────────────────────────────────────

class TestBatchFindContoursExtra:
    def test_single_mask(self):
        result = batch_find_contours([_square_mask()])
        assert len(result) == 1
        assert len(result[0]) >= 1

    def test_mixed_masks(self):
        masks = [_square_mask(), np.zeros((32, 32), dtype=np.uint8)]
        result = batch_find_contours(masks)
        assert len(result) == 2
        assert len(result[0]) >= 1
        assert len(result[1]) == 0

    def test_two_squares_mask_in_batch(self):
        result = batch_find_contours([_two_squares_mask()])
        assert len(result[0]) == 2

    def test_multiple_identical(self):
        masks = [_square_mask()] * 5
        result = batch_find_contours(masks)
        assert len(result) == 5
        for r in result:
            assert len(r) >= 1

    def test_all_inner_are_ndarray(self):
        result = batch_find_contours([_square_mask()])
        for cnt in result[0]:
            assert isinstance(cnt, np.ndarray)
