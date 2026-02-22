"""Tests for puzzle_reconstruction.algorithms.contour_tracker."""
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


# ─── ContourInfo ─────────────────────────────────────────────────────────────

class TestContourInfo:
    def test_fields_stored(self):
        info = _make_info()
        assert info.area == pytest.approx(400.0)
        assert info.perimeter == pytest.approx(80.0)
        assert info.centroid == pytest.approx((20.0, 20.0))

    def test_negative_area_raises(self):
        with pytest.raises(ValueError):
            ContourInfo(
                contour=_rect_contour(),
                area=-1.0,
                perimeter=10.0,
                bbox=(0, 0, 5, 5),
                centroid=(2.5, 2.5),
            )

    def test_negative_perimeter_raises(self):
        with pytest.raises(ValueError):
            ContourInfo(
                contour=_rect_contour(),
                area=10.0,
                perimeter=-1.0,
                bbox=(0, 0, 5, 5),
                centroid=(2.5, 2.5),
            )

    def test_len(self):
        info = _make_info()
        assert len(info) == 4

    def test_default_params_empty(self):
        info = _make_info()
        assert info.params == {}


# ─── TrackState ──────────────────────────────────────────────────────────────

class TestTrackState:
    def test_fields_stored(self):
        ts = TrackState(track_id=0, info=_make_info(), age=3, lost=1)
        assert ts.track_id == 0
        assert ts.age == 3
        assert ts.lost == 1

    def test_negative_track_id_raises(self):
        with pytest.raises(ValueError):
            TrackState(track_id=-1, info=_make_info())

    def test_negative_age_raises(self):
        with pytest.raises(ValueError):
            TrackState(track_id=0, info=_make_info(), age=-1)

    def test_negative_lost_raises(self):
        with pytest.raises(ValueError):
            TrackState(track_id=0, info=_make_info(), lost=-1)

    def test_default_age_zero(self):
        ts = TrackState(track_id=0, info=_make_info())
        assert ts.age == 0

    def test_default_lost_zero(self):
        ts = TrackState(track_id=0, info=_make_info())
        assert ts.lost == 0


# ─── find_contours ───────────────────────────────────────────────────────────

class TestFindContours:
    def test_returns_list(self):
        result = find_contours(_square_mask())
        assert isinstance(result, list)

    def test_finds_at_least_one_contour(self):
        result = find_contours(_square_mask())
        assert len(result) >= 1

    def test_empty_mask_no_contours(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        result = find_contours(mask)
        assert len(result) == 0

    def test_two_squares_finds_two(self):
        result = find_contours(_two_squares_mask())
        assert len(result) == 2

    def test_non_2d_mask_raises(self):
        with pytest.raises(ValueError):
            find_contours(np.zeros((32, 32, 3), dtype=np.uint8))

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            find_contours(_square_mask(), mode="inner")

    def test_all_mode_accepted(self):
        result = find_contours(_square_mask(), mode="all")
        assert isinstance(result, list)

    def test_contours_are_float32(self):
        result = find_contours(_square_mask())
        for cnt in result:
            assert cnt.dtype == np.float32

    def test_contours_are_n_2(self):
        result = find_contours(_square_mask())
        for cnt in result:
            assert cnt.ndim == 2
            assert cnt.shape[1] == 2


# ─── contour_to_array ────────────────────────────────────────────────────────

class TestContourToArray:
    def test_returns_float32(self):
        cnt = np.array([[[0, 0]], [[1, 0]], [[1, 1]]], dtype=np.int32)
        result = contour_to_array(cnt)
        assert result.dtype == np.float32

    def test_shape_n_2(self):
        cnt = np.array([[[0, 0]], [[1, 0]], [[1, 1]]], dtype=np.int32)
        result = contour_to_array(cnt)
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_already_n_2_unchanged(self):
        cnt = _rect_contour()
        result = contour_to_array(cnt)
        assert result.shape == cnt.shape


# ─── compute_contour_info ────────────────────────────────────────────────────

class TestComputeContourInfo:
    def test_returns_contour_info(self):
        result = compute_contour_info(_rect_contour())
        assert isinstance(result, ContourInfo)

    def test_area_nonnegative(self):
        result = compute_contour_info(_rect_contour())
        assert result.area >= 0.0

    def test_perimeter_nonnegative(self):
        result = compute_contour_info(_rect_contour())
        assert result.perimeter >= 0.0

    def test_bbox_tuple_length_4(self):
        result = compute_contour_info(_rect_contour())
        assert len(result.bbox) == 4

    def test_centroid_tuple_length_2(self):
        result = compute_contour_info(_rect_contour())
        assert len(result.centroid) == 2

    def test_empty_contour_raises(self):
        with pytest.raises(ValueError):
            compute_contour_info(np.zeros((0, 2), dtype=np.float32))


# ─── filter_contours ─────────────────────────────────────────────────────────

class TestFilterContours:
    def test_returns_list(self):
        cnts = find_contours(_square_mask())
        result = filter_contours(cnts, min_area=0.0)
        assert isinstance(result, list)

    def test_filters_small_contours(self):
        cnts = find_contours(_two_squares_mask())
        result = filter_contours(cnts, min_area=10000.0)
        assert len(result) == 0

    def test_keeps_large_contours(self):
        cnts = find_contours(_two_squares_mask())
        result = filter_contours(cnts, min_area=10.0, max_area=float("inf"))
        assert len(result) == len(cnts)

    def test_negative_min_area_raises(self):
        with pytest.raises(ValueError):
            filter_contours([], min_area=-1.0)

    def test_max_area_le_min_area_raises(self):
        with pytest.raises(ValueError):
            filter_contours([], min_area=100.0, max_area=50.0)

    def test_negative_min_perimeter_raises(self):
        with pytest.raises(ValueError):
            filter_contours([], min_perimeter=-1.0)

    def test_empty_input(self):
        result = filter_contours([], min_area=10.0)
        assert result == []


# ─── match_contours ───────────────────────────────────────────────────────────

class TestMatchContours:
    def test_returns_list_of_pairs(self):
        info1 = [_make_info(20.0, 20.0)]
        info2 = [_make_info(22.0, 21.0)]
        result = match_contours(info1, info2)
        assert isinstance(result, list)
        if result:
            assert len(result[0]) == 2

    def test_close_centroids_matched(self):
        info1 = [_make_info(20.0, 20.0)]
        info2 = [_make_info(21.0, 20.0)]
        result = match_contours(info1, info2, max_dist=10.0)
        assert len(result) == 1
        assert result[0] == (0, 0)

    def test_far_centroids_not_matched(self):
        info1 = [_make_info(0.0, 0.0)]
        info2 = [_make_info(200.0, 200.0)]
        result = match_contours(info1, info2, max_dist=10.0)
        assert len(result) == 0

    def test_max_dist_zero_raises(self):
        with pytest.raises(ValueError):
            match_contours([_make_info()], [_make_info()], max_dist=0.0)

    def test_empty_prev_returns_empty(self):
        result = match_contours([], [_make_info()])
        assert result == []

    def test_empty_curr_returns_empty(self):
        result = match_contours([_make_info()], [])
        assert result == []


# ─── track_contour ────────────────────────────────────────────────────────────

class TestTrackContour:
    def test_with_new_info_updates(self):
        state = TrackState(track_id=0, info=_make_info(20.0, 20.0), age=1)
        new_info = _make_info(22.0, 22.0)
        updated = track_contour(state, new_info)
        assert updated.age == 2
        assert updated.lost == 0
        assert updated.info.centroid == pytest.approx((22.0, 22.0))

    def test_without_info_increments_lost(self):
        state = TrackState(track_id=0, info=_make_info(), age=5, lost=1)
        updated = track_contour(state, None)
        assert updated.lost == 2
        assert updated.age == 6

    def test_track_id_preserved(self):
        state = TrackState(track_id=7, info=_make_info())
        updated = track_contour(state, _make_info())
        assert updated.track_id == 7

    def test_returns_new_track_state(self):
        state = TrackState(track_id=0, info=_make_info())
        updated = track_contour(state, _make_info())
        assert isinstance(updated, TrackState)


# ─── batch_find_contours ─────────────────────────────────────────────────────

class TestBatchFindContours:
    def test_returns_list(self):
        result = batch_find_contours([_square_mask()])
        assert isinstance(result, list)

    def test_length_matches(self):
        masks = [_square_mask(), _two_squares_mask()]
        result = batch_find_contours(masks)
        assert len(result) == 2

    def test_empty_input_returns_empty(self):
        assert batch_find_contours([]) == []

    def test_inner_lists_are_lists(self):
        result = batch_find_contours([_square_mask()])
        assert isinstance(result[0], list)
