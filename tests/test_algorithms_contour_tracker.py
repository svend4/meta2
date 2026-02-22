"""Тесты для puzzle_reconstruction.algorithms.contour_tracker."""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _blank_mask(h=64, w=64) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _circle_mask(h=64, w=64, r=20) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (h // 2, w // 2), r, 255, -1)
    return mask


def _rect_contour() -> np.ndarray:
    return np.array([
        [10.0, 10.0],
        [50.0, 10.0],
        [50.0, 50.0],
        [10.0, 50.0],
    ], dtype=np.float32)


def _make_info(cx=0.0, cy=0.0) -> ContourInfo:
    pts = _rect_contour()
    return ContourInfo(
        contour=pts,
        area=100.0,
        perimeter=40.0,
        bbox=(10, 10, 40, 40),
        centroid=(cx, cy),
    )


def _make_state(track_id=0) -> TrackState:
    return TrackState(track_id=track_id, info=_make_info())


# ─── TestContourInfo ──────────────────────────────────────────────────────────

class TestContourInfo:
    def test_valid_construction(self):
        info = _make_info()
        assert info.area == pytest.approx(100.0)

    def test_negative_area_raises(self):
        with pytest.raises(ValueError):
            ContourInfo(
                contour=_rect_contour(),
                area=-1.0,
                perimeter=0.0,
                bbox=(0, 0, 1, 1),
                centroid=(0.0, 0.0),
            )

    def test_negative_perimeter_raises(self):
        with pytest.raises(ValueError):
            ContourInfo(
                contour=_rect_contour(),
                area=0.0,
                perimeter=-1.0,
                bbox=(0, 0, 1, 1),
                centroid=(0.0, 0.0),
            )

    def test_len_returns_n_points(self):
        info = _make_info()
        assert len(info) == 4

    def test_params_default_empty(self):
        info = _make_info()
        assert info.params == {}

    def test_params_stored(self):
        info = ContourInfo(
            contour=_rect_contour(),
            area=0.0,
            perimeter=0.0,
            bbox=(0, 0, 1, 1),
            centroid=(0.0, 0.0),
            params={"source": "test"},
        )
        assert info.params["source"] == "test"


# ─── TestTrackState ───────────────────────────────────────────────────────────

class TestTrackState:
    def test_valid_construction(self):
        ts = _make_state(track_id=3)
        assert ts.track_id == 3

    def test_negative_track_id_raises(self):
        with pytest.raises(ValueError):
            TrackState(track_id=-1, info=_make_info())

    def test_negative_age_raises(self):
        with pytest.raises(ValueError):
            TrackState(track_id=0, info=_make_info(), age=-1)

    def test_negative_lost_raises(self):
        with pytest.raises(ValueError):
            TrackState(track_id=0, info=_make_info(), lost=-1)

    def test_defaults(self):
        ts = _make_state()
        assert ts.age == 0
        assert ts.lost == 0


# ─── TestFindContours ─────────────────────────────────────────────────────────

class TestFindContours:
    def test_blank_mask_empty(self):
        result = find_contours(_blank_mask())
        assert result == []

    def test_circle_mask_one_contour(self):
        result = find_contours(_circle_mask())
        assert len(result) >= 1

    def test_returns_list_of_arrays(self):
        result = find_contours(_circle_mask())
        for cnt in result:
            assert isinstance(cnt, np.ndarray)
            assert cnt.shape[1] == 2

    def test_dtype_float32(self):
        result = find_contours(_circle_mask())
        assert result[0].dtype == np.float32

    def test_mode_external_ok(self):
        result = find_contours(_circle_mask(), mode="external")
        assert len(result) >= 1

    def test_mode_all_ok(self):
        result = find_contours(_circle_mask(), mode="all")
        assert len(result) >= 1

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            find_contours(_circle_mask(), mode="inner")

    def test_3d_mask_raises(self):
        mask3d = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            find_contours(mask3d)


# ─── TestContourToArray ───────────────────────────────────────────────────────

class TestContourToArray:
    def test_returns_float32(self):
        cnt = np.array([[[1, 2]], [[3, 4]]], dtype=np.int32)
        out = contour_to_array(cnt)
        assert out.dtype == np.float32

    def test_shape_n_2(self):
        cnt = np.array([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=np.int32)
        out = contour_to_array(cnt)
        assert out.shape == (3, 2)

    def test_already_flat(self):
        cnt = _rect_contour()
        out = contour_to_array(cnt)
        assert out.shape == (4, 2)


# ─── TestComputeContourInfo ───────────────────────────────────────────────────

class TestComputeContourInfo:
    def test_returns_contour_info(self):
        info = compute_contour_info(_circle_mask().astype(np.float32))
        assert isinstance(info, ContourInfo)

    def test_area_nonneg(self):
        info = compute_contour_info(_rect_contour())
        assert info.area >= 0.0

    def test_perimeter_nonneg(self):
        info = compute_contour_info(_rect_contour())
        assert info.perimeter >= 0.0

    def test_empty_contour_raises(self):
        with pytest.raises(ValueError):
            compute_contour_info(np.zeros((0, 2), dtype=np.float32))

    def test_centroid_is_tuple(self):
        info = compute_contour_info(_rect_contour())
        assert len(info.centroid) == 2


# ─── TestFilterContours ───────────────────────────────────────────────────────

class TestFilterContours:
    def test_min_area_negative_raises(self):
        with pytest.raises(ValueError):
            filter_contours([], min_area=-1.0)

    def test_max_leq_min_raises(self):
        with pytest.raises(ValueError):
            filter_contours([], min_area=10.0, max_area=5.0)

    def test_min_perimeter_negative_raises(self):
        with pytest.raises(ValueError):
            filter_contours([], min_perimeter=-1.0)

    def test_removes_small_contours(self):
        contours = find_contours(_circle_mask())
        # A very high min_area should filter everything out
        result = filter_contours(contours, min_area=1e9)
        assert len(result) == 0

    def test_keeps_valid_contours(self):
        contours = find_contours(_circle_mask())
        result = filter_contours(contours, min_area=0.0, max_area=1e9)
        assert len(result) == len(contours)


# ─── TestMatchContours ────────────────────────────────────────────────────────

class TestMatchContours:
    def test_max_dist_zero_raises(self):
        with pytest.raises(ValueError):
            match_contours([], [], max_dist=0.0)

    def test_max_dist_negative_raises(self):
        with pytest.raises(ValueError):
            match_contours([], [], max_dist=-1.0)

    def test_empty_prev_empty_result(self):
        curr = [_make_info(cx=10.0, cy=10.0)]
        assert match_contours([], curr) == []

    def test_empty_curr_empty_result(self):
        prev = [_make_info(cx=10.0, cy=10.0)]
        assert match_contours(prev, []) == []

    def test_identical_centroids_match(self):
        prev = [_make_info(cx=10.0, cy=10.0)]
        curr = [_make_info(cx=10.0, cy=10.0)]
        matches = match_contours(prev, curr, max_dist=5.0)
        assert len(matches) == 1
        assert matches[0] == (0, 0)

    def test_too_far_no_match(self):
        prev = [_make_info(cx=0.0, cy=0.0)]
        curr = [_make_info(cx=100.0, cy=100.0)]
        matches = match_contours(prev, curr, max_dist=5.0)
        assert len(matches) == 0

    def test_returns_list_of_tuples(self):
        prev = [_make_info(cx=5.0, cy=5.0)]
        curr = [_make_info(cx=6.0, cy=6.0)]
        matches = match_contours(prev, curr, max_dist=10.0)
        for m in matches:
            assert isinstance(m, tuple)
            assert len(m) == 2


# ─── TestTrackContour ─────────────────────────────────────────────────────────

class TestTrackContour:
    def test_with_new_info_age_incremented(self):
        state = _make_state()
        new_info = _make_info(cx=1.0, cy=1.0)
        updated = track_contour(state, new_info)
        assert updated.age == 1

    def test_with_new_info_lost_reset(self):
        state = TrackState(track_id=0, info=_make_info(), lost=3)
        updated = track_contour(state, _make_info())
        assert updated.lost == 0

    def test_with_none_lost_incremented(self):
        state = _make_state()
        updated = track_contour(state, None)
        assert updated.lost == 1

    def test_with_none_age_incremented(self):
        state = _make_state()
        updated = track_contour(state, None)
        assert updated.age == 1

    def test_track_id_preserved(self):
        state = _make_state(track_id=7)
        updated = track_contour(state, _make_info())
        assert updated.track_id == 7

    def test_returns_track_state(self):
        state = _make_state()
        result = track_contour(state, None)
        assert isinstance(result, TrackState)


# ─── TestBatchFindContours ────────────────────────────────────────────────────

class TestBatchFindContours:
    def test_returns_list(self):
        masks = [_circle_mask(), _blank_mask()]
        result = batch_find_contours(masks)
        assert isinstance(result, list)

    def test_length_matches(self):
        masks = [_circle_mask(), _blank_mask(), _circle_mask()]
        result = batch_find_contours(masks)
        assert len(result) == 3

    def test_empty_input(self):
        result = batch_find_contours([])
        assert result == []

    def test_blank_mask_gives_empty_list(self):
        result = batch_find_contours([_blank_mask()])
        assert result[0] == []

    def test_mode_all_ok(self):
        result = batch_find_contours([_circle_mask()], mode="all")
        assert isinstance(result, list)
