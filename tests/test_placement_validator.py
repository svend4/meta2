"""Тесты для puzzle_reconstruction.verification.placement_validator."""
import numpy as np
import pytest

from puzzle_reconstruction.verification.placement_validator import (
    CollisionReport,
    PlacementBox,
    PlacementConfig,
    batch_validate_placements,
    box_iou,
    compute_coverage,
    find_collisions,
    find_duplicate_positions,
    find_out_of_bounds,
    validate_placements,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _box(fid, x=0, y=0, w=10, h=10):
    return PlacementBox(fragment_id=fid, x=x, y=y, w=w, h=h)


def _non_overlapping():
    return [
        _box(0, x=0,  y=0,  w=10, h=10),
        _box(1, x=10, y=0,  w=10, h=10),
        _box(2, x=0,  y=10, w=10, h=10),
    ]


def _overlapping():
    return [
        _box(0, x=0, y=0, w=10, h=10),
        _box(1, x=5, y=0, w=10, h=10),  # overlaps with 0
    ]


# ─── TestPlacementConfig ──────────────────────────────────────────────────────

class TestPlacementConfig:
    def test_defaults(self):
        cfg = PlacementConfig()
        assert cfg.iou_threshold == 0.0
        assert cfg.min_coverage == 0.0
        assert cfg.canvas_w == 0
        assert cfg.canvas_h == 0

    def test_negative_iou_threshold_raises(self):
        with pytest.raises(ValueError):
            PlacementConfig(iou_threshold=-0.1)

    def test_min_coverage_gt_1_raises(self):
        with pytest.raises(ValueError):
            PlacementConfig(min_coverage=1.1)

    def test_negative_min_coverage_raises(self):
        with pytest.raises(ValueError):
            PlacementConfig(min_coverage=-0.1)

    def test_negative_canvas_w_raises(self):
        with pytest.raises(ValueError):
            PlacementConfig(canvas_w=-1)

    def test_negative_canvas_h_raises(self):
        with pytest.raises(ValueError):
            PlacementConfig(canvas_h=-1)

    def test_valid_full_config(self):
        cfg = PlacementConfig(iou_threshold=0.5, min_coverage=0.8,
                              canvas_w=100, canvas_h=200)
        assert cfg.iou_threshold == 0.5
        assert cfg.canvas_w == 100


# ─── TestPlacementBox ─────────────────────────────────────────────────────────

class TestPlacementBox:
    def test_basic_construction(self):
        b = _box(0, x=5, y=3, w=20, h=15)
        assert b.fragment_id == 0
        assert b.x == 5
        assert b.y == 3

    def test_negative_id_raises(self):
        with pytest.raises(ValueError):
            _box(-1)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError):
            PlacementBox(fragment_id=0, x=-1, y=0, w=5, h=5)

    def test_negative_y_raises(self):
        with pytest.raises(ValueError):
            PlacementBox(fragment_id=0, x=0, y=-1, w=5, h=5)

    def test_w_zero_raises(self):
        with pytest.raises(ValueError):
            PlacementBox(fragment_id=0, x=0, y=0, w=0, h=5)

    def test_h_zero_raises(self):
        with pytest.raises(ValueError):
            PlacementBox(fragment_id=0, x=0, y=0, w=5, h=0)

    def test_x2_property(self):
        b = _box(0, x=3, y=0, w=7, h=5)
        assert b.x2 == 10

    def test_y2_property(self):
        b = _box(0, x=0, y=4, w=5, h=6)
        assert b.y2 == 10

    def test_area_property(self):
        b = _box(0, x=0, y=0, w=8, h=6)
        assert b.area == 48

    def test_center_property(self):
        b = _box(0, x=0, y=0, w=10, h=10)
        assert b.center == (5.0, 5.0)

    def test_center_non_square(self):
        b = PlacementBox(fragment_id=0, x=2, y=4, w=6, h=8)
        assert b.center == (5.0, 8.0)


# ─── TestCollisionReport ──────────────────────────────────────────────────────

class TestCollisionReport:
    def _make(self, cols=(), dups=(), oob=(), cov=0.0, n=3):
        return CollisionReport(
            collisions=list(cols),
            duplicates=list(dups),
            out_of_bounds=list(oob),
            coverage=cov,
            n_checked=n,
        )

    def test_is_valid_when_no_issues(self):
        r = self._make()
        assert r.is_valid is True

    def test_invalid_with_collision(self):
        r = self._make(cols=[(0, 1)])
        assert r.is_valid is False

    def test_invalid_with_duplicate(self):
        r = self._make(dups=[(0, 1)])
        assert r.is_valid is False

    def test_invalid_with_oob(self):
        r = self._make(oob=[2])
        assert r.is_valid is False

    def test_n_issues_sum(self):
        r = self._make(cols=[(0, 1)], dups=[(2, 3)], oob=[4])
        assert r.n_issues == 3

    def test_negative_n_checked_raises(self):
        with pytest.raises(ValueError):
            CollisionReport(collisions=[], duplicates=[], out_of_bounds=[],
                            coverage=0.0, n_checked=-1)

    def test_coverage_gt_1_raises(self):
        with pytest.raises(ValueError):
            CollisionReport(collisions=[], duplicates=[], out_of_bounds=[],
                            coverage=1.5, n_checked=0)


# ─── TestBoxIou ───────────────────────────────────────────────────────────────

class TestBoxIou:
    def test_identical_boxes_iou_1(self):
        b = _box(0, x=0, y=0, w=10, h=10)
        assert abs(box_iou(b, b) - 1.0) < 1e-9

    def test_no_overlap_iou_0(self):
        a = _box(0, x=0,  y=0, w=10, h=10)
        b = _box(1, x=20, y=0, w=10, h=10)
        assert box_iou(a, b) == 0.0

    def test_adjacent_no_overlap(self):
        a = _box(0, x=0,  y=0, w=10, h=10)
        b = _box(1, x=10, y=0, w=10, h=10)
        assert box_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = _box(0, x=0, y=0, w=10, h=10)
        b = _box(1, x=5, y=0, w=10, h=10)
        iou = box_iou(a, b)
        assert 0.0 < iou < 1.0

    def test_iou_symmetric(self):
        a = _box(0, x=0, y=0, w=10, h=10)
        b = _box(1, x=5, y=5, w=10, h=10)
        assert abs(box_iou(a, b) - box_iou(b, a)) < 1e-12

    def test_contained_box(self):
        outer = PlacementBox(fragment_id=0, x=0, y=0, w=20, h=20)
        inner = PlacementBox(fragment_id=1, x=5, y=5, w=5,  h=5)
        iou = box_iou(outer, inner)
        # inner area=25, outer=400, union=400, inter=25 → 25/400=0.0625
        assert abs(iou - 25.0 / 400.0) < 1e-9


# ─── TestFindCollisions ───────────────────────────────────────────────────────

class TestFindCollisions:
    def test_no_overlap_empty_result(self):
        assert find_collisions(_non_overlapping()) == []

    def test_overlap_detected(self):
        result = find_collisions(_overlapping())
        assert len(result) > 0

    def test_pair_is_sorted(self):
        result = find_collisions(_overlapping())
        for a, b in result:
            assert a <= b

    def test_empty_list(self):
        assert find_collisions([]) == []

    def test_threshold_filters(self):
        boxes = _overlapping()
        # High threshold → no collision registered
        result = find_collisions(boxes, iou_threshold=0.99)
        assert result == []

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            find_collisions(_overlapping(), iou_threshold=-0.1)

    def test_single_box_no_collision(self):
        assert find_collisions([_box(0)]) == []


# ─── TestFindDuplicatePositions ───────────────────────────────────────────────

class TestFindDuplicatePositions:
    def test_no_duplicates_empty(self):
        assert find_duplicate_positions(_non_overlapping()) == []

    def test_exact_duplicate_detected(self):
        boxes = [_box(0, x=5, y=5), _box(1, x=5, y=5)]
        result = find_duplicate_positions(boxes)
        assert len(result) == 1

    def test_different_positions_no_dup(self):
        boxes = [_box(0, x=0, y=0), _box(1, x=0, y=1)]
        assert find_duplicate_positions(boxes) == []

    def test_pair_ids_correct(self):
        boxes = [_box(3, x=5, y=5), _box(7, x=5, y=5)]
        result = find_duplicate_positions(boxes)
        assert result[0] == (3, 7)

    def test_empty_list(self):
        assert find_duplicate_positions([]) == []

    def test_different_size_same_pos_not_dup(self):
        a = PlacementBox(fragment_id=0, x=0, y=0, w=10, h=10)
        b = PlacementBox(fragment_id=1, x=0, y=0, w=20, h=10)
        assert find_duplicate_positions([a, b]) == []


# ─── TestFindOutOfBounds ──────────────────────────────────────────────────────

class TestFindOutOfBounds:
    def test_no_oob(self):
        boxes = [_box(i, x=i*10, y=0, w=10, h=10) for i in range(3)]
        result = find_out_of_bounds(boxes, canvas_w=100, canvas_h=20)
        assert result == []

    def test_oob_detected(self):
        boxes = [_box(0, x=95, y=0, w=10, h=10)]  # x2=105 > 100
        result = find_out_of_bounds(boxes, canvas_w=100, canvas_h=20)
        assert 0 in result

    def test_y_oob_detected(self):
        boxes = [_box(0, x=0, y=95, w=10, h=10)]
        result = find_out_of_bounds(boxes, canvas_w=200, canvas_h=100)
        assert 0 in result

    def test_canvas_w_0_raises(self):
        with pytest.raises(ValueError):
            find_out_of_bounds([_box(0)], canvas_w=0, canvas_h=10)

    def test_canvas_h_0_raises(self):
        with pytest.raises(ValueError):
            find_out_of_bounds([_box(0)], canvas_w=10, canvas_h=0)

    def test_exactly_at_boundary_no_oob(self):
        b = PlacementBox(fragment_id=0, x=90, y=90, w=10, h=10)
        result = find_out_of_bounds([b], canvas_w=100, canvas_h=100)
        assert result == []


# ─── TestComputeCoverage ──────────────────────────────────────────────────────

class TestComputeCoverage:
    def test_full_coverage(self):
        b = PlacementBox(fragment_id=0, x=0, y=0, w=100, h=100)
        cov = compute_coverage([b], canvas_w=100, canvas_h=100)
        assert abs(cov - 1.0) < 1e-9

    def test_zero_coverage_empty(self):
        cov = compute_coverage([], canvas_w=100, canvas_h=100)
        assert cov == 0.0

    def test_partial_coverage(self):
        b = PlacementBox(fragment_id=0, x=0, y=0, w=50, h=100)
        cov = compute_coverage([b], canvas_w=100, canvas_h=100)
        assert abs(cov - 0.5) < 1e-9

    def test_overlapping_counted_once(self):
        b1 = PlacementBox(fragment_id=0, x=0, y=0, w=60, h=100)
        b2 = PlacementBox(fragment_id=1, x=40, y=0, w=60, h=100)
        cov = compute_coverage([b1, b2], canvas_w=100, canvas_h=100)
        assert abs(cov - 1.0) < 1e-9

    def test_canvas_w_0_raises(self):
        with pytest.raises(ValueError):
            compute_coverage([_box(0)], canvas_w=0, canvas_h=10)

    def test_canvas_h_0_raises(self):
        with pytest.raises(ValueError):
            compute_coverage([_box(0)], canvas_w=10, canvas_h=0)

    def test_out_of_bounds_clipped(self):
        b = PlacementBox(fragment_id=0, x=0, y=0, w=200, h=200)
        cov = compute_coverage([b], canvas_w=100, canvas_h=100)
        assert abs(cov - 1.0) < 1e-9


# ─── TestValidatePlacements ───────────────────────────────────────────────────

class TestValidatePlacements:
    def test_valid_layout_is_valid(self):
        cfg = PlacementConfig(canvas_w=100, canvas_h=100)
        report = validate_placements(_non_overlapping(), cfg)
        assert report.is_valid

    def test_returns_collision_report(self):
        report = validate_placements(_non_overlapping())
        assert isinstance(report, CollisionReport)

    def test_n_checked_set(self):
        boxes = _non_overlapping()
        report = validate_placements(boxes)
        assert report.n_checked == len(boxes)

    def test_collision_detected(self):
        report = validate_placements(_overlapping())
        assert len(report.collisions) > 0

    def test_coverage_computed_when_canvas_set(self):
        cfg = PlacementConfig(canvas_w=100, canvas_h=100)
        boxes = [PlacementBox(fragment_id=0, x=0, y=0, w=50, h=100)]
        report = validate_placements(boxes, cfg)
        assert report.coverage > 0.0

    def test_coverage_zero_without_canvas(self):
        report = validate_placements(_non_overlapping())
        assert report.coverage == 0.0

    def test_duplicate_detected(self):
        boxes = [_box(0, x=5, y=5), _box(1, x=5, y=5)]
        report = validate_placements(boxes)
        assert len(report.duplicates) > 0

    def test_none_cfg_uses_defaults(self):
        report = validate_placements(_non_overlapping(), None)
        assert isinstance(report, CollisionReport)


# ─── TestBatchValidatePlacements ─────────────────────────────────────────────

class TestBatchValidatePlacements:
    def test_empty_batch(self):
        assert batch_validate_placements([]) == []

    def test_single_list(self):
        result = batch_validate_placements([_non_overlapping()])
        assert len(result) == 1

    def test_multiple_lists(self):
        result = batch_validate_placements([
            _non_overlapping(),
            _overlapping(),
            [_box(0)],
        ])
        assert len(result) == 3

    def test_returns_collision_reports(self):
        result = batch_validate_placements([_non_overlapping()])
        assert isinstance(result[0], CollisionReport)

    def test_custom_cfg_applied(self):
        cfg = PlacementConfig(canvas_w=100, canvas_h=100)
        result = batch_validate_placements([_non_overlapping()], cfg)
        assert result[0].coverage >= 0.0
