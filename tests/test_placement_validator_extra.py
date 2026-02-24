"""Extra tests for puzzle_reconstruction/verification/placement_validator.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.verification.placement_validator import (
    PlacementConfig,
    PlacementBox,
    CollisionReport,
    box_iou,
    find_collisions,
    find_duplicate_positions,
    find_out_of_bounds,
    compute_coverage,
    validate_placements,
    batch_validate_placements,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _box(fid=0, x=0, y=0, w=10, h=10) -> PlacementBox:
    return PlacementBox(fragment_id=fid, x=x, y=y, w=w, h=h)


def _report(collisions=None, duplicates=None, oob=None,
            coverage=0.0, n_checked=0) -> CollisionReport:
    return CollisionReport(
        collisions=collisions or [],
        duplicates=duplicates or [],
        out_of_bounds=oob or [],
        coverage=coverage,
        n_checked=n_checked,
    )


# ─── PlacementConfig ──────────────────────────────────────────────────────────

class TestPlacementConfigExtra:
    def test_default_iou_threshold(self):
        assert PlacementConfig().iou_threshold == pytest.approx(0.0)

    def test_default_min_coverage(self):
        assert PlacementConfig().min_coverage == pytest.approx(0.0)

    def test_default_canvas_w(self):
        assert PlacementConfig().canvas_w == 0

    def test_default_canvas_h(self):
        assert PlacementConfig().canvas_h == 0

    def test_negative_iou_raises(self):
        with pytest.raises(ValueError):
            PlacementConfig(iou_threshold=-0.1)

    def test_negative_min_coverage_raises(self):
        with pytest.raises(ValueError):
            PlacementConfig(min_coverage=-0.1)

    def test_min_coverage_gt_one_raises(self):
        with pytest.raises(ValueError):
            PlacementConfig(min_coverage=1.1)

    def test_negative_canvas_w_raises(self):
        with pytest.raises(ValueError):
            PlacementConfig(canvas_w=-1)

    def test_negative_canvas_h_raises(self):
        with pytest.raises(ValueError):
            PlacementConfig(canvas_h=-1)

    def test_custom_values(self):
        cfg = PlacementConfig(iou_threshold=0.5, min_coverage=0.3,
                              canvas_w=100, canvas_h=100)
        assert cfg.iou_threshold == pytest.approx(0.5)
        assert cfg.canvas_w == 100


# ─── PlacementBox ─────────────────────────────────────────────────────────────

class TestPlacementBoxExtra:
    def test_fragment_id_stored(self):
        b = _box(fid=5)
        assert b.fragment_id == 5

    def test_x_y_stored(self):
        b = _box(x=10, y=20)
        assert b.x == 10 and b.y == 20

    def test_w_h_stored(self):
        b = _box(w=30, h=40)
        assert b.w == 30 and b.h == 40

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            PlacementBox(fragment_id=-1, x=0, y=0, w=10, h=10)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError):
            PlacementBox(fragment_id=0, x=-1, y=0, w=10, h=10)

    def test_negative_y_raises(self):
        with pytest.raises(ValueError):
            PlacementBox(fragment_id=0, x=0, y=-1, w=10, h=10)

    def test_zero_w_raises(self):
        with pytest.raises(ValueError):
            PlacementBox(fragment_id=0, x=0, y=0, w=0, h=10)

    def test_zero_h_raises(self):
        with pytest.raises(ValueError):
            PlacementBox(fragment_id=0, x=0, y=0, w=10, h=0)

    def test_x2_property(self):
        b = _box(x=5, w=10)
        assert b.x2 == 15

    def test_y2_property(self):
        b = _box(y=3, h=7)
        assert b.y2 == 10

    def test_area_property(self):
        b = _box(w=4, h=5)
        assert b.area == 20

    def test_center_property(self):
        b = _box(x=0, y=0, w=10, h=10)
        cx, cy = b.center
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(5.0)

    def test_center_offset(self):
        b = _box(x=10, y=20, w=4, h=6)
        cx, cy = b.center
        assert cx == pytest.approx(12.0)
        assert cy == pytest.approx(23.0)


# ─── CollisionReport ──────────────────────────────────────────────────────────

class TestCollisionReportExtra:
    def test_is_valid_no_issues(self):
        r = _report()
        assert r.is_valid is True

    def test_is_valid_with_collision(self):
        r = _report(collisions=[(0, 1)])
        assert r.is_valid is False

    def test_is_valid_with_duplicate(self):
        r = _report(duplicates=[(0, 1)])
        assert r.is_valid is False

    def test_is_valid_with_oob(self):
        r = _report(oob=[0])
        assert r.is_valid is False

    def test_n_issues_zero(self):
        assert _report().n_issues == 0

    def test_n_issues_sum(self):
        r = _report(collisions=[(0, 1)], duplicates=[(2, 3)], oob=[4])
        assert r.n_issues == 3

    def test_negative_n_checked_raises(self):
        with pytest.raises(ValueError):
            CollisionReport(collisions=[], duplicates=[], out_of_bounds=[],
                            coverage=0.0, n_checked=-1)

    def test_coverage_gt_one_raises(self):
        with pytest.raises(ValueError):
            CollisionReport(collisions=[], duplicates=[], out_of_bounds=[],
                            coverage=1.5, n_checked=0)

    def test_coverage_stored(self):
        r = _report(coverage=0.75)
        assert r.coverage == pytest.approx(0.75)


# ─── box_iou ──────────────────────────────────────────────────────────────────

class TestBoxIouExtra:
    def test_identical_boxes_iou_one(self):
        b = _box(x=0, y=0, w=10, h=10)
        assert box_iou(b, b) == pytest.approx(1.0)

    def test_no_overlap_iou_zero(self):
        a = _box(fid=0, x=0, y=0, w=10, h=10)
        b = _box(fid=1, x=20, y=20, w=10, h=10)
        assert box_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = _box(fid=0, x=0, y=0, w=10, h=10)
        b = _box(fid=1, x=5, y=0, w=10, h=10)
        iou = box_iou(a, b)
        assert 0.0 < iou < 1.0

    def test_touching_but_no_overlap(self):
        a = _box(fid=0, x=0, y=0, w=10, h=10)
        b = _box(fid=1, x=10, y=0, w=10, h=10)
        assert box_iou(a, b) == pytest.approx(0.0)

    def test_contained_box(self):
        a = _box(fid=0, x=0, y=0, w=20, h=20)
        b = _box(fid=1, x=5, y=5, w=5, h=5)
        iou = box_iou(a, b)
        # inter=25, union=400+25-25=400
        assert iou == pytest.approx(25.0 / 400.0)

    def test_iou_symmetric(self):
        a = _box(fid=0, x=0, y=0, w=10, h=10)
        b = _box(fid=1, x=5, y=5, w=10, h=10)
        assert box_iou(a, b) == pytest.approx(box_iou(b, a))


# ─── find_collisions ──────────────────────────────────────────────────────────

class TestFindCollisionsExtra:
    def test_no_collisions(self):
        boxes = [_box(0, 0, 0, 10, 10), _box(1, 20, 20, 10, 10)]
        assert find_collisions(boxes) == []

    def test_collision_detected(self):
        boxes = [_box(0, 0, 0, 10, 10), _box(1, 5, 5, 10, 10)]
        result = find_collisions(boxes, iou_threshold=0.0)
        assert len(result) == 1

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            find_collisions([_box()], iou_threshold=-0.1)

    def test_high_threshold_no_collisions(self):
        boxes = [_box(0, 0, 0, 10, 10), _box(1, 5, 5, 10, 10)]
        result = find_collisions(boxes, iou_threshold=0.99)
        assert result == []

    def test_empty_list(self):
        assert find_collisions([]) == []

    def test_single_box(self):
        assert find_collisions([_box()]) == []

    def test_result_fragment_ids_ordered(self):
        boxes = [_box(5, 0, 0, 10, 10), _box(2, 5, 5, 10, 10)]
        result = find_collisions(boxes, iou_threshold=0.0)
        assert result[0] == (2, 5)


# ─── find_duplicate_positions ─────────────────────────────────────────────────

class TestFindDuplicatePositionsExtra:
    def test_no_duplicates(self):
        boxes = [_box(0, 0, 0), _box(1, 20, 20)]
        assert find_duplicate_positions(boxes) == []

    def test_duplicate_detected(self):
        boxes = [_box(0, 5, 5), _box(1, 5, 5)]
        result = find_duplicate_positions(boxes)
        assert len(result) == 1

    def test_duplicate_pair_ids(self):
        boxes = [_box(3, 5, 5), _box(7, 5, 5)]
        result = find_duplicate_positions(boxes)
        assert (3, 7) in result

    def test_empty_list(self):
        assert find_duplicate_positions([]) == []

    def test_different_size_not_duplicate(self):
        boxes = [_box(0, 0, 0, 10, 10), _box(1, 0, 0, 10, 20)]
        assert find_duplicate_positions(boxes) == []


# ─── find_out_of_bounds ───────────────────────────────────────────────────────

class TestFindOutOfBoundsExtra:
    def test_all_in_bounds(self):
        boxes = [_box(0, 0, 0, 10, 10)]
        assert find_out_of_bounds(boxes, 100, 100) == []

    def test_out_of_bounds_x(self):
        boxes = [_box(0, 95, 0, 10, 10)]
        result = find_out_of_bounds(boxes, 100, 100)
        assert 0 in result

    def test_out_of_bounds_y(self):
        boxes = [_box(0, 0, 95, 10, 10)]
        result = find_out_of_bounds(boxes, 100, 100)
        assert 0 in result

    def test_canvas_w_zero_raises(self):
        with pytest.raises(ValueError):
            find_out_of_bounds([_box()], 0, 10)

    def test_canvas_h_zero_raises(self):
        with pytest.raises(ValueError):
            find_out_of_bounds([_box()], 10, 0)

    def test_exactly_fits(self):
        boxes = [_box(0, 0, 0, 10, 10)]
        assert find_out_of_bounds(boxes, 10, 10) == []

    def test_empty_boxes(self):
        assert find_out_of_bounds([], 100, 100) == []


# ─── compute_coverage ─────────────────────────────────────────────────────────

class TestComputeCoverageExtra:
    def test_zero_boxes_zero_coverage(self):
        assert compute_coverage([], 100, 100) == pytest.approx(0.0)

    def test_full_coverage(self):
        boxes = [_box(0, 0, 0, 100, 100)]
        assert compute_coverage(boxes, 100, 100) == pytest.approx(1.0)

    def test_partial_coverage(self):
        boxes = [_box(0, 0, 0, 50, 100)]
        cov = compute_coverage(boxes, 100, 100)
        assert cov == pytest.approx(0.5)

    def test_canvas_w_zero_raises(self):
        with pytest.raises(ValueError):
            compute_coverage([], 0, 100)

    def test_canvas_h_zero_raises(self):
        with pytest.raises(ValueError):
            compute_coverage([], 100, 0)

    def test_overlapping_boxes_not_double_counted(self):
        boxes = [_box(0, 0, 0, 10, 10), _box(1, 0, 0, 10, 10)]
        cov = compute_coverage(boxes, 20, 20)
        assert cov == pytest.approx(0.25)  # 100 / 400

    def test_coverage_in_range(self):
        boxes = [_box(0, 0, 0, 30, 30)]
        cov = compute_coverage(boxes, 100, 100)
        assert 0.0 <= cov <= 1.0


# ─── validate_placements ──────────────────────────────────────────────────────

class TestValidatePlacementsExtra:
    def test_returns_collision_report(self):
        r = validate_placements([_box()])
        assert isinstance(r, CollisionReport)

    def test_n_checked_matches_boxes(self):
        boxes = [_box(0), _box(1, 20, 20)]
        r = validate_placements(boxes)
        assert r.n_checked == 2

    def test_no_issues_with_separated_boxes(self):
        boxes = [_box(0, 0, 0), _box(1, 50, 50)]
        r = validate_placements(boxes)
        assert r.is_valid is True

    def test_collision_reported(self):
        boxes = [_box(0, 0, 0, 10, 10), _box(1, 5, 5, 10, 10)]
        cfg = PlacementConfig(iou_threshold=0.0)
        r = validate_placements(boxes, cfg)
        assert len(r.collisions) > 0

    def test_coverage_with_canvas(self):
        boxes = [_box(0, 0, 0, 50, 100)]
        cfg = PlacementConfig(canvas_w=100, canvas_h=100)
        r = validate_placements(boxes, cfg)
        assert r.coverage == pytest.approx(0.5)

    def test_no_canvas_coverage_zero(self):
        boxes = [_box(0, 0, 0, 50, 50)]
        cfg = PlacementConfig(canvas_w=0, canvas_h=0)
        r = validate_placements(boxes, cfg)
        assert r.coverage == pytest.approx(0.0)

    def test_none_cfg_uses_defaults(self):
        r = validate_placements([_box()], cfg=None)
        assert isinstance(r, CollisionReport)

    def test_empty_boxes(self):
        r = validate_placements([])
        assert r.n_checked == 0
        assert r.is_valid is True


# ─── batch_validate_placements ────────────────────────────────────────────────

class TestBatchValidatePlacementsExtra:
    def test_returns_list(self):
        result = batch_validate_placements([[_box()]])
        assert isinstance(result, list)

    def test_list_length_matches(self):
        result = batch_validate_placements([[_box(0)], [_box(1)]])
        assert len(result) == 2

    def test_empty_batch(self):
        result = batch_validate_placements([])
        assert result == []

    def test_each_element_is_report(self):
        result = batch_validate_placements([[_box()], []])
        for r in result:
            assert isinstance(r, CollisionReport)

    def test_cfg_passed_through(self):
        cfg = PlacementConfig(canvas_w=100, canvas_h=100)
        result = batch_validate_placements([[_box(0, 0, 0, 50, 50)]], cfg)
        assert result[0].coverage == pytest.approx(0.25)
