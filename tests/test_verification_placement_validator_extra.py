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

def _pb(fid=0, x=0, y=0, w=50, h=50):
    return PlacementBox(fragment_id=fid, x=x, y=y, w=w, h=h)


def _grid_boxes():
    """2x2 non-overlapping 50x50 on a 100x100 canvas."""
    return [_pb(0, 0, 0), _pb(1, 50, 0), _pb(2, 0, 50), _pb(3, 50, 50)]


# ─── PlacementConfig ─────────────────────────────────────────────────────────

class TestPlacementConfigExtra:
    def test_defaults(self):
        c = PlacementConfig()
        assert c.iou_threshold == 0.0
        assert c.min_coverage == 0.0
        assert c.canvas_w == 0
        assert c.canvas_h == 0

    def test_negative_iou_raises(self):
        with pytest.raises(ValueError):
            PlacementConfig(iou_threshold=-0.1)

    def test_coverage_out_of_range_raises(self):
        with pytest.raises(ValueError):
            PlacementConfig(min_coverage=1.5)

    def test_negative_canvas_w_raises(self):
        with pytest.raises(ValueError):
            PlacementConfig(canvas_w=-1)

    def test_negative_canvas_h_raises(self):
        with pytest.raises(ValueError):
            PlacementConfig(canvas_h=-1)


# ─── PlacementBox ────────────────────────────────────────────────────────────

class TestPlacementBoxExtra:
    def test_properties(self):
        b = _pb(0, 10, 20, 30, 40)
        assert b.x2 == 40
        assert b.y2 == 60
        assert b.area == 1200
        assert b.center == pytest.approx((25.0, 40.0))

    def test_negative_id_raises(self):
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


# ─── CollisionReport ────────────────────────────────────────────────────────

class TestCollisionReportExtra:
    def test_valid(self):
        r = CollisionReport(
            collisions=[], duplicates=[], out_of_bounds=[],
            coverage=0.5, n_checked=4,
        )
        assert r.is_valid is True
        assert r.n_issues == 0

    def test_invalid_collisions(self):
        r = CollisionReport(
            collisions=[(0, 1)], duplicates=[], out_of_bounds=[],
            coverage=0.5, n_checked=2,
        )
        assert r.is_valid is False
        assert r.n_issues == 1

    def test_invalid_duplicates(self):
        r = CollisionReport(
            collisions=[], duplicates=[(0, 1)], out_of_bounds=[],
            coverage=0.5, n_checked=2,
        )
        assert r.is_valid is False

    def test_invalid_oob(self):
        r = CollisionReport(
            collisions=[], duplicates=[], out_of_bounds=[0],
            coverage=0.5, n_checked=1,
        )
        assert r.is_valid is False

    def test_negative_n_checked_raises(self):
        with pytest.raises(ValueError):
            CollisionReport(
                collisions=[], duplicates=[], out_of_bounds=[],
                coverage=0.0, n_checked=-1,
            )

    def test_coverage_out_of_range_raises(self):
        with pytest.raises(ValueError):
            CollisionReport(
                collisions=[], duplicates=[], out_of_bounds=[],
                coverage=1.5, n_checked=0,
            )


# ─── box_iou ────────────────────────────────────────────────────────────────

class TestBoxIouExtra:
    def test_identical(self):
        a = _pb(0, 0, 0, 50, 50)
        assert box_iou(a, a) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = _pb(0, 0, 0, 50, 50)
        b = _pb(1, 200, 200, 50, 50)
        assert box_iou(a, b) == pytest.approx(0.0)

    def test_partial(self):
        a = _pb(0, 0, 0, 100, 100)
        b = _pb(1, 50, 50, 100, 100)
        iou = box_iou(a, b)
        assert 0.0 < iou < 1.0

    def test_touching(self):
        a = _pb(0, 0, 0, 50, 50)
        b = _pb(1, 50, 0, 50, 50)
        assert box_iou(a, b) == pytest.approx(0.0)


# ─── find_collisions ────────────────────────────────────────────────────────

class TestFindCollisionsExtra:
    def test_no_collisions(self):
        assert find_collisions(_grid_boxes()) == []

    def test_with_collision(self):
        boxes = [_pb(0, 0, 0, 100, 100), _pb(1, 50, 50, 100, 100)]
        assert len(find_collisions(boxes)) == 1

    def test_empty(self):
        assert find_collisions([]) == []

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            find_collisions([], iou_threshold=-0.1)

    def test_high_threshold(self):
        boxes = [_pb(0, 0, 0, 100, 100), _pb(1, 50, 50, 100, 100)]
        assert find_collisions(boxes, iou_threshold=0.9) == []


# ─── find_duplicate_positions ────────────────────────────────────────────────

class TestFindDuplicatePositionsExtra:
    def test_no_duplicates(self):
        assert find_duplicate_positions(_grid_boxes()) == []

    def test_with_duplicate(self):
        boxes = [_pb(0, 0, 0, 50, 50), _pb(1, 0, 0, 50, 50)]
        dups = find_duplicate_positions(boxes)
        assert len(dups) == 1
        assert dups[0] == (0, 1)

    def test_empty(self):
        assert find_duplicate_positions([]) == []


# ─── find_out_of_bounds ──────────────────────────────────────────────────────

class TestFindOutOfBoundsExtra:
    def test_inside(self):
        assert find_out_of_bounds(_grid_boxes(), 100, 100) == []

    def test_outside(self):
        boxes = [_pb(0, 80, 80, 50, 50)]
        oob = find_out_of_bounds(boxes, 100, 100)
        assert oob == [0]

    def test_empty(self):
        assert find_out_of_bounds([], 100, 100) == []

    def test_invalid_canvas_w_raises(self):
        with pytest.raises(ValueError):
            find_out_of_bounds([], 0, 100)

    def test_invalid_canvas_h_raises(self):
        with pytest.raises(ValueError):
            find_out_of_bounds([], 100, 0)


# ─── compute_coverage ────────────────────────────────────────────────────────

class TestComputeCoverageExtra:
    def test_full(self):
        cov = compute_coverage(_grid_boxes(), 100, 100)
        assert cov == pytest.approx(1.0)

    def test_partial(self):
        cov = compute_coverage([_pb(0, 0, 0, 50, 50)], 100, 100)
        assert cov == pytest.approx(0.25)

    def test_empty(self):
        cov = compute_coverage([], 100, 100)
        assert cov == pytest.approx(0.0)

    def test_invalid_canvas_raises(self):
        with pytest.raises(ValueError):
            compute_coverage([], 0, 100)


# ─── validate_placements ────────────────────────────────────────────────────

class TestValidatePlacementsExtra:
    def test_clean(self):
        r = validate_placements(_grid_boxes())
        assert r.is_valid is True
        assert r.n_checked == 4

    def test_with_collision(self):
        boxes = [_pb(0, 0, 0, 100, 100), _pb(1, 50, 50, 100, 100)]
        r = validate_placements(boxes)
        assert len(r.collisions) > 0

    def test_with_canvas(self):
        cfg = PlacementConfig(canvas_w=100, canvas_h=100)
        r = validate_placements(_grid_boxes(), cfg)
        assert r.coverage == pytest.approx(1.0)
        assert r.out_of_bounds == []

    def test_out_of_bounds(self):
        cfg = PlacementConfig(canvas_w=80, canvas_h=80)
        r = validate_placements(_grid_boxes(), cfg)
        assert len(r.out_of_bounds) > 0

    def test_empty(self):
        r = validate_placements([])
        assert r.n_checked == 0
        assert r.is_valid is True


# ─── batch_validate_placements ──────────────────────────────────────────────

class TestBatchValidatePlacementsExtra:
    def test_empty(self):
        assert batch_validate_placements([]) == []

    def test_multiple(self):
        results = batch_validate_placements([_grid_boxes(), [_pb()]])
        assert len(results) == 2
        assert all(isinstance(r, CollisionReport) for r in results)

    def test_with_config(self):
        cfg = PlacementConfig(canvas_w=100, canvas_h=100)
        results = batch_validate_placements([_grid_boxes()], cfg)
        assert results[0].coverage == pytest.approx(1.0)
