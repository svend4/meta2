"""Tests for puzzle_reconstruction.verification.placement_validator"""
import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/meta2')

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


def make_box(fid, x, y, w=50, h=50):
    return PlacementBox(fragment_id=fid, x=x, y=y, w=w, h=h)


# ─── PlacementConfig ──────────────────────────────────────────────────────────

def test_placement_config_defaults():
    cfg = PlacementConfig()
    assert cfg.iou_threshold == 0.0
    assert cfg.min_coverage == 0.0
    assert cfg.canvas_w == 0
    assert cfg.canvas_h == 0


def test_placement_config_invalid_iou():
    with pytest.raises(ValueError):
        PlacementConfig(iou_threshold=-0.1)


def test_placement_config_invalid_coverage():
    with pytest.raises(ValueError):
        PlacementConfig(min_coverage=1.5)


def test_placement_config_invalid_canvas():
    with pytest.raises(ValueError):
        PlacementConfig(canvas_w=-1)


# ─── PlacementBox ─────────────────────────────────────────────────────────────

def test_placement_box_x2_y2():
    b = make_box(0, 10, 20, w=30, h=40)
    assert b.x2 == 40
    assert b.y2 == 60


def test_placement_box_area():
    b = make_box(0, 0, 0, w=30, h=40)
    assert b.area == 1200


def test_placement_box_center():
    b = make_box(0, 0, 0, w=100, h=100)
    cx, cy = b.center
    assert cx == 50.0
    assert cy == 50.0


def test_placement_box_invalid_fid():
    with pytest.raises(ValueError):
        PlacementBox(fragment_id=-1, x=0, y=0, w=10, h=10)


def test_placement_box_invalid_x():
    with pytest.raises(ValueError):
        PlacementBox(fragment_id=0, x=-1, y=0, w=10, h=10)


def test_placement_box_invalid_w():
    with pytest.raises(ValueError):
        PlacementBox(fragment_id=0, x=0, y=0, w=0, h=10)


def test_placement_box_invalid_h():
    with pytest.raises(ValueError):
        PlacementBox(fragment_id=0, x=0, y=0, w=10, h=0)


# ─── CollisionReport ──────────────────────────────────────────────────────────

def test_collision_report_is_valid():
    report = CollisionReport(collisions=[], duplicates=[], out_of_bounds=[], coverage=0.5, n_checked=3)
    assert report.is_valid


def test_collision_report_not_valid_collision():
    report = CollisionReport(collisions=[(0, 1)], duplicates=[], out_of_bounds=[], coverage=0.0, n_checked=2)
    assert not report.is_valid


def test_collision_report_n_issues():
    report = CollisionReport(
        collisions=[(0, 1)],
        duplicates=[(2, 3)],
        out_of_bounds=[4],
        coverage=0.0,
        n_checked=5
    )
    assert report.n_issues == 3


def test_collision_report_invalid_n_checked():
    with pytest.raises(ValueError):
        CollisionReport(collisions=[], duplicates=[], out_of_bounds=[], coverage=0.0, n_checked=-1)


# ─── box_iou ──────────────────────────────────────────────────────────────────

def test_box_iou_no_overlap():
    a = make_box(0, 0, 0, w=50, h=50)
    b = make_box(1, 100, 100, w=50, h=50)
    assert box_iou(a, b) == pytest.approx(0.0)


def test_box_iou_full_overlap():
    a = make_box(0, 0, 0, w=50, h=50)
    b = make_box(1, 0, 0, w=50, h=50)
    assert box_iou(a, b) == pytest.approx(1.0)


def test_box_iou_partial():
    a = make_box(0, 0, 0, w=100, h=100)
    b = make_box(1, 50, 0, w=100, h=100)
    iou = box_iou(a, b)
    # inter=50*100=5000, union=100*100+100*100-5000=15000
    assert iou == pytest.approx(5000.0 / 15000.0, abs=0.01)


def test_box_iou_range():
    a = make_box(0, 0, 0)
    b = make_box(1, 25, 25)
    assert 0.0 <= box_iou(a, b) <= 1.0


# ─── find_collisions ──────────────────────────────────────────────────────────

def test_find_collisions_none():
    boxes = [make_box(i, i * 100, 0) for i in range(3)]
    collisions = find_collisions(boxes, iou_threshold=0.0)
    assert collisions == []


def test_find_collisions_found():
    boxes = [make_box(0, 0, 0, w=100, h=100), make_box(1, 50, 50, w=100, h=100)]
    collisions = find_collisions(boxes, iou_threshold=0.0)
    assert len(collisions) > 0


def test_find_collisions_invalid_threshold():
    with pytest.raises(ValueError):
        find_collisions([], iou_threshold=-0.1)


def test_find_collisions_sorted_ids():
    boxes = [make_box(1, 0, 0, w=100, h=100), make_box(0, 50, 50, w=100, h=100)]
    collisions = find_collisions(boxes, iou_threshold=0.0)
    for a_id, b_id in collisions:
        assert a_id < b_id


# ─── find_duplicate_positions ─────────────────────────────────────────────────

def test_find_duplicate_positions_none():
    boxes = [make_box(i, i * 100, 0) for i in range(3)]
    dups = find_duplicate_positions(boxes)
    assert dups == []


def test_find_duplicate_positions_found():
    boxes = [make_box(0, 10, 10, w=50, h=50), make_box(1, 10, 10, w=50, h=50)]
    dups = find_duplicate_positions(boxes)
    assert len(dups) == 1


def test_find_duplicate_positions_different_size():
    boxes = [make_box(0, 10, 10, w=50, h=50), make_box(1, 10, 10, w=60, h=50)]
    dups = find_duplicate_positions(boxes)
    assert dups == []


# ─── find_out_of_bounds ───────────────────────────────────────────────────────

def test_find_out_of_bounds_inside():
    boxes = [make_box(0, 0, 0, w=50, h=50)]
    oob = find_out_of_bounds(boxes, canvas_w=100, canvas_h=100)
    assert oob == []


def test_find_out_of_bounds_outside():
    boxes = [make_box(0, 80, 0, w=50, h=50)]
    oob = find_out_of_bounds(boxes, canvas_w=100, canvas_h=100)
    assert 0 in oob


def test_find_out_of_bounds_invalid_canvas():
    with pytest.raises(ValueError):
        find_out_of_bounds([], canvas_w=0, canvas_h=100)


# ─── compute_coverage ─────────────────────────────────────────────────────────

def test_compute_coverage_full():
    boxes = [make_box(0, 0, 0, w=100, h=100)]
    cov = compute_coverage(boxes, canvas_w=100, canvas_h=100)
    assert cov == pytest.approx(1.0)


def test_compute_coverage_half():
    boxes = [make_box(0, 0, 0, w=50, h=100)]
    cov = compute_coverage(boxes, canvas_w=100, canvas_h=100)
    assert cov == pytest.approx(0.5)


def test_compute_coverage_empty():
    cov = compute_coverage([], canvas_w=100, canvas_h=100)
    assert cov == pytest.approx(0.0)


def test_compute_coverage_invalid_canvas():
    with pytest.raises(ValueError):
        compute_coverage([], canvas_w=0, canvas_h=100)


# ─── validate_placements ──────────────────────────────────────────────────────

def test_validate_placements_clean():
    boxes = [make_box(i, i * 100, 0) for i in range(3)]
    cfg = PlacementConfig(canvas_w=400, canvas_h=100)
    report = validate_placements(boxes, cfg)
    assert report.is_valid


def test_validate_placements_with_collision():
    boxes = [make_box(0, 0, 0, w=100, h=100), make_box(1, 50, 50, w=100, h=100)]
    cfg = PlacementConfig(iou_threshold=0.0)
    report = validate_placements(boxes, cfg)
    assert len(report.collisions) > 0


def test_validate_placements_n_checked():
    boxes = [make_box(i, i * 100, 0) for i in range(5)]
    report = validate_placements(boxes)
    assert report.n_checked == 5


def test_validate_placements_default_config():
    boxes = [make_box(0, 0, 0)]
    report = validate_placements(boxes)
    assert isinstance(report, CollisionReport)


# ─── batch_validate_placements ────────────────────────────────────────────────

def test_batch_validate_placements_basic():
    box_lists = [
        [make_box(i, i * 100, 0) for i in range(2)],
        [make_box(i, i * 100, 0) for i in range(3)],
    ]
    reports = batch_validate_placements(box_lists)
    assert len(reports) == 2
    assert reports[0].n_checked == 2
    assert reports[1].n_checked == 3


def test_batch_validate_placements_empty():
    reports = batch_validate_placements([])
    assert reports == []
