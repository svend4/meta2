"""Extra tests for puzzle_reconstruction/assembly/overlap_resolver.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.assembly.overlap_resolver import (
    ResolveConfig,
    BBox,
    Overlap,
    ResolveResult,
    compute_overlap,
    detect_overlaps,
    resolve_overlaps,
    compute_total_overlap,
    overlap_ratio,
)


def _box(fid, x=0.0, y=0.0, w=10.0, h=10.0):
    return BBox(fragment_id=fid, x=x, y=y, w=w, h=h)


# ─── ResolveConfig (extra) ───────────────────────────────────────────────────

class TestResolveConfigExtra:
    def test_large_max_iter(self):
        cfg = ResolveConfig(max_iter=10000)
        assert cfg.max_iter == 10000

    def test_large_gap(self):
        cfg = ResolveConfig(gap=100.0)
        assert cfg.gap == pytest.approx(100.0)

    def test_step_scale_one(self):
        cfg = ResolveConfig(step_scale=1.0)
        assert cfg.step_scale == pytest.approx(1.0)

    def test_frozen_ids_multiple(self):
        cfg = ResolveConfig(frozen_ids=[0, 1, 2])
        assert cfg.frozen_ids == [0, 1, 2]

    def test_independent_instances(self):
        c1 = ResolveConfig(max_iter=5)
        c2 = ResolveConfig(max_iter=50)
        assert c1.max_iter != c2.max_iter


# ─── BBox (extra) ────────────────────────────────────────────────────────────

class TestBBoxExtra:
    def test_large_coords(self):
        b = _box(0, 1000.0, 2000.0, 500.0, 300.0)
        assert b.x2 == pytest.approx(1500.0)
        assert b.y2 == pytest.approx(2300.0)

    def test_area_float(self):
        b = _box(0, 0.0, 0.0, 3.5, 4.0)
        assert b.area == pytest.approx(14.0)

    def test_cx_cy_correct(self):
        b = _box(0, 10.0, 20.0, 6.0, 8.0)
        assert b.cx == pytest.approx(13.0)
        assert b.cy == pytest.approx(24.0)

    def test_translate_negative(self):
        b = _box(0, 10.0, 20.0)
        b2 = b.translate(-5.0, -10.0)
        assert b2.x == pytest.approx(5.0)
        assert b2.y == pytest.approx(10.0)

    def test_translate_returns_new_bbox(self):
        b = _box(0, 1.0, 2.0)
        b2 = b.translate(3.0, 4.0)
        assert b is not b2

    def test_fragment_id_preserved_on_translate(self):
        b = _box(99, 0.0, 0.0)
        b2 = b.translate(5.0, 5.0)
        assert b2.fragment_id == 99


# ─── Overlap (extra) ─────────────────────────────────────────────────────────

class TestOverlapExtra:
    def test_zero_area_no_overlap(self):
        ov = Overlap(id_a=0, id_b=1, area=0.0, dx=0.0, dy=0.0)
        assert ov.has_overlap is False

    def test_small_area_has_overlap(self):
        ov = Overlap(id_a=0, id_b=1, area=0.001, dx=0.0, dy=0.0)
        assert ov.has_overlap is True

    def test_pair_key_always_sorted(self):
        ov = Overlap(id_a=99, id_b=1, area=0.0, dx=0.0, dy=0.0)
        assert ov.pair_key == (1, 99)

    def test_dx_dy_stored(self):
        ov = Overlap(id_a=0, id_b=1, area=10.0, dx=3.5, dy=-2.0)
        assert ov.dx == pytest.approx(3.5)
        assert ov.dy == pytest.approx(-2.0)


# ─── ResolveResult (extra) ───────────────────────────────────────────────────

class TestResolveResultExtra:
    def test_empty_boxes_empty_ids(self):
        r = ResolveResult(boxes={}, n_iter=0, resolved=True)
        assert r.fragment_ids == []

    def test_boxes_stored(self):
        boxes = {0: _box(0), 1: _box(1, 20.0, 0.0)}
        r = ResolveResult(boxes=boxes, n_iter=1, resolved=True)
        assert 0 in r.boxes
        assert 1 in r.boxes

    def test_history_default_empty(self):
        r = ResolveResult(boxes={}, n_iter=0, resolved=True)
        assert r.final_n_overlaps == 0

    def test_n_iter_stored(self):
        r = ResolveResult(boxes={}, n_iter=5, resolved=False)
        assert r.n_iter == 5
        assert r.resolved is False


# ─── compute_overlap (extra) ─────────────────────────────────────────────────

class TestComputeOverlapExtra:
    def test_identical_boxes(self):
        a = _box(0, 0.0, 0.0, 10.0, 10.0)
        b = _box(1, 0.0, 0.0, 10.0, 10.0)
        ov = compute_overlap(a, b)
        assert ov.has_overlap is True
        assert ov.area == pytest.approx(100.0)

    def test_no_y_overlap(self):
        a = _box(0, 0.0, 0.0, 10.0, 5.0)
        b = _box(1, 0.0, 5.0, 10.0, 5.0)
        ov = compute_overlap(a, b)
        assert ov.area == pytest.approx(0.0)

    def test_symmetric(self):
        a = _box(0, 0.0, 0.0, 10.0, 10.0)
        b = _box(1, 5.0, 3.0, 10.0, 10.0)
        ov1 = compute_overlap(a, b)
        ov2 = compute_overlap(b, a)
        assert ov1.area == pytest.approx(ov2.area)

    def test_gap_parameter(self):
        a = _box(0, 0.0, 0.0, 5.0, 5.0)
        b = _box(1, 7.0, 0.0, 5.0, 5.0)
        ov_no = compute_overlap(a, b, gap=0.0)
        ov_yes = compute_overlap(a, b, gap=3.0)
        assert ov_no.has_overlap is False
        assert ov_yes.has_overlap is True

    def test_contained_box(self):
        outer = _box(0, 0.0, 0.0, 20.0, 20.0)
        inner = _box(1, 5.0, 5.0, 5.0, 5.0)
        ov = compute_overlap(outer, inner)
        assert ov.area == pytest.approx(25.0)


# ─── detect_overlaps (extra) ─────────────────────────────────────────────────

class TestDetectOverlapsExtra:
    def test_three_boxes_two_overlaps(self):
        boxes = {
            0: _box(0, 0.0, 0.0, 10.0, 10.0),
            1: _box(1, 8.0, 0.0, 10.0, 10.0),
            2: _box(2, 50.0, 0.0, 10.0, 10.0),
        }
        ov = detect_overlaps(boxes)
        assert len(ov) == 1

    def test_all_overlap_types(self):
        boxes = {i: _box(i, 0.0, 0.0) for i in range(3)}
        ov = detect_overlaps(boxes)
        for o in ov:
            assert isinstance(o, Overlap)
            assert o.has_overlap is True

    def test_gap_zero_touching_no_overlap(self):
        boxes = {
            0: _box(0, 0.0, 0.0, 10.0, 10.0),
            1: _box(1, 10.0, 0.0, 10.0, 10.0),
        }
        assert detect_overlaps(boxes, gap=0.0) == []

    def test_gap_positive_touching_overlap(self):
        boxes = {
            0: _box(0, 0.0, 0.0, 10.0, 10.0),
            1: _box(1, 10.0, 0.0, 10.0, 10.0),
        }
        ov = detect_overlaps(boxes, gap=1.0)
        assert len(ov) >= 1


# ─── resolve_overlaps (extra) ────────────────────────────────────────────────

class TestResolveOverlapsExtra:
    def test_three_overlapping_resolved(self):
        boxes = {i: _box(i, i * 3.0, 0.0) for i in range(3)}
        r = resolve_overlaps(boxes)
        assert isinstance(r, ResolveResult)
        assert len(r.boxes) == 3

    def test_frozen_id_stays(self):
        boxes = {
            0: _box(0, 0.0, 0.0),
            1: _box(1, 5.0, 0.0),
        }
        cfg = ResolveConfig(max_iter=10, frozen_ids=[0])
        r = resolve_overlaps(boxes, cfg)
        assert r.boxes[0].x == pytest.approx(0.0)
        assert r.boxes[0].y == pytest.approx(0.0)

    def test_does_not_mutate(self):
        boxes = {0: _box(0, 0.0, 0.0), 1: _box(1, 5.0, 0.0)}
        orig_x1 = boxes[1].x
        resolve_overlaps(boxes)
        assert boxes[1].x == pytest.approx(orig_x1)

    def test_all_ids_present_after_resolve(self):
        boxes = {i: _box(i, i * 2.0, 0.0) for i in range(4)}
        r = resolve_overlaps(boxes)
        assert set(r.boxes.keys()) == {0, 1, 2, 3}


# ─── compute_total_overlap / overlap_ratio (extra) ──────────────────────────

class TestTotalOverlapRatioExtra:
    def test_total_nonneg(self):
        boxes = {0: _box(0, 0.0, 0.0), 1: _box(1, 5.0, 0.0)}
        assert compute_total_overlap(boxes) >= 0.0

    def test_ratio_nonneg(self):
        boxes = {0: _box(0, 0.0, 0.0), 1: _box(1, 5.0, 0.0)}
        assert overlap_ratio(boxes) >= 0.0

    def test_ratio_no_overlap_zero(self):
        boxes = {0: _box(0, 0.0, 0.0), 1: _box(1, 100.0, 0.0)}
        assert overlap_ratio(boxes) == pytest.approx(0.0)

    def test_total_overlap_two_identical(self):
        boxes = {0: _box(0, 0.0, 0.0), 1: _box(1, 0.0, 0.0)}
        total = compute_total_overlap(boxes)
        assert total == pytest.approx(100.0)

    def test_ratio_le_one(self):
        boxes = {i: _box(i, 0.0, 0.0) for i in range(5)}
        assert overlap_ratio(boxes) <= 1.0
