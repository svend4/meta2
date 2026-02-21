"""Тесты для puzzle_reconstruction.assembly.overlap_resolver."""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _box(fid: int, x: float = 0.0, y: float = 0.0,
         w: float = 10.0, h: float = 10.0) -> BBox:
    return BBox(fragment_id=fid, x=x, y=y, w=w, h=h)


def _non_overlapping():
    """Два неперекрывающихся прямоугольника."""
    return {
        0: _box(0, 0.0, 0.0, 10.0, 10.0),
        1: _box(1, 20.0, 0.0, 10.0, 10.0),
    }


def _overlapping():
    """Два перекрывающихся прямоугольника (перекрытие 5×10)."""
    return {
        0: _box(0, 0.0, 0.0, 10.0, 10.0),
        1: _box(1, 5.0, 0.0, 10.0, 10.0),
    }


def _grid_4():
    """Четыре прямоугольника в сетке 2×2 без перекрытий."""
    return {
        0: _box(0, 0.0, 0.0),
        1: _box(1, 10.0, 0.0),
        2: _box(2, 0.0, 10.0),
        3: _box(3, 10.0, 10.0),
    }


# ─── TestResolveConfig ────────────────────────────────────────────────────────

class TestResolveConfig:
    def test_defaults(self):
        cfg = ResolveConfig()
        assert cfg.max_iter == 10
        assert cfg.gap == pytest.approx(1.0)
        assert cfg.step_scale == pytest.approx(0.5)
        assert cfg.frozen_ids == []

    def test_valid_custom(self):
        cfg = ResolveConfig(max_iter=5, gap=2.0, step_scale=0.8,
                            frozen_ids=[0])
        assert cfg.max_iter == 5
        assert cfg.frozen_ids == [0]

    def test_max_iter_one_ok(self):
        cfg = ResolveConfig(max_iter=1)
        assert cfg.max_iter == 1

    def test_max_iter_zero_raises(self):
        with pytest.raises(ValueError):
            ResolveConfig(max_iter=0)

    def test_max_iter_neg_raises(self):
        with pytest.raises(ValueError):
            ResolveConfig(max_iter=-1)

    def test_gap_zero_ok(self):
        cfg = ResolveConfig(gap=0.0)
        assert cfg.gap == 0.0

    def test_gap_neg_raises(self):
        with pytest.raises(ValueError):
            ResolveConfig(gap=-1.0)

    def test_step_scale_small_ok(self):
        cfg = ResolveConfig(step_scale=0.01)
        assert cfg.step_scale == pytest.approx(0.01)

    def test_step_scale_zero_raises(self):
        with pytest.raises(ValueError):
            ResolveConfig(step_scale=0.0)

    def test_step_scale_neg_raises(self):
        with pytest.raises(ValueError):
            ResolveConfig(step_scale=-0.5)


# ─── TestBBox ─────────────────────────────────────────────────────────────────

class TestBBox:
    def test_basic(self):
        b = _box(0, 1.0, 2.0, 5.0, 3.0)
        assert b.fragment_id == 0
        assert b.x == pytest.approx(1.0)
        assert b.y == pytest.approx(2.0)
        assert b.w == pytest.approx(5.0)
        assert b.h == pytest.approx(3.0)

    def test_x2(self):
        b = _box(0, 1.0, 2.0, 5.0, 3.0)
        assert b.x2 == pytest.approx(6.0)

    def test_y2(self):
        b = _box(0, 1.0, 2.0, 5.0, 3.0)
        assert b.y2 == pytest.approx(5.0)

    def test_cx(self):
        b = _box(0, 0.0, 0.0, 10.0, 4.0)
        assert b.cx == pytest.approx(5.0)

    def test_cy(self):
        b = _box(0, 0.0, 0.0, 6.0, 8.0)
        assert b.cy == pytest.approx(4.0)

    def test_area(self):
        b = _box(0, 0.0, 0.0, 4.0, 5.0)
        assert b.area == pytest.approx(20.0)

    def test_translate_positive(self):
        b = _box(0, 1.0, 2.0)
        b2 = b.translate(3.0, 4.0)
        assert b2.x == pytest.approx(4.0)
        assert b2.y == pytest.approx(6.0)

    def test_translate_zero_identity(self):
        b = _box(0, 5.0, 7.0)
        b2 = b.translate(0.0, 0.0)
        assert b2.x == pytest.approx(5.0)
        assert b2.y == pytest.approx(7.0)

    def test_translate_does_not_mutate(self):
        b = _box(0, 1.0, 2.0)
        b.translate(10.0, 10.0)
        assert b.x == pytest.approx(1.0)

    def test_translate_preserves_size(self):
        b = _box(0, 0.0, 0.0, 7.0, 3.0)
        b2 = b.translate(5.0, 5.0)
        assert b2.w == pytest.approx(7.0)
        assert b2.h == pytest.approx(3.0)

    def test_translate_preserves_fragment_id(self):
        b = _box(42, 0.0, 0.0)
        b2 = b.translate(1.0, 1.0)
        assert b2.fragment_id == 42

    def test_w_zero_raises(self):
        with pytest.raises(ValueError):
            BBox(fragment_id=0, x=0.0, y=0.0, w=0.0, h=5.0)

    def test_w_neg_raises(self):
        with pytest.raises(ValueError):
            BBox(fragment_id=0, x=0.0, y=0.0, w=-1.0, h=5.0)

    def test_h_zero_raises(self):
        with pytest.raises(ValueError):
            BBox(fragment_id=0, x=0.0, y=0.0, w=5.0, h=0.0)

    def test_h_neg_raises(self):
        with pytest.raises(ValueError):
            BBox(fragment_id=0, x=0.0, y=0.0, w=5.0, h=-2.0)


# ─── TestOverlap ──────────────────────────────────────────────────────────────

class TestOverlap:
    def _make(self, area=10.0, dx=2.0, dy=0.0):
        return Overlap(id_a=0, id_b=1, area=area, dx=dx, dy=dy)

    def test_pair_key_ordered(self):
        ov = Overlap(id_a=3, id_b=1, area=5.0, dx=0.0, dy=0.0)
        assert ov.pair_key == (1, 3)

    def test_pair_key_already_ordered(self):
        ov = self._make()
        assert ov.pair_key == (0, 1)

    def test_has_overlap_true(self):
        ov = self._make(area=0.1)
        assert ov.has_overlap is True

    def test_has_overlap_false(self):
        ov = self._make(area=0.0)
        assert ov.has_overlap is False

    def test_ids_stored(self):
        ov = Overlap(id_a=5, id_b=9, area=1.0, dx=0.0, dy=0.0)
        assert ov.id_a == 5
        assert ov.id_b == 9


# ─── TestResolveResult ────────────────────────────────────────────────────────

class TestResolveResult:
    def _make(self, n_iter=3, resolved=True, n_ov_last=0):
        boxes = {0: _box(0), 1: _box(1, 20.0, 0.0)}
        history = [(i, max(0, 5 - i)) for i in range(n_iter)]
        history.append((n_iter, n_ov_last))
        return ResolveResult(boxes=boxes, n_iter=n_iter,
                             resolved=resolved, history=history)

    def test_basic(self):
        r = self._make()
        assert r.n_iter == 3
        assert r.resolved is True

    def test_final_n_overlaps(self):
        r = self._make(n_ov_last=2)
        assert r.final_n_overlaps == 2

    def test_final_n_overlaps_empty_history(self):
        r = ResolveResult(boxes={}, n_iter=0, resolved=True, history=[])
        assert r.final_n_overlaps == 0

    def test_fragment_ids(self):
        r = self._make()
        assert set(r.fragment_ids) == {0, 1}

    def test_fragment_ids_empty(self):
        r = ResolveResult(boxes={}, n_iter=0, resolved=True)
        assert r.fragment_ids == []

    def test_n_iter_neg_raises(self):
        with pytest.raises(ValueError):
            ResolveResult(boxes={}, n_iter=-1, resolved=True)

    def test_resolved_false(self):
        r = self._make(resolved=False)
        assert r.resolved is False


# ─── TestComputeOverlap ───────────────────────────────────────────────────────

class TestComputeOverlap:
    def test_no_overlap(self):
        a = _box(0, 0.0, 0.0, 5.0, 5.0)
        b = _box(1, 10.0, 0.0, 5.0, 5.0)
        ov = compute_overlap(a, b)
        assert ov.has_overlap is False
        assert ov.area == pytest.approx(0.0)

    def test_full_overlap(self):
        a = _box(0, 0.0, 0.0, 10.0, 10.0)
        b = _box(1, 0.0, 0.0, 10.0, 10.0)
        ov = compute_overlap(a, b)
        assert ov.has_overlap is True
        assert ov.area > 0.0

    def test_partial_overlap(self):
        a = _box(0, 0.0, 0.0, 10.0, 10.0)
        b = _box(1, 5.0, 0.0, 10.0, 10.0)
        ov = compute_overlap(a, b)
        assert ov.has_overlap is True
        assert ov.area > 0.0

    def test_touching_no_gap(self):
        a = _box(0, 0.0, 0.0, 5.0, 5.0)
        b = _box(1, 5.0, 0.0, 5.0, 5.0)
        ov = compute_overlap(a, b, gap=0.0)
        assert ov.area == pytest.approx(0.0)

    def test_gap_creates_overlap_for_close_boxes(self):
        a = _box(0, 0.0, 0.0, 5.0, 5.0)
        b = _box(1, 7.0, 0.0, 5.0, 5.0)
        ov_no_gap = compute_overlap(a, b, gap=0.0)
        ov_gap = compute_overlap(a, b, gap=3.0)
        assert ov_no_gap.has_overlap is False
        assert ov_gap.has_overlap is True

    def test_ids_stored(self):
        a = _box(3, 0.0, 0.0)
        b = _box(7, 5.0, 0.0)
        ov = compute_overlap(a, b)
        assert ov.id_a == 3
        assert ov.id_b == 7

    def test_dx_direction(self):
        a = _box(0, 0.0, 5.0, 10.0, 5.0)
        b = _box(1, 5.0, 5.0, 10.0, 5.0)
        ov = compute_overlap(a, b)
        assert ov.dx > 0

    def test_dy_direction(self):
        a = _box(0, 5.0, 0.0, 5.0, 10.0)
        b = _box(1, 5.0, 5.0, 5.0, 10.0)
        ov = compute_overlap(a, b)
        assert ov.dy > 0


# ─── TestDetectOverlaps ───────────────────────────────────────────────────────

class TestDetectOverlaps:
    def test_no_overlaps_empty(self):
        ov = detect_overlaps(_non_overlapping())
        assert ov == []

    def test_one_overlap(self):
        ov = detect_overlaps(_overlapping())
        assert len(ov) == 1

    def test_all_overlapping(self):
        boxes = {i: _box(i, 0.0, 0.0) for i in range(4)}
        ov = detect_overlaps(boxes)
        assert len(ov) == 6  # C(4,2) = 6

    def test_grid_no_overlaps(self):
        ov = detect_overlaps(_grid_4())
        assert ov == []

    def test_single_box_no_overlaps(self):
        ov = detect_overlaps({0: _box(0)})
        assert ov == []

    def test_empty_boxes(self):
        ov = detect_overlaps({})
        assert ov == []

    def test_gap_parameter(self):
        boxes = {
            0: _box(0, 0.0, 0.0, 5.0, 5.0),
            1: _box(1, 8.0, 0.0, 5.0, 5.0),
        }
        assert len(detect_overlaps(boxes, gap=0.0)) == 0
        assert len(detect_overlaps(boxes, gap=5.0)) == 1

    def test_overlap_objects_have_area(self):
        for ov in detect_overlaps(_overlapping()):
            assert ov.area > 0.0


# ─── TestResolveOverlaps ──────────────────────────────────────────────────────

class TestResolveOverlaps:
    def test_returns_result(self):
        r = resolve_overlaps(_overlapping())
        assert isinstance(r, ResolveResult)

    def test_all_fragments_in_result(self):
        r = resolve_overlaps(_overlapping())
        assert set(r.boxes.keys()) == {0, 1}

    def test_n_iter_positive(self):
        r = resolve_overlaps(_overlapping())
        assert r.n_iter >= 1

    def test_history_populated(self):
        r = resolve_overlaps(_overlapping())
        assert len(r.history) >= 1

    def test_no_overlap_resolves_immediately(self):
        r = resolve_overlaps(_non_overlapping())
        assert r.resolved is True

    def test_frozen_fragment_not_moved(self):
        boxes = _overlapping()
        orig_x = boxes[0].x
        cfg = ResolveConfig(max_iter=5, frozen_ids=[0])
        r = resolve_overlaps(boxes, cfg)
        assert r.boxes[0].x == pytest.approx(orig_x)

    def test_max_iter_respected(self):
        cfg = ResolveConfig(max_iter=2)
        r = resolve_overlaps(_overlapping(), cfg)
        assert r.n_iter <= 2

    def test_does_not_mutate_input(self):
        boxes = _overlapping()
        orig = {fid: (b.x, b.y) for fid, b in boxes.items()}
        resolve_overlaps(boxes)
        for fid, b in boxes.items():
            assert b.x == pytest.approx(orig[fid][0])
            assert b.y == pytest.approx(orig[fid][1])

    def test_empty_boxes(self):
        r = resolve_overlaps({})
        assert r.boxes == {}
        assert r.resolved is True

    def test_single_box(self):
        r = resolve_overlaps({0: _box(0)})
        assert 0 in r.boxes
        assert r.resolved is True

    def test_grid_already_ok(self):
        r = resolve_overlaps(_grid_4())
        assert r.resolved is True


# ─── TestComputeTotalOverlap ──────────────────────────────────────────────────

class TestComputeTotalOverlap:
    def test_no_overlap_zero(self):
        assert compute_total_overlap(_non_overlapping()) == pytest.approx(0.0)

    def test_overlap_positive(self):
        assert compute_total_overlap(_overlapping()) > 0.0

    def test_empty_boxes_zero(self):
        assert compute_total_overlap({}) == pytest.approx(0.0)

    def test_single_box_zero(self):
        assert compute_total_overlap({0: _box(0)}) == pytest.approx(0.0)

    def test_full_overlap_large(self):
        boxes = {i: _box(i, 0.0, 0.0, 10.0, 10.0) for i in range(3)}
        assert compute_total_overlap(boxes) > 0.0

    def test_gap_increases_overlap(self):
        total_no = compute_total_overlap(_non_overlapping(), gap=0.0)
        total_gap = compute_total_overlap(_non_overlapping(), gap=5.0)
        assert total_gap >= total_no


# ─── TestOverlapRatio ─────────────────────────────────────────────────────────

class TestOverlapRatio:
    def test_no_overlap_zero(self):
        assert overlap_ratio(_non_overlapping()) == pytest.approx(0.0)

    def test_overlap_in_range(self):
        ratio = overlap_ratio(_overlapping())
        assert 0.0 <= ratio <= 1.0

    def test_empty_boxes_zero(self):
        assert overlap_ratio({}) == pytest.approx(0.0)

    def test_single_box_zero(self):
        assert overlap_ratio({0: _box(0)}) == pytest.approx(0.0)

    def test_ratio_not_exceed_one(self):
        boxes = {i: _box(i, 0.0, 0.0) for i in range(5)}
        assert overlap_ratio(boxes) <= 1.0

    def test_resolved_layout_ratio(self):
        cfg = ResolveConfig(max_iter=50, gap=0.0, step_scale=1.0)
        r = resolve_overlaps(_overlapping(), cfg)
        ratio = overlap_ratio(r.boxes, gap=0.0)
        assert 0.0 <= ratio <= 1.0
