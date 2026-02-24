"""Extra tests for puzzle_reconstruction/assembly/collision_detector.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.assembly.collision_detector import (
    PlacedRect,
    CollisionInfo,
    aabb_overlap,
    compute_overlap,
    detect_collisions,
    collision_graph,
    is_collision_free,
    total_overlap_area,
    resolve_greedy,
    batch_detect,
)


def _rect(fid, x, y, w, h):
    return PlacedRect(fragment_id=fid, x=x, y=y, width=w, height=h)


# ─── PlacedRect (extra) ─────────────────────────────────────────────────────

class TestPlacedRectExtra:
    def test_center_offset(self):
        r = _rect(0, 10, 20, 6, 8)
        assert r.center == (13.0, 24.0)

    def test_area_large(self):
        r = _rect(0, 0, 0, 100, 200)
        assert r.area == 20000

    def test_x2_y2_consistent(self):
        r = _rect(0, 5, 10, 15, 20)
        assert r.x2 == 20
        assert r.y2 == 30

    def test_fragment_id_zero_ok(self):
        r = _rect(0, 0, 0, 1, 1)
        assert r.fragment_id == 0

    def test_unit_square(self):
        r = _rect(0, 0, 0, 1, 1)
        assert r.area == 1
        assert r.center == (0.5, 0.5)


# ─── CollisionInfo (extra) ───────────────────────────────────────────────────

class TestCollisionInfoExtra:
    def test_large_overlap(self):
        ci = CollisionInfo(id1=0, id2=1, overlap_w=100, overlap_h=200, overlap_area=20000)
        assert ci.overlap_area == 20000

    def test_pair_order(self):
        ci = CollisionInfo(id1=5, id2=2, overlap_w=1, overlap_h=1, overlap_area=1)
        assert ci.pair == (5, 2)

    def test_resolve_vec_custom(self):
        ci = CollisionInfo(id1=0, id2=1, overlap_w=3, overlap_h=4, overlap_area=12,
                           resolve_vec=(-5, 10))
        assert ci.resolve_vec == (-5, 10)

    def test_zero_overlap_area_ok(self):
        ci = CollisionInfo(id1=0, id2=1, overlap_w=0, overlap_h=0, overlap_area=0)
        assert ci.overlap_area == 0

    def test_ids_stored(self):
        ci = CollisionInfo(id1=10, id2=20, overlap_w=0, overlap_h=0, overlap_area=0)
        assert ci.id1 == 10
        assert ci.id2 == 20


# ─── aabb_overlap (extra) ────────────────────────────────────────────────────

class TestAabbOverlapExtra:
    def test_identical_rects_overlap(self):
        r = _rect(0, 5, 5, 10, 10)
        assert aabb_overlap(r, r) is True

    def test_one_pixel_overlap(self):
        a = _rect(0, 0, 0, 10, 10)
        b = _rect(1, 9, 9, 10, 10)
        assert aabb_overlap(a, b) is True

    def test_large_rects_no_overlap(self):
        a = _rect(0, 0, 0, 100, 100)
        b = _rect(1, 200, 200, 100, 100)
        assert aabb_overlap(a, b) is False

    def test_vertically_aligned_no_overlap(self):
        a = _rect(0, 0, 0, 10, 5)
        b = _rect(1, 0, 5, 10, 5)
        assert aabb_overlap(a, b) is False

    def test_nested_rects(self):
        outer = _rect(0, 0, 0, 100, 100)
        inner = _rect(1, 40, 40, 20, 20)
        assert aabb_overlap(outer, inner) is True


# ─── compute_overlap (extra) ─────────────────────────────────────────────────

class TestComputeOverlapExtra:
    def test_identical_rects_full_overlap(self):
        r = _rect(0, 0, 0, 10, 10)
        result = compute_overlap(r, _rect(1, 0, 0, 10, 10))
        assert result is not None
        assert result.overlap_area == 100

    def test_contained_rect(self):
        outer = _rect(0, 0, 0, 20, 20)
        inner = _rect(1, 5, 5, 5, 5)
        result = compute_overlap(outer, inner)
        assert result is not None
        assert result.overlap_area == 25

    def test_symmetric_overlap(self):
        a = _rect(0, 0, 0, 10, 10)
        b = _rect(1, 5, 0, 10, 10)
        r1 = compute_overlap(a, b)
        r2 = compute_overlap(b, a)
        assert r1.overlap_area == r2.overlap_area

    def test_one_pixel_overlap_area(self):
        a = _rect(0, 0, 0, 10, 10)
        b = _rect(1, 9, 9, 10, 10)
        result = compute_overlap(a, b)
        assert result is not None
        assert result.overlap_area == 1

    def test_no_y_overlap_returns_none(self):
        a = _rect(0, 0, 0, 10, 5)
        b = _rect(1, 0, 5, 10, 5)
        assert compute_overlap(a, b) is None


# ─── detect_collisions (extra) ───────────────────────────────────────────────

class TestDetectCollisionsExtra:
    def test_four_rects_chain(self):
        rects = [
            _rect(0, 0, 0, 10, 10),
            _rect(1, 8, 0, 10, 10),
            _rect(2, 16, 0, 10, 10),
            _rect(3, 24, 0, 10, 10),
        ]
        result = detect_collisions(rects)
        assert len(result) == 3

    def test_all_separate(self):
        rects = [_rect(i, i * 20, 0, 5, 5) for i in range(5)]
        assert detect_collisions(rects) == []

    def test_two_rects_same_position(self):
        rects = [_rect(0, 0, 0, 10, 10), _rect(1, 0, 0, 10, 10)]
        result = detect_collisions(rects)
        assert len(result) == 1

    def test_collision_info_types(self):
        rects = [_rect(0, 0, 0, 10, 10), _rect(1, 5, 0, 10, 10)]
        for ci in detect_collisions(rects):
            assert isinstance(ci, CollisionInfo)
            assert ci.overlap_area > 0


# ─── collision_graph (extra) ─────────────────────────────────────────────────

class TestCollisionGraphExtra:
    def test_two_collisions_triangle(self):
        c1 = CollisionInfo(id1=0, id2=1, overlap_w=1, overlap_h=1, overlap_area=1)
        c2 = CollisionInfo(id1=1, id2=2, overlap_w=1, overlap_h=1, overlap_area=1)
        c3 = CollisionInfo(id1=0, id2=2, overlap_w=1, overlap_h=1, overlap_area=1)
        g = collision_graph([c1, c2, c3])
        assert len(g[0]) == 2
        assert len(g[1]) == 2
        assert len(g[2]) == 2

    def test_single_node_no_collision(self):
        g = collision_graph([])
        assert g == {}

    def test_all_nodes_present(self):
        c = CollisionInfo(id1=10, id2=20, overlap_w=1, overlap_h=1, overlap_area=1)
        g = collision_graph([c])
        assert 10 in g
        assert 20 in g


# ─── is_collision_free (extra) ───────────────────────────────────────────────

class TestIsCollisionFreeExtra:
    def test_many_non_overlapping(self):
        rects = [_rect(i, i * 20, 0, 5, 5) for i in range(10)]
        assert is_collision_free(rects) is True

    def test_all_same_position_not_free(self):
        rects = [_rect(i, 0, 0, 10, 10) for i in range(3)]
        assert is_collision_free(rects) is False

    def test_returns_bool(self):
        assert isinstance(is_collision_free([]), bool)


# ─── total_overlap_area (extra) ──────────────────────────────────────────────

class TestTotalOverlapAreaExtra:
    def test_three_collisions_sum(self):
        c1 = CollisionInfo(id1=0, id2=1, overlap_w=5, overlap_h=5, overlap_area=25)
        c2 = CollisionInfo(id1=1, id2=2, overlap_w=3, overlap_h=3, overlap_area=9)
        c3 = CollisionInfo(id1=0, id2=2, overlap_w=2, overlap_h=2, overlap_area=4)
        assert total_overlap_area([c1, c2, c3]) == 38

    def test_single_zero_area(self):
        c = CollisionInfo(id1=0, id2=1, overlap_w=0, overlap_h=0, overlap_area=0)
        assert total_overlap_area([c]) == 0

    def test_result_nonneg(self):
        c = CollisionInfo(id1=0, id2=1, overlap_w=5, overlap_h=5, overlap_area=25)
        assert total_overlap_area([c]) >= 0


# ─── resolve_greedy (extra) ──────────────────────────────────────────────────

class TestResolveGreedyExtra:
    def test_single_rect_unchanged(self):
        r = _rect(0, 5, 5, 10, 10)
        result = resolve_greedy([r])
        assert result[0].x == 5
        assert result[0].y == 5

    def test_three_overlapping_returns_valid(self):
        rects = [_rect(0, 0, 0, 10, 10), _rect(1, 3, 0, 10, 10), _rect(2, 6, 0, 10, 10)]
        result = resolve_greedy(rects, max_iter=100)
        assert len(result) == 3
        assert all(isinstance(r, PlacedRect) for r in result)

    def test_preserves_fragment_ids(self):
        rects = [_rect(10, 0, 0, 5, 5), _rect(20, 3, 0, 5, 5)]
        result = resolve_greedy(rects, max_iter=50)
        ids = {r.fragment_id for r in result}
        assert ids == {10, 20}


# ─── batch_detect (extra) ────────────────────────────────────────────────────

class TestBatchDetectExtra:
    def test_single_empty_group(self):
        result = batch_detect([[]])
        assert result == [[]]

    def test_mixed_groups(self):
        g1 = [_rect(0, 0, 0, 5, 5), _rect(1, 3, 0, 5, 5)]
        g2 = [_rect(0, 0, 0, 5, 5), _rect(1, 20, 0, 5, 5)]
        result = batch_detect([g1, g2])
        assert len(result[0]) >= 1
        assert len(result[1]) == 0

    def test_five_groups(self):
        groups = [[_rect(0, 0, 0, 5, 5)] for _ in range(5)]
        result = batch_detect(groups)
        assert len(result) == 5

    def test_all_results_are_lists(self):
        g = [_rect(0, 0, 0, 10, 10), _rect(1, 5, 0, 10, 10)]
        result = batch_detect([g])
        assert isinstance(result[0], list)
