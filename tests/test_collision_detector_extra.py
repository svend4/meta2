"""Extra tests for puzzle_reconstruction.assembly.collision_detector."""
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _rect(fid=0, x=0, y=0, w=10, h=10):
    return PlacedRect(fragment_id=fid, x=x, y=y, width=w, height=h)


def _ci(id1=0, id2=1, ow=5, oh=5, oa=25):
    return CollisionInfo(id1=id1, id2=id2, overlap_w=ow, overlap_h=oh, overlap_area=oa)


# ─── TestPlacedRectExtra ─────────────────────────────────────────────────────

class TestPlacedRectExtra:
    def test_fragment_id_stored(self):
        r = _rect(fid=7)
        assert r.fragment_id == 7

    def test_large_dimensions(self):
        r = _rect(w=1000, h=2000)
        assert r.area == 2_000_000

    def test_x2_large(self):
        r = _rect(x=500, w=200)
        assert r.x2 == 700

    def test_y2_large(self):
        r = _rect(y=300, h=100)
        assert r.y2 == 400

    def test_area_square(self):
        r = _rect(w=7, h=7)
        assert r.area == 49

    def test_center_non_integer(self):
        r = _rect(x=0, y=0, w=7, h=7)
        cx, cy = r.center
        assert cx == pytest.approx(3.5)
        assert cy == pytest.approx(3.5)

    def test_width_one(self):
        r = _rect(w=1, h=1)
        assert r.area == 1

    def test_different_width_height(self):
        r = _rect(w=4, h=6)
        assert r.area == 24


# ─── TestCollisionInfoExtra ──────────────────────────────────────────────────

class TestCollisionInfoExtra:
    def test_id1_stored(self):
        ci = _ci(id1=3, id2=5)
        assert ci.id1 == 3

    def test_id2_stored(self):
        ci = _ci(id1=3, id2=5)
        assert ci.id2 == 5

    def test_pair_order(self):
        ci = _ci(id1=0, id2=1)
        assert ci.pair == (0, 1)

    def test_overlap_area_large(self):
        ci = CollisionInfo(id1=0, id2=1, overlap_w=100, overlap_h=200, overlap_area=20000)
        assert ci.overlap_area == 20000

    def test_zero_overlap_w(self):
        ci = CollisionInfo(id1=0, id2=1, overlap_w=0, overlap_h=5, overlap_area=0)
        assert ci.overlap_w == 0

    def test_zero_overlap_h(self):
        ci = CollisionInfo(id1=0, id2=1, overlap_w=5, overlap_h=0, overlap_area=0)
        assert ci.overlap_h == 0


# ─── TestAabbOverlapExtra ────────────────────────────────────────────────────

class TestAabbOverlapExtra:
    def test_partial_overlap_x(self):
        a = _rect(fid=0, x=0, y=0, w=10, h=10)
        b = _rect(fid=1, x=5, y=0, w=10, h=10)
        assert aabb_overlap(a, b) is True

    def test_partial_overlap_y(self):
        a = _rect(fid=0, x=0, y=0, w=10, h=10)
        b = _rect(fid=1, x=0, y=5, w=10, h=10)
        assert aabb_overlap(a, b) is True

    def test_diagonally_separated(self):
        a = _rect(fid=0, x=0, y=0, w=5, h=5)
        b = _rect(fid=1, x=10, y=10, w=5, h=5)
        assert aabb_overlap(a, b) is False

    def test_far_apart(self):
        a = _rect(fid=0, x=0, y=0, w=10, h=10)
        b = _rect(fid=1, x=100, y=100, w=10, h=10)
        assert aabb_overlap(a, b) is False

    def test_one_pixel_overlap(self):
        a = _rect(fid=0, x=0, y=0, w=11, h=11)
        b = _rect(fid=1, x=10, y=10, w=10, h=10)
        # x: [0,11) vs [10,20) → overlap 1px; y: same
        assert aabb_overlap(a, b) is True

    def test_different_sizes_overlap(self):
        a = _rect(fid=0, x=0, y=0, w=100, h=100)
        b = _rect(fid=1, x=50, y=50, w=5, h=5)
        assert aabb_overlap(a, b) is True


# ─── TestComputeOverlapExtra ─────────────────────────────────────────────────

class TestComputeOverlapExtra:
    def test_partial_overlap_area(self):
        a = _rect(fid=0, x=0, y=0, w=10, h=10)
        b = _rect(fid=1, x=5, y=5, w=10, h=10)
        ci = compute_overlap(a, b)
        assert ci is not None
        assert ci.overlap_area == 25

    def test_full_containment_area(self):
        a = _rect(fid=0, x=0, y=0, w=20, h=20)
        b = _rect(fid=1, x=5, y=5, w=5, h=5)
        ci = compute_overlap(a, b)
        assert ci is not None
        assert ci.overlap_area == 25

    def test_symmetric_no_overlap(self):
        a = _rect(fid=0, x=0, y=0, w=5, h=5)
        b = _rect(fid=1, x=20, y=20, w=5, h=5)
        assert compute_overlap(a, b) is None
        assert compute_overlap(b, a) is None

    def test_ids_preserved(self):
        a = _rect(fid=4, x=0, y=0)
        b = _rect(fid=9, x=5, y=0)
        ci = compute_overlap(a, b)
        assert ci.id1 == 4 and ci.id2 == 9

    def test_touching_x_is_none(self):
        a = _rect(fid=0, x=0, y=0, w=10, h=10)
        b = _rect(fid=1, x=10, y=0, w=10, h=10)
        assert compute_overlap(a, b) is None


# ─── TestDetectCollisionsExtra ───────────────────────────────────────────────

class TestDetectCollisionsExtra:
    def test_two_colliding(self):
        rects = [_rect(fid=0, x=0, y=0), _rect(fid=1, x=5, y=0)]
        assert len(detect_collisions(rects)) == 1

    def test_four_no_collision(self):
        rects = [_rect(fid=i, x=i * 20, y=0) for i in range(4)]
        assert detect_collisions(rects) == []

    def test_collision_count_complete(self):
        # n=4 all overlap → C(4,2)=6
        rects = [_rect(fid=i, x=0, y=0) for i in range(4)]
        assert len(detect_collisions(rects)) == 6

    def test_two_no_collision(self):
        rects = [_rect(fid=0, x=0), _rect(fid=1, x=20)]
        assert detect_collisions(rects) == []

    def test_returns_collision_infos(self):
        rects = [_rect(fid=0, x=0), _rect(fid=1, x=5)]
        result = detect_collisions(rects)
        assert all(isinstance(c, CollisionInfo) for c in result)


# ─── TestCollisionGraphExtra ─────────────────────────────────────────────────

class TestCollisionGraphExtra:
    def test_no_self_loops(self):
        ci = _ci(id1=0, id2=1)
        graph = collision_graph([ci])
        assert 0 not in graph.get(0, [])
        assert 1 not in graph.get(1, [])

    def test_two_separate_collisions(self):
        cis = [_ci(id1=0, id2=1), _ci(id1=2, id2=3)]
        graph = collision_graph(cis)
        assert 1 in graph[0]
        assert 3 in graph[2]

    def test_triangle_collision(self):
        cis = [_ci(0, 1), _ci(1, 2), _ci(0, 2)]
        graph = collision_graph(cis)
        assert len(graph[0]) == 2
        assert len(graph[1]) == 2
        assert len(graph[2]) == 2

    def test_both_directions(self):
        ci = _ci(id1=5, id2=8)
        graph = collision_graph([ci])
        assert 8 in graph[5]
        assert 5 in graph[8]


# ─── TestIsCollisionFreeExtra ─────────────────────────────────────────────────

class TestIsCollisionFreeExtra:
    def test_three_spaced_rects(self):
        rects = [_rect(fid=i, x=i * 20, y=0) for i in range(3)]
        assert is_collision_free(rects) is True

    def test_two_overlapping(self):
        rects = [_rect(fid=0, x=0), _rect(fid=1, x=5)]
        assert is_collision_free(rects) is False

    def test_touching_is_free(self):
        rects = [_rect(fid=0, x=0, w=10), _rect(fid=1, x=10, w=10)]
        assert is_collision_free(rects) is True

    def test_two_element_overlap(self):
        rects = [_rect(fid=0, x=0), _rect(fid=1, x=0)]
        assert is_collision_free(rects) is False


# ─── TestTotalOverlapAreaExtra ────────────────────────────────────────────────

class TestTotalOverlapAreaExtra:
    def test_zero_area_collisions(self):
        cis = [CollisionInfo(0, 1, 0, 0, 0), CollisionInfo(1, 2, 0, 0, 0)]
        assert total_overlap_area(cis) == 0

    def test_three_collisions(self):
        cis = [_ci(oa=10), _ci(oa=20), _ci(oa=30)]
        assert total_overlap_area(cis) == 60

    def test_single_collision(self):
        assert total_overlap_area([_ci(oa=42)]) == 42

    def test_large_areas(self):
        cis = [CollisionInfo(0, 1, 100, 100, 10000)]
        assert total_overlap_area(cis) == 10000


# ─── TestResolveGreedyExtra ───────────────────────────────────────────────────

class TestResolveGreedyExtra:
    def test_preserves_fragment_ids(self):
        rects = [_rect(fid=i, x=0, y=0) for i in range(3)]
        result = resolve_greedy(rects)
        fids = {r.fragment_id for r in result}
        assert fids == {0, 1, 2}

    def test_no_collision_result_collision_free(self):
        rects = [_rect(fid=i, x=i * 30, y=0) for i in range(3)]
        result = resolve_greedy(rects)
        assert is_collision_free(result)

    def test_returns_placed_rect_list(self):
        rects = [_rect(fid=i, x=0, y=0) for i in range(2)]
        result = resolve_greedy(rects)
        assert all(isinstance(r, PlacedRect) for r in result)

    def test_single_rect_unchanged(self):
        r = _rect(fid=0, x=5, y=10)
        result = resolve_greedy([r])
        assert result[0].x == 5 and result[0].y == 10


# ─── TestBatchDetectExtra ─────────────────────────────────────────────────────

class TestBatchDetectExtra:
    def test_two_groups_lengths(self):
        groups = [
            [_rect(fid=0, x=0), _rect(fid=1, x=5)],
            [_rect(fid=2, x=0, y=0), _rect(fid=3, x=30, y=0)],
        ]
        result = batch_detect(groups)
        assert len(result[0]) == 1
        assert len(result[1]) == 0

    def test_all_collision_group(self):
        group = [_rect(fid=i, x=0, y=0) for i in range(3)]
        result = batch_detect([group])
        assert len(result[0]) == 3  # C(3,2)

    def test_empty_group_in_batch(self):
        result = batch_detect([[]])
        assert result[0] == []

    def test_single_rect_group(self):
        result = batch_detect([[_rect()]])
        assert result[0] == []

    def test_three_groups(self):
        groups = [
            [_rect(fid=i, x=i * 20) for i in range(3)],
            [_rect(fid=i, x=0) for i in range(3)],
            [],
        ]
        result = batch_detect(groups)
        assert len(result) == 3
        assert result[0] == []
        assert len(result[1]) == 3
        assert result[2] == []
