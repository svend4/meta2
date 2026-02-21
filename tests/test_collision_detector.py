"""Тесты для puzzle_reconstruction.assembly.collision_detector."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _rect(fid=0, x=0, y=0, w=10, h=10):
    return PlacedRect(fragment_id=fid, x=x, y=y, width=w, height=h)


# ─── TestPlacedRect ───────────────────────────────────────────────────────────

class TestPlacedRect:
    def test_basic_creation(self):
        r = _rect()
        assert r.fragment_id == 0
        assert r.width == 10

    def test_x2_property(self):
        r = _rect(x=5, w=10)
        assert r.x2 == 15

    def test_y2_property(self):
        r = _rect(y=3, h=8)
        assert r.y2 == 11

    def test_center_property(self):
        r = _rect(x=0, y=0, w=10, h=10)
        assert r.center == (5.0, 5.0)

    def test_area_property(self):
        r = _rect(w=10, h=20)
        assert r.area == 200

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            _rect(fid=-1)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError):
            _rect(x=-1)

    def test_negative_y_raises(self):
        with pytest.raises(ValueError):
            _rect(y=-1)

    def test_width_zero_raises(self):
        with pytest.raises(ValueError):
            _rect(w=0)

    def test_height_zero_raises(self):
        with pytest.raises(ValueError):
            _rect(h=0)


# ─── TestCollisionInfo ────────────────────────────────────────────────────────

class TestCollisionInfo:
    def _make(self, ow=5, oh=5, oa=25):
        return CollisionInfo(id1=0, id2=1, overlap_w=ow,
                             overlap_h=oh, overlap_area=oa)

    def test_basic_creation(self):
        ci = self._make()
        assert ci.overlap_area == 25

    def test_pair_property(self):
        ci = self._make()
        assert ci.pair == (0, 1)

    def test_negative_overlap_w_raises(self):
        with pytest.raises(ValueError):
            CollisionInfo(id1=0, id2=1, overlap_w=-1, overlap_h=5, overlap_area=0)

    def test_negative_overlap_h_raises(self):
        with pytest.raises(ValueError):
            CollisionInfo(id1=0, id2=1, overlap_w=5, overlap_h=-1, overlap_area=0)

    def test_negative_overlap_area_raises(self):
        with pytest.raises(ValueError):
            CollisionInfo(id1=0, id2=1, overlap_w=5, overlap_h=5, overlap_area=-1)

    def test_zero_values_valid(self):
        ci = CollisionInfo(id1=0, id2=1, overlap_w=0, overlap_h=0, overlap_area=0)
        assert ci.overlap_area == 0

    def test_resolve_vec_default(self):
        ci = self._make()
        assert ci.resolve_vec == (0, 0)


# ─── TestAabbOverlap ──────────────────────────────────────────────────────────

class TestAabbOverlap:
    def test_overlapping(self):
        a = _rect(fid=0, x=0, y=0, w=10, h=10)
        b = _rect(fid=1, x=5, y=5, w=10, h=10)
        assert aabb_overlap(a, b) is True

    def test_no_overlap_x(self):
        a = _rect(fid=0, x=0, y=0, w=10, h=10)
        b = _rect(fid=1, x=20, y=0, w=10, h=10)
        assert aabb_overlap(a, b) is False

    def test_no_overlap_y(self):
        a = _rect(fid=0, x=0, y=0, w=10, h=10)
        b = _rect(fid=1, x=0, y=20, w=10, h=10)
        assert aabb_overlap(a, b) is False

    def test_touching_x_not_overlap(self):
        a = _rect(fid=0, x=0, y=0, w=10, h=10)
        b = _rect(fid=1, x=10, y=0, w=10, h=10)
        assert aabb_overlap(a, b) is False

    def test_touching_y_not_overlap(self):
        a = _rect(fid=0, x=0, y=0, w=10, h=10)
        b = _rect(fid=1, x=0, y=10, w=10, h=10)
        assert aabb_overlap(a, b) is False

    def test_contained(self):
        a = _rect(fid=0, x=0, y=0, w=20, h=20)
        b = _rect(fid=1, x=5, y=5, w=5, h=5)
        assert aabb_overlap(a, b) is True

    def test_identical(self):
        a = _rect(fid=0, x=0, y=0, w=10, h=10)
        b = _rect(fid=1, x=0, y=0, w=10, h=10)
        assert aabb_overlap(a, b) is True


# ─── TestComputeOverlap ───────────────────────────────────────────────────────

class TestComputeOverlap:
    def test_no_overlap_returns_none(self):
        a = _rect(fid=0, x=0, y=0, w=10, h=10)
        b = _rect(fid=1, x=20, y=0, w=10, h=10)
        assert compute_overlap(a, b) is None

    def test_overlap_returns_collision_info(self):
        a = _rect(fid=0, x=0, y=0, w=10, h=10)
        b = _rect(fid=1, x=5, y=0, w=10, h=10)
        ci = compute_overlap(a, b)
        assert isinstance(ci, CollisionInfo)

    def test_overlap_width_correct(self):
        a = _rect(fid=0, x=0, y=0, w=10, h=10)
        b = _rect(fid=1, x=6, y=0, w=10, h=10)
        ci = compute_overlap(a, b)
        assert ci.overlap_w == 4

    def test_overlap_area_nonnegative(self):
        a = _rect(fid=0, x=0, y=0, w=10, h=10)
        b = _rect(fid=1, x=5, y=5, w=10, h=10)
        ci = compute_overlap(a, b)
        assert ci.overlap_area >= 0

    def test_ids_stored(self):
        a = _rect(fid=3, x=0, y=0, w=10, h=10)
        b = _rect(fid=7, x=5, y=0, w=10, h=10)
        ci = compute_overlap(a, b)
        assert ci.id1 == 3 and ci.id2 == 7

    def test_full_overlap(self):
        a = _rect(fid=0, x=0, y=0, w=10, h=10)
        b = _rect(fid=1, x=0, y=0, w=10, h=10)
        ci = compute_overlap(a, b)
        assert ci.overlap_area == 100


# ─── TestDetectCollisions ─────────────────────────────────────────────────────

class TestDetectCollisions:
    def test_no_collisions(self):
        rects = [_rect(fid=i, x=i * 20, y=0) for i in range(3)]
        assert detect_collisions(rects) == []

    def test_one_collision(self):
        rects = [
            _rect(fid=0, x=0, y=0, w=10, h=10),
            _rect(fid=1, x=5, y=0, w=10, h=10),
            _rect(fid=2, x=30, y=0, w=10, h=10),
        ]
        collisions = detect_collisions(rects)
        assert len(collisions) == 1

    def test_all_collide(self):
        rects = [_rect(fid=i, x=0, y=0, w=10, h=10) for i in range(3)]
        collisions = detect_collisions(rects)
        assert len(collisions) == 3  # C(3,2)

    def test_empty_list(self):
        assert detect_collisions([]) == []

    def test_single_rect(self):
        assert detect_collisions([_rect()]) == []

    def test_returns_list_of_collision_info(self):
        rects = [_rect(fid=0, x=0, y=0), _rect(fid=1, x=5, y=5)]
        collisions = detect_collisions(rects)
        assert all(isinstance(c, CollisionInfo) for c in collisions)


# ─── TestCollisionGraph ───────────────────────────────────────────────────────

class TestCollisionGraph:
    def test_empty_collisions(self):
        assert collision_graph([]) == {}

    def test_symmetric(self):
        ci = CollisionInfo(id1=0, id2=1, overlap_w=5, overlap_h=5, overlap_area=25)
        graph = collision_graph([ci])
        assert 1 in graph[0]
        assert 0 in graph[1]

    def test_correct_structure(self):
        rects = [_rect(fid=i, x=0, y=0, w=10, h=10) for i in range(3)]
        collisions = detect_collisions(rects)
        graph = collision_graph(collisions)
        for fid in range(3):
            assert fid in graph
            assert len(graph[fid]) == 2  # каждый связан с двумя другими


# ─── TestIsCollisionFree ──────────────────────────────────────────────────────

class TestIsCollisionFree:
    def test_free_layout(self):
        rects = [_rect(fid=i, x=i * 20, y=0) for i in range(4)]
        assert is_collision_free(rects) is True

    def test_overlapping_layout(self):
        rects = [_rect(fid=0, x=0, y=0), _rect(fid=1, x=5, y=5)]
        assert is_collision_free(rects) is False

    def test_empty(self):
        assert is_collision_free([]) is True

    def test_single(self):
        assert is_collision_free([_rect()]) is True

    def test_touching_not_collision(self):
        rects = [
            _rect(fid=0, x=0, y=0, w=10, h=10),
            _rect(fid=1, x=10, y=0, w=10, h=10),
        ]
        assert is_collision_free(rects) is True


# ─── TestTotalOverlapArea ─────────────────────────────────────────────────────

class TestTotalOverlapArea:
    def test_empty(self):
        assert total_overlap_area([]) == 0

    def test_single(self):
        ci = CollisionInfo(id1=0, id2=1, overlap_w=5, overlap_h=4, overlap_area=20)
        assert total_overlap_area([ci]) == 20

    def test_multiple(self):
        cis = [
            CollisionInfo(id1=0, id2=1, overlap_w=5, overlap_h=4, overlap_area=20),
            CollisionInfo(id1=1, id2=2, overlap_w=3, overlap_h=3, overlap_area=9),
        ]
        assert total_overlap_area(cis) == 29

    def test_nonnegative(self):
        rects = [_rect(fid=i, x=0, y=0) for i in range(3)]
        collisions = detect_collisions(rects)
        assert total_overlap_area(collisions) >= 0


# ─── TestResolveGreedy ────────────────────────────────────────────────────────

class TestResolveGreedy:
    def test_returns_list(self):
        rects = [_rect(fid=i, x=i * 20, y=0) for i in range(3)]
        result = resolve_greedy(rects)
        assert isinstance(result, list)

    def test_same_length(self):
        rects = [_rect(fid=i, x=0, y=0, w=10, h=10) for i in range(3)]
        result = resolve_greedy(rects)
        assert len(result) == len(rects)

    def test_no_collision_unchanged(self):
        rects = [_rect(fid=i, x=i * 20, y=0) for i in range(3)]
        result = resolve_greedy(rects)
        for orig, res in zip(rects, result):
            assert orig.x == res.x and orig.y == res.y

    def test_positions_nonnegative(self):
        rects = [_rect(fid=i, x=0, y=0) for i in range(3)]
        result = resolve_greedy(rects)
        for r in result:
            assert r.x >= 0 and r.y >= 0

    def test_max_iter_zero_raises(self):
        with pytest.raises(ValueError):
            resolve_greedy([_rect()], max_iter=0)

    def test_each_placed_rect(self):
        rects = [_rect(fid=i, x=0, y=0) for i in range(2)]
        result = resolve_greedy(rects)
        assert all(isinstance(r, PlacedRect) for r in result)


# ─── TestBatchDetect ──────────────────────────────────────────────────────────

class TestBatchDetect:
    def test_returns_list(self):
        groups = [[_rect(fid=i, x=i * 20, y=0) for i in range(3)]]
        result = batch_detect(groups)
        assert isinstance(result, list)

    def test_correct_length(self):
        groups = [
            [_rect(fid=0, x=0, y=0), _rect(fid=1, x=5, y=0)],
            [_rect(fid=2, x=0, y=0), _rect(fid=3, x=20, y=0)],
        ]
        result = batch_detect(groups)
        assert len(result) == 2

    def test_empty_groups(self):
        assert batch_detect([]) == []

    def test_no_collision_group(self):
        group = [_rect(fid=i, x=i * 20, y=0) for i in range(3)]
        result = batch_detect([group])
        assert result[0] == []

    def test_each_inner_list(self):
        groups = [[_rect(fid=0, x=0), _rect(fid=1, x=5)]]
        result = batch_detect(groups)
        assert isinstance(result[0], list)
