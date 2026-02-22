"""Tests for puzzle_reconstruction/assembly/collision_detector.py"""
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


# ─── PlacedRect ───────────────────────────────────────────────────────────────

class TestPlacedRect:
    def test_basic_attributes(self):
        r = _rect(0, 10, 20, 30, 40)
        assert r.fragment_id == 0
        assert r.x == 10
        assert r.y == 20
        assert r.width == 30
        assert r.height == 40

    def test_x2(self):
        r = _rect(0, 5, 3, 10, 8)
        assert r.x2 == 15

    def test_y2(self):
        r = _rect(0, 5, 3, 10, 8)
        assert r.y2 == 11

    def test_center(self):
        r = _rect(0, 0, 0, 10, 10)
        assert r.center == (5.0, 5.0)

    def test_center_non_square(self):
        r = _rect(0, 4, 6, 8, 4)
        assert r.center == (8.0, 8.0)

    def test_area(self):
        r = _rect(0, 0, 0, 6, 4)
        assert r.area == 24

    def test_area_1x1(self):
        r = _rect(0, 0, 0, 1, 1)
        assert r.area == 1

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError, match="fragment_id"):
            _rect(-1, 0, 0, 10, 10)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError, match="x"):
            _rect(0, -1, 0, 10, 10)

    def test_negative_y_raises(self):
        with pytest.raises(ValueError, match="y"):
            _rect(0, 0, -1, 10, 10)

    def test_width_zero_raises(self):
        with pytest.raises(ValueError, match="width"):
            _rect(0, 0, 0, 0, 10)

    def test_height_zero_raises(self):
        with pytest.raises(ValueError, match="height"):
            _rect(0, 0, 0, 10, 0)

    def test_width_negative_raises(self):
        with pytest.raises(ValueError):
            _rect(0, 0, 0, -1, 10)

    def test_x_zero_ok(self):
        r = _rect(0, 0, 0, 1, 1)
        assert r.x == 0

    def test_large_fragment_id_ok(self):
        r = _rect(999, 0, 0, 100, 100)
        assert r.fragment_id == 999


# ─── CollisionInfo ────────────────────────────────────────────────────────────

class TestCollisionInfo:
    def test_basic(self):
        ci = CollisionInfo(id1=0, id2=1, overlap_w=5, overlap_h=3, overlap_area=15)
        assert ci.id1 == 0
        assert ci.id2 == 1
        assert ci.overlap_w == 5
        assert ci.overlap_h == 3
        assert ci.overlap_area == 15

    def test_pair_property(self):
        ci = CollisionInfo(id1=3, id2=7, overlap_w=0, overlap_h=0, overlap_area=0)
        assert ci.pair == (3, 7)

    def test_default_resolve_vec(self):
        ci = CollisionInfo(id1=0, id2=1, overlap_w=0, overlap_h=0, overlap_area=0)
        assert ci.resolve_vec == (0, 0)

    def test_custom_resolve_vec(self):
        ci = CollisionInfo(id1=0, id2=1, overlap_w=2, overlap_h=2, overlap_area=4,
                           resolve_vec=(3, -1))
        assert ci.resolve_vec == (3, -1)

    def test_negative_overlap_w_raises(self):
        with pytest.raises(ValueError, match="overlap_w"):
            CollisionInfo(id1=0, id2=1, overlap_w=-1, overlap_h=5, overlap_area=0)

    def test_negative_overlap_h_raises(self):
        with pytest.raises(ValueError, match="overlap_h"):
            CollisionInfo(id1=0, id2=1, overlap_w=5, overlap_h=-1, overlap_area=0)

    def test_negative_overlap_area_raises(self):
        with pytest.raises(ValueError, match="overlap_area"):
            CollisionInfo(id1=0, id2=1, overlap_w=0, overlap_h=0, overlap_area=-1)

    def test_zero_dimensions_ok(self):
        ci = CollisionInfo(id1=0, id2=1, overlap_w=0, overlap_h=0, overlap_area=0)
        assert ci.overlap_area == 0


# ─── aabb_overlap ─────────────────────────────────────────────────────────────

class TestAabbOverlap:
    def test_overlapping_diagonal(self):
        a = _rect(0, 0, 0, 10, 10)
        b = _rect(1, 5, 5, 10, 10)
        assert aabb_overlap(a, b) is True

    def test_far_apart(self):
        a = _rect(0, 0, 0, 10, 10)
        b = _rect(1, 20, 20, 10, 10)
        assert aabb_overlap(a, b) is False

    def test_touching_right_edge_not_overlap(self):
        a = _rect(0, 0, 0, 10, 10)
        b = _rect(1, 10, 0, 10, 10)
        assert aabb_overlap(a, b) is False

    def test_touching_bottom_edge_not_overlap(self):
        a = _rect(0, 0, 0, 10, 10)
        b = _rect(1, 0, 10, 10, 10)
        assert aabb_overlap(a, b) is False

    def test_contained(self):
        a = _rect(0, 0, 0, 20, 20)
        b = _rect(1, 5, 5, 5, 5)
        assert aabb_overlap(a, b) is True

    def test_partial_x_overlap(self):
        a = _rect(0, 0, 0, 10, 10)
        b = _rect(1, 8, 0, 10, 10)
        assert aabb_overlap(a, b) is True

    def test_partial_y_overlap(self):
        a = _rect(0, 0, 0, 10, 10)
        b = _rect(1, 0, 8, 10, 10)
        assert aabb_overlap(a, b) is True

    def test_separated_by_one_pixel(self):
        a = _rect(0, 0, 0, 10, 10)
        b = _rect(1, 11, 0, 10, 10)
        assert aabb_overlap(a, b) is False

    def test_symmetry(self):
        a = _rect(0, 0, 0, 10, 10)
        b = _rect(1, 5, 5, 10, 10)
        assert aabb_overlap(a, b) == aabb_overlap(b, a)


# ─── compute_overlap ──────────────────────────────────────────────────────────

class TestComputeOverlap:
    def test_no_overlap_returns_none(self):
        a = _rect(0, 0, 0, 10, 10)
        b = _rect(1, 20, 0, 10, 10)
        assert compute_overlap(a, b) is None

    def test_touching_returns_none(self):
        a = _rect(0, 0, 0, 10, 10)
        b = _rect(1, 10, 0, 10, 10)
        assert compute_overlap(a, b) is None

    def test_overlap_returns_collision_info(self):
        a = _rect(0, 0, 0, 10, 10)
        b = _rect(1, 5, 0, 10, 10)
        result = compute_overlap(a, b)
        assert isinstance(result, CollisionInfo)

    def test_ids_set_correctly(self):
        a = _rect(3, 0, 0, 10, 10)
        b = _rect(7, 5, 5, 10, 10)
        result = compute_overlap(a, b)
        assert result.id1 == 3
        assert result.id2 == 7

    def test_overlap_width_correct(self):
        a = _rect(0, 0, 0, 10, 10)
        b = _rect(1, 6, 0, 10, 10)
        result = compute_overlap(a, b)
        # a spans [0,10), b spans [6,16) → overlap = 10-6 = 4
        assert result.overlap_w == 4

    def test_overlap_height_correct(self):
        a = _rect(0, 0, 0, 10, 10)
        b = _rect(1, 0, 7, 10, 10)
        result = compute_overlap(a, b)
        assert result.overlap_h == 3

    def test_overlap_area_correct(self):
        a = _rect(0, 0, 0, 10, 10)
        b = _rect(1, 6, 7, 10, 10)
        result = compute_overlap(a, b)
        assert result.overlap_area == 4 * 3

    def test_resolve_vec_nonzero_for_overlap(self):
        a = _rect(0, 0, 0, 10, 10)
        b = _rect(1, 5, 0, 10, 10)
        result = compute_overlap(a, b)
        dx, dy = result.resolve_vec
        assert dx != 0 or dy != 0

    def test_full_containment(self):
        a = _rect(0, 0, 0, 20, 20)
        b = _rect(1, 5, 5, 5, 5)
        result = compute_overlap(a, b)
        assert result is not None
        assert result.overlap_area > 0


# ─── detect_collisions ────────────────────────────────────────────────────────

class TestDetectCollisions:
    def test_no_collisions_empty(self):
        rects = [_rect(0, 0, 0, 5, 5), _rect(1, 10, 0, 5, 5)]
        assert detect_collisions(rects) == []

    def test_one_collision(self):
        rects = [_rect(0, 0, 0, 10, 10), _rect(1, 5, 0, 10, 10)]
        result = detect_collisions(rects)
        assert len(result) == 1

    def test_multiple_collisions(self):
        rects = [
            _rect(0, 0, 0, 10, 10),
            _rect(1, 5, 5, 10, 10),
            _rect(2, 3, 3, 10, 10),
        ]
        result = detect_collisions(rects)
        assert len(result) >= 2

    def test_empty_list(self):
        assert detect_collisions([]) == []

    def test_single_rect(self):
        assert detect_collisions([_rect(0, 0, 0, 10, 10)]) == []

    def test_returns_list_of_collision_infos(self):
        rects = [_rect(0, 0, 0, 10, 10), _rect(1, 5, 0, 10, 10)]
        result = detect_collisions(rects)
        assert isinstance(result[0], CollisionInfo)

    def test_touching_no_collision(self):
        rects = [_rect(0, 0, 0, 10, 10), _rect(1, 10, 0, 10, 10)]
        assert detect_collisions(rects) == []

    def test_collision_pair_i_less_than_j(self):
        rects = [_rect(0, 0, 0, 10, 10), _rect(1, 5, 0, 10, 10)]
        result = detect_collisions(rects)
        # id1 should correspond to index 0 (fid=0), id2 to index 1 (fid=1)
        assert result[0].id1 == 0
        assert result[0].id2 == 1


# ─── collision_graph ──────────────────────────────────────────────────────────

class TestCollisionGraph:
    def test_empty_collisions(self):
        assert collision_graph([]) == {}

    def test_single_collision_bidirectional(self):
        ci = CollisionInfo(id1=0, id2=1, overlap_w=5, overlap_h=5, overlap_area=25)
        g = collision_graph([ci])
        assert 1 in g[0]
        assert 0 in g[1]

    def test_two_collisions_shared_node(self):
        c1 = CollisionInfo(id1=0, id2=1, overlap_w=2, overlap_h=2, overlap_area=4)
        c2 = CollisionInfo(id1=1, id2=2, overlap_w=2, overlap_h=2, overlap_area=4)
        g = collision_graph([c1, c2])
        # Node 1 is adjacent to both 0 and 2
        assert len(g[1]) == 2

    def test_returns_sets(self):
        ci = CollisionInfo(id1=0, id2=1, overlap_w=2, overlap_h=2, overlap_area=4)
        g = collision_graph([ci])
        assert isinstance(g[0], set)

    def test_all_ids_present(self):
        ci = CollisionInfo(id1=3, id2=5, overlap_w=1, overlap_h=1, overlap_area=1)
        g = collision_graph([ci])
        assert 3 in g
        assert 5 in g


# ─── is_collision_free ────────────────────────────────────────────────────────

class TestIsCollisionFree:
    def test_empty_list_free(self):
        assert is_collision_free([]) is True

    def test_single_rect_free(self):
        assert is_collision_free([_rect(0, 0, 0, 10, 10)]) is True

    def test_non_overlapping_free(self):
        rects = [_rect(0, 0, 0, 5, 5), _rect(1, 10, 0, 5, 5)]
        assert is_collision_free(rects) is True

    def test_overlapping_not_free(self):
        rects = [_rect(0, 0, 0, 10, 10), _rect(1, 5, 5, 10, 10)]
        assert is_collision_free(rects) is False

    def test_touching_is_free(self):
        rects = [_rect(0, 0, 0, 10, 10), _rect(1, 10, 0, 10, 10)]
        assert is_collision_free(rects) is True

    def test_three_no_overlap(self):
        rects = [_rect(0, 0, 0, 5, 5), _rect(1, 10, 0, 5, 5), _rect(2, 20, 0, 5, 5)]
        assert is_collision_free(rects) is True

    def test_three_one_overlap(self):
        rects = [_rect(0, 0, 0, 5, 5), _rect(1, 3, 0, 5, 5), _rect(2, 20, 0, 5, 5)]
        assert is_collision_free(rects) is False


# ─── total_overlap_area ───────────────────────────────────────────────────────

class TestTotalOverlapArea:
    def test_empty_returns_zero(self):
        assert total_overlap_area([]) == 0

    def test_single_collision(self):
        ci = CollisionInfo(id1=0, id2=1, overlap_w=5, overlap_h=4, overlap_area=20)
        assert total_overlap_area([ci]) == 20

    def test_multiple_collisions_sum(self):
        c1 = CollisionInfo(id1=0, id2=1, overlap_w=5, overlap_h=4, overlap_area=20)
        c2 = CollisionInfo(id1=0, id2=2, overlap_w=3, overlap_h=3, overlap_area=9)
        assert total_overlap_area([c1, c2]) == 29

    def test_zero_area_collision(self):
        ci = CollisionInfo(id1=0, id2=1, overlap_w=0, overlap_h=0, overlap_area=0)
        assert total_overlap_area([ci]) == 0

    def test_returns_int(self):
        ci = CollisionInfo(id1=0, id2=1, overlap_w=3, overlap_h=3, overlap_area=9)
        result = total_overlap_area([ci])
        assert isinstance(result, int)


# ─── resolve_greedy ───────────────────────────────────────────────────────────

class TestResolveGreedy:
    def test_no_collision_positions_unchanged(self):
        rects = [_rect(0, 0, 0, 5, 5), _rect(1, 10, 0, 5, 5)]
        result = resolve_greedy(rects)
        assert result[0].x == 0
        assert result[1].x == 10

    def test_max_iter_zero_raises(self):
        with pytest.raises(ValueError, match="max_iter"):
            resolve_greedy([_rect(0, 0, 0, 5, 5)], max_iter=0)

    def test_max_iter_negative_raises(self):
        with pytest.raises(ValueError):
            resolve_greedy([_rect(0, 0, 0, 5, 5)], max_iter=-1)

    def test_returns_list(self):
        result = resolve_greedy([_rect(0, 0, 0, 5, 5)])
        assert isinstance(result, list)

    def test_same_length_as_input(self):
        rects = [_rect(0, 0, 0, 5, 5), _rect(1, 10, 0, 5, 5)]
        result = resolve_greedy(rects)
        assert len(result) == 2

    def test_original_not_modified(self):
        r1 = _rect(0, 0, 0, 10, 10)
        r2 = _rect(1, 5, 0, 10, 10)
        orig_x = r1.x
        _ = resolve_greedy([r1, r2])
        assert r1.x == orig_x

    def test_returns_placed_rects(self):
        rects = [_rect(0, 0, 0, 5, 5)]
        result = resolve_greedy(rects)
        assert isinstance(result[0], PlacedRect)

    def test_overlapping_returns_valid_list(self):
        r1 = _rect(0, 0, 0, 10, 10)
        r2 = _rect(1, 5, 0, 10, 10)
        result = resolve_greedy([r1, r2], max_iter=50)
        # Function should run without error and return valid PlacedRects
        assert len(result) == 2
        assert all(isinstance(r, PlacedRect) for r in result)
        assert all(r.x >= 0 and r.y >= 0 for r in result)

    def test_empty_list(self):
        result = resolve_greedy([])
        assert result == []


# ─── batch_detect ─────────────────────────────────────────────────────────────

class TestBatchDetect:
    def test_basic_two_groups(self):
        g1 = [_rect(0, 0, 0, 5, 5), _rect(1, 10, 0, 5, 5)]
        g2 = [_rect(0, 0, 0, 10, 10), _rect(1, 5, 0, 10, 10)]
        result = batch_detect([g1, g2])
        assert len(result) == 2
        assert result[0] == []
        assert len(result[1]) >= 1

    def test_empty_groups(self):
        result = batch_detect([[], []])
        assert result == [[], []]

    def test_empty_outer_list(self):
        result = batch_detect([])
        assert result == []

    def test_returns_list_of_lists(self):
        result = batch_detect([[_rect(0, 0, 0, 5, 5)]])
        assert isinstance(result[0], list)

    def test_single_group_no_collision(self):
        g = [_rect(0, 0, 0, 5, 5), _rect(1, 10, 0, 5, 5)]
        result = batch_detect([g])
        assert result[0] == []

    def test_three_groups(self):
        g1 = [_rect(0, 0, 0, 5, 5)]
        g2 = [_rect(0, 0, 0, 10, 10), _rect(1, 5, 0, 10, 10)]
        g3 = [_rect(0, 0, 0, 3, 3), _rect(1, 1, 1, 3, 3)]
        result = batch_detect([g1, g2, g3])
        assert len(result) == 3
        assert len(result[0]) == 0
        assert len(result[1]) >= 1
        assert len(result[2]) >= 1
