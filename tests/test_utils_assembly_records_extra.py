"""Extra tests for puzzle_reconstruction/utils/assembly_records.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.assembly_records import (
    CollisionRecord,
    CostMatrixRecord,
    FragmentScoreRecord,
    AssemblyScoreRecord,
    OverlapRecord,
    ResolveRecord,
    make_collision_record,
    make_cost_matrix_record,
    make_overlap_record,
)


# ─── CollisionRecord ──────────────────────────────────────────────────────────

class TestCollisionRecordExtra:
    def test_stores_id1(self):
        r = CollisionRecord(id1=3, id2=5, overlap_w=10, overlap_h=8, overlap_area=80)
        assert r.id1 == 3

    def test_stores_id2(self):
        r = CollisionRecord(id1=1, id2=7, overlap_w=5, overlap_h=4, overlap_area=20)
        assert r.id2 == 7

    def test_stores_overlap_area(self):
        r = CollisionRecord(id1=0, id2=1, overlap_w=10, overlap_h=10, overlap_area=100)
        assert r.overlap_area == 100

    def test_pair_key_ordered(self):
        r = CollisionRecord(id1=5, id2=2, overlap_w=0, overlap_h=0, overlap_area=0)
        assert r.pair_key == (2, 5)

    def test_pair_key_already_ordered(self):
        r = CollisionRecord(id1=1, id2=9, overlap_w=0, overlap_h=0, overlap_area=0)
        assert r.pair_key == (1, 9)

    def test_stores_overlap_dimensions(self):
        r = CollisionRecord(id1=0, id2=1, overlap_w=12, overlap_h=7, overlap_area=84)
        assert r.overlap_w == 12
        assert r.overlap_h == 7


# ─── CostMatrixRecord ─────────────────────────────────────────────────────────

class TestCostMatrixRecordExtra:
    def test_stores_n_fragments(self):
        r = CostMatrixRecord(n_fragments=10, method="greedy",
                              min_cost=0.0, max_cost=1.0, mean_cost=0.5)
        assert r.n_fragments == 10

    def test_stores_method(self):
        r = CostMatrixRecord(n_fragments=5, method="hungarian",
                              min_cost=0.0, max_cost=1.0, mean_cost=0.5)
        assert r.method == "hungarian"

    def test_stores_costs(self):
        r = CostMatrixRecord(n_fragments=3, method="m",
                              min_cost=0.1, max_cost=0.9, mean_cost=0.4)
        assert r.min_cost == pytest.approx(0.1)
        assert r.max_cost == pytest.approx(0.9)
        assert r.mean_cost == pytest.approx(0.4)

    def test_default_n_forbidden(self):
        r = CostMatrixRecord(n_fragments=5, method="m",
                              min_cost=0.0, max_cost=1.0, mean_cost=0.5)
        assert r.n_forbidden == 0

    def test_custom_n_forbidden(self):
        r = CostMatrixRecord(n_fragments=5, method="m",
                              min_cost=0.0, max_cost=1.0, mean_cost=0.5,
                              n_forbidden=3)
        assert r.n_forbidden == 3


# ─── FragmentScoreRecord ──────────────────────────────────────────────────────

class TestFragmentScoreRecordExtra:
    def test_stores_fragment_idx(self):
        r = FragmentScoreRecord(fragment_idx=2, local_score=0.8,
                                 n_neighbors=3, is_reliable=True)
        assert r.fragment_idx == 2

    def test_stores_local_score(self):
        r = FragmentScoreRecord(fragment_idx=0, local_score=0.65,
                                 n_neighbors=2, is_reliable=False)
        assert r.local_score == pytest.approx(0.65)

    def test_stores_n_neighbors(self):
        r = FragmentScoreRecord(fragment_idx=0, local_score=0.5,
                                 n_neighbors=4, is_reliable=True)
        assert r.n_neighbors == 4

    def test_stores_is_reliable(self):
        r = FragmentScoreRecord(fragment_idx=0, local_score=0.5,
                                 n_neighbors=2, is_reliable=False)
        assert r.is_reliable is False


# ─── AssemblyScoreRecord ──────────────────────────────────────────────────────

class TestAssemblyScoreRecordExtra:
    def test_stores_global_score(self):
        r = AssemblyScoreRecord(global_score=0.75, coverage=0.9,
                                 mean_local=0.7, n_placed=8, n_reliable=6)
        assert r.global_score == pytest.approx(0.75)

    def test_stores_coverage(self):
        r = AssemblyScoreRecord(global_score=0.8, coverage=0.95,
                                 mean_local=0.7, n_placed=10, n_reliable=8)
        assert r.coverage == pytest.approx(0.95)

    def test_stores_n_placed(self):
        r = AssemblyScoreRecord(global_score=0.8, coverage=0.9,
                                 mean_local=0.7, n_placed=5, n_reliable=4)
        assert r.n_placed == 5

    def test_default_fragment_scores_empty(self):
        r = AssemblyScoreRecord(global_score=0.8, coverage=0.9,
                                 mean_local=0.7, n_placed=5, n_reliable=4)
        assert r.fragment_scores == {}

    def test_custom_fragment_scores(self):
        fs = FragmentScoreRecord(fragment_idx=0, local_score=0.8,
                                  n_neighbors=2, is_reliable=True)
        r = AssemblyScoreRecord(global_score=0.8, coverage=0.9,
                                 mean_local=0.7, n_placed=1, n_reliable=1,
                                 fragment_scores={0: fs})
        assert 0 in r.fragment_scores


# ─── OverlapRecord ────────────────────────────────────────────────────────────

class TestOverlapRecordExtra:
    def test_stores_id_a(self):
        r = OverlapRecord(id_a=3, id_b=5, area=10.0, dx=2.0, dy=1.0)
        assert r.id_a == 3

    def test_stores_id_b(self):
        r = OverlapRecord(id_a=1, id_b=7, area=5.0, dx=0.0, dy=0.0)
        assert r.id_b == 7

    def test_stores_area(self):
        r = OverlapRecord(id_a=0, id_b=1, area=15.5, dx=0.0, dy=0.0)
        assert r.area == pytest.approx(15.5)

    def test_pair_key_ordered(self):
        r = OverlapRecord(id_a=7, id_b=2, area=0.0, dx=0.0, dy=0.0)
        assert r.pair_key == (2, 7)

    def test_pair_key_already_ordered(self):
        r = OverlapRecord(id_a=1, id_b=5, area=0.0, dx=0.0, dy=0.0)
        assert r.pair_key == (1, 5)

    def test_has_overlap_true(self):
        r = OverlapRecord(id_a=0, id_b=1, area=5.0, dx=1.0, dy=0.0)
        assert r.has_overlap is True

    def test_has_overlap_false(self):
        r = OverlapRecord(id_a=0, id_b=1, area=0.0, dx=0.0, dy=0.0)
        assert r.has_overlap is False

    def test_stores_dx_dy(self):
        r = OverlapRecord(id_a=0, id_b=1, area=1.0, dx=3.5, dy=-2.0)
        assert r.dx == pytest.approx(3.5)
        assert r.dy == pytest.approx(-2.0)


# ─── ResolveRecord ────────────────────────────────────────────────────────────

class TestResolveRecordExtra:
    def test_stores_n_iter(self):
        r = ResolveRecord(n_iter=10, resolved=True, final_n_overlaps=0, n_fragments=5)
        assert r.n_iter == 10

    def test_stores_resolved(self):
        r = ResolveRecord(n_iter=5, resolved=False, final_n_overlaps=2, n_fragments=3)
        assert r.resolved is False

    def test_stores_final_n_overlaps(self):
        r = ResolveRecord(n_iter=3, resolved=True, final_n_overlaps=0, n_fragments=4)
        assert r.final_n_overlaps == 0

    def test_stores_n_fragments(self):
        r = ResolveRecord(n_iter=1, resolved=True, final_n_overlaps=0, n_fragments=8)
        assert r.n_fragments == 8


# ─── make_collision_record ────────────────────────────────────────────────────

class TestMakeCollisionRecordExtra:
    def test_returns_collision_record(self):
        r = make_collision_record(0, 1)
        assert isinstance(r, CollisionRecord)

    def test_stores_ids(self):
        r = make_collision_record(3, 7)
        assert r.id1 == 3 and r.id2 == 7

    def test_default_zeros(self):
        r = make_collision_record(0, 1)
        assert r.overlap_area == 0

    def test_custom_area(self):
        r = make_collision_record(0, 1, overlap_w=5, overlap_h=5, overlap_area=25)
        assert r.overlap_area == 25


# ─── make_cost_matrix_record ──────────────────────────────────────────────────

class TestMakeCostMatrixRecordExtra:
    def test_returns_cost_matrix_record(self):
        r = make_cost_matrix_record(5, "greedy")
        assert isinstance(r, CostMatrixRecord)

    def test_stores_n_fragments(self):
        r = make_cost_matrix_record(8, "m")
        assert r.n_fragments == 8

    def test_stores_method(self):
        r = make_cost_matrix_record(3, "hungarian")
        assert r.method == "hungarian"

    def test_default_costs(self):
        r = make_cost_matrix_record(5, "m")
        assert r.min_cost == pytest.approx(0.0)
        assert r.max_cost == pytest.approx(1.0)

    def test_custom_costs(self):
        r = make_cost_matrix_record(5, "m", min_cost=0.1, max_cost=0.9, mean_cost=0.4)
        assert r.min_cost == pytest.approx(0.1)
        assert r.mean_cost == pytest.approx(0.4)


# ─── make_overlap_record ──────────────────────────────────────────────────────

class TestMakeOverlapRecordExtra:
    def test_returns_overlap_record(self):
        r = make_overlap_record(0, 1)
        assert isinstance(r, OverlapRecord)

    def test_stores_ids(self):
        r = make_overlap_record(2, 4)
        assert r.id_a == 2 and r.id_b == 4

    def test_default_area_zero(self):
        r = make_overlap_record(0, 1)
        assert r.area == pytest.approx(0.0)

    def test_custom_area(self):
        r = make_overlap_record(0, 1, area=12.5)
        assert r.area == pytest.approx(12.5)

    def test_custom_dx_dy(self):
        r = make_overlap_record(0, 1, dx=3.0, dy=-1.5)
        assert r.dx == pytest.approx(3.0)
        assert r.dy == pytest.approx(-1.5)

    def test_has_overlap_false(self):
        r = make_overlap_record(0, 1)
        assert r.has_overlap is False

    def test_has_overlap_true(self):
        r = make_overlap_record(0, 1, area=5.0)
        assert r.has_overlap is True
