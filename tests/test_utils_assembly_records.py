"""Tests for puzzle_reconstruction.utils.assembly_records"""
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

def test_collision_record_basic():
    r = CollisionRecord(id1=1, id2=3, overlap_w=10, overlap_h=5, overlap_area=50)
    assert r.id1 == 1
    assert r.id2 == 3
    assert r.overlap_area == 50


def test_collision_record_pair_key_ordering():
    r = CollisionRecord(id1=5, id2=2, overlap_w=0, overlap_h=0, overlap_area=0)
    assert r.pair_key == (2, 5)


def test_collision_record_pair_key_same_order():
    r = CollisionRecord(id1=1, id2=4, overlap_w=0, overlap_h=0, overlap_area=0)
    assert r.pair_key == (1, 4)


# ─── CostMatrixRecord ─────────────────────────────────────────────────────────

def test_cost_matrix_record_basic():
    r = CostMatrixRecord(n_fragments=5, method="greedy", min_cost=0.0,
                         max_cost=1.0, mean_cost=0.5)
    assert r.n_fragments == 5
    assert r.method == "greedy"
    assert r.n_forbidden == 0


def test_cost_matrix_record_with_forbidden():
    r = CostMatrixRecord(n_fragments=3, method="sa", min_cost=0.1,
                         max_cost=0.9, mean_cost=0.5, n_forbidden=2)
    assert r.n_forbidden == 2


# ─── FragmentScoreRecord ──────────────────────────────────────────────────────

def test_fragment_score_record_basic():
    r = FragmentScoreRecord(fragment_idx=0, local_score=0.8,
                            n_neighbors=3, is_reliable=True)
    assert r.fragment_idx == 0
    assert r.local_score == pytest.approx(0.8)
    assert r.is_reliable is True


def test_fragment_score_record_unreliable():
    r = FragmentScoreRecord(fragment_idx=2, local_score=0.1,
                            n_neighbors=0, is_reliable=False)
    assert r.is_reliable is False


# ─── AssemblyScoreRecord ──────────────────────────────────────────────────────

def test_assembly_score_record_basic():
    r = AssemblyScoreRecord(global_score=0.75, coverage=0.9,
                            mean_local=0.7, n_placed=5, n_reliable=4)
    assert r.global_score == pytest.approx(0.75)
    assert r.coverage == pytest.approx(0.9)
    assert r.n_placed == 5


def test_assembly_score_record_fragment_scores_default_empty():
    r = AssemblyScoreRecord(global_score=0.0, coverage=0.0,
                            mean_local=0.0, n_placed=0, n_reliable=0)
    assert r.fragment_scores == {}


def test_assembly_score_record_with_fragment_scores():
    fsr = FragmentScoreRecord(0, 0.9, 2, True)
    r = AssemblyScoreRecord(global_score=0.9, coverage=1.0,
                            mean_local=0.9, n_placed=1, n_reliable=1,
                            fragment_scores={0: fsr})
    assert 0 in r.fragment_scores


# ─── OverlapRecord ────────────────────────────────────────────────────────────

def test_overlap_record_has_overlap_true():
    r = OverlapRecord(id_a=0, id_b=1, area=25.0, dx=5.0, dy=5.0)
    assert r.has_overlap is True


def test_overlap_record_has_overlap_false():
    r = OverlapRecord(id_a=0, id_b=1, area=0.0, dx=0.0, dy=0.0)
    assert r.has_overlap is False


def test_overlap_record_pair_key():
    r = OverlapRecord(id_a=3, id_b=1, area=1.0, dx=0.0, dy=0.0)
    assert r.pair_key == (1, 3)


def test_overlap_record_pair_key_already_ordered():
    r = OverlapRecord(id_a=2, id_b=7, area=0.0, dx=0.0, dy=0.0)
    assert r.pair_key == (2, 7)


# ─── ResolveRecord ────────────────────────────────────────────────────────────

def test_resolve_record_basic():
    r = ResolveRecord(n_iter=10, resolved=True, final_n_overlaps=0, n_fragments=5)
    assert r.n_iter == 10
    assert r.resolved is True
    assert r.final_n_overlaps == 0


def test_resolve_record_unresolved():
    r = ResolveRecord(n_iter=100, resolved=False, final_n_overlaps=3, n_fragments=8)
    assert r.resolved is False
    assert r.final_n_overlaps == 3


# ─── make_collision_record ────────────────────────────────────────────────────

def test_make_collision_record_defaults():
    r = make_collision_record(0, 1)
    assert r.id1 == 0
    assert r.id2 == 1
    assert r.overlap_w == 0
    assert r.overlap_h == 0
    assert r.overlap_area == 0


def test_make_collision_record_custom():
    r = make_collision_record(2, 3, overlap_w=10, overlap_h=5, overlap_area=50)
    assert r.overlap_area == 50


# ─── make_cost_matrix_record ──────────────────────────────────────────────────

def test_make_cost_matrix_record_defaults():
    r = make_cost_matrix_record(4, "greedy")
    assert r.n_fragments == 4
    assert r.method == "greedy"
    assert r.min_cost == pytest.approx(0.0)
    assert r.max_cost == pytest.approx(1.0)
    assert r.mean_cost == pytest.approx(0.5)


def test_make_cost_matrix_record_custom():
    r = make_cost_matrix_record(10, "sa", 0.1, 0.9, 0.4, n_forbidden=3)
    assert r.n_forbidden == 3


# ─── make_overlap_record ──────────────────────────────────────────────────────

def test_make_overlap_record_defaults():
    r = make_overlap_record(0, 1)
    assert r.area == pytest.approx(0.0)
    assert r.dx == pytest.approx(0.0)
    assert r.dy == pytest.approx(0.0)


def test_make_overlap_record_custom():
    r = make_overlap_record(1, 2, area=5.0, dx=2.0, dy=3.0)
    assert r.area == pytest.approx(5.0)
    assert r.has_overlap is True


def test_make_overlap_record_pair_key():
    r = make_overlap_record(5, 2)
    assert r.pair_key == (2, 5)
