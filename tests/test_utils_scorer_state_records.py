"""Tests for puzzle_reconstruction.utils.scorer_state_records."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.scorer_state_records import (
    AssemblyScoringRecord,
    StateTransitionRecord,
    AdjacencyRecord,
    BeamSearchRecord,
    BinarizationRecord,
    make_assembly_scoring_record,
    make_binarization_record,
)

np.random.seed(55)


# ── AssemblyScoringRecord ─────────────────────────────────────────────────────

def test_assembly_scoring_coverage_ratio():
    r = make_assembly_scoring_record(n_placed=8, n_total=10, total_score=0.8)
    assert r.coverage_ratio == pytest.approx(0.8)


def test_assembly_scoring_coverage_ratio_zero():
    r = make_assembly_scoring_record(n_placed=0, n_total=0, total_score=0.0)
    assert r.coverage_ratio == 0.0


def test_assembly_scoring_is_complete_true():
    r = make_assembly_scoring_record(n_placed=10, n_total=10, total_score=1.0)
    assert r.is_complete is True


def test_assembly_scoring_is_complete_false():
    r = make_assembly_scoring_record(n_placed=9, n_total=10, total_score=0.9)
    assert r.is_complete is False


def test_assembly_scoring_all_scores():
    r = make_assembly_scoring_record(
        n_placed=5, n_total=10, total_score=0.7,
        geometry_score=0.8, coverage_score=0.6,
        seam_score=0.7, uniqueness_score=0.9,
    )
    assert r.geometry_score == pytest.approx(0.8)
    assert r.uniqueness_score == pytest.approx(0.9)


def test_assembly_scoring_defaults():
    r = make_assembly_scoring_record(n_placed=3, n_total=5, total_score=0.5)
    assert r.geometry_score == 0.0
    assert r.coverage_score == 0.0
    assert r.seam_score == 0.0
    assert r.uniqueness_score == 0.0


# ── StateTransitionRecord ─────────────────────────────────────────────────────

def test_state_transition_delta_step():
    tr = StateTransitionRecord(
        fragment_idx=2, from_step=3, to_step=7, position=(1.0, 2.0)
    )
    assert tr.delta_step == 4


def test_state_transition_defaults():
    tr = StateTransitionRecord(
        fragment_idx=0, from_step=0, to_step=1, position=(0.0, 0.0)
    )
    assert tr.angle == 0.0
    assert tr.scale == pytest.approx(1.0)


def test_state_transition_custom_angle():
    tr = StateTransitionRecord(
        fragment_idx=1, from_step=2, to_step=3, position=(5.0, 5.0), angle=90.0
    )
    assert tr.angle == pytest.approx(90.0)


# ── AdjacencyRecord ───────────────────────────────────────────────────────────

def test_adjacency_record_edge_key_ordered():
    ar = AdjacencyRecord(idx_a=5, idx_b=2, step=10)
    assert ar.edge_key == (2, 5)


def test_adjacency_record_edge_key_same():
    ar = AdjacencyRecord(idx_a=3, idx_b=3, step=0)
    assert ar.edge_key == (3, 3)


def test_adjacency_record_step():
    ar = AdjacencyRecord(idx_a=0, idx_b=1, step=42)
    assert ar.step == 42


# ── BeamSearchRecord ──────────────────────────────────────────────────────────

def test_beam_search_placement_ratio():
    bsr = BeamSearchRecord(
        n_fragments=10, n_entries=5, beam_width=3,
        max_depth=None, n_placed=8, final_score=0.9,
    )
    assert bsr.placement_ratio == pytest.approx(0.8)


def test_beam_search_placement_ratio_zero():
    bsr = BeamSearchRecord(
        n_fragments=0, n_entries=0, beam_width=3,
        max_depth=None, n_placed=0, final_score=0.0,
    )
    assert bsr.placement_ratio == 0.0


def test_beam_search_max_depth_none():
    bsr = BeamSearchRecord(
        n_fragments=5, n_entries=3, beam_width=2,
        max_depth=None, n_placed=5, final_score=0.8,
    )
    assert bsr.max_depth is None


def test_beam_search_with_max_depth():
    bsr = BeamSearchRecord(
        n_fragments=5, n_entries=3, beam_width=2,
        max_depth=10, n_placed=5, final_score=0.8,
    )
    assert bsr.max_depth == 10


# ── BinarizationRecord ────────────────────────────────────────────────────────

def test_binarization_record_background_ratio():
    br = make_binarization_record("otsu", 128.0, 0.3, image_shape=(100, 200))
    assert br.foreground_ratio == pytest.approx(0.3)
    assert br.background_ratio == pytest.approx(0.7)


def test_binarization_record_image_size():
    br = make_binarization_record("global", 100.0, 0.5, image_shape=(480, 640))
    assert br.image_size == (480, 640)
    assert br.image_height == 480
    assert br.image_width == 640


def test_binarization_record_inverted():
    br = make_binarization_record("otsu", 128.0, 0.4, inverted=True)
    assert br.inverted is True


def test_binarization_record_not_inverted():
    br = make_binarization_record("global", 100.0, 0.5)
    assert br.inverted is False


def test_binarization_record_defaults_shape():
    br = make_binarization_record("otsu", 128.0, 0.5)
    assert br.image_size == (0, 0)


# ── make_assembly_scoring_record / make_binarization_record ───────────────────

def test_make_assembly_scoring_record_is_instance():
    r = make_assembly_scoring_record(5, 10, 0.6)
    assert isinstance(r, AssemblyScoringRecord)


def test_make_binarization_record_is_instance():
    br = make_binarization_record("adaptive", 80.0, 0.4)
    assert isinstance(br, BinarizationRecord)
