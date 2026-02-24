"""Extra tests for puzzle_reconstruction/utils/scorer_state_records.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.scorer_state_records import (
    AssemblyScoringRecord,
    StateTransitionRecord,
    AdjacencyRecord,
    BeamSearchRecord,
    BinarizationRecord,
    make_assembly_scoring_record,
    make_binarization_record,
)


# ─── AssemblyScoringRecord ────────────────────────────────────────────────────

class TestAssemblyScoringRecordExtra:
    def test_coverage_ratio_full(self):
        r = AssemblyScoringRecord(n_placed=10, n_total=10, total_score=0.9,
                                   geometry_score=0.8, coverage_score=0.7,
                                   seam_score=0.6, uniqueness_score=0.5)
        assert r.coverage_ratio == pytest.approx(1.0)

    def test_coverage_ratio_partial(self):
        r = AssemblyScoringRecord(n_placed=5, n_total=10, total_score=0.5,
                                   geometry_score=0.4, coverage_score=0.3,
                                   seam_score=0.2, uniqueness_score=0.1)
        assert r.coverage_ratio == pytest.approx(0.5)

    def test_coverage_ratio_zero_total(self):
        r = AssemblyScoringRecord(n_placed=0, n_total=0, total_score=0.0,
                                   geometry_score=0.0, coverage_score=0.0,
                                   seam_score=0.0, uniqueness_score=0.0)
        assert r.coverage_ratio == pytest.approx(0.0)

    def test_is_complete_true(self):
        r = AssemblyScoringRecord(n_placed=10, n_total=10, total_score=0.9,
                                   geometry_score=0.8, coverage_score=0.7,
                                   seam_score=0.6, uniqueness_score=0.5)
        assert r.is_complete is True

    def test_is_complete_false(self):
        r = AssemblyScoringRecord(n_placed=5, n_total=10, total_score=0.5,
                                   geometry_score=0.4, coverage_score=0.3,
                                   seam_score=0.2, uniqueness_score=0.1)
        assert r.is_complete is False


# ─── StateTransitionRecord ───────────────────────────────────────────────────

class TestStateTransitionRecordExtra:
    def test_delta_step(self):
        r = StateTransitionRecord(fragment_idx=0, from_step=3, to_step=7,
                                   position=(1.0, 2.0))
        assert r.delta_step == 4

    def test_fields_stored(self):
        r = StateTransitionRecord(fragment_idx=5, from_step=0, to_step=1,
                                   position=(10.0, 20.0), angle=45.0, scale=1.5)
        assert r.angle == pytest.approx(45.0) and r.scale == pytest.approx(1.5)


# ─── AdjacencyRecord ─────────────────────────────────────────────────────────

class TestAdjacencyRecordExtra:
    def test_edge_key_ordered(self):
        r = AdjacencyRecord(idx_a=5, idx_b=2, step=0)
        assert r.edge_key == (2, 5)

    def test_edge_key_same(self):
        r = AdjacencyRecord(idx_a=3, idx_b=3, step=1)
        assert r.edge_key == (3, 3)


# ─── BeamSearchRecord ────────────────────────────────────────────────────────

class TestBeamSearchRecordExtra:
    def test_placement_ratio_full(self):
        r = BeamSearchRecord(n_fragments=10, n_entries=50, beam_width=5,
                              max_depth=10, n_placed=10, final_score=0.9)
        assert r.placement_ratio == pytest.approx(1.0)

    def test_placement_ratio_zero_fragments(self):
        r = BeamSearchRecord(n_fragments=0, n_entries=0, beam_width=1,
                              max_depth=None, n_placed=0, final_score=0.0)
        assert r.placement_ratio == pytest.approx(0.0)


# ─── BinarizationRecord ──────────────────────────────────────────────────────

class TestBinarizationRecordExtra:
    def test_background_ratio(self):
        r = BinarizationRecord(method="otsu", threshold=128,
                                foreground_ratio=0.3)
        assert r.background_ratio == pytest.approx(0.7)

    def test_image_size(self):
        r = BinarizationRecord(method="adaptive", threshold=100,
                                foreground_ratio=0.5,
                                image_height=480, image_width=640)
        assert r.image_size == (480, 640)


# ─── make_assembly_scoring_record ─────────────────────────────────────────────

class TestMakeAssemblyScoringRecordExtra:
    def test_returns_record(self):
        r = make_assembly_scoring_record(5, 10, 0.5)
        assert isinstance(r, AssemblyScoringRecord)

    def test_defaults_zero(self):
        r = make_assembly_scoring_record(5, 10, 0.5)
        assert r.geometry_score == pytest.approx(0.0)


# ─── make_binarization_record ─────────────────────────────────────────────────

class TestMakeBinarizationRecordExtra:
    def test_returns_record(self):
        r = make_binarization_record("otsu", 128, 0.3)
        assert isinstance(r, BinarizationRecord)

    def test_image_shape(self):
        r = make_binarization_record("adaptive", 100, 0.5,
                                      image_shape=(480, 640))
        assert r.image_height == 480 and r.image_width == 640
