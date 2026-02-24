"""Extra tests for puzzle_reconstruction/utils/region_seam_records.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.region_seam_records import (
    RegionPairRecord,
    SeamRecord,
    ArrayChunkRecord,
    PairwiseNormRecord,
    make_region_pair_record,
    make_seam_record,
    make_array_chunk_record,
)


# ─── RegionPairRecord ─────────────────────────────────────────────────────────

class TestRegionPairRecordExtra:
    def test_pair_key_ordered(self):
        r = RegionPairRecord(idx_a=5, idx_b=2, score=0.8,
                              color_score=0.7, texture_score=0.6,
                              shape_score=0.5, boundary_score=0.4)
        assert r.pair_key == (2, 5)

    def test_pair_key_already_ordered(self):
        r = RegionPairRecord(idx_a=1, idx_b=3, score=0.8,
                              color_score=0.7, texture_score=0.6,
                              shape_score=0.5, boundary_score=0.4)
        assert r.pair_key == (1, 3)

    def test_fields_stored(self):
        r = RegionPairRecord(idx_a=0, idx_b=1, score=0.9,
                              color_score=0.8, texture_score=0.7,
                              shape_score=0.6, boundary_score=0.5)
        assert r.score == pytest.approx(0.9)
        assert r.color_score == pytest.approx(0.8)

    def test_pair_key_same_index(self):
        r = RegionPairRecord(idx_a=3, idx_b=3, score=0.5,
                              color_score=0.0, texture_score=0.0,
                              shape_score=0.0, boundary_score=0.0)
        assert r.pair_key == (3, 3)


# ─── SeamRecord ───────────────────────────────────────────────────────────────

class TestSeamRecordExtra:
    def test_side_pair_property(self):
        r = SeamRecord(fragment_a=0, side_a=1, fragment_b=2, side_b=3,
                       score=0.8, color_score=0.7,
                       gradient_score=0.6, texture_score=0.5)
        assert r.side_pair == (1, 3)

    def test_fields_stored(self):
        r = SeamRecord(fragment_a=10, side_a=0, fragment_b=20, side_b=2,
                       score=0.9, color_score=0.8,
                       gradient_score=0.7, texture_score=0.6)
        assert r.fragment_a == 10 and r.fragment_b == 20
        assert r.gradient_score == pytest.approx(0.7)


# ─── ArrayChunkRecord ────────────────────────────────────────────────────────

class TestArrayChunkRecordExtra:
    def test_has_remainder_true(self):
        r = ArrayChunkRecord(total_elements=10, chunk_size=3,
                              n_chunks=4, last_chunk_size=1)
        assert r.has_remainder is True

    def test_has_remainder_false(self):
        r = ArrayChunkRecord(total_elements=9, chunk_size=3,
                              n_chunks=3, last_chunk_size=3)
        assert r.has_remainder is False


# ─── PairwiseNormRecord ──────────────────────────────────────────────────────

class TestPairwiseNormRecordExtra:
    def test_fields_stored(self):
        r = PairwiseNormRecord(n_rows=5, n_cols=5, metric="l2",
                                min_dist=0.1, max_dist=10.0, mean_dist=3.5)
        assert r.metric == "l2"
        assert r.min_dist == pytest.approx(0.1)
        assert r.max_dist == pytest.approx(10.0)


# ─── make_region_pair_record ──────────────────────────────────────────────────

class TestMakeRegionPairRecordExtra:
    def test_returns_record(self):
        r = make_region_pair_record(0, 1, 0.8)
        assert isinstance(r, RegionPairRecord)

    def test_defaults_zero(self):
        r = make_region_pair_record(0, 1, 0.5)
        assert r.color_score == pytest.approx(0.0)
        assert r.texture_score == pytest.approx(0.0)

    def test_custom_scores(self):
        r = make_region_pair_record(0, 1, 0.8, color_score=0.7,
                                     shape_score=0.6)
        assert r.color_score == pytest.approx(0.7)
        assert r.shape_score == pytest.approx(0.6)


# ─── make_seam_record ────────────────────────────────────────────────────────

class TestMakeSeamRecordExtra:
    def test_returns_record(self):
        r = make_seam_record(0, 1, 2, 3, 0.9)
        assert isinstance(r, SeamRecord)

    def test_fields_assigned(self):
        r = make_seam_record(10, 0, 20, 2, 0.85,
                              color_score=0.7, gradient_score=0.6)
        assert r.fragment_a == 10 and r.side_b == 2
        assert r.color_score == pytest.approx(0.7)


# ─── make_array_chunk_record ──────────────────────────────────────────────────

class TestMakeArrayChunkRecordExtra:
    def test_exact_division(self):
        r = make_array_chunk_record(9, 3)
        assert r.n_chunks == 3 and r.last_chunk_size == 3

    def test_with_remainder(self):
        r = make_array_chunk_record(10, 3)
        assert r.n_chunks == 4 and r.last_chunk_size == 1

    def test_single_chunk(self):
        r = make_array_chunk_record(5, 10)
        assert r.n_chunks == 1 and r.last_chunk_size == 5

    def test_total_elements_stored(self):
        r = make_array_chunk_record(100, 7)
        assert r.total_elements == 100 and r.chunk_size == 7
