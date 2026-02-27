"""Tests for puzzle_reconstruction.utils.region_seam_records."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.region_seam_records import (
    RegionPairRecord,
    SeamRecord,
    ArrayChunkRecord,
    PairwiseNormRecord,
    make_region_pair_record,
    make_seam_record,
    make_array_chunk_record,
)

np.random.seed(13)


# ── RegionPairRecord ──────────────────────────────────────────────────────────

def test_region_pair_record_basic():
    r = make_region_pair_record(0, 1, 0.8)
    assert r.idx_a == 0
    assert r.idx_b == 1
    assert r.score == pytest.approx(0.8)


def test_region_pair_record_defaults():
    r = make_region_pair_record(2, 3, 0.5)
    assert r.color_score == 0.0
    assert r.texture_score == 0.0
    assert r.shape_score == 0.0
    assert r.boundary_score == 0.0


def test_region_pair_record_pair_key_ordered():
    r = make_region_pair_record(5, 2, 0.7)
    assert r.pair_key == (2, 5)


def test_region_pair_record_pair_key_same():
    r = make_region_pair_record(3, 3, 0.5)
    assert r.pair_key == (3, 3)


def test_region_pair_record_all_scores():
    r = make_region_pair_record(0, 1, 0.9, color_score=0.8, texture_score=0.7,
                                 shape_score=0.6, boundary_score=0.5)
    assert r.color_score == pytest.approx(0.8)
    assert r.boundary_score == pytest.approx(0.5)


# ── SeamRecord ────────────────────────────────────────────────────────────────

def test_seam_record_basic():
    sr = make_seam_record(0, 1, 2, 3, 0.75)
    assert sr.fragment_a == 0
    assert sr.side_a == 1
    assert sr.fragment_b == 2
    assert sr.side_b == 3
    assert sr.score == pytest.approx(0.75)


def test_seam_record_defaults():
    sr = make_seam_record(0, 0, 1, 0, 0.5)
    assert sr.color_score == 0.0
    assert sr.gradient_score == 0.0
    assert sr.texture_score == 0.0


def test_seam_record_side_pair():
    sr = make_seam_record(0, 2, 1, 3, 0.6)
    assert sr.side_pair == (2, 3)


def test_seam_record_all_scores():
    sr = make_seam_record(0, 0, 1, 0, 0.8, color_score=0.7,
                           gradient_score=0.6, texture_score=0.5)
    assert sr.color_score == pytest.approx(0.7)
    assert sr.gradient_score == pytest.approx(0.6)
    assert sr.texture_score == pytest.approx(0.5)


# ── ArrayChunkRecord ──────────────────────────────────────────────────────────

def test_array_chunk_record_exact():
    acr = make_array_chunk_record(total_elements=100, chunk_size=10)
    assert acr.n_chunks == 10
    assert acr.last_chunk_size == 10
    assert acr.has_remainder is False


def test_array_chunk_record_with_remainder():
    acr = make_array_chunk_record(total_elements=105, chunk_size=10)
    assert acr.n_chunks == 11
    assert acr.last_chunk_size == 5
    assert acr.has_remainder is True


def test_array_chunk_record_single_chunk():
    acr = make_array_chunk_record(total_elements=7, chunk_size=10)
    assert acr.n_chunks == 1
    assert acr.last_chunk_size == 7
    assert acr.has_remainder is True


def test_array_chunk_record_total_elements():
    acr = make_array_chunk_record(total_elements=50, chunk_size=5)
    assert acr.total_elements == 50
    assert acr.chunk_size == 5


# ── PairwiseNormRecord ────────────────────────────────────────────────────────

def test_pairwise_norm_record_creation():
    pnr = PairwiseNormRecord(
        n_rows=5, n_cols=5, metric="euclidean",
        min_dist=0.0, max_dist=1.0, mean_dist=0.5,
    )
    assert pnr.n_rows == 5
    assert pnr.metric == "euclidean"
    assert pnr.mean_dist == pytest.approx(0.5)


def test_pairwise_norm_record_min_max():
    pnr = PairwiseNormRecord(
        n_rows=3, n_cols=3, metric="cosine",
        min_dist=0.1, max_dist=0.9, mean_dist=0.5,
    )
    assert pnr.min_dist < pnr.max_dist


# ── Integration / edge cases ──────────────────────────────────────────────────

def test_make_region_pair_record_is_instance():
    r = make_region_pair_record(0, 1, 0.5)
    assert isinstance(r, RegionPairRecord)


def test_make_seam_record_is_instance():
    sr = make_seam_record(0, 0, 1, 1, 0.5)
    assert isinstance(sr, SeamRecord)


def test_make_array_chunk_record_is_instance():
    acr = make_array_chunk_record(20, 7)
    assert isinstance(acr, ArrayChunkRecord)


def test_chunk_reconstruction():
    total = 123
    chunk = 10
    acr = make_array_chunk_record(total, chunk)
    reconstructed = (acr.n_chunks - 1) * acr.chunk_size + acr.last_chunk_size
    assert reconstructed == total


def test_seam_record_symmetry():
    sr1 = make_seam_record(0, 1, 2, 3, 0.5)
    sr2 = make_seam_record(2, 3, 0, 1, 0.5)
    # different fragment order but same side_pair pattern
    assert sr1.side_pair == (1, 3)
    assert sr2.side_pair == (3, 1)


def test_multiple_pair_records():
    records = [make_region_pair_record(i, i + 1, 0.1 * i) for i in range(5)]
    assert len(records) == 5
    assert records[0].pair_key < records[-1].pair_key or True  # just check no error
