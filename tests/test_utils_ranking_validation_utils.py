"""Tests for puzzle_reconstruction.utils.ranking_validation_utils."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.ranking_validation_utils import (
    RankingRunRecord,
    CandidateSummary,
    ScoreVectorRecord,
    ValidationRunRecord,
    BoundaryCheckSummary,
    PaletteComparisonRecord,
    PaletteRankingRecord,
    make_ranking_record,
    make_validation_record,
)

np.random.seed(7)


# ── RankingRunRecord ──────────────────────────────────────────────────────────

def test_ranking_run_record_basic():
    r = make_ranking_record(10, 45, top_score=0.9, label="run1")
    assert r.n_fragments == 10
    assert r.n_pairs_ranked == 45
    assert r.top_score == pytest.approx(0.9)
    assert r.label == "run1"


def test_ranking_run_record_has_results_true():
    r = make_ranking_record(5, 3)
    assert r.has_results is True


def test_ranking_run_record_has_results_false():
    r = make_ranking_record(5, 0)
    assert r.has_results is False


def test_ranking_run_record_negative_fragments_raises():
    with pytest.raises(ValueError):
        RankingRunRecord(n_fragments=-1, n_pairs_ranked=0)


def test_ranking_run_record_invalid_top_score_raises():
    with pytest.raises(ValueError):
        RankingRunRecord(n_fragments=1, n_pairs_ranked=0, top_score=1.5)


# ── CandidateSummary ──────────────────────────────────────────────────────────

def test_candidate_summary_basic():
    cs = CandidateSummary(fragment_id=3, n_candidates=5, best_score=0.7, best_partner=2)
    assert cs.fragment_id == 3
    assert cs.has_candidates is True


def test_candidate_summary_no_candidates():
    cs = CandidateSummary(fragment_id=0, n_candidates=0)
    assert cs.has_candidates is False


def test_candidate_summary_invalid_fragment_id():
    with pytest.raises(ValueError):
        CandidateSummary(fragment_id=-1, n_candidates=1)


def test_candidate_summary_invalid_best_score():
    with pytest.raises(ValueError):
        CandidateSummary(fragment_id=0, n_candidates=1, best_score=2.0)


# ── ScoreVectorRecord ─────────────────────────────────────────────────────────

def test_score_vector_record_max_mean():
    svr = ScoreVectorRecord(n_fragments=3, scores=[0.1, 0.5, 0.9])
    assert svr.max_score == pytest.approx(0.9)
    assert svr.mean_score == pytest.approx(0.5)


def test_score_vector_record_empty_scores():
    svr = ScoreVectorRecord(n_fragments=3, scores=[])
    assert svr.max_score == 0.0
    assert svr.mean_score == 0.0


def test_score_vector_record_wrong_length_raises():
    with pytest.raises(ValueError):
        ScoreVectorRecord(n_fragments=3, scores=[0.1, 0.2])  # neither 0 nor 3


# ── ValidationRunRecord ───────────────────────────────────────────────────────

def test_validation_run_record_basic():
    vr = make_validation_record(step=2, n_pairs=10, n_violations=3, quality_score=0.8)
    assert vr.step == 2
    assert vr.n_violations == 3
    assert vr.quality_score == pytest.approx(0.8)


def test_validation_run_record_violation_rate():
    vr = make_validation_record(step=0, n_pairs=10, n_violations=2, quality_score=0.9)
    assert vr.violation_rate == pytest.approx(0.2)


def test_validation_run_record_zero_pairs():
    vr = make_validation_record(step=0, n_pairs=0, n_violations=0, quality_score=1.0)
    assert vr.violation_rate == 0.0


def test_validation_run_record_is_clean_true():
    vr = make_validation_record(step=0, n_pairs=5, n_violations=0, quality_score=1.0)
    assert vr.is_clean is True


def test_validation_run_record_is_clean_false():
    vr = make_validation_record(step=0, n_pairs=5, n_violations=1, quality_score=0.9)
    assert vr.is_clean is False


def test_validation_run_record_invalid_quality():
    with pytest.raises(ValueError):
        ValidationRunRecord(step=0, n_pairs=1, n_violations=0, quality_score=1.5)


# ── BoundaryCheckSummary ──────────────────────────────────────────────────────

def test_boundary_check_summary_dominant_violation():
    bcs = BoundaryCheckSummary(
        n_assemblies=3, mean_quality=0.8,
        violation_types={"gap": 5, "overlap": 2},
    )
    assert bcs.dominant_violation == "gap"


def test_boundary_check_summary_no_violations():
    bcs = BoundaryCheckSummary(n_assemblies=2, mean_quality=1.0)
    assert bcs.dominant_violation is None


# ── PaletteComparisonRecord ───────────────────────────────────────────────────

def test_palette_comparison_similarity():
    pcr = PaletteComparisonRecord(fragment_id_a=0, fragment_id_b=1, distance=0.0, n_colors=8)
    assert pcr.similarity == pytest.approx(1.0)


def test_palette_comparison_similarity_partial():
    pcr = PaletteComparisonRecord(fragment_id_a=0, fragment_id_b=1, distance=127.5, n_colors=8)
    assert 0.0 < pcr.similarity < 1.0


def test_palette_comparison_negative_distance_raises():
    with pytest.raises(ValueError):
        PaletteComparisonRecord(fragment_id_a=0, fragment_id_b=1, distance=-1.0, n_colors=4)


# ── PaletteRankingRecord ──────────────────────────────────────────────────────

def test_palette_ranking_record_top_k():
    prr = PaletteRankingRecord(query_id=0, ranked_ids=[1, 2, 3], similarities=[0.9, 0.7, 0.5])
    top = prr.top_k(2)
    assert len(top) == 2
    assert top[0] == (1, 0.9)


def test_palette_ranking_record_top_k_exceeds():
    prr = PaletteRankingRecord(query_id=0, ranked_ids=[1], similarities=[0.8])
    top = prr.top_k(10)
    assert len(top) == 1


# ── make_ranking_record / make_validation_record ──────────────────────────────

def test_make_ranking_record_defaults():
    r = make_ranking_record(5, 10)
    assert r.top_score == 0.0
    assert r.label == ""


def test_make_validation_record_fields():
    vr = make_validation_record(1, 20, 4, 0.6)
    assert vr.n_pairs == 20
    assert vr.n_violations == 4
