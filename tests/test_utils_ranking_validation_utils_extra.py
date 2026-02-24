"""Extra tests for puzzle_reconstruction/utils/ranking_validation_utils.py."""
from __future__ import annotations

import pytest

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


# ─── RankingRunRecord ─────────────────────────────────────────────────────────

class TestRankingRunRecordExtra:
    def test_negative_n_fragments_raises(self):
        with pytest.raises(ValueError):
            RankingRunRecord(n_fragments=-1, n_pairs_ranked=0)

    def test_top_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            RankingRunRecord(n_fragments=5, n_pairs_ranked=3, top_score=1.5)

    def test_has_results_true(self):
        r = RankingRunRecord(n_fragments=5, n_pairs_ranked=3)
        assert r.has_results is True

    def test_has_results_false(self):
        r = RankingRunRecord(n_fragments=5, n_pairs_ranked=0)
        assert r.has_results is False

    def test_label_stored(self):
        r = RankingRunRecord(n_fragments=5, n_pairs_ranked=3, label="run1")
        assert r.label == "run1"


# ─── CandidateSummary ────────────────────────────────────────────────────────

class TestCandidateSummaryExtra:
    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            CandidateSummary(fragment_id=-1, n_candidates=3)

    def test_best_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            CandidateSummary(fragment_id=0, n_candidates=3, best_score=1.5)

    def test_has_candidates_true(self):
        c = CandidateSummary(fragment_id=0, n_candidates=5)
        assert c.has_candidates is True

    def test_has_candidates_false(self):
        c = CandidateSummary(fragment_id=0, n_candidates=0)
        assert c.has_candidates is False

    def test_best_partner_none_default(self):
        c = CandidateSummary(fragment_id=0, n_candidates=3)
        assert c.best_partner is None


# ─── ScoreVectorRecord ────────────────────────────────────────────────────────

class TestScoreVectorRecordExtra:
    def test_empty_scores_ok(self):
        r = ScoreVectorRecord(n_fragments=5, scores=[])
        assert r.max_score == pytest.approx(0.0)

    def test_scores_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            ScoreVectorRecord(n_fragments=3, scores=[0.5, 0.6])

    def test_max_score(self):
        r = ScoreVectorRecord(n_fragments=3, scores=[0.3, 0.9, 0.5])
        assert r.max_score == pytest.approx(0.9)

    def test_mean_score(self):
        r = ScoreVectorRecord(n_fragments=2, scores=[0.4, 0.8])
        assert r.mean_score == pytest.approx(0.6)

    def test_mean_score_empty(self):
        r = ScoreVectorRecord(n_fragments=0, scores=[])
        assert r.mean_score == pytest.approx(0.0)


# ─── ValidationRunRecord ─────────────────────────────────────────────────────

class TestValidationRunRecordExtra:
    def test_quality_out_of_range_raises(self):
        with pytest.raises(ValueError):
            ValidationRunRecord(step=0, n_pairs=5, n_violations=1,
                                quality_score=1.5)

    def test_violation_rate(self):
        r = ValidationRunRecord(step=0, n_pairs=10, n_violations=2,
                                quality_score=0.8)
        assert r.violation_rate == pytest.approx(0.2)

    def test_violation_rate_zero_pairs(self):
        r = ValidationRunRecord(step=0, n_pairs=0, n_violations=0,
                                quality_score=1.0)
        assert r.violation_rate == pytest.approx(0.0)

    def test_is_clean_true(self):
        r = ValidationRunRecord(step=0, n_pairs=5, n_violations=0,
                                quality_score=1.0)
        assert r.is_clean is True

    def test_is_clean_false(self):
        r = ValidationRunRecord(step=0, n_pairs=5, n_violations=1,
                                quality_score=0.8)
        assert r.is_clean is False


# ─── BoundaryCheckSummary ─────────────────────────────────────────────────────

class TestBoundaryCheckSummaryExtra:
    def test_dominant_violation_none(self):
        s = BoundaryCheckSummary(n_assemblies=3, mean_quality=0.7)
        assert s.dominant_violation is None

    def test_dominant_violation(self):
        s = BoundaryCheckSummary(n_assemblies=3, mean_quality=0.7,
                                  violation_types={"color": 5, "shape": 2})
        assert s.dominant_violation == "color"

    def test_fields_stored(self):
        s = BoundaryCheckSummary(n_assemblies=5, mean_quality=0.8,
                                  best_quality=0.95, worst_quality=0.5)
        assert s.best_quality == pytest.approx(0.95)
        assert s.worst_quality == pytest.approx(0.5)


# ─── PaletteComparisonRecord ──────────────────────────────────────────────────

class TestPaletteComparisonRecordExtra:
    def test_negative_distance_raises(self):
        with pytest.raises(ValueError):
            PaletteComparisonRecord(fragment_id_a=0, fragment_id_b=1,
                                     distance=-1.0, n_colors=5)

    def test_similarity_zero_distance(self):
        r = PaletteComparisonRecord(fragment_id_a=0, fragment_id_b=1,
                                     distance=0.0, n_colors=5)
        assert r.similarity == pytest.approx(1.0)

    def test_similarity_max_distance(self):
        r = PaletteComparisonRecord(fragment_id_a=0, fragment_id_b=1,
                                     distance=255.0, n_colors=5)
        assert r.similarity == pytest.approx(0.0)

    def test_similarity_partial(self):
        r = PaletteComparisonRecord(fragment_id_a=0, fragment_id_b=1,
                                     distance=127.5, n_colors=5)
        assert 0.0 < r.similarity < 1.0


# ─── PaletteRankingRecord ────────────────────────────────────────────────────

class TestPaletteRankingRecordExtra:
    def test_top_k_returns_k(self):
        r = PaletteRankingRecord(query_id=0,
                                  ranked_ids=[1, 2, 3],
                                  similarities=[0.9, 0.7, 0.5])
        top = r.top_k(2)
        assert len(top) == 2

    def test_top_k_content(self):
        r = PaletteRankingRecord(query_id=0,
                                  ranked_ids=[1, 2, 3],
                                  similarities=[0.9, 0.7, 0.5])
        top = r.top_k(1)
        assert top[0] == (1, 0.9)

    def test_empty_top_k(self):
        r = PaletteRankingRecord(query_id=0)
        assert r.top_k(5) == []


# ─── make_ranking_record ──────────────────────────────────────────────────────

class TestMakeRankingRecordExtra:
    def test_returns_record(self):
        r = make_ranking_record(10, 45)
        assert isinstance(r, RankingRunRecord)

    def test_fields_stored(self):
        r = make_ranking_record(5, 10, top_score=0.8, label="test")
        assert r.n_fragments == 5 and r.top_score == pytest.approx(0.8)


# ─── make_validation_record ──────────────────────────────────────────────────

class TestMakeValidationRecordExtra:
    def test_returns_record(self):
        r = make_validation_record(1, 10, 2, 0.8)
        assert isinstance(r, ValidationRunRecord)

    def test_fields_stored(self):
        r = make_validation_record(0, 5, 0, 1.0)
        assert r.n_violations == 0 and r.quality_score == pytest.approx(1.0)
