"""Extra tests for puzzle_reconstruction/utils/orient_skew_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.orient_skew_utils import (
    OrientMatchConfig,
    OrientMatchEntry,
    OrientMatchSummary,
    SkewCorrConfig,
    SkewCorrEntry,
    SkewCorrSummary,
    make_orient_match_entry,
    summarise_orient_match_entries,
    filter_high_orient_matches,
    filter_low_orient_matches,
    filter_orient_by_score_range,
    filter_orient_by_max_angle,
    top_k_orient_match_entries,
    best_orient_match_entry,
    orient_match_stats,
    compare_orient_summaries,
    batch_summarise_orient_match_entries,
    make_skew_corr_entry,
    summarise_skew_corr_entries,
    filter_high_confidence_skew,
    filter_skew_by_method,
    filter_skew_by_angle_range,
    top_k_skew_entries,
    best_skew_entry,
    skew_corr_stats,
    compare_skew_summaries,
    batch_summarise_skew_corr_entries,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _orient(frag_a=0, frag_b=1, angle=45.0, score=0.8, n=36) -> OrientMatchEntry:
    return OrientMatchEntry(fragment_a=frag_a, fragment_b=frag_b,
                            best_angle=angle, best_score=score,
                            n_angles_tested=n)


def _skew(image_id=0, angle=5.0, conf=0.9, method="auto") -> SkewCorrEntry:
    return SkewCorrEntry(image_id=image_id, angle_deg=angle,
                         confidence=conf, method=method)


# ─── OrientMatchConfig ────────────────────────────────────────────────────────

class TestOrientMatchConfigExtra:
    def test_default_min_score(self):
        assert OrientMatchConfig().min_score == pytest.approx(0.0)

    def test_default_max_angle(self):
        assert OrientMatchConfig().max_angle == pytest.approx(180.0)


# ─── make_orient_match_entry ──────────────────────────────────────────────────

class TestMakeOrientMatchEntryExtra:
    def test_returns_entry(self):
        e = make_orient_match_entry(0, 1, 45.0, 0.8, 36)
        assert isinstance(e, OrientMatchEntry)

    def test_values_stored(self):
        e = make_orient_match_entry(2, 3, 90.0, 0.95, 72)
        assert e.fragment_a == 2 and e.best_angle == pytest.approx(90.0)


# ─── summarise_orient_match_entries ───────────────────────────────────────────

class TestSummariseOrientMatchEntriesExtra:
    def test_empty_returns_summary(self):
        s = summarise_orient_match_entries([])
        assert s.n_entries == 0

    def test_single_entry(self):
        s = summarise_orient_match_entries([_orient(score=0.8)])
        assert s.n_entries == 1
        assert s.max_score == pytest.approx(0.8)

    def test_high_score_count(self):
        entries = [_orient(score=0.8), _orient(score=0.5), _orient(score=0.9)]
        s = summarise_orient_match_entries(entries)
        assert s.high_score_count == 2

    def test_mean_score(self):
        entries = [_orient(score=0.6), _orient(score=0.8)]
        s = summarise_orient_match_entries(entries)
        assert s.mean_score == pytest.approx(0.7)


# ─── filter orient ────────────────────────────────────────────────────────────

class TestFilterOrientExtra:
    def test_filter_high_matches(self):
        entries = [_orient(score=0.8), _orient(score=0.4)]
        result = filter_high_orient_matches(entries, 0.7)
        assert all(e.best_score >= 0.7 for e in result)

    def test_filter_low_matches(self):
        entries = [_orient(score=0.8), _orient(score=0.4)]
        result = filter_low_orient_matches(entries, 0.7)
        assert all(e.best_score < 0.7 for e in result)

    def test_filter_score_range(self):
        entries = [_orient(score=0.3), _orient(score=0.6), _orient(score=0.9)]
        result = filter_orient_by_score_range(entries, 0.5, 0.8)
        assert all(0.5 <= e.best_score <= 0.8 for e in result)

    def test_filter_by_max_angle(self):
        entries = [_orient(angle=30.0), _orient(angle=100.0)]
        result = filter_orient_by_max_angle(entries, 45.0)
        assert len(result) == 1

    def test_top_k(self):
        entries = [_orient(score=0.5), _orient(score=0.9), _orient(score=0.7)]
        top = top_k_orient_match_entries(entries, 2)
        assert top[0].best_score == pytest.approx(0.9)
        assert len(top) == 2

    def test_best_entry_none_empty(self):
        assert best_orient_match_entry([]) is None

    def test_best_entry(self):
        entries = [_orient(score=0.5), _orient(score=0.9)]
        best = best_orient_match_entry(entries)
        assert best.best_score == pytest.approx(0.9)


# ─── orient_match_stats ───────────────────────────────────────────────────────

class TestOrientMatchStatsExtra:
    def test_empty_returns_zeros(self):
        s = orient_match_stats([])
        assert s["count"] == 0

    def test_count_correct(self):
        s = orient_match_stats([_orient(), _orient()])
        assert s["count"] == pytest.approx(2.0)

    def test_min_max(self):
        entries = [_orient(score=0.3), _orient(score=0.9)]
        s = orient_match_stats(entries)
        assert s["min"] == pytest.approx(0.3)
        assert s["max"] == pytest.approx(0.9)


# ─── compare_orient_summaries / batch ────────────────────────────────────────

class TestCompareOrientSummariesExtra:
    def test_returns_dict(self):
        s = summarise_orient_match_entries([_orient()])
        d = compare_orient_summaries(s, s)
        assert isinstance(d, dict)

    def test_delta_zero_identical(self):
        s = summarise_orient_match_entries([_orient()])
        d = compare_orient_summaries(s, s)
        assert d["mean_score_delta"] == pytest.approx(0.0)

    def test_batch_length(self):
        result = batch_summarise_orient_match_entries([[_orient()], []])
        assert len(result) == 2


# ─── SkewCorrConfig ───────────────────────────────────────────────────────────

class TestSkewCorrConfigExtra:
    def test_default_min_confidence(self):
        assert SkewCorrConfig().min_confidence == pytest.approx(0.0)

    def test_default_method(self):
        assert SkewCorrConfig().method == "auto"


# ─── make_skew_corr_entry ─────────────────────────────────────────────────────

class TestMakeSkewCorrEntryExtra:
    def test_returns_entry(self):
        e = make_skew_corr_entry(0, 5.0, 0.9, "auto")
        assert isinstance(e, SkewCorrEntry)

    def test_values_stored(self):
        e = make_skew_corr_entry(3, 10.0, 0.85, "hough")
        assert e.image_id == 3 and e.angle_deg == pytest.approx(10.0)


# ─── summarise_skew_corr_entries ──────────────────────────────────────────────

class TestSummariseSkewCorrEntriesExtra:
    def test_empty_returns_summary(self):
        s = summarise_skew_corr_entries([])
        assert s.n_entries == 0

    def test_dominant_method(self):
        entries = [_skew(method="auto"), _skew(method="auto"),
                   _skew(method="hough")]
        s = summarise_skew_corr_entries(entries)
        assert s.dominant_method == "auto"

    def test_mean_confidence(self):
        entries = [_skew(conf=0.6), _skew(conf=0.8)]
        s = summarise_skew_corr_entries(entries)
        assert s.mean_confidence == pytest.approx(0.7)


# ─── filter skew ──────────────────────────────────────────────────────────────

class TestFilterSkewExtra:
    def test_filter_high_confidence(self):
        entries = [_skew(conf=0.8), _skew(conf=0.3)]
        result = filter_high_confidence_skew(entries, 0.5)
        assert all(e.confidence >= 0.5 for e in result)

    def test_filter_by_method(self):
        entries = [_skew(method="auto"), _skew(method="hough")]
        result = filter_skew_by_method(entries, "hough")
        assert all(e.method == "hough" for e in result)

    def test_filter_by_angle_range(self):
        entries = [_skew(angle=5.0), _skew(angle=20.0), _skew(angle=45.0)]
        result = filter_skew_by_angle_range(entries, 4.0, 25.0)
        assert len(result) == 2

    def test_top_k_by_confidence(self):
        entries = [_skew(conf=0.5), _skew(conf=0.95), _skew(conf=0.7)]
        top = top_k_skew_entries(entries, 2)
        assert top[0].confidence == pytest.approx(0.95)

    def test_best_skew_none_empty(self):
        assert best_skew_entry([]) is None

    def test_best_skew_entry(self):
        entries = [_skew(conf=0.5), _skew(conf=0.95)]
        best = best_skew_entry(entries)
        assert best.confidence == pytest.approx(0.95)


# ─── skew_corr_stats ──────────────────────────────────────────────────────────

class TestSkewCorrStatsExtra:
    def test_empty_returns_zeros(self):
        s = skew_corr_stats([])
        assert s["count"] == 0

    def test_count_correct(self):
        s = skew_corr_stats([_skew(), _skew()])
        assert s["count"] == pytest.approx(2.0)


# ─── compare_skew_summaries / batch ──────────────────────────────────────────

class TestCompareSkewSummariesExtra:
    def test_returns_dict(self):
        s = summarise_skew_corr_entries([_skew()])
        d = compare_skew_summaries(s, s)
        assert isinstance(d, dict)

    def test_delta_zero_identical(self):
        s = summarise_skew_corr_entries([_skew()])
        d = compare_skew_summaries(s, s)
        assert d["mean_confidence_delta"] == pytest.approx(0.0)

    def test_batch_length(self):
        result = batch_summarise_skew_corr_entries([[_skew()], [_skew(), _skew()]])
        assert len(result) == 2
