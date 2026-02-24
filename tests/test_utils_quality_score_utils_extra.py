"""Extra tests for puzzle_reconstruction/utils/quality_score_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.quality_score_utils import (
    QualityScoreConfig,
    QualityScoreEntry,
    QualitySummary,
    make_quality_entry,
    entries_from_reports,
    summarise_quality,
    filter_acceptable,
    filter_rejected,
    filter_by_overall,
    filter_by_blur,
    top_k_quality_entries,
    quality_score_stats,
    compare_quality,
    batch_summarise_quality,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _entry(image_id=0, blur=0.8, noise=0.7, contrast=0.9,
           completeness=0.9, overall=0.8, acceptable=True) -> QualityScoreEntry:
    return QualityScoreEntry(image_id=image_id, blur_score=blur,
                              noise_score=noise, contrast_score=contrast,
                              completeness=completeness, overall=overall,
                              is_acceptable=acceptable)


# ─── QualityScoreConfig ───────────────────────────────────────────────────────

class TestQualityScoreConfigExtra:
    def test_default_min_overall(self):
        assert QualityScoreConfig().min_overall == pytest.approx(0.5)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            QualityScoreConfig(min_overall=1.5)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            QualityScoreConfig(min_blur=-0.1)

    def test_all_defaults_valid(self):
        cfg = QualityScoreConfig()
        assert cfg.min_blur == pytest.approx(0.0)


# ─── QualityScoreEntry ────────────────────────────────────────────────────────

class TestQualityScoreEntryExtra:
    def test_stores_overall(self):
        e = _entry(overall=0.75)
        assert e.overall == pytest.approx(0.75)

    def test_repr_contains_id(self):
        e = _entry(image_id=42)
        assert "42" in repr(e)

    def test_is_acceptable_stored(self):
        e = _entry(acceptable=False)
        assert e.is_acceptable is False


# ─── make_quality_entry ───────────────────────────────────────────────────────

class TestMakeQualityEntryExtra:
    def test_returns_entry(self):
        e = make_quality_entry(0, 0.8, 0.7, 0.9, 0.9, 0.8)
        assert isinstance(e, QualityScoreEntry)

    def test_acceptable_when_above_threshold(self):
        cfg = QualityScoreConfig(min_overall=0.5)
        e = make_quality_entry(0, 0.8, 0.7, 0.9, 0.9, 0.8, cfg=cfg)
        assert e.is_acceptable is True

    def test_rejected_when_below_threshold(self):
        cfg = QualityScoreConfig(min_overall=0.9)
        e = make_quality_entry(0, 0.8, 0.7, 0.9, 0.9, 0.5, cfg=cfg)
        assert e.is_acceptable is False

    def test_meta_stored(self):
        e = make_quality_entry(0, 0.8, 0.7, 0.9, 0.9, 0.8, meta={"src": "cam"})
        assert e.meta["src"] == "cam"


# ─── entries_from_reports ─────────────────────────────────────────────────────

class TestEntriesFromReportsExtra:
    def test_returns_list(self):
        reps = [{"overall": 0.8, "blur_score": 0.9, "noise_score": 0.7,
                  "contrast_score": 0.8, "completeness": 0.9}]
        result = entries_from_reports(reps)
        assert isinstance(result, list) and len(result) == 1

    def test_empty_input(self):
        assert entries_from_reports([]) == []

    def test_overall_stored(self):
        reps = [{"overall": 0.6}]
        entry = entries_from_reports(reps)[0]
        assert entry.overall == pytest.approx(0.6)


# ─── summarise_quality ────────────────────────────────────────────────────────

class TestSummariseQualityExtra:
    def test_empty_returns_summary(self):
        s = summarise_quality([])
        assert s.n_total == 0

    def test_n_acceptable_counted(self):
        entries = [_entry(acceptable=True), _entry(acceptable=False),
                   _entry(acceptable=True)]
        s = summarise_quality(entries)
        assert s.n_acceptable == 2 and s.n_rejected == 1

    def test_mean_overall(self):
        entries = [_entry(overall=0.4), _entry(overall=0.8)]
        s = summarise_quality(entries)
        assert s.mean_overall == pytest.approx(0.6)

    def test_repr_contains_n(self):
        s = summarise_quality([_entry()])
        assert "n=1" in repr(s)


# ─── filter helpers ───────────────────────────────────────────────────────────

class TestFilterQualityExtra:
    def test_filter_acceptable(self):
        entries = [_entry(acceptable=True), _entry(acceptable=False)]
        result = filter_acceptable(entries)
        assert all(e.is_acceptable for e in result)

    def test_filter_rejected(self):
        entries = [_entry(acceptable=True), _entry(acceptable=False)]
        result = filter_rejected(entries)
        assert not any(e.is_acceptable for e in result)

    def test_filter_by_overall(self):
        entries = [_entry(overall=0.3), _entry(overall=0.9)]
        result = filter_by_overall(entries, 0.7)
        assert len(result) == 1

    def test_filter_by_blur(self):
        entries = [_entry(blur=0.2), _entry(blur=0.9)]
        result = filter_by_blur(entries, 0.5)
        assert len(result) == 1

    def test_top_k(self):
        entries = [_entry(overall=0.3), _entry(overall=0.9), _entry(overall=0.6)]
        top = top_k_quality_entries(entries, 2)
        assert top[0].overall == pytest.approx(0.9)
        assert len(top) == 2

    def test_top_k_zero_returns_empty(self):
        entries = [_entry(overall=0.8)]
        assert top_k_quality_entries(entries, 0) == []


# ─── quality_score_stats ──────────────────────────────────────────────────────

class TestQualityScoreStatsExtra:
    def test_empty_returns_zeros(self):
        s = quality_score_stats([])
        assert s["count"] == 0

    def test_count_correct(self):
        s = quality_score_stats([_entry(), _entry()])
        assert s["count"] == 2

    def test_min_max(self):
        entries = [_entry(overall=0.2), _entry(overall=0.8)]
        s = quality_score_stats(entries)
        assert s["min"] == pytest.approx(0.2)
        assert s["max"] == pytest.approx(0.8)


# ─── compare_quality / batch ──────────────────────────────────────────────────

class TestCompareQualityExtra:
    def test_returns_dict(self):
        s = summarise_quality([_entry()])
        d = compare_quality(s, s)
        assert isinstance(d, dict)

    def test_delta_zero_identical(self):
        s = summarise_quality([_entry()])
        d = compare_quality(s, s)
        assert d["mean_overall_delta"] == pytest.approx(0.0)

    def test_batch_length(self):
        reps = [{"overall": 0.8}]
        result = batch_summarise_quality([reps, reps])
        assert len(result) == 2
