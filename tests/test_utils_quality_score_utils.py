"""Tests for puzzle_reconstruction.utils.quality_score_utils."""
import pytest
import numpy as np

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

np.random.seed(7)


# ─── QualityScoreConfig ──────────────────────────────────────────────────────

def test_config_defaults():
    cfg = QualityScoreConfig()
    assert cfg.min_overall == pytest.approx(0.5)
    assert cfg.min_blur == pytest.approx(0.0)


def test_config_invalid_min_overall():
    with pytest.raises(ValueError):
        QualityScoreConfig(min_overall=1.5)


def test_config_invalid_min_blur():
    with pytest.raises(ValueError):
        QualityScoreConfig(min_blur=-0.1)


def test_config_boundary_values():
    cfg = QualityScoreConfig(min_overall=0.0)
    assert cfg.min_overall == pytest.approx(0.0)
    cfg2 = QualityScoreConfig(min_overall=1.0)
    assert cfg2.min_overall == pytest.approx(1.0)


# ─── QualityScoreEntry ───────────────────────────────────────────────────────

def test_entry_repr():
    e = QualityScoreEntry(image_id=1, blur_score=0.7, noise_score=0.8,
                           contrast_score=0.6, completeness=0.9,
                           overall=0.75, is_acceptable=True)
    r = repr(e)
    assert "id=1" in r
    assert "overall=0.750" in r


def test_entry_is_acceptable_true():
    e = make_quality_entry(0, 0.8, 0.9, 0.7, 0.9, 0.8)
    assert e.is_acceptable is True


def test_entry_is_acceptable_false():
    e = make_quality_entry(0, 0.3, 0.3, 0.2, 0.3, 0.3)
    assert e.is_acceptable is False


# ─── make_quality_entry ──────────────────────────────────────────────────────

def test_make_quality_entry_defaults():
    e = make_quality_entry(5, 0.6, 0.7, 0.8, 0.9, 0.75)
    assert e.image_id == 5
    assert e.blur_score == pytest.approx(0.6)
    assert e.overall == pytest.approx(0.75)
    assert e.meta == {}


def test_make_quality_entry_with_meta():
    e = make_quality_entry(1, 0.5, 0.5, 0.5, 0.5, 0.6, meta={"tag": "test"})
    assert e.meta["tag"] == "test"


def test_make_quality_entry_custom_cfg():
    cfg = QualityScoreConfig(min_overall=0.9)
    e = make_quality_entry(0, 0.8, 0.8, 0.8, 0.8, 0.8, cfg=cfg)
    assert e.is_acceptable is False


def test_make_quality_entry_threshold_boundary():
    cfg = QualityScoreConfig(min_overall=0.5)
    e = make_quality_entry(0, 0.5, 0.5, 0.5, 0.5, 0.5, cfg=cfg)
    assert e.is_acceptable is True


# ─── entries_from_reports ────────────────────────────────────────────────────

def test_entries_from_reports_basic():
    reports = [
        {"image_id": 0, "blur_score": 0.8, "noise_score": 0.9,
         "contrast_score": 0.7, "completeness": 0.85, "overall": 0.8},
        {"image_id": 1, "blur_score": 0.3, "noise_score": 0.4,
         "contrast_score": 0.2, "completeness": 0.3, "overall": 0.3},
    ]
    entries = entries_from_reports(reports)
    assert len(entries) == 2
    assert entries[0].image_id == 0
    assert entries[1].overall == pytest.approx(0.3)


def test_entries_from_reports_missing_keys():
    reports = [{}]
    entries = entries_from_reports(reports)
    assert len(entries) == 1
    assert entries[0].overall == pytest.approx(0.0)


def test_entries_from_reports_meta_passthrough():
    reports = [{"image_id": 0, "blur_score": 0.5, "noise_score": 0.5,
                "contrast_score": 0.5, "completeness": 0.5, "overall": 0.6,
                "extra": "hello"}]
    entries = entries_from_reports(reports)
    assert entries[0].meta.get("extra") == "hello"


# ─── summarise_quality ───────────────────────────────────────────────────────

def test_summarise_quality_empty():
    s = summarise_quality([])
    assert s.n_total == 0
    assert s.mean_overall == pytest.approx(0.0)


def test_summarise_quality_counts():
    entries = [
        make_quality_entry(0, 0.8, 0.8, 0.8, 0.8, 0.8),
        make_quality_entry(1, 0.3, 0.3, 0.3, 0.3, 0.3),
        make_quality_entry(2, 0.7, 0.7, 0.7, 0.7, 0.7),
    ]
    s = summarise_quality(entries)
    assert s.n_total == 3
    assert s.n_acceptable == 2
    assert s.n_rejected == 1


def test_summarise_quality_mean_overall():
    entries = [
        make_quality_entry(0, 0.5, 0.5, 0.5, 0.5, 0.6),
        make_quality_entry(1, 0.5, 0.5, 0.5, 0.5, 0.8),
    ]
    s = summarise_quality(entries)
    assert s.mean_overall == pytest.approx(0.7)


def test_summarise_quality_returns_correct_type():
    entries = [make_quality_entry(0, 0.7, 0.7, 0.7, 0.7, 0.7)]
    s = summarise_quality(entries)
    assert isinstance(s, QualitySummary)


# ─── filter_acceptable / filter_rejected ────────────────────────────────────

def test_filter_acceptable():
    entries = [
        make_quality_entry(0, 0.8, 0.8, 0.8, 0.8, 0.8),
        make_quality_entry(1, 0.2, 0.2, 0.2, 0.2, 0.2),
    ]
    result = filter_acceptable(entries)
    assert len(result) == 1
    assert result[0].image_id == 0


def test_filter_rejected():
    entries = [
        make_quality_entry(0, 0.8, 0.8, 0.8, 0.8, 0.8),
        make_quality_entry(1, 0.2, 0.2, 0.2, 0.2, 0.2),
    ]
    result = filter_rejected(entries)
    assert len(result) == 1
    assert result[0].image_id == 1


# ─── filter_by_overall ───────────────────────────────────────────────────────

def test_filter_by_overall():
    entries = [
        make_quality_entry(0, 0.5, 0.5, 0.5, 0.5, 0.3),
        make_quality_entry(1, 0.5, 0.5, 0.5, 0.5, 0.7),
        make_quality_entry(2, 0.5, 0.5, 0.5, 0.5, 0.9),
    ]
    result = filter_by_overall(entries, min_overall=0.6)
    assert len(result) == 2
    assert all(e.overall >= 0.6 for e in result)


# ─── filter_by_blur ──────────────────────────────────────────────────────────

def test_filter_by_blur():
    entries = [
        make_quality_entry(0, 0.9, 0.5, 0.5, 0.5, 0.6),
        make_quality_entry(1, 0.3, 0.5, 0.5, 0.5, 0.6),
    ]
    result = filter_by_blur(entries, min_blur=0.5)
    assert len(result) == 1
    assert result[0].blur_score >= 0.5


# ─── top_k_quality_entries ───────────────────────────────────────────────────

def test_top_k_quality_entries_order():
    entries = [
        make_quality_entry(0, 0.5, 0.5, 0.5, 0.5, 0.6),
        make_quality_entry(1, 0.5, 0.5, 0.5, 0.5, 0.9),
        make_quality_entry(2, 0.5, 0.5, 0.5, 0.5, 0.7),
    ]
    top2 = top_k_quality_entries(entries, 2)
    assert len(top2) == 2
    assert top2[0].overall >= top2[1].overall


def test_top_k_quality_entries_k_zero():
    entries = [make_quality_entry(0, 0.5, 0.5, 0.5, 0.5, 0.6)]
    result = top_k_quality_entries(entries, 0)
    assert result == []


def test_top_k_quality_entries_k_larger_than_list():
    entries = [make_quality_entry(0, 0.5, 0.5, 0.5, 0.5, 0.7)]
    result = top_k_quality_entries(entries, 10)
    assert len(result) == 1


# ─── quality_score_stats ─────────────────────────────────────────────────────

def test_quality_score_stats_keys():
    entries = [
        make_quality_entry(0, 0.8, 0.8, 0.8, 0.8, 0.8),
        make_quality_entry(1, 0.3, 0.3, 0.3, 0.3, 0.3),
    ]
    stats = quality_score_stats(entries)
    for key in ("count", "mean", "std", "min", "max",
                "n_acceptable", "n_rejected"):
        assert key in stats


def test_quality_score_stats_empty():
    stats = quality_score_stats([])
    assert stats["count"] == 0


def test_quality_score_stats_values():
    entries = [
        make_quality_entry(0, 0.5, 0.5, 0.5, 0.5, 0.6),
        make_quality_entry(1, 0.5, 0.5, 0.5, 0.5, 0.8),
    ]
    stats = quality_score_stats(entries)
    assert stats["count"] == 2
    assert stats["mean"] == pytest.approx(0.7)
    assert stats["min"] == pytest.approx(0.6)
    assert stats["max"] == pytest.approx(0.8)


# ─── compare_quality ─────────────────────────────────────────────────────────

def test_compare_quality_keys():
    entries_a = [make_quality_entry(0, 0.8, 0.8, 0.8, 0.8, 0.8)]
    entries_b = [make_quality_entry(0, 0.5, 0.5, 0.5, 0.5, 0.5),
                 make_quality_entry(1, 0.6, 0.6, 0.6, 0.6, 0.6)]
    s_a = summarise_quality(entries_a)
    s_b = summarise_quality(entries_b)
    result = compare_quality(s_a, s_b)
    for key in ("n_total_delta", "n_acceptable_delta", "mean_overall_delta"):
        assert key in result


def test_compare_quality_n_total_delta():
    entries_a = [make_quality_entry(i, 0.6, 0.6, 0.6, 0.6, 0.6) for i in range(3)]
    entries_b = [make_quality_entry(i, 0.6, 0.6, 0.6, 0.6, 0.6) for i in range(5)]
    s_a = summarise_quality(entries_a)
    s_b = summarise_quality(entries_b)
    result = compare_quality(s_a, s_b)
    assert result["n_total_delta"] == -2


# ─── batch_summarise_quality ─────────────────────────────────────────────────

def test_batch_summarise_quality_length():
    report_lists = [
        [{"image_id": 0, "blur_score": 0.8, "noise_score": 0.8,
          "contrast_score": 0.8, "completeness": 0.8, "overall": 0.8}],
        [],
    ]
    results = batch_summarise_quality(report_lists)
    assert len(results) == 2


def test_batch_summarise_quality_empty_list():
    results = batch_summarise_quality([[]])
    assert results[0].n_total == 0


def test_batch_summarise_quality_returns_summaries():
    report_lists = [[{"overall": 0.7}]]
    results = batch_summarise_quality(report_lists)
    assert isinstance(results[0], QualitySummary)
