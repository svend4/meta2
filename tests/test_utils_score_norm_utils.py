"""Tests for puzzle_reconstruction.utils.score_norm_utils."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.score_norm_utils import (
    ScoreNormConfig,
    ScoreNormEntry,
    ScoreNormSummary,
    make_norm_entry,
    entries_from_scores,
    summarise_norm,
    filter_by_normalized_range,
    filter_by_original_range,
    top_k_norm_entries,
    norm_entry_stats,
    compare_norm_summaries,
    batch_summarise_norm,
)

np.random.seed(11)


# ─── ScoreNormConfig ─────────────────────────────────────────────────────────

def test_config_defaults():
    cfg = ScoreNormConfig()
    assert cfg.method == "minmax"
    assert cfg.clip is True
    assert cfg.feature_range == (0.0, 1.0)


def test_config_invalid_method():
    with pytest.raises(ValueError):
        ScoreNormConfig(method="unknown")


def test_config_invalid_feature_range():
    with pytest.raises(ValueError):
        ScoreNormConfig(feature_range=(1.0, 0.0))


def test_config_equal_feature_range():
    with pytest.raises(ValueError):
        ScoreNormConfig(feature_range=(0.5, 0.5))


def test_config_valid_methods():
    for m in ("minmax", "zscore", "rank", "calibrated"):
        cfg = ScoreNormConfig(method=m)
        assert cfg.method == m


# ─── ScoreNormEntry ──────────────────────────────────────────────────────────

def test_entry_delta():
    e = make_norm_entry(0, 0.4, 0.7)
    assert e.delta == pytest.approx(0.3)


def test_entry_negative_idx_raises():
    with pytest.raises(ValueError):
        ScoreNormEntry(idx=-1, original_score=0.5, normalized_score=0.8)


def test_entry_method_preserved():
    e = make_norm_entry(1, 0.3, 0.6, method="zscore")
    assert e.method == "zscore"


def test_entry_delta_negative():
    e = make_norm_entry(0, 0.8, 0.5)
    assert e.delta == pytest.approx(-0.3)


# ─── make_norm_entry ─────────────────────────────────────────────────────────

def test_make_norm_entry_fields():
    e = make_norm_entry(5, 0.3, 0.7, method="rank", meta={"info": "test"})
    assert e.idx == 5
    assert e.original_score == pytest.approx(0.3)
    assert e.normalized_score == pytest.approx(0.7)
    assert e.method == "rank"
    assert e.meta["info"] == "test"


def test_make_norm_entry_no_meta():
    e = make_norm_entry(0, 0.5, 0.5)
    assert e.meta == {}


# ─── entries_from_scores ─────────────────────────────────────────────────────

def test_entries_from_scores_basic():
    orig = [0.2, 0.5, 0.8]
    norm = [0.0, 0.5, 1.0]
    entries = entries_from_scores(orig, norm)
    assert len(entries) == 3
    assert entries[0].original_score == pytest.approx(0.2)
    assert entries[2].normalized_score == pytest.approx(1.0)


def test_entries_from_scores_length_mismatch():
    with pytest.raises(ValueError):
        entries_from_scores([0.1, 0.2], [0.5])


def test_entries_from_scores_indices():
    orig = [0.3, 0.6]
    norm = [0.2, 0.8]
    entries = entries_from_scores(orig, norm)
    assert entries[0].idx == 0
    assert entries[1].idx == 1


def test_entries_from_scores_method():
    orig = [0.3]
    norm = [0.6]
    entries = entries_from_scores(orig, norm, method="zscore")
    assert entries[0].method == "zscore"


# ─── summarise_norm ──────────────────────────────────────────────────────────

def test_summarise_norm_empty():
    s = summarise_norm([])
    assert s.n_total == 0
    assert s.method == ""


def test_summarise_norm_basic():
    entries = entries_from_scores([0.1, 0.5, 0.9], [0.0, 0.5, 1.0])
    s = summarise_norm(entries)
    assert s.n_total == 3
    assert s.original_min == pytest.approx(0.1)
    assert s.original_max == pytest.approx(0.9)
    assert s.normalized_min == pytest.approx(0.0)
    assert s.normalized_max == pytest.approx(1.0)


def test_summarise_norm_repr():
    entries = entries_from_scores([0.2, 0.8], [0.0, 1.0])
    s = summarise_norm(entries)
    r = repr(s)
    assert "n=2" in r


def test_summarise_norm_returns_correct_type():
    entries = entries_from_scores([0.5], [0.5])
    s = summarise_norm(entries)
    assert isinstance(s, ScoreNormSummary)


# ─── filter_by_normalized_range ──────────────────────────────────────────────

def test_filter_by_normalized_range():
    orig = [0.1, 0.5, 0.9]
    norm = [0.0, 0.5, 1.0]
    entries = entries_from_scores(orig, norm)
    result = filter_by_normalized_range(entries, lo=0.3, hi=0.7)
    assert len(result) == 1
    assert result[0].normalized_score == pytest.approx(0.5)


def test_filter_by_normalized_range_all():
    orig = [0.1, 0.5, 0.9]
    norm = [0.0, 0.5, 1.0]
    entries = entries_from_scores(orig, norm)
    result = filter_by_normalized_range(entries, lo=0.0, hi=1.0)
    assert len(result) == 3


def test_filter_by_normalized_range_none():
    orig = [0.1, 0.5]
    norm = [0.0, 0.5]
    entries = entries_from_scores(orig, norm)
    result = filter_by_normalized_range(entries, lo=0.7, hi=1.0)
    assert len(result) == 0


# ─── filter_by_original_range ────────────────────────────────────────────────

def test_filter_by_original_range():
    orig = [0.1, 0.5, 0.9]
    norm = [0.0, 0.5, 1.0]
    entries = entries_from_scores(orig, norm)
    result = filter_by_original_range(entries, lo=0.4, hi=0.6)
    assert len(result) == 1
    assert result[0].original_score == pytest.approx(0.5)


# ─── top_k_norm_entries ──────────────────────────────────────────────────────

def test_top_k_norm_entries_order():
    orig = [0.2, 0.5, 0.8]
    norm = [0.1, 0.6, 0.9]
    entries = entries_from_scores(orig, norm)
    top2 = top_k_norm_entries(entries, k=2)
    assert len(top2) == 2
    assert top2[0].normalized_score >= top2[1].normalized_score


def test_top_k_norm_entries_k_one():
    orig = [0.2, 0.5, 0.8]
    norm = [0.1, 0.6, 0.9]
    entries = entries_from_scores(orig, norm)
    top1 = top_k_norm_entries(entries, k=1)
    assert len(top1) == 1
    assert top1[0].normalized_score == pytest.approx(0.9)


def test_top_k_norm_entries_invalid_k():
    entries = entries_from_scores([0.5], [0.5])
    with pytest.raises(ValueError):
        top_k_norm_entries(entries, k=0)


def test_top_k_norm_entries_k_larger():
    orig = [0.3]
    norm = [0.6]
    entries = entries_from_scores(orig, norm)
    result = top_k_norm_entries(entries, k=100)
    assert len(result) == 1


# ─── norm_entry_stats ────────────────────────────────────────────────────────

def test_norm_entry_stats_empty():
    stats = norm_entry_stats([])
    assert stats["n"] == 0
    assert stats["mean_original"] == pytest.approx(0.0)


def test_norm_entry_stats_basic():
    orig = [0.2, 0.6]
    norm = [0.1, 0.9]
    entries = entries_from_scores(orig, norm)
    stats = norm_entry_stats(entries)
    assert stats["n"] == 2
    assert stats["mean_original"] == pytest.approx(0.4)
    assert stats["mean_normalized"] == pytest.approx(0.5)
    assert stats["mean_delta"] == pytest.approx(0.1)


def test_norm_entry_stats_keys():
    entries = entries_from_scores([0.3], [0.7])
    stats = norm_entry_stats(entries)
    for key in ("n", "mean_original", "mean_normalized", "mean_delta"):
        assert key in stats


# ─── compare_norm_summaries ──────────────────────────────────────────────────

def test_compare_norm_summaries_keys():
    s_a = summarise_norm(entries_from_scores([0.1, 0.9], [0.0, 1.0]))
    s_b = summarise_norm(entries_from_scores([0.2, 0.8], [0.1, 0.9]))
    result = compare_norm_summaries(s_a, s_b)
    for key in ("n_total_delta", "original_min_delta", "original_max_delta",
                "normalized_min_delta", "normalized_max_delta"):
        assert key in result


def test_compare_norm_summaries_self():
    s = summarise_norm(entries_from_scores([0.2, 0.8], [0.0, 1.0]))
    result = compare_norm_summaries(s, s)
    assert result["n_total_delta"] == 0
    assert result["original_min_delta"] == pytest.approx(0.0)


# ─── batch_summarise_norm ────────────────────────────────────────────────────

def test_batch_summarise_norm_length():
    score_lists = [
        ([0.1, 0.5, 0.9], [0.0, 0.5, 1.0]),
        ([0.2, 0.8], [0.1, 0.9]),
    ]
    results = batch_summarise_norm(score_lists)
    assert len(results) == 2


def test_batch_summarise_norm_returns_summaries():
    score_lists = [([0.5], [0.7])]
    results = batch_summarise_norm(score_lists)
    assert isinstance(results[0], ScoreNormSummary)


def test_batch_summarise_norm_method():
    score_lists = [([0.3, 0.7], [0.0, 1.0])]
    results = batch_summarise_norm(score_lists, method="zscore")
    assert results[0].method == "zscore"
