"""Tests for puzzle_reconstruction.utils.normalize_noise_utils."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.normalize_noise_utils import (
    NormResultConfig,
    NormResultEntry,
    NormResultSummary,
    NoiseResultConfig,
    NoiseResultEntry,
    NoiseResultSummary,
    make_norm_result_entry,
    summarise_norm_result_entries,
    filter_norm_by_method,
    filter_norm_by_min_spread,
    top_k_norm_by_spread,
    best_norm_entry,
    norm_spread_stats,
    compare_norm_summaries,
    batch_summarise_norm_entries,
    make_noise_result_entry,
    summarise_noise_result_entries,
    filter_noise_by_method,
    filter_noise_by_max_after,
    filter_noise_by_min_delta,
    top_k_noise_by_delta,
    best_noise_entry,
    noise_delta_stats,
    compare_noise_summaries,
    batch_summarise_noise_entries,
)

np.random.seed(42)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm_entries():
    return [
        make_norm_result_entry(0, "minmax", 0.0, 1.0, 100, 100),
        make_norm_result_entry(1, "zscore", -2.0, 2.0, 50, 50),
        make_norm_result_entry(2, "minmax", 0.5, 0.8, 80, 80),
    ]


def _noise_entries():
    return [
        make_noise_result_entry(0, "gaussian", 30.0, 10.0, 10000),
        make_noise_result_entry(1, "median", 25.0, 15.0, 8000),
        make_noise_result_entry(2, "gaussian", 40.0, 5.0, 12000),
    ]


# ── NormResultEntry ───────────────────────────────────────────────────────────

def test_make_norm_result_entry_spread():
    e = make_norm_result_entry(0, "minmax", 0.0, 5.0, 10, 10)
    assert e.spread == pytest.approx(5.0)


def test_make_norm_result_entry_types():
    e = make_norm_result_entry(1, "zscore", -1.0, 3.0, 20, 30, alpha=0.5)
    assert e.run_id == 1
    assert e.method == "zscore"
    assert e.n_rows == 20
    assert e.n_cols == 30
    assert e.params["alpha"] == pytest.approx(0.5)


def test_make_norm_result_entry_negative_spread_allowed():
    # max_val < min_val results in negative spread — just verify it doesn't crash
    e = make_norm_result_entry(2, "minmax", 5.0, 2.0, 5, 5)
    assert e.spread == pytest.approx(-3.0)


# ── summarise_norm_result_entries ─────────────────────────────────────────────

def test_summarise_norm_empty():
    s = summarise_norm_result_entries([])
    assert s.n_runs == 0
    assert s.best_run_id is None


def test_summarise_norm_normal():
    entries = _norm_entries()
    s = summarise_norm_result_entries(entries)
    assert s.n_runs == 3
    # zscore entry has spread=4.0, which is the largest
    assert s.best_run_id == 1
    assert s.worst_run_id == 2


def test_summarise_norm_method_counts():
    entries = _norm_entries()
    s = summarise_norm_result_entries(entries)
    assert s.method_counts["minmax"] == 2
    assert s.method_counts["zscore"] == 1


def test_summarise_norm_mean_spread():
    entries = _norm_entries()
    s = summarise_norm_result_entries(entries)
    spreads = [1.0, 4.0, 0.3]
    assert s.mean_spread == pytest.approx(sum(spreads) / len(spreads))


# ── Filters ───────────────────────────────────────────────────────────────────

def test_filter_norm_by_method():
    entries = _norm_entries()
    filtered = filter_norm_by_method(entries, "minmax")
    assert all(e.method == "minmax" for e in filtered)
    assert len(filtered) == 2


def test_filter_norm_by_min_spread():
    entries = _norm_entries()
    filtered = filter_norm_by_min_spread(entries, 1.0)
    assert all(e.spread >= 1.0 for e in filtered)


def test_top_k_norm_by_spread():
    entries = _norm_entries()
    top = top_k_norm_by_spread(entries, 2)
    assert len(top) == 2
    assert top[0].spread >= top[1].spread


def test_best_norm_entry():
    entries = _norm_entries()
    best = best_norm_entry(entries)
    assert best is not None
    assert best.spread == pytest.approx(4.0)


def test_best_norm_entry_empty():
    assert best_norm_entry([]) is None


def test_norm_spread_stats_empty():
    stats = norm_spread_stats([])
    assert stats["count"] == 0


def test_norm_spread_stats_values():
    entries = _norm_entries()
    stats = norm_spread_stats(entries)
    assert stats["min"] == pytest.approx(0.3)
    assert stats["max"] == pytest.approx(4.0)
    assert stats["count"] == 3


def test_compare_norm_summaries():
    entries = _norm_entries()
    s1 = summarise_norm_result_entries(entries[:2])
    s2 = summarise_norm_result_entries(entries[1:])
    delta = compare_norm_summaries(s1, s2)
    assert "delta_mean_spread" in delta
    assert "same_best" in delta


def test_batch_summarise_norm_entries():
    groups = [_norm_entries()[:2], _norm_entries()[1:]]
    summaries = batch_summarise_norm_entries(groups)
    assert len(summaries) == 2


# ── NoiseResultEntry ──────────────────────────────────────────────────────────

def test_make_noise_result_entry_delta():
    e = make_noise_result_entry(0, "gaussian", 30.0, 10.0, 5000)
    assert e.noise_delta == pytest.approx(20.0)


def test_make_noise_result_entry_types():
    e = make_noise_result_entry(1, "median", 20.0, 8.0, 4000, kernel=5)
    assert e.image_id == 1
    assert e.method == "median"
    assert e.n_pixels == 4000
    assert e.params["kernel"] == 5


# ── summarise_noise_result_entries ────────────────────────────────────────────

def test_summarise_noise_empty():
    s = summarise_noise_result_entries([])
    assert s.n_images == 0
    assert s.best_image_id is None


def test_summarise_noise_normal():
    entries = _noise_entries()
    s = summarise_noise_result_entries(entries)
    assert s.n_images == 3
    # noise_delta: 20, 10, 35 -> best is index 2 (image_id=2)
    assert s.best_image_id == 2
    assert s.worst_image_id == 1


def test_summarise_noise_mean_delta():
    entries = _noise_entries()
    s = summarise_noise_result_entries(entries)
    assert s.mean_delta == pytest.approx((20.0 + 10.0 + 35.0) / 3)


def test_filter_noise_by_method():
    entries = _noise_entries()
    filtered = filter_noise_by_method(entries, "gaussian")
    assert all(e.method == "gaussian" for e in filtered)
    assert len(filtered) == 2


def test_filter_noise_by_max_after():
    entries = _noise_entries()
    filtered = filter_noise_by_max_after(entries, 10.0)
    assert all(e.noise_after <= 10.0 for e in filtered)


def test_filter_noise_by_min_delta():
    entries = _noise_entries()
    filtered = filter_noise_by_min_delta(entries, 20.0)
    assert all(e.noise_delta >= 20.0 for e in filtered)


def test_top_k_noise_by_delta():
    entries = _noise_entries()
    top = top_k_noise_by_delta(entries, 2)
    assert len(top) == 2
    assert top[0].noise_delta >= top[1].noise_delta


def test_best_noise_entry():
    entries = _noise_entries()
    best = best_noise_entry(entries)
    assert best.noise_delta == pytest.approx(35.0)


def test_noise_delta_stats_empty():
    stats = noise_delta_stats([])
    assert stats["count"] == 0


def test_compare_noise_summaries():
    entries = _noise_entries()
    s1 = summarise_noise_result_entries(entries[:2])
    s2 = summarise_noise_result_entries(entries[1:])
    delta = compare_noise_summaries(s1, s2)
    assert "delta_mean_noise_before" in delta
    assert "same_best" in delta


def test_batch_summarise_noise_entries():
    groups = [_noise_entries()[:2], _noise_entries()[1:]]
    summaries = batch_summarise_noise_entries(groups)
    assert len(summaries) == 2
