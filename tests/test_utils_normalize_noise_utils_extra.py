"""Extra tests for puzzle_reconstruction/utils/normalize_noise_utils.py."""
from __future__ import annotations

import pytest

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _norm_entry(run_id=0, method="minmax", min_val=0.0, max_val=1.0,
                n_rows=10, n_cols=10) -> NormResultEntry:
    return NormResultEntry(run_id=run_id, method=method, min_val=min_val,
                           max_val=max_val, spread=max_val - min_val,
                           n_rows=n_rows, n_cols=n_cols)


def _noise_entry(image_id=0, method="gaussian", noise_before=10.0,
                 noise_after=3.0, n_pixels=1000) -> NoiseResultEntry:
    return NoiseResultEntry(image_id=image_id, method=method,
                            noise_before=noise_before, noise_after=noise_after,
                            noise_delta=noise_before - noise_after,
                            n_pixels=n_pixels)


# ─── NormResultConfig ─────────────────────────────────────────────────────────

class TestNormResultConfigExtra:
    def test_default_preferred_method(self):
        assert NormResultConfig().preferred_method == "minmax"

    def test_default_min_spread(self):
        assert NormResultConfig().min_spread == pytest.approx(0.0)

    def test_custom_values(self):
        cfg = NormResultConfig(preferred_method="zscore", min_spread=0.1)
        assert cfg.preferred_method == "zscore"


# ─── make_norm_result_entry ───────────────────────────────────────────────────

class TestMakeNormResultEntryExtra:
    def test_returns_entry(self):
        e = make_norm_result_entry(0, "minmax", 0.0, 1.0, 10, 10)
        assert isinstance(e, NormResultEntry)

    def test_spread_computed(self):
        e = make_norm_result_entry(0, "minmax", 0.2, 0.8, 10, 10)
        assert e.spread == pytest.approx(0.6)

    def test_params_stored(self):
        e = make_norm_result_entry(1, "zscore", 0.0, 1.0, 5, 5, clip=True)
        assert e.params.get("clip") is True


# ─── summarise_norm_result_entries ────────────────────────────────────────────

class TestSummariseNormResultEntriesExtra:
    def test_empty_returns_summary(self):
        s = summarise_norm_result_entries([])
        assert s.n_runs == 0 and s.best_run_id is None

    def test_single_entry(self):
        s = summarise_norm_result_entries([_norm_entry(run_id=5)])
        assert s.n_runs == 1 and s.best_run_id == 5

    def test_method_counts(self):
        entries = [_norm_entry(method="minmax"), _norm_entry(method="minmax"),
                   _norm_entry(method="zscore")]
        s = summarise_norm_result_entries(entries)
        assert s.method_counts["minmax"] == 2
        assert s.method_counts["zscore"] == 1

    def test_mean_spread(self):
        entries = [_norm_entry(min_val=0.0, max_val=0.4),
                   _norm_entry(min_val=0.0, max_val=0.6)]
        s = summarise_norm_result_entries(entries)
        assert s.mean_spread == pytest.approx(0.5)


# ─── filter/top_k/best norm ───────────────────────────────────────────────────

class TestFilterNormExtra:
    def test_filter_by_method(self):
        entries = [_norm_entry(method="minmax"), _norm_entry(method="zscore")]
        result = filter_norm_by_method(entries, "minmax")
        assert all(e.method == "minmax" for e in result)

    def test_filter_by_min_spread(self):
        entries = [_norm_entry(min_val=0.0, max_val=0.3),
                   _norm_entry(min_val=0.0, max_val=0.9)]
        result = filter_norm_by_min_spread(entries, 0.5)
        assert len(result) == 1

    def test_top_k_by_spread(self):
        entries = [_norm_entry(min_val=0.0, max_val=0.2),
                   _norm_entry(min_val=0.0, max_val=0.8),
                   _norm_entry(min_val=0.0, max_val=0.5)]
        top = top_k_norm_by_spread(entries, 2)
        assert top[0].spread == pytest.approx(0.8)
        assert len(top) == 2

    def test_best_norm_entry_none_empty(self):
        assert best_norm_entry([]) is None

    def test_best_norm_entry(self):
        entries = [_norm_entry(min_val=0.0, max_val=0.3),
                   _norm_entry(min_val=0.0, max_val=0.9)]
        best = best_norm_entry(entries)
        assert best.spread == pytest.approx(0.9)


# ─── norm_spread_stats ────────────────────────────────────────────────────────

class TestNormSpreadStatsExtra:
    def test_empty_returns_zeros(self):
        s = norm_spread_stats([])
        assert s["count"] == 0

    def test_min_max_correct(self):
        entries = [_norm_entry(min_val=0.0, max_val=0.2),
                   _norm_entry(min_val=0.0, max_val=0.8)]
        s = norm_spread_stats(entries)
        assert s["min"] == pytest.approx(0.2)
        assert s["max"] == pytest.approx(0.8)

    def test_count_correct(self):
        entries = [_norm_entry(), _norm_entry(), _norm_entry()]
        assert norm_spread_stats(entries)["count"] == 3


# ─── compare_norm_summaries / batch ───────────────────────────────────────────

class TestCompareNormSummariesExtra:
    def test_returns_dict(self):
        s = summarise_norm_result_entries([_norm_entry()])
        d = compare_norm_summaries(s, s)
        assert isinstance(d, dict)

    def test_delta_zero_identical(self):
        s = summarise_norm_result_entries([_norm_entry()])
        d = compare_norm_summaries(s, s)
        assert d["delta_mean_spread"] == pytest.approx(0.0)

    def test_batch_length(self):
        result = batch_summarise_norm_entries([[_norm_entry()], [_norm_entry()]])
        assert len(result) == 2


# ─── NoiseResultConfig ────────────────────────────────────────────────────────

class TestNoiseResultConfigExtra:
    def test_default_preferred_method(self):
        assert NoiseResultConfig().preferred_method == "gaussian"

    def test_default_max_noise_after_inf(self):
        import math
        assert math.isinf(NoiseResultConfig().max_noise_after)


# ─── make_noise_result_entry ──────────────────────────────────────────────────

class TestMakeNoiseResultEntryExtra:
    def test_returns_entry(self):
        e = make_noise_result_entry(0, "gaussian", 10.0, 3.0, 1000)
        assert isinstance(e, NoiseResultEntry)

    def test_delta_computed(self):
        e = make_noise_result_entry(0, "median", 8.0, 2.0, 500)
        assert e.noise_delta == pytest.approx(6.0)

    def test_params_stored(self):
        e = make_noise_result_entry(1, "bilateral", 5.0, 1.0, 200, kernel=5)
        assert e.params.get("kernel") == 5


# ─── summarise_noise_result_entries ───────────────────────────────────────────

class TestSummariseNoiseResultEntriesExtra:
    def test_empty_returns_summary(self):
        s = summarise_noise_result_entries([])
        assert s.n_images == 0 and s.best_image_id is None

    def test_single_entry_best(self):
        e = _noise_entry(image_id=7)
        s = summarise_noise_result_entries([e])
        assert s.best_image_id == 7

    def test_mean_delta(self):
        entries = [_noise_entry(noise_before=10.0, noise_after=4.0),
                   _noise_entry(noise_before=8.0, noise_after=2.0)]
        s = summarise_noise_result_entries(entries)
        assert s.mean_delta == pytest.approx(6.0)


# ─── filter/top_k/best noise ──────────────────────────────────────────────────

class TestFilterNoiseExtra:
    def test_filter_by_method(self):
        entries = [_noise_entry(method="gaussian"),
                   _noise_entry(method="median")]
        result = filter_noise_by_method(entries, "gaussian")
        assert all(e.method == "gaussian" for e in result)

    def test_filter_by_max_after(self):
        entries = [_noise_entry(noise_before=10.0, noise_after=1.0),
                   _noise_entry(noise_before=10.0, noise_after=8.0)]
        result = filter_noise_by_max_after(entries, 5.0)
        assert len(result) == 1

    def test_filter_by_min_delta(self):
        entries = [_noise_entry(noise_before=10.0, noise_after=8.0),
                   _noise_entry(noise_before=10.0, noise_after=2.0)]
        result = filter_noise_by_min_delta(entries, 5.0)
        assert len(result) == 1

    def test_top_k_by_delta(self):
        entries = [_noise_entry(noise_before=10.0, noise_after=5.0),
                   _noise_entry(noise_before=10.0, noise_after=1.0)]
        top = top_k_noise_by_delta(entries, 1)
        assert top[0].noise_delta == pytest.approx(9.0)

    def test_best_noise_entry_empty(self):
        assert best_noise_entry([]) is None

    def test_best_noise_entry(self):
        entries = [_noise_entry(noise_before=10.0, noise_after=8.0),
                   _noise_entry(noise_before=10.0, noise_after=1.0)]
        best = best_noise_entry(entries)
        assert best.noise_delta == pytest.approx(9.0)


# ─── noise_delta_stats ────────────────────────────────────────────────────────

class TestNoiseDeltaStatsExtra:
    def test_empty_returns_zeros(self):
        s = noise_delta_stats([])
        assert s["count"] == 0

    def test_count_correct(self):
        entries = [_noise_entry(), _noise_entry()]
        assert noise_delta_stats(entries)["count"] == 2

    def test_min_max(self):
        entries = [_noise_entry(noise_before=10.0, noise_after=7.0),
                   _noise_entry(noise_before=10.0, noise_after=2.0)]
        s = noise_delta_stats(entries)
        assert s["min"] == pytest.approx(3.0)
        assert s["max"] == pytest.approx(8.0)


# ─── compare_noise_summaries / batch ─────────────────────────────────────────

class TestCompareNoiseSummariesExtra:
    def test_returns_dict(self):
        s = summarise_noise_result_entries([_noise_entry()])
        d = compare_noise_summaries(s, s)
        assert isinstance(d, dict)

    def test_delta_zero_identical(self):
        s = summarise_noise_result_entries([_noise_entry()])
        d = compare_noise_summaries(s, s)
        assert d["delta_mean_delta"] == pytest.approx(0.0)

    def test_batch_length(self):
        result = batch_summarise_noise_entries([[_noise_entry()], []])
        assert len(result) == 2
