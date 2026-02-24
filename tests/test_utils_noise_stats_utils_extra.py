"""Extra tests for puzzle_reconstruction/utils/noise_stats_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.noise_stats_utils import (
    NoiseStatsConfig,
    NoiseStatsEntry,
    NoiseStatsSummary,
    make_noise_entry,
    entries_from_analysis_results,
    summarise_noise_stats,
    filter_clean_entries,
    filter_noisy_entries,
    filter_by_sigma_range,
    filter_by_snr_range,
    filter_by_jpeg_threshold,
    top_k_cleanest,
    top_k_noisiest,
    best_snr_entry,
    noise_stats_dict,
    compare_noise_summaries,
    batch_summarise_noise_stats,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _entry(image_id=0, sigma=5.0, snr_db=30.0,
           jpeg_level=0.1, grain_level=0.1, quality="clean") -> NoiseStatsEntry:
    return NoiseStatsEntry(image_id=image_id, sigma=sigma, snr_db=snr_db,
                           jpeg_level=jpeg_level, grain_level=grain_level,
                           quality=quality)


# ─── NoiseStatsConfig ─────────────────────────────────────────────────────────

class TestNoiseStatsConfigExtra:
    def test_default_max_sigma(self):
        assert NoiseStatsConfig().max_sigma == pytest.approx(50.0)

    def test_default_snr_threshold(self):
        assert NoiseStatsConfig().snr_threshold == pytest.approx(20.0)

    def test_default_quality_levels(self):
        assert NoiseStatsConfig().quality_levels == 3

    def test_zero_max_sigma_raises(self):
        with pytest.raises(ValueError):
            NoiseStatsConfig(max_sigma=0.0)

    def test_negative_snr_threshold_raises(self):
        with pytest.raises(ValueError):
            NoiseStatsConfig(snr_threshold=-1.0)

    def test_zero_quality_levels_raises(self):
        with pytest.raises(ValueError):
            NoiseStatsConfig(quality_levels=0)


# ─── NoiseStatsEntry ──────────────────────────────────────────────────────────

class TestNoiseStatsEntryExtra:
    def test_negative_image_id_raises(self):
        with pytest.raises(ValueError):
            _entry(image_id=-1)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError):
            _entry(sigma=-1.0)

    def test_jpeg_level_out_of_range_raises(self):
        with pytest.raises(ValueError):
            _entry(jpeg_level=1.5)

    def test_grain_level_out_of_range_raises(self):
        with pytest.raises(ValueError):
            _entry(grain_level=-0.1)

    def test_invalid_quality_raises(self):
        with pytest.raises(ValueError):
            _entry(quality="bad")

    def test_is_clean_true(self):
        assert _entry(quality="clean").is_clean is True

    def test_is_clean_false(self):
        assert _entry(quality="noisy").is_clean is False

    def test_is_noisy_true(self):
        assert _entry(quality="noisy").is_noisy is True

    def test_is_noisy_very_noisy(self):
        assert _entry(quality="very_noisy").is_noisy is True


# ─── make_noise_entry ─────────────────────────────────────────────────────────

class TestMakeNoiseEntryExtra:
    def test_returns_entry(self):
        e = make_noise_entry(0, 5.0, 30.0, 0.1, 0.1, "clean")
        assert isinstance(e, NoiseStatsEntry)

    def test_meta_stored(self):
        e = make_noise_entry(0, 5.0, 30.0, 0.1, 0.1, "clean", meta={"src": "cam"})
        assert e.meta["src"] == "cam"

    def test_empty_meta_default(self):
        e = make_noise_entry(0, 5.0, 30.0, 0.1, 0.1, "clean")
        assert e.meta == {}


# ─── entries_from_analysis_results ───────────────────────────────────────────

class TestEntriesFromAnalysisResultsExtra:
    class _FakeResult:
        def __init__(self, noise_level=1.0, snr_db=25.0,
                     jpeg_artifacts=0.1, grain_level=0.2, quality="clean"):
            self.noise_level = noise_level
            self.snr_db = snr_db
            self.jpeg_artifacts = jpeg_artifacts
            self.grain_level = grain_level
            self.quality = quality

    def test_returns_list(self):
        result = entries_from_analysis_results([self._FakeResult()])
        assert isinstance(result, list) and len(result) == 1

    def test_empty_returns_empty(self):
        assert entries_from_analysis_results([]) == []

    def test_ids_assigned(self):
        results = [self._FakeResult(), self._FakeResult()]
        entries = entries_from_analysis_results(results)
        assert entries[0].image_id == 0 and entries[1].image_id == 1


# ─── summarise_noise_stats ────────────────────────────────────────────────────

class TestSummariseNoiseStatsExtra:
    def test_empty_returns_summary(self):
        s = summarise_noise_stats([])
        assert s.n_total == 0 and s.n_clean == 0

    def test_n_clean_counted(self):
        entries = [_entry(quality="clean"), _entry(quality="noisy"),
                   _entry(quality="clean")]
        s = summarise_noise_stats(entries)
        assert s.n_clean == 2 and s.n_noisy == 1

    def test_mean_sigma(self):
        entries = [_entry(sigma=4.0), _entry(sigma=8.0)]
        s = summarise_noise_stats(entries)
        assert s.mean_sigma == pytest.approx(6.0)

    def test_min_max_sigma(self):
        entries = [_entry(sigma=2.0), _entry(sigma=10.0)]
        s = summarise_noise_stats(entries)
        assert s.min_sigma == pytest.approx(2.0) and s.max_sigma == pytest.approx(10.0)

    def test_repr_contains_n_total(self):
        s = summarise_noise_stats([_entry()])
        assert "n_total=1" in repr(s)


# ─── filters ──────────────────────────────────────────────────────────────────

class TestFilterNoiseStatsExtra:
    def test_filter_clean(self):
        entries = [_entry(quality="clean"), _entry(quality="noisy")]
        assert len(filter_clean_entries(entries)) == 1

    def test_filter_noisy(self):
        entries = [_entry(quality="clean"), _entry(quality="noisy"),
                   _entry(quality="very_noisy")]
        assert len(filter_noisy_entries(entries)) == 2

    def test_filter_sigma_range(self):
        entries = [_entry(sigma=2.0), _entry(sigma=8.0), _entry(sigma=15.0)]
        result = filter_by_sigma_range(entries, lo=5.0, hi=10.0)
        assert len(result) == 1 and result[0].sigma == pytest.approx(8.0)

    def test_filter_snr_range(self):
        entries = [_entry(snr_db=10.0), _entry(snr_db=30.0), _entry(snr_db=50.0)]
        result = filter_by_snr_range(entries, lo=20.0, hi=40.0)
        assert len(result) == 1

    def test_filter_by_jpeg_threshold(self):
        entries = [_entry(jpeg_level=0.2), _entry(jpeg_level=0.8)]
        result = filter_by_jpeg_threshold(entries, max_jpeg=0.5)
        assert len(result) == 1


# ─── top_k / best ─────────────────────────────────────────────────────────────

class TestRankingNoiseStatsExtra:
    def test_top_k_cleanest(self):
        entries = [_entry(sigma=10.0), _entry(sigma=2.0), _entry(sigma=5.0)]
        top = top_k_cleanest(entries, 2)
        assert top[0].sigma == pytest.approx(2.0)

    def test_top_k_noisiest(self):
        entries = [_entry(sigma=10.0), _entry(sigma=2.0), _entry(sigma=5.0)]
        top = top_k_noisiest(entries, 1)
        assert top[0].sigma == pytest.approx(10.0)

    def test_top_k_zero_returns_empty(self):
        assert top_k_cleanest([_entry()], 0) == []

    def test_best_snr_entry(self):
        entries = [_entry(snr_db=20.0), _entry(snr_db=40.0)]
        best = best_snr_entry(entries)
        assert best.snr_db == pytest.approx(40.0)

    def test_best_snr_empty(self):
        assert best_snr_entry([]) is None


# ─── noise_stats_dict ─────────────────────────────────────────────────────────

class TestNoiseStatsDictExtra:
    def test_empty_returns_n_zero(self):
        s = noise_stats_dict([])
        assert s["n"] == 0

    def test_count_and_min_max(self):
        entries = [_entry(sigma=2.0), _entry(sigma=8.0)]
        s = noise_stats_dict(entries)
        assert s["n"] == 2
        assert s["min"] == pytest.approx(2.0)
        assert s["max"] == pytest.approx(8.0)


# ─── compare_noise_summaries / batch ──────────────────────────────────────────

class TestCompareNoiseStatsExtra:
    def test_returns_dict(self):
        s = summarise_noise_stats([_entry()])
        d = compare_noise_summaries(s, s)
        assert isinstance(d, dict)

    def test_delta_zero_identical(self):
        s = summarise_noise_stats([_entry()])
        d = compare_noise_summaries(s, s)
        assert d["delta_mean_sigma"] == pytest.approx(0.0)

    def test_a_cleaner_flag(self):
        low = summarise_noise_stats([_entry(sigma=2.0)])
        high = summarise_noise_stats([_entry(sigma=8.0)])
        d = compare_noise_summaries(low, high)
        assert d["a_cleaner"] is True

    def test_batch_length(self):
        result = batch_summarise_noise_stats([[_entry()], []])
        assert len(result) == 2
