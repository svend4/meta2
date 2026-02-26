"""Tests for puzzle_reconstruction.utils.noise_stats_utils"""
import pytest
import math
from puzzle_reconstruction.utils.noise_stats_utils import (
    NoiseStatsConfig, NoiseStatsEntry, NoiseStatsSummary,
    make_noise_entry, entries_from_analysis_results,
    summarise_noise_stats, filter_clean_entries, filter_noisy_entries,
    filter_by_sigma_range, filter_by_snr_range, filter_by_jpeg_threshold,
    top_k_cleanest, top_k_noisiest, best_snr_entry,
    noise_stats_dict, compare_noise_summaries, batch_summarise_noise_stats,
)


def _make_entry(image_id=0, sigma=5.0, snr_db=30.0, jpeg_level=0.1,
                grain_level=0.2, quality="clean"):
    return make_noise_entry(image_id, sigma, snr_db, jpeg_level, grain_level, quality)


# ── NoiseStatsConfig ──────────────────────────────────────────────────────────

def test_noise_stats_config_defaults():
    cfg = NoiseStatsConfig()
    assert cfg.max_sigma == 50.0
    assert cfg.snr_threshold == 20.0
    assert cfg.quality_levels == 3


def test_noise_stats_config_invalid_max_sigma_raises():
    with pytest.raises(ValueError):
        NoiseStatsConfig(max_sigma=0.0)


def test_noise_stats_config_invalid_snr_raises():
    with pytest.raises(ValueError):
        NoiseStatsConfig(snr_threshold=-1.0)


def test_noise_stats_config_invalid_quality_levels_raises():
    with pytest.raises(ValueError):
        NoiseStatsConfig(quality_levels=0)


# ── NoiseStatsEntry ───────────────────────────────────────────────────────────

def test_noise_entry_negative_id_raises():
    with pytest.raises(ValueError):
        NoiseStatsEntry(image_id=-1, sigma=5.0, snr_db=30.0,
                        jpeg_level=0.1, grain_level=0.2, quality="clean")


def test_noise_entry_negative_sigma_raises():
    with pytest.raises(ValueError):
        NoiseStatsEntry(image_id=0, sigma=-1.0, snr_db=30.0,
                        jpeg_level=0.1, grain_level=0.2, quality="clean")


def test_noise_entry_jpeg_out_of_range_raises():
    with pytest.raises(ValueError):
        NoiseStatsEntry(image_id=0, sigma=5.0, snr_db=30.0,
                        jpeg_level=1.5, grain_level=0.2, quality="clean")


def test_noise_entry_grain_out_of_range_raises():
    with pytest.raises(ValueError):
        NoiseStatsEntry(image_id=0, sigma=5.0, snr_db=30.0,
                        jpeg_level=0.1, grain_level=-0.1, quality="clean")


def test_noise_entry_invalid_quality_raises():
    with pytest.raises(ValueError):
        NoiseStatsEntry(image_id=0, sigma=5.0, snr_db=30.0,
                        jpeg_level=0.1, grain_level=0.2, quality="perfect")


def test_noise_entry_is_clean():
    e = _make_entry(quality="clean")
    assert e.is_clean is True
    assert e.is_noisy is False


def test_noise_entry_is_noisy():
    e = _make_entry(quality="noisy")
    assert e.is_noisy is True
    assert e.is_clean is False


def test_noise_entry_is_very_noisy():
    e = _make_entry(quality="very_noisy")
    assert e.is_noisy is True


# ── make_noise_entry ──────────────────────────────────────────────────────────

def test_make_noise_entry_returns_entry():
    e = make_noise_entry(0, 5.0, 30.0, 0.1, 0.2, "clean")
    assert isinstance(e, NoiseStatsEntry)


def test_make_noise_entry_with_meta():
    e = make_noise_entry(1, 10.0, 25.0, 0.2, 0.3, "noisy", meta={"src": "test"})
    assert e.meta["src"] == "test"


# ── summarise_noise_stats ─────────────────────────────────────────────────────

def test_summarise_noise_stats_empty():
    s = summarise_noise_stats([])
    assert s.n_total == 0
    assert s.mean_sigma == 0.0


def test_summarise_noise_stats_basic():
    entries = [
        _make_entry(image_id=0, sigma=5.0, quality="clean"),
        _make_entry(image_id=1, sigma=15.0, quality="noisy"),
        _make_entry(image_id=2, sigma=25.0, quality="very_noisy"),
    ]
    s = summarise_noise_stats(entries)
    assert s.n_total == 3
    assert s.n_clean == 1
    assert s.n_noisy == 2
    assert abs(s.mean_sigma - (5.0 + 15.0 + 25.0) / 3) < 1e-9
    assert s.max_sigma == 25.0
    assert s.min_sigma == 5.0


def test_summarise_noise_stats_repr():
    s = summarise_noise_stats([_make_entry()])
    r = repr(s)
    assert "n_total" in r


# ── filter functions ──────────────────────────────────────────────────────────

def test_filter_clean_entries():
    entries = [_make_entry(quality="clean"), _make_entry(quality="noisy")]
    clean = filter_clean_entries(entries)
    assert len(clean) == 1
    assert clean[0].quality == "clean"


def test_filter_noisy_entries():
    entries = [
        _make_entry(quality="clean"),
        _make_entry(quality="noisy"),
        _make_entry(quality="very_noisy"),
    ]
    noisy = filter_noisy_entries(entries)
    assert len(noisy) == 2


def test_filter_by_sigma_range():
    entries = [
        _make_entry(image_id=0, sigma=5.0),
        _make_entry(image_id=1, sigma=15.0),
        _make_entry(image_id=2, sigma=30.0),
    ]
    filtered = filter_by_sigma_range(entries, lo=10.0, hi=20.0)
    assert len(filtered) == 1
    assert filtered[0].image_id == 1


def test_filter_by_snr_range():
    entries = [
        _make_entry(image_id=0, snr_db=10.0),
        _make_entry(image_id=1, snr_db=30.0),
    ]
    filtered = filter_by_snr_range(entries, lo=20.0, hi=40.0)
    assert len(filtered) == 1
    assert filtered[0].image_id == 1


def test_filter_by_jpeg_threshold():
    entries = [
        _make_entry(image_id=0, jpeg_level=0.2),
        _make_entry(image_id=1, jpeg_level=0.8),
    ]
    filtered = filter_by_jpeg_threshold(entries, max_jpeg=0.5)
    assert len(filtered) == 1
    assert filtered[0].image_id == 0


# ── ranking functions ─────────────────────────────────────────────────────────

def test_top_k_cleanest():
    entries = [_make_entry(image_id=i, sigma=float(i*10)) for i in range(5)]
    top = top_k_cleanest(entries, k=2)
    assert len(top) == 2
    assert top[0].sigma <= top[1].sigma


def test_top_k_cleanest_zero_k():
    entries = [_make_entry()]
    assert top_k_cleanest(entries, k=0) == []


def test_top_k_noisiest():
    entries = [_make_entry(image_id=i, sigma=float(i*10)) for i in range(5)]
    top = top_k_noisiest(entries, k=2)
    assert len(top) == 2
    assert top[0].sigma >= top[1].sigma


def test_best_snr_entry():
    entries = [
        _make_entry(image_id=0, snr_db=10.0),
        _make_entry(image_id=1, snr_db=40.0),
        _make_entry(image_id=2, snr_db=20.0),
    ]
    best = best_snr_entry(entries)
    assert best is not None
    assert best.image_id == 1


def test_best_snr_entry_empty():
    assert best_snr_entry([]) is None


# ── noise_stats_dict ──────────────────────────────────────────────────────────

def test_noise_stats_dict_empty():
    d = noise_stats_dict([])
    assert d["n"] == 0


def test_noise_stats_dict_basic():
    entries = [_make_entry(image_id=i, sigma=float(i+1)) for i in range(4)]
    d = noise_stats_dict(entries)
    assert d["n"] == 4
    assert "mean" in d
    assert "std" in d


# ── compare_noise_summaries ───────────────────────────────────────────────────

def test_compare_noise_summaries():
    a = summarise_noise_stats([_make_entry(sigma=5.0, snr_db=30.0)])
    b = summarise_noise_stats([_make_entry(sigma=10.0, snr_db=20.0)])
    diff = compare_noise_summaries(a, b)
    assert "delta_mean_sigma" in diff
    assert diff["delta_mean_sigma"] == pytest.approx(-5.0)
    assert diff["a_cleaner"] is True


# ── batch_summarise_noise_stats ───────────────────────────────────────────────

def test_batch_summarise_noise_stats():
    group1 = [_make_entry(image_id=0, sigma=5.0)]
    group2 = [_make_entry(image_id=1, sigma=15.0)]
    summaries = batch_summarise_noise_stats([group1, group2])
    assert len(summaries) == 2
    assert summaries[0].mean_sigma == pytest.approx(5.0)
    assert summaries[1].mean_sigma == pytest.approx(15.0)
