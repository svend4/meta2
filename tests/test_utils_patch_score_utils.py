"""Tests for puzzle_reconstruction.utils.patch_score_utils"""
import pytest
import numpy as np

from puzzle_reconstruction.utils.patch_score_utils import (
    PatchScoreConfig,
    PatchScoreEntry,
    PatchScoreSummary,
    make_patch_entry,
    entries_from_patch_matches,
    summarise_patch_scores,
    filter_good_patch_scores,
    filter_poor_patch_scores,
    filter_patch_by_score_range,
    filter_by_side_pair,
    filter_by_ncc_range,
    top_k_patch_entries,
    best_patch_entry,
    patch_score_stats,
    compare_patch_summaries,
    batch_summarise_patch_scores,
)


# ── PatchScoreConfig ─────────────────────────────────────────────────────────

def test_config_defaults():
    cfg = PatchScoreConfig()
    assert cfg.min_score == 0.0
    assert cfg.max_pairs == 1000
    assert cfg.method == "total"


def test_config_negative_min_score_raises():
    with pytest.raises(ValueError):
        PatchScoreConfig(min_score=-0.1)


def test_config_zero_max_pairs_raises():
    with pytest.raises(ValueError):
        PatchScoreConfig(max_pairs=0)


def test_config_negative_max_pairs_raises():
    with pytest.raises(ValueError):
        PatchScoreConfig(max_pairs=-5)


def test_config_invalid_method_raises():
    with pytest.raises(ValueError):
        PatchScoreConfig(method="invalid_method")


def test_config_valid_methods():
    for m in ("total", "ncc", "ssd", "ssim"):
        cfg = PatchScoreConfig(method=m)
        assert cfg.method == m


# ── PatchScoreEntry ──────────────────────────────────────────────────────────

def _make_entry(pair_id=0, idx1=0, idx2=1, side1=0, side2=1,
                ncc=0.5, ssd=0.5, ssim=0.5, total_score=0.5):
    return PatchScoreEntry(
        pair_id=pair_id, idx1=idx1, idx2=idx2,
        side1=side1, side2=side2,
        ncc=ncc, ssd=ssd, ssim=ssim, total_score=total_score,
    )


def test_entry_pair_property():
    e = _make_entry(idx1=2, idx2=5)
    assert e.pair == (2, 5)


def test_entry_is_good_true():
    e = _make_entry(total_score=0.8)
    assert e.is_good is True


def test_entry_is_good_false():
    e = _make_entry(total_score=0.3)
    assert e.is_good is False


def test_entry_is_good_boundary():
    e_at_boundary = _make_entry(total_score=0.5)
    assert e_at_boundary.is_good is False


def test_entry_negative_pair_id_raises():
    with pytest.raises(ValueError):
        PatchScoreEntry(pair_id=-1, idx1=0, idx2=1,
                        side1=0, side2=0, ncc=0.5, ssd=0.5, ssim=0.5, total_score=0.5)


def test_entry_negative_idx_raises():
    with pytest.raises(ValueError):
        PatchScoreEntry(pair_id=0, idx1=-1, idx2=1,
                        side1=0, side2=0, ncc=0.5, ssd=0.5, ssim=0.5, total_score=0.5)


def test_entry_ssd_out_of_range_raises():
    with pytest.raises(ValueError):
        PatchScoreEntry(pair_id=0, idx1=0, idx2=1,
                        side1=0, side2=0, ncc=0.5, ssd=1.5, ssim=0.5, total_score=0.5)


def test_entry_total_score_out_of_range_raises():
    with pytest.raises(ValueError):
        PatchScoreEntry(pair_id=0, idx1=0, idx2=1,
                        side1=0, side2=0, ncc=0.5, ssd=0.5, ssim=0.5, total_score=1.5)


# ── make_patch_entry ─────────────────────────────────────────────────────────

def test_make_patch_entry_basic():
    e = make_patch_entry(0, 1, 2, 0, 1, 0.6, 0.4, 0.7, 0.6)
    assert e.pair_id == 0
    assert e.total_score == pytest.approx(0.6)


def test_make_patch_entry_with_meta():
    e = make_patch_entry(0, 0, 1, 0, 0, 0.5, 0.5, 0.5, 0.5, meta={"key": "val"})
    assert e.meta.get("key") == "val"


def test_make_patch_entry_default_rank():
    e = make_patch_entry(0, 0, 1, 0, 0, 0.5, 0.5, 0.5, 0.5)
    assert e.rank == 0


# ── entries_from_patch_matches ───────────────────────────────────────────────

class _FakePM:
    def __init__(self, idx1, idx2, side1, side2, ncc, ssd, ssim, total_score):
        self.idx1 = idx1; self.idx2 = idx2
        self.side1 = side1; self.side2 = side2
        self.ncc = ncc; self.ssd = ssd; self.ssim = ssim
        self.total_score = total_score


def test_entries_from_patch_matches_count():
    pms = [_FakePM(0, 1, 0, 1, 0.5, 0.5, 0.5, 0.6) for _ in range(5)]
    entries = entries_from_patch_matches(pms)
    assert len(entries) == 5


def test_entries_from_patch_matches_ranks_assigned():
    pms = [_FakePM(0, i, 0, 0, 0.5, 0.5, 0.5, float(i) / 10) for i in range(5)]
    entries = entries_from_patch_matches(pms)
    ranks = [e.rank for e in entries]
    assert all(r > 0 for r in ranks)


def test_entries_from_patch_matches_empty():
    entries = entries_from_patch_matches([])
    assert entries == []


# ── summarise_patch_scores ───────────────────────────────────────────────────

def test_summarise_patch_scores_empty():
    s = summarise_patch_scores([])
    assert s.n_total == 0
    assert s.mean_total == 0.0


def test_summarise_patch_scores_basic():
    entries = [_make_entry(total_score=0.6), _make_entry(total_score=0.4)]
    s = summarise_patch_scores(entries)
    assert s.n_total == 2
    assert s.n_good == 1
    assert s.n_poor == 1
    assert s.mean_total == pytest.approx(0.5)


def test_summarise_patch_scores_repr():
    entries = [_make_entry(total_score=0.7)]
    s = summarise_patch_scores(entries)
    r = repr(s)
    assert "n_total=1" in r


def test_summarise_patch_scores_max_min():
    entries = [_make_entry(total_score=0.2), _make_entry(total_score=0.9)]
    s = summarise_patch_scores(entries)
    assert s.max_total == pytest.approx(0.9)
    assert s.min_total == pytest.approx(0.2)


# ── filter functions ─────────────────────────────────────────────────────────

def test_filter_good_patch_scores():
    entries = [_make_entry(total_score=0.8), _make_entry(total_score=0.3)]
    good = filter_good_patch_scores(entries)
    assert len(good) == 1
    assert good[0].total_score == 0.8


def test_filter_poor_patch_scores():
    entries = [_make_entry(total_score=0.8), _make_entry(total_score=0.3)]
    poor = filter_poor_patch_scores(entries)
    assert len(poor) == 1
    assert poor[0].total_score == 0.3


def test_filter_patch_by_score_range():
    entries = [_make_entry(total_score=v) for v in [0.1, 0.4, 0.7, 1.0]]
    result = filter_patch_by_score_range(entries, lo=0.3, hi=0.8)
    assert all(0.3 <= e.total_score <= 0.8 for e in result)


def test_filter_by_side_pair():
    entries = [
        _make_entry(side1=0, side2=1),
        _make_entry(side1=1, side2=2),
        _make_entry(side1=0, side2=1),
    ]
    result = filter_by_side_pair(entries, side1=0, side2=1)
    assert len(result) == 2


def test_filter_by_ncc_range():
    entries = [_make_entry(ncc=v) for v in [0.1, 0.5, 0.9]]
    result = filter_by_ncc_range(entries, lo=0.4, hi=0.6)
    assert len(result) == 1
    assert result[0].ncc == pytest.approx(0.5)


# ── ranking ──────────────────────────────────────────────────────────────────

def test_top_k_patch_entries_returns_k():
    entries = [_make_entry(total_score=float(i)/10) for i in range(8)]
    top = top_k_patch_entries(entries, k=3)
    assert len(top) == 3


def test_top_k_patch_entries_sorted():
    entries = [_make_entry(total_score=float(i)/10) for i in range(5)]
    top = top_k_patch_entries(entries, k=5)
    scores = [e.total_score for e in top]
    assert scores == sorted(scores, reverse=True)


def test_top_k_patch_entries_zero():
    entries = [_make_entry()]
    assert top_k_patch_entries(entries, k=0) == []


def test_best_patch_entry_basic():
    entries = [_make_entry(total_score=0.3), _make_entry(total_score=0.9)]
    best = best_patch_entry(entries)
    assert best.total_score == pytest.approx(0.9)


def test_best_patch_entry_empty():
    assert best_patch_entry([]) is None


# ── patch_score_stats ────────────────────────────────────────────────────────

def test_patch_score_stats_empty():
    stats = patch_score_stats([])
    assert stats["n"] == 0
    assert stats["mean"] == 0.0


def test_patch_score_stats_keys():
    entries = [_make_entry(total_score=0.5)]
    stats = patch_score_stats(entries)
    for key in ("n", "mean", "min", "max", "std"):
        assert key in stats


def test_patch_score_stats_std_zero_identical():
    entries = [_make_entry(total_score=0.5) for _ in range(5)]
    stats = patch_score_stats(entries)
    assert stats["std"] == pytest.approx(0.0, abs=1e-10)


# ── compare_patch_summaries ───────────────────────────────────────────────────

def test_compare_patch_summaries_keys():
    e_a = [_make_entry(total_score=0.7)]
    e_b = [_make_entry(total_score=0.5)]
    s_a = summarise_patch_scores(e_a)
    s_b = summarise_patch_scores(e_b)
    comp = compare_patch_summaries(s_a, s_b)
    assert "delta_mean_total" in comp
    assert "a_better" in comp
    assert comp["a_better"] is True


# ── batch_summarise_patch_scores ─────────────────────────────────────────────

def test_batch_summarise_patch_scores():
    groups = [[_make_entry(total_score=0.6)], [_make_entry(total_score=0.4)]]
    summaries = batch_summarise_patch_scores(groups)
    assert len(summaries) == 2
    assert summaries[0].n_total == 1


def test_batch_summarise_patch_scores_empty():
    assert batch_summarise_patch_scores([]) == []
