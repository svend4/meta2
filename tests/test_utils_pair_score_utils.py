"""Tests for puzzle_reconstruction.utils.pair_score_utils"""
import pytest
from puzzle_reconstruction.utils.pair_score_utils import (
    PairScoreConfig, PairScoreEntry, PairScoreSummary,
    make_pair_score_entry, entries_from_pair_results,
    summarise_pair_scores, filter_strong_pair_matches,
    filter_weak_pair_matches, filter_pair_by_score_range,
    filter_pair_by_channel, filter_pair_by_dominant_channel,
    top_k_pair_entries, best_pair_entry,
    pair_score_stats, compare_pair_summaries, batch_summarise_pair_scores,
)


def _make_entry(frag_i=0, frag_j=1, score=0.5, channels=None):
    return make_pair_score_entry(frag_i, frag_j, score, channels=channels or {})


# ── PairScoreConfig ───────────────────────────────────────────────────────────

def test_pair_score_config_defaults():
    cfg = PairScoreConfig()
    assert cfg.good_threshold == 0.7
    assert cfg.poor_threshold == 0.3


def test_pair_score_config_invalid_good_threshold_raises():
    with pytest.raises(ValueError):
        PairScoreConfig(good_threshold=1.5)


def test_pair_score_config_invalid_poor_threshold_raises():
    with pytest.raises(ValueError):
        PairScoreConfig(poor_threshold=-0.1)


# ── PairScoreEntry ────────────────────────────────────────────────────────────

def test_pair_score_entry_pair_key_canonical():
    e = _make_entry(frag_i=3, frag_j=1)
    assert e.pair_key == (1, 3)


def test_pair_score_entry_pair_key_already_ordered():
    e = _make_entry(frag_i=1, frag_j=3)
    assert e.pair_key == (1, 3)


def test_pair_score_entry_dominant_channel_none():
    e = _make_entry(channels={})
    assert e.dominant_channel is None


def test_pair_score_entry_dominant_channel():
    e = _make_entry(channels={"color": 0.9, "shape": 0.5, "texture": 0.7})
    assert e.dominant_channel == "color"


def test_pair_score_entry_is_strong_match_true():
    e = _make_entry(score=0.8)
    assert e.is_strong_match is True


def test_pair_score_entry_is_strong_match_false():
    e = _make_entry(score=0.5)
    assert e.is_strong_match is False


# ── make_pair_score_entry ─────────────────────────────────────────────────────

def test_make_pair_score_entry_returns_entry():
    e = make_pair_score_entry(0, 1, 0.75, channels={"c": 0.6}, method="test")
    assert isinstance(e, PairScoreEntry)
    assert e.method == "test"


def test_make_pair_score_entry_default_channels():
    e = make_pair_score_entry(0, 1, 0.5)
    assert e.channels == {}


# ── entries_from_pair_results ─────────────────────────────────────────────────

def test_entries_from_pair_results_basic():
    pairs = [(0, 1), (1, 2), (2, 3)]
    scores = [0.8, 0.5, 0.3]
    entries = entries_from_pair_results(pairs, scores)
    assert len(entries) == 3
    assert entries[0].score == pytest.approx(0.8)


def test_entries_from_pair_results_mismatch_raises():
    with pytest.raises(ValueError):
        entries_from_pair_results([(0, 1)], [0.5, 0.3])


def test_entries_from_pair_results_with_channels():
    pairs = [(0, 1)]
    scores = [0.7]
    chs = [{"color": 0.8, "shape": 0.6}]
    entries = entries_from_pair_results(pairs, scores, channel_lists=chs)
    assert entries[0].channels == {"color": 0.8, "shape": 0.6}


# ── summarise_pair_scores ─────────────────────────────────────────────────────

def test_summarise_pair_scores_empty():
    s = summarise_pair_scores([])
    assert s.n_entries == 0
    assert s.mean_score == 0.0


def test_summarise_pair_scores_basic():
    entries = [_make_entry(score=0.2), _make_entry(score=0.8), _make_entry(score=0.5)]
    s = summarise_pair_scores(entries)
    assert s.n_entries == 3
    assert abs(s.mean_score - (0.2 + 0.8 + 0.5) / 3) < 1e-9
    assert s.min_score == pytest.approx(0.2)
    assert s.max_score == pytest.approx(0.8)


def test_summarise_pair_scores_strong_matches():
    entries = [_make_entry(score=0.9), _make_entry(score=0.5), _make_entry(score=0.8)]
    s = summarise_pair_scores(entries)
    assert s.n_strong_matches == 2


def test_summarise_pair_scores_channel_means():
    entries = [
        _make_entry(channels={"color": 0.8}),
        _make_entry(channels={"color": 0.6}),
    ]
    s = summarise_pair_scores(entries)
    assert abs(s.channel_means["color"] - 0.7) < 1e-9


# ── filter functions ──────────────────────────────────────────────────────────

def test_filter_strong_pair_matches():
    entries = [_make_entry(score=0.3), _make_entry(score=0.8), _make_entry(score=0.9)]
    strong = filter_strong_pair_matches(entries, threshold=0.7)
    assert len(strong) == 2
    assert all(e.score >= 0.7 for e in strong)


def test_filter_weak_pair_matches():
    entries = [_make_entry(score=0.2), _make_entry(score=0.8)]
    weak = filter_weak_pair_matches(entries, threshold=0.3)
    assert len(weak) == 1
    assert weak[0].score < 0.3


def test_filter_pair_by_score_range():
    entries = [_make_entry(score=v) for v in [0.1, 0.5, 0.9]]
    filtered = filter_pair_by_score_range(entries, lo=0.4, hi=0.6)
    assert len(filtered) == 1
    assert filtered[0].score == pytest.approx(0.5)


def test_filter_pair_by_channel():
    entries = [
        _make_entry(channels={"color": 0.8}),
        _make_entry(channels={"color": 0.3}),
        _make_entry(channels={}),
    ]
    filtered = filter_pair_by_channel(entries, "color", min_val=0.5)
    assert len(filtered) == 1


def test_filter_pair_by_dominant_channel():
    entries = [
        _make_entry(channels={"color": 0.9, "shape": 0.5}),
        _make_entry(channels={"shape": 0.9, "color": 0.5}),
    ]
    filtered = filter_pair_by_dominant_channel(entries, "color")
    assert len(filtered) == 1
    assert filtered[0].dominant_channel == "color"


# ── ranking functions ─────────────────────────────────────────────────────────

def test_top_k_pair_entries():
    entries = [_make_entry(score=float(i)/10) for i in range(10)]
    top = top_k_pair_entries(entries, k=3)
    assert len(top) == 3
    assert top[0].score >= top[1].score >= top[2].score


def test_top_k_pair_entries_empty():
    assert top_k_pair_entries([], k=5) == []


def test_best_pair_entry_returns_max():
    entries = [_make_entry(score=0.3), _make_entry(score=0.9), _make_entry(score=0.5)]
    best = best_pair_entry(entries)
    assert best is not None
    assert best.score == pytest.approx(0.9)


def test_best_pair_entry_empty():
    assert best_pair_entry([]) is None


# ── pair_score_stats ──────────────────────────────────────────────────────────

def test_pair_score_stats_empty():
    d = pair_score_stats([])
    assert d["count"] == 0


def test_pair_score_stats_basic():
    entries = [_make_entry(score=0.8), _make_entry(score=0.6)]
    d = pair_score_stats(entries)
    assert d["count"] == 2
    assert "mean" in d
    assert "std" in d
    assert "min" in d
    assert "max" in d


# ── compare_pair_summaries ────────────────────────────────────────────────────

def test_compare_pair_summaries():
    s1 = summarise_pair_scores([_make_entry(score=0.8)])
    s2 = summarise_pair_scores([_make_entry(score=0.5)])
    diff = compare_pair_summaries(s1, s2)
    assert "d_mean_score" in diff
    assert diff["d_mean_score"] == pytest.approx(0.3, abs=1e-9)


# ── batch_summarise_pair_scores ───────────────────────────────────────────────

def test_batch_summarise_pair_scores():
    g1 = [_make_entry(score=0.9)]
    g2 = [_make_entry(score=0.4)]
    summaries = batch_summarise_pair_scores([g1, g2])
    assert len(summaries) == 2
    assert summaries[0].mean_score == pytest.approx(0.9)
    assert summaries[1].mean_score == pytest.approx(0.4)


def test_batch_summarise_pair_scores_empty():
    summaries = batch_summarise_pair_scores([])
    assert summaries == []
