"""Extra tests for puzzle_reconstruction/utils/pair_score_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.pair_score_utils import (
    PairScoreConfig,
    PairScoreEntry,
    PairScoreSummary,
    make_pair_score_entry,
    entries_from_pair_results,
    summarise_pair_scores,
    filter_strong_pair_matches,
    filter_weak_pair_matches,
    filter_pair_by_score_range,
    filter_pair_by_channel,
    filter_pair_by_dominant_channel,
    top_k_pair_entries,
    best_pair_entry,
    pair_score_stats,
    compare_pair_summaries,
    batch_summarise_pair_scores,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _entry(frag_i=0, frag_j=1, score=0.7,
           channels=None, method="pair_scorer") -> PairScoreEntry:
    return PairScoreEntry(frag_i=frag_i, frag_j=frag_j, score=score,
                          channels=channels or {}, method=method)


# ─── PairScoreConfig ──────────────────────────────────────────────────────────

class TestPairScoreConfigExtra:
    def test_default_good_threshold(self):
        assert PairScoreConfig().good_threshold == pytest.approx(0.7)

    def test_default_poor_threshold(self):
        assert PairScoreConfig().poor_threshold == pytest.approx(0.3)

    def test_good_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError):
            PairScoreConfig(good_threshold=1.5)

    def test_poor_threshold_negative_raises(self):
        with pytest.raises(ValueError):
            PairScoreConfig(poor_threshold=-0.1)


# ─── PairScoreEntry ───────────────────────────────────────────────────────────

class TestPairScoreEntryExtra:
    def test_pair_key_ordered(self):
        e = _entry(frag_i=3, frag_j=1)
        assert e.pair_key == (1, 3)

    def test_is_strong_match_true(self):
        e = _entry(score=0.8)
        assert e.is_strong_match is True

    def test_is_strong_match_false(self):
        e = _entry(score=0.5)
        assert e.is_strong_match is False

    def test_dominant_channel_none_when_empty(self):
        e = _entry(channels={})
        assert e.dominant_channel is None

    def test_dominant_channel(self):
        e = _entry(channels={"color": 0.9, "edge": 0.3})
        assert e.dominant_channel == "color"

    def test_stores_method(self):
        e = _entry(method="custom")
        assert e.method == "custom"


# ─── make_pair_score_entry ────────────────────────────────────────────────────

class TestMakePairScoreEntryExtra:
    def test_returns_entry(self):
        e = make_pair_score_entry(0, 1, 0.7)
        assert isinstance(e, PairScoreEntry)

    def test_channels_stored(self):
        e = make_pair_score_entry(0, 1, 0.7, channels={"edge": 0.8})
        assert e.channels["edge"] == pytest.approx(0.8)

    def test_empty_channels_default(self):
        e = make_pair_score_entry(0, 1, 0.5)
        assert e.channels == {}


# ─── entries_from_pair_results ────────────────────────────────────────────────

class TestEntriesFromPairResultsExtra:
    def test_returns_list(self):
        result = entries_from_pair_results([(0, 1)], [0.8])
        assert isinstance(result, list) and len(result) == 1

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            entries_from_pair_results([(0, 1), (1, 2)], [0.8])

    def test_score_stored(self):
        entries = entries_from_pair_results([(0, 1)], [0.75])
        assert entries[0].score == pytest.approx(0.75)

    def test_empty_input(self):
        assert entries_from_pair_results([], []) == []


# ─── summarise_pair_scores ────────────────────────────────────────────────────

class TestSummarisePairScoresExtra:
    def test_empty_returns_summary(self):
        s = summarise_pair_scores([])
        assert s.n_entries == 0

    def test_strong_matches_counted(self):
        entries = [_entry(score=0.8), _entry(score=0.4), _entry(score=0.9)]
        s = summarise_pair_scores(entries)
        assert s.n_strong_matches == 2

    def test_channel_means_computed(self):
        entries = [_entry(channels={"edge": 0.8}),
                   _entry(channels={"edge": 0.6})]
        s = summarise_pair_scores(entries)
        assert s.channel_means["edge"] == pytest.approx(0.7)

    def test_mean_score(self):
        entries = [_entry(score=0.4), _entry(score=0.8)]
        s = summarise_pair_scores(entries)
        assert s.mean_score == pytest.approx(0.6)


# ─── filter helpers ───────────────────────────────────────────────────────────

class TestFilterPairScoreExtra:
    def test_filter_strong(self):
        entries = [_entry(score=0.8), _entry(score=0.3)]
        result = filter_strong_pair_matches(entries, 0.7)
        assert len(result) == 1

    def test_filter_weak(self):
        entries = [_entry(score=0.8), _entry(score=0.2)]
        result = filter_weak_pair_matches(entries, 0.3)
        assert len(result) == 1

    def test_filter_score_range(self):
        entries = [_entry(score=0.2), _entry(score=0.5), _entry(score=0.9)]
        result = filter_pair_by_score_range(entries, 0.4, 0.7)
        assert len(result) == 1 and result[0].score == pytest.approx(0.5)

    def test_filter_by_channel(self):
        entries = [_entry(channels={"edge": 0.9}),
                   _entry(channels={"edge": 0.2})]
        result = filter_pair_by_channel(entries, "edge", 0.5)
        assert len(result) == 1

    def test_filter_by_dominant_channel(self):
        entries = [_entry(channels={"edge": 0.9, "color": 0.3}),
                   _entry(channels={"edge": 0.3, "color": 0.9})]
        result = filter_pair_by_dominant_channel(entries, "edge")
        assert len(result) == 1


# ─── top_k / best ─────────────────────────────────────────────────────────────

class TestRankPairScoreExtra:
    def test_top_k_entries(self):
        entries = [_entry(score=0.3), _entry(score=0.9), _entry(score=0.6)]
        top = top_k_pair_entries(entries, 2)
        assert top[0].score == pytest.approx(0.9)
        assert len(top) == 2

    def test_best_pair_entry_empty(self):
        assert best_pair_entry([]) is None

    def test_best_pair_entry(self):
        entries = [_entry(score=0.3), _entry(score=0.9)]
        best = best_pair_entry(entries)
        assert best.score == pytest.approx(0.9)


# ─── pair_score_stats ─────────────────────────────────────────────────────────

class TestPairScoreStatsExtra:
    def test_empty_returns_zeros(self):
        s = pair_score_stats([])
        assert s["count"] == 0

    def test_count_and_min_max(self):
        entries = [_entry(score=0.2), _entry(score=0.8)]
        s = pair_score_stats(entries)
        assert s["count"] == 2
        assert s["min"] == pytest.approx(0.2)
        assert s["max"] == pytest.approx(0.8)

    def test_n_strong(self):
        entries = [_entry(score=0.8), _entry(score=0.5)]
        s = pair_score_stats(entries)
        assert s["n_strong"] == 1


# ─── compare_pair_summaries ───────────────────────────────────────────────────

class TestComparePairSummariesExtra:
    def test_returns_dict(self):
        s = summarise_pair_scores([_entry()])
        d = compare_pair_summaries(s, s)
        assert isinstance(d, dict)

    def test_delta_zero_identical(self):
        s = summarise_pair_scores([_entry()])
        d = compare_pair_summaries(s, s)
        assert d["d_mean_score"] == pytest.approx(0.0)

    def test_batch_length(self):
        result = batch_summarise_pair_scores([[_entry()], [_entry(), _entry()]])
        assert len(result) == 2
