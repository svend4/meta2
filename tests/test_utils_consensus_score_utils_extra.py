"""Extra tests for puzzle_reconstruction/utils/consensus_score_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.consensus_score_utils import (
    ConsensusScoreConfig,
    ConsensusScoreEntry,
    ConsensusSummary,
    make_consensus_entry,
    entries_from_votes,
    summarise_consensus,
    filter_consensus_pairs,
    filter_non_consensus,
    filter_by_vote_fraction,
    top_k_consensus_entries,
    consensus_score_stats,
    agreement_score,
    compare_consensus,
    batch_summarise_consensus,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _entry(pair=(0, 1), votes=3, n=4, consensus=True) -> ConsensusScoreEntry:
    return ConsensusScoreEntry(
        pair=pair, vote_count=votes, n_methods=n, is_consensus=consensus,
    )


def _entries(n=5) -> list:
    return [
        _entry(pair=(i, i+1), votes=i+1, n=5,
               consensus=(i+1)/5 >= 0.5)
        for i in range(n)
    ]


def _summary(entries=None) -> ConsensusSummary:
    return summarise_consensus(entries or _entries(4))


# ─── ConsensusScoreConfig ─────────────────────────────────────────────────────

class TestConsensusScoreConfigExtra:
    def test_default_min_vote_fraction(self):
        assert ConsensusScoreConfig().min_vote_fraction == pytest.approx(0.5)

    def test_default_min_pairs(self):
        assert ConsensusScoreConfig().min_pairs == 1

    def test_default_weight_by_score(self):
        assert ConsensusScoreConfig().weight_by_score is False

    def test_min_vote_fraction_below_zero_raises(self):
        with pytest.raises(ValueError):
            ConsensusScoreConfig(min_vote_fraction=-0.1)

    def test_min_vote_fraction_above_one_raises(self):
        with pytest.raises(ValueError):
            ConsensusScoreConfig(min_vote_fraction=1.1)

    def test_min_pairs_zero_raises(self):
        with pytest.raises(ValueError):
            ConsensusScoreConfig(min_pairs=0)

    def test_min_pairs_negative_raises(self):
        with pytest.raises(ValueError):
            ConsensusScoreConfig(min_pairs=-1)

    def test_valid_custom_values(self):
        cfg = ConsensusScoreConfig(min_vote_fraction=0.6, min_pairs=3)
        assert cfg.min_vote_fraction == pytest.approx(0.6)
        assert cfg.min_pairs == 3


# ─── ConsensusScoreEntry ──────────────────────────────────────────────────────

class TestConsensusScoreEntryExtra:
    def test_stores_pair(self):
        assert _entry(pair=(2, 5)).pair == (2, 5)

    def test_stores_vote_count(self):
        assert _entry(votes=3).vote_count == 3

    def test_stores_n_methods(self):
        assert _entry(n=6).n_methods == 6

    def test_stores_is_consensus(self):
        assert _entry(consensus=False).is_consensus is False

    def test_vote_fraction_computed(self):
        e = _entry(votes=3, n=4)
        assert e.vote_fraction == pytest.approx(0.75)

    def test_vote_fraction_zero_methods(self):
        e = ConsensusScoreEntry(pair=(0, 1), vote_count=0, n_methods=0, is_consensus=False)
        assert e.vote_fraction == pytest.approx(0.0)

    def test_default_meta_empty(self):
        assert _entry().meta == {}

    def test_repr_is_str(self):
        assert isinstance(repr(_entry()), str)


# ─── make_consensus_entry ─────────────────────────────────────────────────────

class TestMakeConsensusEntryExtra:
    def test_returns_entry(self):
        e = make_consensus_entry((0, 1), 3, 4)
        assert isinstance(e, ConsensusScoreEntry)

    def test_is_consensus_above_threshold(self):
        e = make_consensus_entry((0, 1), 3, 4, threshold=0.5)
        assert e.is_consensus is True

    def test_is_not_consensus_below_threshold(self):
        e = make_consensus_entry((0, 1), 1, 4, threshold=0.5)
        assert e.is_consensus is False

    def test_none_meta_empty(self):
        e = make_consensus_entry((0, 1), 2, 3, meta=None)
        assert e.meta == {}


# ─── entries_from_votes ───────────────────────────────────────────────────────

class TestEntriesFromVotesExtra:
    def test_returns_list(self):
        votes = {frozenset([0, 1]): 3}
        result = entries_from_votes(votes, n_methods=4)
        assert isinstance(result, list)

    def test_length_matches(self):
        votes = {frozenset([0, 1]): 3, frozenset([1, 2]): 2}
        result = entries_from_votes(votes, n_methods=4)
        assert len(result) == 2

    def test_empty_votes(self):
        assert entries_from_votes({}, n_methods=4) == []

    def test_all_are_entries(self):
        votes = {frozenset([0, 1]): 3}
        for e in entries_from_votes(votes, n_methods=4):
            assert isinstance(e, ConsensusScoreEntry)


# ─── summarise_consensus ──────────────────────────────────────────────────────

class TestSummariseConsensusExtra:
    def test_returns_summary(self):
        assert isinstance(summarise_consensus(_entries()), ConsensusSummary)

    def test_n_pairs_correct(self):
        assert summarise_consensus(_entries(5)).n_pairs == 5

    def test_empty_entries(self):
        s = summarise_consensus([])
        assert s.n_pairs == 0 and s.agreement_score == pytest.approx(0.0)

    def test_agreement_in_range(self):
        s = summarise_consensus(_entries(4))
        assert 0.0 <= s.agreement_score <= 1.0

    def test_repr_is_str(self):
        assert isinstance(repr(_summary()), str)


# ─── filter functions ─────────────────────────────────────────────────────────

class TestFilterConsensusExtra:
    def test_filter_consensus_only_true(self):
        entries = [_entry(consensus=True), _entry(consensus=False)]
        result = filter_consensus_pairs(entries)
        assert all(e.is_consensus for e in result)

    def test_filter_non_consensus_only_false(self):
        entries = [_entry(consensus=True), _entry(consensus=False)]
        result = filter_non_consensus(entries)
        assert all(not e.is_consensus for e in result)

    def test_filter_by_vote_fraction(self):
        entries = [_entry(votes=1, n=4), _entry(votes=3, n=4)]
        result = filter_by_vote_fraction(entries, min_fraction=0.5)
        assert all(e.vote_fraction >= 0.5 for e in result)

    def test_empty_input(self):
        assert filter_consensus_pairs([]) == []


# ─── top_k_consensus_entries ──────────────────────────────────────────────────

class TestTopKConsensusEntriesExtra:
    def test_returns_list(self):
        assert isinstance(top_k_consensus_entries(_entries(), 3), list)

    def test_length_at_most_k(self):
        result = top_k_consensus_entries(_entries(5), 3)
        assert len(result) <= 3

    def test_k_larger_than_n(self):
        result = top_k_consensus_entries(_entries(3), 10)
        assert len(result) == 3

    def test_empty_input(self):
        assert top_k_consensus_entries([], 3) == []


# ─── consensus_score_stats ────────────────────────────────────────────────────

class TestConsensusScoreStatsExtra:
    def test_returns_dict(self):
        assert isinstance(consensus_score_stats(_entries()), dict)

    def test_keys_present(self):
        stats = consensus_score_stats(_entries(4))
        for k in ("count", "mean_fraction", "std_fraction",
                  "min_fraction", "max_fraction", "n_consensus", "n_non_consensus"):
            assert k in stats

    def test_count_correct(self):
        assert consensus_score_stats(_entries(6))["count"] == 6

    def test_empty_entries(self):
        stats = consensus_score_stats([])
        assert stats["count"] == 0


# ─── agreement_score ──────────────────────────────────────────────────────────

class TestAgreementScoreExtra:
    def test_all_consensus_is_one(self):
        entries = [_entry(consensus=True)] * 4
        assert agreement_score(entries) == pytest.approx(1.0)

    def test_none_consensus_is_zero(self):
        entries = [_entry(consensus=False)] * 4
        assert agreement_score(entries) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        assert agreement_score([]) == pytest.approx(0.0)

    def test_in_range(self):
        s = agreement_score(_entries(5))
        assert 0.0 <= s <= 1.0


# ─── compare_consensus ────────────────────────────────────────────────────────

class TestCompareConsensusExtra:
    def test_returns_dict(self):
        s = _summary()
        assert isinstance(compare_consensus(s, s), dict)

    def test_keys_present(self):
        s = _summary()
        d = compare_consensus(s, s)
        for k in ("n_pairs_delta", "n_consensus_delta", "agreement_delta",
                  "mean_fraction_delta"):
            assert k in d

    def test_identical_zero_delta(self):
        s = _summary()
        d = compare_consensus(s, s)
        assert d["agreement_delta"] == pytest.approx(0.0)


# ─── batch_summarise_consensus ────────────────────────────────────────────────

class TestBatchSummariseConsensusExtra:
    def test_returns_list(self):
        result = batch_summarise_consensus(
            [{frozenset([0, 1]): 3}], [4]
        )
        assert isinstance(result, list)

    def test_length_matches(self):
        result = batch_summarise_consensus(
            [{frozenset([0, 1]): 3}, {frozenset([1, 2]): 2}],
            [4, 4],
        )
        assert len(result) == 2

    def test_each_is_summary(self):
        for s in batch_summarise_consensus([{frozenset([0, 1]): 3}], [4]):
            assert isinstance(s, ConsensusSummary)

    def test_empty_returns_empty(self):
        assert batch_summarise_consensus([], []) == []
