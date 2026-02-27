"""Tests for puzzle_reconstruction.utils.consensus_score_utils"""
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


# ─── ConsensusScoreConfig ────────────────────────────────────────────────────

def test_consensus_score_config_defaults():
    cfg = ConsensusScoreConfig()
    assert cfg.min_vote_fraction == 0.5
    assert cfg.min_pairs == 1
    assert cfg.weight_by_score is False


def test_consensus_score_config_invalid_vote_fraction():
    with pytest.raises(ValueError, match="min_vote_fraction"):
        ConsensusScoreConfig(min_vote_fraction=1.5)


def test_consensus_score_config_invalid_min_pairs():
    with pytest.raises(ValueError, match="min_pairs"):
        ConsensusScoreConfig(min_pairs=0)


def test_consensus_score_config_boundary():
    cfg = ConsensusScoreConfig(min_vote_fraction=0.0)
    assert cfg.min_vote_fraction == 0.0


# ─── ConsensusScoreEntry ─────────────────────────────────────────────────────

def test_consensus_score_entry_vote_fraction():
    e = ConsensusScoreEntry(pair=(0, 1), vote_count=3, n_methods=5, is_consensus=True)
    assert e.vote_fraction == pytest.approx(0.6)


def test_consensus_score_entry_zero_methods():
    e = ConsensusScoreEntry(pair=(0, 1), vote_count=0, n_methods=0, is_consensus=False)
    assert e.vote_fraction == 0.0


def test_consensus_score_entry_repr():
    e = ConsensusScoreEntry(pair=(1, 2), vote_count=2, n_methods=4, is_consensus=False)
    r = repr(e)
    assert "ConsensusScoreEntry" in r


# ─── make_consensus_entry ────────────────────────────────────────────────────

def test_make_consensus_entry_above_threshold():
    e = make_consensus_entry((0, 1), vote_count=3, n_methods=4, threshold=0.5)
    assert e.is_consensus is True
    assert e.vote_fraction == pytest.approx(0.75)


def test_make_consensus_entry_below_threshold():
    e = make_consensus_entry((0, 1), vote_count=1, n_methods=4, threshold=0.5)
    assert e.is_consensus is False


def test_make_consensus_entry_at_threshold():
    e = make_consensus_entry((0, 1), vote_count=2, n_methods=4, threshold=0.5)
    assert e.is_consensus is True  # 2/4 = 0.5 >= 0.5


def test_make_consensus_entry_meta():
    e = make_consensus_entry((0, 1), 2, 4, meta={"info": "test"})
    assert e.meta["info"] == "test"


def test_make_consensus_entry_zero_methods():
    e = make_consensus_entry((0, 1), 0, 0)
    assert e.is_consensus is False


# ─── entries_from_votes ───────────────────────────────────────────────────────

def test_entries_from_votes_basic():
    pair_votes = {
        frozenset({0, 1}): 3,
        frozenset({2, 3}): 1,
    }
    entries = entries_from_votes(pair_votes, n_methods=4, threshold=0.5)
    assert len(entries) == 2
    fracs = [e.vote_fraction for e in entries]
    assert any(abs(f - 0.75) < 1e-9 for f in fracs)
    assert any(abs(f - 0.25) < 1e-9 for f in fracs)


def test_entries_from_votes_empty():
    entries = entries_from_votes({}, n_methods=3)
    assert entries == []


# ─── summarise_consensus ─────────────────────────────────────────────────────

def test_summarise_consensus_empty():
    s = summarise_consensus([])
    assert s.n_pairs == 0
    assert s.mean_vote_fraction == 0.0
    assert s.agreement_score == 0.0


def test_summarise_consensus_basic():
    entries = [
        make_consensus_entry((0, 1), 4, 4, threshold=0.5),  # consensus
        make_consensus_entry((2, 3), 1, 4, threshold=0.5),  # not consensus
    ]
    s = summarise_consensus(entries)
    assert s.n_pairs == 2
    assert s.n_consensus == 1
    assert s.agreement_score == pytest.approx(0.5)
    assert s.mean_vote_fraction == pytest.approx(0.625)


# ─── filter_consensus_pairs / filter_non_consensus ───────────────────────────

def test_filter_consensus_pairs():
    entries = [
        make_consensus_entry((0, 1), 3, 4),
        make_consensus_entry((2, 3), 1, 4),
    ]
    result = filter_consensus_pairs(entries)
    assert all(e.is_consensus for e in result)
    assert len(result) == 1


def test_filter_non_consensus():
    entries = [
        make_consensus_entry((0, 1), 4, 4),
        make_consensus_entry((2, 3), 1, 4),
    ]
    result = filter_non_consensus(entries)
    assert all(not e.is_consensus for e in result)
    assert len(result) == 1


# ─── filter_by_vote_fraction ─────────────────────────────────────────────────

def test_filter_by_vote_fraction():
    entries = [
        make_consensus_entry((0, 1), 1, 4),   # 0.25
        make_consensus_entry((2, 3), 2, 4),   # 0.5
        make_consensus_entry((4, 5), 4, 4),   # 1.0
    ]
    result = filter_by_vote_fraction(entries, min_fraction=0.5)
    assert len(result) == 2


# ─── top_k_consensus_entries ─────────────────────────────────────────────────

def test_top_k_consensus_entries():
    entries = [
        make_consensus_entry((0, 1), 1, 4),
        make_consensus_entry((2, 3), 3, 4),
        make_consensus_entry((4, 5), 4, 4),
    ]
    top2 = top_k_consensus_entries(entries, k=2)
    assert len(top2) == 2
    assert top2[0].vote_fraction == pytest.approx(1.0)


def test_top_k_consensus_entries_k_zero():
    entries = [make_consensus_entry((0, 1), 2, 4)]
    result = top_k_consensus_entries(entries, k=0)
    assert result == []


# ─── consensus_score_stats ───────────────────────────────────────────────────

def test_consensus_score_stats_empty():
    stats = consensus_score_stats([])
    assert stats["count"] == 0
    assert stats["mean_fraction"] == 0.0


def test_consensus_score_stats_basic():
    entries = [
        make_consensus_entry((0, 1), 2, 4),   # 0.5
        make_consensus_entry((2, 3), 4, 4),   # 1.0
    ]
    stats = consensus_score_stats(entries)
    assert stats["count"] == 2
    assert stats["mean_fraction"] == pytest.approx(0.75)
    assert stats["min_fraction"] == pytest.approx(0.5)
    assert stats["max_fraction"] == pytest.approx(1.0)
    assert stats["n_consensus"] == 2
    assert stats["n_non_consensus"] == 0


# ─── agreement_score ─────────────────────────────────────────────────────────

def test_agreement_score_empty():
    assert agreement_score([]) == 0.0


def test_agreement_score_all_consensus():
    entries = [
        make_consensus_entry((0, 1), 4, 4),
        make_consensus_entry((2, 3), 3, 4),
    ]
    assert agreement_score(entries) == pytest.approx(1.0)


def test_agreement_score_half():
    entries = [
        make_consensus_entry((0, 1), 4, 4),
        make_consensus_entry((2, 3), 1, 4),
    ]
    assert agreement_score(entries) == pytest.approx(0.5)


# ─── compare_consensus ───────────────────────────────────────────────────────

def test_compare_consensus():
    entries_a = [make_consensus_entry((0, 1), 3, 4), make_consensus_entry((2, 3), 3, 4)]
    entries_b = [make_consensus_entry((4, 5), 1, 4)]
    s_a = summarise_consensus(entries_a)
    s_b = summarise_consensus(entries_b)
    diff = compare_consensus(s_a, s_b)
    assert diff["n_pairs_delta"] == 1
    assert diff["n_consensus_delta"] == 2


# ─── batch_summarise_consensus ───────────────────────────────────────────────

def test_batch_summarise_consensus():
    votes1 = {frozenset({0, 1}): 3}
    votes2 = {frozenset({2, 3}): 1}
    results = batch_summarise_consensus([votes1, votes2], [4, 4])
    assert len(results) == 2
    assert results[0].n_pairs == 1
    assert results[1].n_pairs == 1
