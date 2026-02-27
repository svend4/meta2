"""Tests for puzzle_reconstruction.utils.rank_result_utils."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.rank_result_utils import (
    RankResultConfig,
    RankResultEntry,
    RankResultSummary,
    make_rank_result_entry,
    entries_from_ranked_pairs,
    summarise_rank_results,
    filter_high_rank_entries,
    filter_low_rank_entries,
    filter_by_rank_position,
    filter_rank_by_score_range,
    filter_rank_by_dominant_channel,
    top_k_rank_entries,
    best_rank_entry,
    rerank_entries,
    rank_result_stats,
    compare_rank_summaries,
    batch_summarise_rank_results,
)

np.random.seed(99)


# ─── RankResultConfig ────────────────────────────────────────────────────────

def test_config_defaults():
    cfg = RankResultConfig()
    assert cfg.good_threshold == pytest.approx(0.7)
    assert cfg.poor_threshold == pytest.approx(0.3)
    assert cfg.top_k == 10


def test_config_invalid_good_threshold():
    with pytest.raises(ValueError):
        RankResultConfig(good_threshold=1.5)


def test_config_invalid_poor_threshold():
    with pytest.raises(ValueError):
        RankResultConfig(poor_threshold=-0.1)


def test_config_invalid_top_k():
    with pytest.raises(ValueError):
        RankResultConfig(top_k=-1)


def test_config_top_k_zero():
    cfg = RankResultConfig(top_k=0)
    assert cfg.top_k == 0


# ─── RankResultEntry ─────────────────────────────────────────────────────────

def test_entry_pair_key_sorted():
    e = make_rank_result_entry(5, 2, 0.8, 1)
    assert e.pair_key == (2, 5)


def test_entry_pair_key_same_order():
    e = make_rank_result_entry(1, 3, 0.5, 2)
    assert e.pair_key == (1, 3)


def test_entry_is_top_match_true():
    e = make_rank_result_entry(0, 1, 0.9, 1)
    assert e.is_top_match is True


def test_entry_is_top_match_false():
    e = make_rank_result_entry(0, 1, 0.9, 2)
    assert e.is_top_match is False


def test_entry_dominant_channel():
    e = make_rank_result_entry(0, 1, 0.8, 1,
                                channel_scores={"R": 0.9, "G": 0.5, "B": 0.3})
    assert e.dominant_channel == "R"


def test_entry_dominant_channel_none():
    e = make_rank_result_entry(0, 1, 0.8, 1)
    assert e.dominant_channel is None


# ─── make_rank_result_entry ──────────────────────────────────────────────────

def test_make_rank_result_entry_fields():
    e = make_rank_result_entry(1, 2, 0.75, 3, method="custom",
                                params={"alpha": 0.5})
    assert e.frag_i == 1
    assert e.frag_j == 2
    assert e.score == pytest.approx(0.75)
    assert e.rank == 3
    assert e.method == "custom"
    assert e.params["alpha"] == pytest.approx(0.5)


# ─── entries_from_ranked_pairs ───────────────────────────────────────────────

def test_entries_from_ranked_pairs_basic():
    pairs = [(0, 1), (1, 2), (2, 3)]
    scores = [0.9, 0.7, 0.5]
    entries = entries_from_ranked_pairs(pairs, scores)
    assert len(entries) == 3
    assert entries[0].frag_i == 0
    assert entries[0].score == pytest.approx(0.9)


def test_entries_from_ranked_pairs_custom_ranks():
    pairs = [(0, 1), (1, 2)]
    scores = [0.8, 0.6]
    entries = entries_from_ranked_pairs(pairs, scores, ranks=[2, 1])
    assert entries[0].rank == 2
    assert entries[1].rank == 1


def test_entries_from_ranked_pairs_length_mismatch():
    with pytest.raises(ValueError):
        entries_from_ranked_pairs([(0, 1)], [0.5, 0.6])


def test_entries_from_ranked_pairs_default_ranks():
    pairs = [(0, 1), (1, 2)]
    scores = [0.8, 0.6]
    entries = entries_from_ranked_pairs(pairs, scores)
    assert entries[0].rank == 1
    assert entries[1].rank == 2


# ─── summarise_rank_results ──────────────────────────────────────────────────

def test_summarise_rank_results_basic():
    entries = [
        make_rank_result_entry(0, 1, 0.9, 1),
        make_rank_result_entry(1, 2, 0.6, 2),
        make_rank_result_entry(2, 3, 0.3, 3),
    ]
    s = summarise_rank_results(entries)
    assert s.n_entries == 3
    assert s.mean_score == pytest.approx(0.6)
    assert s.max_score == pytest.approx(0.9)
    assert s.min_score == pytest.approx(0.3)
    assert s.n_top_matches == 1


def test_summarise_rank_results_empty():
    s = summarise_rank_results([])
    assert s.n_entries == 0
    assert s.mean_score == pytest.approx(0.0)


def test_summarise_rank_results_mean_rank():
    entries = [
        make_rank_result_entry(0, 1, 0.9, 1),
        make_rank_result_entry(1, 2, 0.6, 3),
    ]
    s = summarise_rank_results(entries)
    assert s.mean_rank == pytest.approx(2.0)


# ─── filter_high_rank_entries ────────────────────────────────────────────────

def test_filter_high_rank_entries():
    entries = [
        make_rank_result_entry(0, 1, 0.9, 1),
        make_rank_result_entry(1, 2, 0.5, 2),
        make_rank_result_entry(2, 3, 0.3, 3),
    ]
    result = filter_high_rank_entries(entries, threshold=0.7)
    assert len(result) == 1
    assert result[0].score == pytest.approx(0.9)


# ─── filter_low_rank_entries ─────────────────────────────────────────────────

def test_filter_low_rank_entries():
    entries = [
        make_rank_result_entry(0, 1, 0.9, 1),
        make_rank_result_entry(1, 2, 0.2, 2),
    ]
    result = filter_low_rank_entries(entries, threshold=0.3)
    assert len(result) == 1
    assert result[0].score == pytest.approx(0.2)


# ─── filter_by_rank_position ─────────────────────────────────────────────────

def test_filter_by_rank_position():
    entries = [
        make_rank_result_entry(0, 1, 0.9, 1),
        make_rank_result_entry(1, 2, 0.7, 2),
        make_rank_result_entry(2, 3, 0.5, 5),
    ]
    result = filter_by_rank_position(entries, max_rank=2)
    assert len(result) == 2
    assert all(e.rank <= 2 for e in result)


# ─── filter_rank_by_score_range ──────────────────────────────────────────────

def test_filter_rank_by_score_range():
    entries = [
        make_rank_result_entry(0, 1, 0.2, 1),
        make_rank_result_entry(1, 2, 0.5, 2),
        make_rank_result_entry(2, 3, 0.9, 3),
    ]
    result = filter_rank_by_score_range(entries, lo=0.4, hi=0.8)
    assert len(result) == 1
    assert result[0].score == pytest.approx(0.5)


# ─── filter_rank_by_dominant_channel ────────────────────────────────────────

def test_filter_rank_by_dominant_channel():
    entries = [
        make_rank_result_entry(0, 1, 0.9, 1,
                                channel_scores={"R": 0.9, "G": 0.5}),
        make_rank_result_entry(1, 2, 0.7, 2,
                                channel_scores={"R": 0.3, "G": 0.8}),
    ]
    result = filter_rank_by_dominant_channel(entries, "R")
    assert len(result) == 1
    assert result[0].frag_i == 0


# ─── top_k_rank_entries ──────────────────────────────────────────────────────

def test_top_k_rank_entries():
    entries = [
        make_rank_result_entry(0, 1, 0.5, 3),
        make_rank_result_entry(1, 2, 0.9, 1),
        make_rank_result_entry(2, 3, 0.7, 2),
    ]
    top2 = top_k_rank_entries(entries, k=2)
    assert len(top2) == 2
    assert top2[0].score == pytest.approx(0.9)


# ─── best_rank_entry ─────────────────────────────────────────────────────────

def test_best_rank_entry():
    entries = [
        make_rank_result_entry(0, 1, 0.5, 1),
        make_rank_result_entry(1, 2, 0.9, 2),
    ]
    best = best_rank_entry(entries)
    assert best.score == pytest.approx(0.9)


def test_best_rank_entry_empty():
    assert best_rank_entry([]) is None


# ─── rerank_entries ──────────────────────────────────────────────────────────

def test_rerank_entries_descending():
    entries = [
        make_rank_result_entry(0, 1, 0.3, 1),
        make_rank_result_entry(1, 2, 0.9, 2),
        make_rank_result_entry(2, 3, 0.6, 3),
    ]
    reranked = rerank_entries(entries)
    assert reranked[0].score == pytest.approx(0.9)
    assert reranked[0].rank == 1


def test_rerank_entries_ascending():
    entries = [
        make_rank_result_entry(0, 1, 0.3, 2),
        make_rank_result_entry(1, 2, 0.9, 1),
    ]
    reranked = rerank_entries(entries, ascending=True)
    assert reranked[0].score == pytest.approx(0.3)
    assert reranked[0].rank == 1


# ─── rank_result_stats ───────────────────────────────────────────────────────

def test_rank_result_stats_keys():
    entries = [make_rank_result_entry(0, 1, 0.8, 1)]
    stats = rank_result_stats(entries)
    for key in ("count", "mean_score", "std_score", "min_score",
                "max_score", "mean_rank"):
        assert key in stats


def test_rank_result_stats_empty():
    stats = rank_result_stats([])
    assert stats["count"] == 0


# ─── compare_rank_summaries ──────────────────────────────────────────────────

def test_compare_rank_summaries_keys():
    entries_a = [make_rank_result_entry(0, 1, 0.9, 1)]
    entries_b = [make_rank_result_entry(0, 1, 0.5, 1),
                 make_rank_result_entry(1, 2, 0.3, 2)]
    s_a = summarise_rank_results(entries_a)
    s_b = summarise_rank_results(entries_b)
    result = compare_rank_summaries(s_a, s_b)
    for key in ("d_mean_score", "d_std_score", "d_mean_rank",
                "d_n_top", "d_n_entries"):
        assert key in result


def test_compare_rank_summaries_values():
    entries = [make_rank_result_entry(0, 1, 0.8, 1)]
    s = summarise_rank_results(entries)
    result = compare_rank_summaries(s, s)
    assert result["d_mean_score"] == pytest.approx(0.0)
    assert result["d_n_entries"] == 0


# ─── batch_summarise_rank_results ────────────────────────────────────────────

def test_batch_summarise_rank_results_length():
    groups = [
        [make_rank_result_entry(0, 1, 0.8, 1)],
        [make_rank_result_entry(0, 1, 0.5, 1),
         make_rank_result_entry(1, 2, 0.3, 2)],
        [],
    ]
    results = batch_summarise_rank_results(groups)
    assert len(results) == 3


def test_batch_summarise_rank_results_empty_group():
    results = batch_summarise_rank_results([[]])
    assert results[0].n_entries == 0
