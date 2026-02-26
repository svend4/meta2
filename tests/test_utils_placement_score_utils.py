"""Tests for puzzle_reconstruction.utils.placement_score_utils."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.placement_score_utils import (
    PlacementScoreConfig,
    PlacementScoreEntry,
    PlacementSummary,
    make_placement_entry,
    entries_from_history,
    summarise_placement,
    filter_positive_steps,
    filter_by_min_score,
    top_k_steps,
    rank_fragments,
    placement_score_stats,
    compare_placements,
    batch_summarise,
)

np.random.seed(0)


# ─── PlacementScoreConfig ────────────────────────────────────────────────────

def test_config_defaults():
    cfg = PlacementScoreConfig()
    assert cfg.min_score == pytest.approx(0.0)
    assert cfg.coverage_weight == pytest.approx(0.5)
    assert cfg.overlap_penalty_weight == pytest.approx(0.5)
    assert cfg.prefer_full_placement is True


def test_config_invalid_min_score():
    with pytest.raises(ValueError):
        PlacementScoreConfig(min_score=1.5)


def test_config_invalid_coverage_weight():
    with pytest.raises(ValueError):
        PlacementScoreConfig(coverage_weight=-0.1)


def test_config_invalid_overlap_weight():
    with pytest.raises(ValueError):
        PlacementScoreConfig(overlap_penalty_weight=2.0)


def test_config_valid_boundary_values():
    cfg = PlacementScoreConfig(min_score=0.0, coverage_weight=1.0,
                                overlap_penalty_weight=0.0)
    assert cfg.min_score == pytest.approx(0.0)


# ─── PlacementScoreEntry ─────────────────────────────────────────────────────

def test_entry_creation():
    e = PlacementScoreEntry(step=0, fragment_idx=1, score_delta=0.1,
                             cumulative_score=0.1)
    assert e.step == 0
    assert e.fragment_idx == 1
    assert e.score_delta == pytest.approx(0.1)


def test_entry_negative_step_raises():
    with pytest.raises(ValueError):
        PlacementScoreEntry(step=-1, fragment_idx=0, score_delta=0.1,
                             cumulative_score=0.1)


def test_entry_negative_fragment_idx_raises():
    with pytest.raises(ValueError):
        PlacementScoreEntry(step=0, fragment_idx=-1, score_delta=0.1,
                             cumulative_score=0.1)


def test_entry_repr():
    e = PlacementScoreEntry(step=2, fragment_idx=3, score_delta=0.25,
                             cumulative_score=0.75)
    r = repr(e)
    assert "step=2" in r
    assert "idx=3" in r


# ─── make_placement_entry ────────────────────────────────────────────────────

def test_make_placement_entry_basic():
    e = make_placement_entry(0, 1, 0.2, 0.2, position=(10.0, 20.0))
    assert isinstance(e, PlacementScoreEntry)
    assert e.position == (10.0, 20.0)


def test_make_placement_entry_with_meta():
    e = make_placement_entry(1, 2, 0.3, 0.5, meta={"key": "val"})
    assert e.meta["key"] == "val"


def test_make_placement_entry_score_delta_float():
    e = make_placement_entry(0, 0, 1, 1)
    assert isinstance(e.score_delta, float)


# ─── entries_from_history ────────────────────────────────────────────────────

def test_entries_from_history_basic():
    history = [
        {"step": 0, "idx": 0, "score_delta": 0.1, "position": (0.0, 0.0)},
        {"step": 1, "idx": 1, "score_delta": 0.2, "position": (10.0, 0.0)},
    ]
    entries = entries_from_history(history)
    assert len(entries) == 2
    assert entries[0].step == 0
    assert entries[1].step == 1


def test_entries_from_history_cumulative():
    history = [
        {"step": 0, "score_delta": 0.3},
        {"step": 1, "score_delta": 0.4},
    ]
    entries = entries_from_history(history)
    assert entries[1].cumulative_score == pytest.approx(0.7)


def test_entries_from_history_sorted_by_step():
    history = [
        {"step": 2, "score_delta": 0.1},
        {"step": 0, "score_delta": 0.2},
        {"step": 1, "score_delta": 0.3},
    ]
    entries = entries_from_history(history)
    steps = [e.step for e in entries]
    assert steps == sorted(steps)


def test_entries_from_history_empty():
    assert entries_from_history([]) == []


# ─── summarise_placement ─────────────────────────────────────────────────────

def test_summarise_placement_basic():
    entries = [
        make_placement_entry(0, 0, 0.3, 0.3),
        make_placement_entry(1, 1, 0.2, 0.5),
        make_placement_entry(2, 2, -0.1, 0.4),
    ]
    s = summarise_placement(entries)
    assert isinstance(s, PlacementSummary)
    assert s.n_placed == 3
    assert s.final_score == pytest.approx(0.4)
    assert s.max_delta == pytest.approx(0.3)
    assert s.min_delta == pytest.approx(-0.1)


def test_summarise_placement_empty():
    s = summarise_placement([])
    assert s.n_placed == 0
    assert s.final_score == pytest.approx(0.0)


def test_summarise_placement_mean_delta():
    entries = [
        make_placement_entry(0, 0, 0.4, 0.4),
        make_placement_entry(1, 1, 0.6, 1.0),
    ]
    s = summarise_placement(entries)
    assert s.mean_delta == pytest.approx(0.5)


# ─── filter_positive_steps ───────────────────────────────────────────────────

def test_filter_positive_steps():
    entries = [
        make_placement_entry(0, 0, 0.3, 0.3),
        make_placement_entry(1, 1, -0.1, 0.2),
        make_placement_entry(2, 2, 0.5, 0.7),
    ]
    result = filter_positive_steps(entries)
    assert len(result) == 2
    assert all(e.score_delta > 0 for e in result)


# ─── filter_by_min_score ─────────────────────────────────────────────────────

def test_filter_by_min_score():
    entries = [
        make_placement_entry(0, 0, 0.3, 0.3),
        make_placement_entry(1, 1, 0.2, 0.5),
        make_placement_entry(2, 2, 0.1, 0.6),
    ]
    result = filter_by_min_score(entries, min_score=0.5)
    assert len(result) == 2
    assert all(e.cumulative_score >= 0.5 for e in result)


# ─── top_k_steps ─────────────────────────────────────────────────────────────

def test_top_k_steps():
    entries = [
        make_placement_entry(0, 0, 0.1, 0.1),
        make_placement_entry(1, 1, 0.5, 0.6),
        make_placement_entry(2, 2, 0.3, 0.9),
    ]
    top2 = top_k_steps(entries, 2)
    assert len(top2) == 2
    assert top2[0].score_delta >= top2[1].score_delta


def test_top_k_steps_k_larger_than_list():
    entries = [make_placement_entry(0, 0, 0.5, 0.5)]
    result = top_k_steps(entries, 10)
    assert len(result) == 1


# ─── rank_fragments ──────────────────────────────────────────────────────────

def test_rank_fragments_order():
    entries = [
        make_placement_entry(0, 2, 0.1, 0.1),
        make_placement_entry(1, 5, 0.8, 0.9),
        make_placement_entry(2, 3, 0.4, 1.3),
    ]
    ranked = rank_fragments(entries)
    assert ranked[0][0] == 5  # highest delta
    assert ranked[0][1] == pytest.approx(0.8)


def test_rank_fragments_returns_tuples():
    entries = [make_placement_entry(0, 1, 0.5, 0.5)]
    ranked = rank_fragments(entries)
    assert len(ranked) == 1
    assert isinstance(ranked[0], tuple)


# ─── placement_score_stats ───────────────────────────────────────────────────

def test_placement_score_stats_keys():
    entries = [
        make_placement_entry(0, 0, 0.2, 0.2),
        make_placement_entry(1, 1, -0.1, 0.1),
        make_placement_entry(2, 2, 0.4, 0.5),
    ]
    stats = placement_score_stats(entries)
    for key in ("n", "final_score", "mean_delta", "std_delta",
                "max_delta", "min_delta", "n_positive", "n_negative"):
        assert key in stats


def test_placement_score_stats_empty():
    stats = placement_score_stats([])
    assert stats["n"] == 0


def test_placement_score_stats_n_positive():
    entries = [
        make_placement_entry(0, 0, 0.3, 0.3),
        make_placement_entry(1, 1, -0.1, 0.2),
        make_placement_entry(2, 2, 0.5, 0.7),
    ]
    stats = placement_score_stats(entries)
    assert stats["n_positive"] == 2
    assert stats["n_negative"] == 1


# ─── compare_placements ──────────────────────────────────────────────────────

def test_compare_placements_better_b():
    s_a = summarise_placement([make_placement_entry(0, 0, 0.3, 0.3)])
    s_b = summarise_placement([make_placement_entry(0, 0, 0.8, 0.8)])
    result = compare_placements(s_a, s_b)
    assert result["better"] == "b"
    assert result["score_diff"] == pytest.approx(0.5)


def test_compare_placements_better_a():
    s_a = summarise_placement([make_placement_entry(0, 0, 0.9, 0.9)])
    s_b = summarise_placement([make_placement_entry(0, 0, 0.4, 0.4)])
    result = compare_placements(s_a, s_b)
    assert result["better"] == "a"


def test_compare_placements_tie():
    entries = [make_placement_entry(0, 0, 0.5, 0.5)]
    s = summarise_placement(entries)
    result = compare_placements(s, s)
    assert result["better"] == "tie"


# ─── batch_summarise ─────────────────────────────────────────────────────────

def test_batch_summarise_length():
    histories = [
        [{"step": 0, "score_delta": 0.3}],
        [{"step": 0, "score_delta": 0.2}, {"step": 1, "score_delta": 0.1}],
    ]
    results = batch_summarise(histories)
    assert len(results) == 2


def test_batch_summarise_empty_histories():
    results = batch_summarise([[], []])
    assert all(s.n_placed == 0 for s in results)


def test_batch_summarise_returns_summaries():
    histories = [[{"step": 0, "score_delta": 0.5}]]
    results = batch_summarise(histories)
    assert isinstance(results[0], PlacementSummary)
