"""Extra tests for puzzle_reconstruction/utils/placement_score_utils.py."""
from __future__ import annotations

import pytest

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _entry(step=0, frag=0, delta=0.1, cum=0.1,
           pos=(0.0, 0.0)) -> PlacementScoreEntry:
    return PlacementScoreEntry(step=step, fragment_idx=frag,
                               score_delta=delta, cumulative_score=cum,
                               position=pos)


# ─── PlacementScoreConfig ─────────────────────────────────────────────────────

class TestPlacementScoreConfigExtra:
    def test_default_min_score(self):
        assert PlacementScoreConfig().min_score == pytest.approx(0.0)

    def test_default_coverage_weight(self):
        assert PlacementScoreConfig().coverage_weight == pytest.approx(0.5)

    def test_min_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            PlacementScoreConfig(min_score=1.5)

    def test_coverage_weight_out_of_range_raises(self):
        with pytest.raises(ValueError):
            PlacementScoreConfig(coverage_weight=-0.1)

    def test_overlap_penalty_weight_out_of_range_raises(self):
        with pytest.raises(ValueError):
            PlacementScoreConfig(overlap_penalty_weight=2.0)


# ─── PlacementScoreEntry ──────────────────────────────────────────────────────

class TestPlacementScoreEntryExtra:
    def test_negative_step_raises(self):
        with pytest.raises(ValueError):
            PlacementScoreEntry(step=-1, fragment_idx=0,
                                score_delta=0.1, cumulative_score=0.1)

    def test_negative_fragment_idx_raises(self):
        with pytest.raises(ValueError):
            PlacementScoreEntry(step=0, fragment_idx=-1,
                                score_delta=0.1, cumulative_score=0.1)

    def test_stores_position(self):
        e = _entry(pos=(5.0, 10.0))
        assert e.position == (5.0, 10.0)

    def test_repr_contains_step(self):
        e = _entry(step=3)
        assert "3" in repr(e)


# ─── PlacementSummary ─────────────────────────────────────────────────────────

class TestPlacementSummaryExtra:
    def test_negative_n_placed_raises(self):
        with pytest.raises(ValueError):
            PlacementSummary(entries=[], n_placed=-1, final_score=0.0,
                              mean_delta=0.0, max_delta=0.0, min_delta=0.0)

    def test_repr_contains_n_placed(self):
        s = PlacementSummary(entries=[], n_placed=5, final_score=0.8,
                              mean_delta=0.1, max_delta=0.3, min_delta=0.0)
        assert "5" in repr(s)


# ─── make_placement_entry ─────────────────────────────────────────────────────

class TestMakePlacementEntryExtra:
    def test_returns_entry(self):
        e = make_placement_entry(0, 1, 0.1, 0.5)
        assert isinstance(e, PlacementScoreEntry)

    def test_meta_stored(self):
        e = make_placement_entry(0, 1, 0.1, 0.5, meta={"algo": "greedy"})
        assert e.meta["algo"] == "greedy"

    def test_position_stored(self):
        e = make_placement_entry(0, 1, 0.1, 0.5, position=(3.0, 7.0))
        assert e.position == (3.0, 7.0)


# ─── entries_from_history ─────────────────────────────────────────────────────

class TestEntriesFromHistoryExtra:
    def test_returns_list(self):
        h = [{"step": 0, "idx": 0, "score_delta": 0.1}]
        result = entries_from_history(h)
        assert isinstance(result, list) and len(result) == 1

    def test_empty_history(self):
        assert entries_from_history([]) == []

    def test_cumulative_accumulates(self):
        h = [{"step": 0, "idx": 0, "score_delta": 0.2},
             {"step": 1, "idx": 1, "score_delta": 0.3}]
        entries = entries_from_history(h)
        assert entries[-1].cumulative_score == pytest.approx(0.5)


# ─── summarise_placement ──────────────────────────────────────────────────────

class TestSummarisePlacementExtra:
    def test_empty_returns_summary(self):
        s = summarise_placement([])
        assert s.n_placed == 0 and s.final_score == pytest.approx(0.0)

    def test_single_entry(self):
        e = _entry(step=0, delta=0.5, cum=0.5)
        s = summarise_placement([e])
        assert s.n_placed == 1 and s.final_score == pytest.approx(0.5)

    def test_final_score_is_last_cumulative(self):
        entries = [_entry(step=0, delta=0.3, cum=0.3),
                   _entry(step=1, delta=0.2, cum=0.5)]
        s = summarise_placement(entries)
        assert s.final_score == pytest.approx(0.5)

    def test_mean_delta(self):
        entries = [_entry(delta=0.2), _entry(step=1, delta=0.4, cum=0.6)]
        s = summarise_placement(entries)
        assert s.mean_delta == pytest.approx(0.3)


# ─── filter helpers ───────────────────────────────────────────────────────────

class TestFilterPlacementExtra:
    def test_filter_positive_steps(self):
        entries = [_entry(delta=0.1), _entry(step=1, delta=-0.05, cum=0.05)]
        result = filter_positive_steps(entries)
        assert len(result) == 1

    def test_filter_by_min_score(self):
        entries = [_entry(cum=0.2), _entry(step=1, delta=0.4, cum=0.8)]
        result = filter_by_min_score(entries, 0.5)
        assert len(result) == 1

    def test_top_k_steps(self):
        entries = [_entry(delta=0.1), _entry(step=1, delta=0.5, cum=0.6),
                   _entry(step=2, delta=0.3, cum=0.9)]
        top = top_k_steps(entries, 2)
        assert top[0].score_delta == pytest.approx(0.5)
        assert len(top) == 2

    def test_rank_fragments(self):
        entries = [_entry(frag=0, delta=0.3), _entry(step=1, frag=1, delta=0.7, cum=1.0)]
        ranked = rank_fragments(entries)
        assert ranked[0][0] == 1  # frag 1 has higher delta


# ─── placement_score_stats ────────────────────────────────────────────────────

class TestPlacementScoreStatsExtra:
    def test_empty_returns_zeros(self):
        s = placement_score_stats([])
        assert s["n"] == 0

    def test_n_positive_n_negative(self):
        entries = [_entry(delta=0.3), _entry(step=1, delta=-0.1, cum=0.2)]
        s = placement_score_stats(entries)
        assert s["n_positive"] == 1
        assert s["n_negative"] == 1

    def test_final_score(self):
        entries = [_entry(delta=0.5, cum=0.5)]
        s = placement_score_stats(entries)
        assert s["final_score"] == pytest.approx(0.5)


# ─── compare_placements ───────────────────────────────────────────────────────

class TestComparePlacementsExtra:
    def test_returns_dict(self):
        s = summarise_placement([_entry()])
        d = compare_placements(s, s)
        assert isinstance(d, dict)

    def test_tie_when_equal(self):
        s = summarise_placement([_entry(delta=0.5, cum=0.5)])
        d = compare_placements(s, s)
        assert d["better"] == "tie"

    def test_better_b(self):
        sa = summarise_placement([_entry(delta=0.3, cum=0.3)])
        sb = summarise_placement([_entry(delta=0.8, cum=0.8)])
        d = compare_placements(sa, sb)
        assert d["better"] == "b"


# ─── batch_summarise ──────────────────────────────────────────────────────────

class TestBatchSummariseExtra:
    def test_returns_list(self):
        h = [{"step": 0, "idx": 0, "score_delta": 0.1}]
        result = batch_summarise([h, h])
        assert len(result) == 2

    def test_empty_histories(self):
        result = batch_summarise([[], []])
        assert all(s.n_placed == 0 for s in result)
