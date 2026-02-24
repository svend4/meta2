"""Extra tests for puzzle_reconstruction/utils/seq_gap_utils.py (iter-234)."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.seq_gap_utils import (
    SequenceScoreConfig,
    SequenceScoreEntry,
    SequenceScoreSummary,
    make_sequence_score_entry,
    summarise_sequence_score_entries,
    filter_full_sequences,
    filter_sequence_by_min_score,
    filter_sequence_by_algorithm,
    top_k_sequence_entries,
    best_sequence_entry,
    sequence_score_stats,
    compare_sequence_summaries,
    batch_summarise_sequence_score_entries,
    GapScoreConfig,
    GapScoreEntry,
    GapScoreSummary,
    make_gap_score_entry,
    summarise_gap_score_entries,
    filter_overlapping_gaps,
    filter_gap_by_category,
    filter_gap_by_max_distance,
    top_k_closest_gaps,
    best_gap_entry,
    gap_score_stats,
    compare_gap_summaries,
    batch_summarise_gap_score_entries,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _seq_entry(
    seq_id: int = 0,
    total_score: float = 0.8,
    n_fragments: int = 5,
    algorithm: str = "greedy",
    is_full: bool = True,
) -> SequenceScoreEntry:
    return make_sequence_score_entry(
        seq_id=seq_id,
        order=list(range(n_fragments)),
        total_score=total_score,
        n_fragments=n_fragments,
        algorithm=algorithm,
        is_full=is_full,
    )


def _gap_entry(
    id1: int = 0,
    id2: int = 1,
    gap_x: float = 1.0,
    gap_y: float = 2.0,
    distance: float = 5.0,
    category: str = "near",
) -> GapScoreEntry:
    return make_gap_score_entry(
        id1=id1, id2=id2, gap_x=gap_x, gap_y=gap_y,
        distance=distance, category=category,
    )


def _seq_entries_mixed() -> list:
    return [
        _seq_entry(seq_id=0, total_score=0.9, algorithm="greedy", is_full=True),
        _seq_entry(seq_id=1, total_score=0.5, algorithm="beam", is_full=False),
        _seq_entry(seq_id=2, total_score=0.7, algorithm="greedy", is_full=True),
        _seq_entry(seq_id=3, total_score=0.3, algorithm="mcts", is_full=False),
        _seq_entry(seq_id=4, total_score=0.6, algorithm="beam", is_full=True),
    ]


def _gap_entries_mixed() -> list:
    return [
        _gap_entry(id1=0, id2=1, distance=1.0, category="overlap"),
        _gap_entry(id1=1, id2=2, distance=3.0, category="touching"),
        _gap_entry(id1=2, id2=3, distance=8.0, category="near"),
        _gap_entry(id1=3, id2=4, distance=15.0, category="far"),
        _gap_entry(id1=4, id2=5, distance=0.5, category="overlap"),
    ]


# ─── SequenceScoreConfig ────────────────────────────────────────────────────

class TestSequenceScoreConfigExtra:
    def test_default_min_score(self):
        assert SequenceScoreConfig().min_score == pytest.approx(0.0)

    def test_default_require_full(self):
        assert SequenceScoreConfig().require_full is True

    def test_custom_min_score(self):
        cfg = SequenceScoreConfig(min_score=0.5)
        assert cfg.min_score == pytest.approx(0.5)

    def test_custom_require_full_false(self):
        cfg = SequenceScoreConfig(require_full=False)
        assert cfg.require_full is False

    def test_both_custom(self):
        cfg = SequenceScoreConfig(min_score=0.9, require_full=False)
        assert cfg.min_score == pytest.approx(0.9)
        assert cfg.require_full is False


# ─── SequenceScoreEntry ─────────────────────────────────────────────────────

class TestSequenceScoreEntryExtra:
    def test_fields_stored(self):
        e = _seq_entry(seq_id=7, total_score=0.75, n_fragments=3)
        assert e.seq_id == 7
        assert e.total_score == pytest.approx(0.75)
        assert e.n_fragments == 3

    def test_default_algorithm(self):
        e = SequenceScoreEntry(seq_id=0, order=[], total_score=0.0, n_fragments=0)
        assert e.algorithm == "greedy"

    def test_default_is_full(self):
        e = SequenceScoreEntry(seq_id=0, order=[], total_score=0.0, n_fragments=0)
        assert e.is_full is True

    def test_order_preserved(self):
        e = _seq_entry(n_fragments=4)
        assert e.order == [0, 1, 2, 3]

    def test_custom_algorithm(self):
        e = _seq_entry(algorithm="beam")
        assert e.algorithm == "beam"

    def test_is_full_false(self):
        e = _seq_entry(is_full=False)
        assert e.is_full is False


# ─── SequenceScoreSummary ───────────────────────────────────────────────────

class TestSequenceScoreSummaryExtra:
    def test_fields(self):
        s = SequenceScoreSummary(
            n_entries=3, mean_score=0.5, mean_fragments=4.0,
            min_score=0.2, max_score=0.8, n_full=2, algorithms=["greedy"],
        )
        assert s.n_entries == 3
        assert s.mean_score == pytest.approx(0.5)

    def test_algorithms_list(self):
        s = SequenceScoreSummary(
            n_entries=1, mean_score=0.0, mean_fragments=0.0,
            min_score=0.0, max_score=0.0, n_full=0, algorithms=["a", "b"],
        )
        assert s.algorithms == ["a", "b"]

    def test_n_full_zero(self):
        s = SequenceScoreSummary(
            n_entries=2, mean_score=0.0, mean_fragments=0.0,
            min_score=0.0, max_score=0.0, n_full=0, algorithms=[],
        )
        assert s.n_full == 0

    def test_min_max_order(self):
        s = SequenceScoreSummary(
            n_entries=2, mean_score=0.5, mean_fragments=3.0,
            min_score=0.1, max_score=0.9, n_full=1, algorithms=[],
        )
        assert s.min_score <= s.max_score

    def test_mean_fragments_float(self):
        s = SequenceScoreSummary(
            n_entries=1, mean_score=0.0, mean_fragments=3.5,
            min_score=0.0, max_score=0.0, n_full=0, algorithms=[],
        )
        assert s.mean_fragments == pytest.approx(3.5)


# ─── make_sequence_score_entry ──────────────────────────────────────────────

class TestMakeSequenceScoreEntryExtra:
    def test_returns_entry(self):
        e = make_sequence_score_entry(0, [1, 2], 0.5, 2)
        assert isinstance(e, SequenceScoreEntry)

    def test_seq_id(self):
        e = make_sequence_score_entry(42, [0], 0.1, 1)
        assert e.seq_id == 42

    def test_order_list(self):
        e = make_sequence_score_entry(0, [3, 1, 2], 0.5, 3)
        assert e.order == [3, 1, 2]

    def test_total_score(self):
        e = make_sequence_score_entry(0, [], 0.99, 0)
        assert e.total_score == pytest.approx(0.99)

    def test_algorithm_default(self):
        e = make_sequence_score_entry(0, [], 0.0, 0)
        assert e.algorithm == "greedy"

    def test_algorithm_custom(self):
        e = make_sequence_score_entry(0, [], 0.0, 0, algorithm="mcts")
        assert e.algorithm == "mcts"


# ─── summarise_sequence_score_entries ───────────────────────────────────────

class TestSummariseSequenceScoreEntriesExtra:
    def test_empty_returns_zero(self):
        s = summarise_sequence_score_entries([])
        assert s.n_entries == 0
        assert s.mean_score == pytest.approx(0.0)

    def test_single_entry(self):
        s = summarise_sequence_score_entries([_seq_entry(total_score=0.8)])
        assert s.n_entries == 1
        assert s.mean_score == pytest.approx(0.8)

    def test_n_full_count(self):
        entries = [_seq_entry(is_full=True), _seq_entry(is_full=False), _seq_entry(is_full=True)]
        s = summarise_sequence_score_entries(entries)
        assert s.n_full == 2

    def test_algorithms_sorted(self):
        entries = [_seq_entry(algorithm="beam"), _seq_entry(algorithm="greedy")]
        s = summarise_sequence_score_entries(entries)
        assert s.algorithms == ["beam", "greedy"]

    def test_min_max_scores(self):
        entries = _seq_entries_mixed()
        s = summarise_sequence_score_entries(entries)
        assert s.min_score == pytest.approx(0.3)
        assert s.max_score == pytest.approx(0.9)

    def test_mean_score(self):
        entries = [_seq_entry(total_score=0.4), _seq_entry(total_score=0.6)]
        s = summarise_sequence_score_entries(entries)
        assert s.mean_score == pytest.approx(0.5)

    def test_mean_fragments(self):
        entries = [_seq_entry(n_fragments=3), _seq_entry(n_fragments=7)]
        s = summarise_sequence_score_entries(entries)
        assert s.mean_fragments == pytest.approx(5.0)


# ─── filter_full_sequences ──────────────────────────────────────────────────

class TestFilterFullSequencesExtra:
    def test_all_full(self):
        entries = [_seq_entry(is_full=True) for _ in range(3)]
        assert len(filter_full_sequences(entries)) == 3

    def test_none_full(self):
        entries = [_seq_entry(is_full=False) for _ in range(3)]
        assert len(filter_full_sequences(entries)) == 0

    def test_mixed(self):
        entries = _seq_entries_mixed()
        result = filter_full_sequences(entries)
        assert all(e.is_full for e in result)
        assert len(result) == 3

    def test_empty(self):
        assert filter_full_sequences([]) == []

    def test_preserves_order(self):
        entries = [
            _seq_entry(seq_id=10, is_full=True),
            _seq_entry(seq_id=20, is_full=False),
            _seq_entry(seq_id=30, is_full=True),
        ]
        result = filter_full_sequences(entries)
        assert [e.seq_id for e in result] == [10, 30]


# ─── filter_sequence_by_min_score ───────────────────────────────────────────

class TestFilterSequenceByMinScoreExtra:
    def test_threshold_zero(self):
        entries = _seq_entries_mixed()
        assert len(filter_sequence_by_min_score(entries, 0.0)) == 5

    def test_threshold_high(self):
        entries = _seq_entries_mixed()
        result = filter_sequence_by_min_score(entries, 0.85)
        assert len(result) == 1
        assert result[0].total_score == pytest.approx(0.9)

    def test_threshold_exact(self):
        entries = [_seq_entry(total_score=0.5)]
        assert len(filter_sequence_by_min_score(entries, 0.5)) == 1

    def test_empty_list(self):
        assert filter_sequence_by_min_score([], 0.5) == []

    def test_all_below(self):
        entries = [_seq_entry(total_score=0.1), _seq_entry(total_score=0.2)]
        assert len(filter_sequence_by_min_score(entries, 0.5)) == 0


# ─── filter_sequence_by_algorithm ───────────────────────────────────────────

class TestFilterSequenceByAlgorithmExtra:
    def test_filter_greedy(self):
        entries = _seq_entries_mixed()
        result = filter_sequence_by_algorithm(entries, "greedy")
        assert len(result) == 2
        assert all(e.algorithm == "greedy" for e in result)

    def test_filter_beam(self):
        entries = _seq_entries_mixed()
        result = filter_sequence_by_algorithm(entries, "beam")
        assert len(result) == 2

    def test_nonexistent_algorithm(self):
        entries = _seq_entries_mixed()
        assert filter_sequence_by_algorithm(entries, "random") == []

    def test_empty(self):
        assert filter_sequence_by_algorithm([], "greedy") == []

    def test_single_match(self):
        entries = _seq_entries_mixed()
        result = filter_sequence_by_algorithm(entries, "mcts")
        assert len(result) == 1


# ─── top_k_sequence_entries ─────────────────────────────────────────────────

class TestTopKSequenceEntriesExtra:
    def test_top_1(self):
        entries = _seq_entries_mixed()
        result = top_k_sequence_entries(entries, 1)
        assert len(result) == 1
        assert result[0].total_score == pytest.approx(0.9)

    def test_top_3(self):
        entries = _seq_entries_mixed()
        result = top_k_sequence_entries(entries, 3)
        assert len(result) == 3
        assert result[0].total_score >= result[1].total_score >= result[2].total_score

    def test_k_greater_than_n(self):
        entries = _seq_entries_mixed()
        result = top_k_sequence_entries(entries, 100)
        assert len(result) == len(entries)

    def test_empty(self):
        assert top_k_sequence_entries([], 5) == []

    def test_descending_order(self):
        entries = _seq_entries_mixed()
        result = top_k_sequence_entries(entries, 5)
        scores = [e.total_score for e in result]
        assert scores == sorted(scores, reverse=True)


# ─── best_sequence_entry ────────────────────────────────────────────────────

class TestBestSequenceEntryExtra:
    def test_returns_best(self):
        entries = _seq_entries_mixed()
        best = best_sequence_entry(entries)
        assert best is not None
        assert best.total_score == pytest.approx(0.9)

    def test_empty_returns_none(self):
        assert best_sequence_entry([]) is None

    def test_single(self):
        e = _seq_entry(total_score=0.42)
        assert best_sequence_entry([e]).total_score == pytest.approx(0.42)

    def test_returns_entry_type(self):
        entries = _seq_entries_mixed()
        assert isinstance(best_sequence_entry(entries), SequenceScoreEntry)

    def test_tie_returns_one(self):
        entries = [_seq_entry(total_score=0.5), _seq_entry(total_score=0.5)]
        best = best_sequence_entry(entries)
        assert best.total_score == pytest.approx(0.5)


# ─── sequence_score_stats ───────────────────────────────────────────────────

class TestSequenceScoreStatsExtra:
    def test_empty(self):
        stats = sequence_score_stats([])
        assert stats["count"] == 0
        assert stats["mean"] == pytest.approx(0.0)

    def test_single_entry_std_zero(self):
        stats = sequence_score_stats([_seq_entry(total_score=0.5)])
        assert stats["std"] == pytest.approx(0.0)

    def test_count(self):
        entries = _seq_entries_mixed()
        stats = sequence_score_stats(entries)
        assert stats["count"] == pytest.approx(5.0)

    def test_min_max(self):
        entries = _seq_entries_mixed()
        stats = sequence_score_stats(entries)
        assert stats["min"] == pytest.approx(0.3)
        assert stats["max"] == pytest.approx(0.9)

    def test_mean(self):
        entries = [_seq_entry(total_score=0.2), _seq_entry(total_score=0.8)]
        stats = sequence_score_stats(entries)
        assert stats["mean"] == pytest.approx(0.5)

    def test_std_positive_for_varied(self):
        entries = _seq_entries_mixed()
        assert sequence_score_stats(entries)["std"] > 0.0


# ─── compare_sequence_summaries ─────────────────────────────────────────────

class TestCompareSequenceSummariesExtra:
    def test_identical_summaries(self):
        s = summarise_sequence_score_entries(_seq_entries_mixed())
        delta = compare_sequence_summaries(s, s)
        assert delta["mean_score_delta"] == pytest.approx(0.0)
        assert delta["mean_fragments_delta"] == pytest.approx(0.0)
        assert delta["n_full_delta"] == pytest.approx(0.0)

    def test_a_better_than_b(self):
        a = summarise_sequence_score_entries([_seq_entry(total_score=0.9)])
        b = summarise_sequence_score_entries([_seq_entry(total_score=0.5)])
        delta = compare_sequence_summaries(a, b)
        assert delta["mean_score_delta"] > 0.0

    def test_b_better_than_a(self):
        a = summarise_sequence_score_entries([_seq_entry(total_score=0.3)])
        b = summarise_sequence_score_entries([_seq_entry(total_score=0.7)])
        delta = compare_sequence_summaries(a, b)
        assert delta["mean_score_delta"] < 0.0

    def test_n_full_delta(self):
        a = summarise_sequence_score_entries([_seq_entry(is_full=True)])
        b = summarise_sequence_score_entries([_seq_entry(is_full=False)])
        delta = compare_sequence_summaries(a, b)
        assert delta["n_full_delta"] == pytest.approx(1.0)

    def test_returns_dict(self):
        s = summarise_sequence_score_entries([_seq_entry()])
        assert isinstance(compare_sequence_summaries(s, s), dict)


# ─── batch_summarise_sequence_score_entries ─────────────────────────────────

class TestBatchSummariseSequenceScoreEntriesExtra:
    def test_single_group(self):
        groups = [_seq_entries_mixed()]
        result = batch_summarise_sequence_score_entries(groups)
        assert len(result) == 1
        assert isinstance(result[0], SequenceScoreSummary)

    def test_multiple_groups(self):
        groups = [
            [_seq_entry(total_score=0.5)],
            [_seq_entry(total_score=0.9)],
        ]
        result = batch_summarise_sequence_score_entries(groups)
        assert len(result) == 2

    def test_empty_groups(self):
        assert batch_summarise_sequence_score_entries([]) == []

    def test_group_with_empty_list(self):
        result = batch_summarise_sequence_score_entries([[]])
        assert len(result) == 1
        assert result[0].n_entries == 0

    def test_summaries_independent(self):
        groups = [
            [_seq_entry(total_score=0.2)],
            [_seq_entry(total_score=0.8)],
        ]
        result = batch_summarise_sequence_score_entries(groups)
        assert result[0].mean_score == pytest.approx(0.2)
        assert result[1].mean_score == pytest.approx(0.8)


# ─── GapScoreConfig ─────────────────────────────────────────────────────────

class TestGapScoreConfigExtra:
    def test_default_near_threshold(self):
        assert GapScoreConfig().near_threshold == pytest.approx(10.0)

    def test_default_overlap_penalty(self):
        assert GapScoreConfig().overlap_penalty == pytest.approx(1.0)

    def test_custom_near_threshold(self):
        cfg = GapScoreConfig(near_threshold=5.0)
        assert cfg.near_threshold == pytest.approx(5.0)

    def test_custom_overlap_penalty(self):
        cfg = GapScoreConfig(overlap_penalty=2.5)
        assert cfg.overlap_penalty == pytest.approx(2.5)

    def test_both_custom(self):
        cfg = GapScoreConfig(near_threshold=20.0, overlap_penalty=0.5)
        assert cfg.near_threshold == pytest.approx(20.0)
        assert cfg.overlap_penalty == pytest.approx(0.5)


# ─── GapScoreEntry ──────────────────────────────────────────────────────────

class TestGapScoreEntryExtra:
    def test_fields_stored(self):
        e = _gap_entry(id1=3, id2=4, gap_x=1.5, gap_y=2.5, distance=7.0)
        assert e.id1 == 3
        assert e.id2 == 4
        assert e.gap_x == pytest.approx(1.5)
        assert e.gap_y == pytest.approx(2.5)
        assert e.distance == pytest.approx(7.0)

    def test_default_category(self):
        e = GapScoreEntry(id1=0, id2=1, gap_x=0.0, gap_y=0.0, distance=0.0)
        assert e.category == "near"

    def test_custom_category(self):
        e = _gap_entry(category="far")
        assert e.category == "far"

    def test_overlap_category(self):
        e = _gap_entry(category="overlap")
        assert e.category == "overlap"

    def test_touching_category(self):
        e = _gap_entry(category="touching")
        assert e.category == "touching"


# ─── GapScoreSummary ────────────────────────────────────────────────────────

class TestGapScoreSummaryExtra:
    def test_fields(self):
        s = GapScoreSummary(
            n_entries=5, mean_distance=3.0, mean_gap_x=1.0, mean_gap_y=2.0,
            n_overlapping=1, n_touching=1, n_near=2, n_far=1,
            min_distance=0.5, max_distance=15.0,
        )
        assert s.n_entries == 5
        assert s.mean_distance == pytest.approx(3.0)

    def test_category_counts_sum(self):
        s = GapScoreSummary(
            n_entries=5, mean_distance=0.0, mean_gap_x=0.0, mean_gap_y=0.0,
            n_overlapping=2, n_touching=1, n_near=1, n_far=1,
            min_distance=0.0, max_distance=0.0,
        )
        assert s.n_overlapping + s.n_touching + s.n_near + s.n_far == s.n_entries

    def test_min_max_distance(self):
        s = GapScoreSummary(
            n_entries=2, mean_distance=5.0, mean_gap_x=0.0, mean_gap_y=0.0,
            n_overlapping=0, n_touching=0, n_near=1, n_far=1,
            min_distance=1.0, max_distance=9.0,
        )
        assert s.min_distance <= s.max_distance

    def test_mean_gap_xy(self):
        s = GapScoreSummary(
            n_entries=1, mean_distance=0.0, mean_gap_x=3.0, mean_gap_y=4.0,
            n_overlapping=0, n_touching=0, n_near=0, n_far=0,
            min_distance=0.0, max_distance=0.0,
        )
        assert s.mean_gap_x == pytest.approx(3.0)
        assert s.mean_gap_y == pytest.approx(4.0)

    def test_zero_entries(self):
        s = GapScoreSummary(
            n_entries=0, mean_distance=0.0, mean_gap_x=0.0, mean_gap_y=0.0,
            n_overlapping=0, n_touching=0, n_near=0, n_far=0,
            min_distance=0.0, max_distance=0.0,
        )
        assert s.n_entries == 0


# ─── make_gap_score_entry ───────────────────────────────────────────────────

class TestMakeGapScoreEntryExtra:
    def test_returns_entry(self):
        e = make_gap_score_entry(0, 1, 1.0, 2.0, 3.0)
        assert isinstance(e, GapScoreEntry)

    def test_ids(self):
        e = make_gap_score_entry(10, 20, 0.0, 0.0, 0.0)
        assert e.id1 == 10
        assert e.id2 == 20

    def test_gap_values(self):
        e = make_gap_score_entry(0, 1, 1.5, 2.5, 3.5)
        assert e.gap_x == pytest.approx(1.5)
        assert e.gap_y == pytest.approx(2.5)

    def test_distance(self):
        e = make_gap_score_entry(0, 1, 0.0, 0.0, 7.7)
        assert e.distance == pytest.approx(7.7)

    def test_default_category(self):
        e = make_gap_score_entry(0, 1, 0.0, 0.0, 0.0)
        assert e.category == "near"

    def test_custom_category(self):
        e = make_gap_score_entry(0, 1, 0.0, 0.0, 0.0, category="far")
        assert e.category == "far"


# ─── summarise_gap_score_entries ────────────────────────────────────────────

class TestSummariseGapScoreEntriesExtra:
    def test_empty(self):
        s = summarise_gap_score_entries([])
        assert s.n_entries == 0
        assert s.mean_distance == pytest.approx(0.0)

    def test_single(self):
        s = summarise_gap_score_entries([_gap_entry(distance=5.0)])
        assert s.n_entries == 1
        assert s.mean_distance == pytest.approx(5.0)

    def test_category_counts(self):
        entries = _gap_entries_mixed()
        s = summarise_gap_score_entries(entries)
        assert s.n_overlapping == 2
        assert s.n_touching == 1
        assert s.n_near == 1
        assert s.n_far == 1

    def test_min_max_distance(self):
        entries = _gap_entries_mixed()
        s = summarise_gap_score_entries(entries)
        assert s.min_distance == pytest.approx(0.5)
        assert s.max_distance == pytest.approx(15.0)

    def test_mean_distance(self):
        entries = [_gap_entry(distance=2.0), _gap_entry(distance=4.0)]
        s = summarise_gap_score_entries(entries)
        assert s.mean_distance == pytest.approx(3.0)

    def test_mean_gap_xy(self):
        entries = [
            _gap_entry(gap_x=1.0, gap_y=3.0),
            _gap_entry(gap_x=3.0, gap_y=5.0),
        ]
        s = summarise_gap_score_entries(entries)
        assert s.mean_gap_x == pytest.approx(2.0)
        assert s.mean_gap_y == pytest.approx(4.0)


# ─── filter_overlapping_gaps ────────────────────────────────────────────────

class TestFilterOverlappingGapsExtra:
    def test_returns_overlaps_only(self):
        entries = _gap_entries_mixed()
        result = filter_overlapping_gaps(entries)
        assert all(e.category == "overlap" for e in result)

    def test_count(self):
        entries = _gap_entries_mixed()
        assert len(filter_overlapping_gaps(entries)) == 2

    def test_no_overlaps(self):
        entries = [_gap_entry(category="near"), _gap_entry(category="far")]
        assert filter_overlapping_gaps(entries) == []

    def test_empty(self):
        assert filter_overlapping_gaps([]) == []

    def test_all_overlaps(self):
        entries = [_gap_entry(category="overlap") for _ in range(4)]
        assert len(filter_overlapping_gaps(entries)) == 4


# ─── filter_gap_by_category ─────────────────────────────────────────────────

class TestFilterGapByCategoryExtra:
    def test_near(self):
        entries = _gap_entries_mixed()
        result = filter_gap_by_category(entries, "near")
        assert len(result) == 1
        assert all(e.category == "near" for e in result)

    def test_far(self):
        entries = _gap_entries_mixed()
        assert len(filter_gap_by_category(entries, "far")) == 1

    def test_touching(self):
        entries = _gap_entries_mixed()
        assert len(filter_gap_by_category(entries, "touching")) == 1

    def test_nonexistent(self):
        entries = _gap_entries_mixed()
        assert filter_gap_by_category(entries, "unknown") == []

    def test_empty(self):
        assert filter_gap_by_category([], "near") == []


# ─── filter_gap_by_max_distance ─────────────────────────────────────────────

class TestFilterGapByMaxDistanceExtra:
    def test_all_pass(self):
        entries = _gap_entries_mixed()
        assert len(filter_gap_by_max_distance(entries, 100.0)) == 5

    def test_strict(self):
        entries = _gap_entries_mixed()
        result = filter_gap_by_max_distance(entries, 1.0)
        assert len(result) == 2  # distance 1.0 and 0.5

    def test_none_pass(self):
        entries = _gap_entries_mixed()
        assert len(filter_gap_by_max_distance(entries, 0.1)) == 0

    def test_exact_boundary(self):
        entries = [_gap_entry(distance=5.0)]
        assert len(filter_gap_by_max_distance(entries, 5.0)) == 1

    def test_empty(self):
        assert filter_gap_by_max_distance([], 10.0) == []


# ─── top_k_closest_gaps ────────────────────────────────────────────────────

class TestTopKClosestGapsExtra:
    def test_top_1(self):
        entries = _gap_entries_mixed()
        result = top_k_closest_gaps(entries, 1)
        assert len(result) == 1
        assert result[0].distance == pytest.approx(0.5)

    def test_top_3(self):
        entries = _gap_entries_mixed()
        result = top_k_closest_gaps(entries, 3)
        assert len(result) == 3
        assert result[0].distance <= result[1].distance <= result[2].distance

    def test_k_greater_than_n(self):
        entries = _gap_entries_mixed()
        result = top_k_closest_gaps(entries, 100)
        assert len(result) == len(entries)

    def test_empty(self):
        assert top_k_closest_gaps([], 5) == []

    def test_ascending_order(self):
        entries = _gap_entries_mixed()
        result = top_k_closest_gaps(entries, 5)
        dists = [e.distance for e in result]
        assert dists == sorted(dists)


# ─── best_gap_entry ─────────────────────────────────────────────────────────

class TestBestGapEntryExtra:
    def test_returns_closest(self):
        entries = _gap_entries_mixed()
        best = best_gap_entry(entries)
        assert best is not None
        assert best.distance == pytest.approx(0.5)

    def test_empty_returns_none(self):
        assert best_gap_entry([]) is None

    def test_single(self):
        e = _gap_entry(distance=3.3)
        assert best_gap_entry([e]).distance == pytest.approx(3.3)

    def test_returns_entry_type(self):
        entries = _gap_entries_mixed()
        assert isinstance(best_gap_entry(entries), GapScoreEntry)

    def test_tie_returns_one(self):
        entries = [_gap_entry(distance=2.0), _gap_entry(distance=2.0)]
        best = best_gap_entry(entries)
        assert best.distance == pytest.approx(2.0)


# ─── gap_score_stats ────────────────────────────────────────────────────────

class TestGapScoreStatsExtra:
    def test_empty(self):
        stats = gap_score_stats([])
        assert stats["count"] == 0
        assert stats["mean"] == pytest.approx(0.0)

    def test_single_std_zero(self):
        stats = gap_score_stats([_gap_entry(distance=4.0)])
        assert stats["std"] == pytest.approx(0.0)

    def test_count(self):
        entries = _gap_entries_mixed()
        stats = gap_score_stats(entries)
        assert stats["count"] == pytest.approx(5.0)

    def test_min_max(self):
        entries = _gap_entries_mixed()
        stats = gap_score_stats(entries)
        assert stats["min"] == pytest.approx(0.5)
        assert stats["max"] == pytest.approx(15.0)

    def test_mean(self):
        entries = [_gap_entry(distance=2.0), _gap_entry(distance=6.0)]
        stats = gap_score_stats(entries)
        assert stats["mean"] == pytest.approx(4.0)

    def test_std_positive_for_varied(self):
        entries = _gap_entries_mixed()
        assert gap_score_stats(entries)["std"] > 0.0


# ─── compare_gap_summaries ──────────────────────────────────────────────────

class TestCompareGapSummariesExtra:
    def test_identical(self):
        s = summarise_gap_score_entries(_gap_entries_mixed())
        delta = compare_gap_summaries(s, s)
        assert delta["mean_distance_delta"] == pytest.approx(0.0)
        assert delta["n_overlapping_delta"] == pytest.approx(0.0)
        assert delta["n_far_delta"] == pytest.approx(0.0)

    def test_a_closer(self):
        a = summarise_gap_score_entries([_gap_entry(distance=1.0)])
        b = summarise_gap_score_entries([_gap_entry(distance=5.0)])
        delta = compare_gap_summaries(a, b)
        assert delta["mean_distance_delta"] < 0.0

    def test_b_closer(self):
        a = summarise_gap_score_entries([_gap_entry(distance=10.0)])
        b = summarise_gap_score_entries([_gap_entry(distance=2.0)])
        delta = compare_gap_summaries(a, b)
        assert delta["mean_distance_delta"] > 0.0

    def test_overlapping_delta(self):
        a = summarise_gap_score_entries([_gap_entry(category="overlap")])
        b = summarise_gap_score_entries([_gap_entry(category="near")])
        delta = compare_gap_summaries(a, b)
        assert delta["n_overlapping_delta"] == pytest.approx(1.0)

    def test_returns_dict(self):
        s = summarise_gap_score_entries([_gap_entry()])
        assert isinstance(compare_gap_summaries(s, s), dict)


# ─── batch_summarise_gap_score_entries ──────────────────────────────────────

class TestBatchSummariseGapScoreEntriesExtra:
    def test_single_group(self):
        groups = [_gap_entries_mixed()]
        result = batch_summarise_gap_score_entries(groups)
        assert len(result) == 1
        assert isinstance(result[0], GapScoreSummary)

    def test_multiple_groups(self):
        groups = [
            [_gap_entry(distance=1.0)],
            [_gap_entry(distance=5.0)],
        ]
        result = batch_summarise_gap_score_entries(groups)
        assert len(result) == 2

    def test_empty(self):
        assert batch_summarise_gap_score_entries([]) == []

    def test_group_with_empty(self):
        result = batch_summarise_gap_score_entries([[]])
        assert len(result) == 1
        assert result[0].n_entries == 0

    def test_summaries_independent(self):
        groups = [
            [_gap_entry(distance=2.0)],
            [_gap_entry(distance=8.0)],
        ]
        result = batch_summarise_gap_score_entries(groups)
        assert result[0].mean_distance == pytest.approx(2.0)
        assert result[1].mean_distance == pytest.approx(8.0)
