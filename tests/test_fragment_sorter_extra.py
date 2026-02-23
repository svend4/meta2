"""Extra tests for puzzle_reconstruction.assembly.fragment_sorter."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.assembly.fragment_sorter import (
    FragmentSortInfo,
    SortConfig,
    SortedFragment,
    assign_positions,
    batch_sort,
    reorder_by_positions,
    sort_by_area,
    sort_by_id,
    sort_by_score,
    sort_fragments,
    sort_random,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _frag(fid=0, area=1.0, score=0.5):
    return FragmentSortInfo(fragment_id=fid, area=area, score=score)


def _frags(n=4):
    return [_frag(fid=i, area=float(i + 1), score=float(i) / n) for i in range(n)]


# ─── TestSortConfigExtra ─────────────────────────────────────────────────────

class TestSortConfigExtra:
    def test_id_strategy_default(self):
        cfg = SortConfig(strategy="id")
        assert cfg.strategy == "id"

    def test_reverse_true(self):
        cfg = SortConfig(reverse=True)
        assert cfg.reverse is True

    def test_seed_zero_valid(self):
        cfg = SortConfig(seed=0)
        assert cfg.seed == 0

    def test_seed_large_valid(self):
        cfg = SortConfig(strategy="random", seed=9999)
        assert cfg.seed == 9999

    def test_all_valid_strategies(self):
        for s in ("id", "area", "score", "random"):
            SortConfig(strategy=s)

    def test_invalid_strategy_message(self):
        with pytest.raises(ValueError):
            SortConfig(strategy="bogus")


# ─── TestFragmentSortInfoExtra ────────────────────────────────────────────────

class TestFragmentSortInfoExtra:
    def test_id_zero_valid(self):
        f = FragmentSortInfo(fragment_id=0)
        assert f.fragment_id == 0

    def test_large_id(self):
        f = FragmentSortInfo(fragment_id=999)
        assert f.fragment_id == 999

    def test_area_zero_valid(self):
        f = FragmentSortInfo(fragment_id=0, area=0.0)
        assert f.area == 0.0

    def test_score_zero_valid(self):
        f = FragmentSortInfo(fragment_id=0, score=0.0)
        assert f.score == 0.0

    def test_high_score(self):
        f = FragmentSortInfo(fragment_id=0, score=100.0)
        assert f.score == 100.0

    def test_meta_stored(self):
        f = FragmentSortInfo(fragment_id=0, meta={"x": 1})
        assert f.meta["x"] == 1

    def test_meta_empty_by_default(self):
        f = FragmentSortInfo(fragment_id=1)
        assert f.meta == {}


# ─── TestSortedFragmentExtra ─────────────────────────────────────────────────

class TestSortedFragmentExtra:
    def test_position_zero_valid(self):
        sf = SortedFragment(fragment_id=0, position=0)
        assert sf.position == 0

    def test_large_position(self):
        sf = SortedFragment(fragment_id=0, position=999)
        assert sf.position == 999

    def test_area_stored(self):
        sf = SortedFragment(fragment_id=1, position=0, area=42.0)
        assert sf.area == 42.0

    def test_score_stored(self):
        sf = SortedFragment(fragment_id=1, position=0, score=0.88)
        assert sf.score == pytest.approx(0.88)

    def test_info_contains_position(self):
        sf = SortedFragment(fragment_id=2, position=5)
        assert "5" in sf.info

    def test_info_is_str(self):
        sf = SortedFragment(fragment_id=0, position=0)
        assert isinstance(sf.info, str)


# ─── TestSortByIdExtra ────────────────────────────────────────────────────────

class TestSortByIdExtra:
    def test_already_sorted(self):
        frags = [_frag(fid=i) for i in range(5)]
        result = sort_by_id(frags)
        assert [f.fragment_id for f in result] == list(range(5))

    def test_reverse_sorted(self):
        frags = [_frag(fid=i) for i in reversed(range(5))]
        result = sort_by_id(frags)
        assert [f.fragment_id for f in result] == list(range(5))

    def test_descending(self):
        frags = [_frag(fid=i) for i in range(5)]
        result = sort_by_id(frags, reverse=True)
        ids = [f.fragment_id for f in result]
        assert ids == sorted(ids, reverse=True)

    def test_two_elements(self):
        frags = [_frag(fid=2), _frag(fid=0)]
        result = sort_by_id(frags)
        assert result[0].fragment_id == 0

    def test_preserves_area(self):
        frags = [_frag(fid=1, area=50.0), _frag(fid=0, area=10.0)]
        result = sort_by_id(frags)
        assert result[0].area == 10.0


# ─── TestSortByAreaExtra ─────────────────────────────────────────────────────

class TestSortByAreaExtra:
    def test_ascending_correct(self):
        frags = [_frag(area=5.0), _frag(area=1.0), _frag(area=3.0)]
        result = sort_by_area(frags)
        assert [f.area for f in result] == [1.0, 3.0, 5.0]

    def test_descending_correct(self):
        frags = [_frag(area=5.0), _frag(area=1.0), _frag(area=3.0)]
        result = sort_by_area(frags, reverse=True)
        assert [f.area for f in result] == [5.0, 3.0, 1.0]

    def test_single_element(self):
        result = sort_by_area([_frag(area=7.0)])
        assert len(result) == 1

    def test_does_not_modify_original(self):
        frags = [_frag(fid=i, area=float(i)) for i in range(3, 0, -1)]
        orig = [f.area for f in frags]
        sort_by_area(frags)
        assert [f.area for f in frags] == orig


# ─── TestSortByScoreExtra ────────────────────────────────────────────────────

class TestSortByScoreExtra:
    def test_ascending(self):
        frags = [_frag(score=0.9), _frag(score=0.1), _frag(score=0.5)]
        result = sort_by_score(frags)
        scores = [f.score for f in result]
        assert scores == sorted(scores)

    def test_descending(self):
        frags = [_frag(score=0.9), _frag(score=0.1), _frag(score=0.5)]
        result = sort_by_score(frags, reverse=True)
        scores = [f.score for f in result]
        assert scores == sorted(scores, reverse=True)

    def test_single(self):
        result = sort_by_score([_frag(score=0.75)])
        assert len(result) == 1

    def test_all_equal_scores(self):
        frags = [_frag(score=0.5) for _ in range(4)]
        result = sort_by_score(frags)
        assert len(result) == 4


# ─── TestSortRandomExtra ─────────────────────────────────────────────────────

class TestSortRandomExtra:
    def test_seed_reproducible(self):
        frags = [_frag(fid=i) for i in range(8)]
        r1 = [f.fragment_id for f in sort_random(frags, seed=42)]
        r2 = [f.fragment_id for f in sort_random(frags, seed=42)]
        assert r1 == r2

    def test_length_preserved(self):
        frags = [_frag(fid=i) for i in range(10)]
        result = sort_random(frags, seed=0)
        assert len(result) == 10

    def test_same_elements(self):
        frags = [_frag(fid=i) for i in range(6)]
        result = sort_random(frags, seed=0)
        assert {f.fragment_id for f in result} == set(range(6))

    def test_single_element(self):
        result = sort_random([_frag(fid=5)], seed=0)
        assert result[0].fragment_id == 5


# ─── TestSortFragmentsExtra ───────────────────────────────────────────────────

class TestSortFragmentsExtra:
    def test_id_strategy_ascending(self):
        frags = _frags()
        cfg = SortConfig(strategy="id")
        result = sort_fragments(frags, cfg)
        ids = [f.fragment_id for f in result]
        assert ids == sorted(ids)

    def test_area_strategy_ascending(self):
        frags = _frags()
        cfg = SortConfig(strategy="area")
        result = sort_fragments(frags, cfg)
        areas = [f.area for f in result]
        assert areas == sorted(areas)

    def test_score_descending(self):
        frags = _frags()
        cfg = SortConfig(strategy="score", reverse=True)
        result = sort_fragments(frags, cfg)
        scores = [f.score for f in result]
        assert scores == sorted(scores, reverse=True)

    def test_random_preserves_count(self):
        frags = _frags()
        cfg = SortConfig(strategy="random", seed=7)
        result = sort_fragments(frags, cfg)
        assert len(result) == len(frags)

    def test_none_config_defaults(self):
        frags = _frags()
        result = sort_fragments(frags, None)
        ids = [f.fragment_id for f in result]
        assert ids == sorted(ids)


# ─── TestAssignPositionsExtra ─────────────────────────────────────────────────

class TestAssignPositionsExtra:
    def test_position_sequence(self):
        frags = _frags()
        result = assign_positions(frags)
        positions = [sf.position for sf in result]
        assert positions == list(range(len(frags)))

    def test_single_frag(self):
        result = assign_positions([_frag(fid=5, area=10.0)])
        assert result[0].position == 0
        assert result[0].fragment_id == 5

    def test_area_forwarded(self):
        f = _frag(fid=0, area=77.7)
        result = assign_positions([f])
        assert result[0].area == pytest.approx(77.7)

    def test_score_forwarded(self):
        f = _frag(fid=0, score=0.88)
        result = assign_positions([f])
        assert result[0].score == pytest.approx(0.88)

    def test_returns_sorted_fragments(self):
        frags = _frags()
        result = assign_positions(frags)
        assert all(isinstance(sf, SortedFragment) for sf in result)


# ─── TestReorderByPositionsExtra ─────────────────────────────────────────────

class TestReorderByPositionsExtra:
    def test_reverse_positions(self):
        n = 5
        sfs = [SortedFragment(fragment_id=i, position=n - 1 - i) for i in range(n)]
        result = reorder_by_positions(sfs)
        positions = [sf.position for sf in result]
        assert positions == sorted(positions)

    def test_single(self):
        sf = SortedFragment(fragment_id=0, position=0)
        result = reorder_by_positions([sf])
        assert result[0].position == 0

    def test_already_ordered(self):
        sfs = [SortedFragment(fragment_id=i, position=i) for i in range(4)]
        result = reorder_by_positions(sfs)
        assert [sf.position for sf in result] == [0, 1, 2, 3]

    def test_all_positions_present(self):
        n = 6
        sfs = [SortedFragment(fragment_id=i, position=(i * 3) % n) for i in range(n)]
        result = reorder_by_positions(sfs)
        assert len(result) == n


# ─── TestBatchSortExtra ───────────────────────────────────────────────────────

class TestBatchSortExtra:
    def test_sorted_fragment_types(self):
        result = batch_sort([_frags()])
        assert all(isinstance(sf, SortedFragment) for sf in result[0])

    def test_positions_zero_based(self):
        result = batch_sort([_frags()])
        positions = [sf.position for sf in result[0]]
        assert sorted(positions) == list(range(len(_frags())))

    def test_area_strategy_batch(self):
        cfg = SortConfig(strategy="area")
        result = batch_sort([_frags()], cfg)
        areas = [sf.area for sf in result[0]]
        assert areas == sorted(areas)

    def test_score_descending_batch(self):
        cfg = SortConfig(strategy="score", reverse=True)
        result = batch_sort([_frags()], cfg)
        scores = [sf.score for sf in result[0]]
        assert scores == sorted(scores, reverse=True)

    def test_multiple_groups_length(self):
        result = batch_sort([_frags(3), _frags(5)])
        assert len(result[0]) == 3
        assert len(result[1]) == 5

    def test_empty_group(self):
        result = batch_sort([[]])
        assert result[0] == []
