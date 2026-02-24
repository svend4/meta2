"""Extra tests for puzzle_reconstruction/assembly/fragment_sorter.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.assembly.fragment_sorter import (
    SortConfig,
    FragmentSortInfo,
    SortedFragment,
    sort_by_id,
    sort_by_area,
    sort_by_score,
    sort_random,
    sort_fragments,
    assign_positions,
    reorder_by_positions,
    batch_sort,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _frags():
    """Three fragments with different areas and scores."""
    return [
        FragmentSortInfo(fragment_id=2, area=100.0, score=0.5),
        FragmentSortInfo(fragment_id=0, area=300.0, score=0.9),
        FragmentSortInfo(fragment_id=1, area=200.0, score=0.1),
    ]


# ─── SortConfig ─────────────────────────────────────────────────────────────

class TestSortConfigExtra:
    def test_defaults(self):
        c = SortConfig()
        assert c.strategy == "id"
        assert c.reverse is False
        assert c.seed == 0

    def test_valid_strategies(self):
        for s in ("area", "score", "id", "random"):
            SortConfig(strategy=s)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            SortConfig(strategy="bad")

    def test_negative_seed_raises(self):
        with pytest.raises(ValueError):
            SortConfig(seed=-1)


# ─── FragmentSortInfo ───────────────────────────────────────────────────────

class TestFragmentSortInfoExtra:
    def test_valid(self):
        f = FragmentSortInfo(fragment_id=0, area=50.0, score=0.5)
        assert f.fragment_id == 0

    def test_negative_id_raises(self):
        with pytest.raises(ValueError):
            FragmentSortInfo(fragment_id=-1)

    def test_negative_area_raises(self):
        with pytest.raises(ValueError):
            FragmentSortInfo(fragment_id=0, area=-1.0)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            FragmentSortInfo(fragment_id=0, score=-0.1)


# ─── SortedFragment ─────────────────────────────────────────────────────────

class TestSortedFragmentExtra:
    def test_valid(self):
        sf = SortedFragment(fragment_id=0, position=0, area=10.0, score=0.5)
        assert sf.fragment_id == 0
        assert sf.position == 0

    def test_negative_id_raises(self):
        with pytest.raises(ValueError):
            SortedFragment(fragment_id=-1, position=0)

    def test_negative_position_raises(self):
        with pytest.raises(ValueError):
            SortedFragment(fragment_id=0, position=-1)

    def test_negative_area_raises(self):
        with pytest.raises(ValueError):
            SortedFragment(fragment_id=0, position=0, area=-1.0)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            SortedFragment(fragment_id=0, position=0, score=-0.1)

    def test_info_property(self):
        sf = SortedFragment(fragment_id=3, position=1, area=100.0, score=0.5)
        assert "3" in sf.info
        assert "1" in sf.info


# ─── sort_by_id ─────────────────────────────────────────────────────────────

class TestSortByIdExtra:
    def test_ascending(self):
        result = sort_by_id(_frags())
        ids = [f.fragment_id for f in result]
        assert ids == [0, 1, 2]

    def test_descending(self):
        result = sort_by_id(_frags(), reverse=True)
        ids = [f.fragment_id for f in result]
        assert ids == [2, 1, 0]


# ─── sort_by_area ───────────────────────────────────────────────────────────

class TestSortByAreaExtra:
    def test_ascending(self):
        result = sort_by_area(_frags())
        areas = [f.area for f in result]
        assert areas == sorted(areas)

    def test_descending(self):
        result = sort_by_area(_frags(), reverse=True)
        areas = [f.area for f in result]
        assert areas == sorted(areas, reverse=True)


# ─── sort_by_score ──────────────────────────────────────────────────────────

class TestSortByScoreExtra:
    def test_ascending(self):
        result = sort_by_score(_frags())
        scores = [f.score for f in result]
        assert scores == sorted(scores)

    def test_descending(self):
        result = sort_by_score(_frags(), reverse=True)
        scores = [f.score for f in result]
        assert scores == sorted(scores, reverse=True)


# ─── sort_random ────────────────────────────────────────────────────────────

class TestSortRandomExtra:
    def test_deterministic(self):
        r1 = sort_random(_frags(), seed=42)
        r2 = sort_random(_frags(), seed=42)
        assert [f.fragment_id for f in r1] == [f.fragment_id for f in r2]

    def test_negative_seed_raises(self):
        with pytest.raises(ValueError):
            sort_random(_frags(), seed=-1)

    def test_same_length(self):
        result = sort_random(_frags(), seed=0)
        assert len(result) == 3


# ─── sort_fragments ─────────────────────────────────────────────────────────

class TestSortFragmentsExtra:
    def test_by_id(self):
        cfg = SortConfig(strategy="id")
        result = sort_fragments(_frags(), cfg)
        ids = [f.fragment_id for f in result]
        assert ids == [0, 1, 2]

    def test_by_area(self):
        cfg = SortConfig(strategy="area")
        result = sort_fragments(_frags(), cfg)
        areas = [f.area for f in result]
        assert areas == sorted(areas)

    def test_by_score(self):
        cfg = SortConfig(strategy="score", reverse=True)
        result = sort_fragments(_frags(), cfg)
        scores = [f.score for f in result]
        assert scores == sorted(scores, reverse=True)

    def test_random(self):
        cfg = SortConfig(strategy="random", seed=42)
        result = sort_fragments(_frags(), cfg)
        assert len(result) == 3

    def test_default_config(self):
        result = sort_fragments(_frags())
        ids = [f.fragment_id for f in result]
        assert ids == [0, 1, 2]


# ─── assign_positions ───────────────────────────────────────────────────────

class TestAssignPositionsExtra:
    def test_positions(self):
        frags = sort_by_id(_frags())
        sfs = assign_positions(frags)
        assert [sf.position for sf in sfs] == [0, 1, 2]
        assert all(isinstance(sf, SortedFragment) for sf in sfs)

    def test_empty(self):
        assert assign_positions([]) == []


# ─── reorder_by_positions ───────────────────────────────────────────────────

class TestReorderByPositionsExtra:
    def test_reorders(self):
        sfs = [
            SortedFragment(fragment_id=0, position=2),
            SortedFragment(fragment_id=1, position=0),
            SortedFragment(fragment_id=2, position=1),
        ]
        result = reorder_by_positions(sfs)
        assert [sf.position for sf in result] == [0, 1, 2]

    def test_empty(self):
        assert reorder_by_positions([]) == []


# ─── batch_sort ─────────────────────────────────────────────────────────────

class TestBatchSortExtra:
    def test_empty(self):
        assert batch_sort([]) == []

    def test_length(self):
        results = batch_sort([_frags(), _frags()])
        assert len(results) == 2

    def test_result_type(self):
        results = batch_sort([_frags()])
        assert all(isinstance(sf, SortedFragment) for sf in results[0])
