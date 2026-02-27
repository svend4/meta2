"""Tests for puzzle_reconstruction/assembly/fragment_sorter.py"""
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


# ── SortConfig ────────────────────────────────────────────────────────────────

class TestSortConfig:
    def test_default_values(self):
        cfg = SortConfig()
        assert cfg.strategy == "id"
        assert cfg.reverse is False
        assert cfg.seed == 0

    def test_valid_strategies(self):
        for s in ("area", "score", "id", "random"):
            cfg = SortConfig(strategy=s)
            assert cfg.strategy == s

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy должна быть"):
            SortConfig(strategy="unknown")

    def test_negative_seed_raises(self):
        with pytest.raises(ValueError, match="seed должен быть >= 0"):
            SortConfig(seed=-1)

    def test_reverse_true(self):
        cfg = SortConfig(reverse=True)
        assert cfg.reverse is True

    def test_seed_zero_ok(self):
        cfg = SortConfig(seed=0)
        assert cfg.seed == 0


# ── FragmentSortInfo ──────────────────────────────────────────────────────────

class TestFragmentSortInfo:
    def test_valid_construction(self):
        f = FragmentSortInfo(fragment_id=0, area=100.0, score=0.5)
        assert f.fragment_id == 0
        assert f.area == 100.0
        assert f.score == 0.5

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError, match="fragment_id должен быть >= 0"):
            FragmentSortInfo(fragment_id=-1)

    def test_negative_area_raises(self):
        with pytest.raises(ValueError, match="area должна быть >= 0"):
            FragmentSortInfo(fragment_id=0, area=-1.0)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError, match="score должен быть >= 0"):
            FragmentSortInfo(fragment_id=0, score=-0.1)

    def test_default_values(self):
        f = FragmentSortInfo(fragment_id=0)
        assert f.area == 0.0
        assert f.score == 0.0
        assert f.meta == {}

    def test_meta_stored(self):
        f = FragmentSortInfo(fragment_id=0, meta={"key": "val"})
        assert f.meta["key"] == "val"

    def test_zero_area_ok(self):
        f = FragmentSortInfo(fragment_id=0, area=0.0)
        assert f.area == 0.0


# ── SortedFragment ────────────────────────────────────────────────────────────

class TestSortedFragment:
    def test_valid_construction(self):
        sf = SortedFragment(fragment_id=0, position=1, area=50.0, score=0.7)
        assert sf.fragment_id == 0
        assert sf.position == 1

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError, match="fragment_id должен быть >= 0"):
            SortedFragment(fragment_id=-1, position=0)

    def test_negative_position_raises(self):
        with pytest.raises(ValueError, match="position должна быть >= 0"):
            SortedFragment(fragment_id=0, position=-1)

    def test_negative_area_raises(self):
        with pytest.raises(ValueError, match="area должна быть >= 0"):
            SortedFragment(fragment_id=0, position=0, area=-5.0)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError, match="score должен быть >= 0"):
            SortedFragment(fragment_id=0, position=0, score=-0.1)

    def test_info_property_returns_string(self):
        sf = SortedFragment(fragment_id=2, position=1, area=100.0, score=0.5)
        info = sf.info
        assert isinstance(info, str)
        assert "2" in info


# ── sort_by_id ────────────────────────────────────────────────────────────────

class TestSortById:
    def _make(self, ids):
        return [FragmentSortInfo(fragment_id=i) for i in ids]

    def test_ascending(self):
        frags = self._make([3, 1, 4, 1, 5])
        result = sort_by_id(frags, reverse=False)
        ids = [f.fragment_id for f in result]
        assert ids == sorted(ids)

    def test_descending(self):
        frags = self._make([3, 1, 4])
        result = sort_by_id(frags, reverse=True)
        ids = [f.fragment_id for f in result]
        assert ids == sorted(ids, reverse=True)

    def test_empty(self):
        assert sort_by_id([], reverse=False) == []

    def test_single(self):
        frags = self._make([7])
        assert sort_by_id(frags)[0].fragment_id == 7


# ── sort_by_area ──────────────────────────────────────────────────────────────

class TestSortByArea:
    def _make(self, areas):
        return [FragmentSortInfo(fragment_id=i, area=a)
                for i, a in enumerate(areas)]

    def test_ascending(self):
        frags = self._make([300, 100, 200])
        result = sort_by_area(frags, reverse=False)
        areas = [f.area for f in result]
        assert areas == sorted(areas)

    def test_descending(self):
        frags = self._make([300, 100, 200])
        result = sort_by_area(frags, reverse=True)
        areas = [f.area for f in result]
        assert areas == sorted(areas, reverse=True)

    def test_empty(self):
        assert sort_by_area([]) == []

    def test_equal_areas(self):
        frags = self._make([100, 100, 100])
        result = sort_by_area(frags)
        assert len(result) == 3


# ── sort_by_score ─────────────────────────────────────────────────────────────

class TestSortByScore:
    def _make(self, scores):
        return [FragmentSortInfo(fragment_id=i, score=s)
                for i, s in enumerate(scores)]

    def test_ascending(self):
        frags = self._make([0.7, 0.3, 0.5])
        result = sort_by_score(frags, reverse=False)
        scores = [f.score for f in result]
        assert scores == sorted(scores)

    def test_descending(self):
        frags = self._make([0.7, 0.3, 0.5])
        result = sort_by_score(frags, reverse=True)
        scores = [f.score for f in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty(self):
        assert sort_by_score([]) == []


# ── sort_random ───────────────────────────────────────────────────────────────

class TestSortRandom:
    def _make(self, n):
        return [FragmentSortInfo(fragment_id=i, area=float(i))
                for i in range(n)]

    def test_same_seed_same_order(self):
        frags = self._make(10)
        r1 = sort_random(frags, seed=42)
        r2 = sort_random(frags, seed=42)
        assert [f.fragment_id for f in r1] == [f.fragment_id for f in r2]

    def test_different_seeds_may_differ(self):
        frags = self._make(20)
        r1 = sort_random(frags, seed=0)
        r2 = sort_random(frags, seed=1)
        # Different seeds should give different orders (probabilistically)
        ids1 = [f.fragment_id for f in r1]
        ids2 = [f.fragment_id for f in r2]
        # They might be the same, but with 20 elements it's very unlikely
        # Just check both have all elements
        assert set(ids1) == set(ids2) == set(range(20))

    def test_all_elements_preserved(self):
        frags = self._make(5)
        result = sort_random(frags, seed=0)
        assert set(f.fragment_id for f in result) == {0, 1, 2, 3, 4}

    def test_negative_seed_raises(self):
        frags = self._make(3)
        with pytest.raises(ValueError, match="seed должен быть >= 0"):
            sort_random(frags, seed=-1)

    def test_empty_list(self):
        assert sort_random([], seed=0) == []


# ── sort_fragments ────────────────────────────────────────────────────────────

class TestSortFragments:
    def _make(self, n):
        return [FragmentSortInfo(fragment_id=i, area=float(n - i),
                                 score=float(i) / n)
                for i in range(n)]

    def test_default_config_sorts_by_id(self):
        frags = self._make(4)
        import random
        random.shuffle(frags)
        result = sort_fragments(frags)
        ids = [f.fragment_id for f in result]
        assert ids == sorted(ids)

    def test_area_strategy(self):
        frags = self._make(4)
        cfg = SortConfig(strategy="area")
        result = sort_fragments(frags, cfg)
        areas = [f.area for f in result]
        assert areas == sorted(areas)

    def test_score_strategy(self):
        frags = self._make(4)
        cfg = SortConfig(strategy="score")
        result = sort_fragments(frags, cfg)
        scores = [f.score for f in result]
        assert scores == sorted(scores)

    def test_random_strategy(self):
        frags = self._make(10)
        cfg = SortConfig(strategy="random", seed=42)
        result = sort_fragments(frags, cfg)
        assert set(f.fragment_id for f in result) == set(range(10))

    def test_reverse_id_strategy(self):
        frags = self._make(4)
        cfg = SortConfig(strategy="id", reverse=True)
        result = sort_fragments(frags, cfg)
        ids = [f.fragment_id for f in result]
        assert ids == sorted(ids, reverse=True)

    def test_none_cfg_uses_default(self):
        frags = self._make(3)
        result = sort_fragments(frags, cfg=None)
        assert len(result) == 3


# ── assign_positions ──────────────────────────────────────────────────────────

class TestAssignPositions:
    def _make(self, ids):
        return [FragmentSortInfo(fragment_id=i) for i in ids]

    def test_positions_zero_based(self):
        frags = self._make([5, 3, 7])
        result = assign_positions(frags)
        positions = [sf.position for sf in result]
        assert positions == [0, 1, 2]

    def test_fragment_ids_preserved(self):
        frags = self._make([5, 3, 7])
        result = assign_positions(frags)
        ids = [sf.fragment_id for sf in result]
        assert ids == [5, 3, 7]

    def test_empty_list(self):
        assert assign_positions([]) == []

    def test_returns_sorted_fragment_instances(self):
        frags = self._make([0, 1])
        result = assign_positions(frags)
        for sf in result:
            assert isinstance(sf, SortedFragment)

    def test_area_and_score_preserved(self):
        frags = [FragmentSortInfo(fragment_id=0, area=50.0, score=0.8)]
        result = assign_positions(frags)
        assert result[0].area == 50.0
        assert result[0].score == 0.8


# ── reorder_by_positions ──────────────────────────────────────────────────────

class TestReorderByPositions:
    def test_sorts_by_position(self):
        sf0 = SortedFragment(fragment_id=5, position=2)
        sf1 = SortedFragment(fragment_id=3, position=0)
        sf2 = SortedFragment(fragment_id=7, position=1)
        result = reorder_by_positions([sf0, sf1, sf2])
        positions = [sf.position for sf in result]
        assert positions == [0, 1, 2]

    def test_empty_list(self):
        assert reorder_by_positions([]) == []

    def test_single_element(self):
        sf = SortedFragment(fragment_id=0, position=0)
        result = reorder_by_positions([sf])
        assert result[0].fragment_id == 0

    def test_already_sorted_unchanged(self):
        sfs = [SortedFragment(fragment_id=i, position=i) for i in range(4)]
        result = reorder_by_positions(sfs)
        for i, sf in enumerate(result):
            assert sf.position == i


# ── batch_sort ────────────────────────────────────────────────────────────────

class TestBatchSort:
    def _make_list(self, ids):
        return [FragmentSortInfo(fragment_id=i) for i in ids]

    def test_output_length(self):
        lists = [self._make_list([2, 0, 1]), self._make_list([5, 3])]
        result = batch_sort(lists)
        assert len(result) == 2

    def test_each_element_is_list_of_sorted_fragments(self):
        lists = [self._make_list([1, 0])]
        result = batch_sort(lists)
        assert isinstance(result[0], list)
        for sf in result[0]:
            assert isinstance(sf, SortedFragment)

    def test_positions_assigned(self):
        lists = [self._make_list([2, 0, 1])]
        result = batch_sort(lists)
        positions = [sf.position for sf in result[0]]
        assert sorted(positions) == [0, 1, 2]

    def test_empty_lists(self):
        result = batch_sort([])
        assert result == []

    def test_cfg_applied(self):
        lists = [self._make_list([3, 1, 2])]
        cfg = SortConfig(strategy="id", reverse=False)
        result = batch_sort(lists, cfg)
        ids = [sf.fragment_id for sf in result[0]]
        assert ids == sorted(ids)
