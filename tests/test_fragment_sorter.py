"""Тесты для puzzle_reconstruction.assembly.fragment_sorter."""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _frag(fid, area=1.0, score=0.5):
    return FragmentSortInfo(fragment_id=fid, area=area, score=score)


def _frags():
    return [
        _frag(3, area=30.0, score=0.3),
        _frag(1, area=10.0, score=0.9),
        _frag(2, area=20.0, score=0.6),
        _frag(0, area=5.0,  score=0.1),
    ]


# ─── TestSortConfig ───────────────────────────────────────────────────────────

class TestSortConfig:
    def test_defaults(self):
        cfg = SortConfig()
        assert cfg.strategy == "id"
        assert cfg.reverse is False
        assert cfg.seed == 0

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            SortConfig(strategy="unknown")

    def test_negative_seed(self):
        with pytest.raises(ValueError):
            SortConfig(seed=-1)

    def test_valid_area(self):
        cfg = SortConfig(strategy="area", reverse=True)
        assert cfg.strategy == "area"
        assert cfg.reverse is True

    def test_valid_score(self):
        cfg = SortConfig(strategy="score")
        assert cfg.strategy == "score"

    def test_valid_random(self):
        cfg = SortConfig(strategy="random", seed=42)
        assert cfg.strategy == "random"
        assert cfg.seed == 42


# ─── TestFragmentSortInfo ─────────────────────────────────────────────────────

class TestFragmentSortInfo:
    def test_basic_construction(self):
        f = FragmentSortInfo(fragment_id=5, area=100.0, score=0.8)
        assert f.fragment_id == 5
        assert f.area == 100.0
        assert f.score == 0.8

    def test_defaults(self):
        f = FragmentSortInfo(fragment_id=0)
        assert f.area == 0.0
        assert f.score == 0.0

    def test_negative_id_raises(self):
        with pytest.raises(ValueError):
            FragmentSortInfo(fragment_id=-1)

    def test_negative_area_raises(self):
        with pytest.raises(ValueError):
            FragmentSortInfo(fragment_id=0, area=-1.0)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            FragmentSortInfo(fragment_id=0, score=-0.1)

    def test_meta_default_empty(self):
        f = FragmentSortInfo(fragment_id=0)
        assert f.meta == {}

    def test_meta_custom(self):
        f = FragmentSortInfo(fragment_id=1, meta={"key": "val"})
        assert f.meta["key"] == "val"


# ─── TestSortedFragment ───────────────────────────────────────────────────────

class TestSortedFragment:
    def test_basic_construction(self):
        sf = SortedFragment(fragment_id=2, position=0)
        assert sf.fragment_id == 2
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

    def test_info_prop_is_string(self):
        sf = SortedFragment(fragment_id=1, position=2, area=50.0, score=0.75)
        assert isinstance(sf.info, str)
        assert "1" in sf.info
        assert "2" in sf.info


# ─── TestSortById ─────────────────────────────────────────────────────────────

class TestSortById:
    def test_ascending_order(self):
        frags = _frags()
        result = sort_by_id(frags)
        ids = [f.fragment_id for f in result]
        assert ids == sorted(ids)

    def test_descending_order(self):
        frags = _frags()
        result = sort_by_id(frags, reverse=True)
        ids = [f.fragment_id for f in result]
        assert ids == sorted(ids, reverse=True)

    def test_empty_list(self):
        assert sort_by_id([]) == []

    def test_single_element(self):
        f = _frag(7)
        result = sort_by_id([f])
        assert len(result) == 1

    def test_does_not_modify_original(self):
        frags = _frags()
        original_ids = [f.fragment_id for f in frags]
        sort_by_id(frags)
        assert [f.fragment_id for f in frags] == original_ids


# ─── TestSortByArea ───────────────────────────────────────────────────────────

class TestSortByArea:
    def test_ascending_order(self):
        frags = _frags()
        result = sort_by_area(frags)
        areas = [f.area for f in result]
        assert areas == sorted(areas)

    def test_descending_order(self):
        frags = _frags()
        result = sort_by_area(frags, reverse=True)
        areas = [f.area for f in result]
        assert areas == sorted(areas, reverse=True)

    def test_empty_list(self):
        assert sort_by_area([]) == []

    def test_equal_areas_stable(self):
        frags = [_frag(i, area=10.0) for i in range(3)]
        result = sort_by_area(frags)
        assert len(result) == 3


# ─── TestSortByScore ──────────────────────────────────────────────────────────

class TestSortByScore:
    def test_ascending_order(self):
        frags = _frags()
        result = sort_by_score(frags)
        scores = [f.score for f in result]
        assert scores == sorted(scores)

    def test_descending_order(self):
        frags = _frags()
        result = sort_by_score(frags, reverse=True)
        scores = [f.score for f in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_list(self):
        assert sort_by_score([]) == []


# ─── TestSortRandom ───────────────────────────────────────────────────────────

class TestSortRandom:
    def test_same_seed_same_order(self):
        frags = _frags()
        r1 = [f.fragment_id for f in sort_random(frags, seed=7)]
        r2 = [f.fragment_id for f in sort_random(frags, seed=7)]
        assert r1 == r2

    def test_different_seeds_likely_different(self):
        frags = [_frag(i) for i in range(10)]
        r1 = [f.fragment_id for f in sort_random(frags, seed=0)]
        r2 = [f.fragment_id for f in sort_random(frags, seed=99)]
        # Very unlikely to be identical for 10 elements
        assert r1 != r2

    def test_preserves_all_elements(self):
        frags = _frags()
        result = sort_random(frags, seed=1)
        assert set(f.fragment_id for f in result) == set(f.fragment_id for f in frags)

    def test_negative_seed_raises(self):
        with pytest.raises(ValueError):
            sort_random(_frags(), seed=-1)

    def test_empty_list(self):
        assert sort_random([]) == []


# ─── TestSortFragments ────────────────────────────────────────────────────────

class TestSortFragments:
    def test_default_cfg_sorts_by_id(self):
        frags = _frags()
        result = sort_fragments(frags)
        ids = [f.fragment_id for f in result]
        assert ids == sorted(ids)

    def test_area_strategy(self):
        frags = _frags()
        cfg = SortConfig(strategy="area")
        result = sort_fragments(frags, cfg)
        areas = [f.area for f in result]
        assert areas == sorted(areas)

    def test_score_strategy(self):
        frags = _frags()
        cfg = SortConfig(strategy="score", reverse=True)
        result = sort_fragments(frags, cfg)
        scores = [f.score for f in result]
        assert scores == sorted(scores, reverse=True)

    def test_random_strategy(self):
        frags = _frags()
        cfg = SortConfig(strategy="random", seed=42)
        result = sort_fragments(frags, cfg)
        assert len(result) == len(frags)

    def test_none_cfg_uses_defaults(self):
        frags = _frags()
        result = sort_fragments(frags, None)
        ids = [f.fragment_id for f in result]
        assert ids == sorted(ids)


# ─── TestAssignPositions ──────────────────────────────────────────────────────

class TestAssignPositions:
    def test_positions_are_0_based(self):
        frags = _frags()
        result = assign_positions(frags)
        positions = [sf.position for sf in result]
        assert positions == list(range(len(frags)))

    def test_fragment_ids_preserved(self):
        frags = sort_by_id(_frags())
        result = assign_positions(frags)
        ids = [sf.fragment_id for sf in result]
        expected = [f.fragment_id for f in frags]
        assert ids == expected

    def test_returns_sorted_fragments(self):
        frags = _frags()
        result = assign_positions(frags)
        assert all(isinstance(sf, SortedFragment) for sf in result)

    def test_empty_list(self):
        assert assign_positions([]) == []

    def test_areas_and_scores_copied(self):
        f = _frag(0, area=42.0, score=0.77)
        result = assign_positions([f])
        assert result[0].area == 42.0
        assert result[0].score == 0.77


# ─── TestReorderByPositions ───────────────────────────────────────────────────

class TestReorderByPositions:
    def test_reorder_ascending(self):
        sfs = [
            SortedFragment(fragment_id=2, position=2),
            SortedFragment(fragment_id=0, position=0),
            SortedFragment(fragment_id=1, position=1),
        ]
        result = reorder_by_positions(sfs)
        assert [sf.position for sf in result] == [0, 1, 2]

    def test_empty_list(self):
        assert reorder_by_positions([]) == []

    def test_preserves_all_elements(self):
        sfs = [SortedFragment(fragment_id=i, position=i) for i in range(5)]
        result = reorder_by_positions(sfs)
        assert len(result) == 5


# ─── TestBatchSort ────────────────────────────────────────────────────────────

class TestBatchSort:
    def test_empty_batch(self):
        assert batch_sort([]) == []

    def test_single_list(self):
        result = batch_sort([_frags()])
        assert len(result) == 1
        assert all(isinstance(sf, SortedFragment) for sf in result[0])

    def test_multiple_lists(self):
        batch = [_frags(), _frags()[:2], [_frag(0)]]
        result = batch_sort(batch)
        assert len(result) == 3

    def test_positions_consistent(self):
        result = batch_sort([_frags()])
        positions = [sf.position for sf in result[0]]
        assert positions == list(range(len(_frags())))

    def test_custom_cfg(self):
        cfg = SortConfig(strategy="score", reverse=True)
        result = batch_sort([_frags()], cfg)
        scores = [sf.score for sf in result[0]]
        assert scores == sorted(scores, reverse=True)
