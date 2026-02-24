"""Extra tests for puzzle_reconstruction/assembly/assembly_state.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.assembly.assembly_state import (
    PlacedFragment,
    AssemblyState,
    create_state,
    place_fragment,
    remove_fragment,
    add_adjacency,
    get_neighbors,
    compute_coverage,
    is_complete,
    to_dict,
    from_dict,
)


# ─── PlacedFragment (extra) ──────────────────────────────────────────────────

class TestPlacedFragmentExtra:
    def test_defaults(self):
        pf = PlacedFragment(idx=0, position=(1.0, 2.0))
        assert pf.angle == pytest.approx(0.0)
        assert pf.scale == pytest.approx(1.0)
        assert pf.meta == {}

    def test_custom_angle(self):
        pf = PlacedFragment(idx=1, position=(0.0, 0.0), angle=90.0)
        assert pf.angle == pytest.approx(90.0)

    def test_custom_scale(self):
        pf = PlacedFragment(idx=2, position=(0.0, 0.0), scale=2.0)
        assert pf.scale == pytest.approx(2.0)

    def test_large_idx_ok(self):
        pf = PlacedFragment(idx=9999, position=(0.0, 0.0))
        assert pf.idx == 9999

    def test_meta_stored(self):
        pf = PlacedFragment(idx=0, position=(0.0, 0.0), meta={"label": "A"})
        assert pf.meta["label"] == "A"

    def test_position_stored(self):
        pf = PlacedFragment(idx=3, position=(5.0, 7.0))
        assert pf.position == (5.0, 7.0)


# ─── AssemblyState (extra) ───────────────────────────────────────────────────

class TestAssemblyStateExtra:
    def test_initial_placed_empty(self):
        s = create_state(4)
        assert len(s.placed) == 0

    def test_initial_adjacency_empty(self):
        s = create_state(3)
        assert len(s.adjacency) == 0

    def test_initial_step_zero(self):
        s = create_state(5)
        assert s.step == 0

    def test_n_fragments_stored(self):
        s = create_state(7)
        assert s.n_fragments == 7

    def test_n_fragments_zero_raises(self):
        with pytest.raises(ValueError):
            create_state(0)


# ─── place_fragment (extra) ──────────────────────────────────────────────────

class TestPlaceFragmentExtra:
    def test_returns_new_state(self):
        s = create_state(3)
        s2 = place_fragment(s, 0, position=(0.0, 0.0))
        assert s2 is not s

    def test_original_unchanged(self):
        s = create_state(3)
        place_fragment(s, 0, position=(1.0, 1.0))
        assert 0 not in s.placed

    def test_placed_count_increases(self):
        s = create_state(3)
        s2 = place_fragment(s, 1, position=(0.0, 0.0))
        assert len(s2.placed) == 1

    def test_out_of_range_idx_raises(self):
        s = create_state(3)
        with pytest.raises(ValueError):
            place_fragment(s, 5, position=(0.0, 0.0))

    def test_angle_stored(self):
        s = create_state(3)
        s2 = place_fragment(s, 0, position=(0.0, 0.0), angle=45.0)
        assert s2.placed[0].angle == pytest.approx(45.0)

    def test_scale_stored(self):
        s = create_state(3)
        s2 = place_fragment(s, 0, position=(0.0, 0.0), scale=1.5)
        assert s2.placed[0].scale == pytest.approx(1.5)

    def test_duplicate_place_raises(self):
        s = create_state(3)
        s2 = place_fragment(s, 0, position=(0.0, 0.0))
        with pytest.raises(ValueError):
            place_fragment(s2, 0, position=(1.0, 0.0))

    def test_step_incremented(self):
        s = create_state(3)
        s2 = place_fragment(s, 0, position=(0.0, 0.0))
        assert s2.step == s.step + 1


# ─── remove_fragment (extra) ─────────────────────────────────────────────────

class TestRemoveFragmentExtra:
    def test_removes_fragment(self):
        s = create_state(3)
        s2 = place_fragment(s, 1, position=(0.0, 0.0))
        s3 = remove_fragment(s2, 1)
        assert 1 not in s3.placed

    def test_returns_new_state(self):
        s = create_state(3)
        s2 = place_fragment(s, 0, position=(0.0, 0.0))
        s3 = remove_fragment(s2, 0)
        assert s3 is not s2

    def test_original_unchanged(self):
        s = create_state(3)
        s2 = place_fragment(s, 0, position=(0.0, 0.0))
        remove_fragment(s2, 0)
        assert 0 in s2.placed

    def test_remove_not_placed_raises(self):
        s = create_state(3)
        with pytest.raises((ValueError, KeyError)):
            remove_fragment(s, 2)

    def test_other_fragments_remain(self):
        s = create_state(4)
        s2 = place_fragment(s, 0, position=(0.0, 0.0))
        s3 = place_fragment(s2, 1, position=(1.0, 0.0))
        s4 = remove_fragment(s3, 0)
        assert 1 in s4.placed


# ─── add_adjacency (extra) ───────────────────────────────────────────────────

class TestAddAdjacencyExtra:
    def test_adds_edge(self):
        s = create_state(4)
        s2 = place_fragment(s, 0, position=(0.0, 0.0))
        s3 = place_fragment(s2, 1, position=(1.0, 0.0))
        s4 = add_adjacency(s3, 0, 1)
        assert 1 in get_neighbors(s4, 0)

    def test_returns_new_state(self):
        s = create_state(3)
        s2 = place_fragment(s, 0, position=(0.0, 0.0))
        s3 = place_fragment(s2, 1, position=(1.0, 0.0))
        s4 = add_adjacency(s3, 0, 1)
        assert s4 is not s3

    def test_symmetric_adjacency(self):
        s = create_state(4)
        s2 = place_fragment(s, 0, position=(0.0, 0.0))
        s3 = place_fragment(s2, 2, position=(1.0, 0.0))
        s4 = add_adjacency(s3, 0, 2)
        assert 0 in get_neighbors(s4, 2)
        assert 2 in get_neighbors(s4, 0)

    def test_same_idx_raises(self):
        s = create_state(3)
        s2 = place_fragment(s, 0, position=(0.0, 0.0))
        with pytest.raises(ValueError):
            add_adjacency(s2, 0, 0)


# ─── get_neighbors (extra) ───────────────────────────────────────────────────

class TestGetNeighborsExtra:
    def test_no_neighbors_empty_list(self):
        s = create_state(3)
        s2 = place_fragment(s, 0, position=(0.0, 0.0))
        assert get_neighbors(s2, 0) == []

    def test_multiple_neighbors(self):
        s = create_state(5)
        for i in range(4):
            s = place_fragment(s, i, position=(float(i), 0.0))
        s = add_adjacency(s, 0, 1)
        s = add_adjacency(s, 0, 2)
        s = add_adjacency(s, 0, 3)
        neighbors = get_neighbors(s, 0)
        assert set(neighbors) == {1, 2, 3}

    def test_unplaced_idx_returns_empty(self):
        s = create_state(3)
        result = get_neighbors(s, 0)
        assert result == []


# ─── compute_coverage (extra) ────────────────────────────────────────────────

class TestComputeCoverageExtra:
    def test_empty_state_zero(self):
        s = create_state(4)
        assert compute_coverage(s) == pytest.approx(0.0)

    def test_all_placed_one(self):
        s = create_state(3)
        for i in range(3):
            s = place_fragment(s, i, position=(float(i), 0.0))
        assert compute_coverage(s) == pytest.approx(1.0)

    def test_partial_coverage(self):
        s = create_state(4)
        s = place_fragment(s, 0, position=(0.0, 0.0))
        s = place_fragment(s, 1, position=(1.0, 0.0))
        assert compute_coverage(s) == pytest.approx(0.5)

    def test_one_of_ten(self):
        s = create_state(10)
        s = place_fragment(s, 0, position=(0.0, 0.0))
        assert compute_coverage(s) == pytest.approx(0.1)


# ─── is_complete (extra) ─────────────────────────────────────────────────────

class TestIsCompleteExtra:
    def test_empty_not_complete(self):
        s = create_state(3)
        assert not is_complete(s)

    def test_partial_not_complete(self):
        s = create_state(3)
        s = place_fragment(s, 0, position=(0.0, 0.0))
        assert not is_complete(s)

    def test_all_placed_complete(self):
        s = create_state(3)
        for i in range(3):
            s = place_fragment(s, i, position=(float(i), 0.0))
        assert is_complete(s)

    def test_single_fragment_complete(self):
        s = create_state(1)
        s = place_fragment(s, 0, position=(0.0, 0.0))
        assert is_complete(s)


# ─── to_dict / from_dict (extra) ─────────────────────────────────────────────

class TestToDictFromDictExtra:
    def test_roundtrip_empty(self):
        s = create_state(3)
        d = to_dict(s)
        s2 = from_dict(d)
        assert s2.n_fragments == 3
        assert len(s2.placed) == 0

    def test_roundtrip_with_fragments(self):
        s = create_state(4)
        s = place_fragment(s, 0, position=(1.0, 2.0))
        s = place_fragment(s, 2, position=(3.0, 4.0))
        d = to_dict(s)
        s2 = from_dict(d)
        assert set(s2.placed.keys()) == {0, 2}

    def test_roundtrip_with_adjacency(self):
        s = create_state(3)
        s = place_fragment(s, 0, position=(0.0, 0.0))
        s = place_fragment(s, 1, position=(1.0, 0.0))
        s = add_adjacency(s, 0, 1)
        d = to_dict(s)
        s2 = from_dict(d)
        assert 1 in get_neighbors(s2, 0)

    def test_to_dict_returns_dict(self):
        s = create_state(2)
        assert isinstance(to_dict(s), dict)

    def test_position_preserved_roundtrip(self):
        s = create_state(2)
        s = place_fragment(s, 0, position=(7.5, 3.2))
        d = to_dict(s)
        s2 = from_dict(d)
        pos = s2.placed[0].position
        assert pos[0] == pytest.approx(7.5)
        assert pos[1] == pytest.approx(3.2)

    def test_angle_preserved_roundtrip(self):
        s = create_state(2)
        s = place_fragment(s, 0, position=(0.0, 0.0), angle=30.0)
        d = to_dict(s)
        s2 = from_dict(d)
        assert s2.placed[0].angle == pytest.approx(30.0)
