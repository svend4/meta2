"""Tests for assembly/assembly_state.py."""
import pytest

from puzzle_reconstruction.assembly.assembly_state import (
    AssemblyState,
    PlacedFragment,
    add_adjacency,
    compute_coverage,
    create_state,
    from_dict,
    get_neighbors,
    is_complete,
    place_fragment,
    remove_fragment,
    to_dict,
)


# ─── PlacedFragment ───────────────────────────────────────────────────────────

class TestPlacedFragment:
    def test_basic_creation(self):
        pf = PlacedFragment(idx=3, position=(10.0, 20.0))
        assert pf.idx == 3
        assert pf.position == (10.0, 20.0)

    def test_default_angle_zero(self):
        pf = PlacedFragment(idx=0, position=(0.0, 0.0))
        assert pf.angle == 0.0

    def test_default_scale_one(self):
        pf = PlacedFragment(idx=0, position=(0.0, 0.0))
        assert pf.scale == 1.0

    def test_default_meta_empty(self):
        pf = PlacedFragment(idx=0, position=(0.0, 0.0))
        assert pf.meta == {}

    def test_custom_angle_and_scale(self):
        pf = PlacedFragment(idx=1, position=(5.0, 5.0), angle=45.0, scale=2.0)
        assert pf.angle == 45.0
        assert pf.scale == 2.0

    def test_meta_stored(self):
        pf = PlacedFragment(idx=0, position=(0.0, 0.0), meta={"score": 0.9})
        assert pf.meta["score"] == pytest.approx(0.9)

    def test_repr_contains_idx(self):
        pf = PlacedFragment(idx=7, position=(1.0, 2.0))
        assert "7" in repr(pf)

    def test_repr_contains_angle(self):
        pf = PlacedFragment(idx=0, position=(0.0, 0.0), angle=30.0)
        assert "30.0" in repr(pf)


# ─── AssemblyState ────────────────────────────────────────────────────────────

class TestAssemblyState:
    def test_basic_creation(self):
        state = AssemblyState(n_fragments=5)
        assert state.n_fragments == 5

    def test_placed_default_empty(self):
        state = AssemblyState(n_fragments=3)
        assert state.placed == {}

    def test_adjacency_default_empty(self):
        state = AssemblyState(n_fragments=3)
        assert state.adjacency == {}

    def test_step_default_zero(self):
        state = AssemblyState(n_fragments=3)
        assert state.step == 0

    def test_repr_contains_n(self):
        state = AssemblyState(n_fragments=10)
        assert "10" in repr(state)

    def test_repr_contains_placed_count(self):
        state = AssemblyState(n_fragments=10)
        assert "placed=0" in repr(state)


# ─── create_state ─────────────────────────────────────────────────────────────

class TestCreateState:
    def test_returns_assembly_state(self):
        state = create_state(5)
        assert isinstance(state, AssemblyState)

    def test_n_fragments_set(self):
        state = create_state(7)
        assert state.n_fragments == 7

    def test_placed_empty(self):
        state = create_state(5)
        assert len(state.placed) == 0

    def test_adjacency_empty(self):
        state = create_state(5)
        assert len(state.adjacency) == 0

    def test_step_is_zero(self):
        state = create_state(5)
        assert state.step == 0

    def test_n_fragments_zero_raises(self):
        with pytest.raises(ValueError):
            create_state(0)

    def test_n_fragments_negative_raises(self):
        with pytest.raises(ValueError):
            create_state(-1)

    def test_n_fragments_one_valid(self):
        state = create_state(1)
        assert state.n_fragments == 1


# ─── place_fragment ───────────────────────────────────────────────────────────

class TestPlaceFragment:
    def setup_method(self):
        self.state = create_state(5)

    def test_returns_assembly_state(self):
        new = place_fragment(self.state, 0, (0.0, 0.0))
        assert isinstance(new, AssemblyState)

    def test_original_unchanged(self):
        place_fragment(self.state, 0, (0.0, 0.0))
        assert len(self.state.placed) == 0

    def test_fragment_in_new_placed(self):
        new = place_fragment(self.state, 2, (5.0, 10.0))
        assert 2 in new.placed

    def test_position_stored(self):
        new = place_fragment(self.state, 0, (3.0, 7.0))
        assert new.placed[0].position == (3.0, 7.0)

    def test_angle_stored(self):
        new = place_fragment(self.state, 0, (0.0, 0.0), angle=90.0)
        assert new.placed[0].angle == 90.0

    def test_scale_stored(self):
        new = place_fragment(self.state, 0, (0.0, 0.0), scale=2.0)
        assert new.placed[0].scale == 2.0

    def test_meta_kwargs_stored(self):
        new = place_fragment(self.state, 0, (0.0, 0.0), score=0.8)
        assert new.placed[0].meta["score"] == pytest.approx(0.8)

    def test_step_incremented(self):
        new = place_fragment(self.state, 0, (0.0, 0.0))
        assert new.step == self.state.step + 1

    def test_idx_out_of_range_raises(self):
        with pytest.raises(ValueError):
            place_fragment(self.state, 10, (0.0, 0.0))

    def test_negative_idx_raises(self):
        with pytest.raises(ValueError):
            place_fragment(self.state, -1, (0.0, 0.0))

    def test_duplicate_placement_raises(self):
        new = place_fragment(self.state, 0, (0.0, 0.0))
        with pytest.raises(ValueError):
            place_fragment(new, 0, (1.0, 1.0))

    def test_adjacency_entry_created(self):
        new = place_fragment(self.state, 0, (0.0, 0.0))
        assert 0 in new.adjacency


# ─── remove_fragment ──────────────────────────────────────────────────────────

class TestRemoveFragment:
    def setup_method(self):
        self.state = create_state(5)
        self.state_with = place_fragment(self.state, 2, (1.0, 2.0))

    def test_returns_assembly_state(self):
        new = remove_fragment(self.state_with, 2)
        assert isinstance(new, AssemblyState)

    def test_original_unchanged(self):
        remove_fragment(self.state_with, 2)
        assert 2 in self.state_with.placed

    def test_fragment_removed(self):
        new = remove_fragment(self.state_with, 2)
        assert 2 not in new.placed

    def test_step_incremented(self):
        new = remove_fragment(self.state_with, 2)
        assert new.step == self.state_with.step + 1

    def test_not_placed_raises_key_error(self):
        with pytest.raises(KeyError):
            remove_fragment(self.state, 0)

    def test_adjacency_cleaned_up(self):
        s1 = place_fragment(self.state, 0, (0.0, 0.0))
        s2 = place_fragment(s1, 1, (10.0, 0.0))
        s3 = add_adjacency(s2, 0, 1)
        s4 = remove_fragment(s3, 0)
        # idx 0 removed from adjacency entirely
        assert 0 not in s4.adjacency
        # idx 1's neighbors no longer include 0
        assert 0 not in s4.adjacency.get(1, set())


# ─── add_adjacency ────────────────────────────────────────────────────────────

class TestAddAdjacency:
    def setup_method(self):
        self.state = create_state(5)

    def test_returns_assembly_state(self):
        new = add_adjacency(self.state, 0, 1)
        assert isinstance(new, AssemblyState)

    def test_original_unchanged(self):
        add_adjacency(self.state, 0, 1)
        assert self.state.adjacency == {}

    def test_bilateral_edge(self):
        new = add_adjacency(self.state, 0, 1)
        assert 1 in new.adjacency.get(0, set())
        assert 0 in new.adjacency.get(1, set())

    def test_self_adjacency_raises(self):
        with pytest.raises(ValueError):
            add_adjacency(self.state, 2, 2)

    def test_multiple_edges(self):
        s1 = add_adjacency(self.state, 0, 1)
        s2 = add_adjacency(s1, 0, 2)
        assert 1 in s2.adjacency[0]
        assert 2 in s2.adjacency[0]

    def test_idempotent(self):
        s1 = add_adjacency(self.state, 0, 1)
        s2 = add_adjacency(s1, 0, 1)
        assert len(s2.adjacency[0]) == 1  # still just 1 neighbor


# ─── get_neighbors ────────────────────────────────────────────────────────────

class TestGetNeighbors:
    def test_no_neighbors_returns_empty(self):
        state = create_state(5)
        assert get_neighbors(state, 0) == []

    def test_neighbors_sorted(self):
        s1 = add_adjacency(create_state(10), 0, 5)
        s2 = add_adjacency(s1, 0, 2)
        result = get_neighbors(s2, 0)
        assert result == sorted(result)

    def test_correct_neighbors(self):
        s1 = add_adjacency(create_state(5), 0, 1)
        s2 = add_adjacency(s1, 0, 3)
        result = get_neighbors(s2, 0)
        assert set(result) == {1, 3}

    def test_returns_list(self):
        state = create_state(3)
        assert isinstance(get_neighbors(state, 0), list)


# ─── compute_coverage ─────────────────────────────────────────────────────────

class TestComputeCoverage:
    def test_no_fragments_placed_zero(self):
        state = create_state(5)
        assert compute_coverage(state) == pytest.approx(0.0)

    def test_all_placed_is_one(self):
        state = create_state(3)
        for i in range(3):
            state = place_fragment(state, i, (float(i), 0.0))
        assert compute_coverage(state) == pytest.approx(1.0)

    def test_partial_coverage(self):
        state = create_state(4)
        state = place_fragment(state, 0, (0.0, 0.0))
        assert compute_coverage(state) == pytest.approx(0.25)

    def test_returns_float(self):
        state = create_state(3)
        assert isinstance(compute_coverage(state), float)


# ─── is_complete ──────────────────────────────────────────────────────────────

class TestIsComplete:
    def test_empty_not_complete(self):
        state = create_state(3)
        assert is_complete(state) is False

    def test_all_placed_complete(self):
        state = create_state(2)
        state = place_fragment(state, 0, (0.0, 0.0))
        state = place_fragment(state, 1, (10.0, 0.0))
        assert is_complete(state) is True

    def test_partial_not_complete(self):
        state = create_state(3)
        state = place_fragment(state, 0, (0.0, 0.0))
        assert is_complete(state) is False


# ─── to_dict / from_dict ──────────────────────────────────────────────────────

class TestToDictFromDict:
    def _build_state(self):
        state = create_state(3)
        state = place_fragment(state, 0, (1.0, 2.0), angle=45.0, scale=1.5, score=0.9)
        state = place_fragment(state, 1, (5.0, 6.0))
        state = add_adjacency(state, 0, 1)
        return state

    def test_to_dict_returns_dict(self):
        state = self._build_state()
        d = to_dict(state)
        assert isinstance(d, dict)

    def test_to_dict_has_required_keys(self):
        state = self._build_state()
        d = to_dict(state)
        assert "n_fragments" in d
        assert "placed" in d
        assert "adjacency" in d
        assert "step" in d

    def test_roundtrip_n_fragments(self):
        state = self._build_state()
        restored = from_dict(to_dict(state))
        assert restored.n_fragments == state.n_fragments

    def test_roundtrip_step(self):
        state = self._build_state()
        restored = from_dict(to_dict(state))
        assert restored.step == state.step

    def test_roundtrip_placed_keys(self):
        state = self._build_state()
        restored = from_dict(to_dict(state))
        assert set(restored.placed.keys()) == set(state.placed.keys())

    def test_roundtrip_position(self):
        state = self._build_state()
        restored = from_dict(to_dict(state))
        assert restored.placed[0].position == pytest.approx((1.0, 2.0))

    def test_roundtrip_angle(self):
        state = self._build_state()
        restored = from_dict(to_dict(state))
        assert restored.placed[0].angle == pytest.approx(45.0)

    def test_roundtrip_scale(self):
        state = self._build_state()
        restored = from_dict(to_dict(state))
        assert restored.placed[0].scale == pytest.approx(1.5)

    def test_roundtrip_adjacency(self):
        state = self._build_state()
        restored = from_dict(to_dict(state))
        assert 1 in restored.adjacency.get(0, set())
        assert 0 in restored.adjacency.get(1, set())

    def test_empty_state_roundtrip(self):
        state = create_state(10)
        restored = from_dict(to_dict(state))
        assert restored.n_fragments == 10
        assert len(restored.placed) == 0
        assert len(restored.adjacency) == 0
