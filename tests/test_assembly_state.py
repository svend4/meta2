"""Тесты для puzzle_reconstruction/assembly/assembly_state.py."""
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


# ─── helpers ─────────────────────────────────────────────────────────────────

def _empty(n=5):
    return create_state(n)


def _filled(n=4):
    """Состояние с размещёнными фрагментами 0..n-1."""
    state = _empty(n)
    for i in range(n):
        state = place_fragment(state, i, (float(i * 10), 0.0))
    return state


# ─── PlacedFragment ───────────────────────────────────────────────────────────

class TestPlacedFragment:
    def _make(self, **kw):
        defaults = dict(idx=2, position=(10.0, 20.0))
        defaults.update(kw)
        return PlacedFragment(**defaults)

    def test_idx_stored(self):
        assert self._make().idx == 2

    def test_position_stored(self):
        p = self._make(position=(5.5, 7.3))
        assert p.position[0] == pytest.approx(5.5)
        assert p.position[1] == pytest.approx(7.3)

    def test_angle_default_zero(self):
        assert self._make().angle == pytest.approx(0.0)

    def test_scale_default_one(self):
        assert self._make().scale == pytest.approx(1.0)

    def test_meta_default_empty(self):
        assert isinstance(self._make().meta, dict)

    def test_angle_stored(self):
        p = PlacedFragment(idx=0, position=(0.0, 0.0), angle=45.0)
        assert p.angle == pytest.approx(45.0)

    def test_scale_stored(self):
        p = PlacedFragment(idx=0, position=(0.0, 0.0), scale=2.5)
        assert p.scale == pytest.approx(2.5)

    def test_meta_stored(self):
        p = PlacedFragment(idx=0, position=(0.0, 0.0), meta={"score": 0.9})
        assert p.meta["score"] == pytest.approx(0.9)

    def test_repr_contains_class(self):
        assert "PlacedFragment" in repr(self._make())

    def test_repr_contains_idx(self):
        r = repr(self._make(idx=7))
        assert "7" in r

    def test_position_is_pair(self):
        p = self._make()
        assert len(p.position) == 2


# ─── AssemblyState ────────────────────────────────────────────────────────────

class TestAssemblyState:
    def test_n_fragments_stored(self):
        s = AssemblyState(n_fragments=8)
        assert s.n_fragments == 8

    def test_placed_default_empty(self):
        s = AssemblyState(n_fragments=3)
        assert s.placed == {}

    def test_adjacency_default_empty(self):
        s = AssemblyState(n_fragments=3)
        assert s.adjacency == {}

    def test_step_default_zero(self):
        s = AssemblyState(n_fragments=3)
        assert s.step == 0

    def test_repr_contains_class(self):
        assert "AssemblyState" in repr(AssemblyState(n_fragments=5))

    def test_repr_contains_n(self):
        r = repr(AssemblyState(n_fragments=12))
        assert "12" in r

    def test_repr_contains_placed_count(self):
        s = AssemblyState(n_fragments=5)
        r = repr(s)
        assert "0" in r   # placed=0


# ─── create_state ─────────────────────────────────────────────────────────────

class TestCreateState:
    def test_returns_assembly_state(self):
        assert isinstance(create_state(5), AssemblyState)

    def test_n_fragments_set(self):
        assert create_state(7).n_fragments == 7

    def test_placed_empty(self):
        assert create_state(5).placed == {}

    def test_adjacency_empty(self):
        assert create_state(5).adjacency == {}

    def test_step_zero(self):
        assert create_state(5).step == 0

    def test_n_zero_raises(self):
        with pytest.raises(ValueError):
            create_state(0)

    def test_n_negative_raises(self):
        with pytest.raises(ValueError):
            create_state(-3)

    def test_n_one_ok(self):
        s = create_state(1)
        assert s.n_fragments == 1


# ─── place_fragment ───────────────────────────────────────────────────────────

class TestPlaceFragment:
    def test_returns_assembly_state(self):
        s = _empty()
        r = place_fragment(s, 0, (0.0, 0.0))
        assert isinstance(r, AssemblyState)

    def test_original_unchanged(self):
        s = _empty()
        place_fragment(s, 0, (0.0, 0.0))
        assert 0 not in s.placed

    def test_idx_in_placed(self):
        s = place_fragment(_empty(), 2, (5.0, 3.0))
        assert 2 in s.placed

    def test_position_stored(self):
        s = place_fragment(_empty(), 0, (7.5, 12.0))
        assert s.placed[0].position == pytest.approx((7.5, 12.0))

    def test_angle_stored(self):
        s = place_fragment(_empty(), 0, (0.0, 0.0), angle=90.0)
        assert s.placed[0].angle == pytest.approx(90.0)

    def test_scale_stored(self):
        s = place_fragment(_empty(), 0, (0.0, 0.0), scale=1.5)
        assert s.placed[0].scale == pytest.approx(1.5)

    def test_meta_kwargs_stored(self):
        s = place_fragment(_empty(), 0, (0.0, 0.0), score=0.8, side=2)
        assert s.placed[0].meta.get("score") == pytest.approx(0.8)
        assert s.placed[0].meta.get("side") == 2

    def test_step_incremented(self):
        s = _empty()
        s2 = place_fragment(s, 0, (0.0, 0.0))
        assert s2.step == s.step + 1

    def test_idx_out_of_range_low_raises(self):
        with pytest.raises(ValueError):
            place_fragment(_empty(5), -1, (0.0, 0.0))

    def test_idx_out_of_range_high_raises(self):
        with pytest.raises(ValueError):
            place_fragment(_empty(5), 5, (0.0, 0.0))

    def test_already_placed_raises(self):
        s = place_fragment(_empty(), 0, (0.0, 0.0))
        with pytest.raises(ValueError):
            place_fragment(s, 0, (1.0, 1.0))

    def test_adjacency_entry_created(self):
        s = place_fragment(_empty(), 3, (0.0, 0.0))
        assert 3 in s.adjacency

    def test_multiple_fragments(self):
        s = _empty(5)
        for i in range(5):
            s = place_fragment(s, i, (float(i), 0.0))
        assert len(s.placed) == 5


# ─── remove_fragment ──────────────────────────────────────────────────────────

class TestRemoveFragment:
    def test_returns_assembly_state(self):
        s = place_fragment(_empty(), 0, (0.0, 0.0))
        assert isinstance(remove_fragment(s, 0), AssemblyState)

    def test_original_unchanged(self):
        s = place_fragment(_empty(), 0, (0.0, 0.0))
        remove_fragment(s, 0)
        assert 0 in s.placed

    def test_idx_removed_from_placed(self):
        s = place_fragment(_empty(), 1, (0.0, 0.0))
        s2 = remove_fragment(s, 1)
        assert 1 not in s2.placed

    def test_step_incremented(self):
        s = place_fragment(_empty(), 0, (0.0, 0.0))
        s2 = remove_fragment(s, 0)
        assert s2.step == s.step + 1

    def test_not_placed_raises(self):
        with pytest.raises(KeyError):
            remove_fragment(_empty(), 0)

    def test_adjacency_entry_removed(self):
        s = place_fragment(_empty(), 0, (0.0, 0.0))
        s = place_fragment(s, 1, (10.0, 0.0))
        s = add_adjacency(s, 0, 1)
        s2 = remove_fragment(s, 0)
        assert 0 not in s2.adjacency

    def test_removed_from_neighbors_adjacency(self):
        s = place_fragment(_empty(), 0, (0.0, 0.0))
        s = place_fragment(s, 1, (10.0, 0.0))
        s = add_adjacency(s, 0, 1)
        s2 = remove_fragment(s, 0)
        neighbors = s2.adjacency.get(1, set())
        assert 0 not in neighbors

    def test_other_fragment_still_placed(self):
        s = place_fragment(_empty(), 0, (0.0, 0.0))
        s = place_fragment(s, 1, (10.0, 0.0))
        s2 = remove_fragment(s, 0)
        assert 1 in s2.placed


# ─── add_adjacency ────────────────────────────────────────────────────────────

class TestAddAdjacency:
    def test_returns_assembly_state(self):
        s = _empty()
        assert isinstance(add_adjacency(s, 0, 1), AssemblyState)

    def test_original_unchanged(self):
        s = _empty()
        add_adjacency(s, 0, 1)
        assert 0 not in s.adjacency or 1 not in s.adjacency.get(0, set())

    def test_idx1_in_idx2_neighbors(self):
        s = add_adjacency(_empty(), 0, 1)
        assert 1 in s.adjacency.get(0, set())

    def test_idx2_in_idx1_neighbors(self):
        s = add_adjacency(_empty(), 0, 1)
        assert 0 in s.adjacency.get(1, set())

    def test_self_adjacency_raises(self):
        with pytest.raises(ValueError):
            add_adjacency(_empty(), 3, 3)

    def test_multiple_calls_accumulate(self):
        s = _empty()
        s = add_adjacency(s, 0, 1)
        s = add_adjacency(s, 0, 2)
        assert 1 in s.adjacency[0]
        assert 2 in s.adjacency[0]

    def test_bidirectional_multiple(self):
        s = add_adjacency(_empty(), 1, 3)
        assert 3 in s.adjacency.get(1, set())
        assert 1 in s.adjacency.get(3, set())


# ─── get_neighbors ────────────────────────────────────────────────────────────

class TestGetNeighbors:
    def test_returns_list(self):
        assert isinstance(get_neighbors(_empty(), 0), list)

    def test_no_neighbors_empty(self):
        assert get_neighbors(_empty(), 0) == []

    def test_unknown_idx_empty(self):
        assert get_neighbors(_empty(), 99) == []

    def test_one_neighbor(self):
        s = add_adjacency(_empty(), 0, 1)
        assert get_neighbors(s, 0) == [1]

    def test_sorted_output(self):
        s = _empty()
        s = add_adjacency(s, 0, 3)
        s = add_adjacency(s, 0, 1)
        s = add_adjacency(s, 0, 2)
        assert get_neighbors(s, 0) == [1, 2, 3]

    def test_both_directions(self):
        s = add_adjacency(_empty(), 2, 4)
        assert 4 in get_neighbors(s, 2)
        assert 2 in get_neighbors(s, 4)


# ─── compute_coverage ─────────────────────────────────────────────────────────

class TestComputeCoverage:
    def test_empty_is_zero(self):
        assert compute_coverage(_empty(5)) == pytest.approx(0.0)

    def test_complete_is_one(self):
        assert compute_coverage(_filled(4)) == pytest.approx(1.0)

    def test_half_is_half(self):
        s = _empty(4)
        s = place_fragment(s, 0, (0.0, 0.0))
        s = place_fragment(s, 1, (0.0, 0.0))
        assert compute_coverage(s) == pytest.approx(0.5)

    def test_in_range(self):
        v = compute_coverage(_empty(5))
        assert 0.0 <= v <= 1.0

    def test_one_of_five(self):
        s = place_fragment(_empty(5), 0, (0.0, 0.0))
        assert compute_coverage(s) == pytest.approx(0.2)

    def test_returns_float(self):
        assert isinstance(compute_coverage(_empty()), float)


# ─── is_complete ──────────────────────────────────────────────────────────────

class TestIsComplete:
    def test_empty_false(self):
        assert is_complete(_empty(3)) is False

    def test_all_placed_true(self):
        assert is_complete(_filled(3)) is True

    def test_partial_false(self):
        s = place_fragment(_empty(3), 0, (0.0, 0.0))
        assert is_complete(s) is False

    def test_n_minus_1_false(self):
        s = _empty(3)
        s = place_fragment(s, 0, (0.0, 0.0))
        s = place_fragment(s, 1, (0.0, 0.0))
        assert is_complete(s) is False

    def test_returns_bool(self):
        assert isinstance(is_complete(_empty()), bool)


# ─── to_dict ──────────────────────────────────────────────────────────────────

class TestToDict:
    def test_has_n_fragments(self):
        d = to_dict(_empty(5))
        assert d["n_fragments"] == 5

    def test_has_placed(self):
        assert "placed" in to_dict(_empty())

    def test_has_adjacency(self):
        assert "adjacency" in to_dict(_empty())

    def test_has_step(self):
        assert "step" in to_dict(_empty())

    def test_placed_is_dict(self):
        assert isinstance(to_dict(_empty())["placed"], dict)

    def test_step_value(self):
        s = place_fragment(_empty(), 0, (0.0, 0.0))
        assert to_dict(s)["step"] == s.step

    def test_placed_fragment_has_position(self):
        s = place_fragment(_empty(), 0, (3.5, 7.2))
        d = to_dict(s)
        pf = list(d["placed"].values())[0]
        assert pf["position"][0] == pytest.approx(3.5)

    def test_returns_dict(self):
        assert isinstance(to_dict(_empty()), dict)


# ─── from_dict ────────────────────────────────────────────────────────────────

class TestFromDict:
    def _roundtrip(self, state):
        return from_dict(to_dict(state))

    def test_returns_assembly_state(self):
        assert isinstance(self._roundtrip(_empty()), AssemblyState)

    def test_n_fragments_preserved(self):
        assert self._roundtrip(_empty(7)).n_fragments == 7

    def test_step_preserved(self):
        s = place_fragment(_empty(), 0, (0.0, 0.0))
        assert self._roundtrip(s).step == s.step

    def test_placed_keys_preserved(self):
        s = _filled(3)
        r = self._roundtrip(s)
        assert set(r.placed.keys()) == {0, 1, 2}

    def test_position_preserved(self):
        s = place_fragment(_empty(), 2, (8.0, 15.0))
        r = self._roundtrip(s)
        assert r.placed[2].position == pytest.approx((8.0, 15.0))

    def test_adjacency_preserved(self):
        s = add_adjacency(_empty(), 0, 1)
        r = self._roundtrip(s)
        assert 1 in r.adjacency.get(0, set())
        assert 0 in r.adjacency.get(1, set())

    def test_empty_state_roundtrip(self):
        s = _empty(4)
        r = self._roundtrip(s)
        assert r.placed == {}
        assert r.n_fragments == 4

    def test_full_state_roundtrip(self):
        s = _filled(4)
        s = add_adjacency(s, 0, 1)
        r = self._roundtrip(s)
        assert is_complete(r)
        assert 1 in r.adjacency.get(0, set())
