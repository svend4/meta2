"""Extra tests for puzzle_reconstruction/assembly/assembly_state.py"""
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


# ─── TestPlacedFragmentExtra ──────────────────────────────────────────────────

class TestPlacedFragmentExtra:
    def test_idx_zero_valid(self):
        pf = PlacedFragment(idx=0, position=(0.0, 0.0))
        assert pf.idx == 0

    def test_large_idx(self):
        pf = PlacedFragment(idx=9999, position=(100.0, 200.0))
        assert pf.idx == 9999

    def test_multiple_meta_keys(self):
        pf = PlacedFragment(idx=0, position=(0.0, 0.0),
                             meta={"score": 0.9, "method": "ncc", "rank": 1})
        assert pf.meta["method"] == "ncc"
        assert pf.meta["rank"] == 1

    def test_negative_angle_valid(self):
        pf = PlacedFragment(idx=0, position=(0.0, 0.0), angle=-45.0)
        assert pf.angle == pytest.approx(-45.0)

    def test_large_angle_valid(self):
        pf = PlacedFragment(idx=0, position=(0.0, 0.0), angle=360.0)
        assert pf.angle == pytest.approx(360.0)

    def test_small_scale_valid(self):
        pf = PlacedFragment(idx=0, position=(0.0, 0.0), scale=0.1)
        assert pf.scale == pytest.approx(0.1)

    def test_repr_is_string(self):
        pf = PlacedFragment(idx=5, position=(1.0, 2.0))
        assert isinstance(repr(pf), str)

    def test_position_negative_coords_valid(self):
        pf = PlacedFragment(idx=0, position=(-10.0, -20.0))
        assert pf.position == (-10.0, -20.0)


# ─── TestAssemblyStateExtra ───────────────────────────────────────────────────

class TestAssemblyStateExtra:
    def test_n_fragments_1(self):
        s = AssemblyState(n_fragments=1)
        assert s.n_fragments == 1

    def test_n_fragments_100(self):
        s = AssemblyState(n_fragments=100)
        assert s.n_fragments == 100

    def test_placed_is_dict(self):
        s = AssemblyState(n_fragments=5)
        assert isinstance(s.placed, dict)

    def test_adjacency_is_dict(self):
        s = AssemblyState(n_fragments=5)
        assert isinstance(s.adjacency, dict)

    def test_step_is_int(self):
        s = AssemblyState(n_fragments=5)
        assert isinstance(s.step, int)


# ─── TestCreateStateExtra ────────────────────────────────────────────────────

class TestCreateStateExtra:
    def test_large_n(self):
        s = create_state(100)
        assert s.n_fragments == 100

    def test_n_2(self):
        s = create_state(2)
        assert s.n_fragments == 2

    def test_creates_fresh_state(self):
        s1 = create_state(5)
        s2 = create_state(5)
        # Mutating one should not affect the other
        s1.placed[0] = None
        assert 0 not in s2.placed


# ─── TestPlaceFragmentExtra ───────────────────────────────────────────────────

class TestPlaceFragmentExtra:
    def test_place_all_n_fragments(self):
        n = 5
        state = create_state(n)
        for i in range(n):
            state = place_fragment(state, i, (float(i), 0.0))
        assert len(state.placed) == n

    def test_step_increments_each_time(self):
        state = create_state(5)
        for i in range(5):
            state = place_fragment(state, i, (float(i), 0.0))
        assert state.step == 5

    def test_position_large_coords(self):
        state = create_state(5)
        new = place_fragment(state, 0, (1000.0, 2000.0))
        assert new.placed[0].position == (1000.0, 2000.0)

    def test_angle_neg_stored(self):
        state = create_state(5)
        new = place_fragment(state, 0, (0.0, 0.0), angle=-30.0)
        assert new.placed[0].angle == pytest.approx(-30.0)

    def test_scale_half(self):
        state = create_state(5)
        new = place_fragment(state, 0, (0.0, 0.0), scale=0.5)
        assert new.placed[0].scale == pytest.approx(0.5)

    def test_multiple_meta(self):
        state = create_state(5)
        new = place_fragment(state, 0, (0.0, 0.0), score=0.9, rank=1)
        assert new.placed[0].meta["score"] == pytest.approx(0.9)
        assert new.placed[0].meta["rank"] == 1

    def test_place_fragment_2_and_3(self):
        state = create_state(5)
        state = place_fragment(state, 2, (5.0, 5.0))
        state = place_fragment(state, 3, (10.0, 5.0))
        assert 2 in state.placed
        assert 3 in state.placed


# ─── TestRemoveFragmentExtra ──────────────────────────────────────────────────

class TestRemoveFragmentExtra:
    def test_remove_then_replace(self):
        state = create_state(5)
        state = place_fragment(state, 0, (1.0, 2.0))
        state = remove_fragment(state, 0)
        state = place_fragment(state, 0, (3.0, 4.0))
        assert state.placed[0].position == (3.0, 4.0)

    def test_remove_last_placed(self):
        state = create_state(3)
        state = place_fragment(state, 0, (0.0, 0.0))
        state = place_fragment(state, 1, (1.0, 0.0))
        state = remove_fragment(state, 1)
        assert 1 not in state.placed
        assert 0 in state.placed

    def test_step_increments_on_remove(self):
        state = create_state(5)
        state = place_fragment(state, 0, (0.0, 0.0))
        prev_step = state.step
        state = remove_fragment(state, 0)
        assert state.step == prev_step + 1


# ─── TestAddAdjacencyExtra ────────────────────────────────────────────────────

class TestAddAdjacencyExtra:
    def test_chain_0_1_2(self):
        state = create_state(5)
        s1 = add_adjacency(state, 0, 1)
        s2 = add_adjacency(s1, 1, 2)
        assert 1 in s2.adjacency[0]
        assert 2 in s2.adjacency[1]
        assert 1 in s2.adjacency[2]

    def test_three_nodes_complete_graph(self):
        state = create_state(5)
        s = state
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            s = add_adjacency(s, a, b)
        assert 1 in s.adjacency[0]
        assert 2 in s.adjacency[0]
        assert 0 in s.adjacency[1]
        assert 2 in s.adjacency[1]

    def test_bidirectional_confirmed(self):
        state = create_state(10)
        s = add_adjacency(state, 3, 7)
        assert 7 in s.adjacency[3]
        assert 3 in s.adjacency[7]

    def test_add_then_remove_clears_edge(self):
        state = create_state(5)
        s1 = place_fragment(state, 0, (0.0, 0.0))
        s2 = place_fragment(s1, 1, (1.0, 0.0))
        s3 = add_adjacency(s2, 0, 1)
        s4 = remove_fragment(s3, 0)
        assert 0 not in s4.adjacency


# ─── TestGetNeighborsExtra ────────────────────────────────────────────────────

class TestGetNeighborsExtra:
    def test_single_neighbor(self):
        s = add_adjacency(create_state(5), 0, 4)
        assert get_neighbors(s, 0) == [4]

    def test_multiple_neighbors_sorted(self):
        s = create_state(10)
        for n in [7, 2, 5]:
            s = add_adjacency(s, 0, n)
        result = get_neighbors(s, 0)
        assert result == [2, 5, 7]

    def test_no_adjacency_returns_empty_list(self):
        s = create_state(5)
        assert get_neighbors(s, 3) == []

    def test_neighbors_are_ints(self):
        s = add_adjacency(create_state(5), 0, 1)
        result = get_neighbors(s, 0)
        for n in result:
            assert isinstance(n, int)


# ─── TestComputeCoverageExtra ─────────────────────────────────────────────────

class TestComputeCoverageExtra:
    def test_half_coverage(self):
        state = create_state(4)
        state = place_fragment(state, 0, (0.0, 0.0))
        state = place_fragment(state, 1, (1.0, 0.0))
        assert compute_coverage(state) == pytest.approx(0.5)

    def test_three_of_five(self):
        state = create_state(5)
        for i in range(3):
            state = place_fragment(state, i, (float(i), 0.0))
        assert compute_coverage(state) == pytest.approx(0.6)

    def test_single_of_one_is_one(self):
        state = create_state(1)
        state = place_fragment(state, 0, (0.0, 0.0))
        assert compute_coverage(state) == pytest.approx(1.0)

    def test_coverage_in_range(self):
        state = create_state(5)
        c = compute_coverage(state)
        assert 0.0 <= c <= 1.0


# ─── TestIsCompleteExtra ──────────────────────────────────────────────────────

class TestIsCompleteExtra:
    def test_n_1_complete_after_one(self):
        state = create_state(1)
        state = place_fragment(state, 0, (0.0, 0.0))
        assert is_complete(state) is True

    def test_returns_bool(self):
        state = create_state(3)
        assert isinstance(is_complete(state), bool)

    def test_n_10_all_placed(self):
        state = create_state(10)
        for i in range(10):
            state = place_fragment(state, i, (float(i), 0.0))
        assert is_complete(state) is True

    def test_n_10_nine_placed_not_complete(self):
        state = create_state(10)
        for i in range(9):
            state = place_fragment(state, i, (float(i), 0.0))
        assert is_complete(state) is False


# ─── TestToDictFromDictExtra ──────────────────────────────────────────────────

class TestToDictFromDictExtra:
    def _build(self):
        s = create_state(4)
        s = place_fragment(s, 0, (1.0, 2.0), angle=30.0)
        s = place_fragment(s, 1, (3.0, 4.0))
        s = place_fragment(s, 2, (5.0, 6.0), score=0.8)
        s = add_adjacency(s, 0, 1)
        s = add_adjacency(s, 1, 2)
        return s

    def test_meta_preserved_roundtrip(self):
        s = self._build()
        r = from_dict(to_dict(s))
        assert r.placed[2].meta.get("score") == pytest.approx(0.8)

    def test_adjacency_complex_roundtrip(self):
        s = self._build()
        r = from_dict(to_dict(s))
        assert 1 in r.adjacency[0]
        assert 2 in r.adjacency[1]

    def test_to_dict_is_serializable(self):
        import json
        s = self._build()
        d = to_dict(s)
        # Should be JSON-serializable (no sets, no numpy types)
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_multiple_roundtrips_stable(self):
        s = self._build()
        s2 = from_dict(to_dict(s))
        s3 = from_dict(to_dict(s2))
        assert s3.n_fragments == s.n_fragments
        assert s3.step == s.step
