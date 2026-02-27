"""
Integration tests for puzzle_reconstruction.assembly.assembly_state module.

~50 tests across 5 classes covering:
- create_state: initial invariants
- place_fragment / remove_fragment: placement lifecycle
- add_adjacency / get_neighbors: adjacency graph
- compute_coverage / is_complete: state metrics
- to_dict / from_dict: serialization roundtrip
"""
from __future__ import annotations

import json
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


# ─── TestCreateState ──────────────────────────────────────────────────────────

class TestCreateState:
    """Tests for create_state()."""

    def test_returns_assembly_state(self):
        s = create_state(4)
        assert isinstance(s, AssemblyState)

    def test_n_fragments_stored(self):
        s = create_state(7)
        assert s.n_fragments == 7

    def test_placed_empty(self):
        s = create_state(3)
        assert len(s.placed) == 0

    def test_adjacency_empty(self):
        s = create_state(3)
        assert len(s.adjacency) == 0

    def test_step_zero(self):
        s = create_state(5)
        assert s.step == 0

    def test_n_fragments_one(self):
        s = create_state(1)
        assert s.n_fragments == 1

    def test_large_n_fragments(self):
        s = create_state(100)
        assert s.n_fragments == 100

    def test_invalid_zero(self):
        with pytest.raises(ValueError):
            create_state(0)

    def test_invalid_negative(self):
        with pytest.raises(ValueError):
            create_state(-1)

    def test_coverage_initially_zero(self):
        s = create_state(5)
        assert compute_coverage(s) == 0.0


# ─── TestPlaceRemoveFragment ──────────────────────────────────────────────────

class TestPlaceRemoveFragment:
    """Tests for place_fragment() and remove_fragment()."""

    def test_place_returns_new_state(self):
        s = create_state(4)
        s2 = place_fragment(s, 0, (10.0, 20.0))
        assert s2 is not s

    def test_place_fragment_in_placed(self):
        s = create_state(4)
        s2 = place_fragment(s, 0, (0.0, 0.0))
        assert 0 in s2.placed

    def test_place_stores_position(self):
        s = create_state(4)
        s2 = place_fragment(s, 1, (5.0, 7.0))
        pf = s2.placed[1]
        assert pf.position == (5.0, 7.0)

    def test_place_stores_angle(self):
        s = create_state(4)
        s2 = place_fragment(s, 0, (0.0, 0.0), angle=45.0)
        assert s2.placed[0].angle == 45.0

    def test_place_stores_scale(self):
        s = create_state(4)
        s2 = place_fragment(s, 0, (0.0, 0.0), scale=2.0)
        assert s2.placed[0].scale == 2.0

    def test_place_increments_step(self):
        s = create_state(4)
        s2 = place_fragment(s, 0, (0.0, 0.0))
        assert s2.step == s.step + 1

    def test_original_state_unchanged(self):
        s = create_state(4)
        place_fragment(s, 0, (0.0, 0.0))
        assert len(s.placed) == 0

    def test_place_out_of_range_raises(self):
        s = create_state(4)
        with pytest.raises(ValueError):
            place_fragment(s, 10, (0.0, 0.0))

    def test_place_negative_index_raises(self):
        s = create_state(4)
        with pytest.raises(ValueError):
            place_fragment(s, -1, (0.0, 0.0))

    def test_place_duplicate_raises(self):
        s = create_state(4)
        s2 = place_fragment(s, 0, (0.0, 0.0))
        with pytest.raises(ValueError):
            place_fragment(s2, 0, (1.0, 1.0))

    def test_place_meta_stored(self):
        s = create_state(4)
        s2 = place_fragment(s, 0, (0.0, 0.0), score=0.9, side="right")
        assert s2.placed[0].meta["score"] == pytest.approx(0.9)
        assert s2.placed[0].meta["side"] == "right"

    def test_remove_fragment_removes(self):
        s = create_state(4)
        s2 = place_fragment(s, 0, (0.0, 0.0))
        s3 = remove_fragment(s2, 0)
        assert 0 not in s3.placed

    def test_remove_increments_step(self):
        s = create_state(4)
        s2 = place_fragment(s, 0, (0.0, 0.0))
        s3 = remove_fragment(s2, 0)
        assert s3.step == s2.step + 1

    def test_remove_nonexistent_raises(self):
        s = create_state(4)
        with pytest.raises(KeyError):
            remove_fragment(s, 0)

    def test_remove_cleans_adjacency(self):
        s = create_state(4)
        s2 = place_fragment(s, 0, (0.0, 0.0))
        s3 = place_fragment(s2, 1, (10.0, 0.0))
        s4 = add_adjacency(s3, 0, 1)
        s5 = remove_fragment(s4, 0)
        assert 0 not in s5.adjacency
        assert 0 not in s5.adjacency.get(1, set())


# ─── TestAdjacency ────────────────────────────────────────────────────────────

class TestAdjacency:
    """Tests for add_adjacency() and get_neighbors()."""

    def test_add_adjacency_returns_new_state(self):
        s = create_state(4)
        s2 = add_adjacency(s, 0, 1)
        assert s2 is not s

    def test_adjacency_symmetric(self):
        s = create_state(4)
        s2 = add_adjacency(s, 0, 1)
        assert 1 in s2.adjacency.get(0, set())
        assert 0 in s2.adjacency.get(1, set())

    def test_get_neighbors_empty(self):
        s = create_state(4)
        assert get_neighbors(s, 0) == []

    def test_get_neighbors_after_add(self):
        s = create_state(4)
        s2 = add_adjacency(s, 0, 2)
        assert 2 in get_neighbors(s2, 0)

    def test_get_neighbors_sorted(self):
        s = create_state(6)
        s2 = add_adjacency(s, 0, 3)
        s3 = add_adjacency(s2, 0, 1)
        neighbors = get_neighbors(s3, 0)
        assert neighbors == sorted(neighbors)

    def test_self_adjacency_raises(self):
        s = create_state(4)
        with pytest.raises(ValueError):
            add_adjacency(s, 2, 2)

    def test_multiple_adjacency_edges(self):
        s = create_state(5)
        s2 = add_adjacency(s, 0, 1)
        s3 = add_adjacency(s2, 0, 2)
        s4 = add_adjacency(s3, 0, 3)
        assert len(get_neighbors(s4, 0)) == 3

    def test_original_unchanged_after_add(self):
        s = create_state(4)
        add_adjacency(s, 0, 1)
        assert 1 not in s.adjacency.get(0, set())


# ─── TestCoverageAndComplete ──────────────────────────────────────────────────

class TestCoverageAndComplete:
    """Tests for compute_coverage() and is_complete()."""

    def test_coverage_zero_initially(self):
        s = create_state(5)
        assert compute_coverage(s) == pytest.approx(0.0)

    def test_coverage_partial(self):
        s = create_state(4)
        s2 = place_fragment(s, 0, (0.0, 0.0))
        assert compute_coverage(s2) == pytest.approx(0.25)

    def test_coverage_full(self):
        s = create_state(3)
        s2 = place_fragment(s, 0, (0.0, 0.0))
        s3 = place_fragment(s2, 1, (10.0, 0.0))
        s4 = place_fragment(s3, 2, (20.0, 0.0))
        assert compute_coverage(s4) == pytest.approx(1.0)

    def test_is_complete_false_initially(self):
        s = create_state(3)
        assert not is_complete(s)

    def test_is_complete_true_when_all_placed(self):
        s = create_state(2)
        s2 = place_fragment(s, 0, (0.0, 0.0))
        s3 = place_fragment(s2, 1, (10.0, 0.0))
        assert is_complete(s3)

    def test_is_complete_false_after_remove(self):
        s = create_state(2)
        s2 = place_fragment(s, 0, (0.0, 0.0))
        s3 = place_fragment(s2, 1, (10.0, 0.0))
        s4 = remove_fragment(s3, 1)
        assert not is_complete(s4)

    def test_coverage_in_unit_interval(self):
        s = create_state(5)
        s2 = place_fragment(s, 0, (0.0, 0.0))
        c = compute_coverage(s2)
        assert 0.0 <= c <= 1.0


# ─── TestSerialisation ────────────────────────────────────────────────────────

class TestSerialisation:
    """Tests for to_dict() and from_dict() roundtrip."""

    def _build_state(self):
        s = create_state(3)
        s2 = place_fragment(s, 0, (0.0, 0.0), angle=10.0, score=0.8)
        s3 = place_fragment(s2, 1, (50.0, 0.0))
        s4 = add_adjacency(s3, 0, 1)
        return s4

    def test_to_dict_returns_dict(self):
        d = to_dict(self._build_state())
        assert isinstance(d, dict)

    def test_to_dict_has_required_keys(self):
        d = to_dict(self._build_state())
        assert "n_fragments" in d
        assert "placed" in d
        assert "adjacency" in d
        assert "step" in d

    def test_from_dict_returns_assembly_state(self):
        s = from_dict(to_dict(self._build_state()))
        assert isinstance(s, AssemblyState)

    def test_roundtrip_n_fragments(self):
        s = self._build_state()
        s2 = from_dict(to_dict(s))
        assert s2.n_fragments == s.n_fragments

    def test_roundtrip_placed_count(self):
        s = self._build_state()
        s2 = from_dict(to_dict(s))
        assert len(s2.placed) == len(s.placed)

    def test_roundtrip_position(self):
        s = self._build_state()
        s2 = from_dict(to_dict(s))
        assert s2.placed[0].position == pytest.approx((0.0, 0.0))

    def test_roundtrip_angle(self):
        s = self._build_state()
        s2 = from_dict(to_dict(s))
        assert s2.placed[0].angle == pytest.approx(10.0)

    def test_roundtrip_adjacency(self):
        s = self._build_state()
        s2 = from_dict(to_dict(s))
        assert 1 in s2.adjacency.get(0, set())

    def test_dict_json_serialisable(self):
        d = to_dict(self._build_state())
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_roundtrip_step(self):
        s = self._build_state()
        s2 = from_dict(to_dict(s))
        assert s2.step == s.step
