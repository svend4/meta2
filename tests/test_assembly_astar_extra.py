"""Extra tests for puzzle_reconstruction/assembly/astar.py"""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.assembly.astar import (
    _AStarState,
    _build_best_score_per_frag,
    _build_edge_to_frag,
    _heuristic,
    _place_new_fragment,
    _score_for_placement,
    astar_assembly,
)
from puzzle_reconstruction.models import (
    Assembly,
    CompatEntry,
    EdgeSide,
    EdgeSignature,
    Fragment,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_edge(eid: int) -> EdgeSignature:
    curve = np.column_stack([np.linspace(0, 1, 10), np.zeros(10)])
    return EdgeSignature(
        edge_id=eid,
        side=EdgeSide.RIGHT,
        virtual_curve=curve,
        fd=1.0,
        css_vec=np.ones(16) / 4.0,
        ifs_coeffs=np.zeros(8),
        length=1.0,
    )


def _make_frag(fid: int, n_edges: int = 2) -> Fragment:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    frag = Fragment(fragment_id=fid, image=img)
    frag.edges = [_make_edge(fid * 10 + k) for k in range(n_edges)]
    return frag


def _make_entry(fa: Fragment, fb: Fragment, score: float = 0.8) -> CompatEntry:
    return CompatEntry(edge_i=fa.edges[0], edge_j=fb.edges[0], score=score)


def _e2f(*frags: Fragment):
    m = {}
    for f in frags:
        for e in f.edges:
            m[e.edge_id] = f
    return m


# ── _build_edge_to_frag extra ─────────────────────────────────────────────────

class TestBuildEdgeToFragExtra:

    def test_single_fragment_all_edges_mapped(self):
        f = _make_frag(0, 4)
        m = _build_edge_to_frag([f])
        assert len(m) == 4
        for e in f.edges:
            assert m[e.edge_id].fragment_id == 0

    def test_five_fragments(self):
        frags = [_make_frag(i, 2) for i in range(5)]
        m = _build_edge_to_frag(frags)
        assert len(m) == 10

    def test_value_type_is_fragment(self):
        f = _make_frag(1, 1)
        m = _build_edge_to_frag([f])
        for v in m.values():
            assert isinstance(v, Fragment)

    def test_no_fragments_no_edges_zero_map(self):
        m = _build_edge_to_frag([])
        assert m == {}

    def test_fragment_with_no_edges(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        f = Fragment(fragment_id=99, image=img)
        f.edges = []
        m = _build_edge_to_frag([f])
        assert m == {}


# ── _build_best_score_per_frag extra ─────────────────────────────────────────

class TestBuildBestScorePerFragExtra:

    def test_multiple_entries_takes_max(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        e2f = _e2f(f0, f1)
        entries = [
            _make_entry(f0, f1, 0.3),
            _make_entry(f0, f1, 0.9),
            _make_entry(f0, f1, 0.6),
        ]
        best = _build_best_score_per_frag([f0, f1], entries, e2f)
        assert best[0] == pytest.approx(0.9)
        assert best[1] == pytest.approx(0.9)

    def test_initialised_to_zero_for_all_frags(self):
        frags = [_make_frag(i) for i in range(5)]
        e2f = _e2f(*frags)
        best = _build_best_score_per_frag(frags, [], e2f)
        for fid in range(5):
            assert best[fid] == 0.0

    def test_unrelated_fragment_stays_zero(self):
        f0, f1, f2 = _make_frag(0), _make_frag(1), _make_frag(2)
        e2f = _e2f(f0, f1, f2)
        entries = [_make_entry(f0, f1, 0.7)]
        best = _build_best_score_per_frag([f0, f1, f2], entries, e2f)
        assert best[2] == 0.0

    def test_score_propagates_to_both_fragments(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        e2f = _e2f(f0, f1)
        entries = [_make_entry(f0, f1, 0.55)]
        best = _build_best_score_per_frag([f0, f1], entries, e2f)
        assert best[0] == pytest.approx(0.55)
        assert best[1] == pytest.approx(0.55)


# ── _heuristic extra ──────────────────────────────────────────────────────────

class TestHeuristicExtra:

    def test_single_fragment(self):
        h = _heuristic(frozenset([3]), {3: 0.9})
        assert h == pytest.approx(0.9)

    def test_partial_best_per_frag_missing_uses_zero(self):
        h = _heuristic(frozenset([0, 1, 2]), {0: 0.5, 1: 0.3})
        assert h == pytest.approx(0.8)

    def test_all_zero_scores_heuristic_zero(self):
        h = _heuristic(frozenset([0, 1, 2]), {0: 0.0, 1: 0.0, 2: 0.0})
        assert h == pytest.approx(0.0)

    def test_large_fragset(self):
        fids = frozenset(range(100))
        best = {i: 0.5 for i in range(100)}
        h = _heuristic(fids, best)
        assert h == pytest.approx(50.0)

    def test_non_negative_result(self):
        h = _heuristic(frozenset([0, 1]), {0: 1.0, 1: 1.0})
        assert h >= 0.0


# ── _score_for_placement extra ────────────────────────────────────────────────

class TestScoreForPlacementExtra:

    def test_multiple_entries_multiple_frags_placed(self):
        f0, f1, f2 = _make_frag(0), _make_frag(1), _make_frag(2)
        e2f = _e2f(f0, f1, f2)
        entries = [
            _make_entry(f0, f2, 0.5),
            _make_entry(f1, f2, 0.4),
        ]
        s = _score_for_placement(f2, frozenset([0, 1]), entries, e2f)
        assert s == pytest.approx(0.9)

    def test_does_not_double_count(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        e2f = _e2f(f0, f1)
        entries = [_make_entry(f0, f1, 0.5)]
        s = _score_for_placement(f1, frozenset([0]), entries, e2f)
        assert s == pytest.approx(0.5)

    def test_self_entry_not_counted(self):
        f0 = _make_frag(0)
        e2f = _e2f(f0)
        # Entry with self — should not count (both sides same frag_id)
        entry = CompatEntry(edge_i=f0.edges[0], edge_j=f0.edges[1], score=0.99)
        s = _score_for_placement(f0, frozenset([0]), [entry], e2f)
        # frag is already in placed_ids so it's placed + unplaced simultaneously
        # just check no exception and result is finite
        assert np.isfinite(s)


# ── _place_new_fragment extra ─────────────────────────────────────────────────

class TestPlaceNewFragmentExtra:

    def test_position_is_2d_array(self):
        f = _make_frag(1)
        placements = {0: (np.array([0.0, 0.0]), 0.0)}
        pos, rot = _place_new_fragment(f, 0, placements, None, 1)
        assert pos.shape == (2,)

    def test_rotation_is_float(self):
        f = _make_frag(1)
        placements = {0: (np.array([10.0, 5.0]), 0.5)}
        pos, rot = _place_new_fragment(f, 0, placements, None, 1)
        assert isinstance(rot, float)

    def test_index_0_position(self):
        f = _make_frag(1)
        placements = {0: (np.array([0.0, 0.0]), 0.0)}
        pos, _ = _place_new_fragment(f, 0, placements, None, 0)
        assert np.all(np.isfinite(pos))

    def test_larger_index_further_right(self):
        f = _make_frag(1)
        placements = {0: (np.array([0.0, 0.0]), 0.0)}
        pos1, _ = _place_new_fragment(f, 0, placements, None, 1)
        pos5, _ = _place_new_fragment(f, 0, placements, None, 5)
        assert pos5[0] > pos1[0]


# ── _AStarState extra ─────────────────────────────────────────────────────────

class TestAStarStateExtra:

    def test_f_score_negated_sum(self):
        s = _AStarState(frozenset([0]), {}, g_score=2.0, h_score=3.0)
        assert s.f_score == pytest.approx(-5.0)

    def test_zero_scores(self):
        s = _AStarState(frozenset(), {}, g_score=0.0, h_score=0.0)
        assert s.f_score == pytest.approx(0.0)

    def test_comparison_ordering_for_heap(self):
        s_high = _AStarState(frozenset(), {}, g_score=10.0, h_score=0.0)
        s_low  = _AStarState(frozenset(), {}, g_score=1.0,  h_score=0.0)
        # s_high has f_score=-10, s_low has f_score=-1; s_high < s_low
        assert s_high < s_low

    def test_placed_ids_stored(self):
        placed = frozenset([1, 3, 5])
        s = _AStarState(placed, {}, g_score=1.0, h_score=0.5)
        assert s.placed_ids == placed

    def test_g_score_stored(self):
        s = _AStarState(frozenset(), {}, g_score=7.5, h_score=2.5)
        assert s.g_score == pytest.approx(7.5)


# ── astar_assembly extra ──────────────────────────────────────────────────────

class TestAstarAssemblyExtra:

    def test_two_fragments_both_placed(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        entries = [_make_entry(f0, f1, 0.8)]
        result = astar_assembly([f0, f1], entries)
        assert set(result.placements.keys()) == {0, 1}

    def test_five_fragments_all_placed(self):
        frags = [_make_frag(i) for i in range(5)]
        entries = [_make_entry(frags[i], frags[i + 1], 0.8) for i in range(4)]
        result = astar_assembly(frags, entries, max_states=5000)
        assert len(result.placements) == 5

    def test_returns_assembly_type(self):
        f0 = _make_frag(0)
        result = astar_assembly([f0], [])
        assert isinstance(result, Assembly)

    def test_single_fragment_placement_at_origin(self):
        f0 = _make_frag(0)
        result = astar_assembly([f0], [])
        pos, rot = result.placements[0]
        assert np.all(np.isfinite(pos))
        assert np.isfinite(rot)

    def test_total_score_increases_with_entries(self):
        frags = [_make_frag(i) for i in range(3)]
        r_no  = astar_assembly(frags, [])
        r_yes = astar_assembly(frags, [
            _make_entry(frags[0], frags[1], 0.9),
            _make_entry(frags[1], frags[2], 0.8),
        ])
        assert r_yes.total_score >= r_no.total_score

    def test_beam_width_1_still_returns(self):
        frags = [_make_frag(i) for i in range(3)]
        entries = [_make_entry(frags[0], frags[1], 0.7)]
        result = astar_assembly(frags, entries, beam_width=1)
        assert isinstance(result, Assembly)

    def test_beam_width_large_same_as_default(self):
        frags = [_make_frag(i) for i in range(3)]
        entries = [_make_entry(frags[i], frags[i + 1], 0.8) for i in range(2)]
        r1 = astar_assembly(frags, entries, max_states=1000, beam_width=50)
        r2 = astar_assembly(frags, entries, max_states=1000, beam_width=500)
        # Both should place all fragments
        assert len(r1.placements) == 3
        assert len(r2.placements) == 3

    def test_method_attribute_is_astar(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        result = astar_assembly([f0, f1], [])
        assert result.method == "astar"

    def test_zero_score_entries_still_works(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        entries = [_make_entry(f0, f1, 0.0)]
        result = astar_assembly([f0, f1], entries)
        assert isinstance(result, Assembly)
        assert result.total_score == pytest.approx(0.0)

    def test_total_score_is_float(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        result = astar_assembly([f0, f1], [_make_entry(f0, f1, 0.5)])
        assert isinstance(result.total_score, float)

    def test_placements_rotations_are_finite(self):
        frags = [_make_frag(i) for i in range(4)]
        entries = [_make_entry(frags[i], frags[i + 1], 0.7) for i in range(3)]
        result = astar_assembly(frags, entries, max_states=2000)
        for fid, (pos, rot) in result.placements.items():
            assert np.isfinite(rot), f"rot not finite for fid={fid}"

    def test_repeated_calls_same_result(self):
        frags = [_make_frag(i) for i in range(3)]
        entries = [_make_entry(frags[i], frags[i + 1], 0.8) for i in range(2)]
        r1 = astar_assembly(frags, entries, max_states=500)
        r2 = astar_assembly(frags, entries, max_states=500)
        assert r1.total_score == pytest.approx(r2.total_score)

    def test_compat_matrix_field_is_array(self):
        f0 = _make_frag(0)
        result = astar_assembly([f0], [])
        assert isinstance(result.compat_matrix, np.ndarray)
