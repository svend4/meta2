"""Tests for puzzle_reconstruction/assembly/astar.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.assembly.astar import (
    astar_assembly,
    _build_edge_to_frag,
    _build_best_score_per_frag,
    _score_for_placement,
    _heuristic,
    _place_new_fragment,
    _AStarState,
)
from puzzle_reconstruction.models import Fragment, CompatEntry, Assembly, EdgeSignature, EdgeSide


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


def _make_entry(frag_a: Fragment, frag_b: Fragment, score: float = 0.8) -> CompatEntry:
    return CompatEntry(
        edge_i=frag_a.edges[0],
        edge_j=frag_b.edges[0],
        score=score,
    )


# ── _build_edge_to_frag ───────────────────────────────────────────────────────

class TestBuildEdgeToFrag:

    def test_basic_mapping(self):
        f0 = _make_frag(0, 2)
        f1 = _make_frag(1, 2)
        mapping = _build_edge_to_frag([f0, f1])
        for edge in f0.edges + f1.edges:
            assert edge.edge_id in mapping

    def test_correct_fragment_returned(self):
        f0 = _make_frag(0, 1)
        mapping = _build_edge_to_frag([f0])
        assert mapping[f0.edges[0].edge_id].fragment_id == 0

    def test_empty_fragments(self):
        assert _build_edge_to_frag([]) == {}


# ── _build_best_score_per_frag ────────────────────────────────────────────────

class TestBuildBestScorePerFrag:

    def test_returns_dict_for_all_frags(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        entries = [_make_entry(f0, f1, 0.7)]
        e2f = _build_edge_to_frag([f0, f1])
        best = _build_best_score_per_frag([f0, f1], entries, e2f)
        assert 0 in best and 1 in best

    def test_best_score_values(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        entries = [_make_entry(f0, f1, 0.9)]
        e2f = _build_edge_to_frag([f0, f1])
        best = _build_best_score_per_frag([f0, f1], entries, e2f)
        assert best[0] == pytest.approx(0.9)
        assert best[1] == pytest.approx(0.9)

    def test_no_entries_zero_score(self):
        f0 = _make_frag(0)
        e2f = _build_edge_to_frag([f0])
        best = _build_best_score_per_frag([f0], [], e2f)
        assert best[0] == 0.0


# ── _heuristic ─────────────────────────────────────────────────────────────────

class TestHeuristic:

    def test_empty_unplaced_zero(self):
        assert _heuristic(frozenset(), {0: 0.8, 1: 0.7}) == pytest.approx(0.0)

    def test_sum_of_best_scores(self):
        h = _heuristic(frozenset([0, 1]), {0: 0.8, 1: 0.7})
        assert h == pytest.approx(1.5)

    def test_missing_frag_treated_as_zero(self):
        h = _heuristic(frozenset([99]), {0: 0.8})
        assert h == pytest.approx(0.0)


# ── _score_for_placement ──────────────────────────────────────────────────────

class TestScoreForPlacement:

    def test_no_entries_zero_score(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        e2f = _build_edge_to_frag([f0, f1])
        s = _score_for_placement(f1, frozenset([0]), [], e2f)
        assert s == pytest.approx(0.0)

    def test_entry_counted_when_anchor_placed(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        entries = [_make_entry(f0, f1, 0.75)]
        e2f = _build_edge_to_frag([f0, f1])
        s = _score_for_placement(f1, frozenset([0]), entries, e2f)
        assert s == pytest.approx(0.75)

    def test_not_counted_when_anchor_not_placed(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        entries = [_make_entry(f0, f1, 0.75)]
        e2f = _build_edge_to_frag([f0, f1])
        # No fragments placed → score should be 0
        s = _score_for_placement(f1, frozenset(), entries, e2f)
        assert s == pytest.approx(0.0)


# ── _place_new_fragment ────────────────────────────────────────────────────────

class TestPlaceNewFragment:

    def test_returns_position_and_angle(self):
        f1 = _make_frag(1)
        placements = {0: (np.array([0.0, 0.0]), 0.0)}
        pos, rot = _place_new_fragment(f1, 0, placements, None, 1)
        assert pos.shape == (2,)
        assert isinstance(rot, float)

    def test_offset_increases_with_index(self):
        f1 = _make_frag(1)
        placements = {0: (np.array([0.0, 0.0]), 0.0)}
        pos1, _ = _place_new_fragment(f1, 0, placements, None, 1)
        pos2, _ = _place_new_fragment(f1, 0, placements, None, 2)
        assert pos2[0] > pos1[0]

    def test_no_anchor_still_works(self):
        f1 = _make_frag(1)
        pos, rot = _place_new_fragment(f1, 999, {}, None, 0)
        assert np.isfinite(pos).all()


# ── _AStarState ────────────────────────────────────────────────────────────────

class TestAStarState:

    def test_lt_comparison(self):
        s1 = _AStarState(frozenset([0]), {}, 1.0, 0.5)
        s2 = _AStarState(frozenset([0, 1]), {}, 2.0, 0.0)
        # States should be comparable for heap ordering
        assert (s1 < s2) or (s2 < s1) or (s1 == s2)

    def test_f_score_is_negated(self):
        s = _AStarState(frozenset(), {}, g_score=3.0, h_score=2.0)
        assert s.f_score == pytest.approx(-(3.0 + 2.0))


# ── astar_assembly ─────────────────────────────────────────────────────────────

class TestAstarAssembly:

    def test_returns_assembly(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        entries = [_make_entry(f0, f1, 0.8)]
        result = astar_assembly([f0, f1], entries)
        assert isinstance(result, Assembly)

    def test_empty_fragments(self):
        result = astar_assembly([], [])
        assert isinstance(result, Assembly)
        assert result.placements == {}

    def test_single_fragment(self):
        f0 = _make_frag(0)
        result = astar_assembly([f0], [])
        assert isinstance(result, Assembly)

    def test_method_is_astar(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        result = astar_assembly([f0, f1], [])
        assert result.method == "astar"

    def test_all_fragments_placed(self):
        frags = [_make_frag(i) for i in range(3)]
        entries = [
            _make_entry(frags[0], frags[1], 0.9),
            _make_entry(frags[1], frags[2], 0.8),
        ]
        result = astar_assembly(frags, entries, max_states=500)
        assert len(result.placements) == 3

    def test_total_score_non_negative(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        entries = [_make_entry(f0, f1, 0.7)]
        result = astar_assembly([f0, f1], entries)
        assert result.total_score >= 0.0

    def test_higher_score_entry_leads_to_higher_score(self):
        f0, f1 = _make_frag(0), _make_frag(1)
        r_low  = astar_assembly([f0, f1], [_make_entry(f0, f1, 0.1)])
        r_high = astar_assembly([f0, f1], [_make_entry(f0, f1, 0.9)])
        assert r_high.total_score >= r_low.total_score

    def test_no_entries_still_returns(self):
        frags = [_make_frag(i) for i in range(4)]
        result = astar_assembly(frags, [])
        assert isinstance(result, Assembly)

    def test_max_states_1_returns_partial(self):
        frags = [_make_frag(i) for i in range(5)]
        entries = [_make_entry(frags[i], frags[i+1], 0.8) for i in range(4)]
        result = astar_assembly(frags, entries, max_states=1)
        assert isinstance(result, Assembly)

    def test_placements_positions_finite(self):
        frags = [_make_frag(i) for i in range(3)]
        entries = [_make_entry(frags[i], frags[i+1], 0.8) for i in range(2)]
        result = astar_assembly(frags, entries, max_states=1000)
        for fid, (pos, rot) in result.placements.items():
            assert np.all(np.isfinite(pos))
            assert np.isfinite(rot)
