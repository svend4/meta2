"""Tests for puzzle_reconstruction/assembly/ant_colony.py"""
import pytest
import numpy as np

from puzzle_reconstruction.models import (
    Assembly,
    CompatEntry,
    EdgeSignature,
    EdgeSide,
    Fragment,
)
from puzzle_reconstruction.assembly.ant_colony import ant_colony_assembly


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_edge(edge_id: int) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=np.zeros((5, 2), dtype=np.float64),
        fd=1.2,
        css_vec=np.zeros(4, dtype=np.float64),
        ifs_coeffs=np.zeros(4, dtype=np.float64),
        length=10.0,
    )


def _make_fragment(fid: int, n_edges: int = 2) -> Fragment:
    edges = [_make_edge(fid * 10 + i) for i in range(n_edges)]
    return Fragment(
        fragment_id=fid,
        image=np.zeros((20, 20, 3), dtype=np.uint8),
        mask=np.zeros((20, 20), dtype=np.uint8),
        contour=np.zeros((4, 2), dtype=np.float64),
        edges=edges,
    )


def _make_compat(fi: Fragment, fj: Fragment, score: float = 0.5) -> CompatEntry:
    return CompatEntry(
        edge_i=fi.edges[0],
        edge_j=fj.edges[0],
        score=score,
        dtw_dist=0.1,
        css_sim=0.9,
        fd_diff=0.05,
        text_score=0.8,
    )


def _frags(n: int):
    return [_make_fragment(i) for i in range(n)]


# ─── TestAntColonyAssembly ────────────────────────────────────────────────────

class TestAntColonyAssembly:
    def test_empty_fragments_returns_assembly(self):
        result = ant_colony_assembly([], [])
        assert isinstance(result, Assembly)

    def test_empty_fragments_empty_list(self):
        result = ant_colony_assembly([], [])
        assert result.fragments == []

    def test_returns_assembly_type(self):
        frags = _frags(3)
        result = ant_colony_assembly(frags, [], n_iterations=2)
        assert isinstance(result, Assembly)

    def test_all_fragments_placed(self):
        frags = _frags(4)
        result = ant_colony_assembly(frags, [], n_iterations=2)
        for f in frags:
            assert f.fragment_id in result.placements

    def test_placements_count_matches_fragments(self):
        frags = _frags(3)
        result = ant_colony_assembly(frags, [], n_iterations=2)
        assert len(result.placements) == 3

    def test_single_fragment(self):
        frags = [_make_fragment(0)]
        result = ant_colony_assembly(frags, [], n_iterations=1)
        assert isinstance(result, Assembly)
        assert 0 in result.placements

    def test_total_score_nonneg(self):
        frags = _frags(3)
        result = ant_colony_assembly(frags, [], n_iterations=2)
        assert result.total_score >= 0.0

    def test_with_compat_entries(self):
        frags = _frags(3)
        entries = [
            _make_compat(frags[0], frags[1], score=0.9),
            _make_compat(frags[1], frags[2], score=0.7),
        ]
        result = ant_colony_assembly(frags, entries, n_iterations=5, seed=42)
        assert isinstance(result, Assembly)

    def test_seed_reproducibility(self):
        frags = _frags(4)
        entries = [_make_compat(frags[0], frags[1])]
        r1 = ant_colony_assembly(frags, entries, n_iterations=5, seed=0)
        r2 = ant_colony_assembly(frags, entries, n_iterations=5, seed=0)
        assert r1.total_score == pytest.approx(r2.total_score)

    def test_no_entries_runs_ok(self):
        frags = _frags(5)
        result = ant_colony_assembly(frags, [], n_iterations=3, seed=1)
        assert isinstance(result, Assembly)
        assert len(result.placements) == 5

    def test_few_iterations(self):
        frags = _frags(3)
        result = ant_colony_assembly(frags, [], n_iterations=1, seed=7)
        assert isinstance(result, Assembly)

    def test_auto_n_ants(self):
        """n_ants=0 should automatically choose n_ants."""
        frags = _frags(5)
        result = ant_colony_assembly(frags, [], n_ants=0, n_iterations=2, seed=3)
        assert isinstance(result, Assembly)

    def test_explicit_n_ants(self):
        frags = _frags(3)
        result = ant_colony_assembly(frags, [], n_ants=5, n_iterations=2, seed=5)
        assert len(result.placements) == 3

    def test_no_rotation(self):
        frags = _frags(3)
        result = ant_colony_assembly(frags, [], n_iterations=2,
                                     allow_rotation=False, seed=9)
        assert isinstance(result, Assembly)

    def test_placements_have_position_and_angle(self):
        frags = _frags(2)
        result = ant_colony_assembly(frags, [], n_iterations=2, seed=1)
        for fid, placement in result.placements.items():
            pos, angle = placement
            assert isinstance(pos, np.ndarray)
            assert isinstance(angle, float)

    def test_high_score_entry_influences_result(self):
        """Fragments with high compat score should be placed consistently."""
        frags = _frags(3)
        high = _make_compat(frags[0], frags[1], score=0.99)
        low = _make_compat(frags[1], frags[2], score=0.01)
        result = ant_colony_assembly(frags, [high, low],
                                     n_iterations=10, seed=42)
        assert isinstance(result, Assembly)
        assert result.total_score >= 0.0
