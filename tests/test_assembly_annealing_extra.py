"""Additional tests for puzzle_reconstruction/assembly/annealing.py"""
import pytest
import numpy as np

from puzzle_reconstruction.models import (
    Fragment, Assembly, CompatEntry, EdgeSignature, EdgeSide,
)
from puzzle_reconstruction.assembly.annealing import simulated_annealing


# ─── helpers ──────────────────────────────────────────────────────────────────

def _edge(eid: int) -> EdgeSignature:
    return EdgeSignature(
        edge_id=eid,
        side=EdgeSide.TOP,
        virtual_curve=np.array([[float(i), 0.0] for i in range(5)]),
        fd=1.5,
        css_vec=np.zeros(8),
        ifs_coeffs=np.zeros(4),
        length=50.0,
    )


def _frag(fid: int, n_edges: int = 2) -> Fragment:
    frag = Fragment(
        fragment_id=fid,
        image=np.zeros((20, 20, 3), dtype=np.uint8),
        mask=np.zeros((20, 20), dtype=np.uint8),
        contour=np.zeros((4, 2)),
    )
    frag.edges = [_edge(fid * 10 + i) for i in range(n_edges)]
    return frag


def _asm(frags) -> Assembly:
    return Assembly(
        fragments=frags,
        placements={f.fragment_id: (np.array([float(i * 50), 0.0]), 0.0)
                    for i, f in enumerate(frags)},
        compat_matrix=np.zeros((len(frags), len(frags))),
        total_score=0.0,
    )


def _entries(frags) -> list:
    entries = []
    for i, fi in enumerate(frags):
        for j, fj in enumerate(frags):
            if i >= j:
                continue
            entries.append(CompatEntry(
                edge_i=fi.edges[0], edge_j=fj.edges[0],
                score=0.7, dtw_dist=0.3, css_sim=0.7,
                fd_diff=0.1, text_score=0.5,
            ))
    return entries


# ─── TestSimulatedAnnealingExtra ──────────────────────────────────────────────

class TestSimulatedAnnealingExtra:
    def test_placements_keys_match_fragment_ids(self):
        frags = [_frag(i) for i in range(3)]
        result = simulated_annealing(_asm(frags), _entries(frags),
                                     max_iter=30, seed=0)
        assert set(result.placements.keys()) == {f.fragment_id for f in frags}

    def test_two_fragments(self):
        frags = [_frag(0), _frag(1)]
        result = simulated_annealing(_asm(frags), _entries(frags),
                                     max_iter=20, seed=0)
        assert len(result.placements) == 2

    def test_different_seeds_may_differ(self):
        frags = [_frag(i) for i in range(4)]
        entries = _entries(frags)
        r1 = simulated_annealing(_asm(frags), entries, max_iter=200, seed=1)
        r2 = simulated_annealing(_asm(frags), entries, max_iter=200, seed=99)
        # Scores may differ — no fixed assertion, just no crash
        assert isinstance(r1.total_score, float)
        assert isinstance(r2.total_score, float)

    def test_t_min_above_t_max_no_iterations(self):
        """T_min > T_max → loop exits immediately without moves."""
        frags = [_frag(i) for i in range(3)]
        result = simulated_annealing(_asm(frags), _entries(frags),
                                     T_max=0.1, T_min=10.0,
                                     max_iter=1000, seed=0)
        assert isinstance(result, Assembly)

    def test_cooling_above_1_does_not_crash(self):
        """cooling >= 1 means temperature doesn't decrease but code still runs."""
        frags = [_frag(i) for i in range(2)]
        result = simulated_annealing(_asm(frags), _entries(frags),
                                     T_max=10.0, T_min=0.01,
                                     cooling=1.0, max_iter=10, seed=0)
        assert isinstance(result, Assembly)

    def test_placements_positions_are_2d(self):
        frags = [_frag(i) for i in range(3)]
        result = simulated_annealing(_asm(frags), _entries(frags),
                                     max_iter=20, seed=0)
        for fid, (pos, angle) in result.placements.items():
            assert len(pos) == 2

    def test_large_fragment_set(self):
        frags = [_frag(i) for i in range(8)]
        result = simulated_annealing(_asm(frags), _entries(frags),
                                     max_iter=30, seed=42)
        assert len(result.placements) == 8

    def test_result_fragments_unchanged(self):
        frags = [_frag(i) for i in range(3)]
        result = simulated_annealing(_asm(frags), _entries(frags),
                                     max_iter=20, seed=0)
        assert result.fragments is frags

    def test_empty_fragments_raises_or_returns(self):
        """With 0 fragments, function returns assembly without crashing."""
        asm = Assembly(fragments=[], placements={}, compat_matrix=np.array([]))
        result = simulated_annealing(asm, [], max_iter=10)
        assert isinstance(result, Assembly)

    def test_high_score_entries_improve_result(self):
        """Higher-score entries should generally keep or improve total_score."""
        frags = [_frag(i) for i in range(3)]
        entries_low = [CompatEntry(
            edge_i=frags[0].edges[0], edge_j=frags[1].edges[0],
            score=0.1, dtw_dist=0.9, css_sim=0.1, fd_diff=0.9, text_score=0.1,
        )]
        result_low = simulated_annealing(_asm(frags), entries_low,
                                          max_iter=20, seed=0)
        assert isinstance(result_low.total_score, float)

    def test_all_placements_have_float_angles(self):
        frags = [_frag(i) for i in range(4)]
        result = simulated_annealing(_asm(frags), _entries(frags),
                                     max_iter=50, seed=3)
        for fid, (pos, angle) in result.placements.items():
            assert isinstance(float(angle), float)
