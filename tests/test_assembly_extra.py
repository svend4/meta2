"""Additional tests for greedy/SA/beam_search assembly algorithms."""
import pytest
import numpy as np

from puzzle_reconstruction.models import (
    Assembly, CompatEntry, EdgeSide, EdgeSignature, Fragment,
)
from puzzle_reconstruction.assembly.greedy import greedy_assembly
from puzzle_reconstruction.assembly.annealing import simulated_annealing
from puzzle_reconstruction.assembly.beam_search import beam_search


# ─── helpers ──────────────────────────────────────────────────────────────────

def _edge(eid: int, length: float = 60.0) -> EdgeSignature:
    return EdgeSignature(
        edge_id=eid,
        side=EdgeSide.TOP,
        virtual_curve=np.zeros((8, 2)),
        fd=1.3,
        css_vec=np.zeros(8),
        ifs_coeffs=np.zeros(4),
        length=length,
    )


def _frag(fid: int) -> Fragment:
    return Fragment(
        fragment_id=fid,
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        mask=np.zeros((16, 16), dtype=np.uint8),
        contour=np.zeros((4, 2)),
        edges=[_edge(fid * 10 + i) for i in range(2)],
    )


def _entry(fi, fj, score=0.7):
    return CompatEntry(
        edge_i=fi.edges[0], edge_j=fj.edges[0],
        score=score, dtw_dist=0.2, css_sim=score, fd_diff=0.1, text_score=0.5,
    )


def _greedy_asm(n: int) -> Assembly:
    frags = [_frag(i) for i in range(n)]
    entries = [_entry(frags[i], frags[(i+1) % n]) for i in range(n-1)]
    return greedy_assembly(frags, entries)


# ─── TestGreedyExtra ──────────────────────────────────────────────────────────

class TestGreedyExtra:
    def test_fragments_list_unchanged(self):
        frags = [_frag(i) for i in range(3)]
        asm = greedy_assembly(frags, [])
        assert asm.fragments is frags

    def test_total_score_nonneg(self):
        frags = [_frag(i) for i in range(3)]
        entries = [_entry(frags[0], frags[1]), _entry(frags[1], frags[2])]
        asm = greedy_assembly(frags, entries)
        assert asm.total_score >= 0.0

    def test_compat_matrix_is_ndarray(self):
        frags = [_frag(i) for i in range(3)]
        asm = greedy_assembly(frags, [_entry(frags[0], frags[1])])
        assert isinstance(asm.compat_matrix, np.ndarray)

    def test_all_positions_finite(self):
        frags = [_frag(i) for i in range(4)]
        asm = greedy_assembly(frags, [])
        for pos, angle in asm.placements.values():
            assert np.all(np.isfinite(pos))
            assert np.isfinite(angle)

    def test_high_score_entry_uses_both_fragments(self):
        frags = [_frag(0), _frag(1), _frag(2)]
        high = _entry(frags[0], frags[1], score=0.99)
        asm = greedy_assembly(frags, [high])
        assert 0 in asm.placements
        assert 1 in asm.placements

    def test_many_fragments_all_placed(self):
        frags = [_frag(i) for i in range(8)]
        asm = greedy_assembly(frags, [])
        assert len(asm.placements) == 8

    def test_positions_are_2d_vectors(self):
        frags = [_frag(i) for i in range(3)]
        asm = greedy_assembly(frags, [])
        for pos, _ in asm.placements.values():
            assert len(pos) == 2


# ─── TestSimulatedAnnealingExtra ──────────────────────────────────────────────

class TestSimulatedAnnealingExtra:
    def test_total_score_finite(self):
        asm0 = _greedy_asm(3)
        result = simulated_annealing(asm0, [], max_iter=10, seed=0)
        assert np.isfinite(result.total_score)

    def test_custom_cooling_no_crash(self):
        asm0 = _greedy_asm(3)
        result = simulated_annealing(asm0, [], cooling=0.99, max_iter=10, seed=1)
        assert isinstance(result, Assembly)

    def test_custom_t_min_t_max_no_crash(self):
        asm0 = _greedy_asm(3)
        result = simulated_annealing(asm0, [], T_max=100.0, T_min=0.001,
                                     max_iter=20, seed=2)
        assert isinstance(result, Assembly)

    def test_compat_matrix_is_ndarray(self):
        asm0 = _greedy_asm(3)
        result = simulated_annealing(asm0, [], max_iter=5, seed=3)
        assert isinstance(result.compat_matrix, np.ndarray)

    def test_fragments_list_same_as_input(self):
        frags = [_frag(i) for i in range(3)]
        asm0 = greedy_assembly(frags, [])
        result = simulated_annealing(asm0, [], max_iter=5, seed=4)
        assert result.fragments is frags

    def test_2_fragments_both_placed(self):
        frags = [_frag(0), _frag(1)]
        asm0 = greedy_assembly(frags, [_entry(frags[0], frags[1])])
        result = simulated_annealing(asm0, [], max_iter=10, seed=5)
        assert len(result.placements) == 2


# ─── TestBeamSearchExtra ──────────────────────────────────────────────────────

class TestBeamSearchExtra:
    def test_total_score_nonneg(self):
        frags = [_frag(i) for i in range(3)]
        asm = beam_search(frags, [], beam_width=3)
        assert asm.total_score >= 0.0

    def test_compat_matrix_is_ndarray(self):
        frags = [_frag(i) for i in range(3)]
        asm = beam_search(frags, [])
        assert isinstance(asm.compat_matrix, np.ndarray)

    def test_fragments_list_same_as_input(self):
        frags = [_frag(i) for i in range(3)]
        asm = beam_search(frags, [])
        assert asm.fragments is frags

    def test_all_positions_finite(self):
        frags = [_frag(i) for i in range(4)]
        entries = [_entry(frags[i], frags[(i+1) % 4]) for i in range(3)]
        asm = beam_search(frags, entries, beam_width=2)
        for pos, angle in asm.placements.values():
            assert np.all(np.isfinite(pos))

    def test_max_depth_param_no_crash(self):
        frags = [_frag(i) for i in range(4)]
        asm = beam_search(frags, [], max_depth=2)
        assert isinstance(asm, Assembly)

    def test_beam_width_1_places_all(self):
        frags = [_frag(i) for i in range(4)]
        asm = beam_search(frags, [], beam_width=1)
        assert len(asm.placements) == 4

    def test_single_fragment_at_origin(self):
        frags = [_frag(42)]
        asm = beam_search(frags, [])
        pos, angle = asm.placements[42]
        np.testing.assert_allclose(pos, [0.0, 0.0], atol=1e-10)
