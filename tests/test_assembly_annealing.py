"""Тесты для puzzle_reconstruction/assembly/annealing.py."""
import pytest
import numpy as np

from puzzle_reconstruction.models import (
    Fragment, Assembly, CompatEntry, EdgeSignature, EdgeSide,
)
from puzzle_reconstruction.assembly.annealing import simulated_annealing


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_edge(edge_id: int) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=np.zeros((10, 2)),
        fd=1.5,
        css_vec=np.zeros(8),
        ifs_coeffs=np.zeros(4),
        length=80.0,
    )


def _make_fragment(fid: int, n_edges: int = 2) -> Fragment:
    frag = Fragment(
        fragment_id=fid,
        image=np.zeros((32, 32, 3), dtype=np.uint8),
        mask=np.zeros((32, 32), dtype=np.uint8),
        contour=np.zeros((8, 2)),
    )
    frag.edges = [_make_edge(fid * 10 + i) for i in range(n_edges)]
    return frag


def _make_entry(ei: EdgeSignature, ej: EdgeSignature,
                score: float = 0.5) -> CompatEntry:
    return CompatEntry(
        edge_i=ei, edge_j=ej, score=score,
        dtw_dist=0.0, css_sim=score, fd_diff=0.0, text_score=0.0,
    )


def _make_assembly(frags):
    """Build a minimal Assembly with identity placements."""
    placements = {f.fragment_id: (np.array([float(i * 100), 0.0]), 0.0)
                  for i, f in enumerate(frags)}
    return Assembly(
        fragments=frags,
        placements=placements,
        compat_matrix=np.zeros((len(frags), len(frags))),
        total_score=0.0,
    )


def _make_entries(frags):
    entries = []
    for i, fi in enumerate(frags):
        for j, fj in enumerate(frags):
            if i >= j:
                continue
            entries.append(_make_entry(fi.edges[0], fj.edges[0], score=0.5))
    return entries


# ─── simulated_annealing ──────────────────────────────────────────────────────

class TestSimulatedAnnealing:
    def test_returns_assembly(self):
        frags = [_make_fragment(i) for i in range(3)]
        asm = _make_assembly(frags)
        entries = _make_entries(frags)
        result = simulated_annealing(asm, entries, max_iter=20, seed=0)
        assert isinstance(result, Assembly)

    def test_single_fragment_returned_as_is(self):
        frags = [_make_fragment(0)]
        asm = _make_assembly(frags)
        result = simulated_annealing(asm, [], max_iter=10)
        assert isinstance(result, Assembly)
        assert len(result.fragments) == 1

    def test_all_fragments_in_result(self):
        frags = [_make_fragment(i) for i in range(4)]
        asm = _make_assembly(frags)
        entries = _make_entries(frags)
        result = simulated_annealing(asm, entries, max_iter=50, seed=1)
        assert len(result.fragments) == 4

    def test_placements_contain_all_fragments(self):
        frags = [_make_fragment(i) for i in range(3)]
        asm = _make_assembly(frags)
        entries = _make_entries(frags)
        result = simulated_annealing(asm, entries, max_iter=30, seed=2)
        for f in frags:
            assert f.fragment_id in result.placements

    def test_total_score_is_float(self):
        frags = [_make_fragment(i) for i in range(3)]
        asm = _make_assembly(frags)
        entries = _make_entries(frags)
        result = simulated_annealing(asm, entries, max_iter=20, seed=0)
        assert isinstance(result.total_score, float)

    def test_total_score_nonneg(self):
        frags = [_make_fragment(i) for i in range(3)]
        asm = _make_assembly(frags)
        entries = _make_entries(frags)
        result = simulated_annealing(asm, entries, max_iter=20, seed=0)
        assert result.total_score >= 0.0

    def test_deterministic_with_same_seed(self):
        frags = [_make_fragment(i) for i in range(4)]
        entries = _make_entries(frags)
        r1 = simulated_annealing(_make_assembly(frags), entries, max_iter=50, seed=7)
        r2 = simulated_annealing(_make_assembly(frags), entries, max_iter=50, seed=7)
        assert r1.total_score == pytest.approx(r2.total_score)

    def test_empty_entries_accepted(self):
        frags = [_make_fragment(i) for i in range(3)]
        asm = _make_assembly(frags)
        result = simulated_annealing(asm, [], max_iter=20, seed=0)
        assert isinstance(result, Assembly)

    def test_high_cooling_terminates_quickly(self):
        frags = [_make_fragment(i) for i in range(3)]
        asm = _make_assembly(frags)
        entries = _make_entries(frags)
        # cooling=0.01 → temperature drops below T_min almost immediately
        result = simulated_annealing(
            asm, entries, T_max=1.0, T_min=0.9, cooling=0.01, max_iter=1000, seed=0
        )
        assert isinstance(result, Assembly)

    def test_max_iter_zero_returns_assembly(self):
        frags = [_make_fragment(i) for i in range(2)]
        asm = _make_assembly(frags)
        entries = _make_entries(frags)
        result = simulated_annealing(asm, entries, max_iter=0, seed=0)
        assert isinstance(result, Assembly)

    def test_placements_values_are_pairs(self):
        frags = [_make_fragment(i) for i in range(3)]
        asm = _make_assembly(frags)
        entries = _make_entries(frags)
        result = simulated_annealing(asm, entries, max_iter=30, seed=0)
        for fid, val in result.placements.items():
            pos, angle = val
            assert isinstance(float(angle), float)

    def test_ocr_score_preserved(self):
        frags = [_make_fragment(i) for i in range(2)]
        asm = _make_assembly(frags)
        asm.ocr_score = 0.88
        entries = _make_entries(frags)
        result = simulated_annealing(asm, entries, max_iter=10, seed=0)
        assert result.ocr_score == pytest.approx(0.88)

    def test_compat_matrix_preserved(self):
        frags = [_make_fragment(i) for i in range(2)]
        asm = _make_assembly(frags)
        asm.compat_matrix = np.eye(2)
        entries = _make_entries(frags)
        result = simulated_annealing(asm, entries, max_iter=10, seed=0)
        assert result.compat_matrix.shape == (2, 2)
