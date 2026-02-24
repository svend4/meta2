"""Extra tests for puzzle_reconstruction/assembly/genetic.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.models import (
    EdgeSignature,
    EdgeSide,
    Fragment,
    CompatEntry,
    Assembly,
)
from puzzle_reconstruction.assembly.genetic import genetic_assembly


# ─── helpers ──────────────────────────────────────────────────────────────────

def _edge(edge_id: int, side: EdgeSide = EdgeSide.TOP) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id, side=side,
        virtual_curve=np.zeros((10, 2)), fd=1.0,
        css_vec=np.zeros(8), ifs_coeffs=np.zeros(4), length=100.0,
    )


def _fragment(fid: int, n_edges: int = 2) -> Fragment:
    base = fid * 10
    edges = [_edge(base + i) for i in range(n_edges)]
    return Fragment(
        fragment_id=fid,
        image=np.zeros((20, 20, 3), dtype=np.uint8),
        mask=np.ones((20, 20), dtype=np.uint8) * 255,
        contour=np.array([[0, 0], [20, 0], [20, 20], [0, 20]]),
        edges=edges,
    )


def _compat(e1: EdgeSignature, e2: EdgeSignature, score: float) -> CompatEntry:
    return CompatEntry(
        edge_i=e1, edge_j=e2, score=score,
        dtw_dist=1.0 - score, css_sim=score,
        fd_diff=0.1, text_score=score,
    )


# ─── genetic_assembly ──────────────────────────────────────────────────────

class TestGeneticAssemblyExtra:
    def test_empty(self):
        result = genetic_assembly([], [])
        assert isinstance(result, Assembly)
        assert len(result.fragments) == 0

    def test_single_fragment(self):
        f = _fragment(0)
        result = genetic_assembly([f], [], population_size=5, n_generations=2)
        assert isinstance(result, Assembly)
        assert len(result.fragments) == 1

    def test_two_fragments(self):
        f0, f1 = _fragment(0), _fragment(1)
        e = _compat(f0.edges[0], f1.edges[0], 0.9)
        result = genetic_assembly([f0, f1], [e],
                                  population_size=10, n_generations=5)
        assert isinstance(result, Assembly)
        assert len(result.fragments) == 2
        assert result.total_score >= 0.0

    def test_three_fragments(self):
        frags = [_fragment(i) for i in range(3)]
        entries = [
            _compat(frags[0].edges[0], frags[1].edges[0], 0.8),
            _compat(frags[1].edges[1], frags[2].edges[0], 0.7),
        ]
        result = genetic_assembly(frags, entries,
                                  population_size=10, n_generations=5,
                                  allow_rotation=False)
        assert isinstance(result, Assembly)
        assert len(result.placements) == 3

    def test_no_rotation(self):
        f0, f1 = _fragment(0), _fragment(1)
        e = _compat(f0.edges[0], f1.edges[0], 0.9)
        result = genetic_assembly([f0, f1], [e],
                                  population_size=10, n_generations=5,
                                  allow_rotation=False)
        assert isinstance(result, Assembly)

    def test_deterministic_seed(self):
        frags = [_fragment(i) for i in range(3)]
        entries = [_compat(frags[0].edges[0], frags[1].edges[0], 0.8)]
        r1 = genetic_assembly(frags, entries, population_size=10,
                              n_generations=5, seed=42)
        r2 = genetic_assembly(frags, entries, population_size=10,
                              n_generations=5, seed=42)
        assert r1.total_score == pytest.approx(r2.total_score)

    def test_placements_dict(self):
        frags = [_fragment(i) for i in range(2)]
        result = genetic_assembly(frags, [], population_size=5,
                                  n_generations=2)
        assert isinstance(result.placements, dict)
        for fid in [0, 1]:
            assert fid in result.placements
