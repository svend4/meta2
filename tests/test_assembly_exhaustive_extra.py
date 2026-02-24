"""Extra tests for puzzle_reconstruction/assembly/exhaustive.py."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from puzzle_reconstruction.models import (
    EdgeSignature,
    EdgeSide,
    Fragment,
    CompatEntry,
    Assembly,
)
from puzzle_reconstruction.assembly.exhaustive import (
    exhaustive_assembly,
    MAX_EXACT_N,
    WARN_N,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _edge(edge_id: int, side: EdgeSide = EdgeSide.TOP) -> EdgeSignature:
    """Create a minimal EdgeSignature."""
    return EdgeSignature(
        edge_id=edge_id,
        side=side,
        virtual_curve=np.zeros((10, 2)),
        fd=1.0,
        css_vec=np.zeros(8),
        ifs_coeffs=np.zeros(4),
        length=100.0,
    )


def _fragment(fid: int, n_edges: int = 2) -> Fragment:
    """Create a minimal Fragment with n_edges edges."""
    base_eid = fid * 10
    edges = [_edge(base_eid + i) for i in range(n_edges)]
    return Fragment(
        fragment_id=fid,
        image=np.zeros((20, 20, 3), dtype=np.uint8),
        mask=np.ones((20, 20), dtype=np.uint8) * 255,
        contour=np.array([[0, 0], [20, 0], [20, 20], [0, 20]]),
        edges=edges,
    )


def _compat(e1: EdgeSignature, e2: EdgeSignature, score: float) -> CompatEntry:
    """Create a CompatEntry between two edges."""
    return CompatEntry(
        edge_i=e1,
        edge_j=e2,
        score=score,
        dtw_dist=1.0 - score,
        css_sim=score,
        fd_diff=0.1,
        text_score=score,
    )


# ─── exhaustive_assembly ────────────────────────────────────────────────────

class TestExhaustiveAssemblyExtra:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            exhaustive_assembly([], [])

    def test_single_fragment(self):
        f = _fragment(0)
        result = exhaustive_assembly([f], [])
        assert isinstance(result, Assembly)
        assert len(result.fragments) == 1

    def test_two_fragments(self):
        f0 = _fragment(0)
        f1 = _fragment(1)
        e = _compat(f0.edges[0], f1.edges[0], 0.9)
        result = exhaustive_assembly([f0, f1], [e])
        assert isinstance(result, Assembly)
        assert len(result.fragments) == 2

    def test_three_fragments(self):
        frags = [_fragment(i) for i in range(3)]
        entries = [
            _compat(frags[0].edges[0], frags[1].edges[0], 0.8),
            _compat(frags[1].edges[1], frags[2].edges[0], 0.7),
            _compat(frags[0].edges[1], frags[2].edges[1], 0.6),
        ]
        result = exhaustive_assembly(frags, entries, allow_rotation=False)
        assert isinstance(result, Assembly)
        assert result.total_score >= 0.0

    def test_no_rotation(self):
        f0 = _fragment(0)
        f1 = _fragment(1)
        e = _compat(f0.edges[0], f1.edges[0], 0.9)
        result = exhaustive_assembly([f0, f1], [e], allow_rotation=False)
        assert isinstance(result, Assembly)

    def test_returns_assembly_with_placements(self):
        f0 = _fragment(0)
        f1 = _fragment(1)
        e = _compat(f0.edges[0], f1.edges[0], 0.9)
        result = exhaustive_assembly([f0, f1], [e], allow_rotation=False)
        assert isinstance(result.placements, dict)

    def test_warns_at_warn_n(self):
        frags = [_fragment(i) for i in range(WARN_N)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            exhaustive_assembly(frags, [], allow_rotation=False)
            assert any("медленным" in str(warning.message) for warning in w)

    def test_constants(self):
        assert MAX_EXACT_N >= 8
        assert WARN_N <= MAX_EXACT_N
