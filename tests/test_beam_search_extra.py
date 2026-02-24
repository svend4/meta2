"""Extra tests for puzzle_reconstruction/assembly/beam_search.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.assembly.beam_search import (
    Hypothesis,
    beam_search,
)
from puzzle_reconstruction.models import (
    Assembly,
    CompatEntry,
    EdgeSide,
    EdgeSignature,
    Fragment,
)


# ─── helpers ────────────────────────────────────────────────────────────────

def _edge(edge_id: int, side: EdgeSide = EdgeSide.RIGHT) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=side,
        virtual_curve=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        fd=1.5,
        css_vec=np.zeros(8),
        ifs_coeffs=np.zeros(4),
        length=80.0,
    )


def _frag(fid: int, n_edges: int = 2) -> Fragment:
    frag = Fragment(
        fragment_id=fid,
        image=np.zeros((32, 32, 3), dtype=np.uint8),
        mask=np.zeros((32, 32), dtype=np.uint8),
        contour=np.zeros((8, 2)),
    )
    frag.edges = [_edge(fid * 10 + i) for i in range(n_edges)]
    return frag


def _entry(ei: EdgeSignature, ej: EdgeSignature, score: float = 0.7) -> CompatEntry:
    return CompatEntry(
        edge_i=ei, edge_j=ej, score=score,
        dtw_dist=0.0, css_sim=score, fd_diff=0.0, text_score=0.0,
    )


def _entries(frags) -> list:
    result = []
    for i, fi in enumerate(frags):
        for j, fj in enumerate(frags):
            if i >= j:
                continue
            result.append(_entry(fi.edges[0], fj.edges[0], 0.5 + 0.1 * (i + j)))
    return sorted(result, key=lambda e: e.score, reverse=True)


# ─── Hypothesis (extra) ──────────────────────────────────────────────────────

class TestHypothesisExtra:
    def test_default_score_zero(self):
        h = Hypothesis(placements={}, placed_ids=set())
        assert h.score == pytest.approx(0.0)

    def test_default_last_entries_empty(self):
        h = Hypothesis(placements={}, placed_ids=set())
        assert h.last_entries == []

    def test_placed_ids_stored(self):
        h = Hypothesis(placements={}, placed_ids={0, 1, 2})
        assert 0 in h.placed_ids
        assert 2 in h.placed_ids

    def test_placements_stored(self):
        placements = {0: (np.zeros(2), 0.0)}
        h = Hypothesis(placements=placements, placed_ids={0})
        assert 0 in h.placements

    def test_custom_score(self):
        h = Hypothesis(placements={}, placed_ids=set(), score=0.75)
        assert h.score == pytest.approx(0.75)

    def test_custom_last_entries(self):
        f0 = _frag(0)
        f1 = _frag(1)
        e = _entry(f0.edges[0], f1.edges[0])
        h = Hypothesis(placements={}, placed_ids=set(), last_entries=[e])
        assert len(h.last_entries) == 1

    def test_empty_placed_ids(self):
        h = Hypothesis(placements={}, placed_ids=set())
        assert len(h.placed_ids) == 0

    def test_score_negative_ok(self):
        h = Hypothesis(placements={}, placed_ids=set(), score=-1.0)
        assert h.score == pytest.approx(-1.0)


# ─── beam_search (extra) ─────────────────────────────────────────────────────

class TestBeamSearchExtra:
    def test_returns_assembly(self):
        frags = [_frag(i) for i in range(2)]
        result = beam_search(frags, _entries(frags))
        assert isinstance(result, Assembly)

    def test_no_entries_still_returns(self):
        frags = [_frag(i) for i in range(3)]
        result = beam_search(frags, [])
        assert isinstance(result, Assembly)

    def test_all_fragments_placed(self):
        frags = [_frag(i) for i in range(3)]
        result = beam_search(frags, _entries(frags))
        assert len(result.placements) == len(frags)

    def test_single_fragment(self):
        frags = [_frag(0)]
        result = beam_search(frags, [])
        assert isinstance(result, Assembly)
        assert len(result.placements) == 1

    def test_beam_width_one(self):
        frags = [_frag(i) for i in range(3)]
        result = beam_search(frags, _entries(frags), beam_width=1)
        assert isinstance(result, Assembly)

    def test_beam_width_large(self):
        frags = [_frag(i) for i in range(4)]
        result = beam_search(frags, _entries(frags), beam_width=50)
        assert isinstance(result, Assembly)

    def test_max_depth_limits_expansion(self):
        frags = [_frag(i) for i in range(5)]
        result = beam_search(frags, _entries(frags), beam_width=3, max_depth=2)
        assert isinstance(result, Assembly)

    def test_no_fragments_returns_empty_assembly(self):
        result = beam_search([], [])
        assert isinstance(result, Assembly)
        assert len(result.placements) == 0

    def test_two_fragments_high_score(self):
        f0 = _frag(0)
        f1 = _frag(1)
        ents = [_entry(f0.edges[0], f1.edges[0], score=0.95)]
        result = beam_search([f0, f1], ents)
        assert isinstance(result, Assembly)

    def test_duplicate_entries_handled(self):
        f0 = _frag(0)
        f1 = _frag(1)
        e = _entry(f0.edges[0], f1.edges[0])
        result = beam_search([f0, f1], [e, e, e])
        assert isinstance(result, Assembly)

    def test_placements_dict_keys_are_ints(self):
        frags = [_frag(i) for i in range(2)]
        result = beam_search(frags, _entries(frags))
        for k in result.placements:
            assert isinstance(k, int)

    def test_four_fragments(self):
        frags = [_frag(i) for i in range(4)]
        result = beam_search(frags, _entries(frags))
        assert isinstance(result, Assembly)
        assert len(result.placements) == 4
