"""Тесты для puzzle_reconstruction/assembly/beam_search.py."""
import pytest
import numpy as np

from puzzle_reconstruction.models import (
    Fragment, Assembly, CompatEntry, EdgeSignature, EdgeSide,
)
from puzzle_reconstruction.assembly.beam_search import (
    Hypothesis,
    beam_search,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_edge(edge_id: int) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
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


def _make_entries(frags):
    entries = []
    for i, fi in enumerate(frags):
        for j, fj in enumerate(frags):
            if i >= j:
                continue
            s = min(0.1 * (i + j + 1), 1.0)
            entries.append(_make_entry(fi.edges[0], fj.edges[0], score=s))
    # sort descending by score
    return sorted(entries, key=lambda e: e.score, reverse=True)


# ─── Hypothesis ───────────────────────────────────────────────────────────────

class TestHypothesis:
    def test_creation(self):
        hyp = Hypothesis(placements={}, placed_ids=set())
        assert hyp.score == pytest.approx(0.0)
        assert hyp.last_entries == []

    def test_score_stored(self):
        hyp = Hypothesis(placements={}, placed_ids=set(), score=3.14)
        assert hyp.score == pytest.approx(3.14)

    def test_placed_ids_stored(self):
        hyp = Hypothesis(placements={}, placed_ids={0, 1, 2})
        assert 0 in hyp.placed_ids

    def test_last_entries_default_empty(self):
        hyp = Hypothesis(placements={}, placed_ids=set())
        assert hyp.last_entries == []


# ─── beam_search ──────────────────────────────────────────────────────────────

class TestBeamSearch:
    def test_empty_fragments_returns_assembly(self):
        result = beam_search([], [])
        assert isinstance(result, Assembly)

    def test_empty_fragments_empty_placements(self):
        result = beam_search([], [])
        assert result.placements == {}

    def test_single_fragment_returns_assembly(self):
        frags = [_make_fragment(0)]
        result = beam_search(frags, [])
        assert isinstance(result, Assembly)

    def test_single_fragment_placed(self):
        frags = [_make_fragment(0)]
        result = beam_search(frags, [])
        assert 0 in result.placements

    def test_returns_assembly(self):
        frags = [_make_fragment(i) for i in range(3)]
        entries = _make_entries(frags)
        result = beam_search(frags, entries)
        assert isinstance(result, Assembly)

    def test_all_fragments_placed(self):
        frags = [_make_fragment(i) for i in range(4)]
        entries = _make_entries(frags)
        result = beam_search(frags, entries, beam_width=2)
        for f in frags:
            assert f.fragment_id in result.placements

    def test_placements_count_matches_fragments(self):
        frags = [_make_fragment(i) for i in range(4)]
        entries = _make_entries(frags)
        result = beam_search(frags, entries)
        assert len(result.placements) == len(frags)

    def test_total_score_nonneg(self):
        frags = [_make_fragment(i) for i in range(3)]
        entries = _make_entries(frags)
        result = beam_search(frags, entries)
        assert result.total_score >= 0.0

    def test_total_score_is_float(self):
        frags = [_make_fragment(i) for i in range(3)]
        entries = _make_entries(frags)
        result = beam_search(frags, entries)
        assert isinstance(result.total_score, float)

    def test_beam_width_1_runs(self):
        frags = [_make_fragment(i) for i in range(3)]
        entries = _make_entries(frags)
        result = beam_search(frags, entries, beam_width=1)
        assert isinstance(result, Assembly)

    def test_beam_width_large_runs(self):
        frags = [_make_fragment(i) for i in range(4)]
        entries = _make_entries(frags)
        result = beam_search(frags, entries, beam_width=100)
        assert isinstance(result, Assembly)

    def test_max_depth_limits_run(self):
        frags = [_make_fragment(i) for i in range(5)]
        entries = _make_entries(frags)
        result = beam_search(frags, entries, max_depth=2)
        assert isinstance(result, Assembly)
        # All fragments must still be placed (orphans filled)
        assert len(result.placements) == len(frags)

    def test_max_depth_none_uses_all(self):
        frags = [_make_fragment(i) for i in range(3)]
        entries = _make_entries(frags)
        result = beam_search(frags, entries, max_depth=None)
        assert len(result.placements) == len(frags)

    def test_empty_entries_places_all_as_orphans(self):
        frags = [_make_fragment(i) for i in range(4)]
        result = beam_search(frags, [])
        assert len(result.placements) == len(frags)

    def test_placement_values_are_pos_angle_pairs(self):
        frags = [_make_fragment(i) for i in range(3)]
        entries = _make_entries(frags)
        result = beam_search(frags, entries)
        for fid, (pos, angle) in result.placements.items():
            assert hasattr(pos, '__len__')
            assert isinstance(float(angle), float)

    def test_two_fragments_with_entry(self):
        frags = [_make_fragment(0), _make_fragment(1)]
        entries = [_make_entry(frags[0].edges[0], frags[1].edges[0], score=0.9)]
        result = beam_search(frags, entries)
        assert 0 in result.placements
        assert 1 in result.placements

    def test_fragments_stored_in_assembly(self):
        frags = [_make_fragment(i) for i in range(3)]
        entries = _make_entries(frags)
        result = beam_search(frags, entries)
        assert result.fragments is frags
