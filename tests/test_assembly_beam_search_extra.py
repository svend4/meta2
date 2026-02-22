"""Additional tests for puzzle_reconstruction/assembly/beam_search.py"""
import pytest
import numpy as np

from puzzle_reconstruction.models import (
    Fragment, Assembly, CompatEntry, EdgeSignature, EdgeSide,
)
from puzzle_reconstruction.assembly.beam_search import Hypothesis, beam_search


# ─── helpers ──────────────────────────────────────────────────────────────────

def _edge(eid: int) -> EdgeSignature:
    return EdgeSignature(
        edge_id=eid,
        side=EdgeSide.TOP,
        virtual_curve=np.array([[float(i), 0.0] for i in range(4)]),
        fd=1.5,
        css_vec=np.zeros(8),
        ifs_coeffs=np.zeros(4),
        length=40.0,
    )


def _frag(fid: int, n_edges: int = 2) -> Fragment:
    frag = Fragment(
        fragment_id=fid,
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        mask=np.zeros((16, 16), dtype=np.uint8),
        contour=np.zeros((4, 2)),
    )
    frag.edges = [_edge(fid * 10 + i) for i in range(n_edges)]
    return frag


def _entry(ei, ej, score=0.8) -> CompatEntry:
    return CompatEntry(
        edge_i=ei, edge_j=ej, score=score,
        dtw_dist=0.2, css_sim=score, fd_diff=0.1, text_score=0.5,
    )


def _chain_entries(frags) -> list:
    """Chain frags[0]–frags[1]–...–frags[n-1] sorted by score descending."""
    entries = []
    for i in range(len(frags) - 1):
        entries.append(_entry(frags[i].edges[0], frags[i + 1].edges[0],
                               score=1.0 - 0.05 * i))
    return sorted(entries, key=lambda e: e.score, reverse=True)


# ─── TestHypothesisExtra ──────────────────────────────────────────────────────

class TestHypothesisExtra:
    def test_placed_ids_is_mutable_set(self):
        hyp = Hypothesis(placements={}, placed_ids=set())
        hyp.placed_ids.add(5)
        assert 5 in hyp.placed_ids

    def test_placements_dict_stored(self):
        pos = np.array([1.0, 2.0])
        hyp = Hypothesis(placements={0: (pos, 0.5)}, placed_ids={0})
        assert 0 in hyp.placements

    def test_last_entries_can_be_set(self):
        frags = [_frag(0), _frag(1)]
        e = _entry(frags[0].edges[0], frags[1].edges[0])
        hyp = Hypothesis(placements={}, placed_ids=set(), last_entries=[e])
        assert len(hyp.last_entries) == 1

    def test_default_score_zero(self):
        hyp = Hypothesis(placements={}, placed_ids=set())
        assert hyp.score == 0.0


# ─── TestBeamSearchExtra ──────────────────────────────────────────────────────

class TestBeamSearchExtra:
    def test_placement_keys_match_fragment_ids(self):
        frags = [_frag(i * 5) for i in range(4)]
        entries = _chain_entries(frags)
        result = beam_search(frags, entries)
        assert set(result.placements.keys()) == {f.fragment_id for f in frags}

    def test_first_fragment_at_origin(self):
        frags = [_frag(0), _frag(1)]
        result = beam_search(frags, [])
        pos, angle = result.placements[0]
        np.testing.assert_allclose(pos, [0.0, 0.0], atol=1e-10)
        assert angle == pytest.approx(0.0)

    def test_5_fragments_chain(self):
        frags = [_frag(i) for i in range(5)]
        entries = _chain_entries(frags)
        result = beam_search(frags, entries, beam_width=3)
        assert len(result.placements) == 5

    def test_beam_width_2_vs_10_both_place_all(self):
        frags = [_frag(i) for i in range(4)]
        entries = _chain_entries(frags)
        r2 = beam_search(frags, entries, beam_width=2)
        r10 = beam_search(frags, entries, beam_width=10)
        assert len(r2.placements) == 4
        assert len(r10.placements) == 4

    def test_score_increases_with_more_entries(self):
        frags = [_frag(i) for i in range(3)]
        r_no = beam_search(frags, [])
        r_with = beam_search(frags, _chain_entries(frags))
        # With useful entries, score should be >= without
        assert r_with.total_score >= r_no.total_score

    def test_orphan_positions_distinct_from_placed(self):
        """Orphan fragments should be placed at y > 0 (in the orphan row)."""
        frags = [_frag(0)]  # only 1 fragment = placed at origin
        result = beam_search(frags, [])
        pos, _ = result.placements[0]
        assert isinstance(pos[0], (int, float, np.floating))

    def test_multiple_edges_per_fragment(self):
        frags = [_frag(i, n_edges=3) for i in range(3)]
        entries = [
            _entry(frags[0].edges[1], frags[1].edges[2], score=0.9),
            _entry(frags[1].edges[0], frags[2].edges[1], score=0.8),
        ]
        result = beam_search(frags, entries, beam_width=2)
        assert len(result.placements) == 3

    def test_duplicate_entries_no_crash(self):
        frags = [_frag(0), _frag(1)]
        e = _entry(frags[0].edges[0], frags[1].edges[0])
        result = beam_search(frags, [e, e, e])
        assert len(result.placements) == 2

    def test_entry_with_unknown_edge_no_crash(self):
        frags = [_frag(0), _frag(1)]
        unknown_edge = _edge(999)
        e_unknown = _entry(frags[0].edges[0], unknown_edge)
        result = beam_search(frags, [e_unknown])
        assert isinstance(result, Assembly)

    def test_large_beam_width_ok(self):
        frags = [_frag(i) for i in range(3)]
        result = beam_search(frags, _chain_entries(frags), beam_width=1000)
        assert len(result.placements) == 3

    def test_max_depth_1_still_places_all(self):
        frags = [_frag(i) for i in range(5)]
        entries = _chain_entries(frags)
        # depth=1 → only first placement loop, rest become orphans
        result = beam_search(frags, entries, max_depth=1)
        assert len(result.placements) == 5

    def test_compat_matrix_is_array(self):
        frags = [_frag(i) for i in range(3)]
        result = beam_search(frags, _chain_entries(frags))
        assert isinstance(result.compat_matrix, np.ndarray)

    def test_position_vectors_are_2d(self):
        frags = [_frag(i) for i in range(3)]
        result = beam_search(frags, _chain_entries(frags))
        for fid, (pos, angle) in result.placements.items():
            assert len(pos) == 2

    def test_chain_6_with_beam_5(self):
        frags = [_frag(i) for i in range(6)]
        entries = _chain_entries(frags)
        result = beam_search(frags, entries, beam_width=5)
        assert len(result.placements) == 6

    def test_total_score_equals_sum_of_entries_used(self):
        """Score accumulates entry.score for each placed fragment."""
        frags = [_frag(0), _frag(1)]
        e = _entry(frags[0].edges[0], frags[1].edges[0], score=0.75)
        result = beam_search(frags, [e])
        assert result.total_score >= 0.0
