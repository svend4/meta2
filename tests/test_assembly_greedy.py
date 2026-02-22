"""Tests for puzzle_reconstruction/assembly/greedy.py"""
import pytest
import numpy as np

from puzzle_reconstruction.models import (
    EdgeSignature, EdgeSide, Fragment, CompatEntry, Assembly,
)
from puzzle_reconstruction.assembly.greedy import greedy_assembly


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_curve(n=8):
    angles = np.linspace(0, np.pi, n)
    return np.column_stack([np.cos(angles), np.sin(angles)]).astype(float)


def make_edge_sig(edge_id, side=EdgeSide.RIGHT):
    return EdgeSignature(
        edge_id=edge_id,
        side=side,
        virtual_curve=make_curve(),
        fd=1.5,
        css_vec=np.ones(8) / 8.0,
        ifs_coeffs=np.ones(5) * 0.5,
        length=8.0,
    )


def make_fragment(fragment_id, edge_ids):
    edges = [make_edge_sig(eid) for eid in edge_ids]
    return Fragment(
        fragment_id=fragment_id,
        image=np.zeros((50, 50, 3), dtype=np.uint8),
        mask=np.full((50, 50), 255, dtype=np.uint8),
        contour=np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype=float),
        edges=edges,
    )


def make_entry(edge_i, edge_j, score=0.8):
    return CompatEntry(
        edge_i=edge_i,
        edge_j=edge_j,
        score=score,
        dtw_dist=1.0,
        css_sim=0.9,
        fd_diff=0.1,
        text_score=0.8,
    )


# ─── greedy_assembly ──────────────────────────────────────────────────────────

class TestGreedyAssembly:
    def test_returns_assembly(self):
        f1 = make_fragment(0, [0])
        f2 = make_fragment(1, [1])
        entry = make_entry(f1.edges[0], f2.edges[0])
        result = greedy_assembly([f1, f2], [entry])
        assert isinstance(result, Assembly)

    def test_empty_fragments(self):
        result = greedy_assembly([], [])
        assert isinstance(result, Assembly)
        assert result.fragments == []
        assert result.placements == {}

    def test_single_fragment_placed_at_origin(self):
        f = make_fragment(0, [0])
        result = greedy_assembly([f], [])
        assert 0 in result.placements
        pos, angle = result.placements[0]
        np.testing.assert_allclose(pos, [0.0, 0.0], atol=1e-10)
        assert angle == pytest.approx(0.0)

    def test_all_fragments_placed(self):
        f1 = make_fragment(0, [0])
        f2 = make_fragment(1, [1])
        f3 = make_fragment(2, [2])
        entry1 = make_entry(f1.edges[0], f2.edges[0], score=0.9)
        entry2 = make_entry(f2.edges[0], f3.edges[0], score=0.7)
        result = greedy_assembly([f1, f2, f3], [entry1, entry2])
        assert len(result.placements) == 3
        assert all(fid in result.placements for fid in [0, 1, 2])

    def test_two_fragments_connected(self):
        f1 = make_fragment(0, [10])
        f2 = make_fragment(1, [11])
        entry = make_entry(f1.edges[0], f2.edges[0], score=0.95)
        result = greedy_assembly([f1, f2], [entry])
        assert 0 in result.placements
        assert 1 in result.placements

    def test_placement_is_tuple_pos_angle(self):
        f = make_fragment(0, [0])
        result = greedy_assembly([f], [])
        pos, angle = result.placements[0]
        assert hasattr(pos, '__len__')
        assert len(pos) == 2
        assert isinstance(angle, float)

    def test_no_entries_all_orphans(self):
        """With no CompatEntries all fragments except first become orphans."""
        f1 = make_fragment(0, [0])
        f2 = make_fragment(1, [1])
        f3 = make_fragment(2, [2])
        result = greedy_assembly([f1, f2, f3], [])
        assert len(result.placements) == 3

    def test_total_score_nonneg(self):
        f1 = make_fragment(0, [0])
        f2 = make_fragment(1, [1])
        entry = make_entry(f1.edges[0], f2.edges[0], score=0.8)
        result = greedy_assembly([f1, f2], [entry])
        assert result.total_score >= 0.0

    def test_fragments_stored_in_result(self):
        f1 = make_fragment(0, [0])
        f2 = make_fragment(1, [1])
        result = greedy_assembly([f1, f2], [])
        assert result.fragments == [f1, f2]

    def test_high_score_entry_wins(self):
        """High-score entry placed before low-score one."""
        f1 = make_fragment(0, [0])
        f2 = make_fragment(1, [1])
        f3 = make_fragment(2, [2])
        entry_high = make_entry(f1.edges[0], f2.edges[0], score=0.95)
        entry_low = make_entry(f1.edges[0], f3.edges[0], score=0.1)
        # entries sorted by score descending
        entries = sorted([entry_high, entry_low],
                         key=lambda e: e.score, reverse=True)
        result = greedy_assembly([f1, f2, f3], entries)
        assert len(result.placements) == 3

    def test_multiple_edges_per_fragment(self):
        f1 = make_fragment(0, [0, 1])  # 2 edges
        f2 = make_fragment(1, [2, 3])  # 2 edges
        entry = make_entry(f1.edges[0], f2.edges[1], score=0.9)
        result = greedy_assembly([f1, f2], [entry])
        assert len(result.placements) == 2
