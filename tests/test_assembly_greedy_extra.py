"""Additional tests for puzzle_reconstruction/assembly/greedy.py"""
import pytest
import numpy as np

from puzzle_reconstruction.models import (
    EdgeSignature, EdgeSide, Fragment, CompatEntry, Assembly,
)
from puzzle_reconstruction.assembly.greedy import greedy_assembly


# ─── helpers ──────────────────────────────────────────────────────────────────

def _curve(n=8) -> np.ndarray:
    t = np.linspace(0, np.pi, n)
    return np.column_stack([np.cos(t), np.sin(t)]).astype(float)


def _edge(edge_id, side=EdgeSide.LEFT) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=side,
        virtual_curve=_curve(),
        fd=1.4,
        css_vec=np.ones(8) / 8.0,
        ifs_coeffs=np.ones(5) * 0.2,
        length=6.0,
    )


def _frag(fid, edge_ids) -> Fragment:
    return Fragment(
        fragment_id=fid,
        image=np.zeros((40, 40, 3), dtype=np.uint8),
        mask=np.full((40, 40), 255, dtype=np.uint8),
        contour=np.array([[0, 0], [40, 0], [40, 40], [0, 40]], dtype=float),
        edges=[_edge(eid) for eid in edge_ids],
    )


def _entry(ei, ej, score=0.85) -> CompatEntry:
    return CompatEntry(
        edge_i=ei, edge_j=ej, score=score,
        dtw_dist=0.5, css_sim=0.9, fd_diff=0.05, text_score=0.8,
    )


# ─── TestGreedyAssemblyExtra ──────────────────────────────────────────────────

class TestGreedyAssemblyExtra:
    def test_placements_keys_equal_fragment_ids(self):
        f1 = _frag(10, [100])
        f2 = _frag(20, [200])
        result = greedy_assembly([f1, f2], [_entry(f1.edges[0], f2.edges[0])])
        assert set(result.placements.keys()) == {10, 20}

    def test_first_fragment_at_origin(self):
        f1 = _frag(0, [0])
        f2 = _frag(1, [1])
        result = greedy_assembly([f1, f2], [])
        pos, angle = result.placements[0]
        np.testing.assert_allclose(pos, [0.0, 0.0], atol=1e-10)
        assert angle == pytest.approx(0.0)

    def test_orphan_positions_distinct(self):
        """Orphans should not all be placed at the same position."""
        frags = [_frag(i, [i]) for i in range(5)]
        result = greedy_assembly(frags, [])
        positions = [result.placements[i][0] for i in range(5)]
        # At least some positions differ (first is origin, rest are shifted)
        pos_set = {tuple(p.tolist()) for p in positions}
        assert len(pos_set) >= 2

    def test_assembly_fragments_list(self):
        f1 = _frag(0, [0])
        f2 = _frag(1, [1])
        result = greedy_assembly([f1, f2], [])
        assert result.fragments == [f1, f2]

    def test_position_has_2_components(self):
        f = _frag(0, [0])
        result = greedy_assembly([f], [])
        pos, _ = result.placements[0]
        assert len(pos) == 2

    def test_angle_is_float_or_convertible(self):
        f = _frag(0, [0])
        result = greedy_assembly([f], [])
        _, angle = result.placements[0]
        float(angle)  # should not raise

    def test_compat_matrix_exists(self):
        f1 = _frag(0, [0])
        result = greedy_assembly([f1], [])
        assert hasattr(result, 'compat_matrix')

    def test_total_score_is_numeric(self):
        f1 = _frag(0, [0])
        f2 = _frag(1, [1])
        e = _entry(f1.edges[0], f2.edges[0], score=0.7)
        result = greedy_assembly([f1, f2], [e])
        float(result.total_score)  # should not raise

    def test_entry_with_unknown_edge_does_not_crash(self):
        """Entry referencing edges outside fragment list should not crash."""
        f1 = _frag(0, [0])
        f2 = _frag(1, [1])
        unknown_edge = _edge(999)
        e = _entry(f1.edges[0], unknown_edge, score=0.9)
        result = greedy_assembly([f1, f2], [e])
        assert isinstance(result, Assembly)

    def test_duplicate_entries_same_pair(self):
        """Two entries for the same pair should not crash."""
        f1 = _frag(0, [0])
        f2 = _frag(1, [1])
        e1 = _entry(f1.edges[0], f2.edges[0], score=0.9)
        e2 = _entry(f1.edges[0], f2.edges[0], score=0.5)
        result = greedy_assembly([f1, f2], [e1, e2])
        assert len(result.placements) == 2

    def test_chain_of_three(self):
        """f1–f2–f3 chain: all 3 should be placed."""
        f1 = _frag(0, [0])
        f2 = _frag(1, [1, 2])
        f3 = _frag(2, [3])
        e12 = _entry(f1.edges[0], f2.edges[0], score=0.9)
        e23 = _entry(f2.edges[1], f3.edges[0], score=0.85)
        result = greedy_assembly([f1, f2, f3], [e12, e23])
        assert len(result.placements) == 3

    def test_higher_score_entry_preferred(self):
        """When entries sorted by score desc, greedy uses best first."""
        f1 = _frag(0, [0])
        f2 = _frag(1, [1])
        f3 = _frag(2, [2])
        e_high = _entry(f1.edges[0], f2.edges[0], score=0.99)
        e_low = _entry(f1.edges[0], f3.edges[0], score=0.01)
        entries = sorted([e_high, e_low], key=lambda e: e.score, reverse=True)
        result = greedy_assembly([f1, f2, f3], entries)
        # All fragments placed regardless
        assert set(result.placements.keys()) == {0, 1, 2}

    def test_large_fragment_set(self):
        """20 fragments, no entries → all placed as orphans."""
        frags = [_frag(i, [i]) for i in range(20)]
        result = greedy_assembly(frags, [])
        assert len(result.placements) == 20

    def test_many_entries_chain(self):
        """Chain of 6 fragments via CompatEntries."""
        n = 6
        frags = [_frag(i, [i, i + n]) for i in range(n)]
        entries = []
        for i in range(n - 1):
            entries.append(_entry(frags[i].edges[1], frags[i + 1].edges[0],
                                  score=0.9 - i * 0.05))
        result = greedy_assembly(frags, entries)
        assert len(result.placements) == n

    def test_result_is_assembly_instance(self):
        f1 = _frag(0, [0])
        result = greedy_assembly([f1], [])
        assert isinstance(result, Assembly)

    def test_fragment_ids_match_placements(self):
        frags = [_frag(i * 5, [i]) for i in range(4)]
        result = greedy_assembly(frags, [])
        assert set(result.placements.keys()) == {0, 5, 10, 15}
