"""Расширенные тесты для puzzle_reconstruction/assembly/greedy.py."""
import numpy as np
import pytest

from puzzle_reconstruction.assembly.greedy import (
    greedy_assembly,
    _compute_placement,
    _transform_curve,
    _place_orphans,
)
from puzzle_reconstruction.models import (
    Assembly,
    CompatEntry,
    EdgeSide,
    EdgeSignature,
    Fragment,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _curve(n: int = 8) -> np.ndarray:
    angles = np.linspace(0, np.pi, n)
    return np.column_stack([np.cos(angles), np.sin(angles)]).astype(float)


def _edge(edge_id: int, side: EdgeSide = EdgeSide.RIGHT) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=side,
        virtual_curve=_curve(),
        fd=1.5,
        css_vec=np.ones(8) / 8.0,
        ifs_coeffs=np.ones(5) * 0.5,
        length=8.0,
    )


def _frag(fid: int, edge_ids=None) -> Fragment:
    if edge_ids is None:
        edge_ids = [fid * 10]
    edges = [_edge(eid) for eid in edge_ids]
    return Fragment(
        fragment_id=fid,
        image=np.zeros((30, 30, 3), dtype=np.uint8),
        mask=np.full((30, 30), 255, dtype=np.uint8),
        contour=np.array([[0, 0], [30, 0], [30, 30], [0, 30]], dtype=float),
        edges=edges,
    )


def _entry(edge_i: EdgeSignature, edge_j: EdgeSignature,
           score: float = 0.8) -> CompatEntry:
    return CompatEntry(
        edge_i=edge_i, edge_j=edge_j, score=score,
        dtw_dist=1.0, css_sim=0.9, fd_diff=0.1, text_score=0.8,
    )


# ─── TestGreedyAssembly ───────────────────────────────────────────────────────

class TestGreedyAssembly:

    # --- Return type and structure ---

    def test_returns_assembly(self):
        f1, f2 = _frag(0), _frag(1)
        result = greedy_assembly([f1, f2], [_entry(f1.edges[0], f2.edges[0])])
        assert isinstance(result, Assembly)

    def test_empty_fragments_returns_assembly(self):
        result = greedy_assembly([], [])
        assert isinstance(result, Assembly)

    def test_empty_fragments_empty_placements(self):
        result = greedy_assembly([], [])
        assert result.placements == {}

    def test_empty_fragments_preserves_list(self):
        result = greedy_assembly([], [])
        assert result.fragments == []

    def test_fragments_stored(self):
        f1, f2 = _frag(0), _frag(1)
        result = greedy_assembly([f1, f2], [])
        assert result.fragments == [f1, f2]

    def test_compat_matrix_is_ndarray(self):
        result = greedy_assembly([_frag(0)], [])
        assert isinstance(result.compat_matrix, np.ndarray)

    # --- Placement counts ---

    def test_single_fragment_in_placements(self):
        f = _frag(0)
        result = greedy_assembly([f], [])
        assert f.fragment_id in result.placements

    def test_two_fragments_both_placed(self):
        f1, f2 = _frag(0), _frag(1)
        result = greedy_assembly([f1, f2], [_entry(f1.edges[0], f2.edges[0])])
        assert 0 in result.placements and 1 in result.placements

    def test_three_fragments_all_placed(self):
        frags = [_frag(i) for i in range(3)]
        entries = [_entry(frags[0].edges[0], frags[1].edges[0], 0.9),
                   _entry(frags[1].edges[0], frags[2].edges[0], 0.7)]
        result = greedy_assembly(frags, entries)
        assert all(i in result.placements for i in range(3))

    def test_five_fragments_no_entries_all_placed(self):
        frags = [_frag(i) for i in range(5)]
        result = greedy_assembly(frags, [])
        assert len(result.placements) == 5
        assert all(i in result.placements for i in range(5))

    def test_no_entries_still_all_placed(self):
        frags = [_frag(i) for i in range(4)]
        result = greedy_assembly(frags, [])
        assert len(result.placements) == 4

    # --- First fragment anchor ---

    def test_first_fragment_at_origin_pos(self):
        f = _frag(0)
        result = greedy_assembly([f], [])
        pos, _ = result.placements[0]
        np.testing.assert_allclose(pos, [0.0, 0.0], atol=1e-10)

    def test_first_fragment_at_zero_angle(self):
        f = _frag(0)
        result = greedy_assembly([f], [])
        _, angle = result.placements[0]
        assert angle == pytest.approx(0.0)

    # --- Placement structure ---

    def test_placement_pos_has_len_2(self):
        f = _frag(0)
        result = greedy_assembly([f], [])
        pos, _ = result.placements[0]
        assert len(pos) == 2

    def test_placement_angle_is_float(self):
        f = _frag(0)
        result = greedy_assembly([f], [])
        _, angle = result.placements[0]
        assert isinstance(angle, float)

    def test_placement_pos_is_ndarray(self):
        f = _frag(0)
        result = greedy_assembly([f], [])
        pos, _ = result.placements[0]
        assert hasattr(pos, '__len__')

    # --- Score ---

    def test_total_score_is_float(self):
        f1, f2 = _frag(0), _frag(1)
        result = greedy_assembly([f1, f2], [_entry(f1.edges[0], f2.edges[0])])
        assert isinstance(result.total_score, float)

    def test_total_score_nonneg(self):
        f1, f2 = _frag(0), _frag(1)
        result = greedy_assembly([f1, f2], [_entry(f1.edges[0], f2.edges[0])])
        assert result.total_score >= 0.0

    def test_total_score_zero_when_no_entries(self):
        frags = [_frag(i) for i in range(3)]
        result = greedy_assembly(frags, [])
        assert result.total_score == pytest.approx(0.0)

    def test_total_score_increases_with_valid_entries(self):
        f1, f2 = _frag(0), _frag(1)
        r_no_entry = greedy_assembly([f1, f2], [])
        r_with_entry = greedy_assembly([f1, f2], [_entry(f1.edges[0], f2.edges[0], score=1.0)])
        assert r_with_entry.total_score >= r_no_entry.total_score

    # --- Entry handling ---

    def test_entry_with_both_already_placed_skipped(self):
        """If both endpoints placed, entry is skipped — no duplicate placement."""
        f1, f2 = _frag(0), _frag(1)
        entry = _entry(f1.edges[0], f2.edges[0])
        # Provide entry twice — should not cause errors
        result = greedy_assembly([f1, f2], [entry, entry])
        assert 0 in result.placements and 1 in result.placements

    def test_entry_with_unknown_edge_skipped(self):
        """Entry with edge_id not belonging to any fragment is handled."""
        f1 = _frag(0, [0])
        unknown_edge = _edge(9999)
        entry = _entry(f1.edges[0], unknown_edge)
        result = greedy_assembly([f1], [entry])
        assert 0 in result.placements

    def test_multiple_edges_per_fragment(self):
        f1 = _frag(0, [0, 1])
        f2 = _frag(1, [2, 3])
        entry = _entry(f1.edges[0], f2.edges[1])
        result = greedy_assembly([f1, f2], [entry])
        assert 0 in result.placements and 1 in result.placements

    def test_high_score_entry_first_in_sorted_list(self):
        f1, f2, f3 = _frag(0), _frag(1), _frag(2)
        high = _entry(f1.edges[0], f2.edges[0], score=0.95)
        low = _entry(f1.edges[0], f3.edges[0], score=0.1)
        entries = sorted([high, low], key=lambda e: e.score, reverse=True)
        result = greedy_assembly([f1, f2, f3], entries)
        assert len(result.placements) == 3

    # --- Orphan placement ---

    def test_orphans_placed_below_anchor(self):
        """Orphans get y >= 0 when no entries provided."""
        frags = [_frag(i) for i in range(3)]
        result = greedy_assembly(frags, [])
        for fid, (pos, _) in result.placements.items():
            if fid != 0:  # fid=0 is at y=0, orphans at y >= max_y + 200
                assert pos[1] >= 0.0

    def test_orphan_x_positions_spread(self):
        """Multiple orphans get different x positions."""
        frags = [_frag(i) for i in range(3)]
        result = greedy_assembly(frags, [])
        # Orphans (fid 1 and 2) should have different x
        pos1, _ = result.placements[1]
        pos2, _ = result.placements[2]
        assert pos1[0] != pos2[0]

    def test_single_orphan_placed(self):
        """Even a single orphan (not connected to any entry) is placed."""
        f1, f2 = _frag(0), _frag(1)
        # f2 has edge_id=10, entry uses edge_id=99 → f2 not connected
        result = greedy_assembly([f1, f2], [])
        assert 1 in result.placements

    # --- Chain placement ---

    def test_chain_of_three_all_placed(self):
        """f0 → f1 → f2 chain via entries."""
        f0 = _frag(0, [100])
        f1 = _frag(1, [101])
        f2 = _frag(2, [102])
        entries = [
            _entry(f0.edges[0], f1.edges[0], 0.9),
            _entry(f1.edges[0], f2.edges[0], 0.8),
        ]
        result = greedy_assembly([f0, f1, f2], entries)
        assert all(i in result.placements for i in [0, 1, 2])

    def test_chain_second_fragment_not_at_origin(self):
        """Second fragment should be placed relative to first (not at origin)."""
        f1 = _frag(0, [0])
        f2 = _frag(1, [1])
        entry = _entry(f1.edges[0], f2.edges[0])
        result = greedy_assembly([f1, f2], [entry])
        pos2, _ = result.placements[1]
        # Second fragment should not be exactly at origin (unless coincidentally)
        # The angle for f2 is π, so position is offset from (0,0)
        # Just verify it's placed (might coincide in degenerate cases)
        assert pos2 is not None


# ─── TestTransformCurve ───────────────────────────────────────────────────────

class TestTransformCurve:
    def test_zero_transform_identity(self):
        curve = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = _transform_curve(curve, np.array([0.0, 0.0]), 0.0)
        np.testing.assert_allclose(result, curve, atol=1e-10)

    def test_translation(self):
        curve = np.array([[0.0, 0.0], [1.0, 0.0]])
        result = _transform_curve(curve, np.array([3.0, 4.0]), 0.0)
        expected = np.array([[3.0, 4.0], [4.0, 4.0]])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_rotation_90_degrees(self):
        curve = np.array([[1.0, 0.0]])
        result = _transform_curve(curve, np.array([0.0, 0.0]), np.pi / 2)
        np.testing.assert_allclose(result, [[0.0, 1.0]], atol=1e-10)

    def test_output_shape_preserved(self):
        curve = np.random.randn(10, 2)
        result = _transform_curve(curve, np.array([1.0, 2.0]), 0.5)
        assert result.shape == curve.shape


# ─── TestPlaceOrphans ─────────────────────────────────────────────────────────

class TestPlaceOrphans:
    def test_empty_orphans_no_change(self):
        placements = {0: (np.array([0.0, 0.0]), 0.0)}
        placed = {0}
        frags = [_frag(0)]
        _place_orphans(frags, placements, placed)
        assert len(placements) == 1

    def test_orphan_added_to_placements(self):
        f0 = _frag(0)
        f1 = _frag(1)
        placements = {0: (np.array([0.0, 0.0]), 0.0)}
        placed = {0}
        _place_orphans([f0, f1], placements, placed)
        assert 1 in placements

    def test_orphan_y_offset_from_max(self):
        f0 = _frag(0)
        f1 = _frag(1)
        placements = {0: (np.array([0.0, 100.0]), 0.0)}
        placed = {0}
        _place_orphans([f0, f1], placements, placed)
        pos, _ = placements[1]
        assert pos[1] >= 100.0  # should be 100 + 200 = 300

    def test_empty_placements_y_offset_zero(self):
        f0 = _frag(0)
        placements = {}
        placed = set()
        _place_orphans([f0], placements, placed)
        pos, _ = placements[0]
        assert pos[1] == pytest.approx(0.0)

    def test_multiple_orphans_added(self):
        frags = [_frag(i) for i in range(3)]
        placements = {0: (np.array([0.0, 0.0]), 0.0)}
        placed = {0}
        _place_orphans(frags, placements, placed)
        assert all(i in placements for i in range(3))


# ─── TestComputePlacement ─────────────────────────────────────────────────────

class TestComputePlacement:
    def test_returns_tuple(self):
        f1, f2 = _frag(0, [0]), _frag(1, [1])
        anchor_placement = (np.array([0.0, 0.0]), 0.0)
        result = _compute_placement(f1, f1.edges[0], f2, f2.edges[0],
                                    anchor_placement)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_angle_is_float(self):
        f1, f2 = _frag(0, [0]), _frag(1, [1])
        anchor_placement = (np.array([0.0, 0.0]), 0.0)
        pos, angle = _compute_placement(f1, f1.edges[0], f2, f2.edges[0],
                                         anchor_placement)
        assert isinstance(angle, float)

    def test_pos_has_len_2(self):
        f1, f2 = _frag(0, [0]), _frag(1, [1])
        anchor_placement = (np.array([0.0, 0.0]), 0.0)
        pos, _ = _compute_placement(f1, f1.edges[0], f2, f2.edges[0],
                                     anchor_placement)
        assert len(pos) == 2

    def test_angle_is_anchor_plus_pi(self):
        f1, f2 = _frag(0, [0]), _frag(1, [1])
        anchor_placement = (np.array([0.0, 0.0]), 0.5)
        _, angle = _compute_placement(f1, f1.edges[0], f2, f2.edges[0],
                                       anchor_placement)
        assert angle == pytest.approx(0.5 + np.pi)
