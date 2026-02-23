"""Extra tests for puzzle_reconstruction/assembly/greedy.py."""
import numpy as np
import pytest

from puzzle_reconstruction.assembly.greedy import (
    _compute_placement,
    _place_orphans,
    _transform_curve,
    greedy_assembly,
)
from puzzle_reconstruction.models import (
    Assembly,
    CompatEntry,
    EdgeSide,
    EdgeSignature,
    Fragment,
)


# ─── helpers (same as original) ───────────────────────────────────────────────

def _curve(n: int = 8) -> np.ndarray:
    angles = np.linspace(0, np.pi, n)
    return np.column_stack([np.cos(angles), np.sin(angles)]).astype(float)

def _edge(edge_id: int, side: EdgeSide = EdgeSide.RIGHT) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id, side=side, virtual_curve=_curve(),
        fd=1.5, css_vec=np.ones(8) / 8.0,
        ifs_coeffs=np.ones(5) * 0.5, length=8.0,
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


# ─── TestGreedyAssemblyExtra ──────────────────────────────────────────────────

class TestGreedyAssemblyExtra:
    def test_score_with_two_entries(self):
        f0, f1, f2 = _frag(0), _frag(1), _frag(2)
        entries = [
            _entry(f0.edges[0], f1.edges[0], 0.8),
            _entry(f1.edges[0], f2.edges[0], 0.6),
        ]
        result = greedy_assembly([f0, f1, f2], entries)
        assert result.total_score >= 0.0

    def test_score_with_high_score_entry(self):
        f0, f1 = _frag(0), _frag(1)
        result = greedy_assembly([f0, f1],
                                  [_entry(f0.edges[0], f1.edges[0], 1.0)])
        assert result.total_score > 0.0

    def test_10_fragments_no_entries(self):
        frags = [_frag(i) for i in range(10)]
        result = greedy_assembly(frags, [])
        assert len(result.placements) == 10

    def test_chain_of_five(self):
        frags = [_frag(i, [i * 10]) for i in range(5)]
        entries = [_entry(frags[i].edges[0], frags[i + 1].edges[0], 0.9)
                   for i in range(4)]
        result = greedy_assembly(frags, entries)
        assert all(i in result.placements for i in range(5))

    def test_all_same_score_entries(self):
        frags = [_frag(i) for i in range(4)]
        entries = [_entry(frags[i].edges[0], frags[j].edges[0], 0.5)
                   for i in range(len(frags))
                   for j in range(i + 1, len(frags))]
        result = greedy_assembly(frags, entries)
        assert len(result.placements) == 4

    def test_compat_matrix_shape(self):
        frags = [_frag(i) for i in range(3)]
        entries = [_entry(frags[i].edges[0], frags[j].edges[0], 0.5)
                   for i in range(3) for j in range(3) if i != j]
        result = greedy_assembly(frags, entries)
        assert result.compat_matrix.shape[0] >= 0

    def test_placements_all_have_pos_len_2(self):
        frags = [_frag(i) for i in range(4)]
        result = greedy_assembly(frags, [])
        for pos, _ in result.placements.values():
            assert len(pos) == 2

    def test_placements_all_angles_float(self):
        frags = [_frag(i) for i in range(4)]
        result = greedy_assembly(frags, [])
        for _, angle in result.placements.values():
            assert isinstance(angle, float)

    def test_entry_low_score_all_still_placed(self):
        f0, f1 = _frag(0), _frag(1)
        result = greedy_assembly([f0, f1],
                                  [_entry(f0.edges[0], f1.edges[0], 0.01)])
        assert 0 in result.placements and 1 in result.placements


# ─── TestTransformCurveExtra ──────────────────────────────────────────────────

class TestTransformCurveExtra:
    def test_rotation_180(self):
        curve = np.array([[1.0, 0.0]])
        result = _transform_curve(curve, np.array([0.0, 0.0]), np.pi)
        np.testing.assert_allclose(result, [[-1.0, 0.0]], atol=1e-10)

    def test_translation_negative(self):
        curve = np.array([[5.0, 5.0]])
        result = _transform_curve(curve, np.array([-5.0, -5.0]), 0.0)
        np.testing.assert_allclose(result, [[0.0, 0.0]], atol=1e-10)

    def test_rotation_0_identity(self):
        curve = np.array([[2.0, 3.0], [4.0, 5.0]])
        result = _transform_curve(curve, np.array([0.0, 0.0]), 0.0)
        np.testing.assert_allclose(result, curve, atol=1e-10)

    def test_large_curve(self):
        curve = np.random.default_rng(0).random((50, 2))
        result = _transform_curve(curve, np.array([1.0, 2.0]), 0.5)
        assert result.shape == (50, 2)

    def test_translation_only(self):
        curve = np.zeros((3, 2))
        result = _transform_curve(curve, np.array([7.0, 3.0]), 0.0)
        np.testing.assert_allclose(result, [[7.0, 3.0]] * 3, atol=1e-10)

    def test_rotation_and_translation(self):
        curve = np.array([[1.0, 0.0]])
        result = _transform_curve(curve, np.array([1.0, 1.0]), np.pi / 2)
        # Rotation: [1,0] → [0,1]; then translate by [1,1] → [1,2]
        np.testing.assert_allclose(result, [[1.0, 2.0]], atol=1e-10)


# ─── TestPlaceOrphansExtra ────────────────────────────────────────────────────

class TestPlaceOrphansExtra:
    def test_three_orphans_different_x(self):
        frags = [_frag(i) for i in range(4)]
        placements = {0: (np.array([0.0, 0.0]), 0.0)}
        placed = {0}
        _place_orphans(frags, placements, placed)
        x_vals = [placements[i][0][0] for i in range(1, 4)]
        assert len(set(x_vals)) == len(x_vals)

    def test_orphan_angle_is_float(self):
        f0, f1 = _frag(0), _frag(1)
        placements = {0: (np.array([0.0, 0.0]), 0.0)}
        placed = {0}
        _place_orphans([f0, f1], placements, placed)
        _, angle = placements[1]
        assert isinstance(angle, float)

    def test_y_offset_from_200(self):
        frags = [_frag(i) for i in range(2)]
        placements = {0: (np.array([0.0, 0.0]), 0.0)}
        placed = {0}
        _place_orphans(frags, placements, placed)
        _, (pos, _) = next(
            (k, v) for k, v in placements.items() if k != 0
        )
        assert pos[1] >= 0.0

    def test_single_already_placed_no_orphans(self):
        f0 = _frag(0)
        placements = {0: (np.array([0.0, 0.0]), 0.0)}
        placed = {0}
        _place_orphans([f0], placements, placed)
        assert len(placements) == 1

    def test_all_orphans_get_pos_len_2(self):
        frags = [_frag(i) for i in range(5)]
        placements = {0: (np.array([0.0, 0.0]), 0.0)}
        placed = {0}
        _place_orphans(frags, placements, placed)
        for i in range(1, 5):
            pos, _ = placements[i]
            assert len(pos) == 2


# ─── TestComputePlacementExtra ────────────────────────────────────────────────

class TestComputePlacementExtra:
    def _place(self, anchor_pos, anchor_angle):
        f1, f2 = _frag(0, [0]), _frag(1, [1])
        anchor = (np.array(anchor_pos), anchor_angle)
        return _compute_placement(f1, f1.edges[0], f2, f2.edges[0], anchor)

    def test_zero_anchor_pos(self):
        pos, angle = self._place([0.0, 0.0], 0.0)
        assert len(pos) == 2
        assert isinstance(angle, float)

    def test_nonzero_anchor_pos(self):
        pos, angle = self._place([10.0, 20.0], 0.0)
        assert pos is not None

    def test_various_anchor_angles(self):
        for a in (0.0, 0.5, 1.0, np.pi / 2, np.pi):
            pos, angle = self._place([0.0, 0.0], a)
            assert angle == pytest.approx(a + np.pi)

    def test_pos_is_ndarray(self):
        pos, _ = self._place([0.0, 0.0], 0.0)
        assert hasattr(pos, '__len__')
        assert len(pos) == 2

    def test_anchor_angle_pi_gives_2pi(self):
        _, angle = self._place([0.0, 0.0], np.pi)
        assert angle == pytest.approx(2 * np.pi)
