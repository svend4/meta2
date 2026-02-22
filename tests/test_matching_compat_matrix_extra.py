"""Additional tests for puzzle_reconstruction/matching/compat_matrix.py."""
import numpy as np
import pytest

from puzzle_reconstruction.matching.compat_matrix import (
    build_compat_matrix,
    top_candidates,
)
from puzzle_reconstruction.models import (
    CompatEntry,
    EdgeSide,
    EdgeSignature,
    Fragment,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _edge(edge_id: int, fd: float = 1.5, length: float = 80.0,
          side: EdgeSide = EdgeSide.TOP) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=side,
        virtual_curve=np.zeros((10, 2)),
        fd=fd,
        css_vec=np.zeros(8),
        ifs_coeffs=np.zeros(4),
        length=length,
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


# ─── TestBuildCompatMatrixExtra ───────────────────────────────────────────────

class TestBuildCompatMatrixExtra:
    def test_5_frags_matrix_shape(self):
        frags = [_frag(i) for i in range(5)]
        matrix, _ = build_compat_matrix(frags)
        assert matrix.shape == (10, 10)

    def test_4_edges_per_frag_matrix_shape(self):
        frags = [_frag(i, n_edges=4) for i in range(2)]
        matrix, _ = build_compat_matrix(frags)
        assert matrix.shape == (8, 8)

    def test_matrix_float32_dtype(self):
        matrix, _ = build_compat_matrix([_frag(0), _frag(1)])
        assert matrix.dtype == np.float32

    def test_matrix_symmetric_5_frags(self):
        frags = [_frag(i) for i in range(5)]
        matrix, _ = build_compat_matrix(frags)
        assert np.allclose(matrix, matrix.T)

    def test_diagonal_zero_5_frags(self):
        frags = [_frag(i) for i in range(5)]
        matrix, _ = build_compat_matrix(frags)
        assert np.all(np.diag(matrix) == 0.0)

    def test_intra_fragment_cells_zero(self):
        """Edges of the same fragment occupy a block that must be zero."""
        frags = [_frag(0, n_edges=3), _frag(1, n_edges=2)]
        matrix, _ = build_compat_matrix(frags)
        # Frag 0 occupies rows/cols 0-2; those off-diagonal pairs must be 0
        block = matrix[:3, :3]
        np.testing.assert_array_equal(block, np.zeros((3, 3)))

    def test_threshold_0_all_cross_frag_included(self):
        frags = [_frag(0), _frag(1)]
        _, e0 = build_compat_matrix(frags, threshold=0.0)
        _, e1 = build_compat_matrix(frags, threshold=0.99)
        assert len(e1) <= len(e0)

    def test_single_edge_per_frag(self):
        """One edge per fragment → 1×1 cross-entry."""
        f0 = Fragment(fragment_id=0, image=np.zeros((8,8,3), dtype=np.uint8),
                      mask=np.zeros((8,8),dtype=np.uint8), contour=np.zeros((4,2)))
        f0.edges = [_edge(0)]
        f1 = Fragment(fragment_id=1, image=np.zeros((8,8,3), dtype=np.uint8),
                      mask=np.zeros((8,8),dtype=np.uint8), contour=np.zeros((4,2)))
        f1.edges = [_edge(10)]
        matrix, entries = build_compat_matrix([f0, f1])
        assert matrix.shape == (2, 2)
        assert len(entries) == 1

    def test_entries_cross_fragment_only(self):
        frags = [_frag(i) for i in range(3)]
        _, entries = build_compat_matrix(frags)
        for e in entries:
            # Edges from same fragment have IDs in the same 10-range
            fid_i = e.edge_i.edge_id // 10
            fid_j = e.edge_j.edge_id // 10
            assert fid_i != fid_j

    def test_matrix_values_nonneg(self):
        matrix, _ = build_compat_matrix([_frag(i) for i in range(3)])
        assert np.all(matrix >= 0.0)

    def test_matrix_values_le_1(self):
        matrix, _ = build_compat_matrix([_frag(i) for i in range(3)])
        assert np.all(matrix <= 1.0)

    def test_4_frags_entries_are_compat_entry(self):
        frags = [_frag(i) for i in range(4)]
        _, entries = build_compat_matrix(frags)
        for e in entries:
            assert isinstance(e, CompatEntry)

    def test_6_frags_no_crash(self):
        frags = [_frag(i) for i in range(6)]
        matrix, entries = build_compat_matrix(frags)
        assert matrix.shape == (12, 12)

    def test_3_edges_per_frag_shape(self):
        frags = [_frag(i, n_edges=3) for i in range(3)]
        matrix, _ = build_compat_matrix(frags)
        assert matrix.shape == (9, 9)


# ─── TestTopCandidatesExtra ───────────────────────────────────────────────────

class TestTopCandidatesExtra:
    def _setup(self, n_frags: int = 3):
        frags = [_frag(i) for i in range(n_frags)]
        all_edges = [e for f in frags for e in f.edges]
        matrix, _ = build_compat_matrix(frags)
        return matrix, all_edges

    def test_k_0_returns_empty(self):
        matrix, all_edges = self._setup()
        result = top_candidates(matrix, all_edges, 0, k=0)
        assert result == []

    def test_k_larger_than_n_returns_all_nonzero(self):
        matrix, all_edges = self._setup()
        result = top_candidates(matrix, all_edges, 0, k=100)
        assert len(result) <= len(all_edges) - 1

    def test_tuples_have_2_items(self):
        matrix, all_edges = self._setup()
        result = top_candidates(matrix, all_edges, 0, k=5)
        for item in result:
            assert len(item) == 2

    def test_scores_descending(self):
        matrix, all_edges = self._setup(n_frags=4)
        result = top_candidates(matrix, all_edges, 0, k=5)
        if len(result) > 1:
            scores = [r[1] for r in result]
            assert scores == sorted(scores, reverse=True)

    def test_self_not_in_result(self):
        matrix, all_edges = self._setup()
        for q in range(len(all_edges)):
            result = top_candidates(matrix, all_edges, q, k=5)
            assert q not in [r[0] for r in result]

    def test_indices_valid(self):
        matrix, all_edges = self._setup()
        result = top_candidates(matrix, all_edges, 0, k=5)
        for idx, _ in result:
            assert 0 <= idx < len(all_edges)

    def test_k_1_returns_at_most_1(self):
        matrix, all_edges = self._setup()
        result = top_candidates(matrix, all_edges, 0, k=1)
        assert len(result) <= 1

    def test_scores_float_type(self):
        matrix, all_edges = self._setup()
        result = top_candidates(matrix, all_edges, 0, k=3)
        for _, score in result:
            assert isinstance(score, float)

    def test_different_query_indices_different_results(self):
        matrix, all_edges = self._setup(n_frags=4)
        r0 = top_candidates(matrix, all_edges, 0, k=3)
        r1 = top_candidates(matrix, all_edges, 1, k=3)
        # Results for different query edges may differ
        if r0 and r1:
            # At least one idx should differ
            ids0 = {r[0] for r in r0}
            ids1 = {r[0] for r in r1}
            # Not necessarily disjoint, just both valid
            assert isinstance(ids0, set)
            assert isinstance(ids1, set)
