"""Расширенные тесты для puzzle_reconstruction/matching/compat_matrix.py."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _edge(edge_id: int, fd: float = 1.5, length: float = 80.0) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=np.zeros((10, 2)),
        fd=fd,
        css_vec=np.zeros(8),
        ifs_coeffs=np.zeros(4),
        length=length,
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


# ─── TestBuildCompatMatrix ────────────────────────────────────────────────────

class TestBuildCompatMatrix:
    def test_returns_tuple_of_2(self):
        frags = [_frag(0), _frag(1)]
        result = build_compat_matrix(frags)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_matrix_is_ndarray(self):
        matrix, _ = build_compat_matrix([_frag(0), _frag(1)])
        assert isinstance(matrix, np.ndarray)

    def test_entries_is_list(self):
        _, entries = build_compat_matrix([_frag(0), _frag(1)])
        assert isinstance(entries, list)

    def test_matrix_dtype_float32(self):
        matrix, _ = build_compat_matrix([_frag(0), _frag(1)])
        assert matrix.dtype == np.float32

    def test_matrix_shape_n_edges_by_n_edges(self):
        # 2 frags × 2 edges each = 4 edges total
        frags = [_frag(0), _frag(1)]
        matrix, _ = build_compat_matrix(frags)
        assert matrix.shape == (4, 4)

    def test_matrix_symmetric(self):
        matrix, _ = build_compat_matrix([_frag(0), _frag(1)])
        assert np.allclose(matrix, matrix.T)

    def test_diagonal_zero(self):
        matrix, _ = build_compat_matrix([_frag(0), _frag(1)])
        assert np.all(np.diag(matrix) == 0.0)

    def test_same_fragment_edges_zero(self):
        frags = [_frag(0), _frag(1)]
        matrix, _ = build_compat_matrix(frags)
        # Edges 0 and 1 belong to frag 0 → matrix[0,1] and matrix[1,0] == 0
        assert matrix[0, 1] == pytest.approx(0.0)
        assert matrix[1, 0] == pytest.approx(0.0)

    def test_entries_sorted_by_score_descending(self):
        frags = [_frag(0), _frag(1), _frag(2)]
        _, entries = build_compat_matrix(frags)
        if len(entries) > 1:
            scores = [e.score for e in entries]
            assert scores == sorted(scores, reverse=True)

    def test_entries_all_compat_entry(self):
        frags = [_frag(0), _frag(1)]
        _, entries = build_compat_matrix(frags)
        for e in entries:
            assert isinstance(e, CompatEntry)

    def test_entries_scores_nonneg(self):
        frags = [_frag(0), _frag(1)]
        _, entries = build_compat_matrix(frags)
        for e in entries:
            assert e.score >= 0.0

    def test_entries_scores_le_1(self):
        frags = [_frag(0), _frag(1)]
        _, entries = build_compat_matrix(frags)
        for e in entries:
            assert e.score <= 1.0

    def test_threshold_filters_entries(self):
        frags = [_frag(0), _frag(1), _frag(2)]
        _, all_entries = build_compat_matrix(frags, threshold=0.0)
        _, filtered = build_compat_matrix(frags, threshold=0.9)
        assert len(filtered) <= len(all_entries)

    def test_three_fragments_larger_matrix(self):
        # 3 frags × 2 edges = 6 edges → 6×6 matrix
        frags = [_frag(i) for i in range(3)]
        matrix, _ = build_compat_matrix(frags)
        assert matrix.shape == (6, 6)

    def test_matrix_values_in_0_1(self):
        frags = [_frag(0), _frag(1)]
        matrix, _ = build_compat_matrix(frags)
        assert np.all(matrix >= 0.0)
        assert np.all(matrix <= 1.0)

    def test_empty_fragments_empty_matrix(self):
        matrix, entries = build_compat_matrix([])
        assert matrix.shape == (0, 0)
        assert entries == []

    def test_single_fragment_empty_entries(self):
        frags = [_frag(0)]
        matrix, entries = build_compat_matrix(frags)
        assert entries == []

    def test_no_cross_fragment_entries_for_single_frag(self):
        frags = [_frag(0)]
        matrix, entries = build_compat_matrix(frags)
        # Edges of same fragment → no cross-matching
        assert all(e.edge_i.edge_id != e.edge_j.edge_id for e in entries)

    def test_high_threshold_fewer_entries(self):
        frags = [_frag(i) for i in range(4)]
        _, e_low = build_compat_matrix(frags, threshold=0.0)
        _, e_high = build_compat_matrix(frags, threshold=0.5)
        assert len(e_high) <= len(e_low)

    def test_different_lengths_reduces_score(self):
        # Fragment with very different edge lengths → lower score (length penalty)
        f0 = Fragment(fragment_id=0, image=np.zeros((32,32,3),dtype=np.uint8),
                      mask=np.zeros((32,32),dtype=np.uint8), contour=np.zeros((8,2)))
        f0.edges = [_edge(0, length=80.0)]
        f1 = Fragment(fragment_id=1, image=np.zeros((32,32,3),dtype=np.uint8),
                      mask=np.zeros((32,32),dtype=np.uint8), contour=np.zeros((8,2)))
        f1.edges = [_edge(10, length=10.0)]  # Very short → penalty
        _, entries_short = build_compat_matrix([f0, f1])

        f2 = Fragment(fragment_id=2, image=np.zeros((32,32,3),dtype=np.uint8),
                      mask=np.zeros((32,32),dtype=np.uint8), contour=np.zeros((8,2)))
        f2.edges = [_edge(20, length=80.0)]
        f3 = Fragment(fragment_id=3, image=np.zeros((32,32,3),dtype=np.uint8),
                      mask=np.zeros((32,32),dtype=np.uint8), contour=np.zeros((8,2)))
        f3.edges = [_edge(30, length=80.0)]  # Same length → no penalty
        _, entries_same = build_compat_matrix([f2, f3])

        score_short = entries_short[0].score if entries_short else 0.0
        score_same = entries_same[0].score if entries_same else 0.0
        assert score_same >= score_short


# ─── TestTopCandidates ────────────────────────────────────────────────────────

class TestTopCandidates:
    def _setup(self):
        frags = [_frag(0), _frag(1), _frag(2)]
        all_edges = [e for f in frags for e in f.edges]
        matrix, _ = build_compat_matrix(frags)
        return matrix, all_edges

    def test_returns_list(self):
        matrix, all_edges = self._setup()
        result = top_candidates(matrix, all_edges, 0, k=3)
        assert isinstance(result, list)

    def test_each_is_tuple(self):
        matrix, all_edges = self._setup()
        result = top_candidates(matrix, all_edges, 0, k=3)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_at_most_k_results(self):
        matrix, all_edges = self._setup()
        k = 2
        result = top_candidates(matrix, all_edges, 0, k=k)
        assert len(result) <= k

    def test_self_excluded(self):
        matrix, all_edges = self._setup()
        result = top_candidates(matrix, all_edges, 0, k=5)
        idx_list = [r[0] for r in result]
        assert 0 not in idx_list

    def test_scores_descending(self):
        matrix, all_edges = self._setup()
        result = top_candidates(matrix, all_edges, 0, k=5)
        if len(result) > 1:
            scores = [r[1] for r in result]
            assert scores == sorted(scores, reverse=True)

    def test_scores_positive(self):
        matrix, all_edges = self._setup()
        result = top_candidates(matrix, all_edges, 0, k=5)
        for idx, score in result:
            assert score > 0.0

    def test_indices_are_int(self):
        matrix, all_edges = self._setup()
        result = top_candidates(matrix, all_edges, 0, k=3)
        for idx, score in result:
            assert isinstance(idx, int)

    def test_scores_are_float(self):
        matrix, all_edges = self._setup()
        result = top_candidates(matrix, all_edges, 0, k=3)
        for idx, score in result:
            assert isinstance(score, float)
