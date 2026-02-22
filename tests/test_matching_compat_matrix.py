"""Tests for puzzle_reconstruction/matching/compat_matrix.py"""
import pytest
import numpy as np

from puzzle_reconstruction.matching.compat_matrix import (
    build_compat_matrix,
    top_candidates,
)
from puzzle_reconstruction.models import Fragment, EdgeSignature, EdgeSide


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


# ─── build_compat_matrix ──────────────────────────────────────────────────────

class TestBuildCompatMatrix:
    def test_empty_fragments_empty_matrix(self):
        mat, entries = build_compat_matrix([])
        assert mat.shape == (0, 0)
        assert entries == []

    def test_single_fragment_single_edge_shape(self):
        """1 fragment with 1 edge → 1×1 matrix, no cross-fragment pairs."""
        f = make_fragment(0, [0])
        mat, entries = build_compat_matrix([f])
        assert mat.shape == (1, 1)
        assert mat[0, 0] == pytest.approx(0.0)

    def test_single_fragment_no_cross_entries(self):
        """Same-fragment edges are skipped."""
        f = make_fragment(0, [0, 1])  # 2 edges, same fragment
        mat, entries = build_compat_matrix([f])
        # entries only for cross-fragment pairs, so empty
        assert entries == []

    def test_returns_tuple(self):
        f = make_fragment(0, [0])
        result = build_compat_matrix([f])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_matrix_dtype_float32(self):
        f = make_fragment(0, [0])
        mat, _ = build_compat_matrix([f])
        assert mat.dtype == np.float32

    def test_two_fragments_same_fragment_skipped(self):
        """2 edges on same fragment → matrix (2,2) but no cross entries."""
        f = make_fragment(0, [0, 1])
        mat, entries = build_compat_matrix([f])
        assert mat.shape == (2, 2)
        assert entries == []

    def test_threshold_filters_entries(self):
        """With high threshold and single-fragment, still 0 entries."""
        f = make_fragment(0, [0])
        _, entries = build_compat_matrix([f], threshold=0.99)
        assert entries == []

    def test_entries_sorted_desc(self):
        """Entries are sorted descending by score."""
        f1 = make_fragment(0, [0])
        f2 = make_fragment(1, [1])
        f3 = make_fragment(2, [2])
        _, entries = build_compat_matrix([f1, f2, f3])
        scores = [e.score for e in entries]
        assert scores == sorted(scores, reverse=True)


# ─── top_candidates ───────────────────────────────────────────────────────────

class TestTopCandidates:
    def make_score_matrix(self, n, values=None):
        mat = np.zeros((n, n), dtype=np.float32)
        if values is None:
            for i in range(n):
                for j in range(n):
                    if i != j:
                        mat[i, j] = float(i + j + 1) / (2 * n)
        else:
            for (i, j), v in values:
                mat[i, j] = v
                mat[j, i] = v
        return mat

    def test_returns_list_of_tuples(self):
        mat = self.make_score_matrix(4)
        result = top_candidates(mat, list(range(4)), 0, k=3)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) for item in result)

    def test_excludes_self(self):
        """The diagonal (self-match) is set to 0 and excluded."""
        mat = np.eye(4, dtype=np.float32)
        result = top_candidates(mat, list(range(4)), 0, k=4)
        # After zeroing row[0], all 1.0 diagonal entries become 0 → only idx 0
        # diagonal was set to 0, so no positive entries
        assert all(idx != 0 for idx, _ in result)

    def test_sorted_desc(self):
        mat = self.make_score_matrix(5)
        result = top_candidates(mat, list(range(5)), 1, k=4)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_k_limits_results(self):
        mat = self.make_score_matrix(6)
        result = top_candidates(mat, list(range(6)), 0, k=2)
        assert len(result) <= 2

    def test_only_positive_scores(self):
        """Entries with score <= 0 are excluded."""
        mat = np.zeros((4, 4), dtype=np.float32)
        mat[0, 2] = 0.8
        mat[2, 0] = 0.8
        result = top_candidates(mat, list(range(4)), 0, k=4)
        # Only index 2 has positive score for edge 0
        assert all(s > 0.0 for _, s in result)

    def test_all_zero_row_returns_empty(self):
        """All zeros in row → no positive candidates."""
        mat = np.zeros((3, 3), dtype=np.float32)
        result = top_candidates(mat, list(range(3)), 0)
        assert result == []

    def test_returns_index_score_pairs(self):
        mat = np.zeros((4, 4), dtype=np.float32)
        mat[1, 3] = 0.7
        mat[3, 1] = 0.7
        mat[1, 2] = 0.5
        mat[2, 1] = 0.5
        result = top_candidates(mat, list(range(4)), 1, k=2)
        # Should return [(3, 0.7), (2, 0.5)]
        assert len(result) == 2
        top_idx, top_score = result[0]
        assert top_idx == 3
        assert top_score == pytest.approx(0.7)

    def test_result_indices_are_int(self):
        mat = self.make_score_matrix(4)
        result = top_candidates(mat, list(range(4)), 0, k=3)
        for idx, _ in result:
            assert isinstance(idx, int)

    def test_result_scores_are_float(self):
        mat = self.make_score_matrix(4)
        result = top_candidates(mat, list(range(4)), 0, k=3)
        for _, score in result:
            assert isinstance(score, float)
