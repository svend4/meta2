"""
Утилиты работы с матрицами оценок совместимости.

Score matrix utility functions for building, transforming, filtering and
analysing pairwise compatibility matrices used in fragment assembly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ScoreMatrixConfig:
    """Configuration for score-matrix operations."""

    threshold: float = 0.0
    """Entries with score <= threshold are treated as zero."""

    top_k: int = 10
    """Number of top candidates to return per row in row-level queries."""

    symmetrize: bool = True
    """Whether to enforce symmetry after construction."""

    eps: float = 1e-10
    """Numerical epsilon for near-zero checks."""

    def __post_init__(self) -> None:
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class MatrixStats:
    """Summary statistics of a score matrix."""

    n_edges: int
    n_nonzero: int
    mean_score: float
    max_score: float
    min_score: float
    sparsity: float  # fraction of zero (or below-threshold) entries
    top_pair: Tuple[int, int]

    def to_dict(self) -> dict:
        return {
            "n_edges": self.n_edges,
            "n_nonzero": self.n_nonzero,
            "mean_score": self.mean_score,
            "max_score": self.max_score,
            "min_score": self.min_score,
            "sparsity": self.sparsity,
            "top_pair": list(self.top_pair),
        }


@dataclass
class RankEntry:
    """A single ranked entry (index, score)."""

    idx: int
    score: float

    def __lt__(self, other: "RankEntry") -> bool:
        return self.score > other.score  # descending by default


# ---------------------------------------------------------------------------
# Basic matrix operations
# ---------------------------------------------------------------------------

def zero_diagonal(matrix: np.ndarray) -> np.ndarray:
    """Return a copy of *matrix* with the main diagonal set to zero."""
    m = matrix.copy()
    np.fill_diagonal(m, 0.0)
    return m


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Symmetrize by averaging: M = (M + M^T) / 2."""
    return (matrix + matrix.T) / 2.0


def threshold_matrix(
    matrix: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Zero out all entries with value <= *threshold*."""
    m = matrix.copy()
    m[m <= threshold] = 0.0
    return m


def normalize_rows(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Row-normalize the matrix so each row sums to 1.
    Rows that sum to zero are left as-is (all zeros).
    """
    m = matrix.astype(float)
    row_sums = m.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < eps, 1.0, row_sums)
    return m / row_sums


def top_k_indices(row: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-*k* values in *row* (descending order)."""
    if k <= 0:
        return np.array([], dtype=int)
    k = min(k, len(row))
    return np.argsort(row)[::-1][:k]


# ---------------------------------------------------------------------------
# Matrix analysis
# ---------------------------------------------------------------------------

def matrix_stats(
    matrix: np.ndarray,
    cfg: Optional[ScoreMatrixConfig] = None,
) -> MatrixStats:
    """
    Compute :class:`MatrixStats` for a square score matrix.

    Parameters
    ----------
    matrix : (N, N) array — pairwise score matrix (zeros on diagonal).
    cfg    : optional :class:`ScoreMatrixConfig`.

    Returns
    -------
    :class:`MatrixStats`
    """
    if cfg is None:
        cfg = ScoreMatrixConfig()

    n = matrix.shape[0]
    # Exclude diagonal
    off_diag = matrix.copy()
    np.fill_diagonal(off_diag, 0.0)

    mask = off_diag > cfg.threshold
    vals = off_diag[mask]
    n_nonzero = int(mask.sum())
    total_off = max(n * (n - 1), 1)

    if n_nonzero > 0:
        mean_s = float(vals.mean())
        max_s = float(vals.max())
        min_s = float(vals.min())
        flat_idx = int(np.argmax(off_diag))
        top_pair = (flat_idx // n, flat_idx % n)
    else:
        mean_s = 0.0
        max_s = 0.0
        min_s = 0.0
        top_pair = (0, 0)

    sparsity = 1.0 - n_nonzero / total_off

    return MatrixStats(
        n_edges=n,
        n_nonzero=n_nonzero,
        mean_score=mean_s,
        max_score=max_s,
        min_score=min_s,
        sparsity=sparsity,
        top_pair=top_pair,
    )


def top_k_per_row(
    matrix: np.ndarray,
    k: int,
    exclude_self: bool = True,
) -> List[List[RankEntry]]:
    """
    For each row, return up to *k* top entries as :class:`RankEntry` objects.

    Parameters
    ----------
    matrix       : (N, N) score matrix.
    k            : number of top candidates per row.
    exclude_self : whether to exclude the diagonal (self-matching).

    Returns
    -------
    List of N lists, each containing up to *k* :class:`RankEntry` objects
    sorted by descending score.
    """
    n = matrix.shape[0]
    result: List[List[RankEntry]] = []
    for i in range(n):
        row = matrix[i].copy()
        if exclude_self:
            row[i] = 0.0
        idx_sorted = np.argsort(row)[::-1]
        entries: List[RankEntry] = []
        for idx in idx_sorted:
            if len(entries) >= k:
                break
            if row[idx] > 0.0:
                entries.append(RankEntry(idx=int(idx), score=float(row[idx])))
        result.append(entries)
    return result


def filter_by_threshold(
    matrix: np.ndarray,
    threshold: float,
    cfg: Optional[ScoreMatrixConfig] = None,
) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    """
    Filter the matrix, returning a sparse representation of above-threshold
    entries as (row, col, score) triples.

    Returns
    -------
    (filtered_matrix, pairs) where *pairs* is sorted by descending score.
    """
    m = threshold_matrix(matrix, threshold)
    rows, cols = np.where(m > threshold)
    pairs = sorted(
        [(int(r), int(c), float(m[r, c])) for r, c in zip(rows, cols)],
        key=lambda x: -x[2],
    )
    return m, pairs


# ---------------------------------------------------------------------------
# Cross-fragment utilities
# ---------------------------------------------------------------------------

def intra_fragment_mask(
    n_edges_per_frag: List[int],
) -> np.ndarray:
    """
    Build a boolean mask where True means the two edges belong to the
    *same* fragment (and therefore should not be matched).

    Parameters
    ----------
    n_edges_per_frag : list of ints — number of edges for each fragment.

    Returns
    -------
    (N, N) bool array where N = sum(n_edges_per_frag).
    """
    n = sum(n_edges_per_frag)
    mask = np.zeros((n, n), dtype=bool)
    offset = 0
    for cnt in n_edges_per_frag:
        mask[offset: offset + cnt, offset: offset + cnt] = True
        offset += cnt
    return mask


def apply_intra_fragment_mask(
    matrix: np.ndarray,
    n_edges_per_frag: List[int],
) -> np.ndarray:
    """
    Zero out entries in *matrix* that correspond to same-fragment edge pairs.
    """
    mask = intra_fragment_mask(n_edges_per_frag)
    m = matrix.copy()
    m[mask] = 0.0
    return m


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def batch_matrix_stats(
    matrices: List[np.ndarray],
    cfg: Optional[ScoreMatrixConfig] = None,
) -> List[MatrixStats]:
    """Compute :class:`MatrixStats` for each matrix in *matrices*."""
    return [matrix_stats(m, cfg) for m in matrices]
