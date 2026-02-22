"""Descriptor comparison and aggregation utilities.

Provides building blocks for working with feature descriptors:
normalisation, distance metrics, pooling, matching helpers, and
lightweight batch operations used across the matching pipeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class DescriptorConfig:
    """Parameters for descriptor comparison operations."""
    metric: str = "l2"          # "l2" | "cosine" | "chi2" | "l1"
    normalize: bool = True
    eps: float = 1e-8           # numerical stability floor


# ─── Normalisation ────────────────────────────────────────────────────────────

def l2_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return *v* divided by its L2 norm (safe against zero vectors)."""
    n = np.linalg.norm(v)
    if n < eps:
        return v.copy()
    return v / n


def l1_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return *v* divided by its L1 norm."""
    s = np.abs(v).sum()
    if s < eps:
        return v.copy()
    return v / s


def batch_l2_normalize(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2 normalisation of a 2-D descriptor matrix.

    Parameters
    ----------
    mat : np.ndarray  shape (N, D)

    Returns
    -------
    np.ndarray  shape (N, D)  – each row has unit L2 norm (or zero).
    """
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return mat / norms


# ─── Distance metrics ─────────────────────────────────────────────────────────

def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 1-D descriptors."""
    return float(np.linalg.norm(a - b))


def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Cosine distance in [0, 1] (0 = identical direction)."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 1.0
    cos = float(np.dot(a, b) / (na * nb))
    cos = max(-1.0, min(1.0, cos))
    return (1.0 - cos) / 2.0


def chi2_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Chi-squared distance for normalised histogram descriptors."""
    denom = a + b + eps
    return float(np.sum((a - b) ** 2 / denom))


def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Manhattan / L1 distance."""
    return float(np.sum(np.abs(a - b)))


def descriptor_distance(a: np.ndarray, b: np.ndarray,
                         metric: str = "l2",
                         eps: float = 1e-8) -> float:
    """Dispatch to one of the supported distance metrics.

    Parameters
    ----------
    a, b   : 1-D np.ndarray
    metric : "l2" | "cosine" | "chi2" | "l1"
    """
    if metric == "l2":
        return l2_distance(a, b)
    if metric == "cosine":
        return cosine_distance(a, b, eps)
    if metric == "chi2":
        return chi2_distance(a, b, eps)
    if metric == "l1":
        return l1_distance(a, b)
    raise ValueError(f"Unknown metric: {metric!r}")


# ─── Pairwise distance matrix ─────────────────────────────────────────────────

def pairwise_l2(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """Compute the pairwise L2 distance matrix between two descriptor sets.

    Parameters
    ----------
    mat_a : np.ndarray  shape (M, D)
    mat_b : np.ndarray  shape (N, D)

    Returns
    -------
    np.ndarray  shape (M, N)
    """
    diff = mat_a[:, None, :] - mat_b[None, :, :]   # M × N × D
    return np.sqrt((diff ** 2).sum(axis=2))


def pairwise_cosine(mat_a: np.ndarray, mat_b: np.ndarray,
                    eps: float = 1e-8) -> np.ndarray:
    """Pairwise cosine distance matrix (values in [0, 1])."""
    a_norm = batch_l2_normalize(mat_a, eps)
    b_norm = batch_l2_normalize(mat_b, eps)
    cos_sim = a_norm @ b_norm.T
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return (1.0 - cos_sim) / 2.0


# ─── Nearest-neighbour matching ───────────────────────────────────────────────

@dataclass
class DescriptorMatch:
    """A single descriptor correspondence."""
    query_idx: int
    train_idx: int
    distance: float


def nn_match(desc_q: np.ndarray,
             desc_t: np.ndarray,
             metric: str = "l2") -> List[DescriptorMatch]:
    """Return the nearest-neighbour match for every query descriptor.

    Parameters
    ----------
    desc_q : np.ndarray  shape (M, D)  – query descriptors
    desc_t : np.ndarray  shape (N, D)  – train descriptors
    metric : distance metric to use

    Returns
    -------
    List[DescriptorMatch]  length M
    """
    if len(desc_q) == 0 or len(desc_t) == 0:
        return []

    if metric == "l2":
        dmat = pairwise_l2(desc_q, desc_t)
    elif metric == "cosine":
        dmat = pairwise_cosine(desc_q, desc_t)
    else:
        dmat = pairwise_l2(desc_q, desc_t)  # fallback

    nn_idx = dmat.argmin(axis=1)
    matches = []
    for q, t in enumerate(nn_idx):
        matches.append(DescriptorMatch(
            query_idx=int(q),
            train_idx=int(t),
            distance=float(dmat[q, t]),
        ))
    return matches


def ratio_test(desc_q: np.ndarray,
               desc_t: np.ndarray,
               ratio: float = 0.75,
               metric: str = "l2") -> List[DescriptorMatch]:
    """Lowe's ratio test: keep matches where d1/d2 < *ratio*.

    Returns only matches that survive the ratio filter.
    """
    if len(desc_q) == 0 or len(desc_t) < 2:
        return []
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"ratio must be in (0, 1), got {ratio}")

    if metric == "l2":
        dmat = pairwise_l2(desc_q, desc_t)
    elif metric == "cosine":
        dmat = pairwise_cosine(desc_q, desc_t)
    else:
        dmat = pairwise_l2(desc_q, desc_t)

    # Sort distances for each query
    sorted_idx = np.argsort(dmat, axis=1)
    matches = []
    for q in range(len(desc_q)):
        t1, t2 = sorted_idx[q, 0], sorted_idx[q, 1]
        d1, d2 = dmat[q, t1], dmat[q, t2]
        if d2 > 1e-12 and d1 / d2 < ratio:
            matches.append(DescriptorMatch(
                query_idx=int(q),
                train_idx=int(t1),
                distance=float(d1),
            ))
    return matches


# ─── Descriptor pooling ───────────────────────────────────────────────────────

def mean_pool(descriptors: np.ndarray) -> np.ndarray:
    """Average-pool a set of descriptors into a single vector."""
    if len(descriptors) == 0:
        raise ValueError("Cannot pool empty descriptor set")
    return descriptors.mean(axis=0)


def max_pool(descriptors: np.ndarray) -> np.ndarray:
    """Max-pool a set of descriptors into a single vector."""
    if len(descriptors) == 0:
        raise ValueError("Cannot pool empty descriptor set")
    return descriptors.max(axis=0)


def vlad_encode(descriptors: np.ndarray,
                codebook: np.ndarray,
                normalize: bool = True) -> np.ndarray:
    """Encode descriptors as a VLAD vector given a codebook.

    Parameters
    ----------
    descriptors : np.ndarray  shape (N, D)
    codebook    : np.ndarray  shape (K, D)
    normalize   : if True, L2-normalise the output

    Returns
    -------
    np.ndarray  shape (K * D,)
    """
    k, d = codebook.shape
    if len(descriptors) == 0:
        return np.zeros(k * d, dtype=np.float32)

    # Assign each descriptor to its nearest visual word
    dists = pairwise_l2(descriptors, codebook)   # N × K
    assignments = dists.argmin(axis=1)

    vlad = np.zeros((k, d), dtype=np.float64)
    for i, word in enumerate(assignments):
        vlad[word] += descriptors[i] - codebook[word]

    vlad = vlad.flatten().astype(np.float32)
    if normalize:
        vlad = l2_normalize(vlad).astype(np.float32)
    return vlad


# ─── Batch helpers ────────────────────────────────────────────────────────────

def batch_nn_match(
    query_sets: List[np.ndarray],
    train_set: np.ndarray,
    metric: str = "l2",
) -> List[List[DescriptorMatch]]:
    """Apply :func:`nn_match` from each query set to a single train set."""
    return [nn_match(q, train_set, metric) for q in query_sets]


def top_k_matches(
    matches: List[DescriptorMatch],
    k: int,
) -> List[DescriptorMatch]:
    """Return the *k* matches with the smallest distance."""
    return sorted(matches, key=lambda m: m.distance)[:k]


def filter_matches_by_distance(
    matches: List[DescriptorMatch],
    max_distance: float,
) -> List[DescriptorMatch]:
    """Keep only matches whose distance is ≤ *max_distance*."""
    return [m for m in matches if m.distance <= max_distance]
