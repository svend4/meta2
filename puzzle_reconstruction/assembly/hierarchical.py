"""
Hierarchical (agglomerative) assembler for puzzle fragment reconstruction.

Approach:
    1. Start with each fragment as its own cluster.
    2. Repeatedly merge the two clusters with the highest inter-cluster
       compatibility score (single-linkage or average-linkage).
    3. After merging, update the cluster representative placement.
    4. Repeat until a single cluster remains or no compatible pairs exist.

Cluster merging strategy:
    - Single linkage:  score(A, B) = max score among all (a∈A, b∈B) edge pairs.
    - Average linkage: score(A, B) = mean score among all (a∈A, b∈B) edge pairs.
    - Complete linkage: score(A, B) = min score among all (a∈A, b∈B) edge pairs.

The result is an Assembly with placements computed by positioning each
merged cluster relative to its merge partner.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..models import Fragment, CompatEntry, Assembly


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical assembly."""
    linkage: str = "average"      # "single", "average", "complete"
    min_merge_score: float = 0.0  # Minimum score to merge two clusters
    max_clusters: int = 1         # Stop when this many clusters remain


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------

@dataclass
class Cluster:
    """A cluster of fragments with a shared coordinate frame."""
    cluster_id:  int
    fragment_ids: Set[int] = field(default_factory=set)
    placements:  Dict[int, Tuple[np.ndarray, float]] = field(default_factory=dict)
    total_score: float = 0.0


# ---------------------------------------------------------------------------
# Inter-cluster scoring
# ---------------------------------------------------------------------------

def _inter_cluster_score(
    cluster_a: Cluster,
    cluster_b: Cluster,
    entries: List[CompatEntry],
    edge_to_frag: Dict[int, "Fragment"],
    linkage: str = "average",
) -> float:
    """
    Compute inter-cluster compatibility score between two clusters.

    Args:
        cluster_a, cluster_b: Clusters to compare.
        entries:              All CompatEntry records.
        edge_to_frag:         Map from edge_id to Fragment.
        linkage:              One of "single", "average", "complete".

    Returns:
        Score ∈ [0, 1] (or negative if no connecting entries found).
    """
    scores = []
    for e in entries:
        fi = edge_to_frag.get(getattr(e.edge_i, "edge_id", -1))
        fj = edge_to_frag.get(getattr(e.edge_j, "edge_id", -1))
        if fi is None or fj is None:
            continue
        a_has_i = fi.fragment_id in cluster_a.fragment_ids
        a_has_j = fj.fragment_id in cluster_a.fragment_ids
        b_has_i = fi.fragment_id in cluster_b.fragment_ids
        b_has_j = fj.fragment_id in cluster_b.fragment_ids
        # Edge pair crosses the A-B boundary
        if (a_has_i and b_has_j) or (b_has_i and a_has_j):
            scores.append(e.score)

    if not scores:
        return -1.0  # No connection

    if linkage == "single":
        return float(max(scores))
    elif linkage == "complete":
        return float(min(scores))
    else:  # average
        return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Cluster merge
# ---------------------------------------------------------------------------

def _merge_clusters(
    cluster_a: Cluster,
    cluster_b: Cluster,
    entries: List[CompatEntry],
    edge_to_frag: Dict[int, "Fragment"],
    new_id: int,
    merge_score: float,
) -> Cluster:
    """
    Merge cluster_b into cluster_a, updating placements.

    Fragments in cluster_b are translated so that their best-matching edge
    is positioned next to its partner in cluster_a.

    Returns a new merged Cluster.
    """
    # Determine offset: place cluster_b's centroid next to cluster_a's centroid
    if cluster_a.placements:
        a_positions = np.array([pos for pos, _ in cluster_a.placements.values()])
        a_centroid  = a_positions.mean(axis=0)
    else:
        a_centroid = np.array([0.0, 0.0])

    # Compute offset for cluster_b relative to cluster_a
    offset = a_centroid + np.array([120.0 * len(cluster_a.fragment_ids), 0.0])

    merged = Cluster(
        cluster_id=new_id,
        fragment_ids=cluster_a.fragment_ids | cluster_b.fragment_ids,
        placements=dict(cluster_a.placements),
        total_score=cluster_a.total_score + cluster_b.total_score + merge_score,
    )

    # Add cluster_b's fragments with offset
    for fid, (pos, rot) in cluster_b.placements.items():
        merged.placements[fid] = (pos + offset, rot)

    return merged


# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------

def hierarchical_assembly(
    fragments: List[Fragment],
    entries: List[CompatEntry],
    cfg: Optional[HierarchicalConfig] = None,
) -> Assembly:
    """
    Hierarchical (agglomerative) puzzle assembly.

    Args:
        fragments: All fragments to assemble.
        entries:   Compatibility entries (sorted descending by score).
        cfg:       HierarchicalConfig (uses defaults if None).

    Returns:
        Assembly with placements and total_score.
    """
    if cfg is None:
        cfg = HierarchicalConfig()

    if not fragments:
        return Assembly(fragments=fragments, placements={},
                        compat_matrix=np.array([]), method="hierarchical")

    # Build edge→fragment mapping
    edge_to_frag: Dict[int, Fragment] = {}
    for frag in fragments:
        for edge in frag.edges:
            edge_to_frag[edge.edge_id] = frag

    # Initialise: each fragment is its own cluster
    clusters: Dict[int, Cluster] = {}
    for idx, frag in enumerate(fragments):
        c = Cluster(
            cluster_id=idx,
            fragment_ids={frag.fragment_id},
            placements={frag.fragment_id: (np.array([idx * 120.0, 0.0]), 0.0)},
        )
        clusters[idx] = c

    next_id = len(fragments)

    # Agglomerative merging
    while len(clusters) > cfg.max_clusters:
        # Find best pair to merge
        best_score = -float("inf")
        best_pair:  Optional[Tuple[int, int]] = None
        ids = list(clusters.keys())

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                ci = clusters[ids[i]]
                cj = clusters[ids[j]]
                s = _inter_cluster_score(ci, cj, entries, edge_to_frag, cfg.linkage)
                if s > best_score:
                    best_score = s
                    best_pair  = (ids[i], ids[j])

        # No valid merge found → stop
        if best_pair is None or best_score < cfg.min_merge_score:
            break

        id_a, id_b = best_pair
        merged = _merge_clusters(
            clusters[id_a],
            clusters[id_b],
            entries,
            edge_to_frag,
            new_id=next_id,
            merge_score=max(best_score, 0.0),
        )
        del clusters[id_a]
        del clusters[id_b]
        clusters[next_id] = merged
        next_id += 1

    # Collect all placements from all remaining clusters
    all_placements: Dict[int, Tuple[np.ndarray, float]] = {}
    total_score = 0.0
    for c in clusters.values():
        all_placements.update(c.placements)
        total_score += c.total_score

    return Assembly(
        fragments=fragments,
        placements=all_placements,
        compat_matrix=np.array([]),
        method="hierarchical",
        total_score=float(total_score),
    )


# ---------------------------------------------------------------------------
# Linkage functions (standalone)
# ---------------------------------------------------------------------------

def single_linkage_score(
    scores: List[float],
) -> float:
    """Single-linkage: return the maximum score."""
    if not scores:
        return 0.0
    return float(max(scores))


def average_linkage_score(
    scores: List[float],
) -> float:
    """Average-linkage: return the mean score."""
    if not scores:
        return 0.0
    return float(np.mean(scores))


def complete_linkage_score(
    scores: List[float],
) -> float:
    """Complete-linkage: return the minimum score."""
    if not scores:
        return 0.0
    return float(min(scores))
