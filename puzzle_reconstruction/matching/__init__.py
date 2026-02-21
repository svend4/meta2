"""
Сопоставление краёв фрагментов.

Модули:
    dtw           — Dynamic Time Warping для кривых краёв
    pairwise      — попарное сравнение двух краёв (CSS + DTW + FD + Text)
    compat_matrix — построение полной матрицы совместимости
    icp           — Iterative Closest Point для точного выравнивания контуров
    consensus     — консенсусное голосование по результатам нескольких методов
    graph_match   — графовый анализ (MST, спектральный порядок, random walk)
    feature_match — дескрипторное сопоставление (ORB/SIFT/AKAZE + RANSAC)
"""
from .dtw import dtw_distance, dtw_distance_mirror
from .pairwise import match_score
from .compat_matrix import build_compat_matrix
from .icp import icp_align, contour_icp, align_fragment_edge, ICPResult
from .consensus import (
    build_consensus,
    assembly_to_pairs,
    vote_on_pairs,
    consensus_score_matrix,
    ConsensusResult,
)
from .graph_match import (
    FragmentGraph,
    GraphMatchResult,
    build_fragment_graph,
    mst_ordering,
    spectral_ordering,
    random_walk_similarity,
    degree_centrality,
    analyze_graph,
)
from .feature_match import (
    KeypointMatch,
    FeatureMatchResult,
    extract_features,
    match_descriptors,
    estimate_homography,
    feature_match_pair,
    edge_feature_score,
)

__all__ = [
    "dtw_distance",
    "dtw_distance_mirror",
    "match_score",
    "build_compat_matrix",
    "icp_align",
    "contour_icp",
    "align_fragment_edge",
    "ICPResult",
    "build_consensus",
    "assembly_to_pairs",
    "vote_on_pairs",
    "consensus_score_matrix",
    "ConsensusResult",
    "FragmentGraph",
    "GraphMatchResult",
    "build_fragment_graph",
    "mst_ordering",
    "spectral_ordering",
    "random_walk_similarity",
    "degree_centrality",
    "analyze_graph",
    "KeypointMatch",
    "FeatureMatchResult",
    "extract_features",
    "match_descriptors",
    "estimate_homography",
    "feature_match_pair",
    "edge_feature_score",
]
