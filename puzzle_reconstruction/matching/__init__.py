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
    color_match   — цветовое сопоставление (гистограммы, моменты, профили)
    texture_match — текстурное сопоставление (LBP, Gabor, ориентации градиентов)
    geometric_match — геометрическое сопоставление (площадь, AR, моменты Ху)
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
from .color_match import (
    ColorMatchResult,
    compute_color_histogram,
    histogram_distance,
    compute_color_moments,
    moments_distance,
    edge_color_profile,
    color_match_pair,
    color_compatibility_matrix,
)
from .texture_match import (
    TextureMatchResult,
    compute_lbp_histogram,
    lbp_distance,
    compute_gabor_features,
    gabor_distance,
    gradient_orientation_hist,
    texture_match_pair,
    texture_compatibility_matrix,
)
from .geometric_match import (
    FragmentGeometry,
    GeometricMatchResult,
    compute_fragment_geometry,
    aspect_ratio_similarity,
    area_ratio_similarity,
    hu_moments_similarity,
    edge_length_similarity,
    match_geometry,
    batch_geometry_match,
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
    # Цветовое сопоставление
    "ColorMatchResult",
    "compute_color_histogram",
    "histogram_distance",
    "compute_color_moments",
    "moments_distance",
    "edge_color_profile",
    "color_match_pair",
    "color_compatibility_matrix",
    # Текстурное сопоставление
    "TextureMatchResult",
    "compute_lbp_histogram",
    "lbp_distance",
    "compute_gabor_features",
    "gabor_distance",
    "gradient_orientation_hist",
    "texture_match_pair",
    "texture_compatibility_matrix",
    # Геометрическое сопоставление
    "FragmentGeometry",
    "GeometricMatchResult",
    "compute_fragment_geometry",
    "aspect_ratio_similarity",
    "area_ratio_similarity",
    "hu_moments_similarity",
    "edge_length_similarity",
    "match_geometry",
    "batch_geometry_match",
]
