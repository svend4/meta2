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
    seam_score      — комплексная оценка шва (профиль, цвет, текстура, градиент)
    candidate_ranker — ранжирование пар-кандидатов (score_pair, rank_pairs,
                       filter_by_score, top_k, deduplicate_pairs, batch_rank)
    patch_matcher    — попиксельное (патч-based) сопоставление краёв
                       (PatchMatch, extract_edge_strip, ncc_score, ssd_score,
                       ssim_score, match_edge_strips, match_patch_pair,
                       batch_patch_match)
    boundary_matcher — геометрическое сопоставление контуров (BoundaryMatch,
                       extract_boundary_points, hausdorff_distance,
                       chamfer_distance, frechet_approx, score_boundary_pair,
                       match_boundary_pair, batch_match_boundaries)
    score_normalizer — нормализация оценок совместимости (ScoreNormResult,
                       normalize_minmax, normalize_zscore, normalize_rank,
                       calibrate_scores, combine_scores,
                       normalize_score_matrix, batch_normalize_scores)
    shape_matcher    — сопоставление по форме контура (ShapeMatchResult,
                       hu_moments, hu_distance, zernike_approx,
                       match_shapes, find_best_shape_match, batch_match_shapes)
    affine_matcher   — аффинное сопоставление (AffineMatchResult, estimate_affine,
                       apply_affine_pts, affine_reprojection_error,
                       score_affine_match, match_fragments_affine, batch_affine_match)
    spectral_matcher — спектральное сопоставление (SpectralMatchResult,
                       magnitude_spectrum, log_magnitude, spectrum_correlation,
                       phase_correlation, match_spectra, batch_spectral_match)
    score_combiner   — комбинирование оценок совпадения (ScoreVector, CombinedScore,
                       weighted_combine, min_combine, max_combine, rank_combine,
                       normalize_score_vectors, batch_combine)
    score_aggregator — агрегация оценок от нескольких матчеров (AggregationConfig,
                       AggregatedScore, AggregationReport, aggregate_scores,
                       aggregate_score_matrix, batch_aggregate_scores, filter_aggregated)
    edge_comparator  — сравнение краёв по интенсивности/градиенту/текстуре
                       (EdgeCompConfig, EdgeSample, EdgeCompResult,
                       extract_edge_sample, compare_edge_intensity,
                       compare_edge_gradient, compare_edge_texture,
                       score_edge_comparison, compare_edge_pair, batch_compare_edges)
    orient_matcher   — сопоставление по ориентации краёв
                       (OrientConfig, OrientProfile, OrientMatchResult,
                       compute_orient_profile, orient_similarity,
                       best_orient_angle, match_orient_pair, batch_orient_match)
    global_matcher   — глобальное сопоставление по совокупности оценок
                       (GlobalMatchConfig, GlobalMatch, GlobalMatchResult,
                       aggregate_pair_scores, rank_candidates, global_match,
                       filter_matches, merge_match_results)
    patch_validator    — валидация патчей на границах (PatchValidConfig,
                       PatchScore, PatchValidResult, compute_patch_score,
                       aggregate_patch_scores, validate_patch_pair,
                       batch_validate_patches, filter_valid_pairs)
    pair_scorer        — агрегированная оценка совместимости пар (ScoringWeights,
                       PairScoreResult, aggregate_channels, score_fragment_pair,
                       select_top_pairs, build_score_matrix, batch_score_pairs)
    curve_descriptor   — компактные дескрипторы кривых (CurveDescriptorConfig,
                       CurveDescriptor, compute_fourier_descriptor,
                       compute_curvature_profile, describe_curve,
                       descriptor_distance, descriptor_similarity,
                       batch_describe_curves, find_best_match)
"""
from .dtw import dtw_distance, dtw_distance_mirror
from .curve_descriptor import (
    CurveDescriptorConfig,
    CurveDescriptor,
    compute_fourier_descriptor,
    compute_curvature_profile,
    describe_curve,
    descriptor_distance,
    descriptor_similarity,
    batch_describe_curves,
    find_best_match,
)
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
from .seam_score import (
    SeamScoreResult,
    compute_seam_score,
    seam_score_matrix,
    normalize_seam_scores,
    rank_candidates,
    batch_seam_scores,
)
from .candidate_ranker import (
    CandidatePair,
    score_pair,
    rank_pairs,
    filter_by_score,
    top_k,
    deduplicate_pairs,
    batch_rank,
)
from .patch_matcher import (
    PatchMatch,
    extract_edge_strip,
    ncc_score,
    ssd_score,
    ssim_score,
    match_edge_strips,
    match_patch_pair,
    batch_patch_match,
)
from .boundary_matcher import (
    BoundaryMatch,
    extract_boundary_points,
    hausdorff_distance,
    chamfer_distance,
    frechet_approx,
    score_boundary_pair,
    match_boundary_pair,
    batch_match_boundaries,
)
from .score_normalizer import (
    ScoreNormResult,
    normalize_minmax,
    normalize_zscore,
    normalize_rank,
    calibrate_scores,
    combine_scores,
    normalize_score_matrix,
    batch_normalize_scores,
)
from .shape_matcher import (
    ShapeMatchResult,
    hu_moments,
    hu_distance,
    zernike_approx,
    match_shapes,
    find_best_shape_match,
    batch_match_shapes,
)
from .affine_matcher import (
    AffineMatchResult,
    estimate_affine,
    apply_affine_pts,
    affine_reprojection_error,
    score_affine_match,
    match_fragments_affine,
    batch_affine_match,
)
from .spectral_matcher import (
    SpectralMatchResult,
    magnitude_spectrum,
    log_magnitude,
    spectrum_correlation,
    phase_correlation,
    match_spectra,
    batch_spectral_match,
)
from .score_combiner import (
    ScoreVector,
    CombinedScore,
    weighted_combine,
    min_combine,
    max_combine,
    rank_combine,
    normalize_score_vectors,
    batch_combine,
)
from .score_aggregator import (
    AggregationConfig,
    AggregatedScore,
    AggregationReport,
    aggregate_scores,
    aggregate_score_matrix,
    batch_aggregate_scores,
    filter_aggregated,
)
from .edge_comparator import (
    EdgeCompConfig,
    EdgeSample,
    EdgeCompResult,
    extract_edge_sample,
    compare_edge_intensity,
    compare_edge_gradient,
    compare_edge_texture,
    score_edge_comparison,
    compare_edge_pair,
    batch_compare_edges,
)
from .orient_matcher import (
    OrientConfig,
    OrientProfile,
    OrientMatchResult,
    compute_orient_profile,
    orient_similarity,
    best_orient_angle,
    match_orient_pair,
    batch_orient_match,
)
from .global_matcher import (
    GlobalMatchConfig,
    GlobalMatch,
    GlobalMatchResult,
    aggregate_pair_scores,
    rank_candidates,
    global_match,
    filter_matches,
    merge_match_results,
)
from .patch_validator import (
    PatchValidConfig,
    PatchScore,
    PatchValidResult,
    compute_patch_score,
    aggregate_patch_scores,
    validate_patch_pair,
    batch_validate_patches,
    filter_valid_pairs,
)
from .pair_scorer import (
    ScoringWeights,
    PairScoreResult,
    aggregate_channels,
    score_pair as score_fragment_pair,
    select_top_pairs,
    build_score_matrix,
    batch_score_pairs,
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
    # Комплексная оценка шва
    "SeamScoreResult",
    "compute_seam_score",
    "seam_score_matrix",
    "normalize_seam_scores",
    "rank_candidates",
    "batch_seam_scores",
    # Ранжирование кандидатов
    "CandidatePair",
    "score_pair",
    "rank_pairs",
    "filter_by_score",
    "top_k",
    "deduplicate_pairs",
    "batch_rank",
    # Патч-based сопоставление краёв
    "PatchMatch",
    "extract_edge_strip",
    "ncc_score",
    "ssd_score",
    "ssim_score",
    "match_edge_strips",
    "match_patch_pair",
    "batch_patch_match",
    # Геометрическое сопоставление контуров
    "BoundaryMatch",
    "extract_boundary_points",
    "hausdorff_distance",
    "chamfer_distance",
    "frechet_approx",
    "score_boundary_pair",
    "match_boundary_pair",
    "batch_match_boundaries",
    # Нормализация оценок совместимости
    "ScoreNormResult",
    "normalize_minmax",
    "normalize_zscore",
    "normalize_rank",
    "calibrate_scores",
    "combine_scores",
    "normalize_score_matrix",
    "batch_normalize_scores",
    # Сопоставление по форме контура
    "ShapeMatchResult",
    "hu_moments",
    "hu_distance",
    "zernike_approx",
    "match_shapes",
    "find_best_shape_match",
    "batch_match_shapes",
    # Аффинное сопоставление
    "AffineMatchResult",
    "estimate_affine",
    "apply_affine_pts",
    "affine_reprojection_error",
    "score_affine_match",
    "match_fragments_affine",
    "batch_affine_match",
    # Спектральное сопоставление
    "SpectralMatchResult",
    "magnitude_spectrum",
    "log_magnitude",
    "spectrum_correlation",
    "phase_correlation",
    "match_spectra",
    "batch_spectral_match",
    # Комбинирование оценок совпадения
    "ScoreVector",
    "CombinedScore",
    "weighted_combine",
    "min_combine",
    "max_combine",
    "rank_combine",
    "normalize_score_vectors",
    "batch_combine",
    # Агрегация оценок от нескольких матчеров
    "AggregationConfig",
    "AggregatedScore",
    "AggregationReport",
    "aggregate_scores",
    "aggregate_score_matrix",
    "batch_aggregate_scores",
    "filter_aggregated",
    # Сравнение краёв по интенсивности/градиенту/текстуре
    "EdgeCompConfig",
    "EdgeSample",
    "EdgeCompResult",
    "extract_edge_sample",
    "compare_edge_intensity",
    "compare_edge_gradient",
    "compare_edge_texture",
    "score_edge_comparison",
    "compare_edge_pair",
    "batch_compare_edges",
    # Сопоставление по ориентации краёв
    "OrientConfig",
    "OrientProfile",
    "OrientMatchResult",
    "compute_orient_profile",
    "orient_similarity",
    "best_orient_angle",
    "match_orient_pair",
    "batch_orient_match",
    # Глобальное сопоставление
    "GlobalMatchConfig",
    "GlobalMatch",
    "GlobalMatchResult",
    "aggregate_pair_scores",
    "rank_candidates",
    "global_match",
    "filter_matches",
    "merge_match_results",
    # Валидация совместимости патчей
    "PatchValidConfig",
    "PatchScore",
    "PatchValidResult",
    "compute_patch_score",
    "aggregate_patch_scores",
    "validate_patch_pair",
    "batch_validate_patches",
    "filter_valid_pairs",
    # Агрегированная оценка совместимости пары фрагментов
    "ScoringWeights",
    "PairScoreResult",
    "aggregate_channels",
    "score_fragment_pair",
    "select_top_pairs",
    "build_score_matrix",
    "batch_score_pairs",
    # Дескрипторы кривых краёв
    "CurveDescriptorConfig",
    "CurveDescriptor",
    "compute_fourier_descriptor",
    "compute_curvature_profile",
    "describe_curve",
    "descriptor_distance",
    "descriptor_similarity",
    "batch_describe_curves",
    "find_best_match",
    # Bridge #9 — реестр 20 sleeping matching-модулей
    "build_matcher_registry",
    "list_matchers",
    "get_matcher",
    "get_matcher_category",
    "MATCHER_CATEGORIES",
    "MATCHER_REGISTRY",
]

# Bridge #9 exports
from .bridge import (  # noqa: E402
    build_matcher_registry,
    list_matchers,
    get_matcher,
    get_matcher_category,
    MATCHER_CATEGORIES,
    MATCHER_REGISTRY,
)
