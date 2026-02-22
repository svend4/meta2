"""
Алгоритмы описания и синтеза подписей краёв фрагментов.

Подпакеты:
    tangram/       — геометрическое описание внутреннего многоугольника
    fractal/       — фрактальные характеристики края (Box, Divider, IFS, CSS)

Модули:
    synthesis            — синтез EdgeSignature из танграма и фрактала
    shape_context        — Shape Context дескриптор (Belongie et al., 2002)
    word_segmentation    — сегментация слов/строк (морфология, без OCR)
    fragment_classifier  — классификация типов фрагментов (угол/край/внутр.)
    edge_profile         — 1D профили краёв (яркость, градиент, текстура, DTW)
    line_detector        — обнаружение строк текста (проекция, Хаф, auto)
    fragment_aligner     — субпиксельное выравнивание краёв (фазовая корр., шаблон)
    score_aggregator     — агрегация оценок совместимости (weighted_avg, harmonic, min, max)
    edge_scorer          — многоканальная оценка совместимости краёв (EdgeScore,
                           score_color_compat, score_gradient_compat,
                           score_texture_compat, score_edge_pair, batch_score_edges)
    position_estimator   — оценка абсолютных позиций фрагментов по попарным смещениям
                           (PositionEstimate, build_offset_graph, estimate_positions,
                           refine_positions, positions_to_array, align_to_origin)
    overlap_resolver     — разрешение перекрытий (OverlapConflict,
                           compute_separation_vector, detect_overlap_conflicts,
                           resolve_single_conflict, resolve_all_conflicts,
                           conflict_score)
    rotation_estimator   — оценка угла поворота фрагментов (RotationEstimate,
                           estimate_by_pca, estimate_by_moments, estimate_by_gradient,
                           refine_rotation, estimate_rotation_pair,
                           batch_estimate_rotations)
    gradient_flow        — анализ градиентного поля (GradientField, GradientStats,
                           compute_gradient, compute_magnitude, compute_orientation,
                           compute_divergence, compute_curl, flow_along_boundary,
                           compare_gradient_fields, batch_gradient_fields,
                           compute_gradient_stats)
    region_segmenter     — сегментация на регионы (RegionProps, SegmentationResult,
                           label_connected, compute_region_props, filter_regions,
                           merge_close_regions, region_adjacency, largest_region,
                           regions_to_mask, batch_segment)
    path_planner         — планирование пути обхода фрагментов (PathResult,
                           dijkstra, shortest_path, all_pairs_shortest_paths,
                           topological_sort, find_connected_components,
                           minimum_spanning_tree, batch_dijkstra)
    edge_extractor       — извлечение граничных профилей фрагментов (EdgeSegment,
                           FragmentEdges, detect_boundary, extract_edge_points,
                           split_edge_by_side, compute_edge_length, simplify_edge,
                           extract_fragment_edges, batch_extract_edges)
    contour_tracker      — отслеживание контуров (ContourInfo, TrackState,
                           find_contours, filter_contours, contour_to_array,
                           compute_contour_info, match_contours,
                           track_contour, batch_find_contours)
    region_splitter      — разделение на связные регионы (RegionInfo, SplitResult,
                           find_regions, filter_regions, region_masks,
                           merge_small_regions, largest_region,
                           split_mask_to_crops, batch_find_regions)
    texture_descriptor   — текстурные дескрипторы (TextureDescriptor, compute_lbp,
                           compute_glcm_features, compute_stats_descriptor,
                           compute_texture_descriptor, normalize_descriptor,
                           descriptor_distance, batch_compute_descriptors)
    sift_matcher         — SIFT-сопоставление фрагментов (SiftConfig, MatchResult,
                           extract_keypoints, match_descriptors, compute_homography,
                           sift_match_pair, filter_matches_by_distance,
                           batch_sift_match)
    patch_matcher        — патч-матчинг по скользящему окну (PatchConfig, PatchMatch,
                           extract_patch, ncc_score, ssd_score, sad_score,
                           match_patch_in_image, find_matches, top_matches,
                           batch_patch_match)
    descriptor_aggregator — агрегация дескрипторов (AggregatorConfig,
                            AggregatedDescriptor, l2_normalize,
                            concatenate_descriptors, weighted_average_descriptors,
                            pca_reduce, elementwise_aggregate, aggregate,
                            distance_matrix, batch_aggregate)
    homography_estimator  — оценка гомографии (HomographyConfig, HomographyResult,
                            normalize_points, dlt_homography,
                            compute_reprojection_error, estimate_homography,
                            decompose_homography, warp_points,
                            batch_estimate_homographies)
    descriptor_combiner   — комбинирование дескрипторов (CombineConfig,
                            DescriptorSet, CombineResult, combine_descriptors,
                            combine_selected, batch_combine,
                            descriptor_distance, build_distance_matrix,
                            find_nearest)
    edge_comparator       — сравнение краёв фрагментов (CompareConfig,
                            EdgeCompareResult, dtw_distance, css_similarity,
                            fd_score, ifs_similarity, compare_edges,
                            build_compat_matrix, top_k_matches)
    boundary_descriptor   — дескрипторы граничных сегментов (DescriptorConfig,
                            BoundaryDescriptor, compute_curvature,
                            curvature_histogram, direction_histogram,
                            chord_distribution, extract_descriptor,
                            descriptor_similarity, batch_extract_descriptors)
    contour_smoother      — сглаживание и передискретизация контуров (SmootherConfig,
                            SmoothedContour, smooth_gaussian, resample_contour,
                            compute_arc_length, smooth_and_resample,
                            align_contours, contour_similarity as smooth_contour_similarity,
                            batch_smooth)
"""
from .synthesis import compute_fractal_signature, build_edge_signatures

from .tangram.hull import convex_hull, rdp_simplify, normalize_polygon
from .tangram.classifier import classify_shape, compute_interior_angles
from .tangram.inscriber import fit_tangram

from .fractal.box_counting import box_counting_fd
from .fractal.divider import divider_fd
from .fractal.css import css_similarity_mirror
from .fractal.ifs import fit_ifs

from .shape_context import (
    compute_shape_context,
    shape_context_distance,
    match_shape_contexts,
    normalize_shape_context,
    log_polar_histogram,
    contour_similarity,
    ShapeContextResult,
)
from .word_segmentation import (
    WordBox,
    LineSegment,
    WordSegmentationResult,
    binarize,
    segment_words,
    merge_line_words,
    segment_lines,
    segment_document,
)
from .fragment_classifier import (
    FragmentType,
    FragmentFeatures,
    ClassifyResult,
    compute_texture_features,
    compute_edge_features,
    compute_shape_features,
    detect_text_presence,
    classify_fragment_type,
    classify_fragment,
    batch_classify,
)
from .edge_profile import (
    EdgeProfile,
    ProfileMatchResult,
    extract_intensity_profile,
    extract_gradient_profile,
    extract_texture_profile,
    normalize_profile,
    profile_correlation,
    profile_dtw,
    match_edge_profiles,
    batch_profile_match,
)
from .line_detector import (
    TextLine,
    LineDetectionResult,
    detect_lines_projection,
    detect_lines_hough,
    estimate_line_metrics,
    filter_lines,
    detect_text_lines,
    batch_detect_lines,
)
from .fragment_aligner import (
    AlignmentResult,
    estimate_shift,
    phase_correlation_align,
    template_match_align,
    apply_shift,
    batch_align,
)
from .score_aggregator import (
    AggregationResult,
    weighted_avg,
    harmonic_mean,
    aggregate_scores,
    threshold_filter,
    top_k_pairs,
    batch_aggregate,
)
from .edge_scorer import (
    EdgeScore,
    score_color_compat,
    score_gradient_compat,
    score_texture_compat,
    score_edge_pair,
    batch_score_edges,
)
from .position_estimator import (
    PositionEstimate,
    build_offset_graph,
    estimate_positions,
    refine_positions,
    positions_to_array,
    align_to_origin,
)
from .overlap_resolver import (
    OverlapConflict,
    compute_separation_vector,
    detect_overlap_conflicts,
    resolve_single_conflict,
    resolve_all_conflicts,
    conflict_score,
)
from .rotation_estimator import (
    RotationEstimate,
    estimate_by_pca,
    estimate_by_moments,
    estimate_by_gradient,
    refine_rotation,
    estimate_rotation_pair,
    batch_estimate_rotations,
)
from .gradient_flow import (
    GradientField,
    GradientStats,
    compute_gradient,
    compute_magnitude,
    compute_orientation,
    compute_divergence,
    compute_curl,
    flow_along_boundary,
    compare_gradient_fields,
    batch_gradient_fields,
    compute_gradient_stats,
)
from .region_segmenter import (
    RegionProps,
    SegmentationResult,
    label_connected,
    compute_region_props,
    filter_regions,
    merge_close_regions,
    region_adjacency,
    largest_region,
    regions_to_mask,
    batch_segment,
)
from .path_planner import (
    PathResult,
    dijkstra,
    shortest_path,
    all_pairs_shortest_paths,
    topological_sort,
    find_connected_components,
    minimum_spanning_tree,
    batch_dijkstra,
)
from .edge_extractor import (
    EdgeSegment,
    FragmentEdges,
    detect_boundary,
    extract_edge_points,
    split_edge_by_side,
    compute_edge_length,
    simplify_edge,
    extract_fragment_edges,
    batch_extract_edges,
)
from .contour_tracker import (
    ContourInfo,
    TrackState,
    find_contours,
    filter_contours,
    contour_to_array,
    compute_contour_info,
    match_contours,
    track_contour,
    batch_find_contours,
)
from .region_splitter import (
    RegionInfo,
    SplitResult,
    find_regions,
    filter_regions as filter_split_regions,
    region_masks,
    merge_small_regions,
    largest_region as largest_split_region,
    split_mask_to_crops,
    batch_find_regions,
)
from .texture_descriptor import (
    TextureDescriptor,
    compute_lbp,
    compute_glcm_features,
    compute_stats_descriptor,
    compute_texture_descriptor,
    normalize_descriptor,
    descriptor_distance,
    batch_compute_descriptors,
)
from .sift_matcher import (
    SiftConfig,
    MatchResult,
    extract_keypoints,
    match_descriptors,
    compute_homography,
    sift_match_pair,
    filter_matches_by_distance,
    batch_sift_match,
)
from .patch_matcher import (
    PatchConfig,
    PatchMatch,
    extract_patch,
    ncc_score,
    ssd_score,
    sad_score,
    match_patch_in_image,
    find_matches,
    top_matches,
    batch_patch_match,
)
from .descriptor_aggregator import (
    AggregatorConfig,
    AggregatedDescriptor,
    l2_normalize,
    concatenate_descriptors,
    weighted_average_descriptors,
    pca_reduce,
    elementwise_aggregate,
    aggregate,
    distance_matrix,
    batch_aggregate,
)
from .homography_estimator import (
    HomographyConfig,
    HomographyResult,
    normalize_points,
    dlt_homography,
    compute_reprojection_error,
    estimate_homography,
    decompose_homography,
    warp_points,
    batch_estimate_homographies,
)
from .descriptor_combiner import (
    CombineConfig,
    DescriptorSet,
    CombineResult,
    combine_descriptors,
    combine_selected,
    batch_combine,
    descriptor_distance as descriptor_distance_combine,
    build_distance_matrix,
    find_nearest,
)
from .edge_comparator import (
    CompareConfig,
    EdgeCompareResult,
    dtw_distance,
    css_similarity,
    fd_score,
    ifs_similarity,
    compare_edges,
    build_compat_matrix,
    top_k_matches,
)
from .boundary_descriptor import (
    DescriptorConfig,
    BoundaryDescriptor,
    compute_curvature,
    curvature_histogram,
    direction_histogram,
    chord_distribution,
    extract_descriptor,
    descriptor_similarity,
    batch_extract_descriptors,
)
from .contour_smoother import (
    SmootherConfig,
    SmoothedContour,
    smooth_gaussian,
    resample_contour,
    compute_arc_length,
    smooth_and_resample,
    align_contours,
    contour_similarity as smooth_contour_similarity,
    batch_smooth,
)

__all__ = [
    # Синтез
    "compute_fractal_signature",
    "build_edge_signatures",
    # Танграм
    "convex_hull",
    "rdp_simplify",
    "normalize_polygon",
    "classify_shape",
    "compute_interior_angles",
    "fit_tangram",
    # Фрактал
    "box_counting_fd",
    "divider_fd",
    "css_similarity_mirror",
    "fit_ifs",
    # Shape Context
    "compute_shape_context",
    "shape_context_distance",
    "match_shape_contexts",
    "normalize_shape_context",
    "log_polar_histogram",
    "contour_similarity",
    "ShapeContextResult",
    # Сегментация слов
    "WordBox",
    "LineSegment",
    "WordSegmentationResult",
    "binarize",
    "segment_words",
    "merge_line_words",
    "segment_lines",
    "segment_document",
    # Классификатор фрагментов
    "FragmentType",
    "FragmentFeatures",
    "ClassifyResult",
    "compute_texture_features",
    "compute_edge_features",
    "compute_shape_features",
    "detect_text_presence",
    "classify_fragment_type",
    "classify_fragment",
    "batch_classify",
    # Профили краёв
    "EdgeProfile",
    "ProfileMatchResult",
    "extract_intensity_profile",
    "extract_gradient_profile",
    "extract_texture_profile",
    "normalize_profile",
    "profile_correlation",
    "profile_dtw",
    "match_edge_profiles",
    "batch_profile_match",
    # Обнаружение строк
    "TextLine",
    "LineDetectionResult",
    "detect_lines_projection",
    "detect_lines_hough",
    "estimate_line_metrics",
    "filter_lines",
    "detect_text_lines",
    "batch_detect_lines",
    # Субпиксельное выравнивание
    "AlignmentResult",
    "estimate_shift",
    "phase_correlation_align",
    "template_match_align",
    "apply_shift",
    "batch_align",
    # Агрегация оценок совместимости
    "AggregationResult",
    "weighted_avg",
    "harmonic_mean",
    "aggregate_scores",
    "threshold_filter",
    "top_k_pairs",
    "batch_aggregate",
    # Многоканальная оценка краёв
    "EdgeScore",
    "score_color_compat",
    "score_gradient_compat",
    "score_texture_compat",
    "score_edge_pair",
    "batch_score_edges",
    # Оценка позиций фрагментов
    "PositionEstimate",
    "build_offset_graph",
    "estimate_positions",
    "refine_positions",
    "positions_to_array",
    "align_to_origin",
    # Разрешение перекрытий
    "OverlapConflict",
    "compute_separation_vector",
    "detect_overlap_conflicts",
    "resolve_single_conflict",
    "resolve_all_conflicts",
    "conflict_score",
    # Оценка угла поворота
    "RotationEstimate",
    "estimate_by_pca",
    "estimate_by_moments",
    "estimate_by_gradient",
    "refine_rotation",
    "estimate_rotation_pair",
    "batch_estimate_rotations",
    # Анализ градиентного поля
    "GradientField",
    "GradientStats",
    "compute_gradient",
    "compute_magnitude",
    "compute_orientation",
    "compute_divergence",
    "compute_curl",
    "flow_along_boundary",
    "compare_gradient_fields",
    "batch_gradient_fields",
    "compute_gradient_stats",
    # Сегментация на регионы
    "RegionProps",
    "SegmentationResult",
    "label_connected",
    "compute_region_props",
    "filter_regions",
    "merge_close_regions",
    "region_adjacency",
    "largest_region",
    "regions_to_mask",
    "batch_segment",
    # Планирование пути обхода фрагментов
    "PathResult",
    "dijkstra",
    "shortest_path",
    "all_pairs_shortest_paths",
    "topological_sort",
    "find_connected_components",
    "minimum_spanning_tree",
    "batch_dijkstra",
    # Извлечение граничных профилей фрагментов
    "EdgeSegment",
    "FragmentEdges",
    "detect_boundary",
    "extract_edge_points",
    "split_edge_by_side",
    "compute_edge_length",
    "simplify_edge",
    "extract_fragment_edges",
    "batch_extract_edges",
    # Отслеживание контуров
    "ContourInfo",
    "TrackState",
    "find_contours",
    "filter_contours",
    "contour_to_array",
    "compute_contour_info",
    "match_contours",
    "track_contour",
    "batch_find_contours",
    # Разделение на связные регионы
    "RegionInfo",
    "SplitResult",
    "find_regions",
    "filter_split_regions",
    "region_masks",
    "merge_small_regions",
    "largest_split_region",
    "split_mask_to_crops",
    "batch_find_regions",
    # Текстурные дескрипторы
    "TextureDescriptor",
    "compute_lbp",
    "compute_glcm_features",
    "compute_stats_descriptor",
    "compute_texture_descriptor",
    "normalize_descriptor",
    "descriptor_distance",
    "batch_compute_descriptors",
    # SIFT-сопоставление
    "SiftConfig",
    "MatchResult",
    "extract_keypoints",
    "match_descriptors",
    "compute_homography",
    "sift_match_pair",
    "filter_matches_by_distance",
    "batch_sift_match",
    # Патч-матчинг
    "PatchConfig",
    "PatchMatch",
    "extract_patch",
    "ncc_score",
    "ssd_score",
    "sad_score",
    "match_patch_in_image",
    "find_matches",
    "top_matches",
    "batch_patch_match",
    # Агрегация дескрипторов
    "AggregatorConfig",
    "AggregatedDescriptor",
    "l2_normalize",
    "concatenate_descriptors",
    "weighted_average_descriptors",
    "pca_reduce",
    "elementwise_aggregate",
    "aggregate",
    "distance_matrix",
    "batch_aggregate",
    # Оценка гомографии
    "HomographyConfig",
    "HomographyResult",
    "normalize_points",
    "dlt_homography",
    "compute_reprojection_error",
    "estimate_homography",
    "decompose_homography",
    "warp_points",
    "batch_estimate_homographies",
    # Комбинирование дескрипторов
    "CombineConfig",
    "DescriptorSet",
    "CombineResult",
    "combine_descriptors",
    "combine_selected",
    "batch_combine",
    "descriptor_distance_combine",
    "build_distance_matrix",
    "find_nearest",
    # Сравнение краёв фрагментов
    "CompareConfig",
    "EdgeCompareResult",
    "dtw_distance",
    "css_similarity",
    "fd_score",
    "ifs_similarity",
    "compare_edges",
    "build_compat_matrix",
    "top_k_matches",
    # Дескрипторы граничных сегментов
    "DescriptorConfig",
    "BoundaryDescriptor",
    "compute_curvature",
    "curvature_histogram",
    "direction_histogram",
    "chord_distribution",
    "extract_descriptor",
    "descriptor_similarity",
    "batch_extract_descriptors",
    # Сглаживание и передискретизация контуров
    "SmootherConfig",
    "SmoothedContour",
    "smooth_gaussian",
    "resample_contour",
    "compute_arc_length",
    "smooth_and_resample",
    "align_contours",
    "smooth_contour_similarity",
    "batch_smooth",
]
