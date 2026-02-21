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
]
