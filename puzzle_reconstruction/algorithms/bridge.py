"""
Мост интеграции алгоритмов (Bridge №5) — реестр 34 sleeping modules.

Подключает модули puzzle_reconstruction/algorithms/, которые были
экспортированы в __init__.py, но не использовались в основном пайплайне.

Архитектура:
    Алгоритмы разделены на три категории по уровню применения:

    fragment   — вычисляются один раз на фрагмент (в _process_one):
                 дескрипторы формы, качество, классификация, ориентация

    pair       — вычисляются для пары краёв/фрагментов (в match/verify):
                 компараторы, оценщики швов, выравниватели патчей

    assembly   — применяются на уровне всей сборки (в assemble):
                 планировщик путей, оценщик перекрытий, позиционирование

Использование:
    from puzzle_reconstruction.algorithms.bridge import (
        build_algorithm_registry,
        list_algorithms,
        get_algorithm,
        ALGORITHM_CATEGORIES,
    )

    registry = build_algorithm_registry()
    alg = registry.get("fragment_classifier")
    if alg:
        result = alg(fragment)

    # Получить только fragment-уровневые алгоритмы:
    names = ALGORITHM_CATEGORIES["fragment"]
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── Категории алгоритмов ─────────────────────────────────────────────────────

ALGORITHM_CATEGORIES: Dict[str, List[str]] = {
    # Применяются к одному Fragment (image / contour / mask)
    "fragment": [
        "boundary_descriptor",
        "color_palette",
        "color_space",
        "contour_smoother",
        "contour_tracker",
        "edge_extractor",
        "edge_profile",
        "fourier_descriptor",
        "fragment_classifier",
        "fragment_quality",
        "gradient_flow",
        "line_detector",
        "region_segmenter",
        "region_splitter",
        "rotation_estimator",
        "shape_context",
        "texture_descriptor",
        "word_segmentation",
    ],
    # Применяются к паре краёв / фрагментов (img_a, img_b, side_a, side_b, ...)
    "pair": [
        "edge_comparator",
        "edge_filter",
        "edge_scorer",
        "fragment_aligner",
        "homography_estimator",
        "overlap_resolver",
        "patch_aligner",
        "patch_matcher",
        "region_scorer",
        "score_aggregator",
        "seam_evaluator",
        "sift_matcher",
    ],
    # Применяются ко всей сборке (Assembly / fragments / compat_matrix)
    "assembly": [
        "descriptor_aggregator",
        "descriptor_combiner",
        "path_planner",
        "position_estimator",
    ],
}


# ─── Реестр алгоритмов ────────────────────────────────────────────────────────

def build_algorithm_registry() -> Dict[str, Callable[..., Any]]:
    """
    Строит словарь {algorithm_name: main_callable}.

    Каждый callable принимает аргументы, специфичные для алгоритма.
    Все import-ы выполняются внутри try/except, поэтому недоступные
    зависимости не останавливают инициализацию.

    Returns:
        Словарь зарегистрированных алгоритмов.
    """
    registry: Dict[str, Callable[..., Any]] = {}

    # ── FRAGMENT-LEVEL ─────────────────────────────────────────────────────────

    # boundary_descriptor: extract_descriptor(points, fragment_id, edge_id, cfg)
    try:
        from .boundary_descriptor import extract_descriptor
        registry["boundary_descriptor"] = extract_descriptor
    except Exception:
        pass

    # color_palette: compute_palette(img, fragment_id, cfg)
    try:
        from .color_palette import compute_palette
        registry["color_palette"] = compute_palette
    except Exception:
        pass

    # color_space: compute_color_histogram(img, cfg, fragment_id)
    try:
        from .color_space import compute_color_histogram
        registry["color_space"] = compute_color_histogram
    except Exception:
        pass

    # contour_smoother: smooth_and_resample(points, cfg)
    try:
        from .contour_smoother import smooth_and_resample
        registry["contour_smoother"] = smooth_and_resample
    except Exception:
        pass

    # contour_tracker: track_contour(state, new_info)
    try:
        from .contour_tracker import track_contour
        registry["contour_tracker"] = track_contour
    except Exception:
        pass

    # edge_extractor: extract_fragment_edges(img, threshold, epsilon)
    try:
        from .edge_extractor import extract_fragment_edges
        registry["edge_extractor"] = extract_fragment_edges
    except Exception:
        pass

    # edge_profile: match_edge_profiles(img1, img2, side1, side2, ...)
    try:
        from .edge_profile import match_edge_profiles
        registry["edge_profile"] = match_edge_profiles
    except Exception:
        pass

    # fourier_descriptor: compute_fd(points, fragment_id, edge_id, cfg)
    try:
        from .fourier_descriptor import compute_fd
        registry["fourier_descriptor"] = compute_fd
    except Exception:
        pass

    # fragment_classifier: classify_fragment(img, ...)
    try:
        from .fragment_classifier import classify_fragment
        registry["fragment_classifier"] = classify_fragment
    except Exception:
        pass

    # fragment_quality: assess_fragment(img, mask, cfg, fragment_id)
    try:
        from .fragment_quality import assess_fragment
        registry["fragment_quality"] = assess_fragment
    except Exception:
        pass

    # gradient_flow: compute_gradient_stats(field, threshold, n_orientation_bins)
    try:
        from .gradient_flow import compute_gradient_stats, compute_gradient
        registry["gradient_flow"] = compute_gradient
    except Exception:
        pass

    # line_detector: detect_text_lines(img, method, ...)
    try:
        from .line_detector import detect_text_lines
        registry["line_detector"] = detect_text_lines
    except Exception:
        pass

    # region_segmenter: label_connected(img, connectivity, threshold)
    try:
        from .region_segmenter import label_connected
        registry["region_segmenter"] = label_connected
    except Exception:
        pass

    # region_splitter: find_regions(mask)
    try:
        from .region_splitter import find_regions
        registry["region_splitter"] = find_regions
    except Exception:
        pass

    # rotation_estimator: estimate_rotation_pair(img1, img2, method)
    try:
        from .rotation_estimator import estimate_rotation_pair, batch_estimate_rotations
        registry["rotation_estimator"] = estimate_rotation_pair
    except Exception:
        pass

    # shape_context: contour_similarity(contour_a, contour_b, ...)
    try:
        from .shape_context import contour_similarity
        registry["shape_context"] = contour_similarity
    except Exception:
        pass

    # texture_descriptor: compute_texture_descriptor(img, method, ...)
    try:
        from .texture_descriptor import compute_texture_descriptor
        registry["texture_descriptor"] = compute_texture_descriptor
    except Exception:
        pass

    # word_segmentation: segment_document(img, ...)
    try:
        from .word_segmentation import segment_document
        registry["word_segmentation"] = segment_document
    except Exception:
        pass

    # ── PAIR-LEVEL ─────────────────────────────────────────────────────────────

    # edge_comparator: compare_edges(edge_a, edge_b, cfg)
    try:
        from .edge_comparator import compare_edges
        registry["edge_comparator"] = compare_edges
    except Exception:
        pass

    # edge_filter: apply_edge_filter(results, cfg)
    try:
        from .edge_filter import apply_edge_filter
        registry["edge_filter"] = apply_edge_filter
    except Exception:
        pass

    # edge_scorer: score_edge_pair(img1, img2, idx1, idx2, side1, side2, ...)
    try:
        from .edge_scorer import score_edge_pair
        registry["edge_scorer"] = score_edge_pair
    except Exception:
        pass

    # fragment_aligner: align_patches(patch_a, patch_b, cfg)
    try:
        from .fragment_aligner import align_patches
        registry["fragment_aligner"] = align_patches
    except Exception:
        pass

    # homography_estimator: estimate_homography(src_pts, dst_pts, cfg)
    try:
        from .homography_estimator import estimate_homography
        registry["homography_estimator"] = estimate_homography
    except Exception:
        pass

    # overlap_resolver: resolve_all_conflicts(state, contours, ...)
    try:
        from .overlap_resolver import resolve_all_conflicts
        registry["overlap_resolver"] = resolve_all_conflicts
    except Exception:
        pass

    # patch_aligner: align_patches(patch_a, patch_b, cfg)
    try:
        from .patch_aligner import align_patches as _pa_align
        registry["patch_aligner"] = _pa_align
    except Exception:
        pass

    # patch_matcher: find_matches(img1, img2, cfg)
    try:
        from .patch_matcher import find_matches
        registry["patch_matcher"] = find_matches
    except Exception:
        pass

    # region_scorer: score_region_pair(patch_a, bbox_a, patch_b, bbox_b, cfg)
    try:
        from .region_scorer import score_region_pair
        registry["region_scorer"] = score_region_pair
    except Exception:
        pass

    # score_aggregator: aggregate_scores(scores, weights, method)
    try:
        from .score_aggregator import aggregate_scores
        registry["score_aggregator"] = aggregate_scores
    except Exception:
        pass

    # seam_evaluator: evaluate_seam(img_a, side_a, img_b, side_b, cfg)
    try:
        from .seam_evaluator import evaluate_seam
        registry["seam_evaluator"] = evaluate_seam
    except Exception:
        pass

    # sift_matcher: sift_match_pair(img1, img2, cfg)
    try:
        from .sift_matcher import sift_match_pair
        registry["sift_matcher"] = sift_match_pair
    except Exception:
        pass

    # ── ASSEMBLY-LEVEL ─────────────────────────────────────────────────────────

    # descriptor_aggregator: aggregate(descriptors, cfg)
    try:
        from .descriptor_aggregator import aggregate
        registry["descriptor_aggregator"] = aggregate
    except Exception:
        pass

    # descriptor_combiner: combine_descriptors(desc_set, cfg)
    try:
        from .descriptor_combiner import combine_descriptors
        registry["descriptor_combiner"] = combine_descriptors
    except Exception:
        pass

    # path_planner: shortest_path(score_matrix, start, end)
    try:
        from .path_planner import shortest_path
        registry["path_planner"] = shortest_path
    except Exception:
        pass

    # position_estimator: estimate_positions(offset_graph, root)
    try:
        from .position_estimator import estimate_positions
        registry["position_estimator"] = estimate_positions
    except Exception:
        pass

    return registry


# ─── Глобальный реестр ────────────────────────────────────────────────────────

ALGORITHM_REGISTRY: Dict[str, Callable[..., Any]] = {}


def _ensure_registry() -> None:
    global ALGORITHM_REGISTRY
    if not ALGORITHM_REGISTRY:
        ALGORITHM_REGISTRY = build_algorithm_registry()


def list_algorithms(category: Optional[str] = None) -> List[str]:
    """
    Список всех зарегистрированных алгоритмов.

    Args:
        category: Фильтр по категории ('fragment', 'pair', 'assembly').
                  None → все категории.

    Returns:
        Отсортированный список имён алгоритмов.
    """
    _ensure_registry()
    if category is not None:
        names = set(ALGORITHM_CATEGORIES.get(category, []))
        return sorted(n for n in ALGORITHM_REGISTRY if n in names)
    return sorted(ALGORITHM_REGISTRY.keys())


def get_algorithm(name: str) -> Optional[Callable[..., Any]]:
    """
    Возвращает callable для алгоритма по имени.

    Args:
        name: Имя алгоритма (из ALGORITHM_CATEGORIES).

    Returns:
        Callable или None если алгоритм недоступен.
    """
    _ensure_registry()
    fn = ALGORITHM_REGISTRY.get(name)
    if fn is None:
        logger.debug("algorithm %r not available", name)
    return fn


def get_category(name: str) -> Optional[str]:
    """Возвращает категорию алгоритма ('fragment'/'pair'/'assembly') или None."""
    for cat, names in ALGORITHM_CATEGORIES.items():
        if name in names:
            return cat
    return None
