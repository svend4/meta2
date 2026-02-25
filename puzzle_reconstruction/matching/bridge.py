"""
Мост интеграции сопоставления (Bridge #9) — реестр 20 sleeping matching modules.

Подключает модули puzzle_reconstruction/matching/, которые экспортированы в
__init__.py, но не использовались в основном пайплайне напрямую.

Уже подключённые модули (не требуют Bridge #9):
    compat_matrix   → Bridge #3 (build_compat_matrix в pipeline/main)
    consensus       → pipeline.py (run_consistency_check)
    pairwise        → compat_matrix.py (score_pair)
    patch_matcher   → compat_matrix.py (patch scoring)
    score_aggregator → compat_matrix.py (aggregate scores)
    score_normalizer → pipeline.py
    score_combiner  → compat_matrix.py (combine scores)

Sleeping-модули (20 штук), подключаемые через Bridge #9:
    affine_matcher   — аффинное сопоставление фрагментов.
    boundary_matcher — сопоставление по граничным профилям.
    candidate_ranker — ранжирование кандидатов по нескольким критериям.
    color_match      — сопоставление по цветовым гистограммам.
    curve_descriptor — дескрипторы кривых для сравнения контуров.
    dtw              — DTW-расстояние для временных рядов профилей.
    edge_comparator  — попарное сравнение краёв.
    feature_match    — SIFT/ORB-сопоставление ключевых точек.
    geometric_match  — геометрическая верификация сопоставлений.
    global_matcher   — глобальное многоступенчатое сопоставление.
    graph_match      — граф-матчинг (спектральный / венгерский).
    icp              — итерационное ближайших точек.
    matcher_registry — реестр матчеров (register / dispatch).
    orient_matcher   — сопоставление с учётом ориентации.
    pair_scorer      — взвешенная оценка пар.
    patch_validator  — валидация патчей по геометрии.
    seam_score       — оценка качества шва.
    shape_matcher    — сопоставление по форме (Hu-моменты, контур).
    spectral_matcher — спектральное сопоставление (Фурье-дескрипторы).
    texture_match    — сопоставление по текстуре.

Использование:
    from puzzle_reconstruction.matching.bridge import (
        build_matcher_registry,
        list_matchers,
        get_matcher,
        MATCHER_CATEGORIES,
    )

    fn = get_matcher("dtw_distance")
    if fn:
        dist = fn(profile_a, profile_b)
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── Категории матчеров ───────────────────────────────────────────────────────

MATCHER_CATEGORIES: Dict[str, List[str]] = {
    # Метрики и расстояния
    "distance": [
        "dtw_distance",
        "compute_fourier_descriptor",
        "compute_seam_score",
        "extract_edge_sample",
    ],
    # Точечное / геометрическое сопоставление
    "geometric": [
        "estimate_affine",
        "compute_fragment_geometry",
        "icp_align",
        "compute_orient_profile",
        "hu_moments",
    ],
    # Интенсивность / цвет / текстура
    "appearance": [
        "extract_boundary_points",
        "compute_color_histogram",
        "extract_features",
        "compute_lbp_histogram",
        "spectrum_correlation",
    ],
    # Patch и валидация
    "patch": [
        "validate_patch_pair",
    ],
    # Ранжирование и оценка
    "ranking": [
        "rank_pairs",
        "score_pair",
        "global_match",
    ],
    # Граф-матчинг
    "graph": [
        "build_fragment_graph",
    ],
    # Реестр матчеров (dispatcher pattern)
    "registry": [
        "register_matcher",
        "get_matcher_fn",
    ],
}


# ─── Реестр ───────────────────────────────────────────────────────────────────

def build_matcher_registry() -> Dict[str, Callable]:
    """
    Строит словарь {matcher_name: callable} для 20 sleeping matching-модулей.

    Returns:
        Словарь зарегистрированных функций сопоставления.
    """
    registry: Dict[str, Callable] = {}

    # ── dtw ────────────────────────────────────────────────────────────────

    try:
        from .dtw import dtw_distance
        registry["dtw_distance"] = dtw_distance
    except Exception:
        pass

    # ── curve_descriptor ───────────────────────────────────────────────────

    try:
        from .curve_descriptor import compute_fourier_descriptor
        registry["compute_fourier_descriptor"] = compute_fourier_descriptor
    except Exception:
        pass

    # ── seam_score ─────────────────────────────────────────────────────────

    try:
        from .seam_score import compute_seam_score
        registry["compute_seam_score"] = compute_seam_score
    except Exception:
        pass

    # ── edge_comparator ────────────────────────────────────────────────────

    try:
        from .edge_comparator import extract_edge_sample
        registry["extract_edge_sample"] = extract_edge_sample
    except Exception:
        pass

    # ── affine_matcher ─────────────────────────────────────────────────────

    try:
        from .affine_matcher import estimate_affine
        registry["estimate_affine"] = estimate_affine
    except Exception:
        pass

    # ── geometric_match ────────────────────────────────────────────────────

    try:
        from .geometric_match import compute_fragment_geometry
        registry["compute_fragment_geometry"] = compute_fragment_geometry
    except Exception:
        pass

    # ── icp ────────────────────────────────────────────────────────────────

    try:
        from .icp import icp_align
        registry["icp_align"] = icp_align
    except Exception:
        pass

    # ── orient_matcher ─────────────────────────────────────────────────────

    try:
        from .orient_matcher import compute_orient_profile
        registry["compute_orient_profile"] = compute_orient_profile
    except Exception:
        pass

    # ── shape_matcher ──────────────────────────────────────────────────────

    try:
        from .shape_matcher import hu_moments
        registry["hu_moments"] = hu_moments
    except Exception:
        pass

    # ── boundary_matcher ───────────────────────────────────────────────────

    try:
        from .boundary_matcher import extract_boundary_points
        registry["extract_boundary_points"] = extract_boundary_points
    except Exception:
        pass

    # ── color_match ────────────────────────────────────────────────────────

    try:
        from .color_match import compute_color_histogram
        registry["compute_color_histogram"] = compute_color_histogram
    except Exception:
        pass

    # ── feature_match ──────────────────────────────────────────────────────

    try:
        from .feature_match import extract_features
        registry["extract_features"] = extract_features
    except Exception:
        pass

    # ── texture_match ──────────────────────────────────────────────────────

    try:
        from .texture_match import compute_lbp_histogram
        registry["compute_lbp_histogram"] = compute_lbp_histogram
    except Exception:
        pass

    # ── spectral_matcher ───────────────────────────────────────────────────

    try:
        from .spectral_matcher import spectrum_correlation
        registry["spectrum_correlation"] = spectrum_correlation
    except Exception:
        pass

    # ── patch_validator ────────────────────────────────────────────────────

    try:
        from .patch_validator import validate_patch_pair
        registry["validate_patch_pair"] = validate_patch_pair
    except Exception:
        pass

    # ── candidate_ranker ───────────────────────────────────────────────────

    try:
        from .candidate_ranker import rank_pairs
        registry["rank_pairs"] = rank_pairs
    except Exception:
        pass

    # ── pair_scorer ────────────────────────────────────────────────────────

    try:
        from .pair_scorer import score_pair
        registry["score_pair"] = score_pair
    except Exception:
        pass

    # ── global_matcher ─────────────────────────────────────────────────────

    try:
        from .global_matcher import global_match
        registry["global_match"] = global_match
    except Exception:
        pass

    # ── graph_match ────────────────────────────────────────────────────────

    try:
        from .graph_match import build_fragment_graph
        registry["build_fragment_graph"] = build_fragment_graph
    except Exception:
        pass

    # ── matcher_registry ───────────────────────────────────────────────────

    try:
        from .matcher_registry import register as register_matcher
        registry["register_matcher"] = register_matcher
    except Exception:
        pass

    try:
        from .matcher_registry import get_matcher as get_matcher_fn
        registry["get_matcher_fn"] = get_matcher_fn
    except Exception:
        pass

    return registry


# ─── Глобальный реестр ────────────────────────────────────────────────────────

MATCHER_REGISTRY: Dict[str, Callable] = {}


def _ensure_registry() -> None:
    global MATCHER_REGISTRY
    if not MATCHER_REGISTRY:
        MATCHER_REGISTRY = build_matcher_registry()


def list_matchers(category: Optional[str] = None) -> List[str]:
    """
    Список всех зарегистрированных matcher-функций.

    Args:
        category: Фильтр по категории ('distance', 'geometric', 'appearance',
                  'patch', 'ranking', 'graph', 'registry'). None → все.

    Returns:
        Отсортированный список имён.
    """
    _ensure_registry()
    if category is not None:
        names = set(MATCHER_CATEGORIES.get(category, []))
        return sorted(n for n in MATCHER_REGISTRY if n in names)
    return sorted(MATCHER_REGISTRY.keys())


def get_matcher(name: str) -> Optional[Callable]:
    """
    Возвращает callable по имени matcher-функции.

    Returns:
        Callable или None если функция недоступна.
    """
    _ensure_registry()
    fn = MATCHER_REGISTRY.get(name)
    if fn is None:
        logger.debug("matcher %r not available", name)
    return fn


def get_matcher_category(name: str) -> Optional[str]:
    """Возвращает категорию matcher-функции или None."""
    for cat, names in MATCHER_CATEGORIES.items():
        if name in names:
            return cat
    return None
