"""
Мост интеграции оценки (Bridge #8) — реестр 9 sleeping scoring modules.

Подключает модули puzzle_reconstruction/scoring/, которые экспортированы в
__init__.py, но не использовались в основном пайплайне напрямую.

Уже подключённые модули (не требуют Bridge #8):
    score_normalizer   → pipeline.py (normalize_score_matrix)
    threshold_selector → pipeline.py (select_threshold)
    consistency_checker → pipeline.py (_consistency_check)

Sleeping-модули (9 штук), подключаемые через Bridge #8:
    boundary_scorer    — оценка граничной совместимости (интенсивность,
                         градиент, цвет) для пар смежных фрагментов.
    evidence_aggregator — агрегация разнородных оценок в EvidenceScore.
    gap_scorer         — оценка зазора / перекрытия между фрагментами.
    global_ranker      — глобальное ранжирование пар по матрице оценок.
    match_evaluator    — оценка качества совпадения (TP/FP/FN → precision/recall).
    match_scorer       — попиксельная оценка совместимости по каналам.
    pair_filter        — фильтрация кандидатных пар по порогам.
    pair_ranker        — ранжирование пар кандидатов.
    rank_fusion        — слияние нескольких рейтингов (RRF / Borda / взвешенное).

Использование:
    from puzzle_reconstruction.scoring.bridge import (
        build_scoring_registry,
        list_scorers,
        get_scorer,
        SCORING_CATEGORIES,
    )

    registry = build_scoring_registry()
    score_boundary = registry.get("score_boundary")
    if score_boundary:
        result = score_boundary(img1, img2)

    # Применить pipeline: filter → rank → fuse
    fuse = registry.get("fuse_rankings")
    if fuse:
        fused_matrix = fuse({"css": css_scores, "dtw": dtw_scores})
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── Категории ────────────────────────────────────────────────────────────────

SCORING_CATEGORIES: Dict[str, List[str]] = {
    # Попарная оценка совместимости
    "pair": [
        "score_boundary",
        "batch_score_boundaries",
        "score_boundary_matrix",
        "score_match",
        "score_gap",
        "aggregate_evidence",
        "rank_by_evidence",
    ],
    # Фильтрация и ранжирование кандидатов
    "filter_rank": [
        "filter_pairs",
        "compute_pair_score",
        "rank_pairs",
    ],
    # Слияние рейтингов
    "fusion": [
        "fuse_rankings",
        "reciprocal_rank_fusion",
        "borda_count",
        "score_fusion",
        "normalize_scores",
    ],
    # Оценка матчинга (TP/FP/FN)
    "evaluation": [
        "evaluate_match",
    ],
}


# ─── Реестр ───────────────────────────────────────────────────────────────────

def build_scoring_registry() -> Dict[str, Callable]:
    """
    Строит словарь {scorer_name: callable} для 9 sleeping scoring-модулей.

    Все import-ы выполняются внутри try/except, поэтому недоступные
    зависимости не останавливают инициализацию.

    Returns:
        Словарь зарегистрированных scorer-функций.
    """
    registry: Dict[str, Callable] = {}

    # ── boundary_scorer ────────────────────────────────────────────────────

    try:
        from .boundary_scorer import score_boundary
        registry["score_boundary"] = score_boundary
    except Exception:
        pass

    try:
        from .boundary_scorer import batch_score_boundaries
        registry["batch_score_boundaries"] = batch_score_boundaries
    except Exception:
        pass

    try:
        from .boundary_scorer import score_matrix as score_boundary_matrix
        registry["score_boundary_matrix"] = score_boundary_matrix
    except Exception:
        pass

    # ── match_scorer ───────────────────────────────────────────────────────

    try:
        from .match_scorer import score_channel as score_match
        registry["score_match"] = score_match
    except Exception:
        pass

    # ── gap_scorer ─────────────────────────────────────────────────────────

    try:
        from .gap_scorer import score_gap
        registry["score_gap"] = score_gap
    except Exception:
        pass

    # ── evidence_aggregator ────────────────────────────────────────────────

    try:
        from .evidence_aggregator import aggregate_evidence
        registry["aggregate_evidence"] = aggregate_evidence
    except Exception:
        pass

    try:
        from .evidence_aggregator import rank_by_evidence
        registry["rank_by_evidence"] = rank_by_evidence
    except Exception:
        pass

    # ── pair_filter ────────────────────────────────────────────────────────

    try:
        from .pair_filter import filter_pairs
        registry["filter_pairs"] = filter_pairs
    except Exception:
        pass

    # ── pair_ranker ────────────────────────────────────────────────────────

    try:
        from .pair_ranker import compute_pair_score
        registry["compute_pair_score"] = compute_pair_score
    except Exception:
        pass

    try:
        from .pair_ranker import rank_pairs as rank_pairs_fn
        registry["rank_pairs"] = rank_pairs_fn
    except Exception:
        pass

    # ── global_ranker ──────────────────────────────────────────────────────

    try:
        from .global_ranker import rank_pairs as global_rank_pairs
        # Используем имя из модуля, чтобы не конфликтовать с pair_ranker
        if "rank_pairs" not in registry:
            registry["rank_pairs"] = global_rank_pairs
        registry["global_rank_pairs"] = global_rank_pairs
    except Exception:
        pass

    # ── rank_fusion ────────────────────────────────────────────────────────

    try:
        from .rank_fusion import fuse_rankings
        registry["fuse_rankings"] = fuse_rankings
    except Exception:
        pass

    try:
        from .rank_fusion import reciprocal_rank_fusion
        registry["reciprocal_rank_fusion"] = reciprocal_rank_fusion
    except Exception:
        pass

    try:
        from .rank_fusion import borda_count
        registry["borda_count"] = borda_count
    except Exception:
        pass

    try:
        from .rank_fusion import score_fusion
        registry["score_fusion"] = score_fusion
    except Exception:
        pass

    try:
        from .rank_fusion import normalize_scores
        registry["normalize_scores"] = normalize_scores
    except Exception:
        pass

    # ── match_evaluator ────────────────────────────────────────────────────

    try:
        from .match_evaluator import evaluate_match
        registry["evaluate_match"] = evaluate_match
    except Exception:
        pass

    return registry


# ─── Глобальный реестр ────────────────────────────────────────────────────────

SCORING_REGISTRY: Dict[str, Callable] = {}


def _ensure_registry() -> None:
    global SCORING_REGISTRY
    if not SCORING_REGISTRY:
        SCORING_REGISTRY = build_scoring_registry()


def list_scorers(category: Optional[str] = None) -> List[str]:
    """
    Список всех зарегистрированных scorer-функций.

    Args:
        category: Фильтр по категории ('pair', 'filter_rank', 'fusion',
                  'evaluation'). None → все категории.

    Returns:
        Отсортированный список имён.
    """
    _ensure_registry()
    if category is not None:
        names = set(SCORING_CATEGORIES.get(category, []))
        return sorted(n for n in SCORING_REGISTRY if n in names)
    return sorted(SCORING_REGISTRY.keys())


def get_scorer(name: str) -> Optional[Callable]:
    """
    Возвращает callable по имени scorer-функции.

    Args:
        name: Имя функции из SCORING_CATEGORIES.

    Returns:
        Callable или None если функция недоступна.
    """
    _ensure_registry()
    fn = SCORING_REGISTRY.get(name)
    if fn is None:
        logger.debug("scorer %r not available", name)
    return fn


def get_scorer_category(name: str) -> Optional[str]:
    """Возвращает категорию scorer-функции или None."""
    for cat, names in SCORING_CATEGORIES.items():
        if name in names:
            return cat
    return None
