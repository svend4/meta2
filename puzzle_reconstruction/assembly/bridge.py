"""
Мост интеграции сборки (Bridge #10) — реестр 16 sleeping assembly modules.

Подключает модули puzzle_reconstruction/assembly/, которые экспортированы в
__init__.py, но не использовались в основном пайплайне напрямую.

Уже подключённые модули (не требуют Bridge #10):
    greedy           → pipeline.py / parallel.py
    annealing        → pipeline.py / parallel.py
    beam_search      → pipeline.py / parallel.py
    gamma_optimizer  → pipeline.py / parallel.py
    genetic          → parallel.py
    ant_colony       → parallel.py
    mcts             → parallel.py
    exhaustive       → parallel.py
    parallel         → pipeline.py
    position_estimator → pipeline.py
    overlap_resolver → pipeline.py

Sleeping-модули (16 штук), подключаемые через Bridge #10:
    assembly_state     — состояние сборки (размещения фрагментов).
    candidate_filter   — фильтрация пар-кандидатов по порогу и топ-K.
    canvas_builder     — построение холста для рендеринга сборки.
    collision_detector — обнаружение столкновений между фрагментами.
    cost_matrix        — матрица стоимостей для оптимизации сборки.
    fragment_arranger  — расстановка фрагментов на сетке/полосе.
    fragment_mapper    — разбиение холста на зоны и маппинг фрагментов.
    fragment_scorer    — оценка качества размещения отдельных фрагментов.
    fragment_sequencer — определение порядка размещения фрагментов.
    fragment_sorter    — сортировка фрагментов по разным критериям.
    gap_analyzer       — анализ зазоров между соседними фрагментами.
    layout_builder     — построение макета сборки из ячеек.
    layout_refiner     — уточнение макета с оптимизацией смещений.
    placement_optimizer — оптимизация размещения жадным методом.
    score_tracker      — отслеживание динамики оценок сборки.
    sequence_planner   — планирование последовательности размещений.

Использование:
    from puzzle_reconstruction.assembly.bridge import (
        build_assembly_registry,
        list_assembly_fns,
        get_assembly_fn,
        ASSEMBLY_CATEGORIES,
    )

    fn = get_assembly_fn("analyze_all_gaps")
    if fn:
        gaps = fn(placements)
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── Категории ────────────────────────────────────────────────────────────────

ASSEMBLY_CATEGORIES: Dict[str, List[str]] = {
    # Состояние и холст
    "state": [
        "create_state",
        "place_fragment",
        "make_empty_canvas",
        "compute_canvas_size",
    ],
    # Фильтрация и ранжирование кандидатов
    "filter": [
        "filter_by_threshold",
        "filter_top_k",
        "filter_by_rank",
    ],
    # Геометрия и коллизии
    "geometry": [
        "aabb_overlap",
        "detect_collisions",
        "compute_gap",
        "find_adjacent",
        "analyze_all_gaps",
    ],
    # Матрица стоимостей
    "cost": [
        "build_from_scores",
        "build_combined",
    ],
    # Расстановка и компоновка
    "layout": [
        "arrange_grid",
        "arrange_strip",
        "create_layout",
        "refine_layout",
        "build_fragment_map",
    ],
    # Оценка и оптимизация
    "scoring": [
        "score_fragment",
        "score_assembly",
        "score_placement",
        "find_best_next",
        "greedy_place",
    ],
    # Сортировка и порядок
    "sequencing": [
        "sequence_greedy",
        "sort_by_score",
        "build_placement_plan",
    ],
    # Трекинг
    "tracking": [
        "create_tracker",
        "record_snapshot",
        "detect_convergence",
    ],
}


# ─── Реестр ───────────────────────────────────────────────────────────────────

def build_assembly_registry() -> Dict[str, Callable]:
    """
    Строит словарь {fn_name: callable} для 16 sleeping assembly-модулей.

    Returns:
        Словарь зарегистрированных функций сборки.
    """
    registry: Dict[str, Callable] = {}

    # ── assembly_state ─────────────────────────────────────────────────────

    try:
        from .assembly_state import create_state
        registry["create_state"] = create_state
    except Exception:
        pass

    try:
        from .assembly_state import place_fragment
        registry["place_fragment"] = place_fragment
    except Exception:
        pass

    # ── candidate_filter ───────────────────────────────────────────────────

    try:
        from .candidate_filter import filter_by_threshold
        registry["filter_by_threshold"] = filter_by_threshold
    except Exception:
        pass

    try:
        from .candidate_filter import filter_top_k
        registry["filter_top_k"] = filter_top_k
    except Exception:
        pass

    try:
        from .candidate_filter import filter_by_rank
        registry["filter_by_rank"] = filter_by_rank
    except Exception:
        pass

    # ── canvas_builder ─────────────────────────────────────────────────────

    try:
        from .canvas_builder import compute_canvas_size
        registry["compute_canvas_size"] = compute_canvas_size
    except Exception:
        pass

    try:
        from .canvas_builder import make_empty_canvas
        registry["make_empty_canvas"] = make_empty_canvas
    except Exception:
        pass

    # ── collision_detector ─────────────────────────────────────────────────

    try:
        from .collision_detector import aabb_overlap
        registry["aabb_overlap"] = aabb_overlap
    except Exception:
        pass

    try:
        from .collision_detector import detect_collisions
        registry["detect_collisions"] = detect_collisions
    except Exception:
        pass

    # ── cost_matrix ────────────────────────────────────────────────────────

    try:
        from .cost_matrix import build_from_scores
        registry["build_from_scores"] = build_from_scores
    except Exception:
        pass

    try:
        from .cost_matrix import build_combined
        registry["build_combined"] = build_combined
    except Exception:
        pass

    # ── fragment_arranger ──────────────────────────────────────────────────

    try:
        from .fragment_arranger import arrange_grid
        registry["arrange_grid"] = arrange_grid
    except Exception:
        pass

    try:
        from .fragment_arranger import arrange_strip
        registry["arrange_strip"] = arrange_strip
    except Exception:
        pass

    # ── fragment_mapper ────────────────────────────────────────────────────

    try:
        from .fragment_mapper import build_fragment_map
        registry["build_fragment_map"] = build_fragment_map
    except Exception:
        pass

    # ── fragment_scorer ────────────────────────────────────────────────────

    try:
        from .fragment_scorer import score_fragment
        registry["score_fragment"] = score_fragment
    except Exception:
        pass

    try:
        from .fragment_scorer import score_assembly
        registry["score_assembly"] = score_assembly
    except Exception:
        pass

    # ── fragment_sequencer ─────────────────────────────────────────────────

    try:
        from .fragment_sequencer import sequence_greedy
        registry["sequence_greedy"] = sequence_greedy
    except Exception:
        pass

    # ── fragment_sorter ────────────────────────────────────────────────────

    try:
        from .fragment_sorter import sort_by_score
        registry["sort_by_score"] = sort_by_score
    except Exception:
        pass

    # ── gap_analyzer ───────────────────────────────────────────────────────

    try:
        from .gap_analyzer import compute_gap
        registry["compute_gap"] = compute_gap
    except Exception:
        pass

    try:
        from .gap_analyzer import find_adjacent
        registry["find_adjacent"] = find_adjacent
    except Exception:
        pass

    try:
        from .gap_analyzer import analyze_all_gaps
        registry["analyze_all_gaps"] = analyze_all_gaps
    except Exception:
        pass

    # ── layout_builder ─────────────────────────────────────────────────────

    try:
        from .layout_builder import create_layout
        registry["create_layout"] = create_layout
    except Exception:
        pass

    # ── layout_refiner ─────────────────────────────────────────────────────

    try:
        from .layout_refiner import refine_layout
        registry["refine_layout"] = refine_layout
    except Exception:
        pass

    # ── placement_optimizer ────────────────────────────────────────────────

    try:
        from .placement_optimizer import score_placement
        registry["score_placement"] = score_placement
    except Exception:
        pass

    try:
        from .placement_optimizer import find_best_next
        registry["find_best_next"] = find_best_next
    except Exception:
        pass

    try:
        from .placement_optimizer import greedy_place
        registry["greedy_place"] = greedy_place
    except Exception:
        pass

    # ── score_tracker ──────────────────────────────────────────────────────

    try:
        from .score_tracker import create_tracker
        registry["create_tracker"] = create_tracker
    except Exception:
        pass

    try:
        from .score_tracker import record_snapshot
        registry["record_snapshot"] = record_snapshot
    except Exception:
        pass

    try:
        from .score_tracker import detect_convergence
        registry["detect_convergence"] = detect_convergence
    except Exception:
        pass

    # ── sequence_planner ───────────────────────────────────────────────────

    try:
        from .sequence_planner import build_placement_plan
        registry["build_placement_plan"] = build_placement_plan
    except Exception:
        pass

    return registry


# ─── Глобальный реестр ────────────────────────────────────────────────────────

ASSEMBLY_REGISTRY: Dict[str, Callable] = {}


def _ensure_registry() -> None:
    global ASSEMBLY_REGISTRY
    if not ASSEMBLY_REGISTRY:
        ASSEMBLY_REGISTRY = build_assembly_registry()


def list_assembly_fns(category: Optional[str] = None) -> List[str]:
    """
    Список всех зарегистрированных assembly-функций.

    Args:
        category: Фильтр по категории ('state', 'filter', 'geometry', 'cost',
                  'layout', 'scoring', 'sequencing', 'tracking'). None → все.

    Returns:
        Отсортированный список имён.
    """
    _ensure_registry()
    if category is not None:
        names = set(ASSEMBLY_CATEGORIES.get(category, []))
        return sorted(n for n in ASSEMBLY_REGISTRY if n in names)
    return sorted(ASSEMBLY_REGISTRY.keys())


def get_assembly_fn(name: str) -> Optional[Callable]:
    """
    Возвращает callable по имени assembly-функции.

    Returns:
        Callable или None если функция недоступна.
    """
    _ensure_registry()
    fn = ASSEMBLY_REGISTRY.get(name)
    if fn is None:
        logger.debug("assembly fn %r not available", name)
    return fn


def get_assembly_category(name: str) -> Optional[str]:
    """Возвращает категорию assembly-функции или None."""
    for cat, names in ASSEMBLY_CATEGORIES.items():
        if name in names:
            return cat
    return None
