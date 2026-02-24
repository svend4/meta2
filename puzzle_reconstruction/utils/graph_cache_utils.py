"""Утилиты кеширования графовых вычислений и управления батч-обработкой.

Модуль предоставляет вспомогательные структуры и функции для:
- Хранения и агрегации результатов графовых алгоритмов
- Управления результатами батч-обработки с кешированием
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# ═══════════════════════════════════════════════════════════════════════════════
# Graph Algorithm Result Analysis
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GraphAlgoConfig:
    """Конфигурация анализа результатов графовых алгоритмов.

    Атрибуты:
        max_cost:         Максимально допустимая стоимость пути.
        min_path_length:  Минимальная длина пути (число узлов).
        require_found:    Учитывать только найденные пути.
    """
    max_cost: float = float("inf")
    min_path_length: int = 1
    require_found: bool = True


@dataclass
class GraphAlgoEntry:
    """Запись результата одного прогона графового алгоритма.

    Атрибуты:
        run_id:       Идентификатор прогона.
        algorithm:    Название алгоритма (например, 'dijkstra').
        found:        Был ли путь найден.
        cost:         Стоимость пути (0.0 если не найден).
        path_length:  Число узлов в пути.
        n_nodes:      Размер графа (число узлов).
        params:       Дополнительные параметры.
    """
    run_id: int
    algorithm: str
    found: bool
    cost: float
    path_length: int
    n_nodes: int
    params: Dict = field(default_factory=dict)


@dataclass
class GraphAlgoSummary:
    """Сводка результатов прогонов графового алгоритма.

    Атрибуты:
        n_runs:       Число прогонов.
        n_found:      Число найденных путей.
        mean_cost:    Средняя стоимость (только найденные).
        min_cost:     Минимальная стоимость (только найденные) или None.
        best_run_id:  ID прогона с минимальной стоимостью или None.
        worst_run_id: ID прогона с максимальной стоимостью или None.
    """
    n_runs: int
    n_found: int
    mean_cost: float
    min_cost: Optional[float]
    best_run_id: Optional[int]
    worst_run_id: Optional[int]


def make_graph_algo_entry(
    run_id: int,
    algorithm: str,
    found: bool,
    cost: float,
    path_length: int,
    n_nodes: int,
    **params,
) -> GraphAlgoEntry:
    """Создать запись результата графового алгоритма.

    Args:
        run_id:      Идентификатор прогона.
        algorithm:   Название алгоритма.
        found:       Найден ли путь.
        cost:        Стоимость пути.
        path_length: Число узлов в пути.
        n_nodes:     Размер графа.
        **params:    Дополнительные параметры.

    Returns:
        :class:`GraphAlgoEntry`.
    """
    return GraphAlgoEntry(
        run_id=run_id,
        algorithm=algorithm,
        found=bool(found),
        cost=float(cost),
        path_length=int(path_length),
        n_nodes=int(n_nodes),
        params=dict(params),
    )


def summarise_graph_algo_entries(
    entries: Sequence[GraphAlgoEntry],
    cfg: Optional[GraphAlgoConfig] = None,
) -> GraphAlgoSummary:
    """Сформировать сводку по результатам графовых алгоритмов.

    Args:
        entries: Список записей.
        cfg:     Конфигурация (None → GraphAlgoConfig()).

    Returns:
        :class:`GraphAlgoSummary`.
    """
    if cfg is None:
        cfg = GraphAlgoConfig()
    if not entries:
        return GraphAlgoSummary(
            n_runs=0,
            n_found=0,
            mean_cost=0.0,
            min_cost=None,
            best_run_id=None,
            worst_run_id=None,
        )
    found = [e for e in entries if e.found] if cfg.require_found else list(entries)
    if not found:
        return GraphAlgoSummary(
            n_runs=len(entries),
            n_found=0,
            mean_cost=0.0,
            min_cost=None,
            best_run_id=None,
            worst_run_id=None,
        )
    costs = [e.cost for e in found]
    mean_cost = sum(costs) / len(costs)
    best = min(found, key=lambda e: e.cost)
    worst = max(found, key=lambda e: e.cost)
    return GraphAlgoSummary(
        n_runs=len(entries),
        n_found=len(found),
        mean_cost=mean_cost,
        min_cost=best.cost,
        best_run_id=best.run_id,
        worst_run_id=worst.run_id,
    )


def filter_graph_algo_by_found(
    entries: Sequence[GraphAlgoEntry],
) -> List[GraphAlgoEntry]:
    """Отфильтровать записи — оставить только с найденными путями."""
    return [e for e in entries if e.found]


def filter_graph_algo_by_max_cost(
    entries: Sequence[GraphAlgoEntry],
    max_cost: float,
) -> List[GraphAlgoEntry]:
    """Отфильтровать записи по максимальной стоимости пути."""
    return [e for e in entries if e.cost <= max_cost]


def filter_graph_algo_by_algorithm(
    entries: Sequence[GraphAlgoEntry],
    algorithm: str,
) -> List[GraphAlgoEntry]:
    """Отфильтровать записи по названию алгоритма."""
    return [e for e in entries if e.algorithm == algorithm]


def filter_graph_algo_by_min_path_length(
    entries: Sequence[GraphAlgoEntry],
    min_length: int,
) -> List[GraphAlgoEntry]:
    """Отфильтровать записи по минимальной длине пути."""
    return [e for e in entries if e.path_length >= min_length]


def top_k_cheapest_paths(
    entries: Sequence[GraphAlgoEntry],
    k: int,
) -> List[GraphAlgoEntry]:
    """Вернуть k прогонов с наименьшей стоимостью пути."""
    found = filter_graph_algo_by_found(entries)
    return sorted(found, key=lambda e: e.cost)[:k]


def best_graph_algo_entry(
    entries: Sequence[GraphAlgoEntry],
) -> Optional[GraphAlgoEntry]:
    """Вернуть прогон с наименьшей стоимостью найденного пути."""
    found = filter_graph_algo_by_found(entries)
    return min(found, key=lambda e: e.cost) if found else None


def graph_algo_cost_stats(
    entries: Sequence[GraphAlgoEntry],
) -> Dict:
    """Вычислить статистику стоимостей: min, max, mean, std."""
    found = filter_graph_algo_by_found(entries)
    if not found:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "count": 0}
    vals = [e.cost for e in found]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {
        "min": min(vals),
        "max": max(vals),
        "mean": mean,
        "std": var ** 0.5,
        "count": len(vals),
    }


def compare_graph_algo_summaries(
    a: GraphAlgoSummary,
    b: GraphAlgoSummary,
) -> Dict:
    """Сравнить две сводки результатов графовых алгоритмов."""
    return {
        "delta_mean_cost": b.mean_cost - a.mean_cost,
        "delta_n_found": b.n_found - a.n_found,
        "delta_n_runs": b.n_runs - a.n_runs,
        "same_best": a.best_run_id == b.best_run_id,
    }


def batch_summarise_graph_algo_entries(
    groups: Sequence[Sequence[GraphAlgoEntry]],
    cfg: Optional[GraphAlgoConfig] = None,
) -> List[GraphAlgoSummary]:
    """Сформировать сводки для нескольких групп записей."""
    return [summarise_graph_algo_entries(g, cfg) for g in groups]


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Processing Result Analysis
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BatchResultConfig:
    """Конфигурация анализа результатов батч-обработки.

    Атрибуты:
        min_success_ratio: Минимально допустимое соотношение успехов.
        max_retries:       Максимально допустимое число повторов.
    """
    min_success_ratio: float = 0.0
    max_retries: int = 0


@dataclass
class BatchResultEntry:
    """Запись результата одного батча.

    Атрибуты:
        batch_id:      Идентификатор батча.
        total:         Число элементов в батче.
        n_success:     Число успешно обработанных.
        n_failed:      Число неудачных.
        n_retried:     Число повторных попыток.
        algorithm:     Метка алгоритма/метода обработки.
        params:        Дополнительные параметры.
    """
    batch_id: int
    total: int
    n_success: int
    n_failed: int
    n_retried: int
    algorithm: str
    params: Dict = field(default_factory=dict)

    @property
    def success_ratio(self) -> float:
        """Соотношение успешно обработанных элементов."""
        return self.n_success / self.total if self.total > 0 else 0.0


@dataclass
class BatchResultSummary:
    """Сводка результатов нескольких батчей.

    Атрибуты:
        n_batches:           Число батчей.
        total_items:         Суммарное число элементов.
        total_success:       Суммарное число успехов.
        total_failed:        Суммарное число неудач.
        mean_success_ratio:  Среднее соотношение успехов.
        best_batch_id:       ID батча с наилучшим соотношением или None.
        worst_batch_id:      ID батча с наихудшим соотношением или None.
    """
    n_batches: int
    total_items: int
    total_success: int
    total_failed: int
    mean_success_ratio: float
    best_batch_id: Optional[int]
    worst_batch_id: Optional[int]


def make_batch_result_entry(
    batch_id: int,
    total: int,
    n_success: int,
    n_failed: int,
    n_retried: int,
    algorithm: str,
    **params,
) -> BatchResultEntry:
    """Создать запись результата батча.

    Args:
        batch_id:   Идентификатор батча.
        total:      Число элементов.
        n_success:  Число успехов.
        n_failed:   Число неудач.
        n_retried:  Число повторов.
        algorithm:  Метка алгоритма.
        **params:   Дополнительные параметры.

    Returns:
        :class:`BatchResultEntry`.
    """
    return BatchResultEntry(
        batch_id=batch_id,
        total=int(total),
        n_success=int(n_success),
        n_failed=int(n_failed),
        n_retried=int(n_retried),
        algorithm=algorithm,
        params=dict(params),
    )


def summarise_batch_result_entries(
    entries: Sequence[BatchResultEntry],
    cfg: Optional[BatchResultConfig] = None,
) -> BatchResultSummary:
    """Сформировать сводку по записям результатов батчей.

    Args:
        entries: Список записей.
        cfg:     Конфигурация (None → BatchResultConfig()).

    Returns:
        :class:`BatchResultSummary`.
    """
    if cfg is None:
        cfg = BatchResultConfig()
    if not entries:
        return BatchResultSummary(
            n_batches=0,
            total_items=0,
            total_success=0,
            total_failed=0,
            mean_success_ratio=0.0,
            best_batch_id=None,
            worst_batch_id=None,
        )
    ratios = [e.success_ratio for e in entries]
    mean_ratio = sum(ratios) / len(ratios)
    best = max(entries, key=lambda e: e.success_ratio)
    worst = min(entries, key=lambda e: e.success_ratio)
    return BatchResultSummary(
        n_batches=len(entries),
        total_items=sum(e.total for e in entries),
        total_success=sum(e.n_success for e in entries),
        total_failed=sum(e.n_failed for e in entries),
        mean_success_ratio=mean_ratio,
        best_batch_id=best.batch_id,
        worst_batch_id=worst.batch_id,
    )


def filter_batch_results_by_min_ratio(
    entries: Sequence[BatchResultEntry],
    min_ratio: float,
) -> List[BatchResultEntry]:
    """Отфильтровать батчи по минимальному соотношению успехов."""
    return [e for e in entries if e.success_ratio >= min_ratio]


def filter_batch_results_by_algorithm(
    entries: Sequence[BatchResultEntry],
    algorithm: str,
) -> List[BatchResultEntry]:
    """Отфильтровать батчи по алгоритму обработки."""
    return [e for e in entries if e.algorithm == algorithm]


def filter_batch_results_by_max_retries(
    entries: Sequence[BatchResultEntry],
    max_retries: int,
) -> List[BatchResultEntry]:
    """Отфильтровать батчи по максимальному числу повторов."""
    return [e for e in entries if e.n_retried <= max_retries]


def top_k_batch_results(
    entries: Sequence[BatchResultEntry],
    k: int,
) -> List[BatchResultEntry]:
    """Вернуть k батчей с наилучшим соотношением успехов."""
    return sorted(entries, key=lambda e: e.success_ratio, reverse=True)[:k]


def best_batch_result_entry(
    entries: Sequence[BatchResultEntry],
) -> Optional[BatchResultEntry]:
    """Вернуть батч с наилучшим соотношением успехов."""
    return max(entries, key=lambda e: e.success_ratio) if entries else None


def batch_result_success_stats(
    entries: Sequence[BatchResultEntry],
) -> Dict:
    """Вычислить статистику соотношений успехов: min, max, mean, std."""
    if not entries:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "count": 0}
    vals = [e.success_ratio for e in entries]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {
        "min": min(vals),
        "max": max(vals),
        "mean": mean,
        "std": var ** 0.5,
        "count": len(vals),
    }


def compare_batch_result_summaries(
    a: BatchResultSummary,
    b: BatchResultSummary,
) -> Dict:
    """Сравнить две сводки результатов батч-обработки."""
    return {
        "delta_mean_success_ratio": b.mean_success_ratio - a.mean_success_ratio,
        "delta_total_items": b.total_items - a.total_items,
        "delta_total_success": b.total_success - a.total_success,
        "same_best": a.best_batch_id == b.best_batch_id,
    }


def batch_summarise_batch_result_entries(
    groups: Sequence[Sequence[BatchResultEntry]],
    cfg: Optional[BatchResultConfig] = None,
) -> List[BatchResultSummary]:
    """Сформировать сводки для нескольких групп записей батч-обработки."""
    return [summarise_batch_result_entries(g, cfg) for g in groups]
