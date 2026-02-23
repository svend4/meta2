"""Утилиты планирования путей и последовательностей сборки пазла.

Предоставляет Config/Entry/Summary-структуры и вспомогательные функции
для анализа результатов поиска кратчайших путей в графе совместимости
фрагментов и планов пошагового размещения.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Dict, List, Optional, Sequence, Tuple


# ─── PathPlan utilities ───────────────────────────────────────────────────────

@dataclass
class PathPlanConfig:
    """Конфигурация для анализа результатов поиска путей."""
    min_cost: float = 0.0
    max_cost: float = float("inf")
    require_found: bool = True


@dataclass
class PathPlanEntry:
    """Запись одного результата поиска пути между двумя вершинами."""
    start: int
    end: int
    path: List[int]
    cost: float
    found: bool
    hops: int = field(init=False)

    def __post_init__(self) -> None:
        self.hops = len(self.path) - 1 if self.path else 0


@dataclass
class PathPlanSummary:
    """Сводная статистика набора PathPlanEntry."""
    n_entries: int
    n_found: int
    n_not_found: int
    mean_cost: float
    mean_hops: float
    min_cost: float
    max_cost: float
    found_rate: float


def make_path_entry(
    start: int,
    end: int,
    path: List[int],
    cost: float,
    found: bool,
) -> PathPlanEntry:
    """Создать PathPlanEntry."""
    return PathPlanEntry(start=start, end=end, path=path, cost=cost, found=found)


def entries_from_path_results(
    results: Sequence[Tuple[int, int, List[int], float, bool]],
) -> List[PathPlanEntry]:
    """Создать список PathPlanEntry из кортежей (start, end, path, cost, found)."""
    return [make_path_entry(s, e, p, c, f) for s, e, p, c, f in results]


def summarise_path_entries(entries: Sequence[PathPlanEntry]) -> PathPlanSummary:
    """Вычислить сводную статистику по списку PathPlanEntry."""
    n = len(entries)
    if n == 0:
        return PathPlanSummary(
            n_entries=0, n_found=0, n_not_found=0,
            mean_cost=0.0, mean_hops=0.0,
            min_cost=0.0, max_cost=0.0, found_rate=0.0,
        )
    found = [e for e in entries if e.found]
    n_found = len(found)
    costs = [e.cost for e in found] if found else [0.0]
    hops = [e.hops for e in found] if found else [0.0]
    return PathPlanSummary(
        n_entries=n,
        n_found=n_found,
        n_not_found=n - n_found,
        mean_cost=mean(costs),
        mean_hops=mean(hops),
        min_cost=min(costs),
        max_cost=max(costs),
        found_rate=n_found / n,
    )


def filter_found_paths(entries: Sequence[PathPlanEntry]) -> List[PathPlanEntry]:
    """Оставить только успешно найденные пути."""
    return [e for e in entries if e.found]


def filter_not_found_paths(entries: Sequence[PathPlanEntry]) -> List[PathPlanEntry]:
    """Оставить только ненайденные пути."""
    return [e for e in entries if not e.found]


def filter_path_by_cost_range(
    entries: Sequence[PathPlanEntry],
    lo: float = 0.0,
    hi: float = float("inf"),
) -> List[PathPlanEntry]:
    """Фильтровать по стоимости пути."""
    return [e for e in entries if e.found and lo <= e.cost <= hi]


def filter_path_by_max_hops(
    entries: Sequence[PathPlanEntry],
    max_hops: int,
) -> List[PathPlanEntry]:
    """Оставить пути не длиннее max_hops шагов."""
    return [e for e in entries if e.found and e.hops <= max_hops]


def top_k_shortest_paths(
    entries: Sequence[PathPlanEntry],
    k: int,
) -> List[PathPlanEntry]:
    """Вернуть k найденных путей с наименьшей стоимостью."""
    found = [e for e in entries if e.found]
    return sorted(found, key=lambda e: e.cost)[:k]


def cheapest_path_entry(entries: Sequence[PathPlanEntry]) -> Optional[PathPlanEntry]:
    """Найденный путь с минимальной стоимостью или None."""
    found = [e for e in entries if e.found]
    return min(found, key=lambda e: e.cost) if found else None


def path_cost_stats(
    entries: Sequence[PathPlanEntry],
) -> Dict[str, float]:
    """Вернуть словарь базовых статистик по стоимостям найденных путей."""
    costs = [e.cost for e in entries if e.found]
    if not costs:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(costs)),
        "mean": mean(costs),
        "std": stdev(costs) if len(costs) > 1 else 0.0,
        "min": min(costs),
        "max": max(costs),
    }


def compare_path_summaries(
    a: PathPlanSummary,
    b: PathPlanSummary,
) -> Dict[str, float]:
    """Сравнить две PathPlanSummary; вернуть разности ключевых метрик."""
    return {
        "found_rate_delta": a.found_rate - b.found_rate,
        "mean_cost_delta": a.mean_cost - b.mean_cost,
        "mean_hops_delta": a.mean_hops - b.mean_hops,
    }


def batch_summarise_path_entries(
    groups: Sequence[Sequence[PathPlanEntry]],
) -> List[PathPlanSummary]:
    """Вычислить сводку для каждой группы PathPlanEntry."""
    return [summarise_path_entries(g) for g in groups]


# ─── AssemblyPlan utilities ───────────────────────────────────────────────────

@dataclass
class AssemblyPlanConfig:
    """Конфигурация для анализа планов сборки."""
    min_coverage: float = 0.0
    min_mean_score: float = 0.0
    strategy: str = "greedy"


@dataclass
class AssemblyPlanEntry:
    """Запись одного плана пошагового размещения фрагментов."""
    plan_id: int
    n_fragments: int
    n_placed: int
    coverage: float
    mean_score: float
    strategy: str
    placement_order: List[int]


@dataclass
class AssemblyPlanSummary:
    """Сводная статистика набора AssemblyPlanEntry."""
    n_plans: int
    mean_coverage: float
    mean_score: float
    mean_placed: float
    min_coverage: float
    max_coverage: float
    strategy: str


def make_assembly_plan_entry(
    plan_id: int,
    n_fragments: int,
    n_placed: int,
    coverage: float,
    mean_score: float,
    strategy: str,
    placement_order: List[int],
) -> AssemblyPlanEntry:
    """Создать AssemblyPlanEntry."""
    return AssemblyPlanEntry(
        plan_id=plan_id,
        n_fragments=n_fragments,
        n_placed=n_placed,
        coverage=coverage,
        mean_score=mean_score,
        strategy=strategy,
        placement_order=placement_order,
    )


def summarise_assembly_plans(
    entries: Sequence[AssemblyPlanEntry],
) -> AssemblyPlanSummary:
    """Вычислить сводную статистику по списку AssemblyPlanEntry."""
    n = len(entries)
    if n == 0:
        return AssemblyPlanSummary(
            n_plans=0, mean_coverage=0.0, mean_score=0.0,
            mean_placed=0.0, min_coverage=0.0, max_coverage=0.0,
            strategy="greedy",
        )
    coverages = [e.coverage for e in entries]
    strategies = {e.strategy for e in entries}
    strategy = next(iter(strategies)) if len(strategies) == 1 else "mixed"
    return AssemblyPlanSummary(
        n_plans=n,
        mean_coverage=mean(coverages),
        mean_score=mean(e.mean_score for e in entries),
        mean_placed=mean(e.n_placed for e in entries),
        min_coverage=min(coverages),
        max_coverage=max(coverages),
        strategy=strategy,
    )


def filter_full_coverage_plans(
    entries: Sequence[AssemblyPlanEntry],
) -> List[AssemblyPlanEntry]:
    """Оставить планы с покрытием 1.0 (все фрагменты размещены)."""
    return [e for e in entries if e.coverage >= 1.0 - 1e-9]


def filter_assembly_plans_by_coverage(
    entries: Sequence[AssemblyPlanEntry],
    min_coverage: float,
) -> List[AssemblyPlanEntry]:
    """Фильтровать планы по минимальному покрытию."""
    return [e for e in entries if e.coverage >= min_coverage]


def filter_assembly_plans_by_score(
    entries: Sequence[AssemblyPlanEntry],
    min_score: float,
) -> List[AssemblyPlanEntry]:
    """Фильтровать планы по минимальной средней оценке."""
    return [e for e in entries if e.mean_score >= min_score]


def filter_assembly_plans_by_strategy(
    entries: Sequence[AssemblyPlanEntry],
    strategy: str,
) -> List[AssemblyPlanEntry]:
    """Фильтровать планы по стратегии."""
    return [e for e in entries if e.strategy == strategy]


def top_k_assembly_plan_entries(
    entries: Sequence[AssemblyPlanEntry],
    k: int,
) -> List[AssemblyPlanEntry]:
    """Вернуть k планов с наибольшим покрытием × средней оценкой."""
    return sorted(entries, key=lambda e: e.coverage * e.mean_score, reverse=True)[:k]


def best_assembly_plan_entry(
    entries: Sequence[AssemblyPlanEntry],
) -> Optional[AssemblyPlanEntry]:
    """Вернуть план с наилучшим покрытием × средней оценкой или None."""
    if not entries:
        return None
    return max(entries, key=lambda e: e.coverage * e.mean_score)


def assembly_plan_stats(
    entries: Sequence[AssemblyPlanEntry],
) -> Dict[str, float]:
    """Вернуть словарь базовых статистик по планам."""
    if not entries:
        return {"count": 0, "mean_coverage": 0.0, "mean_score": 0.0}
    coverages = [e.coverage for e in entries]
    scores = [e.mean_score for e in entries]
    return {
        "count": float(len(entries)),
        "mean_coverage": mean(coverages),
        "std_coverage": stdev(coverages) if len(coverages) > 1 else 0.0,
        "mean_score": mean(scores),
        "std_score": stdev(scores) if len(scores) > 1 else 0.0,
        "min_coverage": min(coverages),
        "max_coverage": max(coverages),
    }


def compare_assembly_plan_summaries(
    a: AssemblyPlanSummary,
    b: AssemblyPlanSummary,
) -> Dict[str, float]:
    """Сравнить две AssemblyPlanSummary; вернуть разности ключевых метрик."""
    return {
        "mean_coverage_delta": a.mean_coverage - b.mean_coverage,
        "mean_score_delta": a.mean_score - b.mean_score,
        "mean_placed_delta": a.mean_placed - b.mean_placed,
    }


def batch_summarise_assembly_plans(
    groups: Sequence[Sequence[AssemblyPlanEntry]],
) -> List[AssemblyPlanSummary]:
    """Вычислить сводку для каждой группы AssemblyPlanEntry."""
    return [summarise_assembly_plans(g) for g in groups]
