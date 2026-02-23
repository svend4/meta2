"""Утилиты анализа и ранжирования результатов сборки пазла.

Предоставляет структуры данных и функции для хранения, фильтрации
и сравнения результатов алгоритмов сборки (генетический, муравьиный, SA…).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─── Config ──────────────────────────────────────────────────────────────────


@dataclass
class AssemblyScoreConfig:
    """Конфигурация для фильтрации и анализа результатов сборки."""

    min_score: float = 0.0
    max_entries: int = 1000
    method: str = "any"

    def __post_init__(self) -> None:
        if self.min_score < 0.0:
            raise ValueError("min_score must be >= 0.0")
        if self.max_entries <= 0:
            raise ValueError("max_entries must be > 0")


# ─── Entry ───────────────────────────────────────────────────────────────────


@dataclass
class AssemblyScoreEntry:
    """Запись об одном результате сборки пазла."""

    run_id: int
    method: str
    n_fragments: int
    total_score: float
    n_iterations: int = 0
    best_iter: int = 0
    rank: int = 0
    meta: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.run_id < 0:
            raise ValueError("run_id must be >= 0")
        if self.n_fragments < 0:
            raise ValueError("n_fragments must be >= 0")
        if self.total_score < 0.0:
            raise ValueError("total_score must be >= 0.0")

    @property
    def is_good(self) -> bool:
        """True если total_score > 0.5."""
        return self.total_score > 0.5

    @property
    def score_per_fragment(self) -> float:
        """Нормализованный балл на фрагмент (0 если фрагментов нет)."""
        if self.n_fragments == 0:
            return 0.0
        return self.total_score / self.n_fragments


# ─── Summary ─────────────────────────────────────────────────────────────────


@dataclass
class AssemblySummary:
    """Агрегированные данные по нескольким результатам сборки."""

    entries: List[AssemblyScoreEntry]
    n_total: int
    n_good: int
    n_poor: int
    mean_score: float
    max_score: float
    min_score: float

    def __repr__(self) -> str:
        return (
            f"AssemblySummary(n_total={self.n_total}, "
            f"n_good={self.n_good}, "
            f"mean_score={self.mean_score:.4f}, "
            f"max_score={self.max_score:.4f})"
        )


# ─── Factory ─────────────────────────────────────────────────────────────────


def make_assembly_entry(
    run_id: int,
    method: str,
    n_fragments: int,
    total_score: float,
    n_iterations: int = 0,
    best_iter: int = 0,
    rank: int = 0,
    meta: Optional[Dict] = None,
) -> AssemblyScoreEntry:
    """Создать запись AssemblyScoreEntry."""
    return AssemblyScoreEntry(
        run_id=run_id,
        method=method,
        n_fragments=n_fragments,
        total_score=float(total_score),
        n_iterations=n_iterations,
        best_iter=best_iter,
        rank=rank,
        meta=meta or {},
    )


def entries_from_assemblies(
    assemblies: List,
    method: str = "unknown",
) -> List[AssemblyScoreEntry]:
    """Построить список записей из объектов Assembly.

    Ожидается, что у каждого объекта есть атрибуты
    ``total_score`` и ``placements``.
    """
    entries: List[AssemblyScoreEntry] = []
    for idx, asm in enumerate(assemblies):
        score = float(getattr(asm, "total_score", 0.0))
        n_frags = len(getattr(asm, "placements", {}))
        entries.append(
            make_assembly_entry(
                run_id=idx,
                method=method,
                n_fragments=n_frags,
                total_score=score,
            )
        )
    # Присвоить ранги по убыванию оценки
    for rank, entry in enumerate(
        sorted(entries, key=lambda e: e.total_score, reverse=True), start=1
    ):
        entry.rank = rank
    return entries


def summarise_assemblies(
    entries: List[AssemblyScoreEntry],
) -> AssemblySummary:
    """Вычислить сводку по списку записей AssemblyScoreEntry."""
    n_total = len(entries)
    if n_total == 0:
        return AssemblySummary(
            entries=[],
            n_total=0,
            n_good=0,
            n_poor=0,
            mean_score=0.0,
            max_score=0.0,
            min_score=0.0,
        )
    scores = [e.total_score for e in entries]
    n_good = sum(1 for e in entries if e.is_good)
    return AssemblySummary(
        entries=list(entries),
        n_total=n_total,
        n_good=n_good,
        n_poor=n_total - n_good,
        mean_score=sum(scores) / n_total,
        max_score=max(scores),
        min_score=min(scores),
    )


# ─── Filters ─────────────────────────────────────────────────────────────────


def filter_good_assemblies(
    entries: List[AssemblyScoreEntry],
) -> List[AssemblyScoreEntry]:
    """Вернуть только записи с is_good == True."""
    return [e for e in entries if e.is_good]


def filter_poor_assemblies(
    entries: List[AssemblyScoreEntry],
) -> List[AssemblyScoreEntry]:
    """Вернуть только записи с is_good == False."""
    return [e for e in entries if not e.is_good]


def filter_by_method(
    entries: List[AssemblyScoreEntry],
    method: str,
) -> List[AssemblyScoreEntry]:
    """Фильтр по полю ``method``."""
    return [e for e in entries if e.method == method]


def filter_by_score_range(
    entries: List[AssemblyScoreEntry],
    lo: float = 0.0,
    hi: float = float("inf"),
) -> List[AssemblyScoreEntry]:
    """Фильтр записей по диапазону total_score."""
    return [e for e in entries if lo <= e.total_score <= hi]


def filter_by_min_fragments(
    entries: List[AssemblyScoreEntry],
    min_fragments: int,
) -> List[AssemblyScoreEntry]:
    """Фильтр записей по минимальному числу фрагментов."""
    return [e for e in entries if e.n_fragments >= min_fragments]


# ─── Ranking ─────────────────────────────────────────────────────────────────


def top_k_assembly_entries(
    entries: List[AssemblyScoreEntry],
    k: int,
) -> List[AssemblyScoreEntry]:
    """Вернуть топ-k записей по убыванию total_score."""
    if k <= 0:
        return []
    return sorted(entries, key=lambda e: e.total_score, reverse=True)[:k]


def best_assembly_entry(
    entries: List[AssemblyScoreEntry],
) -> Optional[AssemblyScoreEntry]:
    """Вернуть запись с наибольшим total_score или None."""
    if not entries:
        return None
    return max(entries, key=lambda e: e.total_score)


# ─── Statistics ──────────────────────────────────────────────────────────────


def assembly_score_stats(
    entries: List[AssemblyScoreEntry],
) -> Dict:
    """Вернуть словарь базовых статистик по total_score."""
    if not entries:
        return {
            "n": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
        }
    scores = [e.total_score for e in entries]
    n = len(scores)
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n
    return {
        "n": n,
        "mean": mean,
        "min": min(scores),
        "max": max(scores),
        "std": variance ** 0.5,
    }


# ─── Comparison ──────────────────────────────────────────────────────────────


def compare_assembly_summaries(
    a: AssemblySummary,
    b: AssemblySummary,
) -> Dict:
    """Сравнить две сводки; вернуть словарь с delta-значениями."""
    return {
        "delta_mean_score": a.mean_score - b.mean_score,
        "delta_max_score": a.max_score - b.max_score,
        "delta_n_good": a.n_good - b.n_good,
        "a_better": a.mean_score >= b.mean_score,
    }


# ─── Batch ───────────────────────────────────────────────────────────────────


def batch_summarise_assemblies(
    groups: List[List[AssemblyScoreEntry]],
) -> List[AssemblySummary]:
    """Применить summarise_assemblies к каждой группе записей."""
    return [summarise_assemblies(g) for g in groups]
