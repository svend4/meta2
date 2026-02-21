"""
Итоговая оценка качества восстановленного документа.

Агрегирует несколько компонентов (геометрия, швы, покрытие, уникальность)
в единую нормированную оценку и предоставляет детальный разбор по составляющим.

Экспортирует:
    ScoreComponent      — одна составляющая итоговой оценки
    AssemblyScoreReport — полный отчёт с компонентами и итогом
    AssemblyScorerParams — параметры взвешивания компонентов
    score_geometry      — оценка геометрической корректности
    score_coverage      — оценка полноты покрытия
    score_seam_quality  — оценка качества швов (средняя)
    score_uniqueness    — оценка уникальности размещений
    compute_assembly_score — полная итоговая оценка
    compare_assemblies  — сравнение двух сборок по оценке
    rank_assemblies     — ранжирование нескольких сборок
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class ScoreComponent:
    """Одна составляющая итоговой оценки.

    Attributes:
        name:    Имя компонента.
        value:   Значение в [0, 1].
        weight:  Вес компонента при суммировании.
        details: Произвольные детали.
    """
    name: str
    value: float
    weight: float = 1.0
    details: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(
                f"ScoreComponent.value must be in [0, 1], got {self.value}"
            )
        if self.weight < 0.0:
            raise ValueError(
                f"ScoreComponent.weight must be >= 0, got {self.weight}"
            )

    def weighted_value(self) -> float:
        return self.value * self.weight


@dataclass
class AssemblyScoreReport:
    """Полный отчёт о качестве сборки.

    Attributes:
        total_score:  Итоговая нормированная оценка [0, 1].
        components:   Список компонентов оценки.
        n_fragments:  Число фрагментов в сборке.
        params:       Использованные параметры.
    """
    total_score: float
    components: List[ScoreComponent] = field(default_factory=list)
    n_fragments: int = 0
    params: Dict[str, object] = field(default_factory=dict)

    def component_by_name(self, name: str) -> Optional[ScoreComponent]:
        for c in self.components:
            if c.name == name:
                return c
        return None

    def to_dict(self) -> dict:
        return {
            "total_score": self.total_score,
            "n_fragments": self.n_fragments,
            "components": [
                {"name": c.name, "value": c.value, "weight": c.weight}
                for c in self.components
            ],
            "params": self.params,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AssemblyScoreReport(total={self.total_score:.4f}, "
            f"n_fragments={self.n_fragments}, "
            f"components={[c.name for c in self.components]})"
        )


@dataclass
class AssemblyScorerParams:
    """Параметры взвешивания компонентов оценки.

    Attributes:
        w_geometry:   Вес геометрической корректности (≥ 0).
        w_coverage:   Вес полноты покрытия (≥ 0).
        w_seam:       Вес качества швов (≥ 0).
        w_uniqueness: Вес уникальности (≥ 0).
        min_coverage: Минимальная доля покрытых позиций для оценки > 0.
    """
    w_geometry: float = 0.30
    w_coverage: float = 0.30
    w_seam: float = 0.25
    w_uniqueness: float = 0.15
    min_coverage: float = 0.0

    def __post_init__(self) -> None:
        for name, val in (
            ("w_geometry", self.w_geometry),
            ("w_coverage", self.w_coverage),
            ("w_seam", self.w_seam),
            ("w_uniqueness", self.w_uniqueness),
        ):
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")
        total = self.w_geometry + self.w_coverage + self.w_seam + self.w_uniqueness
        if total < 1e-8:
            raise ValueError("Sum of all weights must be > 0")

    def total_weight(self) -> float:
        return (self.w_geometry + self.w_coverage +
                self.w_seam + self.w_uniqueness)


# ─── Публичные функции ────────────────────────────────────────────────────────

def score_geometry(
    overlap_ratio: float,
    gap_ratio: float,
    alignment_score: float = 1.0,
) -> float:
    """Оценить геометрическую корректность сборки.

    Оценка снижается при наличии перекрытий (overlap) и зазоров (gap).

    Args:
        overlap_ratio:   Доля перекрывающихся пар (0 = нет перекрытий).
        gap_ratio:       Доля пар с чрезмерными зазорами (0 = нет зазоров).
        alignment_score: Дополнительный множитель выравнивания [0, 1].

    Returns:
        Оценка ∈ [0, 1].

    Raises:
        ValueError: Если любой аргумент вне допустимого диапазона.
    """
    for name, val in (
        ("overlap_ratio", overlap_ratio),
        ("gap_ratio", gap_ratio),
        ("alignment_score", alignment_score),
    ):
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {val}")
    penalty = (overlap_ratio + gap_ratio) / 2.0
    base = max(0.0, 1.0 - penalty)
    return float(base * alignment_score)


def score_coverage(
    n_placed: int,
    n_total: int,
    min_coverage: float = 0.0,
) -> float:
    """Оценить полноту размещения фрагментов.

    Args:
        n_placed:    Число размещённых фрагментов.
        n_total:     Общее число фрагментов.
        min_coverage: Минимальный порог (ниже → оценка 0).

    Returns:
        Оценка ∈ [0, 1].

    Raises:
        ValueError: Если ``n_total`` < 1 или ``n_placed`` < 0 или > ``n_total``.
    """
    if n_total < 1:
        raise ValueError(f"n_total must be >= 1, got {n_total}")
    if n_placed < 0:
        raise ValueError(f"n_placed must be >= 0, got {n_placed}")
    if n_placed > n_total:
        raise ValueError(f"n_placed ({n_placed}) > n_total ({n_total})")
    ratio = float(n_placed) / float(n_total)
    if ratio < min_coverage:
        return 0.0
    return ratio


def score_seam_quality(seam_scores: List[float]) -> float:
    """Вычислить среднюю оценку качества всех швов.

    Args:
        seam_scores: Список оценок отдельных швов ∈ [0, 1].

    Returns:
        Среднее значение ∈ [0, 1], или 1.0 для пустого списка.

    Raises:
        ValueError: Если любая оценка не в [0, 1].
    """
    if not seam_scores:
        return 1.0
    for i, s in enumerate(seam_scores):
        if not (0.0 <= s <= 1.0):
            raise ValueError(
                f"seam_scores[{i}] must be in [0, 1], got {s}"
            )
    return float(np.mean(seam_scores))


def score_uniqueness(
    n_fragments: int,
    n_duplicates: int,
) -> float:
    """Оценить уникальность размещений (штраф за дубликаты).

    Args:
        n_fragments: Общее число размещённых фрагментов.
        n_duplicates: Число дублирующих размещений.

    Returns:
        Оценка ∈ [0, 1].

    Raises:
        ValueError: Если ``n_fragments`` < 0 или ``n_duplicates`` < 0.
    """
    if n_fragments < 0:
        raise ValueError(f"n_fragments must be >= 0, got {n_fragments}")
    if n_duplicates < 0:
        raise ValueError(f"n_duplicates must be >= 0, got {n_duplicates}")
    if n_fragments == 0:
        return 1.0
    penalty = min(float(n_duplicates) / float(n_fragments), 1.0)
    return 1.0 - penalty


def compute_assembly_score(
    n_placed: int,
    n_total: int,
    overlap_ratio: float = 0.0,
    gap_ratio: float = 0.0,
    alignment_score: float = 1.0,
    seam_scores: Optional[List[float]] = None,
    n_duplicates: int = 0,
    params: Optional[AssemblyScorerParams] = None,
) -> AssemblyScoreReport:
    """Вычислить итоговую оценку качества сборки.

    Args:
        n_placed:        Число размещённых фрагментов.
        n_total:         Общее число фрагментов.
        overlap_ratio:   Доля перекрывающихся пар.
        gap_ratio:       Доля пар с зазорами.
        alignment_score: Оценка выравнивания [0, 1].
        seam_scores:     Список оценок швов (или ``None``).
        n_duplicates:    Число дублирующих размещений.
        params:          Параметры взвешивания.

    Returns:
        :class:`AssemblyScoreReport` с итоговой оценкой и компонентами.
    """
    if params is None:
        params = AssemblyScorerParams()
    if seam_scores is None:
        seam_scores = []

    geom = score_geometry(overlap_ratio, gap_ratio, alignment_score)
    cov = score_coverage(n_placed, n_total, params.min_coverage)
    seam = score_seam_quality(seam_scores)
    uniq = score_uniqueness(n_placed, n_duplicates)

    components = [
        ScoreComponent("geometry",   geom, params.w_geometry),
        ScoreComponent("coverage",   cov,  params.w_coverage),
        ScoreComponent("seam",       seam, params.w_seam),
        ScoreComponent("uniqueness", uniq, params.w_uniqueness),
    ]

    total_w = params.total_weight()
    total_score = sum(c.weighted_value() for c in components) / total_w

    return AssemblyScoreReport(
        total_score=float(np.clip(total_score, 0.0, 1.0)),
        components=components,
        n_fragments=n_placed,
        params={
            "w_geometry": params.w_geometry,
            "w_coverage": params.w_coverage,
            "w_seam": params.w_seam,
            "w_uniqueness": params.w_uniqueness,
        },
    )


def compare_assemblies(
    report_a: AssemblyScoreReport,
    report_b: AssemblyScoreReport,
) -> int:
    """Сравнить два отчёта по итоговой оценке.

    Args:
        report_a: Первый отчёт.
        report_b: Второй отчёт.

    Returns:
        1 если A лучше, -1 если B лучше, 0 если равны.
    """
    if report_a.total_score > report_b.total_score + 1e-9:
        return 1
    if report_b.total_score > report_a.total_score + 1e-9:
        return -1
    return 0


def rank_assemblies(
    reports: List[AssemblyScoreReport],
) -> List[Tuple[int, AssemblyScoreReport]]:
    """Ранжировать список отчётов по убыванию итоговой оценки.

    Args:
        reports: Список отчётов.

    Returns:
        Список кортежей (rank_1based, report), отсортированных по убыванию.
        Для пустого списка — пустой список.
    """
    if not reports:
        return []
    sorted_reports = sorted(reports, key=lambda r: r.total_score, reverse=True)
    return [(i + 1, r) for i, r in enumerate(sorted_reports)]
