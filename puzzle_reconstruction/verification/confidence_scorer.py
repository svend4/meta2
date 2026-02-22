"""
Вычисление итоговой уверенности (confidence) в качестве сборки.

Объединяет несколько независимых компонент в единую оценку:

    edge_compat     — средний score CompatEntry для задействованных пар
    layout          — отсутствие перекрытий/зазоров (из layout_verifier)
    text_coherence  — языковая оценка стыков (биграммы / trigrams)
    coverage        — доля фрагментов, участвующих в сборке
    uniqueness      — отсутствие повторных fragment_id в сборке

Конечная оценка: взвешенная сумма компонент → ∈ [0, 1].
Класс `AssemblyConfidence` содержит буквенную оценку (A–F).

Классы:
    ScoreComponent      — одна компонента с именем, весом, значением
    AssemblyConfidence  — итоговый результат (total, components, grade)

Функции:
    score_edge_compat      — компонента по совместимости краёв
    score_layout           — компонента по геометрическому расположению
    score_coverage         — компонента по покрытию фрагментов
    score_uniqueness       — компонента по уникальности размещений
    compute_confidence     — полный пайплайн → AssemblyConfidence
    grade_from_score       — float → 'A'/'B'/'C'/'D'/'F'
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..models import Assembly, CompatEntry, Fragment
from .layout_verifier import (
    ConstraintType,
    verify_layout,
)


# ─── ScoreComponent ───────────────────────────────────────────────────────────

@dataclass
class ScoreComponent:
    """
    Одна компонента итоговой оценки.

    Attributes:
        name:    Идентификатор ('edge_compat', 'layout', …).
        value:   Значение ∈ [0, 1].
        weight:  Вес при суммировании (≥ 0).
        description: Текстовое описание.
    """
    name:        str
    value:       float
    weight:      float  = 1.0
    description: str    = ""

    @property
    def weighted(self) -> float:
        return self.value * self.weight

    def __repr__(self) -> str:
        return (f"ScoreComponent(name={self.name!r}, "
                f"value={self.value:.3f}, weight={self.weight:.2f})")


# ─── AssemblyConfidence ───────────────────────────────────────────────────────

@dataclass
class AssemblyConfidence:
    """
    Итоговая оценка уверенности в качестве сборки.

    Attributes:
        total:      Взвешенная сумма ∈ [0, 1].
        components: Список ScoreComponent.
        grade:      Буквенная оценка ('A'–'F').
        n_fragments: Число фрагментов в сборке.
        assembly_method: Имя метода сборки (из Assembly.method).
    """
    total:           float
    components:      List[ScoreComponent]
    grade:           str
    n_fragments:     int   = 0
    assembly_method: str   = ""

    def get(self, name: str) -> Optional[ScoreComponent]:
        """Возвращает компоненту по имени (None если не найдена)."""
        for c in self.components:
            if c.name == name:
                return c
        return None

    def as_dict(self) -> Dict[str, float]:
        """Словарь {name: value}."""
        return {c.name: c.value for c in self.components}

    def summary(self) -> str:
        lines = [
            f"AssemblyConfidence(grade={self.grade}, total={self.total:.3f},"
            f" n={self.n_fragments}, method={self.assembly_method!r})"
        ]
        for c in self.components:
            lines.append(f"  [{c.name:<16}] {c.value:.3f} × w={c.weight:.2f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"AssemblyConfidence(grade={self.grade}, "
                f"total={self.total:.3f}, "
                f"n_fragments={self.n_fragments})")


# ─── Вспомогательная функция ──────────────────────────────────────────────────

def grade_from_score(score: float) -> str:
    """
    Переводит численную оценку в буквенную.

    Границы:
        A: score ≥ 0.85
        B: score ≥ 0.70
        C: score ≥ 0.55
        D: score ≥ 0.40
        F: score < 0.40
    """
    if score >= 0.85:
        return "A"
    if score >= 0.70:
        return "B"
    if score >= 0.55:
        return "C"
    if score >= 0.40:
        return "D"
    return "F"


# ─── Компоненты оценки ────────────────────────────────────────────────────────

def score_edge_compat(assembly: Assembly,
                       entries:  Sequence[CompatEntry],
                       weight:   float = 1.5) -> ScoreComponent:
    """
    Компонента по средней совместимости задействованных краёв.

    Для каждого Placement проверяет, есть ли CompatEntry c обоими
    fid из пар соседних Placement. Берём топ-K (K = n_placements) записей.

    Args:
        assembly: Сборка.
        entries:  Список CompatEntry.
        weight:   Вес компоненты.

    Returns:
        ScoreComponent с value ∈ [0, 1].
    """
    if not entries or not assembly.placements:
        return ScoreComponent(
            name="edge_compat", value=0.0, weight=weight,
            description="Нет entries или placements")

    # Используем total_score из Assembly как прямую оценку
    value = float(np.clip(assembly.total_score, 0.0, 1.0))

    # Дополнительно: среднее значение entries (если они относятся к этой сборке)
    fid_set = {pl.fragment_id for pl in assembly.placements}
    relevant = [e for e in entries
                if (e.edge_i.edge_id // 10) in fid_set
                and (e.edge_j.edge_id // 10) in fid_set]

    if relevant:
        avg_score = float(np.mean([e.score for e in relevant]))
        # Комбинируем: 60% total_score + 40% avg_entry_score
        value = 0.6 * value + 0.4 * avg_score

    return ScoreComponent(
        name="edge_compat",
        value=float(np.clip(value, 0.0, 1.0)),
        weight=weight,
        description=f"n_relevant_entries={len(relevant)}",
    )


def score_layout(assembly:  Assembly,
                  fragments: Sequence[Fragment],
                  weight:    float = 1.0) -> ScoreComponent:
    """
    Компонента по геометрической согласованности расположения.

    Использует verify_layout: violation_score → inverted → component value.

    Args:
        assembly:  Сборка.
        fragments: Список Fragment.
        weight:    Вес.

    Returns:
        ScoreComponent.
    """
    frags_list = list(fragments)
    if not frags_list or not assembly.placements:
        return ScoreComponent(
            name="layout", value=0.0, weight=weight,
            description="Нет фрагментов или placements")

    result = verify_layout(assembly, frags_list,
                            max_gap=20.0, col_tol=10.0, row_tol=10.0)
    value  = float(np.clip(1.0 - result.violation_score, 0.0, 1.0))

    return ScoreComponent(
        name="layout",
        value=value,
        weight=weight,
        description=(f"violations={len(result.constraints)}, "
                      f"violation_score={result.violation_score:.3f}"),
    )


def score_coverage(assembly:     Assembly,
                    all_fragment_ids: Sequence[int],
                    weight:       float = 0.8) -> ScoreComponent:
    """
    Компонента покрытия: доля известных фрагментов, включённых в сборку.

    Args:
        assembly:         Сборка.
        all_fragment_ids: Все известные fragment_id.
        weight:           Вес.

    Returns:
        ScoreComponent.
    """
    total = len(set(all_fragment_ids))
    if total == 0:
        return ScoreComponent(name="coverage", value=0.0, weight=weight,
                               description="Нет фрагментов")

    placed = len({pl.fragment_id for pl in assembly.placements})
    value  = min(1.0, placed / total)

    return ScoreComponent(
        name="coverage",
        value=float(value),
        weight=weight,
        description=f"placed={placed}/{total}",
    )


def score_uniqueness(assembly: Assembly,
                      weight:   float = 1.2) -> ScoreComponent:
    """
    Компонента уникальности: штраф за повторные fragment_id.

    Если все fid уникальны → 1.0.
    Каждый дубликат снижает оценку на 0.2.

    Args:
        assembly: Сборка.
        weight:   Вес.

    Returns:
        ScoreComponent.
    """
    fids    = [pl.fragment_id for pl in assembly.placements]
    n_total = len(fids)
    n_uniq  = len(set(fids))

    if n_total == 0:
        return ScoreComponent(name="uniqueness", value=0.0, weight=weight,
                               description="Нет placements")

    n_dup  = n_total - n_uniq
    value  = max(0.0, 1.0 - n_dup * 0.2)

    return ScoreComponent(
        name="uniqueness",
        value=float(value),
        weight=weight,
        description=f"duplicates={n_dup}",
    )


def score_assembly_score(assembly: Assembly,
                          weight:   float = 1.0) -> ScoreComponent:
    """
    Компонента на основе Assembly.total_score (нормализованный).

    Args:
        assembly: Сборка.
        weight:   Вес.

    Returns:
        ScoreComponent.
    """
    value = float(np.clip(assembly.total_score, 0.0, 1.0))
    return ScoreComponent(
        name="assembly_score",
        value=value,
        weight=weight,
        description=f"total_score={assembly.total_score:.4f}",
    )


# ─── Полный пайплайн ──────────────────────────────────────────────────────────

def compute_confidence(assembly:         Assembly,
                        fragments:        Sequence[Fragment],
                        entries:          Sequence[CompatEntry],
                        all_fragment_ids: Optional[Sequence[int]] = None,
                        weights:          Optional[Dict[str, float]] = None
                        ) -> AssemblyConfidence:
    """
    Вычисляет итоговую уверенность сборки.

    Args:
        assembly:         Сборка (Assembly).
        fragments:        Список Fragment.
        entries:          Список CompatEntry.
        all_fragment_ids: Все известные fragment_id. Если None — берём из fragments.
        weights:          Переопределение весов. Ключи: 'edge_compat', 'layout',
                          'coverage', 'uniqueness', 'assembly_score'.

    Returns:
        AssemblyConfidence.
    """
    if all_fragment_ids is None:
        all_fragment_ids = [f.fragment_id for f in fragments]

    w = {
        "edge_compat":    1.5,
        "layout":         1.0,
        "coverage":       0.8,
        "uniqueness":     1.2,
        "assembly_score": 1.0,
    }
    if weights:
        w.update(weights)

    components = [
        score_edge_compat(assembly, entries,    weight=w["edge_compat"]),
        score_layout(assembly, fragments,        weight=w["layout"]),
        score_coverage(assembly, all_fragment_ids, weight=w["coverage"]),
        score_uniqueness(assembly,               weight=w["uniqueness"]),
        score_assembly_score(assembly,           weight=w["assembly_score"]),
    ]

    total_weight = sum(c.weight for c in components)
    if total_weight == 0:
        total = 0.0
    else:
        total = sum(c.weighted for c in components) / total_weight

    total = float(np.clip(total, 0.0, 1.0))

    return AssemblyConfidence(
        total=total,
        components=components,
        grade=grade_from_score(total),
        n_fragments=len(assembly.placements),
        assembly_method=assembly.method,
    )
