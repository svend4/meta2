"""Пространственная валидация сборки фрагментов.

Модуль проверяет геометрическую корректность расположения фрагментов:
отсутствие перекрытий, нахождение внутри холста, равномерность зазоров,
полноту покрытия и отсутствие дублирующихся ID.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


_SEVERITY = {"error", "warning", "info"}


# ─── SpatialIssue ─────────────────────────────────────────────────────────────

@dataclass
class SpatialIssue:
    """Одно пространственное нарушение.

    Атрибуты:
        code:      Код нарушения (непустой).
        severity:  'error' | 'warning' | 'info'.
        fragment_ids: Идентификаторы задействованных фрагментов.
        detail:    Текстовое описание нарушения.
    """

    code: str
    severity: str
    fragment_ids: List[int] = field(default_factory=list)
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("code не может быть пустым")
        if self.severity not in _SEVERITY:
            raise ValueError(
                f"severity должен быть одним из {_SEVERITY}, "
                f"получено '{self.severity}'"
            )


# ─── SpatialReport ────────────────────────────────────────────────────────────

@dataclass
class SpatialReport:
    """Итоговый отчёт пространственной валидации.

    Атрибуты:
        issues:         Список найденных нарушений.
        n_fragments:    Число проверенных фрагментов.
        canvas_w/h:     Размеры холста.
        is_valid:       True если нет ошибок (severity='error').
    """

    issues: List[SpatialIssue] = field(default_factory=list)
    n_fragments: int = 0
    canvas_w: float = 0.0
    canvas_h: float = 0.0

    def __post_init__(self) -> None:
        if self.n_fragments < 0:
            raise ValueError(
                f"n_fragments должен быть >= 0, получено {self.n_fragments}"
            )
        if self.canvas_w < 0 or self.canvas_h < 0:
            raise ValueError("canvas_w и canvas_h должны быть >= 0")

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def is_valid(self) -> bool:
        return self.n_errors == 0


# ─── PlacedFragment (lightweight) ─────────────────────────────────────────────

@dataclass
class PlacedFragment:
    """Размещённый фрагмент на холсте.

    Атрибуты:
        fragment_id: Идентификатор (>= 0).
        x, y:        Координаты верхнего левого угла (>= 0).
        width:       Ширина (>= 1).
        height:      Высота (>= 1).
    """

    fragment_id: int
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        if self.x < 0 or self.y < 0:
            raise ValueError("x и y должны быть >= 0")
        if self.width < 1:
            raise ValueError(f"width должен быть >= 1, получено {self.width}")
        if self.height < 1:
            raise ValueError(f"height должен быть >= 1, получено {self.height}")

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return self.width * self.height


# ─── check_unique_ids ─────────────────────────────────────────────────────────

def check_unique_ids(fragments: List[PlacedFragment]) -> List[SpatialIssue]:
    """Проверить уникальность идентификаторов фрагментов.

    Возвращает:
        Список SpatialIssue (severity='error') для дублирующихся ID.
    """
    seen: Dict[int, int] = {}
    issues = []
    for f in fragments:
        if f.fragment_id in seen:
            issues.append(SpatialIssue(
                code="DUPLICATE_ID",
                severity="error",
                fragment_ids=[f.fragment_id],
                detail=f"Дублирующийся fragment_id={f.fragment_id}",
            ))
        else:
            seen[f.fragment_id] = 1
    return issues


# ─── check_within_canvas ──────────────────────────────────────────────────────

def check_within_canvas(
    fragments: List[PlacedFragment],
    canvas_w: float,
    canvas_h: float,
) -> List[SpatialIssue]:
    """Проверить, что все фрагменты находятся внутри холста.

    Аргументы:
        fragments: Список фрагментов.
        canvas_w:  Ширина холста (> 0).
        canvas_h:  Высота холста (> 0).

    Возвращает:
        Список SpatialIssue (severity='error') для выходящих за границу.

    Исключения:
        ValueError: Если canvas_w или canvas_h <= 0.
    """
    if canvas_w <= 0 or canvas_h <= 0:
        raise ValueError(
            f"canvas_w и canvas_h должны быть > 0, "
            f"получено {canvas_w}×{canvas_h}"
        )
    issues = []
    for f in fragments:
        if f.x2 > canvas_w or f.y2 > canvas_h:
            issues.append(SpatialIssue(
                code="OUT_OF_BOUNDS",
                severity="error",
                fragment_ids=[f.fragment_id],
                detail=(
                    f"Фрагмент {f.fragment_id} выходит за холст "
                    f"({f.x2:.1f}>{canvas_w} или {f.y2:.1f}>{canvas_h})"
                ),
            ))
    return issues


# ─── check_no_overlaps ────────────────────────────────────────────────────────

def check_no_overlaps(
    fragments: List[PlacedFragment],
    tolerance: float = 0.0,
) -> List[SpatialIssue]:
    """Проверить отсутствие перекрытий между фрагментами.

    Аргументы:
        fragments: Список фрагментов.
        tolerance: Допустимое перекрытие в пикселях (>= 0).

    Возвращает:
        Список SpatialIssue (severity='error') для перекрывающихся пар.

    Исключения:
        ValueError: Если tolerance < 0.
    """
    if tolerance < 0:
        raise ValueError(
            f"tolerance должен быть >= 0, получено {tolerance}"
        )
    issues = []
    n = len(fragments)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = fragments[i], fragments[j]
            ox = min(a.x2, b.x2) - max(a.x, b.x)
            oy = min(a.y2, b.y2) - max(a.y, b.y)
            if ox > tolerance and oy > tolerance:
                issues.append(SpatialIssue(
                    code="OVERLAP",
                    severity="error",
                    fragment_ids=[a.fragment_id, b.fragment_id],
                    detail=(
                        f"Фрагменты {a.fragment_id} и {b.fragment_id} "
                        f"перекрываются на {ox:.1f}×{oy:.1f} px"
                    ),
                ))
    return issues


# ─── check_coverage ───────────────────────────────────────────────────────────

def check_coverage(
    fragments: List[PlacedFragment],
    canvas_w: float,
    canvas_h: float,
    min_coverage: float = 0.5,
) -> List[SpatialIssue]:
    """Проверить степень покрытия холста фрагментами.

    Аргументы:
        fragments:    Список фрагментов.
        canvas_w/h:   Размеры холста (> 0).
        min_coverage: Минимальная доля покрытия [0, 1].

    Возвращает:
        Список SpatialIssue (severity='warning') при недостаточном покрытии.

    Исключения:
        ValueError: Если min_coverage ∉ [0, 1] или canvas размеры <= 0.
    """
    if canvas_w <= 0 or canvas_h <= 0:
        raise ValueError("canvas_w и canvas_h должны быть > 0")
    if not (0.0 <= min_coverage <= 1.0):
        raise ValueError(
            f"min_coverage должен быть в [0, 1], получено {min_coverage}"
        )
    canvas_area = canvas_w * canvas_h
    if canvas_area < 1e-10:
        return []

    total = sum(f.area for f in fragments)
    coverage = min(total / canvas_area, 1.0)

    issues = []
    if coverage < min_coverage:
        issues.append(SpatialIssue(
            code="LOW_COVERAGE",
            severity="warning",
            detail=(
                f"Покрытие холста {coverage:.1%} ниже порога {min_coverage:.1%}"
            ),
        ))
    return issues


# ─── check_gap_uniformity ─────────────────────────────────────────────────────

def check_gap_uniformity(
    fragments: List[PlacedFragment],
    max_gap_std: float = 10.0,
) -> List[SpatialIssue]:
    """Проверить равномерность зазоров между соседними фрагментами.

    Вычисляет x- и y-зазоры для всех пар и проверяет СКО.

    Аргументы:
        fragments:   Список фрагментов.
        max_gap_std: Максимально допустимое СКО зазоров (>= 0).

    Возвращает:
        Список SpatialIssue (severity='warning') при неравномерных зазорах.

    Исключения:
        ValueError: Если max_gap_std < 0.
    """
    if max_gap_std < 0:
        raise ValueError(
            f"max_gap_std должен быть >= 0, получено {max_gap_std}"
        )
    if len(fragments) < 2:
        return []

    gaps = []
    n = len(fragments)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = fragments[i], fragments[j]
            gx = max(a.x, b.x) - min(a.x2, b.x2)
            gy = max(a.y, b.y) - min(a.y2, b.y2)
            gaps.append(max(gx, gy))

    arr = np.array(gaps, dtype=np.float64)
    std = float(arr.std())

    issues = []
    if std > max_gap_std:
        issues.append(SpatialIssue(
            code="UNEVEN_GAPS",
            severity="warning",
            detail=(
                f"СКО зазоров {std:.2f} превышает порог {max_gap_std:.2f}"
            ),
        ))
    return issues


# ─── validate_spatial ─────────────────────────────────────────────────────────

def validate_spatial(
    fragments: List[PlacedFragment],
    canvas_w: float,
    canvas_h: float,
    overlap_tolerance: float = 0.0,
    min_coverage: float = 0.5,
    max_gap_std: float = 10.0,
) -> SpatialReport:
    """Выполнить полную пространственную валидацию сборки.

    Аргументы:
        fragments:          Список фрагментов.
        canvas_w/h:         Размеры холста.
        overlap_tolerance:  Допустимое перекрытие (>= 0).
        min_coverage:       Минимальная доля покрытия [0, 1].
        max_gap_std:        Максимальное СКО зазоров (>= 0).

    Возвращает:
        SpatialReport.
    """
    issues: List[SpatialIssue] = []
    issues.extend(check_unique_ids(fragments))
    issues.extend(check_within_canvas(fragments, canvas_w, canvas_h))
    issues.extend(check_no_overlaps(fragments, tolerance=overlap_tolerance))
    issues.extend(check_coverage(fragments, canvas_w, canvas_h, min_coverage))
    issues.extend(check_gap_uniformity(fragments, max_gap_std=max_gap_std))

    return SpatialReport(
        issues=issues,
        n_fragments=len(fragments),
        canvas_w=canvas_w,
        canvas_h=canvas_h,
    )


# ─── batch_validate ───────────────────────────────────────────────────────────

def batch_validate(
    assemblies: List[List[PlacedFragment]],
    canvas_w: float,
    canvas_h: float,
    overlap_tolerance: float = 0.0,
    min_coverage: float = 0.5,
) -> List[SpatialReport]:
    """Валидировать несколько сборок.

    Аргументы:
        assemblies:        Список сборок.
        canvas_w/h:        Размеры холста.
        overlap_tolerance: Допустимое перекрытие.
        min_coverage:      Минимальная доля покрытия.

    Возвращает:
        Список SpatialReport.
    """
    return [
        validate_spatial(a, canvas_w, canvas_h, overlap_tolerance, min_coverage)
        for a in assemblies
    ]
