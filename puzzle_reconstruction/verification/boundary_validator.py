"""
Валидация геометрических границ между соседними фрагментами.

Экспортирует:
    BoundaryViolation   — нарушение граничного условия между двумя фрагментами
    BoundaryReport      — сводный отчёт по всем парам соседей
    validate_edge_gap   — проверка зазора / перекрытия на стыке
    validate_alignment  — проверка выравнивания (угол наклона) стыка
    validate_pair       — комплексная проверка одной пары фрагментов
    validate_all_pairs  — проверка всех соседних пар в сборке
    boundary_quality_score — общий балл качества границ
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class BoundaryViolation:
    """Нарушение граничного условия между двумя соседними фрагментами.

    Attributes:
        idx1:          Индекс первого фрагмента.
        idx2:          Индекс второго фрагмента.
        violation_type: Тип нарушения: ``"gap"``, ``"overlap"``, ``"tilt"``.
        severity:      Степень нарушения [0, +∞); 0 = нарушения нет.
        message:       Человекочитаемое описание нарушения.
        params:        Дополнительные параметры проверки.
    """
    idx1: int
    idx2: int
    violation_type: str
    severity: float
    message: str = ""
    params: Dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BoundaryViolation({self.idx1}↔{self.idx2}, "
            f"type={self.violation_type!r}, severity={self.severity:.4f})"
        )


@dataclass
class BoundaryReport:
    """Сводный отчёт по проверке границ для всей сборки.

    Attributes:
        violations:       Список всех обнаруженных нарушений.
        n_pairs_checked:  Количество проверенных пар.
        is_valid:         ``True``, если нарушений с severity > threshold нет.
        overall_score:    Балл качества [0, 1] (1 — идеально).
        params:           Параметры проверки.
    """
    violations: List[BoundaryViolation] = field(default_factory=list)
    n_pairs_checked: int = 0
    is_valid: bool = True
    overall_score: float = 1.0
    params: Dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BoundaryReport(pairs={self.n_pairs_checked}, "
            f"violations={len(self.violations)}, score={self.overall_score:.4f})"
        )


# ─── Приватные утилиты ────────────────────────────────────────────────────────

def _signed_gap(pos1: float, size1: float, pos2: float) -> float:
    """Signed gap: positive → gap exists, negative → overlap."""
    return pos2 - (pos1 + size1)


# ─── Публичные функции ────────────────────────────────────────────────────────

def validate_edge_gap(
    pos1: float,
    size1: float,
    pos2: float,
    max_gap: float = 5.0,
    max_overlap: float = 3.0,
) -> Optional[BoundaryViolation]:
    """Проверить зазор или перекрытие между двумя соседними рёбрами.

    Args:
        pos1:         Координата ближнего края первого фрагмента (пиксели).
        size1:        Размер первого фрагмента вдоль оси стыка.
        pos2:         Координата ближнего края второго фрагмента.
        max_gap:      Максимально допустимый зазор (пиксели).
        max_overlap:  Максимально допустимое перекрытие (пиксели).

    Returns:
        :class:`BoundaryViolation` при нарушении, иначе ``None``.

    Raises:
        ValueError: Если ``max_gap`` < 0 или ``max_overlap`` < 0.
    """
    if max_gap < 0:
        raise ValueError(f"max_gap must be >= 0, got {max_gap}")
    if max_overlap < 0:
        raise ValueError(f"max_overlap must be >= 0, got {max_overlap}")

    gap = _signed_gap(pos1, size1, pos2)
    if gap > max_gap:
        severity = gap - max_gap
        return BoundaryViolation(
            idx1=0, idx2=1,
            violation_type="gap",
            severity=float(severity),
            message=f"Gap of {gap:.2f}px exceeds max {max_gap:.2f}px",
            params={"gap": gap, "max_gap": max_gap},
        )
    if -gap > max_overlap:
        severity = (-gap) - max_overlap
        return BoundaryViolation(
            idx1=0, idx2=1,
            violation_type="overlap",
            severity=float(severity),
            message=f"Overlap of {-gap:.2f}px exceeds max {max_overlap:.2f}px",
            params={"overlap": -gap, "max_overlap": max_overlap},
        )
    return None


def validate_alignment(
    angle_deg: float,
    max_tilt_deg: float = 2.0,
) -> Optional[BoundaryViolation]:
    """Проверить угол наклона стыка между двумя фрагментами.

    Args:
        angle_deg:    Угол отклонения стыковой линии от горизонтали (градусы).
        max_tilt_deg: Максимально допустимый угол наклона.

    Returns:
        :class:`BoundaryViolation` типа ``"tilt"`` при нарушении, иначе ``None``.

    Raises:
        ValueError: Если ``max_tilt_deg`` ≤ 0.
    """
    if max_tilt_deg <= 0:
        raise ValueError(f"max_tilt_deg must be > 0, got {max_tilt_deg}")
    abs_angle = abs(float(angle_deg))
    if abs_angle > max_tilt_deg:
        severity = abs_angle - max_tilt_deg
        return BoundaryViolation(
            idx1=0, idx2=1,
            violation_type="tilt",
            severity=float(severity),
            message=f"Tilt of {abs_angle:.2f}° exceeds max {max_tilt_deg:.2f}°",
            params={"angle_deg": angle_deg, "max_tilt_deg": max_tilt_deg},
        )
    return None


def validate_pair(
    idx1: int,
    idx2: int,
    pos1: float,
    size1: float,
    pos2: float,
    angle_deg: float = 0.0,
    max_gap: float = 5.0,
    max_overlap: float = 3.0,
    max_tilt_deg: float = 2.0,
) -> List[BoundaryViolation]:
    """Комплексная проверка одной пары соседних фрагментов.

    Args:
        idx1:         Индекс первого фрагмента.
        idx2:         Индекс второго фрагмента.
        pos1:         Координата ближнего края первого фрагмента.
        size1:        Размер первого фрагмента вдоль оси стыка.
        pos2:         Координата ближнего края второго фрагмента.
        angle_deg:    Угол наклона стыковой линии.
        max_gap:      Максимально допустимый зазор.
        max_overlap:  Максимально допустимое перекрытие.
        max_tilt_deg: Максимально допустимый угол наклона.

    Returns:
        Список :class:`BoundaryViolation`; пуст, если нарушений нет.
    """
    violations: List[BoundaryViolation] = []

    gap_v = validate_edge_gap(pos1, size1, pos2, max_gap=max_gap, max_overlap=max_overlap)
    if gap_v is not None:
        gap_v.idx1 = idx1
        gap_v.idx2 = idx2
        violations.append(gap_v)

    tilt_v = validate_alignment(angle_deg, max_tilt_deg=max_tilt_deg)
    if tilt_v is not None:
        tilt_v.idx1 = idx1
        tilt_v.idx2 = idx2
        violations.append(tilt_v)

    return violations


def validate_all_pairs(
    pairs: List[Tuple[int, int]],
    positions: List[float],
    sizes: List[float],
    angles: Optional[List[float]] = None,
    max_gap: float = 5.0,
    max_overlap: float = 3.0,
    max_tilt_deg: float = 2.0,
    severity_threshold: float = 0.0,
) -> BoundaryReport:
    """Проверить все соседние пары в сборке.

    Args:
        pairs:              Список пар индексов (i, j) смежных фрагментов.
        positions:          Координата ближнего края каждого фрагмента.
        sizes:              Размер каждого фрагмента вдоль оси стыка.
        angles:             Углы наклона стыков; ``None`` → все 0°.
        max_gap:            Максимально допустимый зазор.
        max_overlap:        Максимально допустимое перекрытие.
        max_tilt_deg:       Максимально допустимый угол наклона.
        severity_threshold: Минимальная severity, при которой нарушение считается
                            критическим для ``is_valid``.

    Returns:
        :class:`BoundaryReport` со всеми обнаруженными нарушениями.
    """
    if not pairs:
        return BoundaryReport(
            violations=[],
            n_pairs_checked=0,
            is_valid=True,
            overall_score=1.0,
            params={
                "max_gap": max_gap,
                "max_overlap": max_overlap,
                "max_tilt_deg": max_tilt_deg,
            },
        )

    if angles is None:
        angles = [0.0] * len(pairs)

    all_violations: List[BoundaryViolation] = []
    for (i, j), angle in zip(pairs, angles):
        vs = validate_pair(
            idx1=i, idx2=j,
            pos1=positions[i], size1=sizes[i], pos2=positions[j],
            angle_deg=angle,
            max_gap=max_gap, max_overlap=max_overlap, max_tilt_deg=max_tilt_deg,
        )
        all_violations.extend(vs)

    critical = [v for v in all_violations if v.severity > severity_threshold]
    is_valid = len(critical) == 0

    score = boundary_quality_score(all_violations, n_pairs=len(pairs))

    return BoundaryReport(
        violations=all_violations,
        n_pairs_checked=len(pairs),
        is_valid=is_valid,
        overall_score=score,
        params={
            "max_gap": max_gap,
            "max_overlap": max_overlap,
            "max_tilt_deg": max_tilt_deg,
            "severity_threshold": severity_threshold,
        },
    )


def boundary_quality_score(
    violations: List[BoundaryViolation],
    n_pairs: int = 1,
    decay: float = 1.0,
) -> float:
    """Вычислить общий балл качества границ.

    Балл равен ``exp(-decay * mean_severity)`` по всем парам,
    где ``mean_severity`` — среднее значение severity (0 при отсутствии нарушений).

    Args:
        violations: Список всех нарушений.
        n_pairs:    Общее число проверенных пар (знаменатель для среднего).
        decay:      Коэффициент затухания (> 0).

    Returns:
        Балл в диапазоне (0, 1]; 1.0 при отсутствии нарушений.

    Raises:
        ValueError: Если ``n_pairs`` < 1 или ``decay`` ≤ 0.
    """
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be >= 1, got {n_pairs}")
    if decay <= 0:
        raise ValueError(f"decay must be > 0, got {decay}")
    if not violations:
        return 1.0
    total_severity = sum(v.severity for v in violations)
    mean_severity = total_severity / n_pairs
    return float(np.exp(-decay * mean_severity))
