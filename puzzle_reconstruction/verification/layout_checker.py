"""
Статистическая проверка геометрической согласованности сборки.

Анализирует позиции фрагментов на предмет перекрытий, зазоров,
выравнивания по сетке и соотношения сторон итогового документа.

Классы:
    LayoutViolationType — перечисление типов нарушений
    LayoutViolation     — одно нарушение расположения
    LayoutCheckResult   — результат проверки (score, violations, bbox)

Функции:
    compute_bounding_box       — AABB для набора позиций фрагментов
    detect_overlaps            — поиск пересечений бокс-пар
    detect_gaps                — аномальные зазоры между соседними фрагментами
    check_grid_alignment       — отклонение от прямоугольной сетки
    check_aspect_ratio         — проверка соотношения сторон сборки
    check_layout               — полная проверка → LayoutCheckResult
    batch_check_layout         — пакетная проверка нескольких сборок
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── LayoutViolationType ──────────────────────────────────────────────────────

class LayoutViolationType(str, Enum):
    OVERLAP      = "overlap"
    GAP          = "gap"
    MISALIGNMENT = "misalignment"
    ASPECT_RATIO = "aspect_ratio"
    BOUNDARY     = "boundary"
    INSUFFICIENT = "insufficient"


# ─── LayoutViolation ──────────────────────────────────────────────────────────

@dataclass
class LayoutViolation:
    """
    Одно геометрическое нарушение в сборке.

    Attributes:
        type:         Тип нарушения.
        severity:     Тяжесть ∈ [0,1].
        fragment_ids: Идентификаторы затронутых фрагментов.
        description:  Текстовое описание.
        values:       Числовые характеристики нарушения.
    """
    type:         LayoutViolationType
    severity:     float
    fragment_ids: List[int]        = field(default_factory=list)
    description:  str              = ""
    values:       Dict[str, float] = field(default_factory=dict)


# ─── LayoutCheckResult ────────────────────────────────────────────────────────

@dataclass
class LayoutCheckResult:
    """
    Результат геометрической проверки сборки.

    Attributes:
        violations:    Список обнаруженных нарушений.
        score:         Качество расположения ∈ [0,1] (1 = идеально).
        n_checked:     Число проверенных фрагментов.
        bounding_box:  (x_min, y_min, w, h) всей сборки или None.
        method_scores: Подоценки {'overlap','gap','alignment','aspect_ratio'}.
    """
    violations:    List[LayoutViolation]
    score:         float
    n_checked:     int
    bounding_box:  Optional[Tuple[float, float, float, float]] = None
    method_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def n_violations(self) -> int:
        return len(self.violations)

    @property
    def max_severity(self) -> float:
        return max((v.severity for v in self.violations), default=0.0)


# ─── Геометрические вспомогательные ──────────────────────────────────────────

def compute_bounding_box(
        positions: Dict[int, Tuple[float, float, float, float]]
) -> Tuple[float, float, float, float]:
    """
    Вычисляет ограничивающий прямоугольник (AABB) набора фрагментов.

    Args:
        positions: {fragment_id: (x, y, w, h)}.

    Returns:
        (x_min, y_min, total_w, total_h).
        Для пустого словаря возвращает (0.0, 0.0, 0.0, 0.0).
    """
    if not positions:
        return (0.0, 0.0, 0.0, 0.0)
    xs  = [v[0] for v in positions.values()]
    ys  = [v[1] for v in positions.values()]
    x2s = [v[0] + v[2] for v in positions.values()]
    y2s = [v[1] + v[3] for v in positions.values()]
    xm, ym = min(xs), min(ys)
    return (xm, ym, max(x2s) - xm, max(y2s) - ym)


def _overlap_area(box1: Tuple, box2: Tuple) -> float:
    """Площадь пересечения двух прямоугольников (x,y,w,h)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ox = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
    oy = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return ox * oy


def _iou_1d(a_start: float, a_end: float,
             b_start: float, b_end: float) -> float:
    """Пересечение/объединение двух отрезков на числовой оси."""
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


# ─── detect_overlaps ──────────────────────────────────────────────────────────

def detect_overlaps(
        fragment_ids: List[int],
        positions:    Dict[int, Tuple[float, float, float, float]],
        min_overlap:  float = 5.0,
) -> List[LayoutViolation]:
    """
    Находит пары фрагментов с площадью перекрытия > ``min_overlap`` px².

    Args:
        fragment_ids: Список идентификаторов.
        positions:    {id: (x, y, w, h)}.
        min_overlap:  Порог площади перекрытия.

    Returns:
        Список LayoutViolation с type=OVERLAP.
    """
    violations: List[LayoutViolation] = []
    ids = [fid for fid in fragment_ids if fid in positions]

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            fi, fj = ids[i], ids[j]
            area = _overlap_area(positions[fi], positions[fj])
            if area > min_overlap:
                min_frag = min(
                    positions[fi][2] * positions[fi][3],
                    positions[fj][2] * positions[fj][3],
                )
                severity = float(np.clip(area / max(min_frag, 1.0), 0.0, 1.0))
                violations.append(LayoutViolation(
                    type=LayoutViolationType.OVERLAP,
                    severity=severity,
                    fragment_ids=[fi, fj],
                    description=f"Overlap {area:.1f}px² between {fi} and {fj}",
                    values={"overlap_area": area},
                ))
    return violations


# ─── detect_gaps ──────────────────────────────────────────────────────────────

def detect_gaps(
        fragment_ids: List[int],
        positions:    Dict[int, Tuple[float, float, float, float]],
        expected_gap: float = 0.0,
        gap_tol:      float = 10.0,
) -> List[LayoutViolation]:
    """
    Находит аномальные зазоры между соседними фрагментами.

    Два фрагмента считаются соседними, если они выровнены по одной оси
    с перекрытием > 50 % по перпендикулярной.

    Args:
        fragment_ids:  Список идентификаторов.
        positions:     {id: (x, y, w, h)}.
        expected_gap:  Ожидаемый зазор (обычно 0).
        gap_tol:       Допустимое отклонение от expected_gap.

    Returns:
        Список LayoutViolation с type=GAP.
    """
    violations: List[LayoutViolation] = []
    ids = [fid for fid in fragment_ids if fid in positions]

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            fi, fj = ids[i], ids[j]
            xi, yi, wi, hi = positions[fi]
            xj, yj, wj, hj = positions[fj]

            # Горизонтальные соседи (перекрытие по Y > 50 %)
            if _iou_1d(yi, yi + hi, yj, yj + hj) > 0.5:
                gap = max(xi, xj) - min(xi + wi, xj + wj)
                if gap > 0 and abs(gap - expected_gap) > gap_tol:
                    sev = float(np.clip(
                        abs(gap - expected_gap) / max(gap_tol * 5, 1.0), 0, 1))
                    violations.append(LayoutViolation(
                        type=LayoutViolationType.GAP,
                        severity=sev,
                        fragment_ids=[fi, fj],
                        description=f"Horizontal gap {gap:.1f}px between {fi} and {fj}",
                        values={"gap": gap, "expected": expected_gap},
                    ))

            # Вертикальные соседи (перекрытие по X > 50 %)
            if _iou_1d(xi, xi + wi, xj, xj + wj) > 0.5:
                gap = max(yi, yj) - min(yi + hi, yj + hj)
                if gap > 0 and abs(gap - expected_gap) > gap_tol:
                    sev = float(np.clip(
                        abs(gap - expected_gap) / max(gap_tol * 5, 1.0), 0, 1))
                    violations.append(LayoutViolation(
                        type=LayoutViolationType.GAP,
                        severity=sev,
                        fragment_ids=[fi, fj],
                        description=f"Vertical gap {gap:.1f}px between {fi} and {fj}",
                        values={"gap": gap, "expected": expected_gap},
                    ))

    return violations


# ─── check_grid_alignment ─────────────────────────────────────────────────────

def check_grid_alignment(
        fragment_ids: List[int],
        positions:    Dict[int, Tuple[float, float, float, float]],
        tol_px:       float = 5.0,
) -> List[LayoutViolation]:
    """
    Проверяет выравнивание фрагментов по прямоугольной сетке.

    Нарушением считается отклонение X или Y от ближайшей координаты сетки
    (кластеры левых/верхних краёв) более чем на ``tol_px``.

    Args:
        fragment_ids: Список идентификаторов.
        positions:    {id: (x, y, w, h)}.
        tol_px:       Пиксельный допуск.

    Returns:
        Список LayoutViolation с type=MISALIGNMENT.
    """
    violations: List[LayoutViolation] = []
    ids = [fid for fid in fragment_ids if fid in positions]
    if len(ids) < 2:
        return violations

    xs = np.array([positions[fid][0] for fid in ids])
    ys = np.array([positions[fid][1] for fid in ids])

    step = max(tol_px, 1.0)
    grid_xs = np.unique(np.round(xs / step).astype(int)) * step
    grid_ys = np.unique(np.round(ys / step).astype(int)) * step

    for fid, xi, yi in zip(ids, xs, ys):
        dx = float(np.min(np.abs(grid_xs - xi)))
        dy = float(np.min(np.abs(grid_ys - yi)))

        if dx > tol_px:
            sev = float(np.clip(dx / (tol_px * 10 + 1e-9), 0, 1))
            violations.append(LayoutViolation(
                type=LayoutViolationType.MISALIGNMENT,
                severity=sev,
                fragment_ids=[fid],
                description=f"X misalignment {dx:.1f}px for fragment {fid}",
                values={"dx": dx, "tol": tol_px},
            ))
        if dy > tol_px:
            sev = float(np.clip(dy / (tol_px * 10 + 1e-9), 0, 1))
            violations.append(LayoutViolation(
                type=LayoutViolationType.MISALIGNMENT,
                severity=sev,
                fragment_ids=[fid],
                description=f"Y misalignment {dy:.1f}px for fragment {fid}",
                values={"dy": dy, "tol": tol_px},
            ))

    return violations


# ─── check_aspect_ratio ───────────────────────────────────────────────────────

def check_aspect_ratio(
        fragment_ids:   List[int],
        positions:      Dict[int, Tuple[float, float, float, float]],
        expected_ratio: Optional[float] = None,
        tol_ratio:      float = 0.3,
) -> List[LayoutViolation]:
    """
    Проверяет соотношение сторон итоговой сборки.

    Args:
        fragment_ids:   Список идентификаторов.
        positions:      {id: (x, y, w, h)}.
        expected_ratio: Ожидаемое W/H (None → проверка не выполняется).
        tol_ratio:      Допустимое относительное отклонение.

    Returns:
        Список из 0 или 1 LayoutViolation с type=ASPECT_RATIO.
    """
    ids = [fid for fid in fragment_ids if fid in positions]
    if not ids or expected_ratio is None:
        return []

    _, _, bw, bh = compute_bounding_box({fid: positions[fid] for fid in ids})
    if bh < 1e-6:
        return []

    actual    = bw / bh
    deviation = abs(actual - expected_ratio) / max(expected_ratio, 1e-6)

    if deviation > tol_ratio:
        sev = float(np.clip(deviation / (tol_ratio * 3 + 1e-9), 0, 1))
        return [LayoutViolation(
            type=LayoutViolationType.ASPECT_RATIO,
            severity=sev,
            fragment_ids=ids,
            description=f"Aspect ratio {actual:.2f} vs expected {expected_ratio:.2f}",
            values={"actual": actual, "expected": expected_ratio,
                    "deviation": deviation},
        )]
    return []


# ─── check_layout ─────────────────────────────────────────────────────────────

def check_layout(
        fragment_ids:   List[int],
        positions:      Dict[int, Tuple[float, float, float, float]],
        overlap_min:    float = 5.0,
        gap_tol:        float = 10.0,
        alignment_tol:  float = 5.0,
        expected_ratio: Optional[float] = None,
        ratio_tol:      float = 0.3,
) -> LayoutCheckResult:
    """
    Полная геометрическая проверка сборки (4 метода).

    violation_score = 0.6·max_severity + 0.4·mean_severity.
    score = 1 − violation_score.

    Args:
        fragment_ids:   Список идентификаторов фрагментов.
        positions:      {id: (x, y, w, h)} в пикселях.
        overlap_min:    Порог площади перекрытия.
        gap_tol:        Допустимый зазор в пикселях.
        alignment_tol:  Допустимое смещение от сетки.
        expected_ratio: Ожидаемое W/H (None → пропустить).
        ratio_tol:      Допустимое относительное отклонение соотношения.

    Returns:
        LayoutCheckResult.
    """
    ids = [fid for fid in fragment_ids if fid in positions]

    if len(ids) < 2:
        bbox = compute_bounding_box({fid: positions[fid] for fid in ids}) if ids else None
        return LayoutCheckResult(
            violations=[],
            score=1.0,
            n_checked=len(ids),
            bounding_box=bbox,
            method_scores={
                "overlap": 1.0, "gap": 1.0,
                "alignment": 1.0, "aspect_ratio": 1.0,
            },
        )

    overlaps   = detect_overlaps(ids, positions, min_overlap=overlap_min)
    gaps       = detect_gaps(ids, positions, gap_tol=gap_tol)
    alignments = check_grid_alignment(ids, positions, tol_px=alignment_tol)
    aspects    = check_aspect_ratio(ids, positions,
                                     expected_ratio=expected_ratio,
                                     tol_ratio=ratio_tol)

    all_violations = overlaps + gaps + alignments + aspects

    def _score_for(viols: List[LayoutViolation]) -> float:
        if not viols:
            return 1.0
        sev = [v.severity for v in viols]
        vs  = 0.6 * max(sev) + 0.4 * float(np.mean(sev))
        return float(np.clip(1.0 - vs, 0.0, 1.0))

    method_scores = {
        "overlap":      _score_for(overlaps),
        "gap":          _score_for(gaps),
        "alignment":    _score_for(alignments),
        "aspect_ratio": _score_for(aspects),
    }

    if all_violations:
        sev   = [v.severity for v in all_violations]
        vs    = 0.6 * max(sev) + 0.4 * float(np.mean(sev))
        score = float(np.clip(1.0 - vs, 0.0, 1.0))
    else:
        score = 1.0

    bbox = compute_bounding_box({fid: positions[fid] for fid in ids})

    return LayoutCheckResult(
        violations=all_violations,
        score=score,
        n_checked=len(ids),
        bounding_box=bbox,
        method_scores=method_scores,
    )


# ─── batch_check_layout ───────────────────────────────────────────────────────

def batch_check_layout(
        fragment_id_groups: List[List[int]],
        position_groups:    List[Dict[int, Tuple[float, float, float, float]]],
        **kwargs,
) -> List[LayoutCheckResult]:
    """
    Пакетная проверка нескольких сборок.

    Args:
        fragment_id_groups: Список списков идентификаторов.
        position_groups:    Список словарей позиций.
        **kwargs:           Передаются в check_layout.

    Returns:
        Список LayoutCheckResult по одному на сборку.

    Raises:
        ValueError: Если длины групп не совпадают.
    """
    if len(fragment_id_groups) != len(position_groups):
        raise ValueError(
            f"Groups length mismatch: "
            f"{len(fragment_id_groups)} vs {len(position_groups)}"
        )
    return [
        check_layout(fids, pos, **kwargs)
        for fids, pos in zip(fragment_id_groups, position_groups)
    ]
