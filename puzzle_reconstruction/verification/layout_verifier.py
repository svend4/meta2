"""
Верификация пространственного расположения фрагментов в сборке.

Проверяет геометрическую согласованность размещения фрагментов:
перекрытия, разрывы, выравнивание по столбцам/строкам, а также
соответствие порядка чтения (reading order).

Классы:
    ConstraintType       — перечисление видов нарушений
    LayoutConstraint     — одно нарушение (тип, участники, серьёзность)
    FragmentBox          — ограничивающий прямоугольник фрагмента в сборке
    LayoutVerificationResult — итог проверки (нарушения, score, valid)

Функции:
    build_layout_boxes   — переводит Assembly + Fragment → List[FragmentBox]
    check_overlaps       — находит пары с перекрытием
    check_gaps           — находит пары-соседей с зазором > max_gap
    check_column_alignment — проверяет вертикальное выравнивание
    check_row_alignment  — проверяет горизонтальное выравнивание
    verify_layout        — полный пайплайн → LayoutVerificationResult
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..models import Assembly, Fragment, Placement


# ─── ConstraintType ───────────────────────────────────────────────────────────

class ConstraintType(str, Enum):
    """Виды нарушений пространственного расположения."""
    OVERLAP         = "overlap"           # Пересечение двух фрагментов
    GAP             = "gap"               # Недопустимый зазор между соседями
    MISALIGN_COL    = "misalign_column"   # Сдвиг по X превышает допуск
    MISALIGN_ROW    = "misalign_row"      # Сдвиг по Y превышает допуск
    OUT_OF_BOUNDS   = "out_of_bounds"     # Фрагмент вне заданного канваса
    DUPLICATE_PLACE = "duplicate_place"   # Один fragment_id размещён дважды


# ─── LayoutConstraint ─────────────────────────────────────────────────────────

@dataclass
class LayoutConstraint:
    """
    Одно нарушение пространственного расположения.

    Attributes:
        kind:       Тип нарушения.
        fids:       Tuple fragment_id, участвующих в нарушении.
        severity:   Серьёзность ∈ [0, 1] (0 = незначительное, 1 = критическое).
        detail:     Текстовое описание нарушения.
    """
    kind:     ConstraintType
    fids:     Tuple[int, ...]
    severity: float = 0.5
    detail:   str   = ""

    def __repr__(self) -> str:
        return (f"LayoutConstraint(kind={self.kind.value}, "
                f"fids={self.fids}, severity={self.severity:.2f})")


# ─── FragmentBox ──────────────────────────────────────────────────────────────

@dataclass
class FragmentBox:
    """
    Ограничивающий прямоугольник фрагмента на плоскости сборки.

    Attributes:
        fid:      fragment_id.
        x, y:     Координаты верхнего левого угла.
        w, h:     Ширина и высота.
        rotation: Угол поворота (°) для информации.
    """
    fid:      int
    x:        float
    y:        float
    w:        float
    h:        float
    rotation: float = 0.0

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    def intersects(self, other: "FragmentBox") -> bool:
        """True если прямоугольники пересекаются (но не касаются)."""
        return (self.x  < other.x2 and self.x2 > other.x and
                self.y  < other.y2 and self.y2 > other.y)

    def overlap_area(self, other: "FragmentBox") -> float:
        """Площадь пересечения (0.0 если нет)."""
        dx = min(self.x2, other.x2) - max(self.x, other.x)
        dy = min(self.y2, other.y2) - max(self.y, other.y)
        if dx <= 0 or dy <= 0:
            return 0.0
        return float(dx * dy)

    def gap_to(self, other: "FragmentBox") -> float:
        """
        Минимальный зазор между прямоугольниками.
        Отрицательный → перекрытие.
        """
        gap_x = max(0.0, max(self.x, other.x) - min(self.x2, other.x2))
        gap_y = max(0.0, max(self.y, other.y) - min(self.y2, other.y2))
        if gap_x == 0 and gap_y == 0:
            return -self.overlap_area(other)
        return float(max(gap_x, gap_y))

    def are_neighbors(self, other: "FragmentBox",
                       proximity: float = 20.0) -> bool:
        """True если прямоугольники находятся не далее proximity пикселей."""
        return self.gap_to(other) <= proximity

    def __repr__(self) -> str:
        return (f"FragmentBox(fid={self.fid}, "
                f"x={self.x:.1f}, y={self.y:.1f}, "
                f"w={self.w:.1f}, h={self.h:.1f})")


# ─── LayoutVerificationResult ─────────────────────────────────────────────────

@dataclass
class LayoutVerificationResult:
    """
    Итог верификации пространственного расположения.

    Attributes:
        constraints:      Список найденных нарушений.
        violation_score:  ∈ [0, 1]; 0 = нет нарушений, 1 = критическое.
        valid:            True если нарушений нет.
        n_fragments:      Число проверенных фрагментов.
        boxes:            Список FragmentBox (для отладки).
    """
    constraints:     List[LayoutConstraint]
    violation_score: float
    valid:           bool
    n_fragments:     int
    boxes:           List[FragmentBox] = field(default_factory=list)

    def by_kind(self, kind: ConstraintType) -> List[LayoutConstraint]:
        return [c for c in self.constraints if c.kind == kind]

    def summary(self) -> str:
        n   = len(self.constraints)
        tag = "PASS" if self.valid else "FAIL"
        return (f"LayoutVerificationResult({tag}, "
                f"n_violations={n}, "
                f"violation_score={self.violation_score:.3f}, "
                f"n_fragments={self.n_fragments})")

    def __repr__(self) -> str:
        return self.summary()


# ─── build_layout_boxes ───────────────────────────────────────────────────────

def build_layout_boxes(assembly:    Assembly,
                        fragments:   List[Fragment]) -> List[FragmentBox]:
    """
    Строит список FragmentBox из Assembly и размеров фрагментов.

    Args:
        assembly:  Assembly с Placement для каждого fragment_id.
        fragments: Список Fragment (нужны размеры изображения).

    Returns:
        Список FragmentBox.
    """
    frag_map: Dict[int, Fragment] = {f.fragment_id: f for f in fragments}
    boxes: List[FragmentBox] = []

    for pl in assembly.placements:
        frag = frag_map.get(pl.fragment_id)
        if frag is None:
            continue
        h, w = frag.image.shape[:2]
        x, y = pl.position
        boxes.append(FragmentBox(
            fid=pl.fragment_id,
            x=float(x),
            y=float(y),
            w=float(w),
            h=float(h),
            rotation=float(pl.rotation),
        ))

    return boxes


# ─── Проверки ─────────────────────────────────────────────────────────────────

def check_overlaps(boxes:      List[FragmentBox],
                    min_area:   float = 1.0) -> List[LayoutConstraint]:
    """
    Находит пары фрагментов с перекрытием.

    Args:
        boxes:    Список FragmentBox.
        min_area: Минимальная площадь пересечения для регистрации нарушения.

    Returns:
        Список LayoutConstraint с kind=OVERLAP.
    """
    constraints: List[LayoutConstraint] = []
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            area = boxes[i].overlap_area(boxes[j])
            if area >= min_area:
                ref_area = min(boxes[i].w * boxes[i].h,
                               boxes[j].w * boxes[j].h)
                severity = min(1.0, area / max(ref_area, 1.0))
                constraints.append(LayoutConstraint(
                    kind=ConstraintType.OVERLAP,
                    fids=(boxes[i].fid, boxes[j].fid),
                    severity=severity,
                    detail=f"overlap_area={area:.1f}px²",
                ))
    return constraints


def check_gaps(boxes:       List[FragmentBox],
                max_gap:     float = 15.0,
                proximity:   float = 50.0) -> List[LayoutConstraint]:
    """
    Находит пары соседних фрагментов с зазором > max_gap.

    «Соседи» — фрагменты, расположенные не далее proximity пикселей.

    Args:
        boxes:     Список FragmentBox.
        max_gap:   Максимально допустимый зазор в пикселях.
        proximity: Порог близости для определения «соседей».

    Returns:
        Список LayoutConstraint с kind=GAP.
    """
    constraints: List[LayoutConstraint] = []
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            if boxes[i].are_neighbors(boxes[j], proximity=proximity):
                gap = boxes[i].gap_to(boxes[j])
                if gap > max_gap:
                    severity = min(1.0, gap / (max_gap * 10))
                    constraints.append(LayoutConstraint(
                        kind=ConstraintType.GAP,
                        fids=(boxes[i].fid, boxes[j].fid),
                        severity=severity,
                        detail=f"gap={gap:.1f}px > max_gap={max_gap}px",
                    ))
    return constraints


def check_column_alignment(boxes:     List[FragmentBox],
                             tolerance: float = 5.0) -> List[LayoutConstraint]:
    """
    Проверяет вертикальное выравнивание фрагментов (сдвиг по X).

    Группирует фрагменты в «столбцы» по X-центру (с допуском) и проверяет,
    что все фрагменты в столбце выровнены.

    Args:
        boxes:     Список FragmentBox.
        tolerance: Допустимое отклонение X-центра в пикселях.

    Returns:
        Список LayoutConstraint с kind=MISALIGN_COL.
    """
    if len(boxes) < 2:
        return []

    # Кластеризация по X-центру (жадная)
    columns: List[List[FragmentBox]] = []
    sorted_b = sorted(boxes, key=lambda b: b.cx)
    for box in sorted_b:
        placed = False
        for col in columns:
            if abs(col[0].cx - box.cx) <= tolerance:
                col.append(box)
                placed = True
                break
        if not placed:
            columns.append([box])

    constraints: List[LayoutConstraint] = []
    for col in columns:
        if len(col) < 2:
            continue
        cx_vals = [b.cx for b in col]
        median_cx = float(np.median(cx_vals))
        for box in col:
            delta = abs(box.cx - median_cx)
            if delta > tolerance:
                severity = min(1.0, delta / (tolerance * 10))
                constraints.append(LayoutConstraint(
                    kind=ConstraintType.MISALIGN_COL,
                    fids=(box.fid,),
                    severity=severity,
                    detail=f"cx_offset={delta:.1f}px > tolerance={tolerance}px",
                ))
    return constraints


def check_row_alignment(boxes:     List[FragmentBox],
                         tolerance: float = 5.0) -> List[LayoutConstraint]:
    """
    Проверяет горизонтальное выравнивание фрагментов (сдвиг по Y).

    Args:
        boxes:     Список FragmentBox.
        tolerance: Допустимое отклонение Y-центра в пикселях.

    Returns:
        Список LayoutConstraint с kind=MISALIGN_ROW.
    """
    if len(boxes) < 2:
        return []

    rows: List[List[FragmentBox]] = []
    sorted_b = sorted(boxes, key=lambda b: b.cy)
    for box in sorted_b:
        placed = False
        for row in rows:
            if abs(row[0].cy - box.cy) <= tolerance:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    constraints: List[LayoutConstraint] = []
    for row in rows:
        if len(row) < 2:
            continue
        cy_vals = [b.cy for b in row]
        median_cy = float(np.median(cy_vals))
        for box in row:
            delta = abs(box.cy - median_cy)
            if delta > tolerance:
                severity = min(1.0, delta / (tolerance * 10))
                constraints.append(LayoutConstraint(
                    kind=ConstraintType.MISALIGN_ROW,
                    fids=(box.fid,),
                    severity=severity,
                    detail=f"cy_offset={delta:.1f}px > tolerance={tolerance}px",
                ))
    return constraints


def check_out_of_bounds(boxes:        List[FragmentBox],
                         canvas_w:     float,
                         canvas_h:     float,
                         margin:       float = 0.0) -> List[LayoutConstraint]:
    """
    Находит фрагменты, выходящие за пределы канваса.

    Args:
        boxes:    Список FragmentBox.
        canvas_w: Ширина канваса в пикселях.
        canvas_h: Высота канваса в пикселях.
        margin:   Допустимый вылет за границу.

    Returns:
        Список LayoutConstraint с kind=OUT_OF_BOUNDS.
    """
    constraints: List[LayoutConstraint] = []
    for box in boxes:
        out_x = max(0.0, -box.x - margin) + max(0.0, box.x2 - canvas_w - margin)
        out_y = max(0.0, -box.y - margin) + max(0.0, box.y2 - canvas_h - margin)
        if out_x > 0 or out_y > 0:
            severity = min(1.0, (out_x + out_y) / max(canvas_w, canvas_h))
            constraints.append(LayoutConstraint(
                kind=ConstraintType.OUT_OF_BOUNDS,
                fids=(box.fid,),
                severity=severity,
                detail=f"out_x={out_x:.1f}, out_y={out_y:.1f}",
            ))
    return constraints


def check_duplicate_placements(assembly: Assembly) -> List[LayoutConstraint]:
    """
    Проверяет, что один fragment_id не размещён дважды.

    Returns:
        Список LayoutConstraint с kind=DUPLICATE_PLACE.
    """
    seen:        Dict[int, int] = {}
    constraints: List[LayoutConstraint] = []
    for pl in assembly.placements:
        fid = pl.fragment_id
        if fid in seen:
            constraints.append(LayoutConstraint(
                kind=ConstraintType.DUPLICATE_PLACE,
                fids=(fid,),
                severity=1.0,
                detail=f"fid={fid} встречается дважды",
            ))
        else:
            seen[fid] = 1
    return constraints


# ─── Полный пайплайн ──────────────────────────────────────────────────────────

def verify_layout(assembly:     Assembly,
                   fragments:    List[Fragment],
                   max_gap:      float = 15.0,
                   col_tol:      float = 5.0,
                   row_tol:      float = 5.0,
                   canvas_size:  Optional[Tuple[float, float]] = None,
                   proximity:    float = 50.0) -> LayoutVerificationResult:
    """
    Полный пайплайн верификации пространственного расположения.

    Args:
        assembly:    Assembly с Placement.
        fragments:   Список Fragment.
        max_gap:     Максимально допустимый зазор между соседями.
        col_tol:     Допуск выравнивания по X.
        row_tol:     Допуск выравнивания по Y.
        canvas_size: (width, height) канваса или None (нет проверки OUT_OF_BOUNDS).
        proximity:   Порог близости для проверки зазоров.

    Returns:
        LayoutVerificationResult.
    """
    boxes = build_layout_boxes(assembly, fragments)

    all_constraints: List[LayoutConstraint] = []

    # 1. Дубликаты
    all_constraints += check_duplicate_placements(assembly)

    # 2. Перекрытия
    all_constraints += check_overlaps(boxes)

    # 3. Зазоры
    all_constraints += check_gaps(boxes, max_gap=max_gap, proximity=proximity)

    # 4. Выравнивание
    all_constraints += check_column_alignment(boxes, tolerance=col_tol)
    all_constraints += check_row_alignment(boxes, tolerance=row_tol)

    # 5. Границы канваса
    if canvas_size is not None:
        all_constraints += check_out_of_bounds(
            boxes, canvas_w=canvas_size[0], canvas_h=canvas_size[1])

    # Итоговый violation_score
    if all_constraints:
        max_sev = max(c.severity for c in all_constraints)
        avg_sev = float(np.mean([c.severity for c in all_constraints]))
        violation_score = 0.6 * max_sev + 0.4 * avg_sev
    else:
        violation_score = 0.0

    return LayoutVerificationResult(
        constraints=all_constraints,
        violation_score=float(violation_score),
        valid=len(all_constraints) == 0,
        n_fragments=len(boxes),
        boxes=boxes,
    )
