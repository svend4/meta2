"""Валидация осевых ограничивающих прямоугольников размещения фрагментов.

Модуль проверяет корректность набора прямоугольных размещений:
коллизии осевых bbox, дублирующиеся позиции, выход за пределы холста
и минимальное покрытие холста.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── PlacementConfig ──────────────────────────────────────────────────────────

@dataclass
class PlacementConfig:
    """Параметры валидации размещений.

    Атрибуты:
        iou_threshold:   Минимальный IoU для регистрации коллизии (>= 0).
        min_coverage:    Минимальное покрытие холста [0, 1].
        canvas_w:        Ширина холста (>= 1, 0 — отключить проверку).
        canvas_h:        Высота холста (>= 1, 0 — отключить проверку).
    """

    iou_threshold: float = 0.0
    min_coverage: float = 0.0
    canvas_w: int = 0
    canvas_h: int = 0

    def __post_init__(self) -> None:
        if self.iou_threshold < 0.0:
            raise ValueError(
                f"iou_threshold должен быть >= 0, получено {self.iou_threshold}"
            )
        if not (0.0 <= self.min_coverage <= 1.0):
            raise ValueError(
                f"min_coverage должен быть в [0, 1], получено {self.min_coverage}"
            )
        if self.canvas_w < 0:
            raise ValueError(
                f"canvas_w должен быть >= 0, получено {self.canvas_w}"
            )
        if self.canvas_h < 0:
            raise ValueError(
                f"canvas_h должен быть >= 0, получено {self.canvas_h}"
            )


# ─── PlacementBox ─────────────────────────────────────────────────────────────

@dataclass
class PlacementBox:
    """Ограничивающий прямоугольник размещения фрагмента.

    Атрибуты:
        fragment_id: Идентификатор фрагмента (>= 0).
        x:           Левая граница (>= 0).
        y:           Верхняя граница (>= 0).
        w:           Ширина (>= 1).
        h:           Высота (>= 1).
    """

    fragment_id: int
    x: int
    y: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        if self.x < 0:
            raise ValueError(f"x должен быть >= 0, получено {self.x}")
        if self.y < 0:
            raise ValueError(f"y должен быть >= 0, получено {self.y}")
        if self.w < 1:
            raise ValueError(f"w должен быть >= 1, получено {self.w}")
        if self.h < 1:
            raise ValueError(f"h должен быть >= 1, получено {self.h}")

    @property
    def x2(self) -> int:
        """Правая граница (не включительно)."""
        return self.x + self.w

    @property
    def y2(self) -> int:
        """Нижняя граница (не включительно)."""
        return self.y + self.h

    @property
    def area(self) -> int:
        """Площадь прямоугольника."""
        return self.w * self.h

    @property
    def center(self) -> Tuple[float, float]:
        """Центр прямоугольника (cx, cy)."""
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


# ─── CollisionReport ──────────────────────────────────────────────────────────

@dataclass
class CollisionReport:
    """Отчёт о результатах валидации набора размещений.

    Атрибуты:
        collisions:   Список пар (id_a, id_b) с IoU > порога.
        duplicates:   Список пар (id_a, id_b) с одинаковыми (x,y,w,h).
        out_of_bounds: Список fragment_id, выходящих за пределы холста.
        coverage:     Доля покрытого холста (0.0 если canvas не задан).
        n_checked:    Число проверенных прямоугольников (>= 0).
    """

    collisions: List[Tuple[int, int]]
    duplicates: List[Tuple[int, int]]
    out_of_bounds: List[int]
    coverage: float
    n_checked: int

    def __post_init__(self) -> None:
        if self.n_checked < 0:
            raise ValueError(
                f"n_checked должен быть >= 0, получено {self.n_checked}"
            )
        if not (0.0 <= self.coverage <= 1.0 + 1e-9):
            raise ValueError(
                f"coverage должен быть в [0, 1], получено {self.coverage}"
            )

    @property
    def is_valid(self) -> bool:
        """True если нет коллизий, дублей и выходов за пределы."""
        return (
            len(self.collisions) == 0
            and len(self.duplicates) == 0
            and len(self.out_of_bounds) == 0
        )

    @property
    def n_issues(self) -> int:
        """Суммарное число проблем."""
        return (
            len(self.collisions)
            + len(self.duplicates)
            + len(self.out_of_bounds)
        )


# ─── box_iou ──────────────────────────────────────────────────────────────────

def box_iou(a: PlacementBox, b: PlacementBox) -> float:
    """Вычислить IoU двух PlacementBox.

    Аргументы:
        a, b: Прямоугольники.

    Возвращает:
        IoU в [0, 1].
    """
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = a.area + b.area - inter
    return float(inter) / float(union) if union > 0 else 0.0


# ─── find_collisions ──────────────────────────────────────────────────────────

def find_collisions(
    boxes: List[PlacementBox],
    iou_threshold: float = 0.0,
) -> List[Tuple[int, int]]:
    """Найти пары прямоугольников с IoU > порога.

    Аргументы:
        boxes:         Список прямоугольников.
        iou_threshold: Порог IoU (>= 0).

    Возвращает:
        Список пар (fragment_id_a, fragment_id_b).

    Исключения:
        ValueError: Если iou_threshold < 0.
    """
    if iou_threshold < 0.0:
        raise ValueError(
            f"iou_threshold должен быть >= 0, получено {iou_threshold}"
        )
    result: List[Tuple[int, int]] = []
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            if box_iou(boxes[i], boxes[j]) > iou_threshold:
                result.append((
                    min(boxes[i].fragment_id, boxes[j].fragment_id),
                    max(boxes[i].fragment_id, boxes[j].fragment_id),
                ))
    return result


# ─── find_duplicate_positions ─────────────────────────────────────────────────

def find_duplicate_positions(
    boxes: List[PlacementBox],
) -> List[Tuple[int, int]]:
    """Найти пары с одинаковыми координатами (x, y, w, h).

    Аргументы:
        boxes: Список прямоугольников.

    Возвращает:
        Список пар (fragment_id_a, fragment_id_b).
    """
    seen: Dict[Tuple[int, int, int, int], int] = {}
    duplicates: List[Tuple[int, int]] = []
    for box in boxes:
        key = (box.x, box.y, box.w, box.h)
        if key in seen:
            duplicates.append((seen[key], box.fragment_id))
        else:
            seen[key] = box.fragment_id
    return duplicates


# ─── find_out_of_bounds ───────────────────────────────────────────────────────

def find_out_of_bounds(
    boxes: List[PlacementBox],
    canvas_w: int,
    canvas_h: int,
) -> List[int]:
    """Найти фрагменты, выходящие за пределы холста.

    Аргументы:
        boxes:    Список прямоугольников.
        canvas_w: Ширина холста (>= 1).
        canvas_h: Высота холста (>= 1).

    Возвращает:
        Список fragment_id нарушителей.

    Исключения:
        ValueError: Если canvas_w < 1 или canvas_h < 1.
    """
    if canvas_w < 1:
        raise ValueError(f"canvas_w должен быть >= 1, получено {canvas_w}")
    if canvas_h < 1:
        raise ValueError(f"canvas_h должен быть >= 1, получено {canvas_h}")
    return [b.fragment_id for b in boxes if b.x2 > canvas_w or b.y2 > canvas_h]


# ─── compute_coverage ─────────────────────────────────────────────────────────

def compute_coverage(
    boxes: List[PlacementBox],
    canvas_w: int,
    canvas_h: int,
) -> float:
    """Вычислить долю холста, покрытую прямоугольниками.

    Аргументы:
        boxes:    Список прямоугольников.
        canvas_w: Ширина холста (>= 1).
        canvas_h: Высота холста (>= 1).

    Возвращает:
        Доля покрытия в [0, 1].

    Исключения:
        ValueError: Если canvas_w < 1 или canvas_h < 1.
    """
    if canvas_w < 1:
        raise ValueError(f"canvas_w должен быть >= 1, получено {canvas_w}")
    if canvas_h < 1:
        raise ValueError(f"canvas_h должен быть >= 1, получено {canvas_h}")

    canvas_area = canvas_w * canvas_h
    if canvas_area == 0:
        return 0.0

    mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for b in boxes:
        x1 = max(0, b.x)
        y1 = max(0, b.y)
        x2 = min(canvas_w, b.x2)
        y2 = min(canvas_h, b.y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1

    covered = int(mask.sum())
    return float(covered) / float(canvas_area)


# ─── validate_placements ──────────────────────────────────────────────────────

def validate_placements(
    boxes: List[PlacementBox],
    cfg: Optional[PlacementConfig] = None,
) -> CollisionReport:
    """Выполнить полную валидацию набора размещений.

    Аргументы:
        boxes: Список прямоугольников.
        cfg:   Параметры (None → PlacementConfig()).

    Возвращает:
        CollisionReport.
    """
    if cfg is None:
        cfg = PlacementConfig()

    collisions = find_collisions(boxes, cfg.iou_threshold)
    duplicates = find_duplicate_positions(boxes)

    oob: List[int] = []
    coverage = 0.0
    if cfg.canvas_w > 0 and cfg.canvas_h > 0:
        oob = find_out_of_bounds(boxes, cfg.canvas_w, cfg.canvas_h)
        coverage = compute_coverage(boxes, cfg.canvas_w, cfg.canvas_h)

    return CollisionReport(
        collisions=collisions,
        duplicates=duplicates,
        out_of_bounds=oob,
        coverage=coverage,
        n_checked=len(boxes),
    )


# ─── batch_validate_placements ───────────────────────────────────────────────

def batch_validate_placements(
    box_lists: List[List[PlacementBox]],
    cfg: Optional[PlacementConfig] = None,
) -> List[CollisionReport]:
    """Валидировать несколько наборов размещений.

    Аргументы:
        box_lists: Список наборов прямоугольников.
        cfg:       Параметры.

    Возвращает:
        Список CollisionReport.
    """
    return [validate_placements(boxes, cfg) for boxes in box_lists]
