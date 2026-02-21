"""Проверка отсутствия перекрытий в сборке фрагментов.

Модуль предоставляет функции для обнаружения и количественной оценки
перекрытий между размещёнными фрагментами: попарная проверка bounding box,
вычисление IoU масок, суммарная площадь перекрытий,
статистика по всей сборке и пакетная обработка.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ─── OverlapRecord ────────────────────────────────────────────────────────────

@dataclass
class OverlapRecord:
    """Запись о перекрытии двух фрагментов.

    Атрибуты:
        idx1:         Индекс первого фрагмента (>= 0).
        idx2:         Индекс второго фрагмента (>= 0).
        overlap_area: Площадь перекрытия в пикселях (>= 0).
        iou:          Intersection over Union ∈ [0, 1].
        params:       Дополнительные параметры.
    """

    idx1: int
    idx2: int
    overlap_area: float
    iou: float
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.idx1 < 0:
            raise ValueError(f"idx1 должен быть >= 0, получено {self.idx1}")
        if self.idx2 < 0:
            raise ValueError(f"idx2 должен быть >= 0, получено {self.idx2}")
        if self.overlap_area < 0.0:
            raise ValueError(
                f"overlap_area должна быть >= 0, получено {self.overlap_area}"
            )
        if not (0.0 <= self.iou <= 1.0):
            raise ValueError(
                f"iou должен быть в [0, 1], получено {self.iou}"
            )

    @property
    def pair(self) -> Tuple[int, int]:
        return (self.idx1, self.idx2)


# ─── ValidationReport ─────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    """Отчёт о валидации перекрытий всей сборки.

    Атрибуты:
        overlaps:        Список обнаруженных перекрытий.
        n_overlaps:      Количество перекрытий (>= 0).
        total_area:      Суммарная площадь перекрытий (>= 0).
        max_iou:         Максимальный IoU среди всех пар.
        is_valid:        True, если перекрытий нет.
        params:          Дополнительные параметры.
    """

    overlaps: List[OverlapRecord]
    n_overlaps: int
    total_area: float
    max_iou: float
    is_valid: bool
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_overlaps < 0:
            raise ValueError(
                f"n_overlaps должен быть >= 0, получено {self.n_overlaps}"
            )
        if self.total_area < 0.0:
            raise ValueError(
                f"total_area должна быть >= 0, получено {self.total_area}"
            )

    def __len__(self) -> int:
        return self.n_overlaps


# ─── bbox_overlap ─────────────────────────────────────────────────────────────

def bbox_overlap(
    bbox1: Tuple[int, int, int, int],
    bbox2: Tuple[int, int, int, int],
) -> float:
    """Вычислить площадь пересечения двух bounding box.

    Аргументы:
        bbox1: (x, y, w, h) первого прямоугольника.
        bbox2: (x, y, w, h) второго прямоугольника.

    Возвращает:
        Площадь пересечения (float, >= 0).

    Исключения:
        ValueError: Если ширина или высота отрицательны.
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    if w1 < 0 or h1 < 0:
        raise ValueError(f"bbox1 имеет отрицательные размеры: w={w1}, h={h1}")
    if w2 < 0 or h2 < 0:
        raise ValueError(f"bbox2 имеет отрицательные размеры: w={w2}, h={h2}")

    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    return float(iw * ih)


# ─── mask_iou ─────────────────────────────────────────────────────────────────

def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Вычислить IoU (Intersection over Union) двух бинарных масок.

    Аргументы:
        mask1: Бинарная маска (bool или uint8, 2-D).
        mask2: Бинарная маска (bool или uint8, 2-D), та же форма.

    Возвращает:
        IoU ∈ [0, 1].

    Исключения:
        ValueError: Если маски имеют разную форму или не 2-D.
    """
    if mask1.ndim != 2:
        raise ValueError(f"mask1 должна быть 2-D, получено ndim={mask1.ndim}")
    if mask2.ndim != 2:
        raise ValueError(f"mask2 должна быть 2-D, получено ndim={mask2.ndim}")
    if mask1.shape != mask2.shape:
        raise ValueError(
            f"Формы масок не совпадают: {mask1.shape} != {mask2.shape}"
        )

    b1 = (mask1 > 0).astype(bool)
    b2 = (mask2 > 0).astype(bool)
    intersection = float((b1 & b2).sum())
    union = float((b1 | b2).sum())
    if union < 1e-9:
        return 0.0
    return intersection / union


# ─── check_pair_overlap ───────────────────────────────────────────────────────

def check_pair_overlap(
    idx1: int,
    idx2: int,
    mask1: np.ndarray,
    mask2: np.ndarray,
    canvas_size: Tuple[int, int],
    pos1: Tuple[int, int] = (0, 0),
    pos2: Tuple[int, int] = (0, 0),
) -> OverlapRecord:
    """Проверить перекрытие двух фрагментов на общем холсте.

    Аргументы:
        idx1:        Индекс первого фрагмента.
        idx2:        Индекс второго фрагмента.
        mask1:       Бинарная маска фрагмента 1 (uint8, 2-D).
        mask2:       Бинарная маска фрагмента 2 (uint8, 2-D).
        canvas_size: (height, width) холста.
        pos1:        (y, x) позиция верхнего левого угла фрагмента 1.
        pos2:        (y, x) позиция верхнего левого угла фрагмента 2.

    Возвращает:
        OverlapRecord с результатами проверки.

    Исключения:
        ValueError: Если маски не 2-D или индексы отрицательны.
    """
    if idx1 < 0:
        raise ValueError(f"idx1 должен быть >= 0, получено {idx1}")
    if idx2 < 0:
        raise ValueError(f"idx2 должен быть >= 0, получено {idx2}")
    if mask1.ndim != 2:
        raise ValueError(f"mask1 должна быть 2-D, получено ndim={mask1.ndim}")
    if mask2.ndim != 2:
        raise ValueError(f"mask2 должна быть 2-D, получено ndim={mask2.ndim}")

    H, W = canvas_size
    canvas1 = np.zeros((H, W), dtype=np.uint8)
    canvas2 = np.zeros((H, W), dtype=np.uint8)

    y1, x1 = pos1
    h1, w1 = mask1.shape
    y2, x2 = pos2
    h2, w2 = mask2.shape

    # Клиппинг к холсту
    def _paste(canvas, mask, y, x):
        my0 = max(0, -y)
        mx0 = max(0, -x)
        cy0 = max(0, y)
        cx0 = max(0, x)
        my1 = mask.shape[0] - max(0, (y + mask.shape[0]) - H)
        mx1 = mask.shape[1] - max(0, (x + mask.shape[1]) - W)
        cy1 = cy0 + (my1 - my0)
        cx1 = cx0 + (mx1 - mx0)
        if my1 > my0 and mx1 > mx0 and cy1 > cy0 and cx1 > cx0:
            canvas[cy0:cy1, cx0:cx1] = mask[my0:my1, mx0:mx1]

    _paste(canvas1, mask1, y1, x1)
    _paste(canvas2, mask2, y2, x2)

    iou = mask_iou(canvas1, canvas2)
    b1 = (canvas1 > 0).astype(bool)
    b2 = (canvas2 > 0).astype(bool)
    area = float((b1 & b2).sum())

    return OverlapRecord(idx1=idx1, idx2=idx2, overlap_area=area, iou=iou)


# ─── validate_assembly ────────────────────────────────────────────────────────

def validate_assembly(
    masks: List[np.ndarray],
    positions: List[Tuple[int, int]],
    canvas_size: Tuple[int, int],
    iou_threshold: float = 0.0,
) -> ValidationReport:
    """Проверить все пары фрагментов на перекрытие.

    Аргументы:
        masks:         Список бинарных масок (uint8, 2-D).
        positions:     Список позиций (y, x) для каждой маски.
        canvas_size:   (height, width) холста.
        iou_threshold: Минимальный IoU для регистрации перекрытия (>= 0).

    Возвращает:
        ValidationReport со списком обнаруженных перекрытий.

    Исключения:
        ValueError: Если длины masks и positions не совпадают или
                    iou_threshold < 0.
    """
    if len(masks) != len(positions):
        raise ValueError(
            f"Длина masks ({len(masks)}) != длина positions ({len(positions)})"
        )
    if iou_threshold < 0.0:
        raise ValueError(
            f"iou_threshold должен быть >= 0, получено {iou_threshold}"
        )

    overlaps: List[OverlapRecord] = []
    n = len(masks)
    for i in range(n):
        for j in range(i + 1, n):
            rec = check_pair_overlap(
                idx1=i, idx2=j,
                mask1=masks[i], mask2=masks[j],
                canvas_size=canvas_size,
                pos1=positions[i], pos2=positions[j],
            )
            if rec.iou > iou_threshold:
                overlaps.append(rec)

    total_area = sum(r.overlap_area for r in overlaps)
    max_iou = max((r.iou for r in overlaps), default=0.0)
    return ValidationReport(
        overlaps=overlaps,
        n_overlaps=len(overlaps),
        total_area=total_area,
        max_iou=max_iou,
        is_valid=len(overlaps) == 0,
    )


# ─── overlap_area_matrix ──────────────────────────────────────────────────────

def overlap_area_matrix(
    masks: List[np.ndarray],
    positions: List[Tuple[int, int]],
    canvas_size: Tuple[int, int],
) -> np.ndarray:
    """Построить симметричную матрицу площадей перекрытий.

    Аргументы:
        masks:       Список бинарных масок.
        positions:   Список позиций (y, x).
        canvas_size: (height, width) холста.

    Возвращает:
        Матрица (N, N) float64; element [i, j] = площадь пересечения масок i и j.
    """
    n = len(masks)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            rec = check_pair_overlap(
                idx1=i, idx2=j,
                mask1=masks[i], mask2=masks[j],
                canvas_size=canvas_size,
                pos1=positions[i], pos2=positions[j],
            )
            matrix[i, j] = rec.overlap_area
            matrix[j, i] = rec.overlap_area
    return matrix


# ─── batch_validate ───────────────────────────────────────────────────────────

def batch_validate(
    assemblies: List[Tuple[List[np.ndarray], List[Tuple[int, int]], Tuple[int, int]]],
    iou_threshold: float = 0.0,
) -> List[ValidationReport]:
    """Пакетная валидация нескольких сборок.

    Аргументы:
        assemblies: Список кортежей (masks, positions, canvas_size).
        iou_threshold: Порог IoU для регистрации перекрытия.

    Возвращает:
        Список ValidationReport, по одному на каждую сборку.
    """
    return [
        validate_assembly(masks, positions, canvas_size, iou_threshold)
        for masks, positions, canvas_size in assemblies
    ]
