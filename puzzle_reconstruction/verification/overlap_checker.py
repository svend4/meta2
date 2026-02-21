"""
Проверка геометрического перекрытия фрагментов в сборке.

Обнаруживает пары фрагментов, полигоны (контуры) которых перекрываются.
Используется для валидации физической корректности реконструкции:
фрагменты документа не должны занимать одно и то же пространство.

Классы:
    OverlapResult — результат проверки одной пары фрагментов

Функции:
    polygon_intersection_area — площадь пересечения двух полигонов
    polygon_union_area        — площадь объединения двух полигонов
    polygon_iou               — IoU ∈ [0,1] для двух полигонов
    check_overlap_pair        — проверка одной пары (idx1, idx2)
    check_all_overlaps        — полная матрица пар (N·(N-1)/2 результатов)
    find_conflicting_pairs    — фильтрация пар с IoU > порога
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ─── OverlapResult ────────────────────────────────────────────────────────────

@dataclass
class OverlapResult:
    """
    Результат проверки перекрытия одной пары фрагментов.

    Attributes:
        idx1:               Индекс первого фрагмента.
        idx2:               Индекс второго фрагмента.
        intersection_area:  Площадь пересечения полигонов (пикс²).
        iou:                IoU = intersection / union ∈ [0,1].
        has_overlap:        True, если iou > iou_thresh.
        method:             Всегда 'polygon'.
        params:             Используемые пороги.
    """
    idx1:               int
    idx2:               int
    intersection_area:  float
    iou:                float
    has_overlap:        bool
    method:             str  = "polygon"
    params:             Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"OverlapResult(({self.idx1},{self.idx2}), "
                f"iou={self.iou:.4f}, "
                f"overlap={self.has_overlap})")


# ─── Вспомогательные ──────────────────────────────────────────────────────────

def _to_contour(poly: np.ndarray) -> np.ndarray:
    """
    Преобразует массив точек в формат контура OpenCV.

    Принимает массив формы (N,2) или (N,1,2) и возвращает (N,1,2) int32.
    """
    pts = np.asarray(poly, dtype=np.float32)
    if pts.ndim == 2:
        pts = pts.reshape(-1, 1, 2)
    return pts.astype(np.int32)


def _poly_area(poly: np.ndarray) -> float:
    """Площадь полигона через cv2.contourArea."""
    cnt = _to_contour(poly)
    return float(cv2.contourArea(cnt))


def _rasterize(poly: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    """Закрашивает полигон на чёрном холсте canvas, возвращает маску."""
    mask = np.zeros_like(canvas, dtype=np.uint8)
    cnt  = _to_contour(poly)
    cv2.fillPoly(mask, [cnt], 255)
    return mask


def _bounding_canvas(poly1: np.ndarray, poly2: np.ndarray) -> np.ndarray:
    """
    Создаёт минимальный общий холст для двух полигонов.
    Возвращает нулевую матрицу нужного размера и смещение (ox, oy).
    """
    pts = np.concatenate([
        np.asarray(poly1, dtype=np.float32).reshape(-1, 2),
        np.asarray(poly2, dtype=np.float32).reshape(-1, 2),
    ], axis=0)
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)
    w = max(int(np.ceil(max_x - min_x)) + 2, 1)
    h = max(int(np.ceil(max_y - min_y)) + 2, 1)
    return np.zeros((h, w), dtype=np.uint8), int(min_x), int(min_y)


# ─── polygon_intersection_area ────────────────────────────────────────────────

def polygon_intersection_area(poly1: np.ndarray,
                               poly2: np.ndarray) -> float:
    """
    Вычисляет площадь пересечения двух полигонов.

    Реализация через растеризацию маски и побитовое И.
    Работает для произвольных (в том числе невыпуклых) полигонов.

    Args:
        poly1: Массив вершин первого полигона формы (N,2) или (N,1,2).
        poly2: Массив вершин второго полигона.

    Returns:
        Площадь пересечения в пикселях² (float ≥ 0).
    """
    canvas, ox, oy = _bounding_canvas(poly1, poly2)

    shifted1 = np.asarray(poly1, dtype=np.float32).reshape(-1, 2) - [ox, oy]
    shifted2 = np.asarray(poly2, dtype=np.float32).reshape(-1, 2) - [ox, oy]

    mask1 = _rasterize(shifted1, canvas)
    mask2 = _rasterize(shifted2, canvas)

    inter = cv2.bitwise_and(mask1, mask2)
    return float(np.count_nonzero(inter))


# ─── polygon_union_area ───────────────────────────────────────────────────────

def polygon_union_area(poly1: np.ndarray,
                        poly2: np.ndarray) -> float:
    """
    Вычисляет площадь объединения двух полигонов.

    Args:
        poly1: Массив вершин первого полигона.
        poly2: Массив вершин второго полигона.

    Returns:
        Площадь объединения в пикселях² (float ≥ 0).
    """
    canvas, ox, oy = _bounding_canvas(poly1, poly2)

    shifted1 = np.asarray(poly1, dtype=np.float32).reshape(-1, 2) - [ox, oy]
    shifted2 = np.asarray(poly2, dtype=np.float32).reshape(-1, 2) - [ox, oy]

    mask1 = _rasterize(shifted1, canvas)
    mask2 = _rasterize(shifted2, canvas)

    union = cv2.bitwise_or(mask1, mask2)
    return float(np.count_nonzero(union))


# ─── polygon_iou ──────────────────────────────────────────────────────────────

def polygon_iou(poly1: np.ndarray,
                 poly2: np.ndarray) -> float:
    """
    Intersection-over-Union двух полигонов ∈ [0,1].

    IoU = intersection_area / union_area.
    Если оба полигона вырождены (union = 0), возвращает 0.0.

    Args:
        poly1: Первый полигон.
        poly2: Второй полигон.

    Returns:
        IoU ∈ [0,1].
    """
    inter = polygon_intersection_area(poly1, poly2)
    if inter < 1e-9:
        return 0.0
    union = polygon_union_area(poly1, poly2)
    if union < 1e-9:
        return 0.0
    return float(np.clip(inter / union, 0.0, 1.0))


# ─── check_overlap_pair ───────────────────────────────────────────────────────

def check_overlap_pair(poly1:      np.ndarray,
                        poly2:      np.ndarray,
                        idx1:       int   = 0,
                        idx2:       int   = 1,
                        iou_thresh: float = 0.05) -> OverlapResult:
    """
    Проверяет перекрытие одной пары полигонов.

    Args:
        poly1:      Первый полигон.
        poly2:      Второй полигон.
        idx1:       Индекс первого фрагмента.
        idx2:       Индекс второго фрагмента.
        iou_thresh: Порог IoU для флага has_overlap.

    Returns:
        OverlapResult с intersection_area, iou, has_overlap.
    """
    inter = polygon_intersection_area(poly1, poly2)
    iou   = polygon_iou(poly1, poly2)
    return OverlapResult(
        idx1=idx1, idx2=idx2,
        intersection_area=inter,
        iou=iou,
        has_overlap=(iou > iou_thresh),
        method="polygon",
        params={"iou_thresh": iou_thresh},
    )


# ─── check_all_overlaps ───────────────────────────────────────────────────────

def check_all_overlaps(polygons:   List[np.ndarray],
                        iou_thresh: float = 0.05) -> List[OverlapResult]:
    """
    Проверяет все пары фрагментов на перекрытие (N·(N-1)/2 пар).

    Args:
        polygons:   Список полигонов (каждый — массив (N,2)).
        iou_thresh: Порог IoU для флага has_overlap.

    Returns:
        Список OverlapResult (по одному на пару i < j).
    """
    n       = len(polygons)
    results = []
    for i in range(n):
        for j in range(i + 1, n):
            results.append(
                check_overlap_pair(polygons[i], polygons[j],
                                   idx1=i, idx2=j,
                                   iou_thresh=iou_thresh)
            )
    return results


# ─── find_conflicting_pairs ───────────────────────────────────────────────────

def find_conflicting_pairs(results:    List[OverlapResult],
                            iou_thresh: Optional[float] = None) -> List[OverlapResult]:
    """
    Возвращает только пары с has_overlap=True (или iou > iou_thresh).

    Args:
        results:    Список OverlapResult (обычно из check_all_overlaps).
        iou_thresh: Переопределяет порог; None → использует has_overlap.

    Returns:
        Отфильтрованный список конфликтующих OverlapResult.
    """
    if iou_thresh is None:
        return [r for r in results if r.has_overlap]
    return [r for r in results if r.iou > iou_thresh]
