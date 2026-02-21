"""
Утилиты для работы с ограничивающими прямоугольниками (bounding boxes).

Предоставляет инструменты для вычисления IoU, пересечений, объединений,
расширения, обрезки изображений и слияния перекрывающихся прямоугольников.

Экспортирует:
    BBox               — ограничивающий прямоугольник (x, y, w, h)
    bbox_iou           — Intersection over Union двух прямоугольников
    bbox_intersection  — пересечение двух прямоугольников
    bbox_union         — минимальный охватывающий прямоугольник двух bbox
    expand_bbox        — расширение bbox на заданный отступ
    crop_image         — вырезать область изображения по bbox
    bboxes_from_mask   — найти bbox связных компонент маски
    merge_overlapping_bboxes — слить перекрывающиеся bbox
    bbox_center        — центр bbox
    bbox_aspect_ratio  — соотношение сторон bbox
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class BBox:
    """Ограничивающий прямоугольник в формате (x, y, w, h).

    Attributes:
        x: Левый край (пикс.).
        y: Верхний край (пикс.).
        w: Ширина (пикс., > 0).
        h: Высота (пикс., > 0).
    """
    x: int
    y: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if self.w <= 0:
            raise ValueError(f"BBox.w must be > 0, got {self.w}")
        if self.h <= 0:
            raise ValueError(f"BBox.h must be > 0, got {self.h}")

    @property
    def x2(self) -> int:
        """Правый край (исключительный)."""
        return self.x + self.w

    @property
    def y2(self) -> int:
        """Нижний край (исключительный)."""
        return self.y + self.h

    @property
    def area(self) -> int:
        """Площадь прямоугольника."""
        return self.w * self.h

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Вернуть как кортеж (x, y, w, h)."""
        return (self.x, self.y, self.w, self.h)

    def __repr__(self) -> str:  # pragma: no cover
        return f"BBox(x={self.x}, y={self.y}, w={self.w}, h={self.h})"


# ─── Публичные функции ────────────────────────────────────────────────────────

def bbox_iou(a: BBox, b: BBox) -> float:
    """Вычислить Intersection over Union двух прямоугольников.

    Args:
        a: Первый прямоугольник.
        b: Второй прямоугольник.

    Returns:
        IoU ∈ [0, 1].
    """
    inter = bbox_intersection(a, b)
    if inter is None:
        return 0.0
    inter_area = inter.area
    union_area = a.area + b.area - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area) / float(union_area)


def bbox_intersection(a: BBox, b: BBox) -> Optional[BBox]:
    """Найти пересечение двух прямоугольников.

    Args:
        a: Первый прямоугольник.
        b: Второй прямоугольник.

    Returns:
        :class:`BBox` пересечения, или ``None`` если не пересекаются.
    """
    ix = max(a.x, b.x)
    iy = max(a.y, b.y)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    if ix2 <= ix or iy2 <= iy:
        return None
    return BBox(x=ix, y=iy, w=ix2 - ix, h=iy2 - iy)


def bbox_union(a: BBox, b: BBox) -> BBox:
    """Найти минимальный охватывающий прямоугольник двух bbox.

    Args:
        a: Первый прямоугольник.
        b: Второй прямоугольник.

    Returns:
        Охватывающий :class:`BBox`.
    """
    x = min(a.x, b.x)
    y = min(a.y, b.y)
    x2 = max(a.x2, b.x2)
    y2 = max(a.y2, b.y2)
    return BBox(x=x, y=y, w=x2 - x, h=y2 - y)


def expand_bbox(
    bbox: BBox,
    padding: int,
    max_h: int = 0,
    max_w: int = 0,
) -> BBox:
    """Расширить bbox на заданный отступ с опциональным ограничением.

    Args:
        bbox:    Исходный прямоугольник.
        padding: Отступ (≥ 0) в пикселях с каждой стороны.
        max_h:   Максимальная высота изображения (0 = без ограничения).
        max_w:   Максимальная ширина изображения (0 = без ограничения).

    Returns:
        Расширенный :class:`BBox`.

    Raises:
        ValueError: Если ``padding`` < 0.
    """
    if padding < 0:
        raise ValueError(f"padding must be >= 0, got {padding}")
    x = max(0, bbox.x - padding)
    y = max(0, bbox.y - padding)
    x2 = bbox.x2 + padding
    y2 = bbox.y2 + padding
    if max_w > 0:
        x2 = min(x2, max_w)
    if max_h > 0:
        y2 = min(y2, max_h)
    w = max(1, x2 - x)
    h = max(1, y2 - y)
    return BBox(x=x, y=y, w=w, h=h)


def crop_image(img: np.ndarray, bbox: BBox) -> np.ndarray:
    """Вырезать область изображения по bbox.

    Координаты автоматически зажимаются до границ изображения.

    Args:
        img:  Изображение (H, W) или (H, W, C).
        bbox: Ограничивающий прямоугольник.

    Returns:
        Вырезанная область того же числа каналов.

    Raises:
        ValueError: Если bbox полностью за пределами изображения.
    """
    h, w = img.shape[:2]
    x = max(0, bbox.x)
    y = max(0, bbox.y)
    x2 = min(w, bbox.x2)
    y2 = min(h, bbox.y2)
    if x2 <= x or y2 <= y:
        raise ValueError(
            f"BBox {bbox.to_tuple()} is outside image bounds ({w}, {h})"
        )
    return img[y:y2, x:x2]


def bboxes_from_mask(
    mask: np.ndarray,
    min_area: int = 1,
) -> List[BBox]:
    """Найти bbox всех связных компонент бинарной маски.

    Args:
        mask:     Бинарное изображение uint8 (H, W).
        min_area: Минимальная площадь компоненты для включения (≥ 1).

    Returns:
        Список :class:`BBox` в порядке убывания площади.

    Raises:
        ValueError: Если ``min_area`` < 1.
    """
    if min_area < 1:
        raise ValueError(f"min_area must be >= 1, got {min_area}")
    binary = (mask > 0).astype(np.uint8) * 255
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    bboxes: List[BBox] = []
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])
        bboxes.append(BBox(x=x, y=y, w=w, h=h))
    bboxes.sort(key=lambda b: b.area, reverse=True)
    return bboxes


def merge_overlapping_bboxes(
    bboxes: List[BBox],
    min_iou: float = 0.0,
) -> List[BBox]:
    """Слить перекрывающиеся bbox жадным алгоритмом.

    Два bbox сливаются в охватывающий, если их IoU ≥ ``min_iou``.
    Итерация продолжается до стабилизации.

    Args:
        bboxes:  Список прямоугольников.
        min_iou: Минимальный IoU для слияния (0 = слить при любом перекрытии).

    Returns:
        Список слитых прямоугольников.

    Raises:
        ValueError: Если ``min_iou`` < 0.
    """
    if min_iou < 0:
        raise ValueError(f"min_iou must be >= 0, got {min_iou}")
    result = list(bboxes)
    changed = True
    while changed:
        changed = False
        merged: List[BBox] = []
        used = [False] * len(result)
        for i in range(len(result)):
            if used[i]:
                continue
            current = result[i]
            for j in range(i + 1, len(result)):
                if used[j]:
                    continue
                iou = bbox_iou(current, result[j])
                if iou > min_iou or bbox_intersection(current, result[j]) is not None and min_iou == 0.0:
                    current = bbox_union(current, result[j])
                    used[j] = True
                    changed = True
            merged.append(current)
            used[i] = True
        result = merged
    return result


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    """Вернуть координаты центра прямоугольника.

    Args:
        bbox: Прямоугольник.

    Returns:
        Кортеж (cx, cy).
    """
    return (bbox.x + bbox.w / 2.0, bbox.y + bbox.h / 2.0)


def bbox_aspect_ratio(bbox: BBox) -> float:
    """Вычислить соотношение сторон bbox.

    Returns:
        min(w, h) / max(w, h) ∈ (0, 1].
    """
    return float(min(bbox.w, bbox.h)) / float(max(bbox.w, bbox.h))
