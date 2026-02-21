"""Разделение изображений на связные регионы.

Модуль предоставляет функции для сегментации изображения на связные
компоненты: анализ связных компонент (cv2.connectedComponentsWithStats),
фильтрация по площади, разбивка маски на суб-маски, слияние небольших
регионов с соседями, а также пакетная обработка.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─── RegionInfo ───────────────────────────────────────────────────────────────

@dataclass
class RegionInfo:
    """Информация об одном связном регионе.

    Атрибуты:
        label:     Метка региона (>= 1, 0 зарезервирован для фона).
        area:      Площадь в пикселях (>= 0).
        bbox:      Ограничивающий прямоугольник (x, y, w, h).
        centroid:  Центр масс (cx, cy).
        mask:      Бинарная маска региона (uint8, 0/255).
        params:    Дополнительные параметры.
    """

    label: int
    area: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    mask: np.ndarray
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label < 0:
            raise ValueError(f"label должен быть >= 0, получено {self.label}")
        if self.area < 0:
            raise ValueError(f"area должна быть >= 0, получено {self.area}")

    def __len__(self) -> int:
        return self.area


# ─── SplitResult ──────────────────────────────────────────────────────────────

@dataclass
class SplitResult:
    """Результат разделения маски на регионы.

    Атрибуты:
        regions:    Список RegionInfo для каждого региона.
        label_map:  Карта меток (int32), форма совпадает с входной маской.
        n_regions:  Количество найденных регионов (без фона).
        params:     Дополнительные параметры.
    """

    regions: List[RegionInfo]
    label_map: np.ndarray
    n_regions: int
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_regions < 0:
            raise ValueError(
                f"n_regions должен быть >= 0, получено {self.n_regions}"
            )

    def __len__(self) -> int:
        return self.n_regions


# ─── find_regions ─────────────────────────────────────────────────────────────

def find_regions(mask: np.ndarray) -> SplitResult:
    """Найти все связные регионы в бинарной маске.

    Аргументы:
        mask: Бинарная маска (uint8 или bool, 2-D; ненулевые значения = объект).

    Возвращает:
        SplitResult с информацией о каждом регионе.

    Исключения:
        ValueError: Если mask не является 2-D массивом.
    """
    if mask.ndim != 2:
        raise ValueError(
            f"mask должна быть 2-D, получено ndim={mask.ndim}"
        )
    bin_mask = (np.asarray(mask) > 0).astype(np.uint8)

    n_labels, label_map, stats, centroids = cv2.connectedComponentsWithStats(
        bin_mask, connectivity=8
    )
    # label 0 = фон
    regions: List[RegionInfo] = []
    for lbl in range(1, n_labels):
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        cx = float(centroids[lbl, 0])
        cy = float(centroids[lbl, 1])
        region_mask = np.where(label_map == lbl, np.uint8(255), np.uint8(0))
        regions.append(
            RegionInfo(
                label=lbl,
                area=area,
                bbox=(x, y, w, h),
                centroid=(cx, cy),
                mask=region_mask,
            )
        )

    return SplitResult(
        regions=regions,
        label_map=label_map.astype(np.int32),
        n_regions=n_labels - 1,
    )


# ─── filter_regions ───────────────────────────────────────────────────────────

def filter_regions(
    split: SplitResult,
    min_area: int = 0,
    max_area: Optional[int] = None,
) -> SplitResult:
    """Оставить только регионы с площадью в заданном диапазоне.

    Аргументы:
        split:    Результат find_regions.
        min_area: Минимальная площадь включительно (>= 0).
        max_area: Максимальная площадь включительно (None — без ограничения).

    Возвращает:
        Новый SplitResult только с подходящими регионами.

    Исключения:
        ValueError: Если min_area < 0 или max_area < min_area.
    """
    if min_area < 0:
        raise ValueError(f"min_area должна быть >= 0, получено {min_area}")
    if max_area is not None and max_area < min_area:
        raise ValueError(
            f"max_area ({max_area}) должна быть >= min_area ({min_area})"
        )

    kept = [
        r for r in split.regions
        if r.area >= min_area and (max_area is None or r.area <= max_area)
    ]

    # Пересобираем label_map
    new_label_map = np.zeros_like(split.label_map)
    for new_lbl, r in enumerate(kept, start=1):
        new_label_map[split.label_map == r.label] = new_lbl
        r = RegionInfo(
            label=new_lbl, area=r.area, bbox=r.bbox,
            centroid=r.centroid, mask=r.mask, params=r.params,
        )

    # Повторно строим список с обновлёнными метками
    updated = []
    for new_lbl, r in enumerate(kept, start=1):
        updated.append(
            RegionInfo(
                label=new_lbl, area=r.area, bbox=r.bbox,
                centroid=r.centroid, mask=r.mask, params=r.params,
            )
        )

    return SplitResult(
        regions=updated,
        label_map=new_label_map,
        n_regions=len(updated),
    )


# ─── region_masks ─────────────────────────────────────────────────────────────

def region_masks(split: SplitResult) -> List[np.ndarray]:
    """Извлечь маски всех регионов как список массивов.

    Аргументы:
        split: Результат find_regions.

    Возвращает:
        Список 2-D массивов uint8 (255 = регион, 0 = фон).
    """
    return [r.mask for r in split.regions]


# ─── merge_small_regions ─────────────────────────────────────────────────────

def merge_small_regions(
    split: SplitResult,
    min_area: int,
) -> SplitResult:
    """Удалить регионы меньше min_area (слить их с фоном).

    Аргументы:
        split:    Результат find_regions.
        min_area: Минимальная площадь для сохранения региона (>= 1).

    Возвращает:
        Новый SplitResult без маленьких регионов.

    Исключения:
        ValueError: Если min_area < 1.
    """
    if min_area < 1:
        raise ValueError(f"min_area должна быть >= 1, получено {min_area}")
    return filter_regions(split, min_area=min_area)


# ─── largest_region ───────────────────────────────────────────────────────────

def largest_region(split: SplitResult) -> Optional[RegionInfo]:
    """Вернуть самый большой регион по площади.

    Аргументы:
        split: Результат find_regions.

    Возвращает:
        RegionInfo самого большого региона или None, если регионов нет.
    """
    if not split.regions:
        return None
    return max(split.regions, key=lambda r: r.area)


# ─── split_mask_to_crops ──────────────────────────────────────────────────────

def split_mask_to_crops(
    image: np.ndarray, split: SplitResult
) -> List[np.ndarray]:
    """Вырезать области изображения, соответствующие каждому региону.

    Аргументы:
        image: Исходное изображение (2-D или 3-D, uint8).
        split: Результат find_regions.

    Возвращает:
        Список кропов изображения по bounding box каждого региона.

    Исключения:
        ValueError: Если image не является 2-D или 3-D массивом.
    """
    if image.ndim not in (2, 3):
        raise ValueError(
            f"image должно быть 2-D или 3-D, получено ndim={image.ndim}"
        )
    crops = []
    for r in split.regions:
        x, y, w, h = r.bbox
        if image.ndim == 2:
            crops.append(image[y : y + h, x : x + w].copy())
        else:
            crops.append(image[y : y + h, x : x + w, :].copy())
    return crops


# ─── batch_find_regions ───────────────────────────────────────────────────────

def batch_find_regions(masks: List[np.ndarray]) -> List[SplitResult]:
    """Найти регионы в каждой маске из списка.

    Аргументы:
        masks: Список бинарных масок (uint8, 2-D).

    Возвращает:
        Список SplitResult по одному на каждую маску.
    """
    return [find_regions(m) for m in masks]
