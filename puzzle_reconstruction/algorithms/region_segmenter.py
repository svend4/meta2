"""
Сегментация изображения на связные регионы.

Предоставляет инструменты для нахождения, анализа и фильтрации
связных компонент в бинарных и серых изображениях фрагментов документа.

Экспортирует:
    RegionProps         — свойства одного региона
    SegmentationResult  — результат сегментации изображения
    label_connected     — разметка связных компонент
    compute_region_props — вычисление свойств регионов
    filter_regions      — фильтрация регионов по критериям
    merge_close_regions — слияние близких регионов
    region_adjacency    — граф смежности регионов
    largest_region      — найти наибольший регион
    regions_to_mask     — перевод набора регионов в бинарную маску
    batch_segment       — пакетная сегментация
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class RegionProps:
    """Свойства одного региона (связной компоненты).

    Attributes:
        label:      Метка компоненты (≥ 1).
        area:       Площадь (число пикселей).
        bbox:       Ограничивающий прямоугольник (x, y, w, h).
        centroid:   Центроид (cx, cy) в пикселях.
        aspect_ratio: min(w,h)/max(w,h) ∈ (0, 1].
        solidity:   area / (ширина×высота bbox) — плотность региона.
        perimeter:  Периметр контура (пиксели).
    """
    label: int
    area: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    centroid: Tuple[float, float]
    aspect_ratio: float
    solidity: float
    perimeter: float

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RegionProps(label={self.label}, area={self.area}, "
            f"centroid=({self.centroid[0]:.1f},{self.centroid[1]:.1f}))"
        )


@dataclass
class SegmentationResult:
    """Результат сегментации изображения.

    Attributes:
        labels:   Матрица меток (H, W) int32; 0 — фон.
        n_labels: Число найденных регионов (без фона).
        props:    Список свойств каждого региона.
        params:   Параметры сегментации.
    """
    labels: np.ndarray
    n_labels: int
    props: List[RegionProps] = field(default_factory=list)
    params: Dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return f"SegmentationResult(n_labels={self.n_labels}, shape={self.labels.shape})"


# ─── Публичные функции ────────────────────────────────────────────────────────

def label_connected(
    img: np.ndarray,
    connectivity: int = 8,
    threshold: int = 128,
) -> SegmentationResult:
    """Размечать связные компоненты бинарного/серого изображения.

    Изображение предварительно бинаризуется по порогу ``threshold``.

    Args:
        img:          Изображение uint8 (2D или BGR; BGR → grey автоматически).
        connectivity: 4 или 8.
        threshold:    Порог бинаризации [0, 255].

    Returns:
        :class:`SegmentationResult`.

    Raises:
        ValueError: Если ``connectivity`` не 4 или 8,
                    или ``threshold`` вне [0, 255].
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    if not (0 <= threshold <= 255):
        raise ValueError(f"threshold must be in [0, 255], got {threshold}")

    gray = _to_gray(img)
    binary = np.where(gray >= threshold, np.uint8(255), np.uint8(0))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=connectivity
    )
    n_regions = n - 1  # exclude background (label 0)
    props = _build_props(labels, stats, centroids, n_regions, binary)
    return SegmentationResult(
        labels=labels.astype(np.int32),
        n_labels=n_regions,
        props=props,
        params={"connectivity": connectivity, "threshold": threshold},
    )


def compute_region_props(result: SegmentationResult) -> List[RegionProps]:
    """Вернуть список свойств регионов из :class:`SegmentationResult`.

    Это удобная обёртка, позволяющая обновить свойства после фильтрации.

    Args:
        result: Результат сегментации.

    Returns:
        Список :class:`RegionProps`.
    """
    return list(result.props)


def filter_regions(
    result: SegmentationResult,
    min_area: int = 0,
    max_area: int = 0,
    min_aspect_ratio: float = 0.0,
    min_solidity: float = 0.0,
) -> SegmentationResult:
    """Отфильтровать регионы по заданным критериям.

    Args:
        result:            Входной результат сегментации.
        min_area:          Минимальная площадь (0 = без ограничений).
        max_area:          Максимальная площадь (0 = без ограничений).
        min_aspect_ratio:  Минимальное соотношение сторон [0, 1].
        min_solidity:      Минимальная плотность [0, 1].

    Returns:
        Новый :class:`SegmentationResult` с отфильтрованными регионами.
        Матрица ``labels`` обновляется (отфильтрованные регионы → 0).

    Raises:
        ValueError: Если ограничения нарушены (min > max).
    """
    if max_area > 0 and min_area > max_area:
        raise ValueError(
            f"min_area ({min_area}) > max_area ({max_area})"
        )
    kept = []
    for p in result.props:
        if min_area > 0 and p.area < min_area:
            continue
        if max_area > 0 and p.area > max_area:
            continue
        if p.aspect_ratio < min_aspect_ratio:
            continue
        if p.solidity < min_solidity:
            continue
        kept.append(p)

    kept_labels: Set[int] = {p.label for p in kept}
    new_labels = result.labels.copy()
    for lab in range(1, result.n_labels + 1):
        if lab not in kept_labels:
            new_labels[new_labels == lab] = 0

    return SegmentationResult(
        labels=new_labels,
        n_labels=len(kept),
        props=kept,
        params=dict(result.params),
    )


def merge_close_regions(
    result: SegmentationResult,
    max_distance: float = 10.0,
) -> SegmentationResult:
    """Слить регионы, центроиды которых ближе чем ``max_distance``.

    Алгоритм: жадное слияние — если два центроида ближе порога,
    меньший регион получает метку большего.

    Args:
        result:       Результат сегментации.
        max_distance: Максимальное расстояние между центроидами (пикс.).

    Returns:
        Новый :class:`SegmentationResult` после слияния.

    Raises:
        ValueError: Если ``max_distance`` < 0.
    """
    if max_distance < 0:
        raise ValueError(f"max_distance must be >= 0, got {max_distance}")
    if len(result.props) <= 1:
        return result

    props = list(result.props)
    label_map: Dict[int, int] = {p.label: p.label for p in props}

    for i in range(len(props)):
        for j in range(i + 1, len(props)):
            cx1, cy1 = props[i].centroid
            cx2, cy2 = props[j].centroid
            dist = float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))
            if dist <= max_distance:
                # Merge j into i (i keeps its label)
                old_label = label_map[props[j].label]
                new_label = label_map[props[i].label]
                for k in label_map:
                    if label_map[k] == old_label:
                        label_map[k] = new_label

    new_labels = result.labels.copy()
    for old, new in label_map.items():
        if old != new:
            new_labels[result.labels == old] = new

    # Rebuild props
    unique_labels = sorted(set(label_map.values()))
    new_props = []
    for lab in unique_labels:
        mask = (new_labels == lab).astype(np.uint8) * 255
        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if n < 2:
            continue
        p = _stats_to_props(lab, stats[1], centroids[1], mask)
        new_props.append(p)

    return SegmentationResult(
        labels=new_labels.astype(np.int32),
        n_labels=len(new_props),
        props=new_props,
        params=dict(result.params),
    )


def region_adjacency(
    result: SegmentationResult,
    dilation_ksize: int = 3,
) -> Dict[int, List[int]]:
    """Построить граф смежности регионов.

    Два региона смежны, если их расширенные маски перекрываются.

    Args:
        result:         Результат сегментации.
        dilation_ksize: Размер ядра расширения (нечётное, ≥ 1).

    Returns:
        Словарь {label: [соседние labels]}.

    Raises:
        ValueError: Если ``dilation_ksize`` < 1.
    """
    if dilation_ksize < 1:
        raise ValueError(f"dilation_ksize must be >= 1, got {dilation_ksize}")
    dilation_ksize = dilation_ksize | 1  # ensure odd

    labels_map = result.labels
    adjacency: Dict[int, List[int]] = {p.label: [] for p in result.props}

    label_ids = [p.label for p in result.props]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_ksize, dilation_ksize))

    masks = {}
    for lab in label_ids:
        m = (labels_map == lab).astype(np.uint8)
        masks[lab] = cv2.dilate(m, k)

    for i, li in enumerate(label_ids):
        for lj in label_ids[i + 1:]:
            if np.any(masks[li] & masks[lj]):
                adjacency[li].append(lj)
                adjacency[lj].append(li)

    return adjacency


def largest_region(result: SegmentationResult) -> Optional[RegionProps]:
    """Найти регион с наибольшей площадью.

    Args:
        result: Результат сегментации.

    Returns:
        :class:`RegionProps` наибольшего региона, или ``None`` для пустого.
    """
    if not result.props:
        return None
    return max(result.props, key=lambda p: p.area)


def regions_to_mask(
    result: SegmentationResult,
    labels: Optional[List[int]] = None,
) -> np.ndarray:
    """Перевести набор регионов в бинарную маску.

    Args:
        result: Результат сегментации.
        labels: Список меток для включения (``None`` — все регионы).

    Returns:
        Бинарная маска (H, W) uint8 (255 — регион, 0 — фон).
    """
    if labels is None:
        labels = [p.label for p in result.props]
    label_set = set(labels)
    mask = np.zeros(result.labels.shape, dtype=np.uint8)
    for lab in label_set:
        mask[result.labels == lab] = 255
    return mask


def batch_segment(
    images: List[np.ndarray],
    connectivity: int = 8,
    threshold: int = 128,
) -> List[SegmentationResult]:
    """Пакетная сегментация списка изображений.

    Args:
        images:       Список изображений uint8.
        connectivity: 4 или 8.
        threshold:    Порог бинаризации.

    Returns:
        Список :class:`SegmentationResult` той же длины.
    """
    return [
        label_connected(img, connectivity=connectivity, threshold=threshold)
        for img in images
    ]


# ─── Приватные ───────────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _stats_to_props(
    label: int,
    stats: np.ndarray,
    centroid: np.ndarray,
    binary: np.ndarray,
) -> RegionProps:
    x = int(stats[cv2.CC_STAT_LEFT])
    y = int(stats[cv2.CC_STAT_TOP])
    w = int(stats[cv2.CC_STAT_WIDTH])
    h = int(stats[cv2.CC_STAT_HEIGHT])
    area = int(stats[cv2.CC_STAT_AREA])
    cx = float(centroid[0])
    cy = float(centroid[1])
    bbox_area = max(w * h, 1)
    ar = float(min(w, h)) / float(max(w, h)) if max(w, h) > 0 else 0.0
    solidity = float(area) / float(bbox_area)

    # Periemeter via contour
    region_mask = (binary > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    perim = sum(cv2.arcLength(c, closed=True) for c in contours)

    return RegionProps(
        label=label,
        area=area,
        bbox=(x, y, w, h),
        centroid=(cx, cy),
        aspect_ratio=ar,
        solidity=solidity,
        perimeter=float(perim),
    )


def _build_props(
    labels: np.ndarray,
    stats: np.ndarray,
    centroids: np.ndarray,
    n_regions: int,
    binary: np.ndarray,
) -> List[RegionProps]:
    props = []
    for lab in range(1, n_regions + 1):
        region_mask = (labels == lab).astype(np.uint8) * 255
        p = _stats_to_props(lab, stats[lab], centroids[lab], region_mask)
        props.append(p)
    return props
