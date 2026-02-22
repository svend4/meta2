"""
Отслеживание контуров фрагментов документа.

Предоставляет инструменты для обнаружения контуров на изображениях,
их фильтрации, сопоставления между кадрами и вычисления характеристик.

Экспортирует:
    ContourInfo          — информация о контуре (площадь, периметр, bbox)
    TrackState           — состояние отслеживания контура
    find_contours        — найти контуры на изображении
    filter_contours      — отфильтровать по площади/периметру
    contour_to_array     — контур → float32 массив (N, 2)
    compute_contour_info — вычислить характеристики контура
    match_contours       — сопоставить контуры между двумя кадрами
    track_contour        — обновить состояние отслеживания
    batch_find_contours  — пакетное обнаружение контуров
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class ContourInfo:
    """Характеристики одного контура.

    Attributes:
        contour:   Массив float32 (N, 2) точек контура в формате (x, y).
        area:      Площадь (пиксели², ≥ 0).
        perimeter: Периметр (пиксели, ≥ 0).
        bbox:      Ограничивающий прямоугольник (x, y, w, h).
        centroid:  Центроид (cx, cy).
        params:    Дополнительные параметры.
    """
    contour: np.ndarray
    area: float
    perimeter: float
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    params: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.area < 0:
            raise ValueError(f"area must be >= 0, got {self.area}")
        if self.perimeter < 0:
            raise ValueError(f"perimeter must be >= 0, got {self.perimeter}")

    def __len__(self) -> int:
        return len(self.contour)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ContourInfo(n_pts={len(self)}, area={self.area:.1f}, "
            f"centroid=({self.centroid[0]:.1f}, {self.centroid[1]:.1f}))"
        )


@dataclass
class TrackState:
    """Состояние отслеживания контура между кадрами.

    Attributes:
        track_id:  Идентификатор трека (≥ 0).
        info:      Текущая :class:`ContourInfo`.
        age:       Возраст трека (число обновлений, ≥ 0).
        lost:      Число пропущенных кадров подряд (≥ 0).
        params:    Дополнительные параметры.
    """
    track_id: int
    info: ContourInfo
    age: int = 0
    lost: int = 0
    params: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.track_id < 0:
            raise ValueError(f"track_id must be >= 0, got {self.track_id}")
        if self.age < 0:
            raise ValueError(f"age must be >= 0, got {self.age}")
        if self.lost < 0:
            raise ValueError(f"lost must be >= 0, got {self.lost}")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TrackState(id={self.track_id}, age={self.age}, "
            f"lost={self.lost})"
        )


# ─── Публичные функции ────────────────────────────────────────────────────────

def find_contours(
    mask: np.ndarray,
    mode: str = "external",
) -> List[np.ndarray]:
    """Найти контуры на бинарной маске.

    Args:
        mask: Бинарное изображение uint8 (H, W).
        mode: ``'external'`` — только внешние контуры;
              ``'all'`` — все контуры.

    Returns:
        Список массивов float32 (N_i, 2) — координаты (x, y) каждого контура.

    Raises:
        ValueError: Если ``mask`` не 2-D или ``mode`` неизвестен.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got ndim={mask.ndim}")
    if mode not in ("external", "all"):
        raise ValueError(
            f"mode must be 'external' or 'all', got {mode!r}"
        )
    retr = cv2.RETR_EXTERNAL if mode == "external" else cv2.RETR_LIST
    src = mask.astype(np.uint8)
    raw, _ = cv2.findContours(src, retr, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for cnt in raw:
        pts = cnt.reshape(-1, 2).astype(np.float32)
        result.append(pts)
    return result


def contour_to_array(contour: np.ndarray) -> np.ndarray:
    """Привести контур к стандартному формату float32 (N, 2).

    Args:
        contour: Массив точек в любом допустимом формате OpenCV.

    Returns:
        Массив float32 (N, 2) точек (x, y).
    """
    arr = np.asarray(contour, dtype=np.float32)
    return arr.reshape(-1, 2)


def compute_contour_info(contour: np.ndarray) -> ContourInfo:
    """Вычислить характеристики контура.

    Args:
        contour: Массив float32 (N, 2) точек (x, y).

    Returns:
        :class:`ContourInfo` с площадью, периметром, bbox и центроидом.

    Raises:
        ValueError: Если контур пустой.
    """
    pts = contour_to_array(contour)
    if len(pts) == 0:
        raise ValueError("contour must not be empty")
    cv_cnt = pts.reshape(-1, 1, 2).astype(np.float32)
    area = float(cv2.contourArea(cv_cnt))
    perimeter = float(cv2.arcLength(cv_cnt, closed=True))
    x, y, w, h = cv2.boundingRect(cv_cnt.astype(np.int32))
    # Центроид через моменты
    M = cv2.moments(cv_cnt)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
    return ContourInfo(
        contour=pts,
        area=area,
        perimeter=perimeter,
        bbox=(int(x), int(y), int(w), int(h)),
        centroid=(float(cx), float(cy)),
    )


def filter_contours(
    contours: List[np.ndarray],
    min_area: float = 10.0,
    max_area: float = float("inf"),
    min_perimeter: float = 0.0,
) -> List[np.ndarray]:
    """Отфильтровать контуры по площади и периметру.

    Args:
        contours:      Список контуров float32 (N_i, 2).
        min_area:      Минимальная площадь (≥ 0).
        max_area:      Максимальная площадь (> min_area).
        min_perimeter: Минимальный периметр (≥ 0).

    Returns:
        Отфильтрованный список контуров.

    Raises:
        ValueError: Если ``min_area`` < 0 или ``max_area`` ≤ ``min_area``.
    """
    if min_area < 0:
        raise ValueError(f"min_area must be >= 0, got {min_area}")
    if max_area <= min_area:
        raise ValueError(
            f"max_area ({max_area}) must be > min_area ({min_area})"
        )
    if min_perimeter < 0:
        raise ValueError(f"min_perimeter must be >= 0, got {min_perimeter}")
    result = []
    for cnt in contours:
        info = compute_contour_info(cnt)
        if (min_area <= info.area <= max_area
                and info.perimeter >= min_perimeter):
            result.append(cnt)
    return result


def match_contours(
    prev: List[ContourInfo],
    curr: List[ContourInfo],
    max_dist: float = 50.0,
) -> List[Tuple[int, int]]:
    """Сопоставить контуры между двумя кадрами по расстоянию центроидов.

    Args:
        prev:     Список :class:`ContourInfo` предыдущего кадра.
        curr:     Список :class:`ContourInfo` текущего кадра.
        max_dist: Максимальное расстояние центроидов для совпадения (> 0).

    Returns:
        Список пар (prev_idx, curr_idx) совпавших контуров.

    Raises:
        ValueError: Если ``max_dist`` ≤ 0.
    """
    if max_dist <= 0:
        raise ValueError(f"max_dist must be > 0, got {max_dist}")
    if not prev or not curr:
        return []
    matched: List[Tuple[int, int]] = []
    used_curr = set()
    for pi, p_info in enumerate(prev):
        px, py = p_info.centroid
        best_dist = max_dist
        best_ci = -1
        for ci, c_info in enumerate(curr):
            if ci in used_curr:
                continue
            cx, cy = c_info.centroid
            dist = float(np.sqrt((px - cx) ** 2 + (py - cy) ** 2))
            if dist < best_dist:
                best_dist = dist
                best_ci = ci
        if best_ci >= 0:
            matched.append((pi, best_ci))
            used_curr.add(best_ci)
    return matched


def track_contour(
    state: TrackState,
    new_info: Optional[ContourInfo],
) -> TrackState:
    """Обновить состояние трека новым наблюдением.

    Args:
        state:    Текущее :class:`TrackState`.
        new_info: Новая :class:`ContourInfo` или ``None`` (контур потерян).

    Returns:
        Обновлённый :class:`TrackState`.
    """
    if new_info is not None:
        return TrackState(
            track_id=state.track_id,
            info=new_info,
            age=state.age + 1,
            lost=0,
            params=state.params,
        )
    return TrackState(
        track_id=state.track_id,
        info=state.info,
        age=state.age + 1,
        lost=state.lost + 1,
        params=state.params,
    )


def batch_find_contours(
    masks: List[np.ndarray],
    mode: str = "external",
) -> List[List[np.ndarray]]:
    """Пакетное обнаружение контуров на списке масок.

    Args:
        masks: Список бинарных масок uint8 (H_i, W_i).
        mode:  Режим: ``'external'`` или ``'all'``.

    Returns:
        Список списков контуров — по одному на каждую маску.
    """
    return [find_contours(mask, mode=mode) for mask in masks]
