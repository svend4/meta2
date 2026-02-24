"""
Извлечение граничных профилей фрагментов документа.

Предоставляет инструменты для обнаружения границ фрагментов,
разбиения контура на стороны и упрощения граничных кривых.

Экспортирует:
    EdgeSegment       — сегмент границы (набор точек и метаданные)
    FragmentEdges     — полный набор граничных сегментов фрагмента
    detect_boundary   — бинарная маска границы фрагмента
    extract_edge_points — набор граничных точек из маски
    split_edge_by_side  — разбить граничные точки по сторонам
    compute_edge_length — длина дуги по точкам
    simplify_edge     — упрощение граничной кривой (RDP-подобный порог)
    extract_fragment_edges — извлечь полный набор граней фрагмента
    batch_extract_edges — пакетное извлечение
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class EdgeSegment:
    """Сегмент границы фрагмента.

    Attributes:
        points: Граничные точки float32 (N, 2) в формате (x, y).
        side:   Сторона: ``'top'``, ``'bottom'``, ``'left'``, ``'right'``
                или ``'unknown'``.
        length: Длина дуги по точкам (пиксели, ≥ 0).
        params: Дополнительные параметры.
    """
    points: np.ndarray
    side: str = "unknown"
    length: float = 0.0
    params: Dict[str, object] = field(default_factory=dict)

    _VALID_SIDES = frozenset({"top", "bottom", "left", "right", "unknown"})

    def __post_init__(self) -> None:
        if self.side not in self._VALID_SIDES:
            raise ValueError(
                f"side must be one of {sorted(self._VALID_SIDES)}, "
                f"got {self.side!r}"
            )
        if self.length < 0:
            raise ValueError(f"length must be >= 0, got {self.length}")

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EdgeSegment(side={self.side!r}, n_pts={len(self)}, "
            f"length={self.length:.1f})"
        )


@dataclass
class FragmentEdges:
    """Набор граничных сегментов фрагмента.

    Attributes:
        segments:   Список :class:`EdgeSegment`.
        n_segments: Количество сегментов.
        params:     Параметры извлечения.
    """
    segments: List[EdgeSegment]
    n_segments: int
    params: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_segments < 0:
            raise ValueError(
                f"n_segments must be >= 0, got {self.n_segments}"
            )

    def __len__(self) -> int:
        return self.n_segments

    def __repr__(self) -> str:  # pragma: no cover
        return f"FragmentEdges(n_segments={self.n_segments})"


# ─── Публичные функции ────────────────────────────────────────────────────────

def detect_boundary(
    img: np.ndarray,
    threshold: int = 10,
) -> np.ndarray:
    """Обнаружить бинарную маску границы (контура) фрагмента.

    Применяет пороговую бинаризацию и выделяет внешний контур.

    Args:
        img:       Изображение uint8 (H, W) или (H, W, C).
        threshold: Порог бинаризации 0–255 (по умолчанию 10).

    Returns:
        Бинарная маска uint8 (H, W): 255 — граница фрагмента, 0 — фон.

    Raises:
        ValueError: Если ``threshold`` вне [0, 255].
    """
    if not (0 <= threshold <= 255):
        raise ValueError(
            f"threshold must be in [0, 255], got {threshold}"
        )
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1,
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)
    boundary = cv2.subtract(binary, eroded)
    return boundary


def extract_edge_points(
    mask: np.ndarray,
) -> np.ndarray:
    """Извлечь граничные точки из бинарной маски.

    Args:
        mask: Бинарная маска uint8 (H, W).

    Returns:
        Массив float32 (N, 2) точек в формате (x, y);
        пустой (0, 2) если границ нет.

    Raises:
        ValueError: Если ``mask`` не 2-D.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got ndim={mask.ndim}")
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    return pts


def split_edge_by_side(
    points: np.ndarray,
    img_shape: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    """Разбить граничные точки по ближайшей стороне изображения.

    Args:
        points:    Массив float32 (N, 2) точек (x, y).
        img_shape: Форма изображения (H, W).

    Returns:
        Словарь ``{'top': ..., 'bottom': ..., 'left': ..., 'right': ...}``
        с массивами float32 (M, 2) для каждой стороны.

    Raises:
        ValueError: Если ``points`` не 2-D или shape некорректна.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"points must be (N, 2), got shape {points.shape}"
        )
    h, w = img_shape
    if h <= 0 or w <= 0:
        raise ValueError(
            f"img_shape must have positive dimensions, got {img_shape}"
        )

    sides: Dict[str, List] = {"top": [], "bottom": [], "left": [], "right": []}
    for pt in points:
        x, y = float(pt[0]), float(pt[1])
        d_top = y
        d_bottom = (h - 1) - y
        d_left = x
        d_right = (w - 1) - x
        min_d = min(d_top, d_bottom, d_left, d_right)
        if min_d == d_top:
            sides["top"].append(pt)
        elif min_d == d_bottom:
            sides["bottom"].append(pt)
        elif min_d == d_left:
            sides["left"].append(pt)
        else:
            sides["right"].append(pt)

    return {
        s: np.array(v, dtype=np.float32).reshape(-1, 2)
        if v else np.zeros((0, 2), dtype=np.float32)
        for s, v in sides.items()
    }


def compute_edge_length(points: np.ndarray) -> float:
    """Вычислить длину дуги по набору точек.

    Args:
        points: Массив float32 (N, 2) точек (x, y).

    Returns:
        Суммарная длина дуги (сумма Евклидовых расстояний соседних точек).
        Для менее чем 2 точек → 0.0.
    """
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.sqrt((diffs ** 2).sum(axis=1))))


def simplify_edge(
    points: np.ndarray,
    epsilon: float = 1.0,
) -> np.ndarray:
    """Упрощение граничной кривой удалением близких точек.

    Точки, расстояние от которых до прямой первая-последняя менее ``epsilon``,
    удаляются итерационно (жадный алгоритм).

    Args:
        points:  Массив float32 (N, 2).
        epsilon: Порог расстояния (≥ 0).

    Returns:
        Упрощённый массив float32 (M, 2), M ≤ N.

    Raises:
        ValueError: Если ``epsilon`` < 0.
    """
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")
    if len(points) <= 2:
        return points.copy()
    # Используем cv2.approxPolyDP
    pts_int = points.reshape(-1, 1, 2).astype(np.float32)
    simplified = cv2.approxPolyDP(pts_int, epsilon=epsilon, closed=False)
    return simplified.reshape(-1, 2).astype(np.float32)


def extract_fragment_edges(
    img: np.ndarray,
    threshold: int = 10,
    epsilon: float = 1.0,
) -> FragmentEdges:
    """Извлечь полный набор граничных сегментов фрагмента.

    Args:
        img:       Изображение uint8 (H, W) или (H, W, C).
        threshold: Порог бинаризации для обнаружения границы.
        epsilon:   Порог упрощения кривой (0 = без упрощения).

    Returns:
        :class:`FragmentEdges` с сегментами для 4 сторон.
    """
    boundary = detect_boundary(img, threshold=threshold)
    pts = extract_edge_points(boundary)

    h, w = img.shape[:2]
    sides_pts = split_edge_by_side(pts, (h, w))

    segments: List[EdgeSegment] = []
    for side_name in ("top", "bottom", "left", "right"):
        side_pts = sides_pts[side_name]
        if len(side_pts) >= 2:
            simplified = simplify_edge(side_pts, epsilon=epsilon)
        else:
            simplified = side_pts
        length = compute_edge_length(simplified)
        segments.append(EdgeSegment(
            points=simplified,
            side=side_name,
            length=length,
            params={"epsilon": epsilon},
        ))

    return FragmentEdges(
        segments=segments,
        n_segments=len(segments),
        params={"threshold": threshold, "epsilon": epsilon},
    )


def batch_extract_edges(
    images: List[np.ndarray],
    threshold: int = 10,
    epsilon: float = 1.0,
) -> List[FragmentEdges]:
    """Пакетное извлечение граничных сегментов.

    Args:
        images:    Список изображений uint8.
        threshold: Порог бинаризации.
        epsilon:   Порог упрощения.

    Returns:
        Список :class:`FragmentEdges` той же длины.
    """
    return [
        extract_fragment_edges(img, threshold=threshold, epsilon=epsilon)
        for img in images
    ]
