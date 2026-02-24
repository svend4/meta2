"""
Извлечение и разбиение контура фрагмента на логические края.
"""
import cv2
import numpy as np
from typing import List, Tuple

from ..models import EdgeSide


def extract_contour(mask: np.ndarray) -> np.ndarray:
    """
    Извлекает упорядоченный внешний контур фрагмента.

    Returns:
        contour: (N, 2) массив точек [x, y] по часовой стрелке.
    """
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("Контур не найден — маска пустая?")
    # Берём самый длинный контур
    c = max(contours, key=cv2.contourArea)
    return c.reshape(-1, 2).astype(np.float32)


def rdp_simplify(contour: np.ndarray, epsilon_ratio: float = 0.005) -> np.ndarray:
    """
    Упрощение контура алгоритмом Рамера-Дугласа-Пекера.

    Args:
        epsilon_ratio: Порог как доля от периметра контура.
    """
    if epsilon_ratio <= 0.0:
        return contour.reshape(-1, 2).astype(np.float32)
    perimeter = cv2.arcLength(contour.reshape(-1, 1, 2).astype(np.float32), True)
    epsilon = epsilon_ratio * perimeter
    simplified = cv2.approxPolyDP(
        contour.reshape(-1, 1, 2).astype(np.float32),
        epsilon, True
    )
    result = simplified.reshape(-1, 2).astype(np.float32)
    if len(result) < 2:
        result = contour.reshape(-1, 2).astype(np.float32)[:2]
    return result


def split_contour_to_edges(contour: np.ndarray,
                           n_sides: int = 4) -> List[Tuple[np.ndarray, EdgeSide]]:
    """
    Разбивает контур на N логических краёв (обычно 4 — как у прямоугольника).

    Метод: находит N точек с наибольшей кривизной как «углы»,
    разбивает контур по ним.

    Returns:
        Список (edge_points, side_label).
    """
    corners = _find_corners(contour, n_sides)
    edges = []
    n = len(contour)

    for k in range(len(corners)):
        start = corners[k]
        end   = corners[(k + 1) % len(corners)]
        if end > start:
            pts = contour[start:end + 1]
        else:
            # Переход через 0
            pts = np.concatenate([contour[start:], contour[:end + 1]])
        side = _label_side(pts, contour)
        edges.append((pts, side))

    return edges


def resample_curve(curve: np.ndarray, n_points: int = 256) -> np.ndarray:
    """
    Равномерно ресэмплирует кривую до n_points точек по длине дуги.
    Нужно для сравнения краёв разной длины.
    """
    diffs = np.diff(curve, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    cumlen = np.concatenate([[0], np.cumsum(seg_lengths)])
    total = cumlen[-1]
    if total == 0:
        return np.tile(curve[0], (n_points, 1))
    t_new = np.linspace(0, total, n_points)
    x_new = np.interp(t_new, cumlen, curve[:, 0])
    y_new = np.interp(t_new, cumlen, curve[:, 1])
    return np.stack([x_new, y_new], axis=1)


def normalize_contour(contour: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Центрирует контур и нормирует масштаб (диагональ = 1).

    Returns:
        (normalized, centroid, scale)
    """
    centroid = contour.mean(axis=0)
    shifted  = contour - centroid
    bbox     = shifted.max(axis=0) - shifted.min(axis=0)
    scale    = np.hypot(bbox[0], bbox[1])
    if scale == 0:
        scale = 1.0
    return shifted / scale, centroid, scale


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _find_corners(contour: np.ndarray, n_corners: int) -> List[int]:
    """
    Находит n_corners индексов с наибольшей локальной кривизной.
    Использует угол между соседними сегментами.
    """
    n = len(contour)
    window = max(5, n // 20)
    angles = np.zeros(n)
    for i in range(n):
        p0 = contour[(i - window) % n]
        p1 = contour[i]
        p2 = contour[(i + window) % n]
        v1 = p0 - p1
        v2 = p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        angles[i] = np.arccos(np.clip(cos_a, -1, 1))

    # Ищем n_corners локальных максимумов с минимальным расстоянием n//n_corners
    min_dist = n // (n_corners * 2)
    corners = []
    used = np.zeros(n, dtype=bool)

    for _ in range(n_corners):
        masked = np.where(used, -1, angles)
        idx = int(np.argmax(masked))
        corners.append(idx)
        lo = max(0, idx - min_dist)
        hi = min(n, idx + min_dist)
        used[lo:hi] = True

    corners.sort()
    return corners


def _label_side(edge_pts: np.ndarray, full_contour: np.ndarray) -> EdgeSide:
    """Определяет сторону (top/bottom/left/right) по положению в bounding box."""
    bbox_min = full_contour.min(axis=0)
    bbox_max = full_contour.max(axis=0)
    center_y = edge_pts[:, 1].mean()
    center_x = edge_pts[:, 0].mean()
    bbox_center = (bbox_min + bbox_max) / 2

    dy = center_y - bbox_center[1]
    dx = center_x - bbox_center[0]

    if abs(dy) > abs(dx):
        return EdgeSide.BOTTOM if dy > 0 else EdgeSide.TOP
    else:
        return EdgeSide.RIGHT if dx > 0 else EdgeSide.LEFT
