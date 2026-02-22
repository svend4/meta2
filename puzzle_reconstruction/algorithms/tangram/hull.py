"""
Convex hull и упрощение контура для танграм-аппроксимации.
"""
import cv2
import numpy as np


def convex_hull(contour: np.ndarray) -> np.ndarray:
    """
    Выпуклая оболочка контура.

    Returns:
        hull: (K, 2) вершины по часовой стрелке.
    """
    pts = contour.reshape(-1, 1, 2).astype(np.float32)
    hull_idx = cv2.convexHull(pts, returnPoints=True)
    return hull_idx.reshape(-1, 2).astype(np.float32)


def rdp_simplify(contour: np.ndarray, epsilon_ratio: float = 0.02) -> np.ndarray:
    """
    Упрощение контура алгоритмом Рамера-Дугласа-Пекера.
    epsilon_ratio задаётся как доля от периметра.
    """
    pts = contour.reshape(-1, 1, 2).astype(np.float32)
    perimeter = cv2.arcLength(pts, True)
    epsilon = epsilon_ratio * perimeter
    simplified = cv2.approxPolyDP(pts, epsilon, True)
    return simplified.reshape(-1, 2).astype(np.float32)


def normalize_polygon(polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Нормализует полигон:
    - Центроид → начало координат
    - Главная ось (по инерции) → ось X
    - Масштаб: диагональ описанного прямоугольника = 1

    Returns:
        (normalized_polygon, centroid, scale, angle_rad)
    """
    centroid = polygon.mean(axis=0)
    shifted  = polygon - centroid

    # Угол главной оси через PCA
    _, _, vt = np.linalg.svd(shifted, full_matrices=False)
    angle = float(np.arctan2(vt[0, 1], vt[0, 0]))

    # Поворот на главную ось
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s], [s, c]])
    rotated = (R @ shifted.T).T

    # Масштабирование
    bbox = rotated.max(axis=0) - rotated.min(axis=0)
    scale = float(np.hypot(bbox[0], bbox[1]))
    if scale == 0:
        scale = 1.0
    normalized = rotated / scale

    return normalized, centroid, scale, angle
