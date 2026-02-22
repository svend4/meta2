"""Утилиты поворота изображений и точек для реконструкции пазла.

Модуль предоставляет функции для поворота изображений, контуров и наборов
точек на произвольный угол, а также утилиты для нормализации углов и
выбора ближайшего дискретного угла из стандартного набора.

Публичный API:
    RotationConfig       — параметры поворота
    rotate_image_angle   — повернуть изображение на произвольный угол
    rotate_points_angle  — повернуть 2-D точки вокруг центра
    normalize_angle      — нормализовать угол в [0, 2π) или (-π, π]
    angle_difference     — минимальная разница между двумя углами
    nearest_discrete     — ближайший угол из дискретного набора
    angles_to_matrix     — набор углов → массив матриц поворота
    batch_rotate_images  — повернуть несколько изображений
    estimate_rotation    — оценить поворот по двум наборам точек (Procrustes)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─── RotationConfig ───────────────────────────────────────────────────────────

@dataclass
class RotationConfig:
    """Параметры операций поворота.

    Атрибуты:
        border_mode:   Режим заполнения границ OpenCV (cv2.BORDER_*).
        border_value:  Значение заполнения при BORDER_CONSTANT.
        interpolation: Интерполяция OpenCV (cv2.INTER_*).
        expand:        True → увеличить холст, чтобы вместить повёрнутое изображение.
    """

    border_mode: int = cv2.BORDER_CONSTANT
    border_value: Tuple[int, int, int] = (255, 255, 255)
    interpolation: int = cv2.INTER_LINEAR
    expand: bool = True

    def __post_init__(self) -> None:
        if self.border_mode not in (
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REFLECT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_WRAP,
        ):
            raise ValueError(
                f"Неизвестный border_mode: {self.border_mode}"
            )
        if self.interpolation not in (
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ):
            raise ValueError(
                f"Неизвестный interpolation: {self.interpolation}"
            )


# ─── rotate_image_angle ───────────────────────────────────────────────────────

def rotate_image_angle(
    image: np.ndarray,
    angle_deg: float,
    cfg: Optional[RotationConfig] = None,
) -> np.ndarray:
    """Повернуть изображение на заданный угол в градусах.

    Аргументы:
        image:     Изображение (H, W) или (H, W, C).
        angle_deg: Угол поворота в градусах (положительный → CCW).
        cfg:       Параметры (None → RotationConfig()).

    Возвращает:
        Повёрнутое изображение (uint8 или исходный dtype).

    Исключения:
        ValueError: Если image не 2-D/3-D.
    """
    if image.ndim not in (2, 3):
        raise ValueError(
            f"image должен быть 2-D или 3-D, получено ndim={image.ndim}"
        )
    if cfg is None:
        cfg = RotationConfig()

    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    if cfg.expand:
        # Вычислить новые размеры холста
        cos_a = abs(math.cos(math.radians(angle_deg)))
        sin_a = abs(math.sin(math.radians(angle_deg)))
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        # Скорректировать сдвиг
        M[0, 2] += (new_w - w) / 2.0
        M[1, 2] += (new_h - h) / 2.0
        dst_size = (new_w, new_h)
    else:
        dst_size = (w, h)

    border_val = cfg.border_value if image.ndim == 3 else cfg.border_value[0]
    result = cv2.warpAffine(
        image, M, dst_size,
        flags=cfg.interpolation,
        borderMode=cfg.border_mode,
        borderValue=border_val,
    )
    return result


# ─── rotate_points_angle ──────────────────────────────────────────────────────

def rotate_points_angle(
    points: np.ndarray,
    angle_rad: float,
    center: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Повернуть массив 2-D точек на заданный угол в радианах.

    Аргументы:
        points:    Массив точек формы (N, 2).
        angle_rad: Угол поворота в радианах (CCW).
        center:    Центр поворота (2,); None → центроид точек.

    Возвращает:
        Повёрнутые точки (N, 2), float64.

    Исключения:
        ValueError: Если points.shape != (N, 2) или N < 1.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"points должен иметь форму (N, 2), получено {pts.shape}"
        )
    if pts.shape[0] < 1:
        raise ValueError("points должен содержать хотя бы 1 точку")

    if center is None:
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    else:
        cx, cy = float(center[0]), float(center[1])

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    shifted = pts - np.array([cx, cy])
    rotated = shifted @ np.array([[cos_a, -sin_a], [sin_a, cos_a]]).T
    return rotated + np.array([cx, cy])


# ─── normalize_angle ──────────────────────────────────────────────────────────

def normalize_angle(
    angle_rad: float,
    half_range: bool = False,
) -> float:
    """Нормализовать угол.

    Аргументы:
        angle_rad:  Угол в радианах.
        half_range: False → [0, 2π);  True → (-π, π].

    Возвращает:
        Нормализованный угол в радианах.
    """
    two_pi = 2.0 * math.pi
    angle = angle_rad % two_pi
    if half_range:
        if angle > math.pi:
            angle -= two_pi
    return float(angle)


# ─── angle_difference ─────────────────────────────────────────────────────────

def angle_difference(a: float, b: float) -> float:
    """Минимальная разница между двумя углами (в радианах).

    Результат всегда неотрицателен и ≤ π.

    Аргументы:
        a: Первый угол в радианах.
        b: Второй угол в радианах.

    Возвращает:
        Минимальное угловое расстояние ∈ [0, π].
    """
    diff = abs(normalize_angle(a) - normalize_angle(b))
    if diff > math.pi:
        diff = 2.0 * math.pi - diff
    return float(diff)


# ─── nearest_discrete ─────────────────────────────────────────────────────────

def nearest_discrete(
    angle_rad: float,
    candidates: List[float],
) -> float:
    """Вернуть угол из candidates, ближайший к angle_rad.

    Аргументы:
        angle_rad:  Входной угол в радианах.
        candidates: Список углов-кандидатов (в радианах).

    Возвращает:
        Ближайший угол из candidates.

    Исключения:
        ValueError: Если candidates пустой.
    """
    if not candidates:
        raise ValueError("candidates не должен быть пустым")

    best = candidates[0]
    best_diff = angle_difference(angle_rad, best)
    for c in candidates[1:]:
        d = angle_difference(angle_rad, c)
        if d < best_diff:
            best_diff = d
            best = c
    return float(best)


# ─── angles_to_matrix ─────────────────────────────────────────────────────────

def angles_to_matrix(angles_rad: np.ndarray) -> np.ndarray:
    """Преобразовать массив углов в набор матриц поворота 2×2.

    Аргументы:
        angles_rad: 1-D массив углов в радианах (N,).

    Возвращает:
        Массив матриц поворота (N, 2, 2), float64.

    Исключения:
        ValueError: Если angles_rad не 1-D.
    """
    angles = np.asarray(angles_rad, dtype=np.float64)
    if angles.ndim != 1:
        raise ValueError(
            f"angles_rad должен быть 1-D, получено ndim={angles.ndim}"
        )
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    matrices = np.stack([
        np.stack([cos_a, -sin_a], axis=1),
        np.stack([sin_a,  cos_a], axis=1),
    ], axis=1)
    return matrices


# ─── batch_rotate_images ──────────────────────────────────────────────────────

def batch_rotate_images(
    images: List[np.ndarray],
    angles_deg: List[float],
    cfg: Optional[RotationConfig] = None,
) -> List[np.ndarray]:
    """Повернуть несколько изображений на соответствующие углы.

    Аргументы:
        images:     Список изображений.
        angles_deg: Список углов в градусах (того же размера).
        cfg:        Параметры (None → RotationConfig()).

    Возвращает:
        Список повёрнутых изображений.

    Исключения:
        ValueError: Если len(images) != len(angles_deg).
    """
    if len(images) != len(angles_deg):
        raise ValueError(
            f"len(images)={len(images)} != len(angles_deg)={len(angles_deg)}"
        )
    return [rotate_image_angle(img, a, cfg) for img, a in zip(images, angles_deg)]


# ─── estimate_rotation ────────────────────────────────────────────────────────

def estimate_rotation(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> float:
    """Оценить угол поворота между двумя наборами точек (Procrustes).

    Находит угол θ, при котором src, повёрнутые на θ, наилучшим образом
    совпадают с dst (в смысле LS). Оба набора предварительно центрируются.

    Аргументы:
        src_points: Исходные точки (N, 2).
        dst_points: Целевые точки (N, 2); N >= 2.

    Возвращает:
        Угол поворота в радианах ∈ (-π, π].

    Исключения:
        ValueError: Если формы не (N, 2) или N < 2.
    """
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)

    for name, arr in (("src_points", src), ("dst_points", dst)):
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"{name} должен иметь форму (N, 2), получено {arr.shape}"
            )
        if arr.shape[0] < 2:
            raise ValueError(
                f"{name} должен содержать минимум 2 точки"
            )

    src_c = src - src.mean(axis=0)
    dst_c = dst - dst.mean(axis=0)

    # SVD cross-covariance
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Извлечь угол из матрицы 2×2
    angle = math.atan2(R[1, 0], R[0, 0])
    return float(angle)
