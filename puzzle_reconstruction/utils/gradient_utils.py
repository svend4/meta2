"""
Утилиты градиентного анализа изображений для реконструкции документов.

Предоставляет функции для вычисления градиентных карт, обнаружения границ
и анализа локальных изменений яркости, используемых при сопоставлении
краёв фрагментов.

Экспортирует:
    GradientConfig          — параметры градиентных операций
    compute_gradient_magnitude — карта величины градиента
    compute_gradient_direction — карта направления градиента (радианы)
    compute_sobel           — детектор границ Собела (возвращает (mag, dx, dy))
    compute_laplacian       — лапласиан изображения
    threshold_gradient      — бинаризация градиентной карты по порогу
    suppress_non_maximum    — подавление немаксимумов вдоль градиента
    compute_edge_density    — плотность пикселей границ в ROI
    batch_compute_gradients — градиент для пакета изображений
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


# ─── GradientConfig ───────────────────────────────────────────────────────────

@dataclass
class GradientConfig:
    """Параметры градиентных операций.

    Attributes:
        ksize:      Размер ядра Собела (нечётное, >= 1).
        normalize:  Нормировать ли результат в [0, 1].
        threshold:  Порог бинаризации для threshold_gradient (0..255).
    """
    ksize:     int   = 3
    normalize: bool  = True
    threshold: float = 32.0

    def __post_init__(self) -> None:
        if self.ksize < 1 or self.ksize % 2 == 0:
            raise ValueError(
                f"ksize must be a positive odd integer, got {self.ksize}"
            )
        if not (0.0 <= self.threshold <= 255.0):
            raise ValueError(
                f"threshold must be in [0, 255], got {self.threshold}"
            )


# ─── compute_gradient_magnitude ───────────────────────────────────────────────

def compute_gradient_magnitude(
    image: np.ndarray,
    cfg: GradientConfig | None = None,
) -> np.ndarray:
    """Вычислить карту величины градиента изображения.

    Args:
        image: Входное изображение (H, W) или (H, W, C), uint8 или float32.
        cfg:   Параметры. None → GradientConfig().

    Returns:
        Карта величины float32 формы (H, W). Если cfg.normalize=True,
        значения ∈ [0, 1], иначе в пикселях/единицах Собела.

    Raises:
        ValueError: Если image не 2-D или 3-D.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2-D or 3-D, got ndim={image.ndim}")
    if cfg is None:
        cfg = GradientConfig()

    gray = _to_gray_float(image)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=cfg.ksize)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=cfg.ksize)
    mag = np.hypot(dx, dy)
    if cfg.normalize:
        m = float(mag.max())
        if m > 0.0:
            mag = mag / m
    return mag.astype(np.float32)


# ─── compute_gradient_direction ───────────────────────────────────────────────

def compute_gradient_direction(
    image: np.ndarray,
    cfg: GradientConfig | None = None,
) -> np.ndarray:
    """Вычислить карту направления градиента (в радианах, (-π, π]).

    Args:
        image: Входное изображение (H, W) или (H, W, C), uint8 или float32.
        cfg:   Параметры. None → GradientConfig().

    Returns:
        Карта направлений float32 формы (H, W), значения ∈ (-π, π].

    Raises:
        ValueError: Если image не 2-D или 3-D.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2-D or 3-D, got ndim={image.ndim}")
    if cfg is None:
        cfg = GradientConfig()

    gray = _to_gray_float(image)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=cfg.ksize)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=cfg.ksize)
    return np.arctan2(dy, dx).astype(np.float32)


# ─── compute_sobel ────────────────────────────────────────────────────────────

def compute_sobel(
    image: np.ndarray,
    cfg: GradientConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Детектор границ Собела — возвращает (magnitude, dx, dy).

    Args:
        image: Входное изображение (H, W) или (H, W, C), uint8 или float32.
        cfg:   Параметры. None → GradientConfig().

    Returns:
        Кортеж (magnitude, dx, dy), каждый — float32 формы (H, W).
        magnitude нормирована в [0, 1] если cfg.normalize=True.

    Raises:
        ValueError: Если image не 2-D или 3-D.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2-D or 3-D, got ndim={image.ndim}")
    if cfg is None:
        cfg = GradientConfig()

    gray = _to_gray_float(image)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=cfg.ksize)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=cfg.ksize)
    mag = np.hypot(dx, dy)
    if cfg.normalize:
        m = float(mag.max())
        if m > 0.0:
            mag = mag / m
    return mag.astype(np.float32), dx.astype(np.float32), dy.astype(np.float32)


# ─── compute_laplacian ────────────────────────────────────────────────────────

def compute_laplacian(
    image: np.ndarray,
    ksize: int = 3,
    normalize: bool = True,
) -> np.ndarray:
    """Вычислить лапласиан изображения (мера резкости/границ второго порядка).

    Args:
        image:     Входное изображение (H, W) или (H, W, C).
        ksize:     Размер ядра лапласиана (нечётное >= 1).
        normalize: Нормировать ли абсолютные значения в [0, 1].

    Returns:
        Лапласиан float32 формы (H, W). Содержит отрицательные значения
        если normalize=False.

    Raises:
        ValueError: Если image не 2-D или 3-D, или ksize некорректен.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2-D or 3-D, got ndim={image.ndim}")
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(
            f"ksize must be a positive odd integer, got {ksize}"
        )

    gray = _to_gray_float(image)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
    if normalize:
        abs_lap = np.abs(lap)
        m = float(abs_lap.max())
        if m > 0.0:
            lap = abs_lap / m
        else:
            lap = abs_lap
    return lap.astype(np.float32)


# ─── threshold_gradient ───────────────────────────────────────────────────────

def threshold_gradient(
    magnitude: np.ndarray,
    threshold: float | None = None,
    cfg: GradientConfig | None = None,
) -> np.ndarray:
    """Бинаризовать карту величины градиента по порогу.

    Args:
        magnitude: Карта градиента float32 (H, W).
        threshold: Явный порог ∈ [0, 1] (если magnitude нормирована)
                   или пиксельный. None → cfg.threshold / 255.
        cfg:       Параметры. None → GradientConfig().

    Returns:
        Бинарная маска bool (H, W): True = граница.

    Raises:
        ValueError: Если magnitude не 2-D.
    """
    if magnitude.ndim != 2:
        raise ValueError(
            f"magnitude must be 2-D, got ndim={magnitude.ndim}"
        )
    if cfg is None:
        cfg = GradientConfig()
    if threshold is None:
        threshold = cfg.threshold / 255.0

    return (magnitude >= threshold).astype(bool)


# ─── suppress_non_maximum ─────────────────────────────────────────────────────

def suppress_non_maximum(
    magnitude: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Подавление немаксимумов — оставить только локальные максимумы вдоль градиента.

    Реализует упрощённое NMS: каждый пиксель сохраняется, только если он
    максимален среди двух соседей вдоль направления градиента
    (8 квантованных направлений).

    Args:
        magnitude: Карта величины float32 (H, W).
        direction: Карта направлений float32 (H, W) в радианах.

    Returns:
        Утончённая карта float32 (H, W): немаксимумы обнулены.

    Raises:
        ValueError: Если формы magnitude и direction не совпадают или не 2-D.
    """
    if magnitude.ndim != 2:
        raise ValueError(
            f"magnitude must be 2-D, got ndim={magnitude.ndim}"
        )
    if direction.shape != magnitude.shape:
        raise ValueError(
            f"magnitude and direction shapes must match: "
            f"{magnitude.shape} vs {direction.shape}"
        )

    h, w = magnitude.shape
    out = np.zeros_like(magnitude)
    # Квантование угла к 4 направлениям (0, 45, 90, 135 degrees)
    angle_deg = np.degrees(direction) % 180.0
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            ang = angle_deg[y, x]
            if ang < 22.5 or ang >= 157.5:
                q, r = magnitude[y, x - 1], magnitude[y, x + 1]
            elif ang < 67.5:
                q, r = magnitude[y - 1, x + 1], magnitude[y + 1, x - 1]
            elif ang < 112.5:
                q, r = magnitude[y - 1, x], magnitude[y + 1, x]
            else:
                q, r = magnitude[y - 1, x - 1], magnitude[y + 1, x + 1]
            val = magnitude[y, x]
            if val >= q and val >= r:
                out[y, x] = val
    return out


# ─── compute_edge_density ─────────────────────────────────────────────────────

def compute_edge_density(
    image: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
    cfg: GradientConfig | None = None,
) -> float:
    """Вычислить долю пикселей границ в области интереса (ROI).

    Args:
        image: Входное изображение (H, W) или (H, W, C).
        roi:   (y1, x1, y2, x2) — регион интереса. None → всё изображение.
        cfg:   Параметры. None → GradientConfig().

    Returns:
        Плотность границ ∈ [0, 1]: доля пикселей, превышающих порог.

    Raises:
        ValueError: Если image не 2-D или 3-D, или roi некорректен.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2-D or 3-D, got ndim={image.ndim}")
    if cfg is None:
        cfg = GradientConfig()

    mag = compute_gradient_magnitude(image, cfg)

    if roi is not None:
        y1, x1, y2, x2 = roi
        h, w = mag.shape
        y1 = max(0, y1); x1 = max(0, x1)
        y2 = min(h, y2); x2 = min(w, x2)
        if y2 <= y1 or x2 <= x1:
            return 0.0
        mag = mag[y1:y2, x1:x2]

    binary = threshold_gradient(mag, cfg=cfg)
    total = binary.size
    if total == 0:
        return 0.0
    return float(binary.sum()) / total


# ─── batch_compute_gradients ──────────────────────────────────────────────────

def batch_compute_gradients(
    images: List[np.ndarray],
    cfg: GradientConfig | None = None,
) -> List[np.ndarray]:
    """Вычислить карты величины градиента для пакета изображений.

    Args:
        images: Список изображений (H, W) или (H, W, C).
        cfg:    Параметры. None → GradientConfig().

    Returns:
        Список карт величины float32, по одной на изображение.

    Raises:
        ValueError: Если images пуст.
    """
    if not images:
        raise ValueError("images must not be empty")
    if cfg is None:
        cfg = GradientConfig()
    return [compute_gradient_magnitude(img, cfg) for img in images]


# ─── helpers ──────────────────────────────────────────────────────────────────

def _to_gray_float(image: np.ndarray) -> np.ndarray:
    """Конвертировать изображение в серое float32 ∈ [0, 255]."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return gray.astype(np.float32)
