"""
Адаптивная бинаризация изображений фрагментов документа.

Предоставляет инструменты для локальной (адаптивной) пороговой бинаризации
с учётом неравномерного освещения — методы Ниблака, Саволы, Бернсена
и стандартного адаптивного порога OpenCV.

Экспортирует:
    ThresholdParams        — параметры бинаризации
    global_threshold       — глобальная бинаризация (Otsu / фиксированный)
    adaptive_mean          — адаптивная бинаризация по среднему
    adaptive_gaussian      — адаптивная бинаризация по гауссиану
    niblack_threshold      — метод Ниблака (µ + k·σ)
    sauvola_threshold      — метод Саволы (µ·(1 + k·(σ/R − 1)))
    bernsen_threshold      — метод Бернсена (среднее локального контраста)
    apply_threshold        — применить метод по параметрам
    batch_threshold        — пакетная бинаризация
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import cv2
import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

_VALID_METHODS = frozenset({
    "global", "otsu", "adaptive_mean", "adaptive_gaussian",
    "niblack", "sauvola", "bernsen",
})


@dataclass
class ThresholdParams:
    """Параметры адаптивной бинаризации.

    Attributes:
        method:      Метод: ``'global'``, ``'otsu'``, ``'adaptive_mean'``,
                     ``'adaptive_gaussian'``, ``'niblack'``, ``'sauvola'``,
                     ``'bernsen'``.
        block_size:  Размер локального блока (нечётное ≥ 3).
        k:           Коэффициент смещения для Ниблака/Саволы (любой float).
        threshold:   Фиксированный порог 0–255 (для метода ``'global'``).
        params:      Дополнительные параметры.
    """
    method: str = "otsu"
    block_size: int = 11
    k: float = 0.2
    threshold: int = 128
    params: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"method must be one of {sorted(_VALID_METHODS)}, "
                f"got {self.method!r}"
            )
        if self.block_size < 3:
            raise ValueError(
                f"block_size must be >= 3, got {self.block_size}"
            )
        if self.block_size % 2 == 0:
            raise ValueError(
                f"block_size must be odd, got {self.block_size}"
            )
        if not (0 <= self.threshold <= 255):
            raise ValueError(
                f"threshold must be in [0, 255], got {self.threshold}"
            )


# ─── Вспомогательная функция ──────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.ndim == 2:
        return arr.copy()
    raise ValueError(f"img must be 2-D or 3-D, got ndim={arr.ndim}")


def _pad_for_block(img: np.ndarray, block_size: int) -> np.ndarray:
    """Добавить зеркальный паддинг, чтобы блок помещался у краёв."""
    pad = block_size // 2
    return np.pad(img.astype(np.float64), pad, mode="reflect")


# ─── Публичные функции ────────────────────────────────────────────────────────

def global_threshold(
    img: np.ndarray,
    threshold: int = 128,
    use_otsu: bool = False,
) -> np.ndarray:
    """Глобальная бинаризация изображения.

    Args:
        img:       Изображение uint8 (H, W) или (H, W, C).
        threshold: Порог 0–255 (игнорируется при ``use_otsu=True``).
        use_otsu:  Если ``True`` — автоматический порог Отсу.

    Returns:
        Бинарная маска uint8 (H, W): 255 — объект, 0 — фон.

    Raises:
        ValueError: Если ``threshold`` вне [0, 255].
    """
    if not (0 <= threshold <= 255):
        raise ValueError(f"threshold must be in [0, 255], got {threshold}")
    gray = _to_gray(img)
    flags = cv2.THRESH_BINARY
    if use_otsu:
        flags |= cv2.THRESH_OTSU
        threshold = 0
    _, binary = cv2.threshold(gray.astype(np.uint8), threshold, 255, flags)
    return binary


def adaptive_mean(
    img: np.ndarray,
    block_size: int = 11,
    c: float = 2.0,
) -> np.ndarray:
    """Адаптивная бинаризация по среднему блока.

    Args:
        img:        Изображение uint8 (H, W) или (H, W, C).
        block_size: Размер блока (нечётное ≥ 3).
        c:          Вычитаемая константа.

    Returns:
        Бинарная маска uint8 (H, W).

    Raises:
        ValueError: Если ``block_size`` < 3 или чётный.
    """
    if block_size < 3:
        raise ValueError(f"block_size must be >= 3, got {block_size}")
    if block_size % 2 == 0:
        raise ValueError(f"block_size must be odd, got {block_size}")
    gray = _to_gray(img).astype(np.uint8)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size, c,
    )


def adaptive_gaussian(
    img: np.ndarray,
    block_size: int = 11,
    c: float = 2.0,
) -> np.ndarray:
    """Адаптивная бинаризация по взвешенному среднему (гауссиан).

    Args:
        img:        Изображение uint8 (H, W) или (H, W, C).
        block_size: Размер блока (нечётное ≥ 3).
        c:          Вычитаемая константа.

    Returns:
        Бинарная маска uint8 (H, W).

    Raises:
        ValueError: Если ``block_size`` < 3 или чётный.
    """
    if block_size < 3:
        raise ValueError(f"block_size must be >= 3, got {block_size}")
    if block_size % 2 == 0:
        raise ValueError(f"block_size must be odd, got {block_size}")
    gray = _to_gray(img).astype(np.uint8)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c,
    )


def niblack_threshold(
    img: np.ndarray,
    block_size: int = 15,
    k: float = -0.2,
) -> np.ndarray:
    """Бинаризация методом Ниблака: T = µ + k·σ.

    Args:
        img:        Изображение uint8 (H, W) или (H, W, C).
        block_size: Размер локального окна (нечётное ≥ 3).
        k:          Коэффициент смещения (обычно отрицательный для текста).

    Returns:
        Бинарная маска uint8 (H, W).

    Raises:
        ValueError: Если ``block_size`` < 3 или чётный.
    """
    if block_size < 3:
        raise ValueError(f"block_size must be >= 3, got {block_size}")
    if block_size % 2 == 0:
        raise ValueError(f"block_size must be odd, got {block_size}")
    gray = _to_gray(img).astype(np.float64)
    padded = _pad_for_block(gray, block_size)
    h, w = gray.shape
    half = block_size // 2
    result = np.zeros((h, w), dtype=np.uint8)
    for r in range(h):
        for c_idx in range(w):
            block = padded[r: r + block_size, c_idx: c_idx + block_size]
            mu = block.mean()
            sigma = block.std()
            t = mu + k * sigma
            result[r, c_idx] = 255 if gray[r, c_idx] > t else 0
    return result


def sauvola_threshold(
    img: np.ndarray,
    block_size: int = 15,
    k: float = 0.2,
    r: float = 128.0,
) -> np.ndarray:
    """Бинаризация методом Саволы: T = µ·(1 + k·(σ/R − 1)).

    Args:
        img:        Изображение uint8 (H, W) или (H, W, C).
        block_size: Размер локального окна (нечётное ≥ 3).
        k:          Коэффициент чувствительности ∈ (0, 1).
        r:          Динамический диапазон σ (обычно 128 для 8-бит).

    Returns:
        Бинарная маска uint8 (H, W).

    Raises:
        ValueError: Если ``block_size`` < 3, чётный или ``r`` ≤ 0.
    """
    if block_size < 3:
        raise ValueError(f"block_size must be >= 3, got {block_size}")
    if block_size % 2 == 0:
        raise ValueError(f"block_size must be odd, got {block_size}")
    if r <= 0:
        raise ValueError(f"r must be > 0, got {r}")
    gray = _to_gray(img).astype(np.float64)
    padded = _pad_for_block(gray, block_size)
    h, w = gray.shape
    result = np.zeros((h, w), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            block = padded[row: row + block_size, col: col + block_size]
            mu = block.mean()
            sigma = block.std()
            t = mu * (1.0 + k * (sigma / r - 1.0))
            result[row, col] = 255 if gray[row, col] > t else 0
    return result


def bernsen_threshold(
    img: np.ndarray,
    block_size: int = 15,
    contrast_threshold: float = 15.0,
) -> np.ndarray:
    """Бинаризация методом Бернсена: T = (max + min) / 2.

    Если локальный контраст ниже ``contrast_threshold`` — пиксель
    относится к фону (0).

    Args:
        img:                Изображение uint8 (H, W) или (H, W, C).
        block_size:         Размер локального окна (нечётное ≥ 3).
        contrast_threshold: Порог контраста для определения фона (≥ 0).

    Returns:
        Бинарная маска uint8 (H, W).

    Raises:
        ValueError: Если ``block_size`` < 3, чётный или
                    ``contrast_threshold`` < 0.
    """
    if block_size < 3:
        raise ValueError(f"block_size must be >= 3, got {block_size}")
    if block_size % 2 == 0:
        raise ValueError(f"block_size must be odd, got {block_size}")
    if contrast_threshold < 0:
        raise ValueError(
            f"contrast_threshold must be >= 0, got {contrast_threshold}"
        )
    gray = _to_gray(img).astype(np.float64)
    padded = _pad_for_block(gray, block_size)
    h, w = gray.shape
    result = np.zeros((h, w), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            block = padded[row: row + block_size, col: col + block_size]
            b_min = block.min()
            b_max = block.max()
            contrast = b_max - b_min
            if contrast < contrast_threshold:
                result[row, col] = 0
            else:
                t = (b_max + b_min) / 2.0
                result[row, col] = 255 if gray[row, col] > t else 0
    return result


def apply_threshold(
    img: np.ndarray,
    params: ThresholdParams,
) -> np.ndarray:
    """Применить метод бинаризации по параметрам.

    Args:
        img:    Изображение uint8 (H, W) или (H, W, C).
        params: :class:`ThresholdParams`.

    Returns:
        Бинарная маска uint8 (H, W).
    """
    method = params.method
    bs = params.block_size
    k = params.k
    if method == "otsu":
        return global_threshold(img, use_otsu=True)
    if method == "global":
        return global_threshold(img, threshold=params.threshold)
    if method == "adaptive_mean":
        return adaptive_mean(img, block_size=bs)
    if method == "adaptive_gaussian":
        return adaptive_gaussian(img, block_size=bs)
    if method == "niblack":
        return niblack_threshold(img, block_size=bs, k=k)
    if method == "sauvola":
        return sauvola_threshold(img, block_size=bs, k=k)
    return bernsen_threshold(img, block_size=bs)


def batch_threshold(
    images: List[np.ndarray],
    params: ThresholdParams,
) -> List[np.ndarray]:
    """Пакетная бинаризация изображений.

    Args:
        images: Список изображений uint8.
        params: :class:`ThresholdParams`.

    Returns:
        Список бинарных масок uint8 той же длины.
    """
    return [apply_threshold(img, params) for img in images]
