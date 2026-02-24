"""Нормализация цвета фрагментов пазла.

Модуль реализует методы выравнивания яркости и цвета: гамма-коррекция,
гистограммная эквализация, CLAHE, баланс белого (Grey World, Max RGB),
а также канальная нормализация.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


_METHODS = {"gamma", "equalize", "clahe", "grey_world", "max_rgb", "minmax"}


# ─── NormConfig ───────────────────────────────────────────────────────────────

@dataclass
class NormConfig:
    """Параметры нормализации цвета.

    Атрибуты:
        method:     Метод: 'gamma' | 'equalize' | 'clahe' |
                    'grey_world' | 'max_rgb' | 'minmax'.
        gamma:      Значение гаммы (> 0, используется при method='gamma').
        clip_limit: Предел контрастности для CLAHE (> 0).
        tile_size:  Размер тайла CLAHE (>= 2, нечётное не требуется).
        target_mean: Целевая яркость для minmax (0–255).
    """

    method: str = "clahe"
    gamma: float = 1.0
    clip_limit: float = 2.0
    tile_size: int = 8
    target_mean: float = 128.0

    def __post_init__(self) -> None:
        if self.method not in _METHODS:
            raise ValueError(
                f"method должен быть одним из {_METHODS}, "
                f"получено '{self.method}'"
            )
        if self.gamma <= 0.0:
            raise ValueError(
                f"gamma должна быть > 0, получено {self.gamma}"
            )
        if self.clip_limit <= 0.0:
            raise ValueError(
                f"clip_limit должен быть > 0, получено {self.clip_limit}"
            )
        if self.tile_size < 2:
            raise ValueError(
                f"tile_size должен быть >= 2, получено {self.tile_size}"
            )
        if not (0.0 <= self.target_mean <= 255.0):
            raise ValueError(
                f"target_mean должен быть в [0, 255], получено {self.target_mean}"
            )


# ─── NormResult ───────────────────────────────────────────────────────────────

@dataclass
class NormResult:
    """Результат нормализации одного изображения.

    Атрибуты:
        image:      Нормализованное изображение (uint8).
        method:     Применённый метод.
        mean_before: Средняя яркость до нормализации.
        mean_after:  Средняя яркость после нормализации.
    """

    image: np.ndarray
    method: str
    mean_before: float
    mean_after: float

    def __post_init__(self) -> None:
        if self.mean_before < 0:
            raise ValueError(
                f"mean_before должна быть >= 0, получено {self.mean_before}"
            )
        if self.mean_after < 0:
            raise ValueError(
                f"mean_after должна быть >= 0, получено {self.mean_after}"
            )

    @property
    def delta_mean(self) -> float:
        """Изменение средней яркости."""
        return self.mean_after - self.mean_before


# ─── Utility ──────────────────────────────────────────────────────────────────

def _mean_brightness(img: np.ndarray) -> float:
    """Средняя яркость пикселей (grayscale или перевод в серый)."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return float(gray.mean())


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


# ─── gamma_correction ─────────────────────────────────────────────────────────

def gamma_correction(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Применить гамма-коррекцию к изображению.

    Аргументы:
        img:   Входное изображение (uint8, 2-D или 3-D).
        gamma: Значение гаммы (> 0).

    Возвращает:
        Изображение uint8 после гамма-коррекции.

    Исключения:
        ValueError: Если gamma <= 0.
    """
    if gamma <= 0.0:
        raise ValueError(f"gamma должна быть > 0, получено {gamma}")
    lut = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(img, lut)


# ─── equalize_histogram ───────────────────────────────────────────────────────

def equalize_histogram(img: np.ndarray) -> np.ndarray:
    """Выровнять гистограмму (глобальная эквализация).

    Для цветных изображений выравнивание применяется к каналу Y (YCrCb).

    Аргументы:
        img: Входное изображение (uint8, 2-D или 3-D BGR).

    Возвращает:
        Изображение uint8.
    """
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


# ─── apply_clahe ──────────────────────────────────────────────────────────────

def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """Применить CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Аргументы:
        img:        Входное изображение (uint8).
        clip_limit: Предел контрастности (> 0).
        tile_size:  Размер тайла (>= 2).

    Возвращает:
        Изображение uint8.

    Исключения:
        ValueError: Если параметры некорректны.
    """
    if clip_limit <= 0.0:
        raise ValueError(f"clip_limit должен быть > 0, получено {clip_limit}")
    if tile_size < 2:
        raise ValueError(f"tile_size должен быть >= 2, получено {tile_size}")
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )
    if img.ndim == 2:
        return clahe.apply(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


# ─── grey_world_balance ───────────────────────────────────────────────────────

def grey_world_balance(img: np.ndarray) -> np.ndarray:
    """Коррекция баланса белого методом Grey World.

    Предполагает, что средний цвет сцены нейтрально-серый.

    Аргументы:
        img: Цветное изображение BGR (uint8).

    Возвращает:
        Изображение BGR uint8 с исправленным балансом.

    Исключения:
        ValueError: Если изображение не трёхканальное.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("grey_world_balance требует трёхканальное (BGR) изображение")
    img_f = img.astype(np.float64)
    mean_per_ch = img_f.mean(axis=(0, 1))  # (B, G, R)
    global_mean = mean_per_ch.mean()
    scale = np.where(mean_per_ch > 1e-6, global_mean / mean_per_ch, 1.0)
    balanced = img_f * scale[np.newaxis, np.newaxis, :]
    return _ensure_uint8(balanced)


# ─── max_rgb_balance ──────────────────────────────────────────────────────────

def max_rgb_balance(img: np.ndarray) -> np.ndarray:
    """Коррекция баланса белого методом Max RGB.

    Масштабирует каналы так, чтобы максимальное значение каждого = 255.

    Аргументы:
        img: Цветное изображение BGR (uint8).

    Возвращает:
        Изображение BGR uint8.

    Исключения:
        ValueError: Если изображение не трёхканальное.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("max_rgb_balance требует трёхканальное (BGR) изображение")
    img_f = img.astype(np.float64)
    ch_max = img_f.max(axis=(0, 1))  # (B, G, R)
    scale = np.where(ch_max > 1e-6, 255.0 / ch_max, 1.0)
    return _ensure_uint8(img_f * scale[np.newaxis, np.newaxis, :])


# ─── minmax_normalize ─────────────────────────────────────────────────────────

def minmax_normalize(img: np.ndarray) -> np.ndarray:
    """Нормализовать диапазон [min, max] → [0, 255].

    Аргументы:
        img: Входное изображение (любой тип).

    Возвращает:
        Изображение uint8.
    """
    img_f = img.astype(np.float64)
    lo, hi = img_f.min(), img_f.max()
    if hi - lo < 1e-10:
        return np.zeros_like(img, dtype=np.uint8)
    return _ensure_uint8((img_f - lo) / (hi - lo) * 255.0)


# ─── normalize_image ──────────────────────────────────────────────────────────

def normalize_image(
    img: np.ndarray,
    cfg: Optional[NormConfig] = None,
) -> NormResult:
    """Применить нормализацию цвета к изображению.

    Аргументы:
        img: Входное изображение (uint8).
        cfg: Параметры (None → NormConfig()).

    Возвращает:
        NormResult.
    """
    if cfg is None:
        cfg = NormConfig()

    mean_before = _mean_brightness(img)

    if cfg.method == "gamma":
        out = gamma_correction(img, cfg.gamma)
    elif cfg.method == "equalize":
        out = equalize_histogram(img)
    elif cfg.method == "clahe":
        out = apply_clahe(img, cfg.clip_limit, cfg.tile_size)
    elif cfg.method == "grey_world":
        out = grey_world_balance(img)
    elif cfg.method == "max_rgb":
        out = max_rgb_balance(img)
    else:  # minmax
        out = minmax_normalize(img)

    return NormResult(
        image=out,
        method=cfg.method,
        mean_before=mean_before,
        mean_after=_mean_brightness(out),
    )


# ─── batch_normalize ──────────────────────────────────────────────────────────

def batch_normalize(
    images: list,
    cfg: Optional[NormConfig] = None,
) -> list:
    """Нормализовать список изображений.

    Аргументы:
        images: Список изображений (uint8).
        cfg:    Параметры (None → NormConfig()).

    Возвращает:
        Список NormResult.
    """
    return [normalize_image(img, cfg) for img in images]
