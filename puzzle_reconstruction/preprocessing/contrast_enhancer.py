"""Улучшение контрастности изображений фрагментов.

Модуль предоставляет несколько стратегий улучшения контраста:
гистограммная эквализация, CLAHE-подобное растяжение, гамма-коррекция
и адаптивное линейное масштабирование.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ─── EnhanceConfig ────────────────────────────────────────────────────────────

@dataclass
class EnhanceConfig:
    """Параметры улучшения контраста.

    Атрибуты:
        method:     Метод: ``"equalize"``, ``"stretch"``, ``"gamma"``,
                    ``"clahe"``.
        gamma:      Значение гаммы (> 0, используется при method="gamma").
        clip_limit: Верхний порог гистограммы [0.0, 1.0] для CLAHE/stretch.
        tile_size:  Размер тайла для CLAHE (>= 1).
        output_range: Диапазон выходных значений (min, max), min < max.
    """

    method: str = "equalize"
    gamma: float = 1.0
    clip_limit: float = 0.03
    tile_size: int = 8
    output_range: Tuple[float, float] = (0.0, 255.0)

    def __post_init__(self) -> None:
        valid_methods = {"equalize", "stretch", "gamma", "clahe"}
        if self.method not in valid_methods:
            raise ValueError(
                f"method должен быть одним из {valid_methods}, "
                f"получено '{self.method}'"
            )
        if self.gamma <= 0:
            raise ValueError(
                f"gamma должна быть > 0, получено {self.gamma}"
            )
        if not (0.0 <= self.clip_limit <= 1.0):
            raise ValueError(
                f"clip_limit должен быть в [0, 1], получено {self.clip_limit}"
            )
        if self.tile_size < 1:
            raise ValueError(
                f"tile_size должен быть >= 1, получено {self.tile_size}"
            )
        lo, hi = self.output_range
        if lo >= hi:
            raise ValueError(
                f"output_range: min ({lo}) должен быть < max ({hi})"
            )


# ─── EnhanceResult ────────────────────────────────────────────────────────────

@dataclass
class EnhanceResult:
    """Результат улучшения контраста.

    Атрибуты:
        image:       Изображение после обработки.
        method:      Применённый метод.
        input_mean:  Среднее значение до обработки.
        output_mean: Среднее значение после обработки.
        input_std:   Стандартное отклонение до обработки.
        output_std:  Стандартное отклонение после обработки.
    """

    image: np.ndarray
    method: str
    input_mean: float
    output_mean: float
    input_std: float
    output_std: float

    @property
    def contrast_gain(self) -> float:
        """Отношение output_std / input_std (0 если input_std == 0)."""
        if self.input_std < 1e-12:
            return 0.0
        return float(self.output_std / self.input_std)

    @property
    def mean_shift(self) -> float:
        """Изменение среднего: output_mean - input_mean."""
        return float(self.output_mean - self.input_mean)


# ─── _normalize ───────────────────────────────────────────────────────────────

def _normalize(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Нормировать массив в диапазон [lo, hi]."""
    a_min, a_max = float(arr.min()), float(arr.max())
    if a_max - a_min < 1e-12:
        return np.full_like(arr, (lo + hi) / 2.0, dtype=float)
    out = (arr.astype(float) - a_min) / (a_max - a_min)
    return out * (hi - lo) + lo


# ─── equalize_histogram ───────────────────────────────────────────────────────

def equalize_histogram(
    image: np.ndarray,
    cfg: Optional[EnhanceConfig] = None,
) -> np.ndarray:
    """Эквализация гистограммы (глобальная).

    Аргументы:
        image: 2D или 3D uint8/float массив.
        cfg:   Параметры.

    Возвращает:
        Изображение с эквализированной гистограммой.
    """
    if cfg is None:
        cfg = EnhanceConfig(method="equalize")

    lo, hi = cfg.output_range
    n_bins = 256

    if image.ndim == 3:
        channels = [equalize_histogram(image[:, :, c], cfg)
                    for c in range(image.shape[2])]
        return np.stack(channels, axis=2).astype(image.dtype)

    flat = image.astype(float).ravel()
    hist, bin_edges = np.histogram(flat, bins=n_bins,
                                   range=(float(flat.min()), float(flat.max())))
    cdf = hist.cumsum().astype(float)
    cdf_min = cdf[cdf > 0][0] if (cdf > 0).any() else 0.0
    total = flat.size

    # CDF нормализованный
    if total - cdf_min < 1e-12:
        equalized = np.full_like(flat, flat[0] if len(flat) > 0 else (lo + hi) / 2.0)
    else:
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        cdf_norm = (cdf - cdf_min) / (total - cdf_min)
        equalized = np.interp(flat, bin_centers, cdf_norm * (hi - lo) + lo)

    return equalized.reshape(image.shape).astype(image.dtype)


# ─── stretch_contrast ─────────────────────────────────────────────────────────

def stretch_contrast(
    image: np.ndarray,
    cfg: Optional[EnhanceConfig] = None,
) -> np.ndarray:
    """Линейное растяжение контраста с отсечением хвостов.

    Аргументы:
        image: Массив изображения.
        cfg:   Параметры (clip_limit задаёт долю хвоста [0, 0.5]).

    Возвращает:
        Изображение с растянутым контрастом.
    """
    if cfg is None:
        cfg = EnhanceConfig(method="stretch")

    lo, hi = cfg.output_range
    clip = float(np.clip(cfg.clip_limit, 0.0, 0.5))

    if image.ndim == 3:
        channels = [stretch_contrast(image[:, :, c], cfg)
                    for c in range(image.shape[2])]
        return np.stack(channels, axis=2).astype(image.dtype)

    flat = image.astype(float).ravel()
    p_lo = float(np.percentile(flat, clip * 100.0))
    p_hi = float(np.percentile(flat, (1.0 - clip) * 100.0))

    if p_hi - p_lo < 1e-12:
        return np.full_like(image, (lo + hi) / 2.0, dtype=image.dtype)

    stretched = np.clip((image.astype(float) - p_lo) / (p_hi - p_lo), 0.0, 1.0)
    return (stretched * (hi - lo) + lo).astype(image.dtype)


# ─── apply_gamma ──────────────────────────────────────────────────────────────

def apply_gamma(
    image: np.ndarray,
    cfg: Optional[EnhanceConfig] = None,
) -> np.ndarray:
    """Гамма-коррекция.

    Аргументы:
        image: Массив изображения (значения в output_range).
        cfg:   Параметры.

    Возвращает:
        Изображение после гамма-коррекции.
    """
    if cfg is None:
        cfg = EnhanceConfig(method="gamma")

    lo, hi = cfg.output_range
    span = hi - lo

    norm = np.clip((image.astype(float) - lo) / span, 0.0, 1.0)
    corrected = np.power(norm, cfg.gamma)
    return (corrected * span + lo).astype(image.dtype)


# ─── clahe_enhance ────────────────────────────────────────────────────────────

def clahe_enhance(
    image: np.ndarray,
    cfg: Optional[EnhanceConfig] = None,
) -> np.ndarray:
    """Тайловая эквализация с ограничением контраста (упрощённый CLAHE).

    Аргументы:
        image: 2D массив.
        cfg:   Параметры.

    Возвращает:
        Изображение после CLAHE-обработки.
    """
    if cfg is None:
        cfg = EnhanceConfig(method="clahe")

    lo, hi = cfg.output_range

    if image.ndim == 3:
        channels = [clahe_enhance(image[:, :, c], cfg)
                    for c in range(image.shape[2])]
        return np.stack(channels, axis=2).astype(image.dtype)

    h, w = image.shape
    ts = cfg.tile_size
    out = np.zeros_like(image, dtype=float)

    for row in range(0, h, ts):
        for col in range(0, w, ts):
            tile = image[row:row + ts, col:col + ts].astype(float)
            # Ограничить гистограмму
            if tile.max() - tile.min() < 1e-12:
                out[row:row + ts, col:col + ts] = (lo + hi) / 2.0
                continue
            scaled = _normalize(tile, lo, hi)
            out[row:row + ts, col:col + ts] = scaled

    return np.clip(out, lo, hi).astype(image.dtype)


# ─── enhance_contrast ─────────────────────────────────────────────────────────

def enhance_contrast(
    image: np.ndarray,
    cfg: Optional[EnhanceConfig] = None,
) -> EnhanceResult:
    """Улучшить контраст изображения выбранным методом.

    Аргументы:
        image: Массив изображения (2D или 3D).
        cfg:   Параметры.

    Возвращает:
        EnhanceResult.

    Исключения:
        ValueError: если image не 2D/3D.
    """
    if image.ndim not in (2, 3):
        raise ValueError(
            f"image должен быть 2D или 3D, получено {image.ndim}D"
        )
    if cfg is None:
        cfg = EnhanceConfig()

    flat_in = image.astype(float).ravel()
    input_mean = float(np.mean(flat_in))
    input_std = float(np.std(flat_in))

    dispatch = {
        "equalize": equalize_histogram,
        "stretch": stretch_contrast,
        "gamma": apply_gamma,
        "clahe": clahe_enhance,
    }
    fn = dispatch[cfg.method]
    enhanced = fn(image, cfg)

    flat_out = enhanced.astype(float).ravel()
    return EnhanceResult(
        image=enhanced,
        method=cfg.method,
        input_mean=input_mean,
        output_mean=float(np.mean(flat_out)),
        input_std=input_std,
        output_std=float(np.std(flat_out)),
    )


# ─── batch_enhance ────────────────────────────────────────────────────────────

def batch_enhance(
    images: List[np.ndarray],
    cfg: Optional[EnhanceConfig] = None,
) -> List[EnhanceResult]:
    """Улучшить контраст набора изображений.

    Аргументы:
        images: Список изображений.
        cfg:    Параметры.

    Возвращает:
        Список EnhanceResult.
    """
    return [enhance_contrast(img, cfg) for img in images]
