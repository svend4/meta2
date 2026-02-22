"""
Утилиты обработки гистограмм изображений.

Экспортирует:
    compute_1d_histogram     — вычисление 1D гистограммы канала
    compute_2d_histogram     — совместная 2D гистограмма двух каналов
    histogram_equalization   — выравнивание гистограммы (global)
    histogram_specification  — спецификация гистограммы (histogram matching)
    earth_mover_distance     — расстояние перемещения земли (Wasserstein-1)
    chi_squared_distance     — χ²-расстояние между двумя гистограммами
    histogram_intersection   — пересечение гистограмм (сходство)
    backproject              — обратное проецирование гистограммы на изображение
    joint_histogram          — совместная гистограмма двух изображений (N каналов)
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


# ─── Публичные функции ────────────────────────────────────────────────────────

def compute_1d_histogram(
    img: np.ndarray,
    channel: int = 0,
    n_bins: int = 256,
    value_range: Tuple[float, float] = (0.0, 256.0),
    normalize: bool = True,
) -> np.ndarray:
    """Вычислить 1D гистограмму заданного канала изображения.

    Args:
        img:         Изображение (H, W) или (H, W, C) uint8.
        channel:     Индекс канала (для одноканального изображения используется 0).
        n_bins:      Число корзин (≥ 1).
        value_range: Диапазон значений (min, max).
        normalize:   Если ``True``, делит на сумму (L1-нормализация).

    Returns:
        Массив float32 формы (n_bins,).

    Raises:
        ValueError: Если ``n_bins`` < 1 или ``channel`` выходит за пределы.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if img.ndim == 2:
        plane = img
    elif img.ndim == 3:
        if channel >= img.shape[2]:
            raise ValueError(
                f"channel {channel} out of range for image with {img.shape[2]} channels"
            )
        plane = img[:, :, channel]
    else:
        raise ValueError(f"Expected 2D or 3D image, got ndim={img.ndim}")

    hist = cv2.calcHist(
        [plane.astype(np.uint8)], [0], None,
        [n_bins], list(value_range),
    ).flatten().astype(np.float32)

    if normalize:
        total = hist.sum()
        if total > 0:
            hist = hist / total
    return hist


def compute_2d_histogram(
    img: np.ndarray,
    channel1: int = 0,
    channel2: int = 1,
    n_bins: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """Вычислить совместную 2D гистограмму двух каналов изображения.

    Args:
        img:      Изображение (H, W, C) с C ≥ 2.
        channel1: Индекс первого канала.
        channel2: Индекс второго канала.
        n_bins:   Число корзин по каждому измерению (≥ 2).
        normalize: L1-нормализация.

    Returns:
        Матрица float32 формы (n_bins, n_bins).

    Raises:
        ValueError: Если изображение одноканальное или индексы выходят за пределы.
    """
    if img.ndim < 3:
        raise ValueError("compute_2d_histogram requires a multi-channel image")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    n_ch = img.shape[2]
    for ch in (channel1, channel2):
        if ch >= n_ch:
            raise ValueError(f"channel {ch} out of range for {n_ch}-channel image")

    p1 = img[:, :, channel1].astype(np.uint8)
    p2 = img[:, :, channel2].astype(np.uint8)
    hist = cv2.calcHist(
        [p1, p2], [0, 1], None,
        [n_bins, n_bins], [0, 256, 0, 256],
    ).astype(np.float32)

    if normalize:
        total = hist.sum()
        if total > 0:
            hist = hist / total
    return hist


def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """Глобальное выравнивание гистограммы одноканального изображения.

    Args:
        img: Изображение (H, W) uint8.

    Returns:
        Выровненное изображение (H, W) uint8.

    Raises:
        ValueError: Если изображение многоканальное.
    """
    if img.ndim != 2:
        raise ValueError(
            f"histogram_equalization expects a 2D grayscale image, got ndim={img.ndim}"
        )
    return cv2.equalizeHist(img.astype(np.uint8))


def histogram_specification(
    src: np.ndarray,
    ref: np.ndarray,
) -> np.ndarray:
    """Спецификация гистограммы: привести гистограмму ``src`` к гистограмме ``ref``.

    Args:
        src: Исходное изображение (H, W) uint8.
        ref: Референсное изображение (H', W') uint8.

    Returns:
        Преобразованное изображение (H, W) uint8.

    Raises:
        ValueError: Если изображения многоканальные.
    """
    for name, arr in (("src", src), ("ref", ref)):
        if arr.ndim != 2:
            raise ValueError(
                f"histogram_specification: {name!r} must be 2D, got ndim={arr.ndim}"
            )
    # Кумулятивные функции распределения (CDF)
    src_hist, _ = np.histogram(src.ravel(), bins=256, range=(0, 256))
    ref_hist, _ = np.histogram(ref.ravel(), bins=256, range=(0, 256))
    src_cdf = src_hist.cumsum().astype(np.float64)
    ref_cdf = ref_hist.cumsum().astype(np.float64)
    src_cdf = src_cdf / (src_cdf[-1] + 1e-12)
    ref_cdf = ref_cdf / (ref_cdf[-1] + 1e-12)

    # Поиск ближайшего значения в ref_cdf для каждого уровня src_cdf
    lut = np.zeros(256, dtype=np.uint8)
    j = 0
    for i in range(256):
        while j < 255 and ref_cdf[j] < src_cdf[i]:
            j += 1
        lut[i] = j
    return lut[src.astype(np.uint8)]


def earth_mover_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Расстояние перемещения земли (Wasserstein-1) между двумя гистограммами.

    Оба вектора нормализуются к единичной сумме перед вычислением.

    Args:
        hist1: Гистограмма (N,) ≥ 0.
        hist2: Гистограмма (N,) ≥ 0 той же длины.

    Returns:
        Неотрицательное расстояние EMD.

    Raises:
        ValueError: Если длины гистограмм не совпадают или длина < 1.
    """
    a = np.asarray(hist1, dtype=np.float64).ravel()
    b = np.asarray(hist2, dtype=np.float64).ravel()
    if len(a) != len(b):
        raise ValueError(
            f"Histograms must have equal length, got {len(a)} and {len(b)}"
        )
    if len(a) < 1:
        raise ValueError("Histograms must not be empty")
    sa = a.sum()
    sb = b.sum()
    na = a / sa if sa > 1e-12 else a
    nb = b / sb if sb > 1e-12 else b
    return float(np.abs(np.cumsum(na) - np.cumsum(nb)).sum())


def chi_squared_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """χ²-расстояние между двумя гистограммами.

    d(h1, h2) = 0.5 × Σ (h1ᵢ − h2ᵢ)² / (h1ᵢ + h2ᵢ + ε)

    Args:
        hist1: Гистограмма (N,) ≥ 0.
        hist2: Гистограмма (N,) ≥ 0 той же длины.

    Returns:
        Неотрицательное расстояние χ².

    Raises:
        ValueError: Если длины гистограмм не совпадают.
    """
    a = np.asarray(hist1, dtype=np.float64).ravel()
    b = np.asarray(hist2, dtype=np.float64).ravel()
    if len(a) != len(b):
        raise ValueError(
            f"Histograms must have equal length, got {len(a)} and {len(b)}"
        )
    denom = a + b + 1e-12
    return float(0.5 * np.sum((a - b) ** 2 / denom))


def histogram_intersection(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Сходство гистограмм через пересечение (Swain & Ballard, 1991).

    s(h1, h2) = Σ min(h1ᵢ, h2ᵢ) / Σ h2ᵢ

    Args:
        hist1: Гистограмма (N,) ≥ 0.
        hist2: Гистограмма (N,) ≥ 0 той же длины.

    Returns:
        Значение в диапазоне [0, 1].

    Raises:
        ValueError: Если длины гистограмм не совпадают.
    """
    a = np.asarray(hist1, dtype=np.float64).ravel()
    b = np.asarray(hist2, dtype=np.float64).ravel()
    if len(a) != len(b):
        raise ValueError(
            f"Histograms must have equal length, got {len(a)} and {len(b)}"
        )
    total_b = b.sum()
    if total_b < 1e-12:
        return 0.0
    return float(np.minimum(a, b).sum() / total_b)


def backproject(
    img: np.ndarray,
    model_hist: np.ndarray,
    n_bins: int = 256,
    channel: int = 0,
) -> np.ndarray:
    """Обратное проецирование гистограммы на изображение.

    Заменяет каждый пиксель (по значению канала) вероятностью из ``model_hist``.

    Args:
        img:        Изображение (H, W) или (H, W, C) uint8.
        model_hist: Нормализованная гистограмма (n_bins,).
        n_bins:     Число корзин.
        channel:    Канал для одноканальной выборки.

    Returns:
        Карта вероятностей (H, W) float32, значения [0, 1].

    Raises:
        ValueError: Если длина ``model_hist`` не равна ``n_bins``.
    """
    hist = np.asarray(model_hist, dtype=np.float32).ravel()
    if len(hist) != n_bins:
        raise ValueError(
            f"model_hist length {len(hist)} does not match n_bins={n_bins}"
        )
    if img.ndim == 2:
        plane = img.astype(np.uint8)
    else:
        plane = img[:, :, channel].astype(np.uint8)

    # LUT: bin index for each 0..255 value
    bin_size = 256.0 / n_bins
    lut_idx = np.clip(
        (np.arange(256, dtype=np.float32) / bin_size).astype(np.int32),
        0, n_bins - 1,
    )
    lut_vals = hist[lut_idx].astype(np.float32)
    return lut_vals[plane]


def joint_histogram(
    img1: np.ndarray,
    img2: np.ndarray,
    n_bins: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """Совместная гистограмма двух одноканальных изображений.

    Изображения должны иметь одинаковые пространственные размеры.

    Args:
        img1:      Первое изображение (H, W) uint8.
        img2:      Второе изображение (H, W) uint8.
        n_bins:    Число корзин по каждому измерению (≥ 2).
        normalize: L1-нормализация.

    Returns:
        Матрица float32 (n_bins, n_bins).

    Raises:
        ValueError: Если размеры изображений не совпадают или ``n_bins`` < 2.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    a = img1 if img1.ndim == 2 else cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    b = img2 if img2.ndim == 2 else cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    if a.shape != b.shape:
        raise ValueError(
            f"Images must have the same shape, got {a.shape} and {b.shape}"
        )
    hist = cv2.calcHist(
        [a.astype(np.uint8), b.astype(np.uint8)], [0, 1], None,
        [n_bins, n_bins], [0, 256, 0, 256],
    ).astype(np.float32)
    if normalize:
        total = hist.sum()
        if total > 0:
            hist = hist / total
    return hist
