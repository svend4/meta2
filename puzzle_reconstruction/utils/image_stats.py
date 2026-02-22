"""
Статистический анализ изображений.

Вычисляет разнообразные статистические характеристики изображения:
энтропия, резкость, гистограммные моменты, градиентная структура.

Классы:
    ImageStats — сводные статистики одного изображения

Функции:
    compute_entropy         — шенноновская энтропия гистограммы
    compute_sharpness       — резкость (дисперсия лапласиана)
    compute_histogram_stats — моменты гистограммы (μ, σ, скос, эксцесс)
    compute_gradient_stats  — статистики градиентной карты (Sobel)
    compute_image_stats     — полный набор статистик → ImageStats
    compare_images          — попарное сравнение двух изображений
    batch_stats             — статистики для списка изображений
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np


# ─── ImageStats ───────────────────────────────────────────────────────────────

@dataclass
class ImageStats:
    """
    Сводные статистики одного изображения.

    Attributes:
        mean:        Среднее значение пикселей [0, 255].
        std:         Стандартное отклонение пикселей.
        entropy:     Шенноновская энтропия гистограммы [бит].
        contrast:    RMS-контраст (= std).
        sharpness:   Дисперсия лапласиана (мера резкости).
        histogram:   Нормированная гистограмма яркости (256 бинов, float32).
        percentiles: Словарь {p: значение} для p ∈ {5, 25, 50, 75, 95}.
        n_pixels:    Общее число пикселей.
        extra:       Дополнительные вычисленные поля.
    """
    mean:        float
    std:         float
    entropy:     float
    contrast:    float
    sharpness:   float
    histogram:   np.ndarray
    percentiles: Dict[int, float]
    n_pixels:    int
    extra:       Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"ImageStats(mean={self.mean:.1f}, std={self.std:.1f}, "
                f"entropy={self.entropy:.2f}, sharpness={self.sharpness:.1f}, "
                f"n={self.n_pixels})")


# ─── Вспомогательные ──────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ─── compute_entropy ──────────────────────────────────────────────────────────

def compute_entropy(img: np.ndarray) -> float:
    """
    Шенноновская энтропия гистограммы яркости.

    Args:
        img: BGR или grayscale изображение.

    Returns:
        Энтропия в битах ∈ [0, 8].
    """
    gray = _to_gray(img)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist[hist > 0].astype(np.float64)
    hist /= hist.sum()
    return float(-np.sum(hist * np.log2(hist)))


# ─── compute_sharpness ────────────────────────────────────────────────────────

def compute_sharpness(img: np.ndarray) -> float:
    """
    Резкость изображения через дисперсию лапласиана.

    Чем больше значение, тем резче изображение.

    Args:
        img: BGR или grayscale изображение.

    Returns:
        Неотрицательное вещественное число (дисперсия лапласиана).
    """
    gray = _to_gray(img)
    lap  = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
    return float(lap.var())


# ─── compute_histogram_stats ──────────────────────────────────────────────────

def compute_histogram_stats(img: np.ndarray,
                             bins: int = 256) -> Dict[str, float]:
    """
    Моменты распределения яркости: среднее, σ, скос, эксцесс.

    Args:
        img:  BGR или grayscale изображение.
        bins: Число бинов гистограммы.

    Returns:
        Словарь {'mean', 'std', 'skewness', 'kurtosis'}.
    """
    gray = _to_gray(img).astype(np.float64)
    n    = gray.size
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0}

    mu  = float(gray.mean())
    sig = float(gray.std())

    if sig < 1e-9:
        return {"mean": mu, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0}

    z   = (gray - mu) / sig
    sk  = float((z ** 3).mean())
    ku  = float((z ** 4).mean() - 3.0)   # excess kurtosis

    return {"mean": mu, "std": sig, "skewness": sk, "kurtosis": ku}


# ─── compute_gradient_stats ───────────────────────────────────────────────────

def compute_gradient_stats(img: np.ndarray) -> Dict[str, float]:
    """
    Статистики карты градиентов (Sobel).

    Args:
        img: BGR или grayscale изображение.

    Returns:
        Словарь {'grad_mean', 'grad_std', 'grad_max', 'grad_energy'}.
    """
    gray = _to_gray(img).astype(np.float64)

    gx   = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag  = np.sqrt(gx ** 2 + gy ** 2)

    n    = max(1, mag.size)
    return {
        "grad_mean":   float(mag.mean()),
        "grad_std":    float(mag.std()),
        "grad_max":    float(mag.max()),
        "grad_energy": float((mag ** 2).sum() / n),
    }


# ─── compute_image_stats ──────────────────────────────────────────────────────

def compute_image_stats(img: np.ndarray,
                         hist_bins: int = 256,
                         percentile_levels: Tuple[int, ...] = (5, 25, 50, 75, 95),
                         include_gradient: bool = True) -> ImageStats:
    """
    Полный набор статистик изображения.

    Args:
        img:               BGR или grayscale изображение.
        hist_bins:         Число бинов нормированной гистограммы.
        percentile_levels: Перцентили для вычисления.
        include_gradient:  Включить градиентные статистики в extra.

    Returns:
        ImageStats.
    """
    gray = _to_gray(img)
    h, w = gray.shape
    n_pixels = int(h * w)

    # Базовые моменты
    gray_f = gray.astype(np.float64)
    mu     = float(gray_f.mean())
    sig    = float(gray_f.std())

    # Энтропия
    entropy = compute_entropy(img)

    # Резкость
    sharpness = compute_sharpness(img)

    # Нормированная гистограмма
    hist_raw = cv2.calcHist([gray], [0], None, [hist_bins],
                             [0, 256]).flatten().astype(np.float32)
    s = hist_raw.sum()
    if s > 0:
        hist_raw /= s

    # Перцентили
    flat  = gray.flatten()
    pcts  = {int(p): float(np.percentile(flat, p)) for p in percentile_levels}

    # Дополнительно: градиент
    extra: Dict[str, float] = {}
    if include_gradient:
        extra.update(compute_gradient_stats(img))

    # Гистограммные высшие моменты
    hstats = compute_histogram_stats(img)
    extra["skewness"] = hstats["skewness"]
    extra["kurtosis"] = hstats["kurtosis"]

    return ImageStats(
        mean=mu,
        std=sig,
        entropy=entropy,
        contrast=sig,
        sharpness=sharpness,
        histogram=hist_raw,
        percentiles=pcts,
        n_pixels=n_pixels,
        extra=extra,
    )


# ─── compare_images ───────────────────────────────────────────────────────────

def compare_images(img1: np.ndarray,
                   img2: np.ndarray) -> Dict[str, float]:
    """
    Попарное сравнение двух изображений по ключевым статистикам.

    Args:
        img1: Первое BGR или grayscale изображение.
        img2: Второе BGR или grayscale изображение.

    Returns:
        Словарь с ключами:
            'mean_diff'      — разность средних,
            'std_ratio'      — отношение стандартных отклонений,
            'entropy_diff'   — разность энтропий,
            'sharpness_ratio'— отношение резкостей,
            'hist_corr'      — коэффициент корреляции гистограмм ∈ [-1, 1],
            'hist_bhatt'     — расстояние Бхаттачарья (меньше → ближе).
    """
    s1 = compute_image_stats(img1, include_gradient=False)
    s2 = compute_image_stats(img2, include_gradient=False)

    # Корреляция гистограмм
    h1 = s1.histogram.reshape(-1, 1).astype(np.float32)
    h2 = s2.histogram.reshape(-1, 1).astype(np.float32)
    hist_corr = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))

    # Расстояние Бхаттачарья
    hist_bhatt = float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))

    std1 = s1.std if s1.std > 1e-9 else 1e-9
    sh1  = s1.sharpness if s1.sharpness > 1e-9 else 1e-9

    return {
        "mean_diff":       s1.mean - s2.mean,
        "std_ratio":       s2.std / std1,
        "entropy_diff":    s1.entropy - s2.entropy,
        "sharpness_ratio": s2.sharpness / sh1,
        "hist_corr":       hist_corr,
        "hist_bhatt":      hist_bhatt,
    }


# ─── batch_stats ──────────────────────────────────────────────────────────────

def batch_stats(images: List[np.ndarray],
                hist_bins: int = 256,
                include_gradient: bool = False) -> List[ImageStats]:
    """
    Вычисляет ImageStats для каждого изображения из списка.

    Args:
        images:           Список BGR или grayscale изображений.
        hist_bins:        Число бинов гистограммы.
        include_gradient: Включать градиентные статистики в extra.

    Returns:
        Список ImageStats (по одному на изображение).
    """
    return [
        compute_image_stats(img,
                             hist_bins=hist_bins,
                             include_gradient=include_gradient)
        for img in images
    ]
