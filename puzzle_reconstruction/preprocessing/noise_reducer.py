"""
Шумоподавление для изображений фрагментов документа.

Применяет различные фильтры к изображениям фрагментов, чтобы снизить
артефакты сканирования (зернистость, JPEG-блоки, полосы) перед
последующим сравнением краёв и OCR.

Классы:
    NoiseReductionResult — результат шумоподавления одного изображения

Функции:
    estimate_noise     — оценка уровня шума (σ по методу Лаплас-среднего)
    gaussian_reduce    — Гауссов фильтр
    median_reduce      — медианный фильтр
    bilateral_reduce   — двусторонний фильтр (edge-preserving)
    auto_reduce        — автовыбор фильтра по оценке уровня шума
    batch_reduce       — пакетное шумоподавление списка изображений
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np


# ─── NoiseReductionResult ─────────────────────────────────────────────────────

@dataclass
class NoiseReductionResult:
    """
    Результат шумоподавления одного изображения.

    Attributes:
        filtered:              Обработанное изображение (uint8).
        method:                Применённый метод ('gaussian', 'median',
                               'bilateral', 'auto').
        params:                Параметры фильтра.
        noise_estimate_before: Оценка σ шума до фильтрации (≥ 0).
        noise_estimate_after:  Оценка σ шума после фильтрации (≥ 0).
    """
    filtered:              np.ndarray
    method:                str
    params:                Dict = field(default_factory=dict)
    noise_estimate_before: float = 0.0
    noise_estimate_after:  float = 0.0

    def __repr__(self) -> str:
        h, w = self.filtered.shape[:2]
        return (f"NoiseReductionResult(method={self.method!r}, "
                f"shape=({h},{w}), "
                f"σ_before={self.noise_estimate_before:.2f}, "
                f"σ_after={self.noise_estimate_after:.2f})")


# ─── estimate_noise ───────────────────────────────────────────────────────────

def estimate_noise(img: np.ndarray) -> float:
    """
    Оценивает уровень шума изображения (σ).

    Использует лапласиан изображения: σ ≈ std(Laplacian(gray)) / √2.
    Работает для серых и BGR изображений.

    Args:
        img: BGR или grayscale изображение uint8.

    Returns:
        Оценка σ ≥ 0 (float).
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap  = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return float(lap.std() / np.sqrt(2.0))


# ─── gaussian_reduce ──────────────────────────────────────────────────────────

def gaussian_reduce(img:   np.ndarray,
                     ksize: int = 5,
                     sigma: float = 0.0) -> NoiseReductionResult:
    """
    Применяет Гауссов фильтр к изображению.

    Args:
        img:   BGR или grayscale изображение uint8.
        ksize: Размер ядра (нечётное, ≥ 1).
        sigma: Стандартное отклонение (0 → автовычисление по ksize).

    Returns:
        NoiseReductionResult с method='gaussian'.
    """
    ksize = max(1, ksize | 1)   # гарантируем нечётность
    noise_before = estimate_noise(img)
    filtered     = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    noise_after  = estimate_noise(filtered)
    return NoiseReductionResult(
        filtered=filtered,
        method="gaussian",
        params={"ksize": ksize, "sigma": sigma},
        noise_estimate_before=noise_before,
        noise_estimate_after=noise_after,
    )


# ─── median_reduce ────────────────────────────────────────────────────────────

def median_reduce(img:   np.ndarray,
                   ksize: int = 5) -> NoiseReductionResult:
    """
    Применяет медианный фильтр к изображению.

    Хорошо устраняет импульсный (salt-and-pepper) шум.

    Args:
        img:   BGR или grayscale изображение uint8.
        ksize: Размер ядра (нечётное, ≥ 1).

    Returns:
        NoiseReductionResult с method='median'.
    """
    ksize = max(1, ksize | 1)
    noise_before = estimate_noise(img)
    filtered     = cv2.medianBlur(img, ksize)
    noise_after  = estimate_noise(filtered)
    return NoiseReductionResult(
        filtered=filtered,
        method="median",
        params={"ksize": ksize},
        noise_estimate_before=noise_before,
        noise_estimate_after=noise_after,
    )


# ─── bilateral_reduce ─────────────────────────────────────────────────────────

def bilateral_reduce(img:         np.ndarray,
                      d:           int   = 9,
                      sigma_color: float = 75.0,
                      sigma_space: float = 75.0) -> NoiseReductionResult:
    """
    Применяет двусторонний (bilateral) фильтр.

    Сохраняет края при сглаживании текстур. Оптимален для
    фотографий и сканов с резкими границами фрагментов.

    Args:
        img:         BGR или grayscale изображение uint8.
        d:           Диаметр пикселевой окрестности (≥ 1).
        sigma_color: Диапазон в пространстве цветов.
        sigma_space: Диапазон в координатном пространстве.

    Returns:
        NoiseReductionResult с method='bilateral'.
    """
    d = max(1, d)
    noise_before = estimate_noise(img)
    filtered     = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    noise_after  = estimate_noise(filtered)
    return NoiseReductionResult(
        filtered=filtered,
        method="bilateral",
        params={"d": d, "sigma_color": sigma_color, "sigma_space": sigma_space},
        noise_estimate_before=noise_before,
        noise_estimate_after=noise_after,
    )


# ─── auto_reduce ──────────────────────────────────────────────────────────────

def auto_reduce(img:         np.ndarray,
                 low_thresh:  float = 5.0,
                 high_thresh: float = 20.0,
                 **kwargs) -> NoiseReductionResult:
    """
    Автоматически выбирает фильтр по оценке уровня шума.

    Стратегия:
        σ < low_thresh  → без обработки (Gaussian с ksize=1)
        σ ∈ [low, high) → bilateral (edge-preserving)
        σ ≥ high_thresh → median (импульсный шум)

    Args:
        img:         BGR или grayscale изображение uint8.
        low_thresh:  Нижний порог σ для выбора фильтра.
        high_thresh: Верхний порог σ.
        **kwargs:    Дополнительные параметры для конкретного фильтра.

    Returns:
        NoiseReductionResult с method='auto' и вложенным методом в params.
    """
    sigma = estimate_noise(img)

    if sigma < low_thresh:
        sub = gaussian_reduce(img, ksize=1, **{k: v for k, v in kwargs.items()
                                                if k in ("sigma",)})
        chosen = "gaussian_trivial"
    elif sigma < high_thresh:
        sub = bilateral_reduce(img, **{k: v for k, v in kwargs.items()
                                        if k in ("d", "sigma_color", "sigma_space")})
        chosen = "bilateral"
    else:
        sub = median_reduce(img, **{k: v for k, v in kwargs.items()
                                     if k in ("ksize",)})
        chosen = "median"

    params = dict(sub.params)
    params["chosen_filter"] = chosen
    params["sigma_before"]  = sigma
    params["low_thresh"]    = low_thresh
    params["high_thresh"]   = high_thresh

    return NoiseReductionResult(
        filtered=sub.filtered,
        method="auto",
        params=params,
        noise_estimate_before=sub.noise_estimate_before,
        noise_estimate_after=sub.noise_estimate_after,
    )


# ─── batch_reduce ─────────────────────────────────────────────────────────────

def batch_reduce(images: List[np.ndarray],
                  method: str = "gaussian",
                  **kwargs) -> List[NoiseReductionResult]:
    """
    Применяет шумоподавление ко всем изображениям в списке.

    Args:
        images: Список BGR или grayscale изображений uint8.
        method: 'gaussian' | 'median' | 'bilateral' | 'auto'.
        **kwargs: Параметры соответствующего фильтра.

    Returns:
        Список NoiseReductionResult той же длины.

    Raises:
        ValueError: Неизвестный метод.
    """
    _dispatch = {
        "gaussian": gaussian_reduce,
        "median":   median_reduce,
        "bilateral": bilateral_reduce,
        "auto":     auto_reduce,
    }
    if method not in _dispatch:
        raise ValueError(
            f"Unknown noise reduction method {method!r}. "
            f"Choose from: {list(_dispatch)}."
        )
    fn = _dispatch[method]
    return [fn(img, **kwargs) for img in images]
