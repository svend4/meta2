"""
Оценка качества изображений-фрагментов.

Анализирует резкость, контраст, уровень шума и полноту содержимого
для принятия решений о пригодности фрагментов к дальнейшей обработке.

Классы:
    QualityReport — сводный отчёт о качестве одного изображения

Функции:
    estimate_blur          — оценка размытости (дисперсия Лапласиана)
    estimate_noise         — оценка уровня шума (высокочастотная энергия)
    estimate_contrast      — оценка контраста (нормированный IQR яркости)
    estimate_completeness  — доля непустых пикселей
    assess_quality         — полный анализ одного изображения
    filter_by_quality      — разделение списка на good / rejected
    batch_assess_quality   — пакетный анализ списка изображений
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── QualityReport ────────────────────────────────────────────────────────────

@dataclass
class QualityReport:
    """
    Сводный отчёт о качестве изображения-фрагмента.

    Все оценки ∈ [0, 1], где 1 — наилучшее качество.

    Attributes:
        blur_score:        1 = чёткое, 0 = сильно размытое.
        noise_score:       1 = чистое, 0 = очень зашумлённое.
        contrast_score:    1 = высокий контраст, 0 = плоское изображение.
        completeness:      Доля непустых пикселей ∈ [0, 1].
        overall:           Взвешенное среднее четырёх метрик.
        is_acceptable:     True, если overall ≥ min_score.
        params:            Параметры анализа.
    """
    blur_score:     float
    noise_score:    float
    contrast_score: float
    completeness:   float
    overall:        float
    is_acceptable:  bool
    params:         Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"QualityReport(overall={self.overall:.3f}, "
                f"acceptable={self.is_acceptable}, "
                f"blur={self.blur_score:.3f}, "
                f"noise={self.noise_score:.3f}, "
                f"contrast={self.contrast_score:.3f})")


# ─── _to_gray ─────────────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ─── estimate_blur ────────────────────────────────────────────────────────────

def estimate_blur(
    img:       np.ndarray,
    max_var:   float = 500.0,
) -> float:
    """
    Оценивает чёткость изображения через дисперсию Лапласиана.

    score = min(laplacian_var / max_var, 1.0).

    Args:
        img:     Grayscale или BGR изображение.
        max_var: Нормировочная константа (дисперсия «идеально чёткого» изображения).

    Returns:
        Оценка чёткости ∈ [0, 1].
    """
    gray = _to_gray(img).astype(np.float64)
    lap  = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
    var  = float(lap.var())
    return float(np.clip(var / max(max_var, 1e-9), 0.0, 1.0))


# ─── estimate_noise ───────────────────────────────────────────────────────────

def estimate_noise(
    img:       np.ndarray,
    max_sigma: float = 30.0,
) -> float:
    """
    Оценивает уровень шума через стандартное отклонение
    высокочастотных компонент (разность между изображением и его сглаженной
    версией).

    noise_score = 1 - min(sigma / max_sigma, 1.0).

    Args:
        img:       Grayscale или BGR изображение.
        max_sigma: Уровень шума, соответствующий оценке 0.

    Returns:
        Оценка чистоты ∈ [0, 1] (1 = мало шума).
    """
    gray      = _to_gray(img).astype(np.float32)
    blurred   = cv2.GaussianBlur(gray, (5, 5), 0)
    residual  = gray.astype(np.float64) - blurred.astype(np.float64)
    sigma     = float(residual.std())
    noisy_frac = np.clip(sigma / max(max_sigma, 1e-9), 0.0, 1.0)
    return float(1.0 - noisy_frac)


# ─── estimate_contrast ────────────────────────────────────────────────────────

def estimate_contrast(
    img: np.ndarray,
) -> float:
    """
    Оценивает контраст через нормированный IQR (межквартильный размах).

    contrast = (p75 - p25) / 255.

    Args:
        img: Grayscale или BGR изображение.

    Returns:
        Оценка контраста ∈ [0, 1].
    """
    gray = _to_gray(img).astype(np.float64)
    p25  = float(np.percentile(gray, 25))
    p75  = float(np.percentile(gray, 75))
    iqr  = p75 - p25
    return float(np.clip(iqr / 255.0, 0.0, 1.0))


# ─── estimate_completeness ────────────────────────────────────────────────────

def estimate_completeness(
    img:          np.ndarray,
    bg_threshold: int = 240,
) -> float:
    """
    Оценивает долю «непустых» (не фоновых) пикселей.

    Пиксели с яркостью ≥ bg_threshold считаются фоном.

    Args:
        img:          Grayscale или BGR изображение.
        bg_threshold: Порог яркости фона (0–255).

    Returns:
        Доля непустых пикселей ∈ [0, 1].
    """
    gray  = _to_gray(img)
    total = gray.size
    if total == 0:
        return 0.0
    non_bg = int(np.count_nonzero(gray < bg_threshold))
    return float(non_bg) / float(total)


# ─── assess_quality ───────────────────────────────────────────────────────────

def assess_quality(
    img:          np.ndarray,
    min_score:    float = 0.4,
    bg_threshold: int   = 240,
    max_blur_var: float = 500.0,
    max_noise:    float = 30.0,
    weights:      Optional[Tuple[float, float, float, float]] = None,
) -> QualityReport:
    """
    Проводит полный анализ качества одного изображения.

    Args:
        img:          Grayscale или BGR изображение uint8.
        min_score:    Порог overall score для is_acceptable.
        bg_threshold: Порог яркости фона для estimate_completeness.
        max_blur_var: max_var для estimate_blur.
        max_noise:    max_sigma для estimate_noise.
        weights:      (w_blur, w_noise, w_contrast, w_completeness).
                      None → равные.

    Returns:
        QualityReport с полными метриками.
    """
    b_score = estimate_blur(img, max_var=max_blur_var)
    n_score = estimate_noise(img, max_sigma=max_noise)
    c_score = estimate_contrast(img)
    comp    = estimate_completeness(img, bg_threshold=bg_threshold)

    if weights is None:
        wb, wn, wc, wcomp = 0.25, 0.25, 0.25, 0.25
    else:
        wb, wn, wc, wcomp = weights
        s = wb + wn + wc + wcomp
        if s > 1e-9:
            wb /= s; wn /= s; wc /= s; wcomp /= s

    overall = float(wb * b_score + wn * n_score + wc * c_score + wcomp * comp)
    overall = float(np.clip(overall, 0.0, 1.0))

    return QualityReport(
        blur_score=b_score,
        noise_score=n_score,
        contrast_score=c_score,
        completeness=comp,
        overall=overall,
        is_acceptable=(overall >= min_score),
        params={
            "min_score":    min_score,
            "bg_threshold": bg_threshold,
            "max_blur_var": max_blur_var,
            "max_noise":    max_noise,
            "weights":      weights,
        },
    )


# ─── filter_by_quality ────────────────────────────────────────────────────────

def filter_by_quality(
    images:    List[np.ndarray],
    min_score: float = 0.4,
    **kwargs,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Разделяет изображения на хорошие (overall ≥ min_score) и отклонённые.

    Args:
        images:    Список изображений.
        min_score: Порог overall score.
        **kwargs:  Параметры assess_quality.

    Returns:
        (good, rejected) — два списка изображений.
    """
    good: List[np.ndarray]     = []
    rejected: List[np.ndarray] = []
    for img in images:
        report = assess_quality(img, min_score=min_score, **kwargs)
        (good if report.is_acceptable else rejected).append(img)
    return good, rejected


# ─── batch_assess_quality ─────────────────────────────────────────────────────

def batch_assess_quality(
    images: List[np.ndarray],
    **kwargs,
) -> List[QualityReport]:
    """
    Проводит анализ качества для каждого изображения в списке.

    Args:
        images:  Список изображений.
        **kwargs: Параметры assess_quality.

    Returns:
        Список QualityReport той же длины.
    """
    return [assess_quality(img, **kwargs) for img in images]
