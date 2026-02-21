"""
Анализ и оценка уровня шума в фрагментах документа.

Оценивает уровень шума, SNR, JPEG-артефакты и зернистость для последующей
фильтрации, контроля качества и адаптивного выбора параметров обработки.

Классы:
    NoiseAnalysisResult — результат анализа шума

Функции:
    estimate_noise_sigma  — оценка σ шума методом лапласиана
    estimate_snr          — отношение сигнал/шум (дБ)
    detect_jpeg_artifacts — индекс JPEG-артефактов по DCT-блокам
    estimate_grain        — зернистость по локальной дисперсии
    analyze_noise         — полный анализ с квалификацией качества
    batch_analyze_noise   — пакетная обработка
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np


# ─── NoiseAnalysisResult ──────────────────────────────────────────────────────

@dataclass
class NoiseAnalysisResult:
    """
    Результат анализа шума изображения.

    Attributes:
        noise_level:    Оценка стандартного отклонения шума σ (шкала 0..255).
        snr_db:         Отношение сигнал/шум в дБ. Может быть inf для константы.
        jpeg_artifacts: Индекс JPEG-артефактов ∈ [0,1] (0 — нет, 1 — сильные).
        grain_level:    Зернистость ∈ [0,1] (локальная дисперсия блоков).
        quality:        Качественная оценка: 'clean' | 'noisy' | 'very_noisy'.
        params:         Параметры анализа.
    """
    noise_level:    float
    snr_db:         float
    jpeg_artifacts: float
    grain_level:    float
    quality:        str
    params:         Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"NoiseAnalysisResult(σ={self.noise_level:.2f}, "
                f"snr={self.snr_db:.1f}dB, "
                f"jpeg={self.jpeg_artifacts:.3f}, "
                f"quality={self.quality!r})")


# ─── estimate_noise_sigma ────────────────────────────────────────────────────

def estimate_noise_sigma(img: np.ndarray,
                          kernel_size: int = 3) -> float:
    """
    Оценивает уровень σ шума методом лапласиана (Immerkær, 1996).

    Применяет лапласиан к grayscale-изображению и оценивает σ через
    средний абсолютный отклик, нормированный на теоретическую константу.

    Args:
        img:         BGR или grayscale изображение (uint8).
        kernel_size: Размер ядра лапласиана (не используется напрямую,
                     в params сохраняется для воспроизводимости).

    Returns:
        σ шума ≥ 0.0 (в шкале пикселей 0..255).
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float64)

    # Ядро лапласиана Immerkær для оценки шума
    kernel = np.array([[1, -2,  1],
                       [-2, 4, -2],
                       [1, -2,  1]], dtype=np.float64)

    filtered = cv2.filter2D(gray_f, -1, kernel)
    h, w     = filtered.shape
    # Нормировочная константа (теоретическая для белого шума)
    norm_c   = np.sqrt(np.pi / 2.0) / (6.0 * (h - 2) * (w - 2))
    sigma    = norm_c * float(np.sum(np.abs(filtered)))
    return float(max(0.0, sigma))


# ─── estimate_snr ─────────────────────────────────────────────────────────────

def estimate_snr(img: np.ndarray) -> float:
    """
    Оценивает отношение сигнал/шум (SNR) в дБ.

    SNR = 20 · log10(μ_signal / σ_noise).
    При σ=0 возвращает float('inf').

    Args:
        img: BGR или grayscale изображение (uint8).

    Returns:
        SNR в дБ (float, может быть inf).
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_sig = float(gray.astype(np.float64).mean())
    sigma    = estimate_noise_sigma(img)
    if sigma < 1e-12:
        return float("inf")
    snr = 20.0 * np.log10(max(mean_sig, 1e-9) / sigma)
    return float(snr)


# ─── detect_jpeg_artifacts ────────────────────────────────────────────────────

def detect_jpeg_artifacts(img:        np.ndarray,
                           block_size: int = 8) -> float:
    """
    Обнаруживает JPEG-артефакты (эффект блочности) по DCT-блокам.

    Вычисляет среднюю разницу яркости на границах 8×8-блоков и
    нормирует её в диапазон [0,1].

    Args:
        img:        BGR или grayscale изображение (uint8).
        block_size: Размер DCT-блока (по умолчанию 8 для JPEG).

    Returns:
        Индекс JPEG-артефактов ∈ [0,1].
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    g    = gray.astype(np.float32)

    # Горизонтальные границы блоков
    h_diffs = []
    for by in range(block_size, h, block_size):
        row_diff = np.abs(g[by, :] - g[by - 1, :]).mean()
        h_diffs.append(row_diff)

    # Вертикальные границы блоков
    v_diffs = []
    for bx in range(block_size, w, block_size):
        col_diff = np.abs(g[:, bx] - g[:, bx - 1]).mean()
        v_diffs.append(col_diff)

    block_diff   = float(np.mean(h_diffs + v_diffs)) if (h_diffs or v_diffs) else 0.0
    # Нормировка: типичный сильный JPEG ~20 ед. → зажимаем в [0,1]
    index = float(np.clip(block_diff / 20.0, 0.0, 1.0))
    return index


# ─── estimate_grain ───────────────────────────────────────────────────────────

def estimate_grain(img:        np.ndarray,
                   block_size: int = 16) -> float:
    """
    Оценивает зернистость по локальной дисперсии малых блоков.

    Разбивает изображение на блоки block_size×block_size и вычисляет
    среднее стандартное отклонение внутри блоков. Нормирует в [0,1].

    Args:
        img:        BGR или grayscale изображение (uint8).
        block_size: Размер блока (пикс).

    Returns:
        Зернистость ∈ [0,1].
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    g    = gray.astype(np.float32)

    stds = []
    for by in range(0, h - block_size + 1, block_size):
        for bx in range(0, w - block_size + 1, block_size):
            block = g[by:by + block_size, bx:bx + block_size]
            stds.append(float(block.std()))

    if not stds:
        return 0.0

    mean_std = float(np.mean(stds))
    # Нормировка: σ=40 соответствует сильному зерну
    return float(np.clip(mean_std / 40.0, 0.0, 1.0))


# ─── analyze_noise ────────────────────────────────────────────────────────────

def analyze_noise(img:            np.ndarray,
                  noise_thresh1:  float = 5.0,
                  noise_thresh2:  float = 15.0,
                  jpeg_block:     int   = 8,
                  grain_block:    int   = 16) -> NoiseAnalysisResult:
    """
    Полный анализ шума изображения.

    Вычисляет noise_level, snr_db, jpeg_artifacts, grain_level и присваивает
    качественную метку:
      - 'clean':     noise_level < noise_thresh1
      - 'noisy':     noise_thresh1 ≤ noise_level < noise_thresh2
      - 'very_noisy': noise_level ≥ noise_thresh2

    Args:
        img:           BGR или grayscale изображение (uint8).
        noise_thresh1: Порог разграничения 'clean' / 'noisy'.
        noise_thresh2: Порог разграничения 'noisy' / 'very_noisy'.
        jpeg_block:    Размер блока для detect_jpeg_artifacts.
        grain_block:   Размер блока для estimate_grain.

    Returns:
        NoiseAnalysisResult с полными метриками.
    """
    nl  = estimate_noise_sigma(img)
    snr = estimate_snr(img)
    ja  = detect_jpeg_artifacts(img, block_size=jpeg_block)
    gr  = estimate_grain(img, block_size=grain_block)

    if nl < noise_thresh1:
        quality = "clean"
    elif nl < noise_thresh2:
        quality = "noisy"
    else:
        quality = "very_noisy"

    return NoiseAnalysisResult(
        noise_level=nl, snr_db=snr,
        jpeg_artifacts=ja, grain_level=gr,
        quality=quality,
        params={
            "noise_thresh1": noise_thresh1,
            "noise_thresh2": noise_thresh2,
            "jpeg_block": jpeg_block,
            "grain_block": grain_block,
        },
    )


# ─── batch_analyze_noise ──────────────────────────────────────────────────────

def batch_analyze_noise(images: List[np.ndarray],
                         **kwargs) -> List[NoiseAnalysisResult]:
    """
    Пакетный анализ шума для списка изображений.

    Args:
        images:  Список BGR или grayscale изображений.
        **kwargs: Параметры, передаваемые в analyze_noise
                  (noise_thresh1, noise_thresh2, jpeg_block, grain_block).

    Returns:
        Список NoiseAnalysisResult (по одному на изображение).
    """
    return [analyze_noise(img, **kwargs) for img in images]
