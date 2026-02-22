"""
Улучшение контраста изображений фрагментов документов.

Реализует несколько алгоритмов повышения контраста:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Histogram Equalization (глобальная)
    - Gamma Correction
    - Linear Stretch (min-max нормализация)
    - Single-Scale Retinex (логарифмическое выравнивание яркости)

Классы:
    ContrastResult — результат улучшения контраста (изображение + метрики)

Функции:
    enhance_clahe    — адаптивное CLAHE в пространстве LAB/HSV
    enhance_histeq   — глобальное выравнивание гистограммы
    enhance_gamma    — степенная гамма-коррекция
    enhance_stretch  — линейное растягивание диапазона [p_lo%, p_hi%]
    enhance_retinex  — Single-Scale Retinex (SSR)
    measure_contrast — RMS-контраст изображения
    auto_enhance     — автовыбор метода по RMS-контрасту
    batch_enhance    — обработка списка изображений
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── ContrastResult ───────────────────────────────────────────────────────────

@dataclass
class ContrastResult:
    """
    Результат улучшения контраста одного изображения.

    Attributes:
        enhanced:        Обработанное изображение (dtype совпадает со входом).
        method:          Название метода.
        contrast_before: RMS-контраст до обработки.
        contrast_after:  RMS-контраст после обработки.
        params:          Параметры метода.
    """
    enhanced:        np.ndarray
    method:          str
    contrast_before: float
    contrast_after:  float
    params:          Dict = field(default_factory=dict)

    @property
    def improvement(self) -> float:
        """Абсолютный прирост RMS-контраста."""
        return self.contrast_after - self.contrast_before

    @property
    def improvement_ratio(self) -> float:
        """Относительный прирост RMS-контраста (0 если до = 0)."""
        if self.contrast_before <= 0.0:
            return 0.0
        return self.improvement / self.contrast_before

    def __repr__(self) -> str:
        return (f"ContrastResult(method={self.method!r}, "
                f"contrast={self.contrast_before:.2f}→{self.contrast_after:.2f}, "
                f"Δ={self.improvement:+.2f})")


# ─── Вспомогательные ──────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def measure_contrast(img: np.ndarray) -> float:
    """
    Вычисляет RMS-контраст (нормированное стандартное отклонение яркости).

    Args:
        img: BGR или grayscale изображение.

    Returns:
        RMS-контраст ∈ [0, 127.5].
    """
    gray = _to_gray(img).astype(np.float32)
    return float(np.std(gray))


# ─── enhance_clahe ────────────────────────────────────────────────────────────

def enhance_clahe(img:        np.ndarray,
                   clip_limit: float = 2.0,
                   tile_size:  int   = 8) -> ContrastResult:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Для цветных изображений применяется к L-каналу в пространстве LAB.
    Для grayscale — напрямую.

    Args:
        img:        BGR или grayscale изображение.
        clip_limit: Порог ограничения контраста.
        tile_size:  Размер тайла (tile_size × tile_size).

    Returns:
        ContrastResult.
    """
    ts  = max(1, int(tile_size))
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(ts, ts))
    contrast_before = measure_contrast(img)

    if img.ndim == 2:
        enhanced = clahe.apply(img)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_eq = clahe.apply(l_chan)
        lab_eq = cv2.merge([l_eq, a_chan, b_chan])
        enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    contrast_after = measure_contrast(enhanced)
    return ContrastResult(
        enhanced=enhanced,
        method="clahe",
        contrast_before=contrast_before,
        contrast_after=contrast_after,
        params={"clip_limit": clip_limit, "tile_size": ts},
    )


# ─── enhance_histeq ───────────────────────────────────────────────────────────

def enhance_histeq(img: np.ndarray) -> ContrastResult:
    """
    Глобальное выравнивание гистограммы.

    Для цветных изображений применяется к V-каналу в пространстве HSV.

    Args:
        img: BGR или grayscale изображение.

    Returns:
        ContrastResult.
    """
    contrast_before = measure_contrast(img)

    if img.ndim == 2:
        enhanced = cv2.equalizeHist(img)
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v_eq = cv2.equalizeHist(v)
        enhanced = cv2.cvtColor(cv2.merge([h, s, v_eq]), cv2.COLOR_HSV2BGR)

    contrast_after = measure_contrast(enhanced)
    return ContrastResult(
        enhanced=enhanced,
        method="histeq",
        contrast_before=contrast_before,
        contrast_after=contrast_after,
        params={},
    )


# ─── enhance_gamma ────────────────────────────────────────────────────────────

# LUT-кэш для ускорения повторных вызовов
_gamma_lut_cache: Dict[float, np.ndarray] = {}


def _build_gamma_lut(gamma: float) -> np.ndarray:
    if gamma not in _gamma_lut_cache:
        inv = 1.0 / max(1e-6, gamma)
        table = (np.arange(256, dtype=np.float32) / 255.0) ** inv * 255.0
        _gamma_lut_cache[gamma] = np.clip(table, 0, 255).astype(np.uint8)
    return _gamma_lut_cache[gamma]


def enhance_gamma(img:   np.ndarray,
                   gamma: float = 1.5) -> ContrastResult:
    """
    Гамма-коррекция: output = input^(1/γ) × 255.

    γ > 1  → осветление (усиление тёмных областей).
    γ < 1  → затемнение.
    γ = 1  → без изменений.

    Args:
        img:   BGR или grayscale изображение.
        gamma: Значение гаммы (> 0).

    Returns:
        ContrastResult.
    """
    gamma = max(1e-6, float(gamma))
    lut   = _build_gamma_lut(gamma)
    contrast_before = measure_contrast(img)
    enhanced = cv2.LUT(img, lut)
    contrast_after  = measure_contrast(enhanced)
    return ContrastResult(
        enhanced=enhanced,
        method="gamma",
        contrast_before=contrast_before,
        contrast_after=contrast_after,
        params={"gamma": gamma},
    )


# ─── enhance_stretch ──────────────────────────────────────────────────────────

def enhance_stretch(img:    np.ndarray,
                     p_low:  float = 2.0,
                     p_high: float = 98.0) -> ContrastResult:
    """
    Линейное растягивание гистограммы по перцентилям.

    Пиксели ниже p_low-го перцентиля → 0, выше p_high-го → 255.

    Args:
        img:    BGR или grayscale изображение.
        p_low:  Нижний перцентиль (%).
        p_high: Верхний перцентиль (%).

    Returns:
        ContrastResult.
    """
    contrast_before = measure_contrast(img)

    def _stretch_channel(ch: np.ndarray) -> np.ndarray:
        lo = float(np.percentile(ch, p_low))
        hi = float(np.percentile(ch, p_high))
        if hi <= lo:
            return ch.copy()
        stretched = (ch.astype(np.float32) - lo) / (hi - lo) * 255.0
        return np.clip(stretched, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        enhanced = _stretch_channel(img)
    else:
        channels = cv2.split(img)
        enhanced = cv2.merge([_stretch_channel(c) for c in channels])

    contrast_after = measure_contrast(enhanced)
    return ContrastResult(
        enhanced=enhanced,
        method="stretch",
        contrast_before=contrast_before,
        contrast_after=contrast_after,
        params={"p_low": p_low, "p_high": p_high},
    )


# ─── enhance_retinex ──────────────────────────────────────────────────────────

def enhance_retinex(img:   np.ndarray,
                     sigma: float = 30.0) -> ContrastResult:
    """
    Single-Scale Retinex (SSR): R = log(I) - log(G_σ * I).

    Нормализует локальную яркость, улучшая контраст в тёмных областях.

    Args:
        img:   BGR или grayscale изображение.
        sigma: σ гауссовского ядра (пространственный масштаб).

    Returns:
        ContrastResult.
    """
    contrast_before = measure_contrast(img)

    ksize = int(max(3, 2 * int(3 * sigma) + 1))
    if ksize % 2 == 0:
        ksize += 1

    def _ssr(ch: np.ndarray) -> np.ndarray:
        ch_f   = ch.astype(np.float32) + 1.0   # avoid log(0)
        blurred = cv2.GaussianBlur(ch_f, (ksize, ksize), sigma)
        blurred = np.maximum(blurred, 1.0)
        retinex = np.log(ch_f) - np.log(blurred)
        # Нормировка в [0, 255]
        rmin, rmax = retinex.min(), retinex.max()
        if rmax > rmin:
            retinex = (retinex - rmin) / (rmax - rmin) * 255.0
        else:
            retinex = np.zeros_like(retinex)
        return np.clip(retinex, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        enhanced = _ssr(img)
    else:
        channels = cv2.split(img)
        enhanced = cv2.merge([_ssr(c) for c in channels])

    contrast_after = measure_contrast(enhanced)
    return ContrastResult(
        enhanced=enhanced,
        method="retinex",
        contrast_before=contrast_before,
        contrast_after=contrast_after,
        params={"sigma": sigma},
    )


# ─── auto_enhance ─────────────────────────────────────────────────────────────

_AUTO_LOW  = 20.0    # RMS ниже → очень низкий контраст
_AUTO_HIGH = 60.0    # RMS выше → контраст достаточный


def auto_enhance(img:        np.ndarray,
                  clip_limit: float = 2.0,
                  gamma:      float = 1.5) -> ContrastResult:
    """
    Автоматически выбирает метод улучшения контраста по RMS-контрасту.

    RMS < 20  → CLAHE (сильное улучшение)
    RMS < 60  → Stretch (умеренное улучшение)
    RMS ≥ 60  → Gamma (лёгкая коррекция или без изменений при γ≈1)

    Args:
        img:        Входное изображение.
        clip_limit: clip_limit для CLAHE.
        gamma:      γ для гамма-коррекции.

    Returns:
        ContrastResult с выбранным методом.
    """
    rms = measure_contrast(img)
    if rms < _AUTO_LOW:
        return enhance_clahe(img, clip_limit=clip_limit)
    elif rms < _AUTO_HIGH:
        return enhance_stretch(img)
    else:
        return enhance_gamma(img, gamma=gamma)


# ─── batch_enhance ────────────────────────────────────────────────────────────

_DISPATCH = {
    "auto":    auto_enhance,
    "clahe":   enhance_clahe,
    "histeq":  enhance_histeq,
    "gamma":   enhance_gamma,
    "stretch": enhance_stretch,
    "retinex": enhance_retinex,
}


def batch_enhance(images: List[np.ndarray],
                   method: str = "auto",
                   **kwargs) -> List[ContrastResult]:
    """
    Применяет улучшение контраста к списку изображений.

    Args:
        images: Список BGR или grayscale изображений.
        method: 'auto' | 'clahe' | 'histeq' | 'gamma' | 'stretch' | 'retinex'.
        **kwargs: Параметры для выбранного метода.

    Returns:
        Список ContrastResult.

    Raises:
        ValueError: Если method не из допустимого набора.
    """
    if method not in _DISPATCH:
        raise ValueError(
            f"Неизвестный метод: {method!r}. Допустимые: {list(_DISPATCH)}"
        )
    fn = _DISPATCH[method]
    return [fn(img, **kwargs) for img in images]
