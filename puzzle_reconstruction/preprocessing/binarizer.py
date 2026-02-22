"""
Расширенная бинаризация изображений документов.

Предоставляет несколько алгоритмов перевода grayscale-изображения
в бинарное (0/255), включая классические пороговые методы и
локально-адаптивные техники.

Классы:
    BinarizeResult — результат бинаризации (маска + метаданные)

Функции:
    binarize_otsu      — глобальная пороговая бинаризация Оцу
    binarize_adaptive  — адаптивная пороговая (mean / gaussian)
    binarize_sauvola   — метод Саувола (локальный порог по μ ± k·σ)
    binarize_niblack   — метод Нибэка (μ − k·σ)
    binarize_bernsen   — метод Бернсена (mid-gray локального диапазона)
    auto_binarize      — автоматический выбор метода по энтропии
    batch_binarize     — обработка списка изображений
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import cv2
import numpy as np


# ─── BinarizeResult ───────────────────────────────────────────────────────────

@dataclass
class BinarizeResult:
    """
    Результат бинаризации одного изображения.

    Attributes:
        binary:    Бинарное изображение (uint8, значения 0 или 255).
        method:    Название метода.
        threshold: Глобальный порог (0 для адаптивных методов).
        inverted:  True если изображение инвертировано (текст белый).
        params:    Параметры метода.
    """
    binary:    np.ndarray
    method:    str
    threshold: float
    inverted:  bool  = False
    params:    Dict  = field(default_factory=dict)

    @property
    def foreground_ratio(self) -> float:
        """Доля белых (255) пикселей ∈ [0, 1]."""
        return float((self.binary > 0).sum()) / max(1, self.binary.size)

    def __repr__(self) -> str:
        return (f"BinarizeResult(method={self.method!r}, "
                f"threshold={self.threshold:.1f}, "
                f"fg={self.foreground_ratio:.3f})")


# ─── Вспомогательные ──────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _apply_invert(binary: np.ndarray, invert: bool) -> np.ndarray:
    return cv2.bitwise_not(binary) if invert else binary


def _integral_images(gray: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (integral, integral_sq) для окна скользящих статистик."""
    f = gray.astype(np.float64)
    integ    = cv2.integral(f)
    integ_sq = cv2.integral(f * f)
    return integ, integ_sq


def _window_mean_std(integ: np.ndarray,
                      integ_sq: np.ndarray,
                      half: int,
                      h: int, w: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет локальные среднее и стандартное отклонение.

    Использует интегральные изображения (O(1) на пиксель).
    """
    # Координаты с паддингом
    y1 = np.clip(np.arange(h) - half,     0, h - 1)
    y2 = np.clip(np.arange(h) + half + 1, 1, h)
    x1 = np.clip(np.arange(w) - half,     0, w - 1)
    x2 = np.clip(np.arange(w) + half + 1, 1, w)

    # Число пикселей в каждом окне
    ny = (y2 - y1).reshape(-1, 1).astype(np.float64)
    nx = (x2 - x1).reshape(1, -1).astype(np.float64)
    n  = ny * nx

    # Сумма и сумма квадратов через интегральные изображения
    # integ имеет размер (h+1, w+1)
    s  = (integ[y2[:, None], x2[None, :]]
          - integ[y1[:, None], x2[None, :]]
          - integ[y2[:, None], x1[None, :]]
          + integ[y1[:, None], x1[None, :]])
    s2 = (integ_sq[y2[:, None], x2[None, :]]
          - integ_sq[y1[:, None], x2[None, :]]
          - integ_sq[y2[:, None], x1[None, :]]
          + integ_sq[y1[:, None], x1[None, :]])

    mean = s / np.maximum(n, 1.0)
    var  = s2 / np.maximum(n, 1.0) - mean ** 2
    std  = np.sqrt(np.maximum(var, 0.0))
    return mean, std


# ─── binarize_otsu ────────────────────────────────────────────────────────────

def binarize_otsu(img:    np.ndarray,
                   invert: bool = False) -> BinarizeResult:
    """
    Бинаризация методом Оцу (глобальный порог).

    Args:
        img:    BGR или grayscale изображение.
        invert: Инвертировать результат (текст → белый).

    Returns:
        BinarizeResult.
    """
    gray = _to_gray(img)
    thr, binary = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = _apply_invert(binary, invert)
    return BinarizeResult(
        binary=binary,
        method="otsu",
        threshold=float(thr),
        inverted=invert,
        params={},
    )


# ─── binarize_adaptive ────────────────────────────────────────────────────────

def binarize_adaptive(img:             np.ndarray,
                       block_size:      int   = 11,
                       c:               float = 2.0,
                       adaptive_method: str   = "gaussian",
                       invert:          bool  = False) -> BinarizeResult:
    """
    Адаптивная пороговая бинаризация.

    Args:
        img:             BGR или grayscale изображение.
        block_size:      Размер блока (нечётное ≥ 3).
        c:               Вычитаемая константа.
        adaptive_method: 'mean' | 'gaussian'.
        invert:          Инвертировать результат.

    Returns:
        BinarizeResult.
    """
    gray = _to_gray(img)
    bs   = max(3, int(block_size) | 1)   # гарантируем нечётное

    method_flag = (cv2.ADAPTIVE_THRESH_GAUSSIAN_C
                   if adaptive_method == "gaussian"
                   else cv2.ADAPTIVE_THRESH_MEAN_C)

    binary = cv2.adaptiveThreshold(
        gray, 255, method_flag, cv2.THRESH_BINARY, bs, c
    )
    binary = _apply_invert(binary, invert)
    return BinarizeResult(
        binary=binary,
        method=f"adaptive_{adaptive_method}",
        threshold=0.0,
        inverted=invert,
        params={"block_size": bs, "c": c, "adaptive_method": adaptive_method},
    )


# ─── binarize_sauvola ─────────────────────────────────────────────────────────

def binarize_sauvola(img:         np.ndarray,
                      window_size: int   = 15,
                      k:           float = 0.2,
                      r:           float = 128.0,
                      invert:      bool  = False) -> BinarizeResult:
    """
    Метод Саувола: T(x,y) = μ(x,y) · [1 + k · (σ(x,y)/R − 1)].

    Хорошо работает при неравномерном освещении.

    Args:
        img:         BGR или grayscale изображение.
        window_size: Размер локального окна (нечётное).
        k:           Параметр чувствительности (обычно 0.1–0.3).
        r:           Нормировочный параметр (по умолчанию 128).
        invert:      Инвертировать результат.

    Returns:
        BinarizeResult.
    """
    gray = _to_gray(img).astype(np.float64)
    h, w = gray.shape
    half  = max(1, window_size // 2)

    integ, integ_sq = _integral_images(gray.astype(np.uint8))
    mean, std       = _window_mean_std(integ, integ_sq, half, h, w)

    thresh  = mean * (1.0 + k * (std / max(1e-6, r) - 1.0))
    binary  = np.where(gray >= thresh, 255, 0).astype(np.uint8)
    binary  = _apply_invert(binary, invert)

    return BinarizeResult(
        binary=binary,
        method="sauvola",
        threshold=0.0,
        inverted=invert,
        params={"window_size": window_size, "k": k, "r": r},
    )


# ─── binarize_niblack ─────────────────────────────────────────────────────────

def binarize_niblack(img:         np.ndarray,
                      window_size: int   = 15,
                      k:           float = -0.2,
                      invert:      bool  = False) -> BinarizeResult:
    """
    Метод Нибэка: T(x,y) = μ(x,y) + k · σ(x,y).

    Args:
        img:         BGR или grayscale изображение.
        window_size: Размер окна (нечётное).
        k:           Параметр (отрицательный для тёмного текста).
        invert:      Инвертировать результат.

    Returns:
        BinarizeResult.
    """
    gray  = _to_gray(img).astype(np.float64)
    h, w  = gray.shape
    half  = max(1, window_size // 2)

    integ, integ_sq = _integral_images(gray.astype(np.uint8))
    mean, std       = _window_mean_std(integ, integ_sq, half, h, w)

    thresh = mean + k * std
    binary = np.where(gray >= thresh, 255, 0).astype(np.uint8)
    binary = _apply_invert(binary, invert)

    return BinarizeResult(
        binary=binary,
        method="niblack",
        threshold=0.0,
        inverted=invert,
        params={"window_size": window_size, "k": k},
    )


# ─── binarize_bernsen ─────────────────────────────────────────────────────────

def binarize_bernsen(img:              np.ndarray,
                      window_size:      int   = 15,
                      contrast_thresh:  float = 15.0,
                      invert:           bool  = False) -> BinarizeResult:
    """
    Метод Бернсена: T = (max + min) / 2 в локальном окне.

    Низкоконтрастные области маркируются как фон.

    Args:
        img:             BGR или grayscale изображение.
        window_size:     Размер окна (нечётное).
        contrast_thresh: Минимальный локальный контраст для бинаризации.
        invert:          Инвертировать результат.

    Returns:
        BinarizeResult.
    """
    gray = _to_gray(img)
    h, w = gray.shape
    half  = max(1, window_size // 2)

    ks = 2 * half + 1
    # Морфологический min/max через дилатацию/эрозию
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
    local_min = cv2.erode(gray, kernel).astype(np.float32)
    local_max = cv2.dilate(gray, kernel).astype(np.float32)

    mid_gray  = (local_min + local_max) / 2.0
    contrast  = local_max - local_min

    gray_f = gray.astype(np.float32)
    binary = np.where(
        (contrast >= contrast_thresh) & (gray_f >= mid_gray),
        255, 0
    ).astype(np.uint8)
    binary = _apply_invert(binary, invert)

    return BinarizeResult(
        binary=binary,
        method="bernsen",
        threshold=0.0,
        inverted=invert,
        params={"window_size": window_size,
                "contrast_thresh": contrast_thresh},
    )


# ─── auto_binarize ────────────────────────────────────────────────────────────

def _image_entropy(gray: np.ndarray) -> float:
    """Шенноновская энтропия гистограммы (оценка информативности)."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist[hist > 0].astype(np.float32)
    hist /= hist.sum()
    return float(-np.sum(hist * np.log2(hist)))


def auto_binarize(img:    np.ndarray,
                   invert: bool = False) -> BinarizeResult:
    """
    Автоматически выбирает метод бинаризации по характеристикам изображения.

    Стратегия:
        Энтропия > 6.5 → Otsu (хорошая глобальная гистограмма)
        Иначе           → Sauvola (неравномерное освещение/низкий контраст)

    Args:
        img:    BGR или grayscale изображение.
        invert: Инвертировать результат.

    Returns:
        BinarizeResult с выбранным методом.
    """
    gray    = _to_gray(img)
    entropy = _image_entropy(gray)

    if entropy > 6.5:
        return binarize_otsu(img, invert=invert)
    else:
        return binarize_sauvola(img, invert=invert)


# ─── batch_binarize ───────────────────────────────────────────────────────────

_DISPATCH = {
    "otsu":             binarize_otsu,
    "adaptive_mean":    lambda img, **kw: binarize_adaptive(img, adaptive_method="mean", **kw),
    "adaptive_gaussian": lambda img, **kw: binarize_adaptive(img, adaptive_method="gaussian", **kw),
    "adaptive":         binarize_adaptive,
    "sauvola":          binarize_sauvola,
    "niblack":          binarize_niblack,
    "bernsen":          binarize_bernsen,
    "auto":             auto_binarize,
}


def batch_binarize(images: List[np.ndarray],
                    method: str = "otsu",
                    **kwargs) -> List[BinarizeResult]:
    """
    Применяет бинаризацию к списку изображений.

    Args:
        images: Список BGR или grayscale изображений.
        method: 'otsu' | 'adaptive' | 'adaptive_mean' | 'adaptive_gaussian'
                | 'sauvola' | 'niblack' | 'bernsen' | 'auto'.
        **kwargs: Параметры для выбранного метода.

    Returns:
        Список BinarizeResult.

    Raises:
        ValueError: Если method не из допустимого набора.
    """
    if method not in _DISPATCH:
        raise ValueError(
            f"Неизвестный метод: {method!r}. Допустимые: {list(_DISPATCH)}"
        )
    fn = _DISPATCH[method]
    return [fn(img, **kwargs) for img in images]
