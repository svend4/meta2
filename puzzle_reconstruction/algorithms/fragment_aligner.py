"""
Субпиксельное выравнивание фрагментов документа.

Реализует два метода уточнения взаимного положения краёв:
фазовую корреляцию (быстро, FFT-based) и шаблонное совмещение
(точный сдвиг вдоль края). Результат используется для компенсации
малых смещений перед финальной оценкой шва.

Классы:
    AlignmentResult — смещение (dx, dy) + уверенность метода

Функции:
    estimate_shift          — 1D субпиксельный сдвиг по кросс-корреляции
    phase_correlation_align — выравнивание фазовой корреляцией
    template_match_align    — выравнивание шаблонным совмещением
    apply_shift             — применение смещения к изображению
    batch_align             — пакетное выравнивание пар
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── AlignmentResult ──────────────────────────────────────────────────────────

@dataclass
class AlignmentResult:
    """
    Результат выравнивания двух фрагментов.

    Attributes:
        dx:         Субпиксельный сдвиг по горизонтали (пикс).
        dy:         Субпиксельный сдвиг по вертикали (пикс).
        confidence: Уверенность в оценке ∈ [0,1].
        method:     Использованный метод ('phase' | 'template').
        params:     Вспомогательные параметры.
    """
    dx:         float
    dy:         float
    confidence: float
    method:     str
    params:     Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"AlignmentResult(dx={self.dx:.2f}, dy={self.dy:.2f}, "
                f"conf={self.confidence:.3f}, method={self.method!r})")


# ─── Вспомогательные ──────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _extract_strip(gray: np.ndarray, side: int, border_px: int) -> np.ndarray:
    """
    Извлекает полосу пикселей вдоль заданного края.

    Args:
        gray:      Grayscale изображение.
        side:      Сторона (0=верх,1=право,2=низ,3=лево).
        border_px: Ширина полосы в пикселях.

    Returns:
        2D массив float32.
    """
    h, w = gray.shape
    b    = max(1, min(border_px, min(h, w) // 2))
    if side == 0:
        return gray[:b, :].astype(np.float32)
    elif side == 1:
        return gray[:, -b:].astype(np.float32)
    elif side == 2:
        return gray[-b:, :].astype(np.float32)
    else:
        return gray[:, :b].astype(np.float32)


def _quadratic_peak(arr: np.ndarray, idx: int) -> float:
    """
    Субпиксельная оценка позиции пика методом квадратичной интерполяции.

    Args:
        arr: 1D массив значений.
        idx: Индекс максимума.

    Returns:
        Субпиксельный индекс пика.
    """
    n = len(arr)
    if idx <= 0 or idx >= n - 1:
        return float(idx)
    y0 = float(arr[idx - 1])
    y1 = float(arr[idx])
    y2 = float(arr[idx + 1])
    denom = 2.0 * (2.0 * y1 - y0 - y2)
    if abs(denom) < 1e-9:
        return float(idx)
    delta = (y2 - y0) / denom
    return float(idx) + delta


# ─── estimate_shift ───────────────────────────────────────────────────────────

def estimate_shift(strip1: np.ndarray,
                    strip2: np.ndarray) -> Tuple[float, float]:
    """
    Оценивает субпиксельный сдвиг между двумя 1D-профилями методом
    нормализованной кросс-корреляции.

    Args:
        strip1: Первый профиль (1D или усреднённый 2D → 1D).
        strip2: Второй профиль той же длины.

    Returns:
        (shift, confidence) — сдвиг в пикселях и уверенность ∈ [0,1].
        Положительный сдвиг означает, что strip2 сдвинут вправо.
    """
    a = strip1.flatten().astype(np.float64)
    b = strip2.flatten().astype(np.float64)
    n = min(len(a), len(b))
    if n < 2:
        return (0.0, 0.0)
    a, b = a[:n], b[:n]

    # Нормализация
    a -= a.mean()
    b -= b.mean()
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return (0.0, 0.5)
    a /= sa
    b /= sb

    # FFT кросс-корреляция
    fa   = np.fft.rfft(a, n=2 * n)
    fb   = np.fft.rfft(b, n=2 * n)
    cc   = np.fft.irfft(fa * np.conj(fb))
    cc   = np.concatenate([cc[n:], cc[:n]])   # центрируем

    peak_idx = int(np.argmax(cc))
    peak_val = float(cc[peak_idx])
    sub_idx  = _quadratic_peak(cc, peak_idx)
    shift    = sub_idx - n   # смещение относительно нуля

    norm_peak = float(np.clip(peak_val / max(float(np.max(np.abs(cc))), 1e-9),
                               0.0, 1.0))
    return (float(shift), norm_peak)


# ─── phase_correlation_align ──────────────────────────────────────────────────

def phase_correlation_align(img1:       np.ndarray,
                              img2:       np.ndarray,
                              side1:      int = 1,
                              side2:      int = 3,
                              border_px:  int = 8,
                              n_samples:  int = 64) -> AlignmentResult:
    """
    Выравнивает два фрагмента методом фазовой корреляции краевых полос.

    Извлекает полосы вдоль указанных краёв, вычисляет 1D нормализованную
    кросс-корреляцию и находит субпиксельный сдвиг вдоль направления края.

    Args:
        img1, img2: BGR или grayscale изображения.
        side1:      Сторона первого фрагмента (0=верх,…,3=лево).
        side2:      Сторона второго фрагмента.
        border_px:  Ширина извлекаемой полосы (пикс).
        n_samples:  Длина профиля для корреляции.

    Returns:
        AlignmentResult с method='phase'.
    """
    g1 = _to_gray(img1)
    g2 = _to_gray(img2)

    s1 = _extract_strip(g1, side1, border_px).mean(axis=0 if side1 in (0, 2) else 1)
    s2 = _extract_strip(g2, side2, border_px).mean(axis=0 if side2 in (0, 2) else 1)

    # Ресэмплинг до n_samples
    def _resample(arr: np.ndarray, n: int) -> np.ndarray:
        if len(arr) == n:
            return arr
        xp = np.arange(len(arr), dtype=np.float64)
        xs = np.linspace(0, len(arr) - 1, n, dtype=np.float64)
        return np.interp(xs, xp, arr).astype(np.float32)

    s1r = _resample(s1, n_samples)
    s2r = _resample(s2, n_samples)

    shift, conf = estimate_shift(s1r, s2r)

    # Сдвиг вдоль края → dx / dy в зависимости от стороны
    if side1 in (0, 2):   # горизонтальный край → сдвиг по X
        dx, dy = float(shift), 0.0
    else:                 # вертикальный край → сдвиг по Y
        dx, dy = 0.0, float(shift)

    return AlignmentResult(
        dx=dx, dy=dy, confidence=float(conf),
        method="phase",
        params={
            "side1": side1, "side2": side2,
            "border_px": border_px, "n_samples": n_samples,
        },
    )


# ─── template_match_align ─────────────────────────────────────────────────────

def template_match_align(img1:        np.ndarray,
                          img2:        np.ndarray,
                          side1:       int = 1,
                          side2:       int = 3,
                          border_px:   int = 8,
                          search_range: int = 10) -> AlignmentResult:
    """
    Выравнивает фрагменты методом шаблонного совмещения.

    Использует полосу края img1 как шаблон и ищет её в расширенной
    полосе img2 в диапазоне ±search_range пикселей.

    Args:
        img1, img2:   BGR или grayscale изображения.
        side1:        Сторона первого фрагмента.
        side2:        Сторона второго фрагмента.
        border_px:    Ширина краевой полосы (пикс).
        search_range: Диапазон поиска (±пикс).

    Returns:
        AlignmentResult с method='template'.
    """
    g1 = _to_gray(img1)
    g2 = _to_gray(img2)

    tmpl  = _extract_strip(g1, side1, border_px).astype(np.float32)
    search = _extract_strip(g2, side2, max(border_px + search_range, border_px + 1))
    search = search.astype(np.float32)

    # Усредняем по перпендикулярному направлению → 1D
    axis = 0 if side1 in (0, 2) else 1
    t1d  = tmpl.mean(axis=axis)
    s1d  = search.mean(axis=axis)

    # Корреляция скользящим окном
    n  = len(t1d)
    m  = len(s1d)
    if n == 0 or m < n:
        return AlignmentResult(dx=0.0, dy=0.0, confidence=0.0,
                                method="template",
                                params={"side1": side1, "side2": side2,
                                        "border_px": border_px,
                                        "search_range": search_range})

    # Нормализованная корреляция
    t_norm = t1d - t1d.mean()
    t_std  = t_norm.std()
    if t_std < 1e-9:
        return AlignmentResult(dx=0.0, dy=0.0, confidence=0.5,
                                method="template",
                                params={"side1": side1, "side2": side2,
                                        "border_px": border_px,
                                        "search_range": search_range})

    scores = np.zeros(m - n + 1, dtype=np.float64)
    for i in range(len(scores)):
        window = s1d[i:i + n]
        w_norm = window - window.mean()
        w_std  = w_norm.std()
        if w_std < 1e-9:
            scores[i] = 0.0
        else:
            scores[i] = float(np.dot(t_norm, w_norm) / (t_std * w_std * n))

    best_idx = int(np.argmax(scores))
    conf     = float(np.clip((scores[best_idx] + 1.0) / 2.0, 0.0, 1.0))
    shift    = float(best_idx) - search_range

    if side1 in (0, 2):
        dx, dy = shift, 0.0
    else:
        dx, dy = 0.0, shift

    return AlignmentResult(
        dx=dx, dy=dy, confidence=conf,
        method="template",
        params={"side1": side1, "side2": side2,
                "border_px": border_px, "search_range": search_range},
    )


# ─── apply_shift ──────────────────────────────────────────────────────────────

def apply_shift(img: np.ndarray,
                 dx:  float,
                 dy:  float) -> np.ndarray:
    """
    Применяет аффинное смещение к изображению.

    Использует cv2.warpAffine с билинейной интерполяцией и заполняет
    выходящие за границы области белым цветом (255).

    Args:
        img: BGR или grayscale изображение (uint8).
        dx:  Сдвиг по горизонтали (пикс, вправо > 0).
        dy:  Сдвиг по вертикали (пикс, вниз > 0).

    Returns:
        Изображение того же размера с применённым сдвигом.
    """
    M  = np.array([[1.0, 0.0, dx],
                   [0.0, 1.0, dy]], dtype=np.float32)
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=255)


# ─── batch_align ──────────────────────────────────────────────────────────────

_ALIGN_DISPATCH = {
    "phase":    phase_correlation_align,
    "template": template_match_align,
}


def batch_align(images: List[np.ndarray],
                 pairs:  List[Tuple[int, int, int, int]],
                 method: str = "phase",
                 **kwargs) -> List[AlignmentResult]:
    """
    Пакетное выравнивание пар фрагментов.

    Args:
        images: Список BGR или grayscale изображений.
        pairs:  Список кортежей (i, side1, j, side2).
        method: 'phase' | 'template'.
        **kwargs: Параметры, передаваемые в выбранный метод.

    Returns:
        Список AlignmentResult (по одному на пару).

    Raises:
        ValueError: Если метод неизвестен.
    """
    if method not in _ALIGN_DISPATCH:
        raise ValueError(
            f"Unknown alignment method {method!r}. "
            f"Available: {sorted(_ALIGN_DISPATCH.keys())}"
        )
    fn = _ALIGN_DISPATCH[method]
    return [
        fn(images[i], images[j], side1=s1, side2=s2, **kwargs)
        for i, s1, j, s2 in pairs
    ]
