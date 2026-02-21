"""
Сопоставление фрагментов методами спектрального анализа (FFT).

Использует амплитудные спектры и фазовую корреляцию для оценки
совместимости пар фрагментов документа.

Экспортирует:
    SpectralMatchResult  — результат спектрального сопоставления
    magnitude_spectrum   — вычисление амплитудного спектра
    log_magnitude        — логарифмическая нормализация спектра
    spectrum_correlation — корреляция двух спектров ∈ [-1, 1]
    phase_correlation    — фазовая корреляция (сдвиг + оценка)
    match_spectra        — сопоставить пару фрагментов
    batch_spectral_match — пакетное сопоставление
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class SpectralMatchResult:
    """Результат спектрального сопоставления двух фрагментов.

    Attributes:
        idx1:        Индекс первого фрагмента.
        idx2:        Индекс второго фрагмента.
        score:       Итоговая оценка совместимости ∈ [0, 1].
        phase_shift: Оценённый сдвиг (dy, dx) в пикселях.
        params:      Параметры алгоритма.
    """
    idx1: int
    idx2: int
    score: float
    phase_shift: Tuple[float, float] = (0.0, 0.0)
    params: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"score must be in [0, 1], got {self.score}"
            )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SpectralMatchResult(idx1={self.idx1}, idx2={self.idx2}, "
            f"score={self.score:.4f}, shift={self.phase_shift})"
        )


# ─── Публичные функции ────────────────────────────────────────────────────────

def magnitude_spectrum(img: np.ndarray) -> np.ndarray:
    """Вычислить амплитудный спектр изображения.

    Args:
        img: Изображение uint8 (H, W) или (H, W, C).

    Returns:
        Амплитуда float64 (H, W) — сдвинутый спектр, нулевая частота в центре.

    Raises:
        ValueError: Если изображение не 2-D или 3-D.
    """
    gray = _to_gray_float(img)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    return np.abs(fft)


def log_magnitude(spectrum: np.ndarray) -> np.ndarray:
    """Нормализовать спектр логарифмическим преобразованием.

    Args:
        spectrum: Амплитудный спектр float64 (H, W), как из
                  :func:`magnitude_spectrum`.

    Returns:
        Нормализованный массив float64 (H, W) ∈ [0, 1].
    """
    log = np.log1p(spectrum)
    mn, mx = log.min(), log.max()
    if mx - mn < 1e-12:
        return np.zeros_like(log)
    return (log - mn) / (mx - mn)


def spectrum_correlation(s1: np.ndarray, s2: np.ndarray) -> float:
    """Вычислить нормированную корреляцию двух спектров.

    Перед корреляцией спектры изменяются в размере до одинаковых
    (min из обоих) и нормализуются.

    Args:
        s1: Амплитудный спектр float64 (H1, W1).
        s2: Амплитудный спектр float64 (H2, W2).

    Returns:
        Корреляция Пирсона ∈ [-1, 1]; при нулевой дисперсии → 0.

    Raises:
        ValueError: Если оба спектра пустые.
    """
    if s1.size == 0 or s2.size == 0:
        raise ValueError("Spectra must not be empty")
    h = min(s1.shape[0], s2.shape[0])
    w = min(s1.shape[1], s2.shape[1])
    a = s1[:h, :w].astype(np.float64).ravel()
    b = s2[:h, :w].astype(np.float64).ravel()
    return float(_pearson(a, b))


def phase_correlation(
    img1: np.ndarray,
    img2: np.ndarray,
    normalize: bool = True,
) -> Tuple[float, float, float]:
    """Оценить сдвиг и оценку совместимости фазовой корреляцией.

    Args:
        img1:      Первое изображение uint8 (H, W) или (H, W, C).
        img2:      Второе изображение uint8 (H, W) или (H, W, C).
        normalize: Приводить ли изображения к одному размеру перед расчётом.

    Returns:
        Кортеж ``(score, dy, dx)``:
          - ``score`` ∈ [0, 1] — нормированный пиковый отклик,
          - ``dy``, ``dx`` — оценённый сдвиг в пикселях.

    Raises:
        ValueError: Если изображение не 2-D или 3-D.
    """
    g1 = _to_gray_float(img1)
    g2 = _to_gray_float(img2)

    if normalize:
        h = min(g1.shape[0], g2.shape[0])
        w = min(g1.shape[1], g2.shape[1])
        g1 = g1[:h, :w]
        g2 = g2[:h, :w]

    f1 = np.fft.fft2(g1)
    f2 = np.fft.fft2(g2)
    cross = f1 * np.conj(f2)
    denom = np.abs(cross)
    denom[denom < 1e-10] = 1e-10
    normalized = cross / denom
    corr_map = np.abs(np.fft.ifft2(normalized))

    peak_val = float(corr_map.max())
    peak_idx = np.unravel_index(np.argmax(corr_map), corr_map.shape)
    dy = float(peak_idx[0])
    dx = float(peak_idx[1])

    h, w = corr_map.shape
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w

    # Нормировать оценку — пиковое значение / (H*W) → [0, 1]
    score = min(1.0, peak_val / max(1.0, float(h * w) ** 0.5))
    return score, dy, dx


def match_spectra(
    img1: np.ndarray,
    img2: np.ndarray,
    idx1: int = 0,
    idx2: int = 1,
    w_corr: float = 0.5,
    w_phase: float = 0.5,
) -> SpectralMatchResult:
    """Сопоставить два фрагмента спектральным методом.

    Итоговая оценка = взвешенная сумма корреляции спектров и
    фазовой корреляции.

    Args:
        img1:    Первое изображение.
        img2:    Второе изображение.
        idx1:    Индекс первого фрагмента.
        idx2:    Индекс второго фрагмента.
        w_corr:  Вес корреляции спектров (≥ 0).
        w_phase: Вес фазовой корреляции (≥ 0); w_corr + w_phase должны быть > 0.

    Returns:
        :class:`SpectralMatchResult`.

    Raises:
        ValueError: Если оба веса нулевые.
    """
    if w_corr < 0:
        raise ValueError(f"w_corr must be >= 0, got {w_corr}")
    if w_phase < 0:
        raise ValueError(f"w_phase must be >= 0, got {w_phase}")
    total_w = w_corr + w_phase
    if total_w <= 0:
        raise ValueError("w_corr + w_phase must be > 0")

    s1 = magnitude_spectrum(img1)
    s2 = magnitude_spectrum(img2)

    # Нормированная корреляция спектров → [0, 1]
    raw_corr = spectrum_correlation(s1, s2)
    corr_score = (raw_corr + 1.0) / 2.0  # [-1,1] → [0,1]

    # Фазовая корреляция
    phase_score, dy, dx = phase_correlation(img1, img2)

    score = (w_corr * corr_score + w_phase * phase_score) / total_w
    score = float(np.clip(score, 0.0, 1.0))

    return SpectralMatchResult(
        idx1=idx1,
        idx2=idx2,
        score=score,
        phase_shift=(dy, dx),
        params={
            "w_corr": w_corr,
            "w_phase": w_phase,
            "corr_score": corr_score,
            "phase_score": phase_score,
        },
    )


def batch_spectral_match(
    query: np.ndarray,
    candidates: List[np.ndarray],
    query_idx: int = 0,
    w_corr: float = 0.5,
    w_phase: float = 0.5,
) -> List[SpectralMatchResult]:
    """Сопоставить один запрос со всеми кандидатами.

    Args:
        query:      Изображение запроса.
        candidates: Список изображений-кандидатов.
        query_idx:  Индекс фрагмента-запроса.
        w_corr:     Вес корреляции спектров.
        w_phase:    Вес фазовой корреляции.

    Returns:
        Список :class:`SpectralMatchResult` в порядке кандидатов.
    """
    return [
        match_spectra(query, cand,
                      idx1=query_idx, idx2=i,
                      w_corr=w_corr, w_phase=w_phase)
        for i, cand in enumerate(candidates)
    ]


# ─── Приватные ───────────────────────────────────────────────────────────────

def _to_gray_float(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        gray = img
    else:
        raise ValueError(f"img must be 2-D or 3-D, got ndim={img.ndim}")
    return gray.astype(np.float64)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a_c = a - a.mean()
    b_c = b - b.mean()
    num = float(np.dot(a_c, b_c))
    denom = float(np.sqrt(np.dot(a_c, a_c) * np.dot(b_c, b_c)))
    if denom < 1e-12:
        return 0.0
    return num / denom
