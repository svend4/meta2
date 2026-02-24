"""
Утилиты обработки одномерных сигналов.

Используются для анализа профилей краёв, яркостных переходов и
текстурных паттернов вдоль границ фрагментов.

Экспортирует:
    smooth_signal          — сглаживание (Gaussian / moving average)
    normalize_signal       — нормализация в заданный диапазон
    find_peaks             — поиск локальных максимумов
    find_valleys           — поиск локальных минимумов
    compute_autocorrelation  — нормированная автокорреляция
    compute_cross_correlation — нормированная кросскорреляция
    signal_energy          — суммарная энергия сигнала (L2)
    segment_signal         — сегментация сигнала по порогу
    resample_signal        — ресэмплинг до заданного числа точек
    phase_shift            — сдвиг с наилучшей кросскорреляцией
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d


# ─── Публичные функции ────────────────────────────────────────────────────────

def smooth_signal(
    signal: np.ndarray,
    method: str = "gaussian",
    sigma: float = 1.0,
    window: int = 5,
) -> np.ndarray:
    """Сгладить одномерный сигнал.

    Args:
        signal: Массив float64 (N,).
        method: ``'gaussian'`` (фильтр Гаусса) или ``'moving_avg'`` (скользящее среднее).
        sigma:  Стандартное отклонение для Gaussian (> 0).
        window: Ширина окна для moving_avg (нечётное, ≥ 1).

    Returns:
        Сглаженный массив float64 той же длины.

    Raises:
        ValueError: Если ``method`` неизвестен, ``sigma`` ≤ 0 или ``window`` < 1.
    """
    s = np.asarray(signal, dtype=np.float64).ravel()
    if method == "gaussian":
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        return gaussian_filter1d(s, sigma=sigma)
    if method == "moving_avg":
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        kernel = np.ones(window, dtype=np.float64) / window
        return np.convolve(s, kernel, mode="same")
    raise ValueError(f"method must be 'gaussian' or 'moving_avg', got {method!r}")


def normalize_signal(
    signal: np.ndarray,
    out_min: float = 0.0,
    out_max: float = 1.0,
) -> np.ndarray:
    """Нормализовать сигнал в диапазон [out_min, out_max].

    Если сигнал константный, возвращает массив, заполненный ``out_min``.

    Args:
        signal:  Массив (N,).
        out_min: Нижняя граница выходного диапазона.
        out_max: Верхняя граница выходного диапазона.

    Returns:
        Нормализованный массив float64 той же длины.

    Raises:
        ValueError: Если ``out_min`` ≥ ``out_max``.
    """
    if out_min >= out_max:
        raise ValueError(
            f"out_min must be < out_max, got {out_min} >= {out_max}"
        )
    s = np.asarray(signal, dtype=np.float64).ravel()
    s_min = s.min()
    s_max = s.max()
    rng = s_max - s_min
    if rng < 1e-12:
        return np.full_like(s, out_min, dtype=np.float64)
    return out_min + (s - s_min) / rng * (out_max - out_min)


def find_peaks(
    signal: np.ndarray,
    min_height: float = 0.0,
    min_distance: int = 1,
) -> np.ndarray:
    """Найти индексы локальных максимумов сигнала.

    Локальный максимум — значение, строго превышающее оба соседних.

    Args:
        signal:       Массив (N,).
        min_height:   Минимальное значение для считывания пиком.
        min_distance: Минимальное расстояние между соседними пиками (≥ 1).

    Returns:
        Массив int64 с индексами пиков.

    Raises:
        ValueError: Если ``min_distance`` < 1.
    """
    if min_distance < 1:
        raise ValueError(f"min_distance must be >= 1, got {min_distance}")
    s = np.asarray(signal, dtype=np.float64).ravel()
    n = len(s)
    if n < 3:
        return np.empty(0, dtype=np.int64)

    candidates = []
    for i in range(1, n - 1):
        if s[i] > s[i - 1] and s[i] > s[i + 1] and s[i] >= min_height:
            candidates.append(i)

    if min_distance <= 1 or len(candidates) == 0:
        return np.array(candidates, dtype=np.int64)

    # Greedy suppression by distance
    selected: List[int] = [candidates[0]]
    for idx in candidates[1:]:
        if idx - selected[-1] >= min_distance:
            selected.append(idx)
    return np.array(selected, dtype=np.int64)


def find_valleys(
    signal: np.ndarray,
    max_depth: float = float("inf"),
    min_distance: int = 1,
) -> np.ndarray:
    """Найти индексы локальных минимумов (долин) сигнала.

    Args:
        signal:       Массив (N,).
        max_depth:    Максимальное значение, при котором считается долиной.
        min_distance: Минимальное расстояние между соседними долинами (≥ 1).

    Returns:
        Массив int64 с индексами долин.

    Raises:
        ValueError: Если ``min_distance`` < 1.
    """
    inverted = -np.asarray(signal, dtype=np.float64)
    return find_peaks(inverted, min_height=-max_depth, min_distance=min_distance)


def compute_autocorrelation(signal: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Вычислить автокорреляцию сигнала.

    Args:
        signal:    Массив (N,).
        normalize: Если ``True``, нормировать на нулевой лаг (autocorr[0] = 1).

    Returns:
        Массив float64 длиной 2N−1 (полная автокорреляция).

    Raises:
        ValueError: Если сигнал пустой.
    """
    s = np.asarray(signal, dtype=np.float64).ravel()
    if len(s) == 0:
        raise ValueError("signal must not be empty")
    ac = np.correlate(s, s, mode="full")
    if normalize:
        peak = ac[len(ac) // 2]
        if abs(peak) > 1e-12:
            ac = ac / peak
    return ac


def compute_cross_correlation(
    signal1: np.ndarray,
    signal2: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Вычислить кросскорреляцию двух сигналов.

    Args:
        signal1:   Массив (N,).
        signal2:   Массив (M,).
        normalize: Нормировать на sqrt(energy1 × energy2).

    Returns:
        Массив float64 длиной N+M−1.

    Raises:
        ValueError: Если один из сигналов пустой.
    """
    s1 = np.asarray(signal1, dtype=np.float64).ravel()
    s2 = np.asarray(signal2, dtype=np.float64).ravel()
    if len(s1) == 0 or len(s2) == 0:
        raise ValueError("Both signals must be non-empty")
    cc = np.correlate(s1, s2, mode="full")
    if normalize:
        e1 = float(np.dot(s1, s1))
        e2 = float(np.dot(s2, s2))
        denom = np.sqrt(e1 * e2)
        if denom > 1e-12:
            cc = cc / denom
    return cc


def signal_energy(signal: np.ndarray) -> float:
    """Вычислить энергию сигнала: sum(x²).

    Args:
        signal: Массив (N,).

    Returns:
        Неотрицательное значение float.
    """
    s = np.asarray(signal, dtype=np.float64).ravel()
    return float(np.dot(s, s))


def segment_signal(
    signal: np.ndarray,
    threshold: float,
    above: bool = True,
) -> List[Tuple[int, int]]:
    """Сегментировать сигнал по порогу.

    Возвращает список отрезков [start, end), где значения выше (или ниже)
    заданного порога.

    Args:
        signal:    Массив (N,).
        threshold: Пороговое значение.
        above:     Если ``True``, ищем сегменты выше порога; иначе — ниже.

    Returns:
        Список кортежей (start_idx, end_idx) — полуоткрытые интервалы.
    """
    s = np.asarray(signal, dtype=np.float64).ravel()
    mask = s >= threshold if above else s < threshold
    segments: List[Tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_seg:
            start = i
            in_seg = True
        elif not v and in_seg:
            segments.append((start, i))
            in_seg = False
    if in_seg:
        segments.append((start, len(s)))
    return segments


def resample_signal(signal: np.ndarray, n_out: int) -> np.ndarray:
    """Ресэмплировать сигнал до ``n_out`` точек линейной интерполяцией.

    Args:
        signal: Массив (N,) с N ≥ 1.
        n_out:  Число выходных точек (≥ 1).

    Returns:
        Массив float64 длиной ``n_out``.

    Raises:
        ValueError: Если ``n_out`` < 1 или сигнал пустой.
    """
    s = np.asarray(signal, dtype=np.float64).ravel()
    if len(s) == 0:
        raise ValueError("signal must not be empty")
    if n_out < 1:
        raise ValueError(f"n_out must be >= 1, got {n_out}")
    if len(s) == 1:
        return np.full(n_out, s[0], dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, len(s))
    x_new = np.linspace(0.0, 1.0, n_out)
    return np.interp(x_new, x_old, s)


def phase_shift(
    signal1: np.ndarray,
    signal2: np.ndarray,
) -> Tuple[int, float]:
    """Найти сдвиг, при котором кросскорреляция максимальна.

    Args:
        signal1: Массив (N,).
        signal2: Массив (N,) той же длины.

    Returns:
        Кортеж (shift, peak_value):
        - ``shift`` — целочисленный сдвиг в диапазоне [-(N-1), N-1].
        - ``peak_value`` — нормированное значение кросскорреляции в пике.

    Raises:
        ValueError: Если сигналы разной длины или пустые.
    """
    s1 = np.asarray(signal1, dtype=np.float64).ravel()
    s2 = np.asarray(signal2, dtype=np.float64).ravel()
    if len(s1) != len(s2):
        raise ValueError(
            f"signal1 and signal2 must have the same length, "
            f"got {len(s1)} and {len(s2)}"
        )
    if len(s1) == 0:
        raise ValueError("Signals must not be empty")
    cc = compute_cross_correlation(s1, s2, normalize=True)
    n = len(s1)
    best_idx = int(np.argmax(cc))
    best_lag = (n - 1) - best_idx
    return best_lag, float(cc[best_idx])
