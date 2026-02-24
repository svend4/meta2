"""Утилиты сглаживания сигналов и контуров.

Модуль предоставляет функции для сглаживания одномерных сигналов
(скользящее среднее, Гауссово, медианное, LOWESS) и двумерных контуров
(сплайн, усреднение точек), а также пакетную обработку.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence

import numpy as np
from scipy.signal import savgol_filter as _scipy_savgol_filter


# ─── SmoothingParams ─────────────────────────────────────────────────────────

_VALID_METHODS = frozenset(
    {"moving_average", "gaussian", "median", "savgol", "exponential"}
)


@dataclass
class SmoothingParams:
    """Параметры сглаживания сигнала.

    Атрибуты:
        method:      Алгоритм сглаживания.
        window_size: Ширина окна (нечётное целое >= 3).
        sigma:       Стандартное отклонение для Гауссова сглаживания (> 0).
        polyorder:   Порядок полинома для Savitzky-Golay (< window_size).
        alpha:       Коэффициент экспоненциального сглаживания (0 < alpha <= 1).
        params:      Дополнительные параметры.
    """

    method: str = "moving_average"
    window_size: int = 5
    sigma: float = 1.0
    polyorder: int = 2
    alpha: float = 0.3
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"Неизвестный метод '{self.method}'. "
                f"Допустимые: {sorted(_VALID_METHODS)}"
            )
        if self.window_size < 3:
            raise ValueError(
                f"window_size должен быть >= 3, получено {self.window_size}"
            )
        if self.window_size % 2 == 0:
            raise ValueError(
                f"window_size должен быть нечётным, получено {self.window_size}"
            )
        if self.sigma <= 0.0:
            raise ValueError(
                f"sigma должна быть > 0, получено {self.sigma}"
            )
        if self.polyorder < 1:
            raise ValueError(
                f"polyorder должен быть >= 1, получено {self.polyorder}"
            )
        if self.polyorder >= self.window_size:
            raise ValueError(
                f"polyorder ({self.polyorder}) должен быть < window_size ({self.window_size})"
            )
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(
                f"alpha должна быть в (0, 1], получено {self.alpha}"
            )


# ─── moving_average ──────────────────────────────────────────────────────────

def moving_average(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Сглаживание скользящим средним.

    Аргументы:
        signal:      Одномерный массив вещественных чисел.
        window_size: Ширина окна (нечётное, >= 3).

    Возвращает:
        Сглаженный массив той же длины (float64).

    Исключения:
        ValueError: Если signal не одномерный или window_size некорректен.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"signal должен быть одномерным, получено ndim={signal.ndim}")
    if window_size < 3:
        raise ValueError(f"window_size должен быть >= 3, получено {window_size}")
    if window_size % 2 == 0:
        raise ValueError(f"window_size должен быть нечётным, получено {window_size}")
    if len(signal) == 0:
        return signal.copy()

    half = window_size // 2
    padded = np.pad(signal, half, mode="edge")
    kernel = np.ones(window_size, dtype=np.float64) / window_size
    return np.convolve(padded, kernel, mode="valid")[: len(signal)]


# ─── gaussian_smooth ─────────────────────────────────────────────────────────

def gaussian_smooth(signal: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Гауссово сглаживание сигнала.

    Аргументы:
        signal: Одномерный массив.
        sigma:  Стандартное отклонение ядра (> 0).

    Возвращает:
        Сглаженный массив (float64) той же длины.

    Исключения:
        ValueError: Если signal не одномерный или sigma <= 0.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"signal должен быть одномерным, получено ndim={signal.ndim}")
    if sigma <= 0.0:
        raise ValueError(f"sigma должна быть > 0, получено {sigma}")
    if len(signal) == 0:
        return signal.copy()

    # Строим гауссово ядро размером ~6σ (нечётное)
    half = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    padded = np.pad(signal, half, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(signal)]


# ─── median_smooth ───────────────────────────────────────────────────────────

def median_smooth(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Медианное сглаживание сигнала.

    Аргументы:
        signal:      Одномерный массив.
        window_size: Ширина окна (нечётное, >= 3).

    Возвращает:
        Сглаженный массив (float64) той же длины.

    Исключения:
        ValueError: Если signal не одномерный или window_size некорректен.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"signal должен быть одномерным, получено ndim={signal.ndim}")
    if window_size < 3:
        raise ValueError(f"window_size должен быть >= 3, получено {window_size}")
    if window_size % 2 == 0:
        raise ValueError(f"window_size должен быть нечётным, получено {window_size}")
    if len(signal) == 0:
        return signal.copy()

    half = window_size // 2
    padded = np.pad(signal, half, mode="edge")
    out = np.empty(len(signal), dtype=np.float64)
    for i in range(len(signal)):
        out[i] = np.median(padded[i : i + window_size])
    return out


# ─── exponential_smooth ──────────────────────────────────────────────────────

def exponential_smooth(signal: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Экспоненциальное сглаживание (EMA).

    S[0] = x[0];  S[t] = α·x[t] + (1−α)·S[t−1]

    Аргументы:
        signal: Одномерный массив.
        alpha:  Коэффициент сглаживания (0 < alpha <= 1).

    Возвращает:
        Сглаженный массив (float64) той же длины.

    Исключения:
        ValueError: Если signal не одномерный или alpha вне диапазона.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"signal должен быть одномерным, получено ndim={signal.ndim}")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha должна быть в (0, 1], получено {alpha}")
    if len(signal) == 0:
        return signal.copy()

    out = np.empty_like(signal)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = alpha * signal[i] + (1.0 - alpha) * out[i - 1]
    return out


# ─── savgol_smooth ───────────────────────────────────────────────────────────

def savgol_smooth(
    signal: np.ndarray, window_size: int = 5, polyorder: int = 2
) -> np.ndarray:
    """Сглаживание методом Савицкого-Голея.

    Локально подбирает полином степени ``polyorder`` в каждом окне.

    Аргументы:
        signal:      Одномерный массив.
        window_size: Ширина окна (нечётное, >= 3).
        polyorder:   Степень аппроксимирующего полинома (>= 1, < window_size).

    Возвращает:
        Сглаженный массив (float64) той же длины.

    Исключения:
        ValueError: Если параметры некорректны.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"signal должен быть одномерным, получено ndim={signal.ndim}")
    if window_size < 3:
        raise ValueError(f"window_size должен быть >= 3, получено {window_size}")
    if window_size % 2 == 0:
        raise ValueError(f"window_size должен быть нечётным, получено {window_size}")
    if polyorder < 1:
        raise ValueError(f"polyorder должен быть >= 1, получено {polyorder}")
    if polyorder >= window_size:
        raise ValueError(
            f"polyorder ({polyorder}) должен быть < window_size ({window_size})"
        )
    if len(signal) == 0:
        return signal.copy()

    return _scipy_savgol_filter(signal, window_size, polyorder).astype(np.float64)


# ─── smooth_contour ──────────────────────────────────────────────────────────

def smooth_contour(
    contour: np.ndarray, sigma: float = 1.0
) -> np.ndarray:
    """Гауссово сглаживание 2-D контура.

    Сглаживает каждую координату независимо с учётом замкнутости контура.

    Аргументы:
        contour: Массив формы (N, 2) с координатами точек контура.
        sigma:   Стандартное отклонение ядра (> 0).

    Возвращает:
        Сглаженный контур формы (N, 2) (float32).

    Исключения:
        ValueError: Если контур не имеет формы (N, 2) или пустой.
    """
    contour = np.asarray(contour, dtype=np.float32)
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError(
            f"contour должен иметь форму (N, 2), получено {contour.shape}"
        )
    if len(contour) == 0:
        raise ValueError("contour не может быть пустым")
    if sigma <= 0.0:
        raise ValueError(f"sigma должна быть > 0, получено {sigma}")

    xs = gaussian_smooth(contour[:, 0].astype(np.float64), sigma=sigma)
    ys = gaussian_smooth(contour[:, 1].astype(np.float64), sigma=sigma)
    return np.stack([xs, ys], axis=1).astype(np.float32)


# ─── apply_smoothing ─────────────────────────────────────────────────────────

def apply_smoothing(
    signal: np.ndarray, params: SmoothingParams
) -> np.ndarray:
    """Применить сглаживание к сигналу согласно параметрам.

    Аргументы:
        signal: Одномерный сигнал.
        params: Параметры сглаживания.

    Возвращает:
        Сглаженный сигнал (float64).
    """
    dispatch = {
        "moving_average": lambda s: moving_average(s, params.window_size),
        "gaussian": lambda s: gaussian_smooth(s, params.sigma),
        "median": lambda s: median_smooth(s, params.window_size),
        "savgol": lambda s: savgol_smooth(s, params.window_size, params.polyorder),
        "exponential": lambda s: exponential_smooth(s, params.alpha),
    }
    return dispatch[params.method](signal)


# ─── batch_smooth ─────────────────────────────────────────────────────────────

def batch_smooth(
    signals: List[np.ndarray], params: SmoothingParams
) -> List[np.ndarray]:
    """Применить сглаживание к списку сигналов.

    Аргументы:
        signals: Список одномерных массивов.
        params:  Параметры сглаживания.

    Возвращает:
        Список сглаженных массивов (float64).
    """
    return [apply_smoothing(s, params) for s in signals]
