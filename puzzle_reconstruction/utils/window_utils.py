"""
Утилиты оконной обработки одномерных последовательностей.

Предоставляет инструменты для скользящих статистик, оконных функций
и разбивки сигналов на перекрывающиеся сегменты.
Применяется для анализа профилей яркости вдоль краёв фрагментов,
нормализации дескрипторов и сглаживания оценок совместимости.

Экспортирует:
    WindowConfig          — параметры оконных операций
    apply_window_function — умножение на весовую оконную функцию
    rolling_mean          — скользящее среднее
    rolling_std           — скользящее стандартное отклонение
    rolling_max           — скользящий максимум
    rolling_min           — скользящий минимум
    compute_overlap       — коэффициент перекрытия двух окон
    split_into_windows    — разбивка на перекрывающиеся окна
    merge_windows         — усреднение перекрывающихся окон обратно
    batch_rolling         — пакетный скользящий расчёт
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np


# ─── WindowConfig ─────────────────────────────────────────────────────────────

@dataclass
class WindowConfig:
    """Параметры оконных операций.

    Attributes:
        size:    Размер окна (>= 1).
        step:    Шаг между окнами (>= 1).
        func:    Тип оконной функции: 'hann' | 'hamming' | 'bartlett' | 'blackman' | 'rect'.
        padding: Стратегия дополнения краёв: 'same' | 'valid'.
    """
    size:    int  = 8
    step:    int  = 1
    func:    Literal["hann", "hamming", "bartlett", "blackman", "rect"] = "rect"
    padding: Literal["same", "valid"] = "same"

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError(f"size must be >= 1, got {self.size}")
        if self.step < 1:
            raise ValueError(f"step must be >= 1, got {self.step}")
        _WINDOW_FUNCS = {"hann", "hamming", "bartlett", "blackman", "rect"}
        if self.func not in _WINDOW_FUNCS:
            raise ValueError(f"func must be one of {_WINDOW_FUNCS}, got {self.func!r}")
        if self.padding not in ("same", "valid"):
            raise ValueError(f"padding must be 'same' or 'valid', got {self.padding!r}")


# ─── apply_window_function ────────────────────────────────────────────────────

def apply_window_function(
    signal: np.ndarray,
    cfg: WindowConfig | None = None,
) -> np.ndarray:
    """Умножить сигнал на оконную функцию заданного размера.

    Если len(signal) != cfg.size, оконная функция интерполируется.

    Args:
        signal: 1-D массив.
        cfg:    Параметры (None → WindowConfig()).

    Returns:
        Массив float64 той же длины, умноженный на оконные веса.

    Raises:
        ValueError: Если signal не 1-D или пуст.
    """
    if cfg is None:
        cfg = WindowConfig()
    s = np.asarray(signal, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(f"signal must be 1-D, got ndim={s.ndim}")
    if len(s) == 0:
        raise ValueError("signal must not be empty")

    n = len(s)
    if cfg.func == "rect":
        return s.copy()

    window_fn = {
        "hann":     np.hanning,
        "hamming":  np.hamming,
        "bartlett": np.bartlett,
        "blackman": np.blackman,
    }[cfg.func]

    w = window_fn(cfg.size).astype(np.float64)
    if len(w) != n:
        x_old = np.linspace(0.0, 1.0, len(w))
        x_new = np.linspace(0.0, 1.0, n)
        w = np.interp(x_new, x_old, w)

    return s * w


# ─── rolling_mean ─────────────────────────────────────────────────────────────

def rolling_mean(
    signal: np.ndarray,
    cfg: WindowConfig | None = None,
) -> np.ndarray:
    """Скользящее среднее.

    В режиме ``padding='same'`` возвращает массив той же длины, что и signal.
    В режиме ``padding='valid'`` возвращает только полные окна.

    Args:
        signal: 1-D массив.
        cfg:    Параметры.

    Returns:
        Массив float64.

    Raises:
        ValueError: Если signal не 1-D или пуст.
    """
    if cfg is None:
        cfg = WindowConfig()
    s = np.asarray(signal, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(f"signal must be 1-D, got ndim={s.ndim}")
    if len(s) == 0:
        raise ValueError("signal must not be empty")
    return _rolling_apply(s, cfg, np.mean)


# ─── rolling_std ──────────────────────────────────────────────────────────────

def rolling_std(
    signal: np.ndarray,
    cfg: WindowConfig | None = None,
) -> np.ndarray:
    """Скользящее стандартное отклонение.

    Args:
        signal: 1-D массив.
        cfg:    Параметры.

    Returns:
        Массив float64.

    Raises:
        ValueError: Если signal не 1-D или пуст.
    """
    if cfg is None:
        cfg = WindowConfig()
    s = np.asarray(signal, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(f"signal must be 1-D, got ndim={s.ndim}")
    if len(s) == 0:
        raise ValueError("signal must not be empty")
    return _rolling_apply(s, cfg, np.std)


# ─── rolling_max ──────────────────────────────────────────────────────────────

def rolling_max(
    signal: np.ndarray,
    cfg: WindowConfig | None = None,
) -> np.ndarray:
    """Скользящий максимум.

    Args:
        signal: 1-D массив.
        cfg:    Параметры.

    Returns:
        Массив float64.

    Raises:
        ValueError: Если signal не 1-D или пуст.
    """
    if cfg is None:
        cfg = WindowConfig()
    s = np.asarray(signal, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(f"signal must be 1-D, got ndim={s.ndim}")
    if len(s) == 0:
        raise ValueError("signal must not be empty")
    return _rolling_apply(s, cfg, np.max)


# ─── rolling_min ──────────────────────────────────────────────────────────────

def rolling_min(
    signal: np.ndarray,
    cfg: WindowConfig | None = None,
) -> np.ndarray:
    """Скользящий минимум.

    Args:
        signal: 1-D массив.
        cfg:    Параметры.

    Returns:
        Массив float64.

    Raises:
        ValueError: Если signal не 1-D или пуст.
    """
    if cfg is None:
        cfg = WindowConfig()
    s = np.asarray(signal, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(f"signal must be 1-D, got ndim={s.ndim}")
    if len(s) == 0:
        raise ValueError("signal must not be empty")
    return _rolling_apply(s, cfg, np.min)


# ─── compute_overlap ──────────────────────────────────────────────────────────

def compute_overlap(
    start_a: int, end_a: int,
    start_b: int, end_b: int,
) -> float:
    """Вычислить коэффициент перекрытия двух окон.

    Коэффициент = intersection / union (Jaccard).

    Args:
        start_a, end_a: Начало и конец первого окна (start < end).
        start_b, end_b: Начало и конец второго окна (start < end).

    Returns:
        float в [0, 1].

    Raises:
        ValueError: Если start >= end для любого окна.
    """
    if start_a >= end_a:
        raise ValueError(f"Window A: start must be < end, got {start_a} >= {end_a}")
    if start_b >= end_b:
        raise ValueError(f"Window B: start must be < end, got {start_b} >= {end_b}")

    inter_start = max(start_a, start_b)
    inter_end   = min(end_a, end_b)
    intersection = max(0, inter_end - inter_start)

    union = (end_a - start_a) + (end_b - start_b) - intersection
    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


# ─── split_into_windows ───────────────────────────────────────────────────────

def split_into_windows(
    signal: np.ndarray,
    cfg: WindowConfig | None = None,
) -> List[np.ndarray]:
    """Разбить сигнал на перекрывающиеся окна.

    Args:
        signal: 1-D массив.
        cfg:    Параметры (size, step, padding).

    Returns:
        Список массивов float64 длиной cfg.size каждый.
        В режиме 'same' края дополняются отражением.
        В режиме 'valid' возвращаются только полные окна.

    Raises:
        ValueError: Если signal не 1-D или пуст.
    """
    if cfg is None:
        cfg = WindowConfig()
    s = np.asarray(signal, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(f"signal must be 1-D, got ndim={s.ndim}")
    if len(s) == 0:
        raise ValueError("signal must not be empty")

    size = cfg.size
    step = cfg.step

    if cfg.padding == "same":
        pad = size // 2
        s_padded = np.pad(s, pad, mode="reflect")
    else:
        s_padded = s
        pad = 0

    windows: List[np.ndarray] = []
    i = 0
    while i + size <= len(s_padded):
        windows.append(s_padded[i:i + size].copy())
        i += step

    return windows


# ─── merge_windows ────────────────────────────────────────────────────────────

def merge_windows(
    windows: List[np.ndarray],
    original_length: int,
    cfg: WindowConfig | None = None,
) -> np.ndarray:
    """Усреднить перекрывающиеся окна обратно в сигнал длиной original_length.

    Каждый элемент выходного массива равен среднему по всем окнам,
    в которые он попадает.

    Args:
        windows:         Список массивов длиной cfg.size.
        original_length: Длина исходного сигнала (>= 1).
        cfg:             Параметры.

    Returns:
        Массив float64 длиной original_length.

    Raises:
        ValueError: Если windows пуст или original_length < 1.
    """
    if cfg is None:
        cfg = WindowConfig()
    if not windows:
        raise ValueError("windows must not be empty")
    if original_length < 1:
        raise ValueError(f"original_length must be >= 1, got {original_length}")

    size = cfg.size
    step = cfg.step

    if cfg.padding == "same":
        pad = size // 2
        total_len = original_length + 2 * pad
    else:
        pad = 0
        total_len = original_length

    out     = np.zeros(total_len, dtype=np.float64)
    counts  = np.zeros(total_len, dtype=np.float64)

    for k, win in enumerate(windows):
        start = k * step
        end   = start + size
        if end > total_len:
            break
        out[start:end]    += win
        counts[start:end] += 1.0

    counts = np.where(counts == 0, 1, counts)
    result = out / counts

    if cfg.padding == "same":
        return result[pad:pad + original_length]
    return result[:original_length]


# ─── batch_rolling ────────────────────────────────────────────────────────────

def batch_rolling(
    signals: List[np.ndarray],
    stat: Literal["mean", "std", "max", "min"] = "mean",
    cfg: WindowConfig | None = None,
) -> List[np.ndarray]:
    """Применить скользящую статистику к пакету сигналов.

    Args:
        signals: Список 1-D массивов.
        stat:    Статистика: 'mean' | 'std' | 'max' | 'min'.
        cfg:     Параметры.

    Returns:
        Список массивов float64.

    Raises:
        ValueError: Если signals пуст или stat неизвестен.
    """
    if not signals:
        raise ValueError("signals must not be empty")
    _FN = {"mean": rolling_mean, "std": rolling_std,
           "max": rolling_max,  "min": rolling_min}
    if stat not in _FN:
        raise ValueError(f"stat must be one of {list(_FN)}, got {stat!r}")
    fn = _FN[stat]
    return [fn(s, cfg) for s in signals]


# ─── Internal helper ──────────────────────────────────────────────────────────

def _rolling_apply(s: np.ndarray, cfg: WindowConfig, fn) -> np.ndarray:
    """Apply fn over windows of s according to cfg."""
    size = cfg.size
    step = cfg.step
    n = len(s)

    if cfg.padding == "same":
        pad = size // 2
        s_padded = np.pad(s, pad, mode="reflect")
    else:
        s_padded = s
        pad = 0

    results = []
    i = 0
    while i + size <= len(s_padded):
        results.append(fn(s_padded[i:i + size]))
        i += step

    if not results:
        return np.array([], dtype=np.float64)

    out = np.array(results, dtype=np.float64)

    # Trim or pad to match original length
    if cfg.padding == "same":
        # Center the result on the original signal
        if len(out) >= n:
            out = out[:n]
        else:
            out = np.pad(out, (0, n - len(out)), mode="edge")
    return out
