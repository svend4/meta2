"""
Статистические утилиты для анализа оценок совместимости и дескрипторов.

Предоставляет функции для вычисления описательных статистик,
нормализации распределений, обнаружения выбросов и агрегации
наборов числовых оценок.

Экспортирует:
    StatsConfig         — параметры статистического анализа
    describe            — описательные статистики массива
    zscore_array        — z-оценки (стандартизация)
    iqr                 — межквартильный размах
    winsorize           — ограничение выбросов по квантилям
    percentile_rank     — ранг значения в процентилях
    outlier_mask        — маска выбросов (метод IQR)
    running_stats       — накопительные статистики
    weighted_mean       — взвешенное среднее
    weighted_std        — взвешенное стандартное отклонение
    batch_describe      — пакетное описание списка массивов
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ─── StatsConfig ──────────────────────────────────────────────────────────────

@dataclass
class StatsConfig:
    """Параметры статистического анализа.

    Attributes:
        outlier_iqr_k:  Множитель IQR для определения выбросов (> 0).
        winsor_low:     Нижний квантиль для winsorize (0 ≤ winsor_low < winsor_high).
        winsor_high:    Верхний квантиль для winsorize (winsor_low < winsor_high ≤ 1).
        ddof:           Степени свободы для std/var (0 или 1).
    """
    outlier_iqr_k: float = 1.5
    winsor_low:    float = 0.05
    winsor_high:   float = 0.95
    ddof:          int   = 0

    def __post_init__(self) -> None:
        if self.outlier_iqr_k <= 0:
            raise ValueError(
                f"outlier_iqr_k must be > 0, got {self.outlier_iqr_k}"
            )
        if not (0.0 <= self.winsor_low < self.winsor_high <= 1.0):
            raise ValueError(
                f"winsor_low/high must satisfy 0 ≤ low < high ≤ 1, "
                f"got {self.winsor_low}/{self.winsor_high}"
            )
        if self.ddof not in (0, 1):
            raise ValueError(f"ddof must be 0 or 1, got {self.ddof}")


# ─── describe ─────────────────────────────────────────────────────────────────

def describe(
    arr: np.ndarray,
    cfg: Optional[StatsConfig] = None,
) -> Dict[str, float]:
    """Вычисляет описательные статистики одномерного массива.

    Args:
        arr: 1-D массив числовых значений (не пустой).
        cfg: Параметры (None → StatsConfig()).

    Returns:
        Словарь: min, max, mean, std, median, q25, q75, iqr.

    Raises:
        ValueError: Если arr не 1-D или пуст.
    """
    if cfg is None:
        cfg = StatsConfig()
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"arr must be 1-D, got ndim={a.ndim}")
    if len(a) == 0:
        raise ValueError("arr must not be empty")

    q25, median, q75 = float(np.percentile(a, 25)), float(np.median(a)), float(np.percentile(a, 75))
    return {
        "min":    float(np.min(a)),
        "max":    float(np.max(a)),
        "mean":   float(np.mean(a)),
        "std":    float(np.std(a, ddof=cfg.ddof)),
        "median": median,
        "q25":    q25,
        "q75":    q75,
        "iqr":    q75 - q25,
    }


# ─── zscore_array ─────────────────────────────────────────────────────────────

def zscore_array(
    arr: np.ndarray,
    cfg: Optional[StatsConfig] = None,
) -> np.ndarray:
    """Стандартизирует массив (z-оценки): (x - mean) / std.

    Если std == 0, возвращает нулевой массив той же формы.

    Args:
        arr: 1-D массив числовых значений (не пустой).
        cfg: Параметры.

    Returns:
        Массив float64 тех же размеров.

    Raises:
        ValueError: Если arr не 1-D или пуст.
    """
    if cfg is None:
        cfg = StatsConfig()
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"arr must be 1-D, got ndim={a.ndim}")
    if len(a) == 0:
        raise ValueError("arr must not be empty")

    mu  = np.mean(a)
    std = np.std(a, ddof=cfg.ddof)
    if std < 1e-12:
        return np.zeros_like(a)
    return (a - mu) / std


# ─── iqr ──────────────────────────────────────────────────────────────────────

def iqr(arr: np.ndarray) -> float:
    """Вычисляет межквартильный размах (Q75 - Q25).

    Args:
        arr: 1-D массив числовых значений (не пустой).

    Returns:
        float ≥ 0.

    Raises:
        ValueError: Если arr не 1-D или пуст.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"arr must be 1-D, got ndim={a.ndim}")
    if len(a) == 0:
        raise ValueError("arr must not be empty")
    return float(np.percentile(a, 75) - np.percentile(a, 25))


# ─── winsorize ────────────────────────────────────────────────────────────────

def winsorize(
    arr: np.ndarray,
    cfg: Optional[StatsConfig] = None,
) -> np.ndarray:
    """Ограничивает выбросы по квантилям [winsor_low, winsor_high].

    Args:
        arr: 1-D массив числовых значений (не пустой).
        cfg: Параметры (winsor_low, winsor_high).

    Returns:
        Массив float64 с теми же размерами, значения зажаты в [lo, hi].

    Raises:
        ValueError: Если arr не 1-D или пуст.
    """
    if cfg is None:
        cfg = StatsConfig()
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"arr must be 1-D, got ndim={a.ndim}")
    if len(a) == 0:
        raise ValueError("arr must not be empty")

    lo = float(np.percentile(a, cfg.winsor_low  * 100))
    hi = float(np.percentile(a, cfg.winsor_high * 100))
    return np.clip(a, lo, hi)


# ─── percentile_rank ──────────────────────────────────────────────────────────

def percentile_rank(arr: np.ndarray, value: float) -> float:
    """Вычисляет ранг значения в массиве в процентилях [0, 100].

    Args:
        arr:   1-D массив числовых значений (не пустой).
        value: Значение для ранжирования.

    Returns:
        float в [0, 100].

    Raises:
        ValueError: Если arr не 1-D или пуст.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"arr must be 1-D, got ndim={a.ndim}")
    if len(a) == 0:
        raise ValueError("arr must not be empty")
    n_below = int(np.sum(a < value))
    return float(n_below) / float(len(a)) * 100.0


# ─── outlier_mask ─────────────────────────────────────────────────────────────

def outlier_mask(
    arr: np.ndarray,
    cfg: Optional[StatsConfig] = None,
) -> np.ndarray:
    """Возвращает булеву маску выбросов методом IQR.

    Выброс: значение < Q25 - k*IQR или > Q75 + k*IQR.

    Args:
        arr: 1-D массив числовых значений (не пустой).
        cfg: Параметры (outlier_iqr_k).

    Returns:
        Булев массив той же длины (True = выброс).

    Raises:
        ValueError: Если arr не 1-D или пуст.
    """
    if cfg is None:
        cfg = StatsConfig()
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"arr must be 1-D, got ndim={a.ndim}")
    if len(a) == 0:
        raise ValueError("arr must not be empty")

    q25 = float(np.percentile(a, 25))
    q75 = float(np.percentile(a, 75))
    spread = (q75 - q25) * cfg.outlier_iqr_k
    return (a < q25 - spread) | (a > q75 + spread)


# ─── running_stats ────────────────────────────────────────────────────────────

def running_stats(arr: np.ndarray) -> Dict[str, np.ndarray]:
    """Вычисляет накопительные (prefix) статистики.

    Args:
        arr: 1-D массив числовых значений (не пустой).

    Returns:
        Словарь с ключами 'cumsum', 'cummax', 'cummin', 'cummean'.
        Все массивы имеют тот же размер и тип float64.

    Raises:
        ValueError: Если arr не 1-D или пуст.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"arr must be 1-D, got ndim={a.ndim}")
    if len(a) == 0:
        raise ValueError("arr must not be empty")

    cumsum  = np.cumsum(a)
    counts  = np.arange(1, len(a) + 1, dtype=np.float64)
    return {
        "cumsum":  cumsum,
        "cummax":  np.maximum.accumulate(a),
        "cummin":  np.minimum.accumulate(a),
        "cummean": cumsum / counts,
    }


# ─── weighted_mean ────────────────────────────────────────────────────────────

def weighted_mean(
    arr:     np.ndarray,
    weights: np.ndarray,
) -> float:
    """Вычисляет взвешенное среднее.

    Args:
        arr:     1-D массив значений (не пустой).
        weights: 1-D массив неотрицательных весов той же длины.
                 Сумма весов должна быть > 0.

    Returns:
        float — взвешенное среднее.

    Raises:
        ValueError: Если массивы не 1-D, пусты, имеют разную длину
                    или сумма весов <= 0.
    """
    a = np.asarray(arr,     dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"arr must be 1-D, got ndim={a.ndim}")
    if w.ndim != 1:
        raise ValueError(f"weights must be 1-D, got ndim={w.ndim}")
    if len(a) == 0:
        raise ValueError("arr must not be empty")
    if len(a) != len(w):
        raise ValueError(
            f"arr and weights must have same length, got {len(a)} vs {len(w)}"
        )
    total_w = float(np.sum(w))
    if total_w <= 0.0:
        raise ValueError(f"Sum of weights must be > 0, got {total_w}")
    return float(np.sum(a * w) / total_w)


# ─── weighted_std ─────────────────────────────────────────────────────────────

def weighted_std(
    arr:     np.ndarray,
    weights: np.ndarray,
) -> float:
    """Вычисляет взвешенное стандартное отклонение.

    Args:
        arr:     1-D массив значений (не пустой).
        weights: 1-D массив неотрицательных весов той же длины.

    Returns:
        float ≥ 0.

    Raises:
        ValueError: Те же условия, что и weighted_mean.
    """
    mu = weighted_mean(arr, weights)
    a  = np.asarray(arr,     dtype=np.float64)
    w  = np.asarray(weights, dtype=np.float64)
    total_w = float(np.sum(w))
    variance = float(np.sum(w * (a - mu) ** 2) / total_w)
    return float(np.sqrt(max(0.0, variance)))


# ─── batch_describe ───────────────────────────────────────────────────────────

def batch_describe(
    arrays: List[np.ndarray],
    cfg: Optional[StatsConfig] = None,
) -> List[Dict[str, float]]:
    """Применяет describe ко списку массивов.

    Args:
        arrays: Список 1-D массивов (не пустой список).
        cfg:    Параметры.

    Returns:
        Список словарей (тот же порядок, что и arrays).

    Raises:
        ValueError: Если список arrays пуст.
    """
    if not arrays:
        raise ValueError("arrays must not be empty")
    return [describe(a, cfg) for a in arrays]
