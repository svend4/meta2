"""
Утилиты интерполяции для сигналов, оценок и позиций.

Предоставляет функции для линейной интерполяции, ресамплинга 1D-сигналов,
заполнения пропусков, билинейной интерполяции 2D-сеток и пакетных операций.
Используется при оптимизации позиций фрагментов, интерполяции матриц оценок
и обработке временных рядов в пайплайне сборки.

Экспортирует:
    InterpolationConfig  — параметры интерполяции
    lerp                 — линейная интерполяция двух скаляров
    lerp_array           — линейная интерполяция двух массивов
    bilinear_interpolate — билинейная интерполяция значения в 2D-сетке
    resample_1d          — ресамплинг 1D-сигнала до целевой длины
    fill_missing         — заполнение NaN-значений интерполяцией (1D)
    interpolate_scores   — мягкое усреднение матрицы оценок
    smooth_interpolate   — скользящая средняя с линейной интерполяцией краёв
    batch_resample       — пакетный ресамплинг списка сигналов
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


# ─── InterpolationConfig ──────────────────────────────────────────────────────

@dataclass
class InterpolationConfig:
    """Параметры интерполяции.

    Attributes:
        method:   Метод интерполяции: 'linear' | 'nearest'.
        clamp:    Ограничивать ли результат диапазоном исходных значений.
        fill_val: Значение для заполнения за пределами диапазона (если clamp=False).
    """
    method:   str   = "linear"
    clamp:    bool  = True
    fill_val: float = 0.0

    def __post_init__(self) -> None:
        if self.method not in ("linear", "nearest"):
            raise ValueError(
                f"method должен быть 'linear' или 'nearest', "
                f"получено {self.method!r}"
            )


# ─── lerp ─────────────────────────────────────────────────────────────────────

def lerp(a: float, b: float, t: float) -> float:
    """Линейная интерполяция между двумя скалярами.

    Args:
        a: Начальное значение.
        b: Конечное значение.
        t: Параметр интерполяции ∈ [0, 1] (0 → a, 1 → b).

    Returns:
        a + t * (b - a).

    Raises:
        ValueError: Если t < 0 или t > 1.
    """
    if t < 0.0 or t > 1.0:
        raise ValueError(f"t должен быть в [0, 1], получено {t}")
    return float(a + t * (b - a))


# ─── lerp_array ───────────────────────────────────────────────────────────────

def lerp_array(
    a: np.ndarray,
    b: np.ndarray,
    t: float,
) -> np.ndarray:
    """Линейная интерполяция двух массивов одинаковой формы.

    Args:
        a: Первый массив.
        b: Второй массив.
        t: Параметр интерполяции ∈ [0, 1].

    Returns:
        Массив float64 той же формы: a + t * (b - a).

    Raises:
        ValueError: Если t вне [0, 1] или формы массивов не совпадают.
    """
    if t < 0.0 or t > 1.0:
        raise ValueError(f"t должен быть в [0, 1], получено {t}")
    a_ = np.asarray(a, dtype=np.float64)
    b_ = np.asarray(b, dtype=np.float64)
    if a_.shape != b_.shape:
        raise ValueError(
            f"Формы массивов не совпадают: {a_.shape} != {b_.shape}"
        )
    return a_ + t * (b_ - a_)


# ─── bilinear_interpolate ─────────────────────────────────────────────────────

def bilinear_interpolate(
    grid: np.ndarray,
    x: float,
    y: float,
) -> float:
    """Билинейная интерполяция значения в непрерывной точке (x, y) 2D-сетки.

    Args:
        grid: 2D массив значений (H, W).
        x:    Горизонтальная координата ∈ [0, W-1].
        y:    Вертикальная координата ∈ [0, H-1].

    Returns:
        Интерполированное значение float.

    Raises:
        ValueError: Если grid не 2D или координаты вне сетки.
    """
    g = np.asarray(grid, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError(f"grid должна быть 2D, получено ndim={g.ndim}")
    H, W = g.shape
    if not (0.0 <= x <= W - 1) or not (0.0 <= y <= H - 1):
        raise ValueError(
            f"Координаты ({x}, {y}) вне диапазона [0, {W-1}] × [0, {H-1}]"
        )
    x0, y0 = int(x), int(y)
    x1 = min(x0 + 1, W - 1)
    y1 = min(y0 + 1, H - 1)
    dx, dy = x - x0, y - y0
    return float(
        g[y0, x0] * (1 - dx) * (1 - dy)
        + g[y0, x1] * dx * (1 - dy)
        + g[y1, x0] * (1 - dx) * dy
        + g[y1, x1] * dx * dy
    )


# ─── resample_1d ──────────────────────────────────────────────────────────────

def resample_1d(
    arr: np.ndarray,
    target_len: int,
    cfg: InterpolationConfig | None = None,
) -> np.ndarray:
    """Ресамплинг 1D-сигнала до заданной длины.

    Args:
        arr:        1D массив (не пустой).
        target_len: Целевая длина (>= 1).
        cfg:        Параметры интерполяции (None → InterpolationConfig()).

    Returns:
        float64 массив длины target_len.

    Raises:
        ValueError: Если arr не 1D, пуст или target_len < 1.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"arr должен быть 1D, получено ndim={a.ndim}")
    if a.size == 0:
        raise ValueError("arr не должен быть пустым")
    if target_len < 1:
        raise ValueError(f"target_len должен быть >= 1, получено {target_len}")
    if cfg is None:
        cfg = InterpolationConfig()

    if len(a) == target_len:
        return a.copy()

    src_x = np.linspace(0.0, 1.0, len(a))
    dst_x = np.linspace(0.0, 1.0, target_len)

    if cfg.method == "nearest":
        indices = np.round(dst_x * (len(a) - 1)).astype(int)
        indices = np.clip(indices, 0, len(a) - 1)
        return a[indices]
    # linear
    return np.interp(dst_x, src_x, a)


# ─── fill_missing ─────────────────────────────────────────────────────────────

def fill_missing(arr: np.ndarray) -> np.ndarray:
    """Заполняет NaN-значения линейной интерполяцией по соседним элементам.

    Только для 1D массивов. NaN на краях заполняются ближайшим не-NaN.

    Args:
        arr: 1D массив float с возможными NaN.

    Returns:
        float64 массив без NaN (той же длины).

    Raises:
        ValueError: Если arr не 1D или пуст.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"arr должен быть 1D, получено ndim={a.ndim}")
    if a.size == 0:
        raise ValueError("arr не должен быть пустым")
    if not np.any(np.isnan(a)):
        return a.copy()

    result = a.copy()
    idx = np.arange(len(a))
    valid = ~np.isnan(a)
    if not valid.any():
        return np.zeros_like(a)
    result = np.interp(idx, idx[valid], a[valid])
    return result


# ─── interpolate_scores ───────────────────────────────────────────────────────

def interpolate_scores(
    matrix: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Сглаживает матрицу оценок смешиванием с транспонированной.

    Приводит к симметрии: result[i,j] = alpha * M[i,j] + (1-alpha) * M[j,i].

    Args:
        matrix: Квадратная 2D матрица float.
        alpha:  Вес оригинальной матрицы ∈ [0, 1].

    Returns:
        Симметричная матрица float64.

    Raises:
        ValueError: Если matrix не квадратная 2D или alpha вне [0, 1].
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"matrix должна быть 2D, получено ndim={m.ndim}")
    if m.shape[0] != m.shape[1]:
        raise ValueError(
            f"matrix должна быть квадратной, получено {m.shape}"
        )
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha должен быть в [0, 1], получено {alpha}")
    return alpha * m + (1.0 - alpha) * m.T


# ─── smooth_interpolate ───────────────────────────────────────────────────────

def smooth_interpolate(
    arr: np.ndarray,
    window: int = 3,
) -> np.ndarray:
    """Скользящая средняя с линейной интерполяцией на краях.

    Args:
        arr:    1D массив (не пустой).
        window: Размер окна (>= 1).

    Returns:
        float64 массив той же длины.

    Raises:
        ValueError: Если arr не 1D, пуст или window < 1.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"arr должен быть 1D, получено ndim={a.ndim}")
    if a.size == 0:
        raise ValueError("arr не должен быть пустым")
    if window < 1:
        raise ValueError(f"window должен быть >= 1, получено {window}")
    if window == 1 or len(a) == 1:
        return a.copy()

    hw = window // 2
    n = len(a)
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - hw)
        hi = min(n, i + hw + 1)
        result[i] = np.mean(a[lo:hi])
    return result


# ─── batch_resample ───────────────────────────────────────────────────────────

def batch_resample(
    arrays: List[np.ndarray],
    target_len: int,
    cfg: InterpolationConfig | None = None,
) -> List[np.ndarray]:
    """Ресамплинг списка 1D-сигналов до одной длины.

    Args:
        arrays:     Список 1D массивов (непустой).
        target_len: Целевая длина (>= 1).
        cfg:        Параметры интерполяции.

    Returns:
        Список float64 массивов длины target_len.

    Raises:
        ValueError: Если список пуст или target_len < 1.
    """
    if not arrays:
        raise ValueError("arrays не должен быть пустым")
    if target_len < 1:
        raise ValueError(f"target_len должен быть >= 1, получено {target_len}")
    return [resample_1d(a, target_len, cfg) for a in arrays]
