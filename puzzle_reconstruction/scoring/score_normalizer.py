"""Нормализация матриц оценок совместимости фрагментов пазла.

Модуль предоставляет методы масштабирования и трансформации матриц
попарных оценок (minmax, z-score, ранговая, softmax, sigmoid),
а также утилиты для взвешенного объединения нескольких матриц.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


_NORM_METHODS = {"minmax", "zscore", "rank", "softmax", "sigmoid"}


# ─── NormMethod ───────────────────────────────────────────────────────────────

@dataclass
class NormMethod:
    """Параметры метода нормализации матрицы оценок.

    Атрибуты:
        method:      'minmax' | 'zscore' | 'rank' | 'softmax' | 'sigmoid'.
        axis:        Ось для softmax (0 — по столбцам, 1 — по строкам, None — глобально).
        temperature: Температура для softmax (> 0).
        eps:         Малая добавка для численной стабильности (> 0).
    """

    method: str = "minmax"
    axis: Optional[int] = None
    temperature: float = 1.0
    eps: float = 1e-10

    def __post_init__(self) -> None:
        if self.method not in _NORM_METHODS:
            raise ValueError(
                f"method должен быть одним из {_NORM_METHODS}, "
                f"получено '{self.method}'"
            )
        if self.axis is not None and self.axis not in (0, 1):
            raise ValueError(
                f"axis должен быть 0, 1 или None, получено {self.axis}"
            )
        if self.temperature <= 0.0:
            raise ValueError(
                f"temperature должна быть > 0, получено {self.temperature}"
            )
        if self.eps <= 0.0:
            raise ValueError(
                f"eps должен быть > 0, получено {self.eps}"
            )


# ─── NormalizedMatrix ─────────────────────────────────────────────────────────

@dataclass
class NormalizedMatrix:
    """Результат нормализации матрицы оценок.

    Атрибуты:
        method:  Применённый метод.
        data:    Нормализованная матрица (2-D float64).
        min_val: Минимальное значение исходной матрицы.
        max_val: Максимальное значение исходной матрицы.
    """

    method: str
    data: np.ndarray
    min_val: float
    max_val: float

    @property
    def shape(self) -> Tuple[int, ...]:
        """Форма матрицы."""
        return self.data.shape


# ─── minmax_normalize_matrix ──────────────────────────────────────────────────

def minmax_normalize_matrix(
    matrix: np.ndarray,
    eps: float = 1e-10,
) -> NormalizedMatrix:
    """Нормализовать матрицу в диапазон [0, 1] методом minmax.

    Аргументы:
        matrix: Двумерная матрица (N × M, float).
        eps:    Малая добавка для предотвращения деления на ноль (> 0).

    Возвращает:
        NormalizedMatrix.

    Исключения:
        ValueError: Если matrix не 2-D или eps <= 0.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"matrix должна быть 2-D, получено ndim={mat.ndim}")
    if eps <= 0.0:
        raise ValueError(f"eps должен быть > 0, получено {eps}")

    min_val = float(mat.min())
    max_val = float(mat.max())
    span = max_val - min_val
    if span < eps:
        normalized = np.zeros_like(mat)
    else:
        normalized = (mat - min_val) / span
    return NormalizedMatrix(method="minmax", data=normalized,
                            min_val=min_val, max_val=max_val)


# ─── zscore_normalize_matrix ──────────────────────────────────────────────────

def zscore_normalize_matrix(
    matrix: np.ndarray,
    eps: float = 1e-10,
) -> NormalizedMatrix:
    """Нормализовать матрицу методом z-score (μ=0, σ=1).

    Аргументы:
        matrix: Двумерная матрица.
        eps:    Малая добавка для предотвращения деления на ноль (> 0).

    Возвращает:
        NormalizedMatrix (min_val/max_val — исходные статистики).

    Исключения:
        ValueError: Если matrix не 2-D.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"matrix должна быть 2-D, получено ndim={mat.ndim}")
    if eps <= 0.0:
        raise ValueError(f"eps должен быть > 0, получено {eps}")

    mu = mat.mean()
    sigma = mat.std()
    if sigma < eps:
        normalized = np.zeros_like(mat)
    else:
        normalized = (mat - mu) / sigma
    return NormalizedMatrix(method="zscore", data=normalized,
                            min_val=float(mu), max_val=float(sigma))


# ─── rank_normalize_matrix ────────────────────────────────────────────────────

def rank_normalize_matrix(
    matrix: np.ndarray,
) -> NormalizedMatrix:
    """Нормализовать матрицу по рангам элементов (0 — наименьший, 1 — наибольший).

    Аргументы:
        matrix: Двумерная матрица.

    Возвращает:
        NormalizedMatrix.

    Исключения:
        ValueError: Если matrix не 2-D.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"matrix должна быть 2-D, получено ndim={mat.ndim}")

    flat = mat.ravel()
    n = len(flat)
    order = np.argsort(flat)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    if n > 1:
        ranks /= float(n - 1)
    normalized = ranks.reshape(mat.shape)
    return NormalizedMatrix(method="rank", data=normalized,
                            min_val=float(mat.min()), max_val=float(mat.max()))


# ─── softmax_normalize_matrix ─────────────────────────────────────────────────

def softmax_normalize_matrix(
    matrix: np.ndarray,
    axis: Optional[int] = None,
    temperature: float = 1.0,
) -> NormalizedMatrix:
    """Применить softmax к матрице оценок.

    Аргументы:
        matrix:      Двумерная матрица.
        axis:        Ось (0 — по столбцам, 1 — по строкам, None — глобально).
        temperature: Температура (> 0).

    Возвращает:
        NormalizedMatrix.

    Исключения:
        ValueError: Если matrix не 2-D или параметры некорректны.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"matrix должна быть 2-D, получено ndim={mat.ndim}")
    if axis is not None and axis not in (0, 1):
        raise ValueError(f"axis должен быть 0, 1 или None, получено {axis}")
    if temperature <= 0.0:
        raise ValueError(f"temperature должна быть > 0, получено {temperature}")

    scaled = mat / temperature
    if axis is None:
        shifted = scaled - scaled.max()
        exp = np.exp(shifted)
        normalized = exp / exp.sum()
    else:
        shifted = scaled - scaled.max(axis=axis, keepdims=True)
        exp = np.exp(shifted)
        normalized = exp / (exp.sum(axis=axis, keepdims=True) + 1e-300)

    return NormalizedMatrix(method="softmax", data=normalized,
                            min_val=float(mat.min()), max_val=float(mat.max()))


# ─── sigmoid_normalize_matrix ─────────────────────────────────────────────────

def sigmoid_normalize_matrix(
    matrix: np.ndarray,
) -> NormalizedMatrix:
    """Применить sigmoid к каждому элементу матрицы.

    Аргументы:
        matrix: Двумерная матрица.

    Возвращает:
        NormalizedMatrix (значения в (0, 1)).

    Исключения:
        ValueError: Если matrix не 2-D.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"matrix должна быть 2-D, получено ndim={mat.ndim}")

    normalized = 1.0 / (1.0 + np.exp(-mat))
    return NormalizedMatrix(method="sigmoid", data=normalized,
                            min_val=float(mat.min()), max_val=float(mat.max()))


# ─── normalize_score_matrix ───────────────────────────────────────────────────

def normalize_score_matrix(
    matrix: np.ndarray,
    cfg: Optional[NormMethod] = None,
) -> NormalizedMatrix:
    """Нормализовать матрицу оценок согласно конфигурации.

    Аргументы:
        matrix: Двумерная матрица.
        cfg:    Параметры (None → NormMethod()).

    Возвращает:
        NormalizedMatrix.
    """
    if cfg is None:
        cfg = NormMethod()

    if cfg.method == "minmax":
        return minmax_normalize_matrix(matrix, cfg.eps)
    if cfg.method == "zscore":
        return zscore_normalize_matrix(matrix, cfg.eps)
    if cfg.method == "rank":
        return rank_normalize_matrix(matrix)
    if cfg.method == "softmax":
        return softmax_normalize_matrix(matrix, cfg.axis, cfg.temperature)
    # sigmoid
    return sigmoid_normalize_matrix(matrix)


# ─── combine_score_matrices ───────────────────────────────────────────────────

def combine_score_matrices(
    matrices: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """Взвешенно объединить несколько матриц оценок.

    Аргументы:
        matrices: Список матриц одинакового размера (2-D).
        weights:  Веса (None → равные). Все веса должны быть >= 0.

    Возвращает:
        Взвешенная сумма матриц (float64), нормированная на сумму весов.

    Исключения:
        ValueError: Если matrices пустой, формы не совпадают или сумма весов = 0.
    """
    if not matrices:
        raise ValueError("matrices не должен быть пустым")

    mats = [np.asarray(m, dtype=np.float64) for m in matrices]
    shape = mats[0].shape
    for i, m in enumerate(mats[1:], start=1):
        if m.shape != shape:
            raise ValueError(
                f"Форма матрицы {i} {m.shape} не совпадает с ожидаемой {shape}"
            )

    if weights is None:
        weights = [1.0] * len(matrices)
    if len(weights) != len(matrices):
        raise ValueError(
            f"Число весов {len(weights)} != число матриц {len(matrices)}"
        )
    weights = [float(w) for w in weights]
    if any(w < 0 for w in weights):
        raise ValueError("Все веса должны быть >= 0")
    total = sum(weights)
    if total < 1e-12:
        raise ValueError("Сумма весов должна быть > 0")

    result = np.zeros(shape, dtype=np.float64)
    for m, w in zip(mats, weights):
        result += m * w
    return result / total


# ─── batch_normalize_matrices ─────────────────────────────────────────────────

def batch_normalize_matrices(
    matrices: List[np.ndarray],
    cfg: Optional[NormMethod] = None,
) -> List[NormalizedMatrix]:
    """Нормализовать несколько матриц.

    Аргументы:
        matrices: Список матриц.
        cfg:      Параметры.

    Возвращает:
        Список NormalizedMatrix.
    """
    return [normalize_score_matrix(m, cfg) for m in matrices]
