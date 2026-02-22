"""
Построение и обработка матриц расстояний между фрагментами.

Предоставляет функции вычисления попарных расстояний по различным метрикам,
их нормализации и преобразования для последующей сборки пазла.

Классы:
    DistanceConfig   — параметры вычисления расстояний

Функции:
    euclidean_distance_matrix  — матрица евклидовых расстояний
    cosine_distance_matrix     — матрица косинусных расстояний
    manhattan_distance_matrix  — матрица Манхэттенских расстояний
    build_distance_matrix      — единая точка входа по типу метрики
    normalize_distance_matrix  — нормализация матрицы в [0, 1]
    to_similarity_matrix       — преобразование расстояний в схожесть
    threshold_distance_matrix  — обнуление элементов выше порога
    top_k_distance_pairs       — топ-k ближайших пар
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── DistanceConfig ───────────────────────────────────────────────────────────

_VALID_METRICS = frozenset({"euclidean", "cosine", "manhattan"})


@dataclass
class DistanceConfig:
    """Параметры вычисления матрицы расстояний.

    Attributes:
        metric:    Метрика расстояния: 'euclidean', 'cosine', 'manhattan'.
        normalize: Нормализовать матрицу в [0, 1] после вычисления.
        eps:       Небольшое смещение для числовой устойчивости (> 0).
    """
    metric:    str   = "euclidean"
    normalize: bool  = True
    eps:       float = 1e-8

    def __post_init__(self) -> None:
        if self.metric not in _VALID_METRICS:
            raise ValueError(
                f"metric must be one of {sorted(_VALID_METRICS)}, "
                f"got {self.metric!r}"
            )
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps}")


# ─── Публичные функции ────────────────────────────────────────────────────────

def euclidean_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Вычислить матрицу попарных евклидовых расстояний.

    Args:
        X: Матрица признаков (N, d).

    Returns:
        Симметричная матрица float64 (N, N) с нулевой диагональю.

    Raises:
        ValueError: Если X не 2-D.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2-D, got ndim={X_arr.ndim}")
    n = len(X_arr)
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i · x_j
    sq = (X_arr ** 2).sum(axis=1, keepdims=True)
    gram = X_arr @ X_arr.T
    dist2 = sq + sq.T - 2.0 * gram
    dist2 = np.maximum(dist2, 0.0)
    mat = np.sqrt(dist2)
    np.fill_diagonal(mat, 0.0)
    return mat


def cosine_distance_matrix(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Вычислить матрицу попарных косинусных расстояний (1 - cosine_sim).

    Args:
        X:   Матрица признаков (N, d).
        eps: Константа устойчивости.

    Returns:
        Симметричная матрица float64 (N, N) в [0, 2] с нулевой диагональю.

    Raises:
        ValueError: Если X не 2-D.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2-D, got ndim={X_arr.ndim}")
    norms = np.linalg.norm(X_arr, axis=1, keepdims=True) + eps
    X_norm = X_arr / norms
    cos_sim = X_norm @ X_norm.T
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    mat = 1.0 - cos_sim
    np.fill_diagonal(mat, 0.0)
    return mat


def manhattan_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Вычислить матрицу попарных Манхэттенских расстояний.

    Args:
        X: Матрица признаков (N, d).

    Returns:
        Симметричная матрица float64 (N, N) с нулевой диагональю.

    Raises:
        ValueError: Если X не 2-D.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2-D, got ndim={X_arr.ndim}")
    n = len(X_arr)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        diff = np.abs(X_arr - X_arr[i])
        mat[i] = diff.sum(axis=1)
    np.fill_diagonal(mat, 0.0)
    return mat


def build_distance_matrix(
    X: np.ndarray,
    cfg: Optional[DistanceConfig] = None,
) -> np.ndarray:
    """Построить матрицу расстояний по заданной метрике.

    Args:
        X:   Матрица признаков (N, d).
        cfg: Параметры; None → DistanceConfig().

    Returns:
        Матрица float64 (N, N), нормализованная если cfg.normalize=True.

    Raises:
        ValueError: Если X не 2-D.
    """
    if cfg is None:
        cfg = DistanceConfig()

    dispatch = {
        "euclidean": lambda: euclidean_distance_matrix(X),
        "cosine":    lambda: cosine_distance_matrix(X, cfg.eps),
        "manhattan": lambda: manhattan_distance_matrix(X),
    }
    mat = dispatch[cfg.metric]()

    if cfg.normalize:
        mat = normalize_distance_matrix(mat)

    return mat


def normalize_distance_matrix(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Нормализовать матрицу расстояний в [0, 1].

    Использует только внедиагональные элементы для вычисления max.
    Диагональ остаётся нулевой.

    Args:
        mat: Квадратная матрица расстояний (N, N).
        eps: Константа устойчивости.

    Returns:
        Нормализованная матрица float64 (N, N).

    Raises:
        ValueError: Если mat не квадратная 2-D.
    """
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(
            f"mat must be a square 2-D array, got shape {mat.shape}"
        )
    n = mat.shape[0]
    mask_diag = ~np.eye(n, dtype=bool)
    off = mat[mask_diag]
    mx = float(off.max()) if off.size > 0 else 0.0
    if mx > eps:
        result = mat / mx
    else:
        result = mat.copy()
    np.fill_diagonal(result, 0.0)
    return result


def to_similarity_matrix(
    dist_mat: np.ndarray,
    method: str = "inverse",
    sigma: float = 1.0,
) -> np.ndarray:
    """Преобразовать матрицу расстояний в матрицу схожести.

    Args:
        dist_mat: Матрица расстояний (N, N), значения >= 0.
        method:   'inverse' → 1/(1+d); 'gaussian' → exp(-d^2/(2*sigma^2)).
        sigma:    Параметр гауссова ядра (> 0, используется при method='gaussian').

    Returns:
        Матрица схожести float64 (N, N), значения в [0, 1].
        Диагональ = 1.0.

    Raises:
        ValueError: Если method неизвестен или sigma <= 0.
    """
    if method not in ("inverse", "gaussian"):
        raise ValueError(
            f"method must be 'inverse' or 'gaussian', got {method!r}"
        )
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    d = np.asarray(dist_mat, dtype=np.float64)
    if method == "inverse":
        sim = 1.0 / (1.0 + d)
    else:
        sim = np.exp(-(d ** 2) / (2.0 * sigma ** 2))

    np.fill_diagonal(sim, 1.0)
    return sim


def threshold_distance_matrix(
    mat: np.ndarray,
    threshold: float,
    fill: float = 0.0,
) -> np.ndarray:
    """Обнулить (заменить) расстояния выше порога.

    Args:
        mat:       Матрица расстояний (N, N).
        threshold: Порог (элементы > threshold заменяются на fill).
        fill:      Значение замены.

    Returns:
        Новая матрица float64.

    Raises:
        ValueError: Если mat не 2-D.
    """
    m = np.asarray(mat, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"mat must be 2-D, got ndim={m.ndim}")
    result = m.copy()
    result[result > threshold] = fill
    return result


def top_k_distance_pairs(
    mat: np.ndarray,
    k: int,
) -> List[Tuple[int, int, float]]:
    """Вернуть k пар с наименьшим расстоянием (без диагонали и дублей).

    Args:
        mat: Симметричная матрица расстояний (N, N).
        k:   Число пар (>= 1).

    Returns:
        Список кортежей (i, j, distance), i < j, отсортированных по возрастанию.

    Raises:
        ValueError: Если k < 1 или mat не 2-D квадратная.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    m = np.asarray(mat, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(
            f"mat must be a square 2-D array, got shape {m.shape}"
        )
    n = m.shape[0]
    pairs: List[Tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, float(m[i, j])))
    pairs.sort(key=lambda x: x[2])
    return pairs[:k]
