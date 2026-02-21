"""
Метрики расстояния и сходства для фрагментов документа.

Предоставляет реализации классических и специализированных метрик
для сравнения дескрипторов, контуров и точечных множеств.

Экспортирует:
    euclidean_distance      — Евклидово расстояние
    cosine_distance         — косинусное расстояние (1 − косинус угла)
    cosine_similarity       — косинусное сходство ∈ [−1, 1]
    manhattan_distance      — расстояние Манхэттен
    chebyshev_distance      — расстояние Чебышёва
    hausdorff_distance      — расстояние Хаусдорфа между точечными множествами
    chamfer_distance        — расстояние Шамфера
    normalized_distance     — нормировка расстояния к [0, 1]
    pairwise_distances      — матрица попарных расстояний
    nearest_neighbor_dist   — расстояние до ближайшего соседа
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np


# ─── Публичные функции ────────────────────────────────────────────────────────

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Евклидово расстояние между двумя векторами.

    Args:
        a: Вектор float (d,).
        b: Вектор float (d,) той же размерности.

    Returns:
        Евклидово расстояние ≥ 0.

    Raises:
        ValueError: Если размерности не совпадают.
    """
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"Shape mismatch: a={a_arr.shape}, b={b_arr.shape}"
        )
    return float(np.sqrt(np.sum((a_arr - b_arr) ** 2)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусное сходство между двумя векторами.

    Args:
        a: Вектор float (d,).
        b: Вектор float (d,).

    Returns:
        Значение ∈ [−1, 1]; 1 = идентичны по направлению.
        Нулевые векторы дают 0.0.

    Raises:
        ValueError: Если размерности не совпадают.
    """
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"Shape mismatch: a={a_arr.shape}, b={b_arr.shape}"
        )
    norm_a = float(np.linalg.norm(a_arr))
    norm_b = float(np.linalg.norm(b_arr))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.clip(np.dot(a_arr, b_arr) / (norm_a * norm_b), -1.0, 1.0))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусное расстояние (1 − cosine_similarity).

    Args:
        a: Вектор float (d,).
        b: Вектор float (d,).

    Returns:
        Значение ∈ [0, 2].

    Raises:
        ValueError: Если размерности не совпадают.
    """
    return float(np.clip(1.0 - cosine_similarity(a, b), 0.0, 2.0))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Расстояние Манхэттен (L1-норма разности).

    Args:
        a: Вектор float (d,).
        b: Вектор float (d,).

    Returns:
        Сумма абсолютных разностей ≥ 0.

    Raises:
        ValueError: Если размерности не совпадают.
    """
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"Shape mismatch: a={a_arr.shape}, b={b_arr.shape}"
        )
    return float(np.sum(np.abs(a_arr - b_arr)))


def chebyshev_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Расстояние Чебышёва (L∞-норма разности).

    Args:
        a: Вектор float (d,).
        b: Вектор float (d,).

    Returns:
        Максимальная абсолютная разность ≥ 0.

    Raises:
        ValueError: Если размерности не совпадают.
    """
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"Shape mismatch: a={a_arr.shape}, b={b_arr.shape}"
        )
    return float(np.max(np.abs(a_arr - b_arr)))


def hausdorff_distance(
    set_a: np.ndarray,
    set_b: np.ndarray,
) -> float:
    """Расстояние Хаусдорфа между двумя точечными множествами.

    H(A, B) = max(h(A,B), h(B,A)), где h(A,B) = max_{a∈A} min_{b∈B} ||a−b||.

    Args:
        set_a: Массив float (N, d) — первое множество точек.
        set_b: Массив float (M, d) — второе множество точек.

    Returns:
        Расстояние Хаусдорфа ≥ 0.

    Raises:
        ValueError: Если множества пустые или размерности не совпадают.
    """
    A = np.asarray(set_a, dtype=np.float64)
    B = np.asarray(set_b, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("set_a and set_b must be 2-D arrays")
    if A.shape[0] == 0 or B.shape[0] == 0:
        raise ValueError("Point sets must not be empty")
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"Dimension mismatch: set_a d={A.shape[1]}, set_b d={B.shape[1]}"
        )
    # h(A, B)
    diffs_ab = A[:, None, :] - B[None, :, :]   # (N, M, d)
    dists_ab = np.sqrt(np.sum(diffs_ab ** 2, axis=2))  # (N, M)
    h_ab = float(np.max(np.min(dists_ab, axis=1)))
    # h(B, A)
    h_ba = float(np.max(np.min(dists_ab, axis=0)))
    return max(h_ab, h_ba)


def chamfer_distance(
    set_a: np.ndarray,
    set_b: np.ndarray,
) -> float:
    """Расстояние Шамфера между двумя точечными множествами.

    C(A, B) = (1/|A|) Σ_{a∈A} min_{b∈B} ||a−b||
            + (1/|B|) Σ_{b∈B} min_{a∈A} ||b−a||.

    Args:
        set_a: Массив float (N, d).
        set_b: Массив float (M, d).

    Returns:
        Расстояние Шамфера ≥ 0.

    Raises:
        ValueError: Если множества пустые или размерности не совпадают.
    """
    A = np.asarray(set_a, dtype=np.float64)
    B = np.asarray(set_b, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("set_a and set_b must be 2-D arrays")
    if A.shape[0] == 0 or B.shape[0] == 0:
        raise ValueError("Point sets must not be empty")
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"Dimension mismatch: set_a d={A.shape[1]}, set_b d={B.shape[1]}"
        )
    diffs = A[:, None, :] - B[None, :, :]   # (N, M, d)
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))  # (N, M)
    term_ab = float(np.mean(np.min(dists, axis=1)))
    term_ba = float(np.mean(np.min(dists, axis=0)))
    return term_ab + term_ba


def normalized_distance(
    dist: float,
    max_dist: float,
) -> float:
    """Нормировать расстояние к диапазону [0, 1].

    Args:
        dist:     Исходное расстояние (≥ 0).
        max_dist: Максимальное ожидаемое расстояние (> 0).

    Returns:
        Нормированное расстояние ∈ [0, 1].

    Raises:
        ValueError: Если ``dist`` < 0 или ``max_dist`` ≤ 0.
    """
    if dist < 0:
        raise ValueError(f"dist must be >= 0, got {dist}")
    if max_dist <= 0:
        raise ValueError(f"max_dist must be > 0, got {max_dist}")
    return float(np.clip(dist / max_dist, 0.0, 1.0))


def pairwise_distances(
    X: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """Матрица попарных расстояний между строками X.

    Args:
        X:      Матрица float (N, d).
        metric: ``'euclidean'``, ``'cosine'``, ``'manhattan'``,
                ``'chebyshev'``.

    Returns:
        Симметричная матрица float64 (N, N) расстояний.

    Raises:
        ValueError: Если X не 2-D или ``metric`` неизвестен.
    """
    _METRICS = {
        "euclidean": euclidean_distance,
        "cosine":    cosine_distance,
        "manhattan": manhattan_distance,
        "chebyshev": chebyshev_distance,
    }
    if metric not in _METRICS:
        raise ValueError(
            f"metric must be one of {sorted(_METRICS)}, got {metric!r}"
        )
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2-D, got ndim={X_arr.ndim}")
    n = len(X_arr)
    fn: Callable[[np.ndarray, np.ndarray], float] = _METRICS[metric]
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = fn(X_arr[i], X_arr[j])
            D[i, j] = d
            D[j, i] = d
    return D


def nearest_neighbor_dist(
    query: np.ndarray,
    candidates: np.ndarray,
) -> float:
    """Расстояние от точки до ближайшего соседа в множестве кандидатов.

    Args:
        query:      Вектор float (d,).
        candidates: Матрица float (N, d) кандидатов.

    Returns:
        Минимальное Евклидово расстояние ≥ 0.

    Raises:
        ValueError: Если ``candidates`` пустой или не 2-D.
    """
    q = np.asarray(query, dtype=np.float64).ravel()
    C = np.asarray(candidates, dtype=np.float64)
    if C.ndim != 2:
        raise ValueError(f"candidates must be 2-D, got ndim={C.ndim}")
    if C.shape[0] == 0:
        raise ValueError("candidates must not be empty")
    diffs = C - q[None, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    return float(np.min(dists))
