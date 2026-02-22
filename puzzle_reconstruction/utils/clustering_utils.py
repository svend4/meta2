"""
Кластеризация фрагментов документа по признаковым дескрипторам.

Предоставляет инструменты для группировки фрагментов методами
k-средних, иерархической кластеризации и оценки качества разбиения.

Экспортирует:
    ClusterResult          — результат кластеризации
    kmeans_cluster         — кластеризация k-средними
    assign_to_clusters     — назначить точки ближайшим центрам
    compute_inertia        — внутрикластерная дисперсия
    silhouette_score_approx — приближённый силуэтный коэффициент
    hierarchical_cluster   — иерархическая кластеризация
    find_optimal_k         — подбор числа кластеров (метод локтя)
    cluster_indices        — индексы точек каждого кластера
    merge_clusters         — объединить несколько кластеров
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class ClusterResult:
    """Результат кластеризации.

    Attributes:
        labels:     Метки кластеров int (N,); значения 0..n_clusters-1.
        centers:    Центры кластеров float64 (k, d).
        n_clusters: Число кластеров (≥ 1).
        inertia:    Суммарная внутрикластерная дисперсия (≥ 0).
        params:     Параметры алгоритма.
    """
    labels: np.ndarray
    centers: np.ndarray
    n_clusters: int
    inertia: float
    params: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_clusters < 1:
            raise ValueError(
                f"n_clusters must be >= 1, got {self.n_clusters}"
            )
        if self.inertia < 0:
            raise ValueError(
                f"inertia must be >= 0, got {self.inertia}"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ClusterResult(k={self.n_clusters}, "
            f"n={len(self)}, inertia={self.inertia:.4f})"
        )


# ─── Публичные функции ────────────────────────────────────────────────────────

def kmeans_cluster(
    X: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 0,
) -> ClusterResult:
    """Кластеризация k-средних (Lloyd's algorithm).

    Args:
        X:           Матрица признаков float (N, d).
        n_clusters:  Число кластеров (1 ≤ k ≤ N).
        max_iter:    Максимальное число итераций (≥ 1).
        tol:         Порог сходимости (изменение инерции).
        random_state: Сид генератора для воспроизводимости.

    Returns:
        :class:`ClusterResult`.

    Raises:
        ValueError: Если ``n_clusters`` вне [1, N] или ``max_iter`` < 1.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    n, d = X_arr.shape
    if n_clusters < 1 or n_clusters > n:
        raise ValueError(
            f"n_clusters must be in [1, {n}], got {n_clusters}"
        )
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")

    rng = np.random.default_rng(random_state)
    # Инициализация: случайный выбор центров
    init_idx = rng.choice(n, size=n_clusters, replace=False)
    centers = X_arr[init_idx].copy()

    labels = np.zeros(n, dtype=np.int64)
    prev_inertia = float("inf")

    for _ in range(max_iter):
        labels = assign_to_clusters(X_arr, centers)
        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                new_centers[k] = X_arr[mask].mean(axis=0)
            else:
                new_centers[k] = centers[k]
        inertia = compute_inertia(X_arr, labels, new_centers)
        if abs(prev_inertia - inertia) < tol:
            centers = new_centers
            break
        prev_inertia = inertia
        centers = new_centers

    final_inertia = compute_inertia(X_arr, labels, centers)
    return ClusterResult(
        labels=labels,
        centers=centers,
        n_clusters=n_clusters,
        inertia=float(final_inertia),
        params={
            "algorithm": "kmeans",
            "max_iter": max_iter,
            "tol": tol,
        },
    )


def assign_to_clusters(
    X: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """Назначить каждую точку ближайшему центру.

    Args:
        X:       Матрица (N, d).
        centers: Центры кластеров (k, d).

    Returns:
        Метки int64 (N,) — индекс ближайшего центра.

    Raises:
        ValueError: Если X и centers имеют разную размерность признаков.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    C = np.asarray(centers, dtype=np.float64)
    if X_arr.ndim != 2 or C.ndim != 2:
        raise ValueError("X and centers must be 2-D")
    if X_arr.shape[1] != C.shape[1]:
        raise ValueError(
            f"Feature dimensions differ: X={X_arr.shape[1]}, "
            f"centers={C.shape[1]}"
        )
    # (N, 1, d) - (1, k, d) → (N, k, d) → sum → (N, k)
    diffs = X_arr[:, None, :] - C[None, :, :]
    dists = np.sum(diffs ** 2, axis=2)
    return np.argmin(dists, axis=1).astype(np.int64)


def compute_inertia(
    X: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
) -> float:
    """Вычислить суммарную внутрикластерную дисперсию.

    Args:
        X:       Матрица (N, d).
        labels:  Метки кластеров (N,).
        centers: Центры (k, d).

    Returns:
        Сумма квадратов расстояний до центров.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    C = np.asarray(centers, dtype=np.float64)
    labs = np.asarray(labels)
    total = 0.0
    for i, c in enumerate(C):
        mask = labs == i
        if mask.any():
            diff = X_arr[mask] - c
            total += float(np.sum(diff ** 2))
    return total


def silhouette_score_approx(
    X: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Приближённый силуэтный коэффициент кластеризации.

    Для каждой точки вычисляет a(i) (среднее расстояние внутри кластера)
    и b(i) (мин. среднее расстояние до ближайшего чужого кластера).

    Args:
        X:      Матрица (N, d).
        labels: Метки кластеров (N,).

    Returns:
        Средний силуэтный коэффициент ∈ [-1, 1].
        Если менее 2 кластеров — возвращает 0.0.

    Raises:
        ValueError: Если X не 2-D.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    labs = np.asarray(labels)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2-D, got ndim={X_arr.ndim}")
    unique = np.unique(labs)
    if len(unique) < 2:
        return 0.0

    n = len(X_arr)
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        same = labs == labs[i]
        same[i] = False
        if same.any():
            a_i = float(np.mean(np.sqrt(np.sum((X_arr[same] - X_arr[i]) ** 2, axis=1))))
        else:
            a_i = 0.0

        b_i = float("inf")
        for k in unique:
            if k == labs[i]:
                continue
            mask = labs == k
            if mask.any():
                mean_d = float(np.mean(
                    np.sqrt(np.sum((X_arr[mask] - X_arr[i]) ** 2, axis=1))
                ))
                if mean_d < b_i:
                    b_i = mean_d

        m = max(a_i, b_i)
        scores[i] = (b_i - a_i) / m if m > 0 else 0.0

    return float(np.mean(scores))


def hierarchical_cluster(
    dist_matrix: np.ndarray,
    n_clusters: int,
    linkage: str = "single",
) -> np.ndarray:
    """Иерархическая кластеризация агломеративным методом.

    Args:
        dist_matrix: Симметричная матрица расстояний (N, N).
        n_clusters:  Желаемое число кластеров (≥ 1).
        linkage:     Метод слияния: ``'single'``, ``'complete'``,
                     ``'average'``.

    Returns:
        Метки int64 (N,).

    Raises:
        ValueError: Если матрица не квадратная или ``linkage`` неизвестен.
    """
    mat = np.asarray(dist_matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(
            f"dist_matrix must be a square 2-D array, got shape {mat.shape}"
        )
    if linkage not in ("single", "complete", "average"):
        raise ValueError(
            f"linkage must be 'single', 'complete' or 'average', "
            f"got {linkage!r}"
        )
    n = mat.shape[0]
    if n_clusters < 1 or n_clusters > n:
        raise ValueError(
            f"n_clusters must be in [1, {n}], got {n_clusters}"
        )

    # Начальное состояние: каждая точка — отдельный кластер
    cluster_ids = list(range(n))
    clusters: Dict[int, List[int]] = {i: [i] for i in range(n)}
    next_id = n

    while len(clusters) > n_clusters:
        ids = sorted(clusters.keys())
        best_d = float("inf")
        best_i, best_j = -1, -1
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                ci, cj = ids[a], ids[b]
                pts_i = clusters[ci]
                pts_j = clusters[cj]
                dists = [mat[p, q] for p in pts_i for q in pts_j]
                if linkage == "single":
                    d = min(dists)
                elif linkage == "complete":
                    d = max(dists)
                else:
                    d = sum(dists) / len(dists)
                if d < best_d:
                    best_d = d
                    best_i, best_j = ci, cj
        if best_i == -1:
            break
        merged = clusters.pop(best_i) + clusters.pop(best_j)
        clusters[next_id] = merged
        next_id += 1

    labels = np.zeros(n, dtype=np.int64)
    for label, (_, members) in enumerate(clusters.items()):
        for idx in members:
            labels[idx] = label
    return labels


def find_optimal_k(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
) -> int:
    """Подобрать оптимальное число кластеров методом локтя.

    Args:
        X:     Матрица (N, d).
        k_min: Минимальное k (≥ 1).
        k_max: Максимальное k (≤ N).

    Returns:
        Оптимальное k.

    Raises:
        ValueError: Если k_min > k_max или k_min < 1.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    n = len(X_arr)
    if k_min < 1:
        raise ValueError(f"k_min must be >= 1, got {k_min}")
    if k_min > k_max:
        raise ValueError(f"k_min ({k_min}) must be <= k_max ({k_max})")
    k_max = min(k_max, n)
    if k_min > k_max:
        return k_min

    inertias = []
    k_range = list(range(k_min, k_max + 1))
    for k in k_range:
        result = kmeans_cluster(X_arr, k)
        inertias.append(result.inertia)

    if len(inertias) <= 1:
        return k_range[0]

    # Метод локтя: найти точку с максимальным вторым производным снижением
    best_k = k_range[0]
    max_decrease = 0.0
    for i in range(1, len(inertias)):
        decrease = inertias[i - 1] - inertias[i]
        if decrease > max_decrease:
            max_decrease = decrease
            best_k = k_range[i]
    return best_k


def cluster_indices(
    labels: np.ndarray,
    n_clusters: Optional[int] = None,
) -> Dict[int, List[int]]:
    """Получить индексы точек для каждого кластера.

    Args:
        labels:     Метки (N,).
        n_clusters: Если задано — включить все кластеры 0..n_clusters-1
                    (даже пустые).

    Returns:
        Словарь {cluster_label: [indices]}.
    """
    labs = np.asarray(labels)
    unique = sorted(np.unique(labs).tolist())
    result: Dict[int, List[int]] = {}
    if n_clusters is not None:
        for k in range(n_clusters):
            result[k] = []
    for k in unique:
        result[k] = sorted(int(i) for i in np.where(labs == k)[0])
    return result


def merge_clusters(
    labels: np.ndarray,
    ids_to_merge: List[int],
    target_id: int,
) -> np.ndarray:
    """Объединить указанные кластеры в один.

    Args:
        labels:       Метки (N,).
        ids_to_merge: Список меток, которые нужно слить.
        target_id:    Метка результирующего кластера.

    Returns:
        Новый массив меток int64 (N,).
    """
    result = np.asarray(labels, dtype=np.int64).copy()
    for cid in ids_to_merge:
        result[result == cid] = target_id
    return result
