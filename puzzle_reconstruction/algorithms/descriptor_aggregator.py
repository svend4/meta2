"""Агрегация дескрипторов из нескольких источников.

Модуль объединяет дескрипторы (текстурные, SIFT, форм) путём
конкатенации, взвешенного усреднения или PCA-сжатия,
а также вычисляет матрицы расстояний между дескрипторами.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


_AGG_MODES = {"concat", "weighted_avg", "pca", "max", "min"}


# ─── AggregatorConfig ─────────────────────────────────────────────────────────

@dataclass
class AggregatorConfig:
    """Параметры агрегации дескрипторов.

    Атрибуты:
        mode:        Режим: 'concat' | 'weighted_avg' | 'pca' | 'max' | 'min'.
        weights:     Веса источников (словарь {имя: вес}, все >= 0).
        n_components: Число PCA-компонент (>= 1, используется при mode='pca').
        normalize:   L2-нормировать результат.
    """

    mode: str = "concat"
    weights: Dict[str, float] = field(default_factory=dict)
    n_components: int = 32
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.mode not in _AGG_MODES:
            raise ValueError(
                f"mode должен быть одним из {_AGG_MODES}, получено '{self.mode}'"
            )
        for name, w in self.weights.items():
            if w < 0.0:
                raise ValueError(
                    f"Вес '{name}' должен быть >= 0, получено {w}"
                )
        if self.n_components < 1:
            raise ValueError(
                f"n_components должен быть >= 1, получено {self.n_components}"
            )


# ─── AggregatedDescriptor ─────────────────────────────────────────────────────

@dataclass
class AggregatedDescriptor:
    """Агрегированный дескриптор одного фрагмента.

    Атрибуты:
        fragment_id: Идентификатор фрагмента (>= 0).
        vector:      Результирующий вектор (1-D float32).
        mode:        Применённый режим агрегации.
        source_dims: Словарь {источник: размерность}.
    """

    fragment_id: int
    vector: np.ndarray
    mode: str
    source_dims: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        self.vector = np.asarray(self.vector, dtype=np.float32)
        if self.vector.ndim != 1:
            raise ValueError(
                f"vector должен быть 1-D, получено ndim={self.vector.ndim}"
            )

    @property
    def dim(self) -> int:
        """Размерность агрегированного вектора."""
        return int(self.vector.shape[0])


# ─── l2_normalize ─────────────────────────────────────────────────────────────

def l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-нормировать вектор (или каждую строку матрицы).

    Аргументы:
        v: Массив (N,) или (M, N).

    Возвращает:
        Нормированный массив float32.
    """
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-10 else v
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)
    return v / norms


# ─── concatenate_descriptors ──────────────────────────────────────────────────

def concatenate_descriptors(
    descriptors: Dict[str, np.ndarray],
    normalize: bool = True,
) -> np.ndarray:
    """Конкатенировать дескрипторы из нескольких источников.

    Аргументы:
        descriptors: Словарь {имя: вектор (1-D)}.
        normalize:   L2-нормировать результат.

    Возвращает:
        Конкатенированный вектор float32.

    Исключения:
        ValueError: Если descriptors пуст.
    """
    if not descriptors:
        raise ValueError("descriptors не может быть пустым")
    parts = [np.asarray(v, dtype=np.float32).ravel() for v in descriptors.values()]
    result = np.concatenate(parts)
    return l2_normalize(result) if normalize else result


# ─── weighted_average_descriptors ─────────────────────────────────────────────

def weighted_average_descriptors(
    descriptors: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Взвешенное усреднение дескрипторов одинаковой размерности.

    Аргументы:
        descriptors: Словарь {имя: вектор (1-D)}.
        weights:     Словарь весов {имя: вес}. None → равные веса.
        normalize:   L2-нормировать результат.

    Возвращает:
        Взвешенно усреднённый вектор float32.

    Исключения:
        ValueError: Если descriptors пуст или размерности не совпадают.
    """
    if not descriptors:
        raise ValueError("descriptors не может быть пустым")

    names = list(descriptors.keys())
    vecs = [np.asarray(descriptors[n], dtype=np.float32).ravel() for n in names]

    dim = vecs[0].shape[0]
    for v in vecs[1:]:
        if v.shape[0] != dim:
            raise ValueError(
                "Все дескрипторы должны иметь одинаковую размерность "
                f"для weighted_avg. Ожидается {dim}, получено {v.shape[0]}"
            )

    if weights is None:
        w_arr = np.ones(len(names), dtype=np.float32)
    else:
        w_arr = np.array(
            [float(weights.get(n, 1.0)) for n in names], dtype=np.float32
        )

    total = w_arr.sum()
    if total < 1e-10:
        total = 1.0

    result = np.zeros(dim, dtype=np.float32)
    for w, v in zip(w_arr, vecs):
        result += w * v
    result /= total

    return l2_normalize(result) if normalize else result


# ─── pca_reduce ───────────────────────────────────────────────────────────────

def pca_reduce(
    matrix: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Снизить размерность матрицы методом PCA (SVD).

    Аргументы:
        matrix:       Матрица (M × D), M образцов, D признаков.
        n_components: Число компонент (>= 1, <= min(M, D)).

    Возвращает:
        Матрица (M × n_components) float32.

    Исключения:
        ValueError: Если n_components некорректно.
    """
    if n_components < 1:
        raise ValueError(
            f"n_components должен быть >= 1, получено {n_components}"
        )
    X = np.asarray(matrix, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"matrix должна быть 2-D, получено ndim={X.ndim}")
    M, D = X.shape
    k = min(n_components, M, D)

    # Центрирование
    X_c = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    reduced = U[:, :k] * S[:k]
    return reduced.astype(np.float32)


# ─── elementwise_aggregate ────────────────────────────────────────────────────

def elementwise_aggregate(
    descriptors: Dict[str, np.ndarray],
    mode: str = "max",
    normalize: bool = True,
) -> np.ndarray:
    """Поэлементная агрегация (max или min) дескрипторов.

    Аргументы:
        descriptors: Словарь {имя: вектор (1-D) одинаковой размерности}.
        mode:        'max' или 'min'.
        normalize:   L2-нормировать результат.

    Возвращает:
        Агрегированный вектор float32.

    Исключения:
        ValueError: Если descriptors пуст, mode неизвестен или размерности не совпадают.
    """
    if not descriptors:
        raise ValueError("descriptors не может быть пустым")
    if mode not in ("max", "min"):
        raise ValueError(f"mode должен быть 'max' или 'min', получено '{mode}'")

    vecs = [np.asarray(v, dtype=np.float32).ravel() for v in descriptors.values()]
    dim = vecs[0].shape[0]
    for v in vecs[1:]:
        if v.shape[0] != dim:
            raise ValueError("Все дескрипторы должны иметь одинаковую размерность")

    stack = np.stack(vecs, axis=0)  # (K, D)
    result = stack.max(axis=0) if mode == "max" else stack.min(axis=0)
    return l2_normalize(result) if normalize else result


# ─── aggregate ────────────────────────────────────────────────────────────────

def aggregate(
    descriptors: Dict[str, np.ndarray],
    cfg: Optional[AggregatorConfig] = None,
) -> np.ndarray:
    """Агрегировать дескрипторы согласно конфигурации.

    Аргументы:
        descriptors: Словарь {источник: вектор (1-D)}.
        cfg:         Параметры агрегации (None → AggregatorConfig()).

    Возвращает:
        Агрегированный вектор float32.
    """
    if cfg is None:
        cfg = AggregatorConfig()

    if cfg.mode == "concat":
        return concatenate_descriptors(descriptors, normalize=cfg.normalize)
    if cfg.mode == "weighted_avg":
        return weighted_average_descriptors(
            descriptors, weights=cfg.weights or None, normalize=cfg.normalize
        )
    if cfg.mode in ("max", "min"):
        return elementwise_aggregate(descriptors, mode=cfg.mode,
                                     normalize=cfg.normalize)
    # pca — requires batch context; fallback to concat
    return concatenate_descriptors(descriptors, normalize=cfg.normalize)


# ─── distance_matrix ──────────────────────────────────────────────────────────

def distance_matrix(
    vectors: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """Вычислить матрицу расстояний между векторами.

    Аргументы:
        vectors: Матрица (N × D).
        metric:  'cosine' | 'euclidean' | 'l1'.

    Возвращает:
        Матрица расстояний (N × N, float32), диагональ = 0.

    Исключения:
        ValueError: Если metric неизвестна или vectors не 2-D.
    """
    _metrics = {"cosine", "euclidean", "l1"}
    if metric not in _metrics:
        raise ValueError(
            f"metric должна быть одной из {_metrics}, получено '{metric}'"
        )
    V = np.asarray(vectors, dtype=np.float32)
    if V.ndim != 2:
        raise ValueError(f"vectors должна быть 2-D, получено ndim={V.ndim}")
    N = V.shape[0]

    if metric == "cosine":
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        V_n = V / norms
        sim = V_n @ V_n.T
        D = np.clip(1.0 - sim, 0.0, 2.0)
    elif metric == "euclidean":
        sq = np.sum(V ** 2, axis=1)
        D = sq[:, None] + sq[None, :] - 2.0 * (V @ V.T)
        D = np.sqrt(np.clip(D, 0.0, None))
    else:  # l1
        D = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            D[i] = np.sum(np.abs(V - V[i]), axis=1)

    np.fill_diagonal(D, 0.0)
    return D.astype(np.float32)


# ─── batch_aggregate ──────────────────────────────────────────────────────────

def batch_aggregate(
    descriptor_groups: List[Dict[str, np.ndarray]],
    cfg: Optional[AggregatorConfig] = None,
) -> List[np.ndarray]:
    """Агрегировать дескрипторы для нескольких фрагментов.

    Аргументы:
        descriptor_groups: Список словарей {источник: вектор}.
        cfg:               Параметры агрегации.

    Возвращает:
        Список агрегированных векторов.
    """
    return [aggregate(g, cfg) for g in descriptor_groups]
