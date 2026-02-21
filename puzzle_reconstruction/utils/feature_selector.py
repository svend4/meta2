"""Выбор и фильтрация признаков для описания фрагментов.

Модуль предоставляет функции для отбора наиболее информативных признаков:
отбор по дисперсии, корреляционный отбор, ранжирование по важности,
PCA-снижение размерности, нормализация признаков и пакетная обработка.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ─── FeatureSet ───────────────────────────────────────────────────────────────

@dataclass
class FeatureSet:
    """Набор признаков одного фрагмента.

    Атрибуты:
        features:  Вектор признаков (float32, 1-D).
        labels:    Имена признаков (опционально).
        fragment_id: Идентификатор фрагмента (>= 0).
        params:    Дополнительные параметры.
    """

    features: np.ndarray
    labels: Optional[List[str]] = None
    fragment_id: int = 0
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, dtype=np.float32)
        if self.features.ndim != 1:
            raise ValueError(
                f"features должен быть 1-D, получено ndim={self.features.ndim}"
            )
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        if self.labels is not None and len(self.labels) != len(self.features):
            raise ValueError(
                f"Длина labels ({len(self.labels)}) не совпадает "
                f"с длиной features ({len(self.features)})"
            )

    def __len__(self) -> int:
        return len(self.features)


# ─── SelectionResult ──────────────────────────────────────────────────────────

@dataclass
class SelectionResult:
    """Результат отбора признаков.

    Атрибуты:
        selected_indices: Индексы отобранных признаков (int64).
        n_selected:       Количество отобранных признаков (>= 0).
        scores:           Оценки значимости для каждого признака.
        params:           Дополнительные параметры.
    """

    selected_indices: np.ndarray
    n_selected: int
    scores: np.ndarray
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.selected_indices = np.asarray(self.selected_indices, dtype=np.int64)
        self.scores = np.asarray(self.scores, dtype=np.float64)
        if self.n_selected < 0:
            raise ValueError(
                f"n_selected должен быть >= 0, получено {self.n_selected}"
            )

    def __len__(self) -> int:
        return self.n_selected


# ─── variance_selection ───────────────────────────────────────────────────────

def variance_selection(
    X: np.ndarray, threshold: float = 0.0
) -> SelectionResult:
    """Отбор признаков по дисперсии: удалить признаки с малой дисперсией.

    Аргументы:
        X:         Матрица признаков формы (N, D).
        threshold: Минимальная дисперсия (>= 0).

    Возвращает:
        SelectionResult с индексами признаков, дисперсия которых >= threshold.

    Исключения:
        ValueError: Если X не 2-D или threshold < 0.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X должен быть 2-D, получено ndim={X.ndim}")
    if threshold < 0.0:
        raise ValueError(f"threshold должен быть >= 0, получено {threshold}")

    variances = X.var(axis=0)
    indices = np.where(variances >= threshold)[0].astype(np.int64)
    return SelectionResult(
        selected_indices=indices,
        n_selected=len(indices),
        scores=variances,
    )


# ─── correlation_selection ────────────────────────────────────────────────────

def correlation_selection(
    X: np.ndarray, max_corr: float = 0.95
) -> SelectionResult:
    """Удаление сильно коррелированных признаков (жадный алгоритм).

    Для каждой пары признаков с |r| > max_corr удаляет второй.

    Аргументы:
        X:        Матрица признаков формы (N, D).
        max_corr: Порог корреляции (0 < max_corr <= 1).

    Возвращает:
        SelectionResult с индексами некоррелированных признаков.

    Исключения:
        ValueError: Если X не 2-D или max_corr вне (0, 1].
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X должен быть 2-D, получено ndim={X.ndim}")
    if not (0.0 < max_corr <= 1.0):
        raise ValueError(
            f"max_corr должен быть в (0, 1], получено {max_corr}"
        )

    D = X.shape[1]
    if D == 0:
        return SelectionResult(
            selected_indices=np.array([], dtype=np.int64),
            n_selected=0,
            scores=np.array([], dtype=np.float64),
        )

    # Вычисляем матрицу корреляций
    std = X.std(axis=0)
    std[std < 1e-12] = 1.0  # избегаем деления на ноль
    Xn = (X - X.mean(axis=0)) / std
    corr_matrix = (Xn.T @ Xn) / max(X.shape[0] - 1, 1)

    keep = np.ones(D, dtype=bool)
    for i in range(D):
        if not keep[i]:
            continue
        for j in range(i + 1, D):
            if not keep[j]:
                continue
            if abs(corr_matrix[i, j]) > max_corr:
                keep[j] = False

    indices = np.where(keep)[0].astype(np.int64)
    # Оценки: средняя |r| каждого признака со всеми остальными
    scores = np.abs(corr_matrix).mean(axis=1)
    return SelectionResult(
        selected_indices=indices,
        n_selected=len(indices),
        scores=scores,
    )


# ─── rank_features ────────────────────────────────────────────────────────────

def rank_features(
    X: np.ndarray, y: np.ndarray
) -> SelectionResult:
    """Ранжирование признаков по корреляции с целевой переменной.

    Вычисляет |r(X[:, i], y)| для каждого признака i.

    Аргументы:
        X: Матрица признаков (N, D).
        y: Целевой вектор (N,).

    Возвращает:
        SelectionResult, отсортированный по убыванию |r|.

    Исключения:
        ValueError: Если размерности не совпадают или X не 2-D.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim != 2:
        raise ValueError(f"X должен быть 2-D, получено ndim={X.ndim}")
    if X.shape[0] != len(y):
        raise ValueError(
            f"Количество строк X ({X.shape[0]}) не совпадает с len(y) ({len(y)})"
        )

    y_std = y.std()
    scores = np.zeros(X.shape[1], dtype=np.float64)
    for i in range(X.shape[1]):
        xi = X[:, i]
        xi_std = xi.std()
        if xi_std < 1e-12 or y_std < 1e-12:
            scores[i] = 0.0
        else:
            scores[i] = abs(np.corrcoef(xi, y)[0, 1])

    order = np.argsort(-scores).astype(np.int64)
    return SelectionResult(
        selected_indices=order,
        n_selected=len(order),
        scores=scores,
    )


# ─── pca_reduce ───────────────────────────────────────────────────────────────

def pca_reduce(
    X: np.ndarray, n_components: int
) -> Tuple[np.ndarray, np.ndarray]:
    """PCA-снижение размерности (SVD без sklearn).

    Аргументы:
        X:            Матрица признаков (N, D), N >= n_components.
        n_components: Количество главных компонент (>= 1, <= min(N, D)).

    Возвращает:
        Кортеж (X_reduced, explained_variance_ratio):
          - X_reduced: (N, n_components) float64
          - explained_variance_ratio: (n_components,) float64

    Исключения:
        ValueError: Если параметры некорректны.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X должен быть 2-D, получено ndim={X.ndim}")
    N, D = X.shape
    max_comp = min(N, D)
    if n_components < 1:
        raise ValueError(f"n_components должен быть >= 1, получено {n_components}")
    if n_components > max_comp:
        raise ValueError(
            f"n_components ({n_components}) > min(N, D) = {max_comp}"
        )

    Xc = X - X.mean(axis=0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    variances = (s ** 2) / max(N - 1, 1)
    total_var = variances.sum()
    if total_var < 1e-15:
        evr = np.zeros(n_components, dtype=np.float64)
    else:
        evr = variances[:n_components] / total_var

    X_reduced = (Xc @ Vt[:n_components].T).astype(np.float64)
    return X_reduced, evr


# ─── normalize_features ───────────────────────────────────────────────────────

def normalize_features(
    X: np.ndarray, method: str = "minmax"
) -> np.ndarray:
    """Нормализация матрицы признаков по столбцам.

    Аргументы:
        X:      Матрица (N, D).
        method: 'minmax' (масштаб [0,1]) или 'zscore' (µ=0, σ=1).

    Возвращает:
        Нормализованная матрица (N, D) float64.

    Исключения:
        ValueError: Если X не 2-D или method неизвестен.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X должен быть 2-D, получено ndim={X.ndim}")
    if method not in ("minmax", "zscore"):
        raise ValueError(
            f"Неизвестный метод '{method}'. Допустимые: 'minmax', 'zscore'"
        )

    Xn = X.copy()
    if method == "minmax":
        mn = X.min(axis=0)
        mx = X.max(axis=0)
        rng = mx - mn
        rng[rng < 1e-12] = 1.0
        Xn = (X - mn) / rng
    else:  # zscore
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma < 1e-12] = 1.0
        Xn = (X - mu) / sigma
    return Xn


# ─── select_top_k ─────────────────────────────────────────────────────────────

def select_top_k(
    selection: SelectionResult, k: int
) -> SelectionResult:
    """Оставить только k лучших признаков из SelectionResult.

    Аргументы:
        selection: Результат rank_features или variance_selection.
        k:         Количество признаков (>= 1).

    Возвращает:
        Новый SelectionResult с первыми k индексами.

    Исключения:
        ValueError: Если k < 1 или k > n_selected.
    """
    if k < 1:
        raise ValueError(f"k должен быть >= 1, получено {k}")
    if k > selection.n_selected:
        raise ValueError(
            f"k ({k}) > n_selected ({selection.n_selected})"
        )
    indices = selection.selected_indices[:k]
    scores = selection.scores[indices] if len(selection.scores) > 0 else selection.scores
    return SelectionResult(
        selected_indices=indices.astype(np.int64),
        n_selected=k,
        scores=scores,
    )


# ─── apply_selection ──────────────────────────────────────────────────────────

def apply_selection(
    X: np.ndarray, selection: SelectionResult
) -> np.ndarray:
    """Применить отбор признаков к матрице X.

    Аргументы:
        X:         Матрица признаков (N, D).
        selection: Результат отбора.

    Возвращает:
        Матрица (N, n_selected) float64.

    Исключения:
        ValueError: Если X не 2-D.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X должен быть 2-D, получено ndim={X.ndim}")
    return X[:, selection.selected_indices]


# ─── batch_select ─────────────────────────────────────────────────────────────

def batch_select(
    feature_sets: List[FeatureSet], selection: SelectionResult
) -> List[FeatureSet]:
    """Применить отбор признаков к списку FeatureSet.

    Аргументы:
        feature_sets: Список FeatureSet.
        selection:    Результат отбора (индексы).

    Возвращает:
        Список новых FeatureSet с отобранными признаками.
    """
    result = []
    for fs in feature_sets:
        new_feat = fs.features[selection.selected_indices]
        new_labels = (
            [fs.labels[i] for i in selection.selected_indices]
            if fs.labels is not None else None
        )
        result.append(
            FeatureSet(
                features=new_feat,
                labels=new_labels,
                fragment_id=fs.fragment_id,
                params=dict(fs.params),
            )
        )
    return result
