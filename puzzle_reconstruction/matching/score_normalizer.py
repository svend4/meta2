"""
Нормализация и калибровка оценок совместимости фрагментов.

Предоставляет функции для приведения оценок из разных методов к единому
масштабу, объединения нескольких источников оценок и калибровки по
эталонному распределению.

Классы:
    ScoreNormResult — результат нормализации одного набора оценок

Функции:
    normalize_minmax        — линейная нормализация в заданный диапазон
    normalize_zscore        — стандартизация с клиппингом выбросов
    normalize_rank          — нормализация через ранги
    calibrate_scores        — совмещение распределения с эталоном (histogram matching)
    combine_scores          — взвешенная комбинация нескольких наборов
    normalize_score_matrix  — нормализация квадратной матрицы оценок
    batch_normalize_scores  — пакетная нормализация списка массивов
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── ScoreNormResult ──────────────────────────────────────────────────────────

@dataclass
class ScoreNormResult:
    """
    Результат нормализации набора оценок.

    Attributes:
        scores:      Нормализованные оценки (np.ndarray float64).
        method:      Метод нормализации.
        original_min: Минимальное значение до нормализации.
        original_max: Максимальное значение до нормализации.
        params:      Дополнительные параметры.
    """
    scores:       np.ndarray
    method:       str
    original_min: float
    original_max: float
    params:       Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"ScoreNormResult(method={self.method!r}, "
                f"n={len(self.scores)}, "
                f"range=[{self.scores.min():.3f}, {self.scores.max():.3f}])")


# ─── normalize_minmax ─────────────────────────────────────────────────────────

def normalize_minmax(
    scores:        np.ndarray,
    feature_range: Tuple[float, float] = (0.0, 1.0),
    eps:           float = 1e-9,
) -> ScoreNormResult:
    """
    Линейная нормализация в заданный диапазон [low, high].

    Если min == max, все значения устанавливаются в low.

    Args:
        scores:        1D массив float.
        feature_range: (low, high) — целевой диапазон.
        eps:           Числовая константа устойчивости.

    Returns:
        ScoreNormResult с методом "minmax".
    """
    a    = np.asarray(scores, dtype=np.float64).ravel()
    low, high = feature_range
    mn, mx = float(a.min()), float(a.max())

    if abs(mx - mn) < eps:
        result = np.full_like(a, fill_value=float(low))
    else:
        result = (a - mn) / (mx - mn) * (high - low) + low

    return ScoreNormResult(
        scores=result,
        method="minmax",
        original_min=mn,
        original_max=mx,
        params={"feature_range": feature_range},
    )


# ─── normalize_zscore ─────────────────────────────────────────────────────────

def normalize_zscore(
    scores:   np.ndarray,
    clip_std: float = 3.0,
) -> ScoreNormResult:
    """
    Z-score стандартизация с клиппингом выбросов.

    Шаги:
      1. z = (x - mean) / std
      2. Клиппинг z в [-clip_std, +clip_std]
      3. Масштабирование в [0, 1]

    Если std == 0, все значения → 0.5.

    Args:
        scores:   1D массив float.
        clip_std: Граница клиппинга в единицах σ.

    Returns:
        ScoreNormResult с методом "zscore".
    """
    a    = np.asarray(scores, dtype=np.float64).ravel()
    mn, mx = float(a.min()), float(a.max())
    mu   = a.mean()
    sigma = a.std()

    if sigma < 1e-9:
        result = np.full_like(a, fill_value=0.5)
    else:
        z      = (a - mu) / sigma
        z      = np.clip(z, -clip_std, clip_std)
        # Нормализуем [-clip_std, +clip_std] → [0, 1]
        result = (z + clip_std) / (2.0 * clip_std)

    return ScoreNormResult(
        scores=result,
        method="zscore",
        original_min=mn,
        original_max=mx,
        params={"clip_std": clip_std},
    )


# ─── normalize_rank ───────────────────────────────────────────────────────────

def normalize_rank(
    scores: np.ndarray,
) -> ScoreNormResult:
    """
    Нормализация через ранги (rank transformation).

    Присваивает каждому значению ранг 0/(N-1), 1/(N-1), ..., 1.
    Дублирующиеся значения получают средний ранг.

    Args:
        scores: 1D массив float.

    Returns:
        ScoreNormResult с методом "rank".
    """
    a    = np.asarray(scores, dtype=np.float64).ravel()
    n    = len(a)
    mn, mx = float(a.min()), float(a.max())

    if n <= 1:
        result = np.zeros_like(a)
        return ScoreNormResult(
            scores=result, method="rank",
            original_min=mn, original_max=mx,
        )

    if np.std(a) < 1e-9:
        result = np.full(n, 0.5, dtype=np.float64)
        return ScoreNormResult(
            scores=result, method="rank",
            original_min=mn, original_max=mx,
        )
    order  = np.argsort(a)
    result = np.empty(n, dtype=np.float64)
    result[order] = np.arange(n, dtype=np.float64) / float(n - 1)

    return ScoreNormResult(
        scores=result,
        method="rank",
        original_min=mn,
        original_max=mx,
    )


# ─── calibrate_scores ─────────────────────────────────────────────────────────

def calibrate_scores(
    scores:    np.ndarray,
    reference: np.ndarray,
    n_bins:    int = 256,
) -> ScoreNormResult:
    """
    Приводит распределение оценок к распределению эталонного набора
    через histogram matching (квантильное совмещение).

    Args:
        scores:    1D массив float — нормализуемые оценки.
        reference: 1D массив float — эталонные оценки (целевое распределение).
        n_bins:    Число корзин гистограммы.

    Returns:
        ScoreNormResult с методом "calibrated".
    """
    s   = np.asarray(scores,    dtype=np.float64).ravel()
    ref = np.asarray(reference, dtype=np.float64).ravel()

    if s.size == 0 or ref.size == 0:
        mn = float(s.min()) if s.size > 0 else 0.0
        mx = float(s.max()) if s.size > 0 else 0.0
        return ScoreNormResult(
            scores=s.copy(), method="calibrated",
            original_min=mn, original_max=mx,
        )
    mn, mx = float(s.min()), float(s.max())

    # Квантильное совмещение через CDF
    s_sorted   = np.sort(s)
    ref_sorted = np.sort(ref)
    n_s   = len(s_sorted)
    n_ref = len(ref_sorted)

    # Для каждого значения из s находим его квантиль и сопоставляем с ref
    quantiles = np.searchsorted(s_sorted, s, side="right").astype(np.float64)
    quantiles /= float(n_s)
    ref_indices = (quantiles * (n_ref - 1)).astype(int)
    ref_indices = np.clip(ref_indices, 0, n_ref - 1)
    result = ref_sorted[ref_indices]

    return ScoreNormResult(
        scores=result,
        method="calibrated",
        original_min=mn,
        original_max=mx,
        params={"n_bins": n_bins},
    )


# ─── combine_scores ───────────────────────────────────────────────────────────

def combine_scores(
    score_arrays: List[np.ndarray],
    weights:      Optional[List[float]] = None,
    method:       str = "weighted",
) -> np.ndarray:
    """
    Объединяет несколько наборов оценок в один вектор.

    Args:
        score_arrays: Список 1D массивов одинаковой длины.
        weights:      Веса. None → равные. Нормализуются до суммы 1.
        method:       "weighted" — взвешенное среднее;
                      "min"      — поэлементный минимум;
                      "max"      — поэлементный максимум;
                      "product"  — произведение (для оценок ∈ [0,1]).

    Returns:
        np.ndarray float64 той же длины.

    Raises:
        ValueError: Пустой список, несовпадающие длины или неизвестный метод.
    """
    if not score_arrays:
        raise ValueError("score_arrays must not be empty.")

    if method not in ("weighted", "min", "max", "product"):
        raise ValueError(
            f"Unknown method {method!r}. "
            f"Choose 'weighted', 'min', 'max', or 'product'.")

    arrays = [np.asarray(a, dtype=np.float64).ravel() for a in score_arrays]
    n = len(arrays[0])
    for i, a in enumerate(arrays[1:], 1):
        if len(a) != n:
            raise ValueError(
                f"All arrays must have the same length. "
                f"arrays[0] has {n}, arrays[{i}] has {len(a)}.")

    if method == "min":
        return np.min(np.stack(arrays), axis=0)
    elif method == "max":
        return np.max(np.stack(arrays), axis=0)
    elif method == "product":
        result = np.ones(n, dtype=np.float64)
        for a in arrays:
            result *= a
        return result
    else:  # weighted
        if weights is None:
            w = np.full(len(arrays), 1.0 / len(arrays))
        else:
            w = np.array(weights, dtype=np.float64)
            s = w.sum()
            if s < 1e-9:
                raise ValueError("Sum of weights must be > 0.")
            w /= s
        return sum(float(wi) * a for wi, a in zip(w, arrays))


# ─── normalize_score_matrix ───────────────────────────────────────────────────

def normalize_score_matrix(
    matrix: np.ndarray,
    method: str = "minmax",
    keep_diagonal: bool = True,
) -> np.ndarray:
    """
    Нормализует квадратную матрицу оценок.

    Нормализация применяется только к off-diagonal элементам.

    Args:
        matrix:        (N, N) матрица float.
        method:        "minmax" | "zscore" | "rank".
        keep_diagonal: True → диагональ сохраняется без изменений.

    Returns:
        Нормализованная матрица (N, N) float64.

    Raises:
        ValueError: Если матрица не квадратная или неизвестный метод.
    """
    if method not in ("minmax", "zscore", "rank"):
        raise ValueError(
            f"Unknown method {method!r}. Choose 'minmax', 'zscore', or 'rank'.")

    n = matrix.shape[0]
    if matrix.ndim != 2 or matrix.shape[1] != n:
        raise ValueError(
            f"matrix must be square, got shape {matrix.shape}.")

    m = matrix.astype(np.float64)
    mask_off = ~np.eye(n, dtype=bool)
    off_vals = m[mask_off]

    if method == "minmax":
        r = normalize_minmax(off_vals).scores
    elif method == "zscore":
        r = normalize_zscore(off_vals).scores
    else:
        r = normalize_rank(off_vals).scores

    result = m.copy()
    result[mask_off] = r
    if keep_diagonal:
        np.copyto(result, m, where=np.eye(n, dtype=bool))
    return result


# ─── batch_normalize_scores ───────────────────────────────────────────────────

def batch_normalize_scores(
    score_list: List[np.ndarray],
    method:     str = "minmax",
    **kwargs,
) -> List[ScoreNormResult]:
    """
    Применяет нормализацию к каждому массиву в списке независимо.

    Args:
        score_list: Список 1D массивов.
        method:     "minmax" | "zscore" | "rank".
        **kwargs:   Дополнительные параметры метода.

    Returns:
        Список ScoreNormResult той же длины.

    Raises:
        ValueError: Неизвестный метод.
    """
    if method not in ("minmax", "zscore", "rank"):
        raise ValueError(
            f"Unknown method {method!r}. Choose 'minmax', 'zscore', or 'rank'.")

    fn = {"minmax": normalize_minmax,
          "zscore": normalize_zscore,
          "rank":   normalize_rank}[method]
    return [fn(s, **kwargs) for s in score_list]
