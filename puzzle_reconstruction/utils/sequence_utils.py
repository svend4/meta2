"""
Утилиты для работы с последовательностями и упорядочиванием элементов.

Предоставляет функции для анализа и преобразования последовательностей:
нормализации, ранжирования, выравнивания и метрик сходства, используемых
при упорядочивании фрагментов в пайплайне реконструкции.

Экспортирует:
    SequenceConfig        — параметры операций над последовательностями
    rank_sequence         — ранги элементов последовательности (1-based)
    normalize_sequence    — нормализация в [0, 1]
    invert_sequence       — инвертирование (max - x + min)
    sliding_scores        — скользящие агрегации по окну
    align_sequences       — выравнивание двух последовательностей к одной длине
    kendall_tau_distance  — расстояние Кендалла между двумя ранжированиями
    longest_increasing    — длина наибольшей возрастающей подпоследовательности
    segment_by_threshold  — разбивка последовательности на сегменты по порогу
    batch_rank            — ранжирование для пакета последовательностей
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np


# ─── SequenceConfig ───────────────────────────────────────────────────────────

@dataclass
class SequenceConfig:
    """Параметры операций над последовательностями.

    Attributes:
        window:     Размер окна для скользящих агрегаций (>= 1).
        agg:        Агрегирующая функция окна: "mean", "max", "min", "sum".
        threshold:  Порог для бинарного разбиения последовательности [0, 1].
    """
    window:    int   = 3
    agg:       Literal["mean", "max", "min", "sum"] = "mean"
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError(
                f"window must be >= 1, got {self.window}"
            )
        if self.agg not in ("mean", "max", "min", "sum"):
            raise ValueError(
                f"agg must be one of 'mean','max','min','sum', got '{self.agg}'"
            )
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(
                f"threshold must be in [0, 1], got {self.threshold}"
            )


# ─── rank_sequence ────────────────────────────────────────────────────────────

def rank_sequence(seq: np.ndarray) -> np.ndarray:
    """Вернуть ранги элементов последовательности (1-based, ascending).

    Наименьший элемент получает ранг 1. При совпадении значений используется
    среднеарифметическое ранков (tied ranks).

    Args:
        seq: 1-D массив числовых значений.

    Returns:
        Массив float64 рангов той же формы, что seq.

    Raises:
        ValueError: Если seq не 1-D или пуст.
    """
    seq = np.asarray(seq, dtype=float)
    if seq.ndim != 1:
        raise ValueError(f"seq must be 1-D, got ndim={seq.ndim}")
    if seq.size == 0:
        raise ValueError("seq must not be empty")

    n = len(seq)
    order = np.argsort(seq)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)

    # Tied ranks: среднее среди одинаковых значений
    unique_vals = np.unique(seq)
    for val in unique_vals:
        mask = seq == val
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()

    return ranks


# ─── normalize_sequence ───────────────────────────────────────────────────────

def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Нормализовать последовательность в [0, 1] по min-max.

    Если все значения одинаковы — возвращает массив из нулей.

    Args:
        seq: 1-D числовой массив.

    Returns:
        Массив float64 в [0, 1].

    Raises:
        ValueError: Если seq не 1-D или пуст.
    """
    seq = np.asarray(seq, dtype=float)
    if seq.ndim != 1:
        raise ValueError(f"seq must be 1-D, got ndim={seq.ndim}")
    if seq.size == 0:
        raise ValueError("seq must not be empty")

    lo, hi = float(seq.min()), float(seq.max())
    if hi == lo:
        return np.zeros_like(seq)
    return (seq - lo) / (hi - lo)


# ─── invert_sequence ──────────────────────────────────────────────────────────

def invert_sequence(seq: np.ndarray) -> np.ndarray:
    """Инвертировать нормализованную последовательность: x → 1 - x.

    Args:
        seq: 1-D числовой массив (ожидается ∈ [0, 1], но не обязательно).

    Returns:
        Массив float64: 1.0 - seq.

    Raises:
        ValueError: Если seq не 1-D или пуст.
    """
    seq = np.asarray(seq, dtype=float)
    if seq.ndim != 1:
        raise ValueError(f"seq must be 1-D, got ndim={seq.ndim}")
    if seq.size == 0:
        raise ValueError("seq must not be empty")
    return 1.0 - seq


# ─── sliding_scores ───────────────────────────────────────────────────────────

def sliding_scores(
    seq: np.ndarray,
    cfg: SequenceConfig | None = None,
) -> np.ndarray:
    """Вычислить скользящую агрегацию по окну.

    Краевые элементы (где окно не умещается полностью) вычисляются
    по доступным соседям (усечённое окно).

    Args:
        seq: 1-D числовой массив.
        cfg: Параметры. None → SequenceConfig().

    Returns:
        Массив float64 той же длины, что seq.

    Raises:
        ValueError: Если seq не 1-D или пуст.
    """
    seq = np.asarray(seq, dtype=float)
    if seq.ndim != 1:
        raise ValueError(f"seq must be 1-D, got ndim={seq.ndim}")
    if seq.size == 0:
        raise ValueError("seq must not be empty")
    if cfg is None:
        cfg = SequenceConfig()

    n = len(seq)
    half = cfg.window // 2
    result = np.empty(n, dtype=float)
    agg_fn = {"mean": np.mean, "max": np.max, "min": np.min, "sum": np.sum}[cfg.agg]

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result[i] = agg_fn(seq[lo:hi])

    return result


# ─── align_sequences ──────────────────────────────────────────────────────────

def align_sequences(
    a: np.ndarray,
    b: np.ndarray,
    target_len: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Выровнять две последовательности к одной длине методом линейной интерполяции.

    Args:
        a:          1-D массив.
        b:          1-D массив.
        target_len: Целевая длина. None → max(len(a), len(b)).

    Returns:
        Пара (a_aligned, b_aligned) — оба длины target_len, dtype float64.

    Raises:
        ValueError: Если a или b не 1-D, или target_len < 1.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"a must be 1-D, got ndim={a.ndim}")
    if b.ndim != 1:
        raise ValueError(f"b must be 1-D, got ndim={b.ndim}")

    if target_len is None:
        target_len = max(len(a), len(b))
    if target_len < 1:
        raise ValueError(f"target_len must be >= 1, got {target_len}")

    def _resample(arr: np.ndarray, n: int) -> np.ndarray:
        if len(arr) == n:
            return arr.copy()
        old_x = np.linspace(0.0, 1.0, len(arr))
        new_x = np.linspace(0.0, 1.0, n)
        return np.interp(new_x, old_x, arr)

    return _resample(a, target_len), _resample(b, target_len)


# ─── kendall_tau_distance ─────────────────────────────────────────────────────

def kendall_tau_distance(perm_a: np.ndarray, perm_b: np.ndarray) -> int:
    """Расстояние Кендалла — число инверсий между двумя перестановками.

    Считает количество пар (i, j) таких, что относительный порядок
    элементов в perm_a и perm_b различается.

    Args:
        perm_a: 1-D массив — первая перестановка индексов [0..n-1].
        perm_b: 1-D массив — вторая перестановка тех же индексов.

    Returns:
        Целое число инверсий ∈ [0, n*(n-1)/2].

    Raises:
        ValueError: Если perm_a и perm_b разной длины или не 1-D.
    """
    perm_a = np.asarray(perm_a, dtype=int)
    perm_b = np.asarray(perm_b, dtype=int)
    if perm_a.ndim != 1 or perm_b.ndim != 1:
        raise ValueError("perm_a and perm_b must be 1-D")
    if len(perm_a) != len(perm_b):
        raise ValueError(
            f"perm_a and perm_b must have same length: "
            f"{len(perm_a)} vs {len(perm_b)}"
        )

    n = len(perm_a)
    # Map perm_b to index: where does each element of perm_a appear in perm_b?
    pos_in_b = np.empty(n, dtype=int)
    for rank, elem in enumerate(perm_b):
        pos_in_b[elem] = rank

    # Count inversions using naïve O(n²) algorithm (suitable for small n)
    mapped = np.array([pos_in_b[x] for x in perm_a], dtype=int)
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if mapped[i] > mapped[j]:
                inversions += 1
    return inversions


# ─── longest_increasing ───────────────────────────────────────────────────────

def longest_increasing(seq: np.ndarray) -> int:
    """Длина наибольшей строго возрастающей подпоследовательности (LIS).

    Args:
        seq: 1-D числовой массив.

    Returns:
        Длина LIS ∈ [0, len(seq)].

    Raises:
        ValueError: Если seq не 1-D.
    """
    seq = np.asarray(seq, dtype=float)
    if seq.ndim != 1:
        raise ValueError(f"seq must be 1-D, got ndim={seq.ndim}")
    if seq.size == 0:
        return 0

    # O(n log n) patience sorting
    tails: list[float] = []
    for x in seq:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(float(x))
        else:
            tails[lo] = float(x)
    return len(tails)


# ─── segment_by_threshold ─────────────────────────────────────────────────────

def segment_by_threshold(
    seq: np.ndarray,
    cfg: SequenceConfig | None = None,
) -> List[tuple[int, int]]:
    """Разбить последовательность на сегменты, где значения >= порога.

    Args:
        seq: 1-D числовой массив (нормализованный, ожидается ∈ [0, 1]).
        cfg: Параметры. None → SequenceConfig().

    Returns:
        Список кортежей (start, end) — включительные индексы сегментов,
        где seq[i] >= threshold. Пустой список если таких нет.

    Raises:
        ValueError: Если seq не 1-D.
    """
    seq = np.asarray(seq, dtype=float)
    if seq.ndim != 1:
        raise ValueError(f"seq must be 1-D, got ndim={seq.ndim}")
    if cfg is None:
        cfg = SequenceConfig()

    segments: List[tuple[int, int]] = []
    in_seg = False
    start = 0

    for i, val in enumerate(seq):
        if val >= cfg.threshold:
            if not in_seg:
                start = i
                in_seg = True
        else:
            if in_seg:
                segments.append((start, i - 1))
                in_seg = False

    if in_seg:
        segments.append((start, len(seq) - 1))

    return segments


# ─── batch_rank ───────────────────────────────────────────────────────────────

def batch_rank(sequences: List[np.ndarray]) -> List[np.ndarray]:
    """Вычислить ранги для пакета последовательностей.

    Args:
        sequences: Список 1-D массивов.

    Returns:
        Список массивов рангов, по одному на входную последовательность.

    Raises:
        ValueError: Если sequences пуст.
    """
    if not sequences:
        raise ValueError("sequences must not be empty")
    return [rank_sequence(np.asarray(s, dtype=float)) for s in sequences]
