"""Утилиты нормализации для дескрипторов и оценочных векторов.

Модуль предоставляет функции нормализации одномерных массивов (L1, L2,
мин-макс, z-нормализация) и матриц оценок (симметризация, диагональная
обнуляция). Используется при подготовке дескрипторов для сопоставления.

Функции:
    l1_normalize        — нормализация L1 (сумма = 1)
    l2_normalize        — нормализация L2 (норма = 1)
    minmax_normalize    — нормализация в [0, 1]
    zscore_normalize    — z-нормализация (μ=0, σ=1)
    softmax             — преобразование в вероятностное распределение
    clamp               — ограничение значений в [lo, hi]
    symmetrize_matrix   — симметризация матрицы ((A + A^T) / 2)
    zero_diagonal       — обнуление диагонали матрицы
    normalize_rows      — нормализация каждой строки матрицы
    batch_l2_normalize  — пакетная L2-нормализация списка векторов
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# ─── l1_normalize ─────────────────────────────────────────────────────────────

def l1_normalize(arr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Нормализовать вектор по L1-норме (сумма абсолютных значений).

    Аргументы:
        arr: 1-D массив float.
        eps: Малое число для защиты от деления на ноль.

    Возвращает:
        Нормализованный вектор float64.
        Если L1-норма < eps, возвращает нулевой вектор.

    Исключения:
        ValueError: Если arr не 1-D.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(
            f"arr должен быть 1-D, получено ndim={arr.ndim}"
        )
    norm = np.abs(arr).sum()
    if norm < eps:
        return np.zeros_like(arr)
    return arr / norm


# ─── l2_normalize ─────────────────────────────────────────────────────────────

def l2_normalize(arr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Нормализовать вектор по L2-норме (евклидова длина = 1).

    Аргументы:
        arr: 1-D массив float.
        eps: Малое число для защиты от деления на ноль.

    Возвращает:
        Нормализованный вектор float64.
        Если L2-норма < eps, возвращает нулевой вектор.

    Исключения:
        ValueError: Если arr не 1-D.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(
            f"arr должен быть 1-D, получено ndim={arr.ndim}"
        )
    norm = np.linalg.norm(arr)
    if norm < eps:
        return np.zeros_like(arr)
    return arr / norm


# ─── minmax_normalize ─────────────────────────────────────────────────────────

def minmax_normalize(arr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Нормализовать вектор в диапазон [0, 1].

    Аргументы:
        arr: 1-D массив float.
        eps: Порог различия max-min.

    Возвращает:
        Нормализованный вектор float64 ∈ [0, 1].
        Если max-min < eps, возвращает нулевой вектор.

    Исключения:
        ValueError: Если arr не 1-D.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(
            f"arr должен быть 1-D, получено ndim={arr.ndim}"
        )
    a_min, a_max = arr.min(), arr.max()
    rng = a_max - a_min
    if rng < eps:
        return np.zeros_like(arr)
    return (arr - a_min) / rng


# ─── zscore_normalize ────────────────────────────────────────────────────────

def zscore_normalize(arr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Нормализовать вектор: (x - μ) / σ.

    Аргументы:
        arr: 1-D массив float.
        eps: Порог стандартного отклонения.

    Возвращает:
        z-нормализованный вектор float64.
        Если σ < eps, возвращает нулевой вектор.

    Исключения:
        ValueError: Если arr не 1-D.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(
            f"arr должен быть 1-D, получено ndim={arr.ndim}"
        )
    mu = arr.mean()
    sigma = arr.std()
    if sigma < eps:
        return np.zeros_like(arr)
    return (arr - mu) / sigma


# ─── softmax ──────────────────────────────────────────────────────────────────

def softmax(arr: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Применить softmax-преобразование к вектору.

    Аргументы:
        arr:         1-D массив float.
        temperature: Температура (> 0); меньше → более острое распределение.

    Возвращает:
        Вектор float64 ∈ [0, 1] с суммой = 1.

    Исключения:
        ValueError: Если arr не 1-D или temperature <= 0.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(
            f"arr должен быть 1-D, получено ndim={arr.ndim}"
        )
    if temperature <= 0.0:
        raise ValueError(
            f"temperature должна быть > 0, получено {temperature}"
        )
    scaled = arr / temperature
    # Численная стабильность: вычитаем максимум
    scaled -= scaled.max()
    exp_arr = np.exp(scaled)
    return exp_arr / exp_arr.sum()


# ─── clamp ───────────────────────────────────────────────────────────────────

def clamp(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Ограничить значения массива диапазоном [lo, hi].

    Аргументы:
        arr: Массив любой формы.
        lo:  Нижняя граница.
        hi:  Верхняя граница.

    Возвращает:
        Массив того же типа с ограниченными значениями.

    Исключения:
        ValueError: Если lo > hi.
    """
    if lo > hi:
        raise ValueError(
            f"lo должен быть <= hi, получено lo={lo}, hi={hi}"
        )
    return np.clip(arr, lo, hi)


# ─── symmetrize_matrix ───────────────────────────────────────────────────────

def symmetrize_matrix(mat: np.ndarray) -> np.ndarray:
    """Симметризовать квадратную матрицу: (A + A^T) / 2.

    Аргументы:
        mat: 2-D квадратная матрица float.

    Возвращает:
        Симметричная матрица float64.

    Исключения:
        ValueError: Если mat не квадратная или не 2-D.
    """
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(
            f"mat должна быть 2-D, получено ndim={mat.ndim}"
        )
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(
            f"mat должна быть квадратной, получено форма {mat.shape}"
        )
    return (mat + mat.T) / 2.0


# ─── zero_diagonal ───────────────────────────────────────────────────────────

def zero_diagonal(mat: np.ndarray) -> np.ndarray:
    """Обнулить диагональ матрицы.

    Аргументы:
        mat: 2-D матрица.

    Возвращает:
        Матрица с нулевой диагональю (копия).

    Исключения:
        ValueError: Если mat не 2-D.
    """
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(
            f"mat должна быть 2-D, получено ndim={mat.ndim}"
        )
    result = mat.copy()
    np.fill_diagonal(result, 0.0)
    return result


# ─── normalize_rows ──────────────────────────────────────────────────────────

def normalize_rows(
    mat:    np.ndarray,
    method: str = "l2",
    eps:    float = 1e-9,
) -> np.ndarray:
    """Нормализовать каждую строку матрицы.

    Аргументы:
        mat:    2-D матрица float.
        method: 'l1' | 'l2' | 'minmax'.
        eps:    Порог для нулевой нормы.

    Возвращает:
        Нормализованная матрица float64.

    Исключения:
        ValueError: Если mat не 2-D или method неизвестный.
    """
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(
            f"mat должна быть 2-D, получено ndim={mat.ndim}"
        )
    if method not in ("l1", "l2", "minmax"):
        raise ValueError(
            f"method должен быть 'l1', 'l2' или 'minmax', получено {method!r}"
        )

    _fn = {"l1": l1_normalize, "l2": l2_normalize, "minmax": minmax_normalize}[method]
    result = np.zeros_like(mat)
    for i, row in enumerate(mat):
        result[i] = _fn(row, eps=eps)
    return result


# ─── batch_l2_normalize ──────────────────────────────────────────────────────

def batch_l2_normalize(
    vectors: List[np.ndarray],
    eps:     float = 1e-9,
) -> List[np.ndarray]:
    """Пакетная L2-нормализация списка векторов.

    Аргументы:
        vectors: Список 1-D массивов.
        eps:     Порог нулевой нормы.

    Возвращает:
        Список L2-нормализованных векторов float64.

    Исключения:
        ValueError: Если любой элемент не 1-D.
    """
    return [l2_normalize(v, eps=eps) for v in vectors]
