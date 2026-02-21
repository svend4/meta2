"""
Утилиты для работы с разреженными матрицами оценок фрагментов.

Предоставляет инструменты фильтрации, нормализации и преобразования
матриц попарных оценок совместимости.

Экспортирует:
    SparseEntry         — запись разреженной матрицы (row, col, value)
    to_sparse_entries   — преобразовать плотную матрицу в список записей
    from_sparse_entries — собрать плотную матрицу из записей
    sparse_top_k        — топ-k записей для каждой строки
    threshold_matrix    — обнулить значения ниже порога
    symmetrize_matrix   — симметризовать матрицу (max(M, M.T))
    normalize_matrix    — построчная нормализация в [0, 1]
    diagonal_zeros      — копия матрицы с нулевой диагональю
    matrix_sparsity     — доля нулевых элементов
    top_k_per_row       — матрица с сохранением только топ-k на строку
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class SparseEntry:
    """Запись разреженной матрицы.

    Attributes:
        row:   Индекс строки (≥ 0).
        col:   Индекс столбца (≥ 0).
        value: Значение элемента.
    """
    row: int
    col: int
    value: float

    def __post_init__(self) -> None:
        if self.row < 0:
            raise ValueError(f"row must be >= 0, got {self.row}")
        if self.col < 0:
            raise ValueError(f"col must be >= 0, got {self.col}")

    def __repr__(self) -> str:  # pragma: no cover
        return f"SparseEntry(row={self.row}, col={self.col}, value={self.value:.4f})"


# ─── Публичные функции ────────────────────────────────────────────────────────

def to_sparse_entries(
    matrix: np.ndarray,
    threshold: float = 0.0,
) -> List[SparseEntry]:
    """Преобразовать плотную матрицу в список ненулевых (выше порога) записей.

    Args:
        matrix:    Двумерный массив float.
        threshold: Минимальное абсолютное значение для включения.

    Returns:
        Список :class:`SparseEntry` в порядке строка→столбец.

    Raises:
        ValueError: Если ``matrix`` не двумерный.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got ndim={mat.ndim}")
    rows, cols = np.where(np.abs(mat) > threshold)
    entries: List[SparseEntry] = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        entries.append(SparseEntry(row=int(r), col=int(c), value=float(mat[r, c])))
    return entries


def from_sparse_entries(
    entries: List[SparseEntry],
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    """Собрать плотную матрицу float64 из разреженных записей.

    Args:
        entries: Список записей.
        n_rows:  Количество строк (> 0).
        n_cols:  Количество столбцов (> 0).

    Returns:
        Плотная матрица float64 (n_rows, n_cols).

    Raises:
        ValueError: Если размеры ≤ 0 или индекс записи выходит за границы.
    """
    if n_rows <= 0:
        raise ValueError(f"n_rows must be > 0, got {n_rows}")
    if n_cols <= 0:
        raise ValueError(f"n_cols must be > 0, got {n_cols}")
    mat = np.zeros((n_rows, n_cols), dtype=np.float64)
    for e in entries:
        if e.row >= n_rows or e.col >= n_cols:
            raise ValueError(
                f"Entry ({e.row}, {e.col}) is out of bounds "
                f"for shape ({n_rows}, {n_cols})"
            )
        mat[e.row, e.col] = e.value
    return mat


def sparse_top_k(
    matrix: np.ndarray,
    k: int,
) -> List[SparseEntry]:
    """Вернуть топ-k записей для каждой строки матрицы.

    Args:
        matrix: Двумерный массив float (N, M).
        k:      Количество записей на строку (≥ 1).

    Returns:
        Список :class:`SparseEntry` — не более ``k`` на строку,
        отсортированных по убыванию значения внутри строки.

    Raises:
        ValueError: Если ``k`` < 1 или матрица не 2-D.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got ndim={mat.ndim}")
    entries: List[SparseEntry] = []
    n_rows, n_cols = mat.shape
    for r in range(n_rows):
        row = mat[r]
        effective_k = min(k, n_cols)
        top_cols = np.argsort(-row)[:effective_k]
        for c in top_cols:
            entries.append(SparseEntry(row=r, col=int(c), value=float(row[c])))
    return entries


def threshold_matrix(
    matrix: np.ndarray,
    threshold: float,
    fill: float = 0.0,
) -> np.ndarray:
    """Обнулить (или заменить) значения ниже порога.

    Args:
        matrix:    Двумерный массив float.
        threshold: Порог (строго: значения < threshold заменяются).
        fill:      Значение замены (по умолчанию 0).

    Returns:
        Новый массив float64 того же размера.

    Raises:
        ValueError: Если матрица не 2-D.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got ndim={mat.ndim}")
    result = mat.copy()
    result[result < threshold] = fill
    return result


def symmetrize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Симметризовать матрицу по максимуму: result[i,j] = max(M[i,j], M[j,i]).

    Args:
        matrix: Квадратная матрица float (N, N).

    Returns:
        Симметричная матрица float64 (N, N).

    Raises:
        ValueError: Если матрица не квадратная или не 2-D.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(
            f"matrix must be a square 2-D array, got shape {mat.shape}"
        )
    return np.maximum(mat, mat.T)


def normalize_matrix(
    matrix: np.ndarray,
    axis: int = 1,
) -> np.ndarray:
    """Построчная (или постолбцовая) нормализация матрицы в [0, 1].

    Если строка (столбец) содержит только нули, остаётся нулевой.

    Args:
        matrix: Двумерный массив float.
        axis:   0 — нормализовать по столбцам, 1 — по строкам (по умолчанию).

    Returns:
        Нормализованная матрица float64.

    Raises:
        ValueError: Если матрица не 2-D или ``axis`` не 0 или 1.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got ndim={mat.ndim}")
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    result = mat.copy()
    if axis == 1:
        for i in range(mat.shape[0]):
            row_max = mat[i].max()
            if row_max > 0:
                result[i] = mat[i] / row_max
    else:
        for j in range(mat.shape[1]):
            col_max = mat[:, j].max()
            if col_max > 0:
                result[:, j] = mat[:, j] / col_max
    return result


def diagonal_zeros(matrix: np.ndarray) -> np.ndarray:
    """Вернуть копию матрицы с нулями на диагонали.

    Args:
        matrix: Квадратная матрица float (N, N).

    Returns:
        Копия float64 с диагональю = 0.

    Raises:
        ValueError: Если матрица не квадратная или не 2-D.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(
            f"matrix must be a square 2-D array, got shape {mat.shape}"
        )
    result = mat.copy()
    np.fill_diagonal(result, 0.0)
    return result


def matrix_sparsity(matrix: np.ndarray) -> float:
    """Вычислить долю нулевых элементов матрицы.

    Args:
        matrix: Двумерный массив float.

    Returns:
        Доля нулей ∈ [0, 1]; для пустой матрицы → 1.0.

    Raises:
        ValueError: Если матрица не 2-D.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got ndim={mat.ndim}")
    if mat.size == 0:
        return 1.0
    return float(np.sum(mat == 0)) / float(mat.size)


def top_k_per_row(
    matrix: np.ndarray,
    k: int,
) -> np.ndarray:
    """Обнулить все элементы строки, кроме топ-k наибольших.

    Args:
        matrix: Двумерный массив float (N, M).
        k:      Количество сохраняемых значений на строку (≥ 1).

    Returns:
        Новая матрица float64 с нулями вне топ-k.

    Raises:
        ValueError: Если ``k`` < 1 или матрица не 2-D.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got ndim={mat.ndim}")
    result = np.zeros_like(mat)
    n_rows, n_cols = mat.shape
    effective_k = min(k, n_cols)
    for i in range(n_rows):
        top_idx = np.argsort(-mat[i])[:effective_k]
        result[i, top_idx] = mat[i, top_idx]
    return result
