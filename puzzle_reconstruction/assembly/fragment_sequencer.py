"""
Определение оптимального порядка (последовательности) фрагментов.

Предоставляет алгоритмы для упорядочивания фрагментов документа
на основе матрицы попарных оценок совместимости.

Экспортирует:
    SequenceResult       — результат определения последовательности
    sequence_greedy      — жадный алгоритм последовательности
    sequence_by_score    — сортировка по индивидуальным оценкам
    compute_sequence_score — суммарная оценка последовательности
    reverse_sequence     — обращение порядка
    rotate_sequence      — циклический сдвиг последовательности
    sequence_to_pairs    — список соседних пар
    find_best_start      — оптимальный начальный элемент
    batch_sequence       — пакетная последовательность нескольких матриц
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class SequenceResult:
    """Результат определения порядка фрагментов.

    Attributes:
        order:        Список индексов фрагментов в найденном порядке.
        total_score:  Суммарная оценка последовательности.
        params:       Параметры алгоритма.
    """
    order: List[int]
    total_score: float
    params: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total_score < 0.0:
            raise ValueError(
                f"total_score must be >= 0, got {self.total_score}"
            )

    def __len__(self) -> int:
        return len(self.order)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SequenceResult(n={len(self.order)}, "
            f"total_score={self.total_score:.4f})"
        )


# ─── Публичные функции ────────────────────────────────────────────────────────

def sequence_greedy(
    score_matrix: np.ndarray,
    start: Optional[int] = None,
) -> SequenceResult:
    """Жадный алгоритм построения последовательности.

    На каждом шаге выбирает непосещённый фрагмент с наибольшей оценкой
    совместимости с текущим.

    Args:
        score_matrix: Квадратная матрица (N, N) float оценок [0, 1].
                      ``score_matrix[i, j]`` — оценка пары (i, j).
        start:        Начальный индекс (0-based). Если ``None`` —
                      выбирается строка с максимальной суммой.

    Returns:
        :class:`SequenceResult` с найденным порядком.

    Raises:
        ValueError: Если матрица не квадратная, пустая или ``start``
                    вне допустимого диапазона.
    """
    mat = np.asarray(score_matrix, dtype=np.float64)
    _validate_square(mat)
    n = mat.shape[0]
    if n == 0:
        return SequenceResult(order=[], total_score=0.0,
                              params={"algorithm": "greedy"})
    if start is None:
        start = int(np.argmax(mat.sum(axis=1)))
    if not (0 <= start < n):
        raise ValueError(f"start must be in [0, {n - 1}], got {start}")

    visited = [False] * n
    order = [start]
    visited[start] = True
    total = 0.0

    for _ in range(n - 1):
        current = order[-1]
        best_score = -1.0
        best_next = -1
        for j in range(n):
            if not visited[j] and mat[current, j] > best_score:
                best_score = mat[current, j]
                best_next = j
        if best_next == -1:
            break
        order.append(best_next)
        visited[best_next] = True
        total += best_score

    return SequenceResult(
        order=order,
        total_score=float(total),
        params={"algorithm": "greedy", "start": start},
    )


def sequence_by_score(
    scores: List[float],
    descending: bool = True,
) -> SequenceResult:
    """Упорядочить фрагменты по индивидуальным оценкам.

    Args:
        scores:     Список оценок (по одной на фрагмент).
        descending: Если ``True`` — от большего к меньшему.

    Returns:
        :class:`SequenceResult` с порядком по оценкам.

    Raises:
        ValueError: Если любая оценка отрицательна.
    """
    for i, s in enumerate(scores):
        if s < 0.0:
            raise ValueError(f"scores[{i}] must be >= 0, got {s}")
    arr = np.asarray(scores, dtype=np.float64)
    if descending:
        order = list(np.argsort(-arr))
    else:
        order = list(np.argsort(arr))
    total = float(arr.sum())
    return SequenceResult(
        order=order,
        total_score=total,
        params={"algorithm": "by_score", "descending": descending},
    )


def compute_sequence_score(
    order: List[int],
    score_matrix: np.ndarray,
) -> float:
    """Вычислить суммарную оценку последовательности пар.

    Args:
        order:        Список индексов фрагментов.
        score_matrix: Матрица оценок (N, N).

    Returns:
        Сумма оценок соседних пар.

    Raises:
        ValueError: Если матрица не квадратная или индекс вне диапазона.
    """
    mat = np.asarray(score_matrix, dtype=np.float64)
    _validate_square(mat)
    n = mat.shape[0]
    if len(order) <= 1:
        return 0.0
    total = 0.0
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        if not (0 <= a < n and 0 <= b < n):
            raise ValueError(
                f"Index out of range: {a} or {b} not in [0, {n - 1}]"
            )
        total += mat[a, b]
    return float(total)


def reverse_sequence(result: SequenceResult) -> SequenceResult:
    """Обратить порядок последовательности.

    Args:
        result: Входной результат последовательности.

    Returns:
        Новый :class:`SequenceResult` с обратным порядком.
    """
    return SequenceResult(
        order=list(reversed(result.order)),
        total_score=result.total_score,
        params=dict(result.params),
    )


def rotate_sequence(result: SequenceResult, start_idx: int) -> SequenceResult:
    """Циклически сдвинуть последовательность так, чтобы начинаться с ``start_idx``.

    Args:
        result:    Входной результат.
        start_idx: Новый начальный элемент (значение, не позиция).

    Returns:
        Новый :class:`SequenceResult` с повёрнутым порядком.

    Raises:
        ValueError: Если ``start_idx`` не найден в ``result.order``.
    """
    order = result.order
    if start_idx not in order:
        raise ValueError(f"start_idx {start_idx} not in order {order}")
    pos = order.index(start_idx)
    new_order = order[pos:] + order[:pos]
    return SequenceResult(
        order=new_order,
        total_score=result.total_score,
        params=dict(result.params),
    )


def sequence_to_pairs(result: SequenceResult) -> List[Tuple[int, int]]:
    """Преобразовать последовательность в список соседних пар.

    Args:
        result: Результат последовательности.

    Returns:
        Список кортежей (a, b) для соседних элементов.
        Для последовательности длины 0 или 1 возвращает [].
    """
    order = result.order
    if len(order) <= 1:
        return []
    return [(order[i], order[i + 1]) for i in range(len(order) - 1)]


def find_best_start(
    order: List[int],
    score_matrix: np.ndarray,
) -> int:
    """Найти оптимальный стартовый индекс для циклической последовательности.

    Перебирает все циклические сдвиги, выбирает тот, при котором
    суммарная оценка (с учётом последней→первой пары) максимальна.

    Args:
        order:        Список индексов фрагментов.
        score_matrix: Матрица оценок (N, N).

    Returns:
        Значение (fragment index), с которого выгоднее начать.

    Raises:
        ValueError: Если ``order`` пустой.
    """
    if not order:
        raise ValueError("order must not be empty")
    mat = np.asarray(score_matrix, dtype=np.float64)
    n_seq = len(order)
    if n_seq == 1:
        return order[0]

    best_start = order[0]
    best_score = -1.0
    for pos in range(n_seq):
        rotated = order[pos:] + order[:pos]
        score = compute_sequence_score(rotated, mat)
        # Include wrap-around pair
        last, first = rotated[-1], rotated[0]
        mat_n = mat.shape[0]
        if 0 <= last < mat_n and 0 <= first < mat_n:
            score += mat[last, first]
        if score > best_score:
            best_score = score
            best_start = order[pos]

    return best_start


def batch_sequence(
    score_matrices: List[np.ndarray],
    start: Optional[int] = None,
) -> List[SequenceResult]:
    """Пакетное построение последовательностей для нескольких матриц.

    Args:
        score_matrices: Список матриц оценок.
        start:          Начальный индекс (передаётся в :func:`sequence_greedy`).

    Returns:
        Список :class:`SequenceResult` той же длины.
    """
    return [
        sequence_greedy(mat, start=start)
        for mat in score_matrices
    ]


# ─── Приватные ───────────────────────────────────────────────────────────────

def _validate_square(mat: np.ndarray) -> None:
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(
            f"score_matrix must be a square 2-D array, got shape {mat.shape}"
        )
