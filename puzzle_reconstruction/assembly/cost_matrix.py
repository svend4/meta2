"""
Построение и преобразование матриц стоимостей для алгоритмов сборки.

Предоставляет функции создания матриц стоимости из матриц оценок/расстояний,
их нормализации, маскирования и агрегации нескольких источников.

Классы:
    CostMatrix — обёртка над np.ndarray с метаданными

Функции:
    build_from_scores     — матрица стоимости = 1 - нормированная оценка
    build_from_distances  — матрица стоимости из расстояний с нормализацией
    build_combined        — взвешенная комбинация нескольких матриц стоимости
    apply_forbidden_mask  — запрет отдельных пар (→ inf / high cost)
    normalize_costs       — нормализация матрицы стоимости
    to_assignment_matrix  — квадратная матрица для венгерского алгоритма
    top_k_candidates      — топ-k кандидатов для каждого фрагмента
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── CostMatrix ───────────────────────────────────────────────────────────────

@dataclass
class CostMatrix:
    """
    Обёртка над квадратной матрицей стоимостей.

    Attributes:
        matrix:       np.ndarray формы (N, N), float32.
        n_fragments:  Число фрагментов (= N).
        method:       Метод построения.
        params:       Дополнительные параметры.
    """
    matrix:      np.ndarray
    n_fragments: int
    method:      str
    params:      Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.matrix.shape != (self.n_fragments, self.n_fragments):
            raise ValueError(
                f"matrix shape {self.matrix.shape} does not match "
                f"n_fragments={self.n_fragments}."
            )

    def __repr__(self) -> str:
        return (f"CostMatrix(n={self.n_fragments}, method={self.method!r}, "
                f"min={self.matrix.min():.3f}, max={self.matrix.max():.3f})")


# ─── build_from_scores ────────────────────────────────────────────────────────

def build_from_scores(
    score_matrix: np.ndarray,
    invert:       bool = True,
    eps:          float = 1e-6,
) -> CostMatrix:
    """
    Строит матрицу стоимости из матрицы оценок совместимости.

    cost = 1 - score  (если invert=True), иначе score нормализуется напрямую.
    Диагональ всегда равна 0.

    Args:
        score_matrix: (N, N) матрица оценок (чем выше → лучше), float.
        invert:       True → cost = 1 - score_norm.
        eps:          Небольшое смещение для численной устойчивости.

    Returns:
        CostMatrix с методом "from_scores".

    Raises:
        ValueError: Если матрица не квадратная.
    """
    n = score_matrix.shape[0]
    if score_matrix.ndim != 2 or score_matrix.shape[1] != n:
        raise ValueError(
            f"score_matrix must be square, got shape {score_matrix.shape}.")

    m = score_matrix.astype(np.float32)
    mn, mx = float(m.min()), float(m.max())
    if abs(mx - mn) < eps:
        norm = np.zeros_like(m)
    else:
        norm = (m - mn) / (mx - mn + eps)

    cost = (1.0 - norm) if invert else norm
    np.fill_diagonal(cost, 0.0)

    return CostMatrix(
        matrix=cost.astype(np.float32),
        n_fragments=n,
        method="from_scores",
        params={"invert": invert},
    )


# ─── build_from_distances ─────────────────────────────────────────────────────

def build_from_distances(
    distance_matrix: np.ndarray,
    normalize:       bool  = True,
    eps:             float = 1e-6,
) -> CostMatrix:
    """
    Строит матрицу стоимости из матрицы расстояний.

    Расстояния уже являются стоимостями; нормализация приводит их к [0, 1].
    Диагональ всегда равна 0.

    Args:
        distance_matrix: (N, N) матрица расстояний (≥ 0).
        normalize:        True → нормализовать в [0, 1].
        eps:              Числовая константа устойчивости.

    Returns:
        CostMatrix с методом "from_distances".

    Raises:
        ValueError: Если матрица не квадратная.
    """
    n = distance_matrix.shape[0]
    if distance_matrix.ndim != 2 or distance_matrix.shape[1] != n:
        raise ValueError(
            f"distance_matrix must be square, got shape {distance_matrix.shape}.")

    m = np.abs(distance_matrix).astype(np.float32)
    np.fill_diagonal(m, 0.0)

    if normalize:
        mx = float(m.max())
        if mx > eps:
            m = m / mx

    return CostMatrix(
        matrix=m,
        n_fragments=n,
        method="from_distances",
        params={"normalize": normalize},
    )


# ─── build_combined ───────────────────────────────────────────────────────────

def build_combined(
    cost_matrices: List[CostMatrix],
    weights:       Optional[List[float]] = None,
) -> CostMatrix:
    """
    Строит взвешенную комбинацию нескольких матриц стоимости.

    Args:
        cost_matrices: Список CostMatrix с одинаковым n_fragments.
        weights:       Список весов (нормализуется до суммы 1.0).
                       None → равные веса.

    Returns:
        CostMatrix с методом "combined".

    Raises:
        ValueError: Если список пуст или n_fragments не совпадают.
    """
    if not cost_matrices:
        raise ValueError("cost_matrices must not be empty.")

    n = cost_matrices[0].n_fragments
    for cm in cost_matrices[1:]:
        if cm.n_fragments != n:
            raise ValueError(
                f"All cost matrices must have the same n_fragments ({n}), "
                f"got {cm.n_fragments}.")

    if weights is None:
        w = np.full(len(cost_matrices), 1.0 / len(cost_matrices))
    else:
        w = np.array(weights, dtype=np.float64)
        s = w.sum()
        if s < 1e-9:
            raise ValueError("Sum of weights must be > 0.")
        w /= s

    combined = np.zeros((n, n), dtype=np.float32)
    for cm, wi in zip(cost_matrices, w):
        combined += float(wi) * cm.matrix
    np.fill_diagonal(combined, 0.0)

    return CostMatrix(
        matrix=combined,
        n_fragments=n,
        method="combined",
        params={"n_sources": len(cost_matrices), "weights": list(w)},
    )


# ─── apply_forbidden_mask ─────────────────────────────────────────────────────

def apply_forbidden_mask(
    cm:          CostMatrix,
    mask:        np.ndarray,
    fill_value:  float = 1.0,
) -> CostMatrix:
    """
    Устанавливает запрещённые пары (mask == True) в fill_value.

    Args:
        cm:         Входная CostMatrix.
        mask:       Булева матрица (N, N). True → запрещённая пара.
        fill_value: Значение для запрещённых пар (≥ 1.0 → очень дорого).

    Returns:
        Новая CostMatrix с методом "masked".

    Raises:
        ValueError: Если форма mask не совпадает с (N, N).
    """
    n = cm.n_fragments
    if mask.shape != (n, n):
        raise ValueError(
            f"mask shape {mask.shape} must match (n_fragments, n_fragments) = ({n}, {n}).")

    result = cm.matrix.copy()
    result[mask.astype(bool)] = fill_value
    np.fill_diagonal(result, 0.0)

    return CostMatrix(
        matrix=result,
        n_fragments=n,
        method="masked",
        params={"fill_value": fill_value, "n_forbidden": int(mask.sum())},
    )


# ─── normalize_costs ──────────────────────────────────────────────────────────

def normalize_costs(
    cm:     CostMatrix,
    method: str = "minmax",
) -> CostMatrix:
    """
    Нормализует матрицу стоимости.

    Args:
        cm:     Входная CostMatrix.
        method: "minmax"  — линейная нормализация в [0, 1];
                "zscore"  — стандартизация (μ=0, σ=1), затем клиппинг в [0, 1];
                "rank"    — замена значений рангами, нормированными в [0, 1].

    Returns:
        Новая CostMatrix с методом "normalized_{method}".

    Raises:
        ValueError: Неизвестный метод.
    """
    if method not in ("minmax", "zscore", "rank"):
        raise ValueError(
            f"Unknown normalization method {method!r}. "
            f"Choose 'minmax', 'zscore', or 'rank'.")

    m = cm.matrix.astype(np.float64)
    n = cm.n_fragments

    # Не учитываем диагональ при вычислении статистик
    mask_diag = ~np.eye(n, dtype=bool)
    off_diag  = m[mask_diag]

    if method == "minmax":
        mn, mx = off_diag.min(), off_diag.max()
        r = m.copy()
        if abs(mx - mn) > 1e-12:
            r[mask_diag] = (off_diag - mn) / (mx - mn)
        else:
            r[mask_diag] = 0.0

    elif method == "zscore":
        mu, sigma = off_diag.mean(), off_diag.std()
        r = m.copy()
        if sigma > 1e-12:
            r[mask_diag] = (off_diag - mu) / sigma
        r = np.clip(r, 0.0, None)
        # Нормализуем ещё раз в [0, 1]
        mx = r[mask_diag].max() if r[mask_diag].size > 0 else 1.0
        if mx > 1e-12:
            r[mask_diag] /= mx

    else:  # rank
        order = np.argsort(off_diag)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(order), dtype=np.float64)
        ranks /= max(len(ranks) - 1, 1)
        r = m.copy()
        r[mask_diag] = ranks

    np.fill_diagonal(r, 0.0)
    return CostMatrix(
        matrix=r.astype(np.float32),
        n_fragments=n,
        method=f"normalized_{method}",
        params={"source_method": cm.method},
    )


# ─── to_assignment_matrix ─────────────────────────────────────────────────────

def to_assignment_matrix(cm: CostMatrix) -> np.ndarray:
    """
    Возвращает квадратную матрицу стоимостей для венгерского алгоритма.

    Диагональ заменяется максимальным значением + 1 (чтобы исключить
    самоприсваивание).

    Args:
        cm: Входная CostMatrix.

    Returns:
        np.ndarray формы (N, N) float32.
    """
    m  = cm.matrix.astype(np.float32).copy()
    mx = float(m.max())
    np.fill_diagonal(m, mx + 1.0)
    return m


# ─── top_k_candidates ─────────────────────────────────────────────────────────

def top_k_candidates(
    cm: CostMatrix,
    k:  int = 5,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Возвращает топ-k кандидатов (наименьшая стоимость) для каждого фрагмента.

    Args:
        cm: Входная CostMatrix.
        k:  Число кандидатов.

    Returns:
        Словарь {fragment_idx: [(candidate_idx, cost), ...]} по возрастанию стоимости.
    """
    n      = cm.n_fragments
    k_eff  = min(k, n - 1)  # без самого себя
    result: Dict[int, List[Tuple[int, float]]] = {}

    for i in range(n):
        row    = cm.matrix[i].copy()
        row[i] = np.inf  # исключаем себя
        if k_eff == 0:
            result[i] = []
            continue
        top_idx = np.argsort(row)[:k_eff]
        result[i] = [(int(j), float(cm.matrix[i, j])) for j in top_idx]

    return result
