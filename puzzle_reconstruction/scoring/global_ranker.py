"""Глобальное ранжирование пар фрагментов по совокупным оценкам.

Модуль агрегирует оценки из нескольких источников (совместимость краёв,
SIFT, текстура), нормирует их, ранжирует пары фрагментов и выдаёт
топ-K кандидатов для каждого фрагмента.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── RankedPair ───────────────────────────────────────────────────────────────

@dataclass
class RankedPair:
    """Ранжированная пара фрагментов.

    Атрибуты:
        idx1:      Индекс первого фрагмента.
        idx2:      Индекс второго фрагмента.
        score:     Агрегированная оценка (float в [0, 1]).
        rank:      Ранг (0 = лучший).
        component_scores: Словарь {источник: оценка}.
    """

    idx1: int
    idx2: int
    score: float
    rank: int
    component_scores: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.idx1 < 0:
            raise ValueError(f"idx1 должен быть >= 0, получено {self.idx1}")
        if self.idx2 < 0:
            raise ValueError(f"idx2 должен быть >= 0, получено {self.idx2}")
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"score должен быть в [0, 1], получено {self.score}"
            )
        if self.rank < 0:
            raise ValueError(f"rank должен быть >= 0, получено {self.rank}")

    @property
    def pair(self) -> Tuple[int, int]:
        return (self.idx1, self.idx2)


# ─── RankingConfig ────────────────────────────────────────────────────────────

@dataclass
class RankingConfig:
    """Параметры глобального ранжирования.

    Атрибуты:
        weights:        Словарь {источник: вес} (все >= 0).
        top_k:          Число лучших кандидатов на фрагмент (>= 1).
        normalize:      Нормировать матрицы перед агрегацией.
        min_score:      Минимальный порог агрегированной оценки (>= 0).
        symmetric:      Симметризовать агрегированную матрицу.
    """

    weights: Dict[str, float] = field(
        default_factory=lambda: {"boundary": 0.5, "sift": 0.3, "texture": 0.2}
    )
    top_k: int = 5
    normalize: bool = True
    min_score: float = 0.0
    symmetric: bool = True

    def __post_init__(self) -> None:
        for name, w in self.weights.items():
            if w < 0.0:
                raise ValueError(
                    f"Вес '{name}' должен быть >= 0, получено {w}"
                )
        if self.top_k < 1:
            raise ValueError(
                f"top_k должен быть >= 1, получено {self.top_k}"
            )
        if self.min_score < 0.0:
            raise ValueError(
                f"min_score должен быть >= 0, получено {self.min_score}"
            )


# ─── normalize_matrix ─────────────────────────────────────────────────────────

def normalize_matrix(M: np.ndarray) -> np.ndarray:
    """Нормировать матрицу оценок к диапазону [0, 1] (min-max).

    Аргументы:
        M: Матрица (N×N, float64), диагональ не учитывается.

    Возвращает:
        Нормированная матрица (float64), диагональ = 0.

    Исключения:
        ValueError: Если M не квадратная 2-D.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(
            f"M должна быть квадратной 2-D, получено shape={M.shape}"
        )
    N = M.shape[0]
    mask = ~np.eye(N, dtype=bool)
    vals = M[mask]
    vmin, vmax = vals.min() if len(vals) else 0.0, vals.max() if len(vals) else 0.0
    if vmax - vmin < 1e-12:
        result = np.zeros_like(M)
    else:
        result = (M - vmin) / (vmax - vmin)
    np.fill_diagonal(result, 0.0)
    return result


# ─── aggregate_score_matrices ─────────────────────────────────────────────────

def aggregate_score_matrices(
    matrices: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
    normalize: bool = True,
    symmetric: bool = True,
) -> np.ndarray:
    """Агрегировать несколько матриц оценок в одну.

    Аргументы:
        matrices:  Словарь {имя: матрица (N×N)}.
        weights:   Словарь {имя: вес}. None → равные веса.
        normalize: Нормировать каждую матрицу перед взвешиванием.
        symmetric: Симметризовать результат.

    Возвращает:
        Агрегированная матрица (N×N, float64) с диагональю = 0.

    Исключения:
        ValueError: Если matrices пуст или размеры не совпадают.
    """
    if not matrices:
        raise ValueError("matrices не может быть пустым")

    names = list(matrices.keys())
    mats = list(matrices.values())

    # Проверяем согласованность размеров
    N = mats[0].shape[0]
    for m in mats[1:]:
        if m.shape != (N, N):
            raise ValueError(
                f"Все матрицы должны иметь одинаковый размер N×N, "
                f"ожидается ({N}, {N}), получено {m.shape}"
            )

    if weights is None:
        weights = {n: 1.0 for n in names}

    total_weight = sum(weights.get(n, 0.0) for n in names)
    if total_weight < 1e-12:
        total_weight = 1.0

    result = np.zeros((N, N), dtype=np.float64)
    for name, mat in zip(names, mats):
        w = weights.get(name, 0.0)
        if w == 0.0:
            continue
        m = mat.astype(np.float64)
        if normalize:
            m = normalize_matrix(m)
        result += w * m

    result /= total_weight
    np.fill_diagonal(result, 0.0)

    if symmetric:
        result = (result + result.T) / 2.0

    return result


# ─── rank_pairs ───────────────────────────────────────────────────────────────

def rank_pairs(
    agg_matrix: np.ndarray,
    min_score: float = 0.0,
) -> List[RankedPair]:
    """Ранжировать все пары (i, j) по убыванию агрегированной оценки.

    Аргументы:
        agg_matrix: Квадратная матрица (N×N).
        min_score:  Нижний порог оценки (пары ниже исключаются).

    Возвращает:
        Список RankedPair, отсортированный по убыванию score.

    Исключения:
        ValueError: Если agg_matrix не квадратная.
    """
    agg_matrix = np.asarray(agg_matrix, dtype=np.float64)
    if agg_matrix.ndim != 2 or agg_matrix.shape[0] != agg_matrix.shape[1]:
        raise ValueError("agg_matrix должна быть квадратной 2-D")

    N = agg_matrix.shape[0]
    pairs: List[Tuple[float, int, int]] = []
    for i in range(N):
        for j in range(i + 1, N):
            score = float(agg_matrix[i, j])
            if score >= min_score:
                pairs.append((score, i, j))

    pairs.sort(key=lambda x: x[0], reverse=True)
    return [
        RankedPair(idx1=i, idx2=j, score=s, rank=r)
        for r, (s, i, j) in enumerate(pairs)
    ]


# ─── top_k_candidates ────────────────────────────────────────────────────────

def top_k_candidates(
    ranked_pairs: List[RankedPair],
    n_fragments: int,
    k: int,
) -> Dict[int, List[RankedPair]]:
    """Выбрать топ-K кандидатов для каждого фрагмента.

    Аргументы:
        ranked_pairs: Отсортированный список RankedPair.
        n_fragments:  Число фрагментов.
        k:            Число лучших кандидатов на фрагмент (>= 1).

    Возвращает:
        Словарь {fragment_id: список лучших RankedPair}.

    Исключения:
        ValueError: Если k < 1 или n_fragments < 1.
    """
    if k < 1:
        raise ValueError(f"k должен быть >= 1, получено {k}")
    if n_fragments < 1:
        raise ValueError(
            f"n_fragments должен быть >= 1, получено {n_fragments}"
        )

    result: Dict[int, List[RankedPair]] = {i: [] for i in range(n_fragments)}
    for pair in ranked_pairs:
        for fid in (pair.idx1, pair.idx2):
            if fid < n_fragments and len(result[fid]) < k:
                result[fid].append(pair)
    return result


# ─── global_rank ──────────────────────────────────────────────────────────────

def global_rank(
    matrices: Dict[str, np.ndarray],
    cfg: Optional[RankingConfig] = None,
) -> List[RankedPair]:
    """Выполнить полное глобальное ранжирование пар.

    Аргументы:
        matrices: Словарь {имя: матрица оценок (N×N)}.
        cfg:      Параметры ранжирования (None → RankingConfig()).

    Возвращает:
        Список RankedPair, отсортированный по убыванию оценки.
    """
    if cfg is None:
        cfg = RankingConfig()
    agg = aggregate_score_matrices(
        matrices,
        weights=cfg.weights,
        normalize=cfg.normalize,
        symmetric=cfg.symmetric,
    )
    return rank_pairs(agg, min_score=cfg.min_score)


# ─── score_vector ─────────────────────────────────────────────────────────────

def score_vector(
    ranked_pairs: List[RankedPair],
    n_fragments: int,
) -> np.ndarray:
    """Вычислить средний рейтинг каждого фрагмента.

    Аргументы:
        ranked_pairs: Список RankedPair.
        n_fragments:  Число фрагментов.

    Возвращает:
        Массив (N,) float64 — средняя оценка участия каждого фрагмента.

    Исключения:
        ValueError: Если n_fragments < 1.
    """
    if n_fragments < 1:
        raise ValueError(
            f"n_fragments должен быть >= 1, получено {n_fragments}"
        )
    scores = np.zeros(n_fragments, dtype=np.float64)
    counts = np.zeros(n_fragments, dtype=np.int64)
    for pair in ranked_pairs:
        for fid in (pair.idx1, pair.idx2):
            if fid < n_fragments:
                scores[fid] += pair.score
                counts[fid] += 1
    mask = counts > 0
    scores[mask] /= counts[mask]
    return scores


# ─── batch_global_rank ────────────────────────────────────────────────────────

def batch_global_rank(
    matrix_groups: List[Dict[str, np.ndarray]],
    cfg: Optional[RankingConfig] = None,
) -> List[List[RankedPair]]:
    """Глобальное ранжирование для нескольких наборов матриц.

    Аргументы:
        matrix_groups: Список словарей {имя: матрица}.
        cfg:           Параметры ранжирования.

    Возвращает:
        Список списков RankedPair.
    """
    return [global_rank(matrices, cfg) for matrices in matrix_groups]
