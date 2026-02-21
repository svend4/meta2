"""Агрегация оценок от нескольких матчеров в итоговую оценку пары.

Модуль позволяет комбинировать векторы оценок (по одной от каждого
матчера/метрики) с помощью различных стратегий: взвешенная сумма,
среднее, максимум, минимум, медиана, произведение и ранговое
агрегирование.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


_VALID_STRATEGIES = {"weighted", "mean", "max", "min", "median", "product", "rank"}


# ─── AggregationConfig ────────────────────────────────────────────────────────

@dataclass
class AggregationConfig:
    """Параметры агрегации оценок.

    Атрибуты:
        strategy:    'weighted' | 'mean' | 'max' | 'min' | 'median' |
                     'product' | 'rank'.
        weights:     Список весов (None → равные; len == n_sources для 'weighted').
        clip_min:    Нижняя граница обрезки итоговой оценки (>= 0).
        clip_max:    Верхняя граница обрезки итоговой оценки (clip_max > clip_min).
        normalize:   Привести итоговые оценки в [0, 1] по всему батчу.
    """

    strategy: str = "mean"
    weights: Optional[List[float]] = None
    clip_min: float = 0.0
    clip_max: float = 1.0
    normalize: bool = False

    def __post_init__(self) -> None:
        if self.strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"strategy должна быть одной из {_VALID_STRATEGIES}, "
                f"получено '{self.strategy}'"
            )
        if self.clip_min < 0.0:
            raise ValueError(
                f"clip_min должен быть >= 0, получено {self.clip_min}"
            )
        if self.clip_max <= self.clip_min:
            raise ValueError(
                f"clip_max ({self.clip_max}) должен быть > clip_min ({self.clip_min})"
            )
        if self.weights is not None:
            if any(w < 0.0 for w in self.weights):
                raise ValueError("Все веса должны быть >= 0")
            if len(self.weights) == 0:
                raise ValueError("weights не должен быть пустым")


# ─── AggregatedScore ─────────────────────────────────────────────────────────

@dataclass
class AggregatedScore:
    """Итоговая агрегированная оценка для одной пары.

    Атрибуты:
        pair:     Пара идентификаторов фрагментов.
        score:    Итоговая оценка (>= 0).
        sources:  Исходные оценки от каждого матчера.
        strategy: Использованная стратегия.
    """

    pair: Tuple[int, int]
    score: float
    sources: List[float]
    strategy: str

    def __post_init__(self) -> None:
        if self.score < 0.0:
            raise ValueError(
                f"score должен быть >= 0, получено {self.score}"
            )
        if not self.strategy:
            raise ValueError("strategy не должна быть пустой")

    @property
    def n_sources(self) -> int:
        """Число исходных оценок."""
        return len(self.sources)

    @property
    def source_max(self) -> float:
        """Максимальное значение среди исходных оценок (0 если нет)."""
        return float(max(self.sources)) if self.sources else 0.0

    @property
    def source_min(self) -> float:
        """Минимальное значение среди исходных оценок (0 если нет)."""
        return float(min(self.sources)) if self.sources else 0.0


# ─── AggregationReport ────────────────────────────────────────────────────────

@dataclass
class AggregationReport:
    """Отчёт по агрегации батча пар.

    Атрибуты:
        scores:       Список AggregatedScore.
        n_pairs:      Число пар (>= 0).
        strategy:     Использованная стратегия.
        mean_score:   Среднее итоговых оценок (>= 0).
    """

    scores: List[AggregatedScore]
    n_pairs: int
    strategy: str
    mean_score: float

    def __post_init__(self) -> None:
        if self.n_pairs < 0:
            raise ValueError(
                f"n_pairs должен быть >= 0, получено {self.n_pairs}"
            )
        if self.mean_score < 0.0:
            raise ValueError(
                f"mean_score должен быть >= 0, получено {self.mean_score}"
            )

    @property
    def best_pair(self) -> Optional[Tuple[int, int]]:
        """Пара с наибольшей итоговой оценкой (None если scores пуст)."""
        if not self.scores:
            return None
        return max(self.scores, key=lambda s: s.score).pair

    @property
    def best_score(self) -> float:
        """Наибольшая итоговая оценка (0 если scores пуст)."""
        if not self.scores:
            return 0.0
        return float(max(s.score for s in self.scores))


# ─── _uniform_weights ─────────────────────────────────────────────────────────

def _uniform_weights(n: int) -> np.ndarray:
    return np.ones(n, dtype=float) / n


# ─── aggregate_scores ─────────────────────────────────────────────────────────

def aggregate_scores(
    source_scores: List[float],
    cfg: Optional[AggregationConfig] = None,
    pair: Tuple[int, int] = (0, 0),
) -> AggregatedScore:
    """Агрегировать вектор оценок в одно число.

    Аргументы:
        source_scores: Список оценок (по одной от каждого матчера).
        cfg:           Параметры (None → AggregationConfig()).
        pair:          Пара идентификаторов.

    Возвращает:
        AggregatedScore.

    Исключения:
        ValueError: Если source_scores пуст или длина не совпадает с весами.
    """
    if cfg is None:
        cfg = AggregationConfig()
    if len(source_scores) == 0:
        raise ValueError("source_scores не должен быть пустым")

    arr = np.array(source_scores, dtype=float)

    if cfg.strategy == "weighted":
        weights = (
            np.array(cfg.weights, dtype=float)
            if cfg.weights is not None
            else _uniform_weights(len(arr))
        )
        if len(weights) != len(arr):
            raise ValueError(
                f"Длина weights ({len(weights)}) не совпадает с "
                f"длиной source_scores ({len(arr)})"
            )
        w_sum = weights.sum()
        score = float((arr * weights).sum() / w_sum) if w_sum > 0 else 0.0
    elif cfg.strategy == "mean":
        score = float(np.mean(arr))
    elif cfg.strategy == "max":
        score = float(np.max(arr))
    elif cfg.strategy == "min":
        score = float(np.min(arr))
    elif cfg.strategy == "median":
        score = float(np.median(arr))
    elif cfg.strategy == "product":
        score = float(np.prod(arr))
    else:  # rank
        ranks = arr.argsort().argsort().astype(float) + 1.0
        score = float(np.mean(ranks / len(arr)))

    score = float(np.clip(score, cfg.clip_min, cfg.clip_max))
    return AggregatedScore(
        pair=pair,
        score=score,
        sources=list(source_scores),
        strategy=cfg.strategy,
    )


# ─── aggregate_score_matrix ───────────────────────────────────────────────────

def aggregate_score_matrix(
    matrices: List[np.ndarray],
    cfg: Optional[AggregationConfig] = None,
) -> np.ndarray:
    """Агрегировать список матриц оценок в одну матрицу.

    Аргументы:
        matrices: Список 2D-матриц оценок одинакового размера.
        cfg:      Параметры.

    Возвращает:
        Агрегированная 2D-матрица.

    Исключения:
        ValueError: Если matrices пуст или матрицы имеют разные размеры.
    """
    if cfg is None:
        cfg = AggregationConfig()
    if len(matrices) == 0:
        raise ValueError("matrices не должен быть пустым")

    shapes = [m.shape for m in matrices]
    if len(set(shapes)) > 1:
        raise ValueError(
            f"Все матрицы должны иметь одинаковый размер, "
            f"получено: {shapes}"
        )

    stack = np.stack([m.astype(float) for m in matrices], axis=0)

    if cfg.strategy == "weighted":
        weights = (
            np.array(cfg.weights, dtype=float)
            if cfg.weights is not None
            else _uniform_weights(len(matrices))
        )
        if len(weights) != len(matrices):
            raise ValueError(
                f"Длина weights ({len(weights)}) не совпадает с "
                f"числом матриц ({len(matrices)})"
            )
        w = weights / (weights.sum() + 1e-12)
        result = np.einsum("k,k...->...", w, stack)
    elif cfg.strategy == "mean":
        result = np.mean(stack, axis=0)
    elif cfg.strategy == "max":
        result = np.max(stack, axis=0)
    elif cfg.strategy == "min":
        result = np.min(stack, axis=0)
    elif cfg.strategy == "median":
        result = np.median(stack, axis=0)
    elif cfg.strategy == "product":
        result = np.prod(stack, axis=0)
    else:  # rank
        result = np.mean(stack, axis=0)

    result = np.clip(result, cfg.clip_min, cfg.clip_max)

    if cfg.normalize:
        r_min, r_max = result.min(), result.max()
        if r_max > r_min:
            result = (result - r_min) / (r_max - r_min)
        else:
            result = np.zeros_like(result)

    return result


# ─── batch_aggregate_scores ───────────────────────────────────────────────────

def batch_aggregate_scores(
    pairs: List[Tuple[int, int]],
    source_score_lists: List[List[float]],
    cfg: Optional[AggregationConfig] = None,
) -> AggregationReport:
    """Агрегировать оценки для батча пар.

    Аргументы:
        pairs:             Список пар идентификаторов.
        source_score_lists: Список векторов оценок (по одному на пару).
        cfg:               Параметры.

    Возвращает:
        AggregationReport.

    Исключения:
        ValueError: Если длины lists не совпадают.
    """
    if cfg is None:
        cfg = AggregationConfig()
    if len(pairs) != len(source_score_lists):
        raise ValueError(
            f"Длины pairs ({len(pairs)}) и source_score_lists "
            f"({len(source_score_lists)}) не совпадают"
        )

    agg_scores: List[AggregatedScore] = []
    for pair, sources in zip(pairs, source_score_lists):
        agg_scores.append(aggregate_scores(sources, cfg, pair))

    mean_score = float(np.mean([s.score for s in agg_scores])) if agg_scores else 0.0
    return AggregationReport(
        scores=agg_scores,
        n_pairs=len(agg_scores),
        strategy=cfg.strategy,
        mean_score=mean_score,
    )


# ─── filter_aggregated ────────────────────────────────────────────────────────

def filter_aggregated(
    report: AggregationReport,
    threshold: float,
) -> List[AggregatedScore]:
    """Оставить только пары с итоговой оценкой >= threshold.

    Аргументы:
        report:    AggregationReport.
        threshold: Пороговое значение (>= 0).

    Возвращает:
        Отфильтрованный список AggregatedScore.

    Исключения:
        ValueError: Если threshold < 0.
    """
    if threshold < 0.0:
        raise ValueError(f"threshold должен быть >= 0, получено {threshold}")
    return [s for s in report.scores if s.score >= threshold]
