"""Оценка качества сопоставления пар фрагментов пазла.

Модуль вычисляет метрики точности, полноты и F1 для найденных
соответствий, агрегирует их по батчу и ранжирует пары по качеству.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ─── EvalConfig ───────────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    """Параметры оценки качества сопоставления.

    Атрибуты:
        min_score:   Нижняя граница диапазона оценок (>= 0).
        max_score:   Верхняя граница (> min_score).
        n_levels:    Число уровней для PR-кривой (>= 2).
        beta:        Вес полноты в F-мере (> 0; 1.0 → F1).
    """

    min_score: float = 0.0
    max_score: float = 1.0
    n_levels: int = 10
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.min_score < 0.0:
            raise ValueError(
                f"min_score должен быть >= 0, получено {self.min_score}"
            )
        if self.max_score <= self.min_score:
            raise ValueError(
                f"max_score должен быть > min_score: "
                f"{self.max_score} <= {self.min_score}"
            )
        if self.n_levels < 2:
            raise ValueError(
                f"n_levels должен быть >= 2, получено {self.n_levels}"
            )
        if self.beta <= 0.0:
            raise ValueError(
                f"beta должен быть > 0, получено {self.beta}"
            )


# ─── MatchEval ────────────────────────────────────────────────────────────────

@dataclass
class MatchEval:
    """Результат оценки одной пары фрагментов.

    Атрибуты:
        pair:   Упорядоченная пара (fragment_id_a, fragment_id_b).
        score:  Оценка совместимости (>= 0).
        tp:     True positives (>= 0).
        fp:     False positives (>= 0).
        fn:     False negatives (>= 0).
    """

    pair: Tuple[int, int]
    score: float
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __post_init__(self) -> None:
        if self.score < 0.0:
            raise ValueError(
                f"score должен быть >= 0, получено {self.score}"
            )
        for name, val in (("tp", self.tp), ("fp", self.fp), ("fn", self.fn)):
            if val < 0:
                raise ValueError(
                    f"{name} должен быть >= 0, получено {val}"
                )

    @property
    def precision(self) -> float:
        """Точность = TP / (TP + FP). 0.0 если TP+FP=0."""
        denom = self.tp + self.fp
        return float(self.tp) / float(denom) if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Полнота = TP / (TP + FN). 0.0 если TP+FN=0."""
        denom = self.tp + self.fn
        return float(self.tp) / float(denom) if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        """F1 = 2·P·R / (P+R). 0.0 если P+R=0."""
        p, r = self.precision, self.recall
        return 2.0 * p * r / (p + r) if (p + r) > 0.0 else 0.0


# ─── EvalReport ───────────────────────────────────────────────────────────────

@dataclass
class EvalReport:
    """Агрегированный отчёт по набору сопоставлений.

    Атрибуты:
        evals:      Список MatchEval.
        n_pairs:    Число оценённых пар (>= 0).
        mean_score: Средняя оценка (>= 0).
        mean_f1:    Средний F1 (>= 0).
    """

    evals: List[MatchEval]
    n_pairs: int
    mean_score: float
    mean_f1: float

    def __post_init__(self) -> None:
        if self.n_pairs < 0:
            raise ValueError(
                f"n_pairs должен быть >= 0, получено {self.n_pairs}"
            )
        if self.mean_score < 0.0:
            raise ValueError(
                f"mean_score должен быть >= 0, получено {self.mean_score}"
            )
        if self.mean_f1 < 0.0:
            raise ValueError(
                f"mean_f1 должен быть >= 0, получено {self.mean_f1}"
            )

    @property
    def best_f1(self) -> float:
        """Максимальный F1 среди всех пар (0.0 если пусто)."""
        if not self.evals:
            return 0.0
        return max(e.f1 for e in self.evals)

    @property
    def best_pair(self) -> Optional[Tuple[int, int]]:
        """Пара с наибольшим F1 (None если нет оценок)."""
        if not self.evals:
            return None
        return max(self.evals, key=lambda e: e.f1).pair


# ─── compute_precision ────────────────────────────────────────────────────────

def compute_precision(tp: int, fp: int) -> float:
    """Вычислить точность.

    Аргументы:
        tp: True positives (>= 0).
        fp: False positives (>= 0).

    Возвращает:
        Precision в [0, 1].

    Исключения:
        ValueError: Если tp < 0 или fp < 0.
    """
    if tp < 0:
        raise ValueError(f"tp должен быть >= 0, получено {tp}")
    if fp < 0:
        raise ValueError(f"fp должен быть >= 0, получено {fp}")
    denom = tp + fp
    return float(tp) / float(denom) if denom > 0 else 0.0


# ─── compute_recall ───────────────────────────────────────────────────────────

def compute_recall(tp: int, fn: int) -> float:
    """Вычислить полноту.

    Аргументы:
        tp: True positives (>= 0).
        fn: False negatives (>= 0).

    Возвращает:
        Recall в [0, 1].

    Исключения:
        ValueError: Если tp < 0 или fn < 0.
    """
    if tp < 0:
        raise ValueError(f"tp должен быть >= 0, получено {tp}")
    if fn < 0:
        raise ValueError(f"fn должен быть >= 0, получено {fn}")
    denom = tp + fn
    return float(tp) / float(denom) if denom > 0 else 0.0


# ─── compute_f_score ──────────────────────────────────────────────────────────

def compute_f_score(precision: float, recall: float, beta: float = 1.0) -> float:
    """Вычислить взвешенную F-меру.

    Аргументы:
        precision: Точность (>= 0).
        recall:    Полнота (>= 0).
        beta:      Вес полноты (> 0).

    Возвращает:
        F_beta в [0, 1].

    Исключения:
        ValueError: Если beta <= 0.
    """
    if beta <= 0.0:
        raise ValueError(f"beta должен быть > 0, получено {beta}")
    b2 = beta ** 2
    denom = b2 * precision + recall
    if denom < 1e-12:
        return 0.0
    return float((1.0 + b2) * precision * recall / ((b2 * precision + recall)))


# ─── evaluate_match ───────────────────────────────────────────────────────────

def evaluate_match(
    pair: Tuple[int, int],
    score: float,
    tp: int,
    fp: int,
    fn: int,
) -> MatchEval:
    """Создать MatchEval для одной пары.

    Аргументы:
        pair:  (id_a, id_b).
        score: Оценка (>= 0).
        tp, fp, fn: Счётчики TP/FP/FN (>= 0).

    Возвращает:
        MatchEval.
    """
    return MatchEval(pair=pair, score=score, tp=tp, fp=fp, fn=fn)


# ─── evaluate_batch_matches ───────────────────────────────────────────────────

def evaluate_batch_matches(
    pairs: List[Tuple[int, int]],
    scores: List[float],
    tp_list: List[int],
    fp_list: List[int],
    fn_list: List[int],
) -> List[MatchEval]:
    """Оценить список пар.

    Аргументы:
        pairs:   Список пар.
        scores:  Оценки (len == len(pairs)).
        tp_list, fp_list, fn_list: Счётчики.

    Возвращает:
        Список MatchEval.

    Исключения:
        ValueError: Если длины не совпадают.
    """
    n = len(pairs)
    for name, lst in (("scores", scores), ("tp_list", tp_list),
                      ("fp_list", fp_list), ("fn_list", fn_list)):
        if len(lst) != n:
            raise ValueError(
                f"Длина {name} ({len(lst)}) != len(pairs) ({n})"
            )
    return [
        evaluate_match(p, s, tp, fp, fn)
        for p, s, tp, fp, fn
        in zip(pairs, scores, tp_list, fp_list, fn_list)
    ]


# ─── aggregate_eval ───────────────────────────────────────────────────────────

def aggregate_eval(evals: List[MatchEval]) -> EvalReport:
    """Агрегировать список MatchEval в отчёт.

    Аргументы:
        evals: Список оценок.

    Возвращает:
        EvalReport.
    """
    n = len(evals)
    if n == 0:
        return EvalReport(evals=[], n_pairs=0, mean_score=0.0, mean_f1=0.0)

    mean_score = float(np.mean([e.score for e in evals]))
    mean_f1 = float(np.mean([e.f1 for e in evals]))
    return EvalReport(
        evals=evals,
        n_pairs=n,
        mean_score=mean_score,
        mean_f1=mean_f1,
    )


# ─── filter_by_score ──────────────────────────────────────────────────────────

def filter_by_score(
    evals: List[MatchEval],
    threshold: float,
) -> List[MatchEval]:
    """Оставить только оценки с score >= threshold.

    Аргументы:
        evals:     Список MatchEval.
        threshold: Порог (>= 0).

    Возвращает:
        Отфильтрованный список.

    Исключения:
        ValueError: Если threshold < 0.
    """
    if threshold < 0.0:
        raise ValueError(
            f"threshold должен быть >= 0, получено {threshold}"
        )
    return [e for e in evals if e.score >= threshold]


# ─── rank_matches ─────────────────────────────────────────────────────────────

def rank_matches(
    evals: List[MatchEval],
    by: str = "f1",
) -> List[MatchEval]:
    """Ранжировать список MatchEval по убыванию f1 или score.

    Аргументы:
        evals: Список MatchEval.
        by:    'f1' | 'score'.

    Возвращает:
        Новый список, упорядоченный по убыванию.

    Исключения:
        ValueError: Если by неизвестен.
    """
    if by not in ("f1", "score"):
        raise ValueError(f"by должен быть 'f1' или 'score', получено '{by}'")
    key = (lambda e: e.f1) if by == "f1" else (lambda e: e.score)
    return sorted(evals, key=key, reverse=True)
