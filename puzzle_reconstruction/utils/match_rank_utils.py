"""Утилиты агрегации результатов ранжирования совпадений и оценок соответствия.

Модуль предоставляет вспомогательные структуры и функции для:
- Хранения и анализа результатов ранжирования пар-кандидатов
- Агрегации и сравнения отчётов оценки совпадений
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# ═══════════════════════════════════════════════════════════════════════════════
# Candidate Ranking Result Analysis
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RankingConfig:
    """Конфигурация анализа результатов ранжирования.

    Атрибуты:
        min_score:   Минимальный порог оценки для включения пары.
        top_k:       Количество лучших пар для анализа.
        deduplicate: Применять ли дедупликацию индексов.
    """
    min_score: float = 0.0
    top_k: int = 10
    deduplicate: bool = False


@dataclass
class RankingEntry:
    """Запись результата ранжирования одной партии пар.

    Атрибуты:
        batch_id:     Идентификатор партии.
        n_pairs:      Число пар в партии.
        n_accepted:   Число пар, принятых выше порога.
        top_score:    Максимальная оценка в партии.
        mean_score:   Средняя оценка принятых пар.
        algorithm:    Метка алгоритма ранжирования.
        params:       Дополнительные параметры.
    """
    batch_id: int
    n_pairs: int
    n_accepted: int
    top_score: float
    mean_score: float
    algorithm: str
    params: Dict = field(default_factory=dict)

    @property
    def acceptance_rate(self) -> float:
        """Доля принятых пар."""
        return self.n_accepted / self.n_pairs if self.n_pairs > 0 else 0.0


@dataclass
class RankingSummary:
    """Сводка результатов нескольких партий ранжирования.

    Атрибуты:
        n_batches:           Число партий.
        total_pairs:         Суммарное число пар.
        total_accepted:      Суммарное число принятых пар.
        mean_top_score:      Средняя максимальная оценка.
        mean_acceptance_rate: Среднее соотношение принятых пар.
        best_batch_id:       ID партии с наилучшей top_score или None.
        worst_batch_id:      ID партии с наихудшей top_score или None.
    """
    n_batches: int
    total_pairs: int
    total_accepted: int
    mean_top_score: float
    mean_acceptance_rate: float
    best_batch_id: Optional[int]
    worst_batch_id: Optional[int]


def make_ranking_entry(
    batch_id: int,
    n_pairs: int,
    n_accepted: int,
    top_score: float,
    mean_score: float,
    algorithm: str,
    **params,
) -> RankingEntry:
    """Создать запись результата ранжирования.

    Args:
        batch_id:   Идентификатор партии.
        n_pairs:    Число пар.
        n_accepted: Число принятых пар.
        top_score:  Максимальная оценка.
        mean_score: Средняя оценка.
        algorithm:  Метка алгоритма.
        **params:   Дополнительные параметры.

    Returns:
        :class:`RankingEntry`.
    """
    return RankingEntry(
        batch_id=batch_id,
        n_pairs=int(n_pairs),
        n_accepted=int(n_accepted),
        top_score=float(top_score),
        mean_score=float(mean_score),
        algorithm=algorithm,
        params=dict(params),
    )


def summarise_ranking_entries(
    entries: Sequence[RankingEntry],
    cfg: Optional[RankingConfig] = None,
) -> RankingSummary:
    """Сформировать сводку результатов ранжирования.

    Args:
        entries: Список записей.
        cfg:     Конфигурация (None → RankingConfig()).

    Returns:
        :class:`RankingSummary`.
    """
    if cfg is None:
        cfg = RankingConfig()
    if not entries:
        return RankingSummary(
            n_batches=0,
            total_pairs=0,
            total_accepted=0,
            mean_top_score=0.0,
            mean_acceptance_rate=0.0,
            best_batch_id=None,
            worst_batch_id=None,
        )
    top_scores = [e.top_score for e in entries]
    rates = [e.acceptance_rate for e in entries]
    best = max(entries, key=lambda e: e.top_score)
    worst = min(entries, key=lambda e: e.top_score)
    return RankingSummary(
        n_batches=len(entries),
        total_pairs=sum(e.n_pairs for e in entries),
        total_accepted=sum(e.n_accepted for e in entries),
        mean_top_score=sum(top_scores) / len(top_scores),
        mean_acceptance_rate=sum(rates) / len(rates),
        best_batch_id=best.batch_id,
        worst_batch_id=worst.batch_id,
    )


def filter_ranking_by_algorithm(
    entries: Sequence[RankingEntry],
    algorithm: str,
) -> List[RankingEntry]:
    """Отфильтровать записи ранжирования по алгоритму."""
    return [e for e in entries if e.algorithm == algorithm]


def filter_ranking_by_min_top_score(
    entries: Sequence[RankingEntry],
    min_top_score: float,
) -> List[RankingEntry]:
    """Отфильтровать записи по минимальной максимальной оценке."""
    return [e for e in entries if e.top_score >= min_top_score]


def filter_ranking_by_min_acceptance(
    entries: Sequence[RankingEntry],
    min_rate: float,
) -> List[RankingEntry]:
    """Отфильтровать записи по минимальному соотношению принятых пар."""
    return [e for e in entries if e.acceptance_rate >= min_rate]


def top_k_ranking_entries(
    entries: Sequence[RankingEntry],
    k: int,
) -> List[RankingEntry]:
    """Вернуть k записей с наивысшей top_score."""
    return sorted(entries, key=lambda e: e.top_score, reverse=True)[:k]


def best_ranking_entry(
    entries: Sequence[RankingEntry],
) -> Optional[RankingEntry]:
    """Вернуть запись с наивысшей top_score."""
    return max(entries, key=lambda e: e.top_score) if entries else None


def ranking_score_stats(
    entries: Sequence[RankingEntry],
) -> Dict:
    """Вычислить статистику top_score: min, max, mean, std."""
    if not entries:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "count": 0}
    vals = [e.top_score for e in entries]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {
        "min": min(vals),
        "max": max(vals),
        "mean": mean,
        "std": var ** 0.5,
        "count": len(vals),
    }


def compare_ranking_summaries(
    a: RankingSummary,
    b: RankingSummary,
) -> Dict:
    """Сравнить две сводки результатов ранжирования."""
    return {
        "delta_mean_top_score": b.mean_top_score - a.mean_top_score,
        "delta_total_accepted": b.total_accepted - a.total_accepted,
        "delta_acceptance_rate": b.mean_acceptance_rate - a.mean_acceptance_rate,
        "same_best": a.best_batch_id == b.best_batch_id,
    }


def batch_summarise_ranking_entries(
    groups: Sequence[Sequence[RankingEntry]],
    cfg: Optional[RankingConfig] = None,
) -> List[RankingSummary]:
    """Сформировать сводки для нескольких групп записей ранжирования."""
    return [summarise_ranking_entries(g, cfg) for g in groups]


# ═══════════════════════════════════════════════════════════════════════════════
# Match Evaluation Result Analysis
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalResultConfig:
    """Конфигурация анализа результатов оценки совпадений.

    Атрибуты:
        min_f1:    Минимальный допустимый F1-score.
        min_score: Минимальный допустимый score.
    """
    min_f1: float = 0.0
    min_score: float = 0.0


@dataclass
class EvalResultEntry:
    """Запись результата оценки набора совпадений.

    Атрибуты:
        run_id:     Идентификатор прогона.
        n_pairs:    Число пар.
        mean_score: Средняя оценка совпадений.
        mean_f1:    Средний F1-score.
        best_f1:    Лучший F1-score.
        algorithm:  Метка алгоритма оценки.
        params:     Дополнительные параметры.
    """
    run_id: int
    n_pairs: int
    mean_score: float
    mean_f1: float
    best_f1: float
    algorithm: str
    params: Dict = field(default_factory=dict)


@dataclass
class EvalResultSummary:
    """Сводка результатов нескольких прогонов оценки совпадений.

    Атрибуты:
        n_runs:         Число прогонов.
        total_pairs:    Суммарное число пар.
        mean_f1:        Среднее mean_f1 по прогонам.
        mean_best_f1:   Среднее best_f1 по прогонам.
        best_run_id:    ID прогона с наивысшим best_f1 или None.
        worst_run_id:   ID прогона с наименьшим best_f1 или None.
    """
    n_runs: int
    total_pairs: int
    mean_f1: float
    mean_best_f1: float
    best_run_id: Optional[int]
    worst_run_id: Optional[int]


def make_eval_result_entry(
    run_id: int,
    n_pairs: int,
    mean_score: float,
    mean_f1: float,
    best_f1: float,
    algorithm: str,
    **params,
) -> EvalResultEntry:
    """Создать запись результата оценки совпадений.

    Args:
        run_id:     Идентификатор прогона.
        n_pairs:    Число пар.
        mean_score: Средняя оценка.
        mean_f1:    Средний F1.
        best_f1:    Лучший F1.
        algorithm:  Метка алгоритма.
        **params:   Дополнительные параметры.

    Returns:
        :class:`EvalResultEntry`.
    """
    return EvalResultEntry(
        run_id=run_id,
        n_pairs=int(n_pairs),
        mean_score=float(mean_score),
        mean_f1=float(mean_f1),
        best_f1=float(best_f1),
        algorithm=algorithm,
        params=dict(params),
    )


def summarise_eval_result_entries(
    entries: Sequence[EvalResultEntry],
    cfg: Optional[EvalResultConfig] = None,
) -> EvalResultSummary:
    """Сформировать сводку по результатам оценки совпадений.

    Args:
        entries: Список записей.
        cfg:     Конфигурация (None → EvalResultConfig()).

    Returns:
        :class:`EvalResultSummary`.
    """
    if cfg is None:
        cfg = EvalResultConfig()
    if not entries:
        return EvalResultSummary(
            n_runs=0,
            total_pairs=0,
            mean_f1=0.0,
            mean_best_f1=0.0,
            best_run_id=None,
            worst_run_id=None,
        )
    f1s = [e.mean_f1 for e in entries]
    best_f1s = [e.best_f1 for e in entries]
    best = max(entries, key=lambda e: e.best_f1)
    worst = min(entries, key=lambda e: e.best_f1)
    return EvalResultSummary(
        n_runs=len(entries),
        total_pairs=sum(e.n_pairs for e in entries),
        mean_f1=sum(f1s) / len(f1s),
        mean_best_f1=sum(best_f1s) / len(best_f1s),
        best_run_id=best.run_id,
        worst_run_id=worst.run_id,
    )


def filter_eval_by_min_f1(
    entries: Sequence[EvalResultEntry],
    min_f1: float,
) -> List[EvalResultEntry]:
    """Отфильтровать записи по минимальному mean_f1."""
    return [e for e in entries if e.mean_f1 >= min_f1]


def filter_eval_by_algorithm(
    entries: Sequence[EvalResultEntry],
    algorithm: str,
) -> List[EvalResultEntry]:
    """Отфильтровать записи оценки по алгоритму."""
    return [e for e in entries if e.algorithm == algorithm]


def top_k_eval_entries(
    entries: Sequence[EvalResultEntry],
    k: int,
) -> List[EvalResultEntry]:
    """Вернуть k записей с наивысшим best_f1."""
    return sorted(entries, key=lambda e: e.best_f1, reverse=True)[:k]


def best_eval_entry(
    entries: Sequence[EvalResultEntry],
) -> Optional[EvalResultEntry]:
    """Вернуть запись с наивысшим best_f1."""
    return max(entries, key=lambda e: e.best_f1) if entries else None


def eval_f1_stats(
    entries: Sequence[EvalResultEntry],
) -> Dict:
    """Вычислить статистику mean_f1: min, max, mean, std."""
    if not entries:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "count": 0}
    vals = [e.mean_f1 for e in entries]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {
        "min": min(vals),
        "max": max(vals),
        "mean": mean,
        "std": var ** 0.5,
        "count": len(vals),
    }


def compare_eval_summaries(
    a: EvalResultSummary,
    b: EvalResultSummary,
) -> Dict:
    """Сравнить две сводки результатов оценки совпадений."""
    return {
        "delta_mean_f1": b.mean_f1 - a.mean_f1,
        "delta_mean_best_f1": b.mean_best_f1 - a.mean_best_f1,
        "delta_total_pairs": b.total_pairs - a.total_pairs,
        "same_best": a.best_run_id == b.best_run_id,
    }


def batch_summarise_eval_entries(
    groups: Sequence[Sequence[EvalResultEntry]],
    cfg: Optional[EvalResultConfig] = None,
) -> List[EvalResultSummary]:
    """Сформировать сводки для нескольких групп записей оценки."""
    return [summarise_eval_result_entries(g, cfg) for g in groups]
