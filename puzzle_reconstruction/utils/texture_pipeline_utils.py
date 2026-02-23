"""Утилиты анализа результатов текстурного сопоставления и пайплайн-трекинга.

Модуль предоставляет структуры данных и функции для:
- Хранения и агрегации результатов текстурного сопоставления пар фрагментов
- Отслеживания прогресса текстурного анализа по батчам
- Сравнения результатов между прогонами
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# Texture Match Analysis
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TextureMatchRecord:
    """Запись результата текстурного сопоставления одной пары фрагментов.

    Атрибуты:
        pair:           Пара индексов фрагментов (idx1, idx2).
        score:          Итоговая оценка сходства (0–1).
        lbp_score:      Оценка LBP-гистограмм (0–1).
        gabor_score:    Оценка Gabor-признаков (0–1).
        gradient_score: Оценка ориентаций градиентов (0–1).
        side1:          Сторона первого фрагмента.
        side2:          Сторона второго фрагмента.
        params:         Дополнительные параметры.
    """
    pair: Tuple[int, int]
    score: float
    lbp_score: float
    gabor_score: float
    gradient_score: float
    side1: int = 0
    side2: int = 0
    params: Dict = field(default_factory=dict)


@dataclass
class TextureMatchSummary:
    """Сводка результатов текстурного сопоставления набора пар.

    Атрибуты:
        n_pairs:         Число пар.
        mean_score:      Средняя итоговая оценка.
        mean_lbp:        Средняя LBP-оценка.
        mean_gabor:      Средняя Gabor-оценка.
        mean_gradient:   Средняя оценка ориентаций.
        best_pair:       Пара с наибольшей оценкой или None.
        best_score:      Наибольшая итоговая оценка.
    """
    n_pairs: int
    mean_score: float
    mean_lbp: float
    mean_gabor: float
    mean_gradient: float
    best_pair: Optional[Tuple[int, int]]
    best_score: float


def make_texture_match_record(
    pair: Tuple[int, int],
    score: float,
    lbp_score: float,
    gabor_score: float,
    gradient_score: float,
    side1: int = 0,
    side2: int = 0,
    **params,
) -> TextureMatchRecord:
    """Создать запись текстурного сопоставления.

    Args:
        pair:           Пара индексов (idx1, idx2).
        score:          Итоговая оценка.
        lbp_score:      LBP-оценка.
        gabor_score:    Gabor-оценка.
        gradient_score: Оценка ориентаций.
        side1:          Сторона первого фрагмента.
        side2:          Сторона второго фрагмента.
        **params:       Дополнительные параметры.

    Returns:
        :class:`TextureMatchRecord`.
    """
    return TextureMatchRecord(
        pair=pair,
        score=float(score),
        lbp_score=float(lbp_score),
        gabor_score=float(gabor_score),
        gradient_score=float(gradient_score),
        side1=int(side1),
        side2=int(side2),
        params=dict(params),
    )


def summarise_texture_matches(
    records: Sequence[TextureMatchRecord],
) -> TextureMatchSummary:
    """Сформировать сводку по результатам текстурного сопоставления.

    Args:
        records: Список записей.

    Returns:
        :class:`TextureMatchSummary`.
    """
    if not records:
        return TextureMatchSummary(
            n_pairs=0,
            mean_score=0.0,
            mean_lbp=0.0,
            mean_gabor=0.0,
            mean_gradient=0.0,
            best_pair=None,
            best_score=0.0,
        )
    n = len(records)
    best = max(records, key=lambda r: r.score)
    return TextureMatchSummary(
        n_pairs=n,
        mean_score=sum(r.score for r in records) / n,
        mean_lbp=sum(r.lbp_score for r in records) / n,
        mean_gabor=sum(r.gabor_score for r in records) / n,
        mean_gradient=sum(r.gradient_score for r in records) / n,
        best_pair=best.pair,
        best_score=best.score,
    )


def filter_texture_by_score(
    records: Sequence[TextureMatchRecord],
    threshold: float,
) -> List[TextureMatchRecord]:
    """Отфильтровать записи по минимальной итоговой оценке."""
    return [r for r in records if r.score >= threshold]


def filter_texture_by_lbp(
    records: Sequence[TextureMatchRecord],
    threshold: float,
) -> List[TextureMatchRecord]:
    """Отфильтровать записи по минимальной LBP-оценке."""
    return [r for r in records if r.lbp_score >= threshold]


def top_k_texture_records(
    records: Sequence[TextureMatchRecord],
    k: int,
) -> List[TextureMatchRecord]:
    """Вернуть k записей с наивысшей итоговой оценкой."""
    return sorted(records, key=lambda r: r.score, reverse=True)[:k]


def best_texture_record(
    records: Sequence[TextureMatchRecord],
) -> Optional[TextureMatchRecord]:
    """Вернуть запись с наивысшей итоговой оценкой."""
    return max(records, key=lambda r: r.score) if records else None


def texture_score_stats(
    records: Sequence[TextureMatchRecord],
) -> Dict:
    """Вычислить статистику score: min, max, mean, std."""
    if not records:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "count": 0}
    vals = [r.score for r in records]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {
        "min": min(vals),
        "max": max(vals),
        "mean": mean,
        "std": var ** 0.5,
        "count": len(vals),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Batch Tracking
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BatchPipelineRecord:
    """Запись результата обработки одного батча пайплайна.

    Атрибуты:
        batch_id:        Идентификатор батча.
        n_items:         Число элементов в батче.
        n_done:          Число успешно обработанных.
        n_failed:        Число упавших.
        n_skipped:       Число пропущенных.
        elapsed:         Время выполнения в секундах.
        stage:           Метка этапа пайплайна.
        params:          Дополнительные параметры.
    """
    batch_id: int
    n_items: int
    n_done: int
    n_failed: int
    n_skipped: int
    elapsed: float
    stage: str
    params: Dict = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Доля успешно обработанных элементов."""
        return self.n_done / self.n_items if self.n_items > 0 else 0.0

    @property
    def throughput(self) -> float:
        """Число обработанных элементов в секунду."""
        return self.n_done / self.elapsed if self.elapsed > 0 else 0.0


@dataclass
class BatchPipelineSummary:
    """Сводка результатов нескольких батчей пайплайна.

    Атрибуты:
        n_batches:           Число батчей.
        total_items:         Суммарное число элементов.
        total_done:          Суммарное число успешных.
        total_failed:        Суммарное число упавших.
        total_elapsed:       Суммарное время выполнения.
        mean_success_rate:   Среднее success_rate.
        best_batch_id:       ID батча с наибольшим success_rate или None.
        worst_batch_id:      ID батча с наименьшим success_rate или None.
    """
    n_batches: int
    total_items: int
    total_done: int
    total_failed: int
    total_elapsed: float
    mean_success_rate: float
    best_batch_id: Optional[int]
    worst_batch_id: Optional[int]


def make_batch_pipeline_record(
    batch_id: int,
    n_items: int,
    n_done: int,
    n_failed: int,
    n_skipped: int,
    elapsed: float,
    stage: str,
    **params,
) -> BatchPipelineRecord:
    """Создать запись результата обработки батча.

    Args:
        batch_id:  Идентификатор батча.
        n_items:   Число элементов.
        n_done:    Число успешных.
        n_failed:  Число упавших.
        n_skipped: Число пропущенных.
        elapsed:   Время выполнения.
        stage:     Этап пайплайна.
        **params:  Дополнительные параметры.

    Returns:
        :class:`BatchPipelineRecord`.
    """
    return BatchPipelineRecord(
        batch_id=int(batch_id),
        n_items=int(n_items),
        n_done=int(n_done),
        n_failed=int(n_failed),
        n_skipped=int(n_skipped),
        elapsed=float(elapsed),
        stage=stage,
        params=dict(params),
    )


def summarise_batch_pipeline(
    records: Sequence[BatchPipelineRecord],
) -> BatchPipelineSummary:
    """Сформировать сводку по результатам батчей пайплайна.

    Args:
        records: Список записей.

    Returns:
        :class:`BatchPipelineSummary`.
    """
    if not records:
        return BatchPipelineSummary(
            n_batches=0,
            total_items=0,
            total_done=0,
            total_failed=0,
            total_elapsed=0.0,
            mean_success_rate=0.0,
            best_batch_id=None,
            worst_batch_id=None,
        )
    rates = [r.success_rate for r in records]
    best = max(records, key=lambda r: r.success_rate)
    worst = min(records, key=lambda r: r.success_rate)
    return BatchPipelineSummary(
        n_batches=len(records),
        total_items=sum(r.n_items for r in records),
        total_done=sum(r.n_done for r in records),
        total_failed=sum(r.n_failed for r in records),
        total_elapsed=sum(r.elapsed for r in records),
        mean_success_rate=sum(rates) / len(rates),
        best_batch_id=best.batch_id,
        worst_batch_id=worst.batch_id,
    )


def filter_batch_by_success_rate(
    records: Sequence[BatchPipelineRecord],
    min_rate: float,
) -> List[BatchPipelineRecord]:
    """Отфильтровать батчи по минимальному success_rate."""
    return [r for r in records if r.success_rate >= min_rate]


def filter_batch_by_stage(
    records: Sequence[BatchPipelineRecord],
    stage: str,
) -> List[BatchPipelineRecord]:
    """Отфильтровать батчи по этапу пайплайна."""
    return [r for r in records if r.stage == stage]


def top_k_batch_records(
    records: Sequence[BatchPipelineRecord],
    k: int,
) -> List[BatchPipelineRecord]:
    """Вернуть k батчей с наибольшим success_rate."""
    return sorted(records, key=lambda r: r.success_rate, reverse=True)[:k]


def batch_throughput_stats(
    records: Sequence[BatchPipelineRecord],
) -> Dict:
    """Вычислить статистику throughput: min, max, mean."""
    if not records:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "count": 0}
    vals = [r.throughput for r in records]
    return {
        "min": min(vals),
        "max": max(vals),
        "mean": sum(vals) / len(vals),
        "count": len(vals),
    }


def compare_batch_summaries(
    a: BatchPipelineSummary,
    b: BatchPipelineSummary,
) -> Dict:
    """Сравнить две сводки по батчам пайплайна."""
    return {
        "delta_total_done": b.total_done - a.total_done,
        "delta_total_failed": b.total_failed - a.total_failed,
        "delta_mean_success_rate": b.mean_success_rate - a.mean_success_rate,
        "delta_total_elapsed": b.total_elapsed - a.total_elapsed,
        "same_best": a.best_batch_id == b.best_batch_id,
    }
