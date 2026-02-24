"""Утилиты агрегации результатов нормализации и шумоподавления.

Модуль предоставляет вспомогательные структуры и функции для:
- Хранения и сравнения результатов нормализации матриц оценок
- Анализа и ранжирования результатов шумоподавления изображений
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# ═══════════════════════════════════════════════════════════════════════════════
# Score Normalization Result Analysis
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NormResultConfig:
    """Конфигурация анализа результатов нормализации матриц.

    Атрибуты:
        preferred_method:  Предпочтительный метод нормализации.
        min_spread:        Минимальный разброс нормализованных значений.
    """
    preferred_method: str = "minmax"
    min_spread: float = 0.0


@dataclass
class NormResultEntry:
    """Запись результата нормализации одной матрицы.

    Атрибуты:
        run_id:    Идентификатор прогона.
        method:    Метод нормализации (например, 'minmax', 'zscore').
        min_val:   Минимальное значение до нормализации.
        max_val:   Максимальное значение до нормализации.
        spread:    Разброс (max_val - min_val).
        n_rows:    Число строк матрицы.
        n_cols:    Число столбцов матрицы.
        params:    Дополнительные параметры.
    """
    run_id: int
    method: str
    min_val: float
    max_val: float
    spread: float
    n_rows: int
    n_cols: int
    params: Dict = field(default_factory=dict)


@dataclass
class NormResultSummary:
    """Сводка результатов нормализации нескольких матриц.

    Атрибуты:
        n_runs:        Число прогонов.
        mean_spread:   Средний разброс.
        method_counts: Словарь с числом прогонов по каждому методу.
        best_run_id:   ID прогона с наибольшим разбросом или None.
        worst_run_id:  ID прогона с наименьшим разбросом или None.
    """
    n_runs: int
    mean_spread: float
    method_counts: Dict
    best_run_id: Optional[int]
    worst_run_id: Optional[int]


def make_norm_result_entry(
    run_id: int,
    method: str,
    min_val: float,
    max_val: float,
    n_rows: int,
    n_cols: int,
    **params,
) -> NormResultEntry:
    """Создать запись результата нормализации.

    Args:
        run_id:  Идентификатор прогона.
        method:  Метод нормализации.
        min_val: Минимальное значение.
        max_val: Максимальное значение.
        n_rows:  Число строк.
        n_cols:  Число столбцов.
        **params: Дополнительные параметры.

    Returns:
        :class:`NormResultEntry`.
    """
    return NormResultEntry(
        run_id=run_id,
        method=method,
        min_val=float(min_val),
        max_val=float(max_val),
        spread=float(max_val) - float(min_val),
        n_rows=int(n_rows),
        n_cols=int(n_cols),
        params=dict(params),
    )


def summarise_norm_result_entries(
    entries: Sequence[NormResultEntry],
    cfg: Optional[NormResultConfig] = None,
) -> NormResultSummary:
    """Сформировать сводку по результатам нормализации.

    Args:
        entries: Список записей.
        cfg:     Конфигурация (None → NormResultConfig()).

    Returns:
        :class:`NormResultSummary`.
    """
    if cfg is None:
        cfg = NormResultConfig()
    if not entries:
        return NormResultSummary(
            n_runs=0,
            mean_spread=0.0,
            method_counts={},
            best_run_id=None,
            worst_run_id=None,
        )
    spreads = [e.spread for e in entries]
    mean_spread = sum(spreads) / len(spreads)
    method_counts: Dict = {}
    for e in entries:
        method_counts[e.method] = method_counts.get(e.method, 0) + 1
    best = max(entries, key=lambda e: e.spread)
    worst = min(entries, key=lambda e: e.spread)
    return NormResultSummary(
        n_runs=len(entries),
        mean_spread=mean_spread,
        method_counts=method_counts,
        best_run_id=best.run_id,
        worst_run_id=worst.run_id,
    )


def filter_norm_by_method(
    entries: Sequence[NormResultEntry],
    method: str,
) -> List[NormResultEntry]:
    """Отфильтровать записи по методу нормализации."""
    return [e for e in entries if e.method == method]


def filter_norm_by_min_spread(
    entries: Sequence[NormResultEntry],
    min_spread: float,
) -> List[NormResultEntry]:
    """Отфильтровать записи по минимальному разбросу."""
    return [e for e in entries if e.spread >= min_spread]


def top_k_norm_by_spread(
    entries: Sequence[NormResultEntry],
    k: int,
) -> List[NormResultEntry]:
    """Вернуть k записей с наибольшим разбросом."""
    return sorted(entries, key=lambda e: e.spread, reverse=True)[:k]


def best_norm_entry(
    entries: Sequence[NormResultEntry],
) -> Optional[NormResultEntry]:
    """Вернуть запись с наибольшим разбросом."""
    return max(entries, key=lambda e: e.spread) if entries else None


def norm_spread_stats(
    entries: Sequence[NormResultEntry],
) -> Dict:
    """Вычислить статистику разбросов: min, max, mean, std."""
    if not entries:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "count": 0}
    vals = [e.spread for e in entries]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {
        "min": min(vals),
        "max": max(vals),
        "mean": mean,
        "std": var ** 0.5,
        "count": len(vals),
    }


def compare_norm_summaries(
    a: NormResultSummary,
    b: NormResultSummary,
) -> Dict:
    """Сравнить две сводки результатов нормализации."""
    return {
        "delta_mean_spread": b.mean_spread - a.mean_spread,
        "delta_n_runs": b.n_runs - a.n_runs,
        "same_best": a.best_run_id == b.best_run_id,
    }


def batch_summarise_norm_entries(
    groups: Sequence[Sequence[NormResultEntry]],
    cfg: Optional[NormResultConfig] = None,
) -> List[NormResultSummary]:
    """Сформировать сводки для нескольких групп нормализованных записей."""
    return [summarise_norm_result_entries(g, cfg) for g in groups]


# ═══════════════════════════════════════════════════════════════════════════════
# Noise Reduction Result Analysis
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NoiseResultConfig:
    """Конфигурация анализа результатов шумоподавления.

    Атрибуты:
        max_noise_after:  Максимально допустимый шум после обработки.
        preferred_method: Предпочтительный метод шумоподавления.
    """
    max_noise_after: float = float("inf")
    preferred_method: str = "gaussian"


@dataclass
class NoiseResultEntry:
    """Запись результата шумоподавления одного изображения.

    Атрибуты:
        image_id:       Идентификатор изображения.
        method:         Метод шумоподавления.
        noise_before:   Оценка шума до обработки.
        noise_after:    Оценка шума после обработки.
        noise_delta:    Снижение шума (noise_before - noise_after).
        n_pixels:       Число пикселей.
        params:         Дополнительные параметры.
    """
    image_id: int
    method: str
    noise_before: float
    noise_after: float
    noise_delta: float
    n_pixels: int
    params: Dict = field(default_factory=dict)


@dataclass
class NoiseResultSummary:
    """Сводка результатов шумоподавления набора изображений.

    Атрибуты:
        n_images:         Число изображений.
        mean_noise_before: Средний шум до обработки.
        mean_noise_after:  Средний шум после обработки.
        mean_delta:        Среднее снижение шума.
        best_image_id:     ID изображения с наибольшим снижением шума или None.
        worst_image_id:    ID изображения с наименьшим снижением шума или None.
    """
    n_images: int
    mean_noise_before: float
    mean_noise_after: float
    mean_delta: float
    best_image_id: Optional[int]
    worst_image_id: Optional[int]


def make_noise_result_entry(
    image_id: int,
    method: str,
    noise_before: float,
    noise_after: float,
    n_pixels: int,
    **params,
) -> NoiseResultEntry:
    """Создать запись результата шумоподавления.

    Args:
        image_id:     Идентификатор изображения.
        method:       Метод шумоподавления.
        noise_before: Оценка шума до обработки.
        noise_after:  Оценка шума после обработки.
        n_pixels:     Число пикселей.
        **params:     Дополнительные параметры.

    Returns:
        :class:`NoiseResultEntry`.
    """
    return NoiseResultEntry(
        image_id=image_id,
        method=method,
        noise_before=float(noise_before),
        noise_after=float(noise_after),
        noise_delta=float(noise_before) - float(noise_after),
        n_pixels=int(n_pixels),
        params=dict(params),
    )


def summarise_noise_result_entries(
    entries: Sequence[NoiseResultEntry],
    cfg: Optional[NoiseResultConfig] = None,
) -> NoiseResultSummary:
    """Сформировать сводку по результатам шумоподавления.

    Args:
        entries: Список записей.
        cfg:     Конфигурация (None → NoiseResultConfig()).

    Returns:
        :class:`NoiseResultSummary`.
    """
    if cfg is None:
        cfg = NoiseResultConfig()
    if not entries:
        return NoiseResultSummary(
            n_images=0,
            mean_noise_before=0.0,
            mean_noise_after=0.0,
            mean_delta=0.0,
            best_image_id=None,
            worst_image_id=None,
        )
    nb = [e.noise_before for e in entries]
    na = [e.noise_after for e in entries]
    deltas = [e.noise_delta for e in entries]
    best = max(entries, key=lambda e: e.noise_delta)
    worst = min(entries, key=lambda e: e.noise_delta)
    return NoiseResultSummary(
        n_images=len(entries),
        mean_noise_before=sum(nb) / len(nb),
        mean_noise_after=sum(na) / len(na),
        mean_delta=sum(deltas) / len(deltas),
        best_image_id=best.image_id,
        worst_image_id=worst.image_id,
    )


def filter_noise_by_method(
    entries: Sequence[NoiseResultEntry],
    method: str,
) -> List[NoiseResultEntry]:
    """Отфильтровать записи по методу шумоподавления."""
    return [e for e in entries if e.method == method]


def filter_noise_by_max_after(
    entries: Sequence[NoiseResultEntry],
    max_noise_after: float,
) -> List[NoiseResultEntry]:
    """Отфильтровать записи по максимально допустимому шуму после обработки."""
    return [e for e in entries if e.noise_after <= max_noise_after]


def filter_noise_by_min_delta(
    entries: Sequence[NoiseResultEntry],
    min_delta: float,
) -> List[NoiseResultEntry]:
    """Отфильтровать записи по минимальному снижению шума."""
    return [e for e in entries if e.noise_delta >= min_delta]


def top_k_noise_by_delta(
    entries: Sequence[NoiseResultEntry],
    k: int,
) -> List[NoiseResultEntry]:
    """Вернуть k записей с наибольшим снижением шума."""
    return sorted(entries, key=lambda e: e.noise_delta, reverse=True)[:k]


def best_noise_entry(
    entries: Sequence[NoiseResultEntry],
) -> Optional[NoiseResultEntry]:
    """Вернуть запись с наибольшим снижением шума."""
    return max(entries, key=lambda e: e.noise_delta) if entries else None


def noise_delta_stats(
    entries: Sequence[NoiseResultEntry],
) -> Dict:
    """Вычислить статистику снижения шума: min, max, mean, std."""
    if not entries:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "count": 0}
    vals = [e.noise_delta for e in entries]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {
        "min": min(vals),
        "max": max(vals),
        "mean": mean,
        "std": var ** 0.5,
        "count": len(vals),
    }


def compare_noise_summaries(
    a: NoiseResultSummary,
    b: NoiseResultSummary,
) -> Dict:
    """Сравнить две сводки результатов шумоподавления."""
    return {
        "delta_mean_noise_before": b.mean_noise_before - a.mean_noise_before,
        "delta_mean_noise_after": b.mean_noise_after - a.mean_noise_after,
        "delta_mean_delta": b.mean_delta - a.mean_delta,
        "same_best": a.best_image_id == b.best_image_id,
    }


def batch_summarise_noise_entries(
    groups: Sequence[Sequence[NoiseResultEntry]],
    cfg: Optional[NoiseResultConfig] = None,
) -> List[NoiseResultSummary]:
    """Сформировать сводки для нескольких групп записей шумоподавления."""
    return [summarise_noise_result_entries(g, cfg) for g in groups]
