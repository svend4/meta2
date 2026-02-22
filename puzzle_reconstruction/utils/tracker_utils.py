"""tracker_utils — утилиты для отслеживания итерационного прогресса.

Предоставляет инструменты для записи, анализа и сравнения числовых
последовательностей в итерационных алгоритмах (оптимизация, сборка,
поиск с возвратом и т.д.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ─── Конфигурация ─────────────────────────────────────────────────────────────

@dataclass
class TrackerConfig:
    """Параметры трекера итераций."""
    window: int = 5
    tol: float = 1e-5
    smooth_window: int = 3
    keep_history: bool = True
    name: str = "tracker"


# ─── Запись одного шага ───────────────────────────────────────────────────────

@dataclass
class StepRecord:
    """Одна запись в истории трекера."""
    step: int
    value: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"StepRecord(step={self.step}, value={self.value:.6f})"


# ─── Трекер ───────────────────────────────────────────────────────────────────

@dataclass
class IterTracker:
    """Трекер числовых значений по итерациям."""
    records: List[StepRecord] = field(default_factory=list)
    config: TrackerConfig = field(default_factory=TrackerConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"IterTracker(n={len(self.records)}, name={self.config.name!r})"


def create_iter_tracker(config: Optional[TrackerConfig] = None, **meta: Any) -> IterTracker:
    """Создать новый пустой трекер.

    Parameters
    ----------
    config:
        Конфигурация трекера. Если None — используются значения по умолчанию.
    **meta:
        Произвольные метаданные, записываемые в трекер.

    Returns
    -------
    IterTracker
    """
    if config is None:
        config = TrackerConfig()
    return IterTracker(config=config, metadata=dict(meta))


def record_step(
    tracker: IterTracker,
    step: int,
    value: float,
    **meta: Any,
) -> IterTracker:
    """Добавить запись шага в трекер.

    Parameters
    ----------
    tracker:
        Трекер для обновления.
    step:
        Номер итерации/шага.
    value:
        Числовое значение (например, потеря, оценка, RMSE).
    **meta:
        Дополнительные поля для записи.

    Returns
    -------
    tracker (для удобства цепочки вызовов)
    """
    record = StepRecord(step=int(step), value=float(value), meta=dict(meta))
    if tracker.config.keep_history:
        tracker.records.append(record)
    else:
        # Сохраняем только последнюю запись
        tracker.records = [record]
    return tracker


# ─── Извлечение значений ──────────────────────────────────────────────────────

def get_values(tracker: IterTracker) -> np.ndarray:
    """Вернуть все значения трекера как массив float64.

    Parameters
    ----------
    tracker:
        Трекер.

    Returns
    -------
    np.ndarray shape (n,)
    """
    return np.array([r.value for r in tracker.records], dtype=np.float64)


def get_steps(tracker: IterTracker) -> np.ndarray:
    """Вернуть все номера шагов трекера как массив int64.

    Parameters
    ----------
    tracker:
        Трекер.

    Returns
    -------
    np.ndarray shape (n,)
    """
    return np.array([r.step for r in tracker.records], dtype=np.int64)


def get_best_record(tracker: IterTracker) -> Optional[StepRecord]:
    """Вернуть запись с максимальным значением.

    Parameters
    ----------
    tracker:
        Трекер.

    Returns
    -------
    StepRecord | None
    """
    if not tracker.records:
        return None
    return max(tracker.records, key=lambda r: r.value)


def get_worst_record(tracker: IterTracker) -> Optional[StepRecord]:
    """Вернуть запись с минимальным значением.

    Parameters
    ----------
    tracker:
        Трекер.

    Returns
    -------
    StepRecord | None
    """
    if not tracker.records:
        return None
    return min(tracker.records, key=lambda r: r.value)


# ─── Анализ конвергенции ──────────────────────────────────────────────────────

def compute_delta(tracker: IterTracker, lag: int = 1) -> np.ndarray:
    """Вычислить разности значений между соседними шагами.

    Parameters
    ----------
    tracker:
        Трекер.
    lag:
        Расстояние между шагами для разности. Должно быть >= 1.

    Returns
    -------
    np.ndarray shape (max(0, n - lag),)

    Raises
    ------
    ValueError
        Если lag < 1.
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    vals = get_values(tracker)
    if len(vals) <= lag:
        return np.array([], dtype=np.float64)
    return vals[lag:] - vals[:-lag]


def is_improving(tracker: IterTracker, window: int = 3, tol: float = 1e-6) -> bool:
    """Проверить, улучшается ли значение в последних `window` шагах.

    Parameters
    ----------
    tracker:
        Трекер.
    window:
        Количество последних шагов для анализа.
    tol:
        Минимальное улучшение, считающееся значимым.

    Returns
    -------
    bool
    """
    vals = get_values(tracker)
    if len(vals) < window:
        return False
    recent = vals[-window:]
    return float(recent[-1] - recent[0]) > tol


def find_plateau_start(
    tracker: IterTracker,
    window: int = 5,
    tol: float = 1e-5,
) -> Optional[int]:
    """Найти шаг, с которого начинается плато (значения стабилизировались).

    Parameters
    ----------
    tracker:
        Трекер.
    window:
        Ширина скользящего окна для проверки стабилизации.
    tol:
        Максимальный диапазон значений в окне, при котором считается плато.

    Returns
    -------
    Номер записи (индекс), с которого началось плато, или None.

    Raises
    ------
    ValueError
        Если window < 2.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    vals = get_values(tracker)
    if len(vals) < window:
        return None
    for i in range(len(vals) - window + 1):
        chunk = vals[i: i + window]
        if float(np.max(chunk) - np.min(chunk)) <= tol:
            return tracker.records[i].step
    return None


# ─── Сглаживание ──────────────────────────────────────────────────────────────

def smooth_values(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Применить скользящее среднее к массиву значений.

    Parameters
    ----------
    values:
        Входной одномерный массив.
    window:
        Ширина окна. Должна быть >= 1.

    Returns
    -------
    np.ndarray той же длины, dtype=float64.

    Raises
    ------
    ValueError
        Если window < 1.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return v.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(v, kernel, mode="same")


# ─── Статистика ───────────────────────────────────────────────────────────────

def tracker_stats(tracker: IterTracker) -> Dict[str, Any]:
    """Сводная статистика по трекеру.

    Parameters
    ----------
    tracker:
        Трекер.

    Returns
    -------
    Словарь с полями:
        n, mean, std, min, max, best_step, worst_step,
        first_step, last_step
    """
    vals = get_values(tracker)
    if len(vals) == 0:
        return {"n": 0}
    best = get_best_record(tracker)
    worst = get_worst_record(tracker)
    return {
        "n": len(vals),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "best_step": best.step if best else None,
        "worst_step": worst.step if worst else None,
        "first_step": tracker.records[0].step,
        "last_step": tracker.records[-1].step,
    }


# ─── Сравнение трекеров ───────────────────────────────────────────────────────

def compare_trackers(
    a: IterTracker,
    b: IterTracker,
) -> Dict[str, Any]:
    """Сравнить два трекера по ключевым метрикам.

    Parameters
    ----------
    a, b:
        Трекеры для сравнения.

    Returns
    -------
    Словарь с полями:
        best_a, best_b, winner ("a", "b" или "tie"),
        delta_best (best_a - best_b), delta_mean
    """
    sa = tracker_stats(a)
    sb = tracker_stats(b)
    if sa.get("n", 0) == 0 and sb.get("n", 0) == 0:
        return {"best_a": None, "best_b": None, "winner": "tie", "delta_best": 0.0, "delta_mean": 0.0}
    best_a = sa.get("max", float("-inf"))
    best_b = sb.get("max", float("-inf"))
    delta_best = float(best_a - best_b)
    mean_a = sa.get("mean", 0.0)
    mean_b = sb.get("mean", 0.0)
    delta_mean = float(mean_a - mean_b)
    if delta_best > 0:
        winner = "a"
    elif delta_best < 0:
        winner = "b"
    else:
        winner = "tie"
    return {
        "best_a": best_a,
        "best_b": best_b,
        "winner": winner,
        "delta_best": delta_best,
        "delta_mean": delta_mean,
    }


# ─── Слияние трекеров ─────────────────────────────────────────────────────────

def merge_trackers(trackers: Sequence[IterTracker]) -> IterTracker:
    """Объединить несколько трекеров в один (конкатенация записей).

    Parameters
    ----------
    trackers:
        Последовательность трекеров.

    Returns
    -------
    Новый IterTracker со всеми записями в порядке добавления.
    """
    merged = create_iter_tracker()
    for t in trackers:
        merged.records.extend(t.records)
    return merged


def window_stats(
    tracker: IterTracker,
    window: int = 5,
) -> List[Dict[str, float]]:
    """Скользящая статистика по окну записей.

    Parameters
    ----------
    tracker:
        Трекер.
    window:
        Размер окна. Должен быть >= 1.

    Returns
    -------
    Список словарей {"mean", "std", "min", "max"} для каждого окна.

    Raises
    ------
    ValueError
        Если window < 1.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    vals = get_values(tracker)
    result: List[Dict[str, float]] = []
    for i in range(len(vals) - window + 1):
        chunk = vals[i: i + window]
        result.append({
            "mean": float(np.mean(chunk)),
            "std": float(np.std(chunk)),
            "min": float(np.min(chunk)),
            "max": float(np.max(chunk)),
        })
    return result


def top_k_records(tracker: IterTracker, k: int) -> List[StepRecord]:
    """Вернуть top-k записей по убыванию значения.

    Parameters
    ----------
    tracker:
        Трекер.
    k:
        Количество лучших записей.

    Returns
    -------
    Список из не более k записей, отсортированных по убыванию value.
    """
    sorted_records = sorted(tracker.records, key=lambda r: r.value, reverse=True)
    return sorted_records[:k]
