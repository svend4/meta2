"""
Отслеживание эволюции оценок в процессе сборки.

Экспортирует:
    ScoreSnapshot       — снимок оценок на одной итерации
    ScoreTracker        — трекер истории оценок по итерациям
    create_tracker      — создание пустого трекера
    record_snapshot     — добавление снимка в трекер
    detect_convergence  — определение момента сходимости
    extract_best_iteration — итерация с наилучшей оценкой
    summarize_tracker   — сводная статистика по всей истории
    smooth_scores       — скользящее среднее оценок
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class ScoreSnapshot:
    """Снимок оценок на одной итерации алгоритма сборки.

    Attributes:
        iteration:   Номер итерации (≥ 0).
        score:       Основная оценка (выше — лучше).
        n_placed:    Количество размещённых фрагментов.
        extra:       Дополнительные метрики (произвольный словарь).
    """
    iteration: int
    score: float
    n_placed: int
    extra: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ScoreSnapshot(iter={self.iteration}, score={self.score:.4f}, "
            f"placed={self.n_placed})"
        )


@dataclass
class ScoreTracker:
    """Хранилище истории оценок.

    Attributes:
        snapshots:   Список снимков, упорядоченных по времени добавления.
        params:      Параметры трекера / дополнительные метаданные.
    """
    snapshots: List[ScoreSnapshot] = field(default_factory=list)
    params: Dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        n = len(self.snapshots)
        best = max((s.score for s in self.snapshots), default=float("nan"))
        return f"ScoreTracker(n_snapshots={n}, best_score={best:.4f})"


# ─── Публичные функции ────────────────────────────────────────────────────────

def create_tracker(**params) -> ScoreTracker:
    """Создать пустой трекер.

    Args:
        **params: Произвольные параметры, сохраняемые в ``tracker.params``.

    Returns:
        Пустой :class:`ScoreTracker`.
    """
    return ScoreTracker(snapshots=[], params=dict(params))


def record_snapshot(
    tracker: ScoreTracker,
    iteration: int,
    score: float,
    n_placed: int,
    **extra: float,
) -> ScoreTracker:
    """Добавить снимок оценок в трекер.

    Args:
        tracker:    Существующий трекер.
        iteration:  Номер итерации.
        score:      Основная оценка.
        n_placed:   Количество размещённых фрагментов.
        **extra:    Дополнительные числовые метрики.

    Returns:
        Тот же объект ``tracker`` с добавленным снимком.
    """
    snap = ScoreSnapshot(
        iteration=iteration,
        score=float(score),
        n_placed=int(n_placed),
        extra={k: float(v) for k, v in extra.items()},
    )
    tracker.snapshots.append(snap)
    return tracker


def detect_convergence(
    tracker: ScoreTracker,
    window: int = 5,
    tol: float = 1e-4,
) -> Optional[int]:
    """Определить итерацию сходимости трекера.

    Сходимость фиксируется, когда размах оценок в скользящем окне
    ``window`` последних итераций не превышает ``tol``.

    Args:
        tracker:  Трекер с историей снимков.
        window:   Размер окна для анализа (≥ 2).
        tol:      Порог изменения оценки.

    Returns:
        Номер итерации первого окна, в котором достигнута сходимость,
        или ``None``, если сходимость не обнаружена.

    Raises:
        ValueError: Если ``window`` < 2.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    scores = [s.score for s in tracker.snapshots]
    if len(scores) < window:
        return None
    for i in range(window - 1, len(scores)):
        chunk = scores[i - window + 1 : i + 1]
        if (max(chunk) - min(chunk)) <= tol:
            return tracker.snapshots[i - window + 1].iteration
    return None


def extract_best_iteration(tracker: ScoreTracker) -> Optional[ScoreSnapshot]:
    """Вернуть снимок с наилучшей оценкой.

    Args:
        tracker: Трекер с историей снимков.

    Returns:
        :class:`ScoreSnapshot` с максимальной ``score``,
        или ``None``, если трекер пуст.
    """
    if not tracker.snapshots:
        return None
    return max(tracker.snapshots, key=lambda s: s.score)


def summarize_tracker(tracker: ScoreTracker) -> Dict[str, object]:
    """Сформировать сводную статистику по всей истории.

    Args:
        tracker: Трекер с историей снимков.

    Returns:
        Словарь с ключами:
        ``n_snapshots``, ``best_score``, ``worst_score``,
        ``mean_score``, ``std_score``, ``first_iteration``,
        ``last_iteration``, ``best_iteration``.
        Если трекер пуст, возвращает ``{"n_snapshots": 0}``.
    """
    snaps = tracker.snapshots
    if not snaps:
        return {"n_snapshots": 0}
    scores = np.array([s.score for s in snaps], dtype=np.float64)
    best_snap = max(snaps, key=lambda s: s.score)
    return {
        "n_snapshots":     len(snaps),
        "best_score":      float(scores.max()),
        "worst_score":     float(scores.min()),
        "mean_score":      float(scores.mean()),
        "std_score":       float(scores.std()),
        "first_iteration": snaps[0].iteration,
        "last_iteration":  snaps[-1].iteration,
        "best_iteration":  best_snap.iteration,
    }


def smooth_scores(
    tracker: ScoreTracker,
    window: int = 3,
) -> np.ndarray:
    """Вычислить скользящее среднее оценок.

    Args:
        tracker: Трекер с историей снимков.
        window:  Размер окна (≥ 1).

    Returns:
        Массив float64 той же длины, что и ``tracker.snapshots``.
        Для пустого трекера — пустой массив.

    Raises:
        ValueError: Если ``window`` < 1.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    scores = np.array([s.score for s in tracker.snapshots], dtype=np.float64)
    if scores.size == 0:
        return scores
    kernel = np.ones(window, dtype=np.float64) / window
    # «same» mode: сохраняет длину; краевые элементы усредняются по неполному окну
    return np.convolve(scores, kernel, mode="same")
