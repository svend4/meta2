"""Отслеживание и агрегация метрик во время работы алгоритмов.

Модуль предоставляет классы и функции для регистрации числовых
метрик, вычисления статистик и экспорта истории в удобном формате.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── MetricRecord ─────────────────────────────────────────────────────────────

@dataclass
class MetricRecord:
    """Одна запись метрики.

    Атрибуты:
        name:  Имя метрики (непустое).
        value: Числовое значение.
        step:  Шаг/итерация (>= 0).
    """

    name: str
    value: float
    step: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name не должен быть пустым")
        if self.step < 0:
            raise ValueError(
                f"step должен быть >= 0, получено {self.step}"
            )


# ─── MetricStats ──────────────────────────────────────────────────────────────

@dataclass
class MetricStats:
    """Статистика по одной метрике.

    Атрибуты:
        name:    Имя метрики.
        count:   Число записей (>= 0).
        mean:    Среднее значение.
        std:     Стандартное отклонение (>= 0).
        minimum: Минимальное значение.
        maximum: Максимальное значение.
        last:    Последнее записанное значение.
    """

    name: str
    count: int
    mean: float
    std: float
    minimum: float
    maximum: float
    last: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name не должен быть пустым")
        if self.count < 0:
            raise ValueError(
                f"count должен быть >= 0, получено {self.count}"
            )
        if self.std < 0.0:
            raise ValueError(
                f"std должен быть >= 0, получено {self.std}"
            )

    @property
    def range(self) -> float:
        """Размах (maximum - minimum)."""
        return self.maximum - self.minimum


# ─── TrackerConfig ────────────────────────────────────────────────────────────

@dataclass
class TrackerConfig:
    """Параметры трекера метрик.

    Атрибуты:
        max_history: Максимальная длина истории на метрику (0 = без ограничений).
        namespace:   Пространство имён трекера (непустое).
    """

    max_history: int = 0
    namespace: str = "default"

    def __post_init__(self) -> None:
        if self.max_history < 0:
            raise ValueError(
                f"max_history должен быть >= 0, получено {self.max_history}"
            )
        if not self.namespace:
            raise ValueError("namespace не должен быть пустым")


# ─── MetricTracker ────────────────────────────────────────────────────────────

class MetricTracker:
    """Трекер числовых метрик с историей и вычислением статистик.

    Аргументы:
        cfg: Параметры (None → TrackerConfig()).
    """

    def __init__(self, cfg: Optional[TrackerConfig] = None) -> None:
        if cfg is None:
            cfg = TrackerConfig()
        self._cfg = cfg
        self._history: Dict[str, List[MetricRecord]] = {}

    # ── запись ────────────────────────────────────────────────────────────────

    def record(self, name: str, value: float, step: int = 0) -> None:
        """Записать значение метрики.

        Аргументы:
            name:  Имя метрики (непустое).
            value: Числовое значение.
            step:  Шаг (>= 0).

        Исключения:
            ValueError: Если name пустое или step < 0.
        """
        rec = MetricRecord(name=name, value=value, step=step)
        if name not in self._history:
            self._history[name] = []
        self._history[name].append(rec)
        # Ограничение истории
        if self._cfg.max_history > 0:
            if len(self._history[name]) > self._cfg.max_history:
                self._history[name] = self._history[name][-self._cfg.max_history:]

    def record_dict(self, metrics: Dict[str, float], step: int = 0) -> None:
        """Записать несколько метрик одновременно.

        Аргументы:
            metrics: Словарь {name: value}.
            step:    Шаг (>= 0).
        """
        for name, value in metrics.items():
            self.record(name, value, step)

    # ── чтение ────────────────────────────────────────────────────────────────

    def history(self, name: str) -> List[MetricRecord]:
        """Вернуть историю записей для метрики.

        Аргументы:
            name: Имя метрики.

        Возвращает:
            Список MetricRecord (пустой если метрика не найдена).
        """
        return list(self._history.get(name, []))

    def last_value(self, name: str) -> Optional[float]:
        """Последнее значение метрики (None если нет записей)."""
        records = self._history.get(name, [])
        return records[-1].value if records else None

    def values(self, name: str) -> List[float]:
        """Список всех значений метрики."""
        return [r.value for r in self._history.get(name, [])]

    def metric_names(self) -> List[str]:
        """Список имён всех зарегистрированных метрик."""
        return list(self._history.keys())

    def has_metric(self, name: str) -> bool:
        """Проверить, есть ли записи для метрики."""
        return name in self._history and len(self._history[name]) > 0

    # ── статистика ────────────────────────────────────────────────────────────

    def stats(self, name: str) -> Optional[MetricStats]:
        """Вычислить статистику по метрике.

        Возвращает None если метрика не найдена или записей нет.
        """
        records = self._history.get(name, [])
        if not records:
            return None
        vals = np.array([r.value for r in records], dtype=float)
        return MetricStats(
            name=name,
            count=len(vals),
            mean=float(np.mean(vals)),
            std=float(np.std(vals)),
            minimum=float(np.min(vals)),
            maximum=float(np.max(vals)),
            last=float(vals[-1]),
        )

    def all_stats(self) -> Dict[str, MetricStats]:
        """Статистика для всех метрик."""
        result: Dict[str, MetricStats] = {}
        for name in self._history:
            s = self.stats(name)
            if s is not None:
                result[name] = s
        return result

    # ── управление ────────────────────────────────────────────────────────────

    def clear(self, name: Optional[str] = None) -> int:
        """Очистить историю.

        Аргументы:
            name: Имя метрики (None = очистить все).

        Возвращает:
            Число удалённых записей.
        """
        if name is not None:
            records = self._history.pop(name, [])
            return len(records)
        total = sum(len(v) for v in self._history.values())
        self._history.clear()
        return total

    def size(self, name: Optional[str] = None) -> int:
        """Число записей для метрики (или всего, если name=None)."""
        if name is not None:
            return len(self._history.get(name, []))
        return sum(len(v) for v in self._history.values())

    @property
    def namespace(self) -> str:
        """Пространство имён трекера."""
        return self._cfg.namespace


# ─── make_tracker ─────────────────────────────────────────────────────────────

def make_tracker(
    max_history: int = 0,
    namespace: str = "default",
) -> MetricTracker:
    """Создать MetricTracker с заданными параметрами.

    Аргументы:
        max_history: Максимальный размер истории (0 = без ограничений).
        namespace:   Пространство имён.

    Возвращает:
        MetricTracker.
    """
    return MetricTracker(TrackerConfig(max_history=max_history, namespace=namespace))


# ─── merge_trackers ───────────────────────────────────────────────────────────

def merge_trackers(target: MetricTracker, source: MetricTracker) -> int:
    """Скопировать все записи из source в target.

    Аргументы:
        target: Целевой трекер.
        source: Исходный трекер.

    Возвращает:
        Число скопированных записей.
    """
    count = 0
    for name, records in source._history.items():
        for rec in records:
            target.record(name, rec.value, rec.step)
            count += 1
    return count


# ─── compute_moving_average ───────────────────────────────────────────────────

def compute_moving_average(
    tracker: MetricTracker,
    name: str,
    window: int = 5,
) -> List[float]:
    """Вычислить скользящее среднее для метрики.

    Аргументы:
        tracker: MetricTracker.
        name:    Имя метрики.
        window:  Ширина окна (>= 1).

    Возвращает:
        Список скользящих средних (той же длины, что и история).

    Исключения:
        ValueError: Если window < 1.
    """
    if window < 1:
        raise ValueError(f"window должен быть >= 1, получено {window}")
    vals = tracker.values(name)
    if not vals:
        return []
    result: List[float] = []
    arr = np.array(vals, dtype=float)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(arr[start:i + 1])))
    return result


# ─── export_metrics ───────────────────────────────────────────────────────────

def export_metrics(
    tracker: MetricTracker,
) -> Dict[str, List[Tuple[int, float]]]:
    """Экспортировать все метрики в словарь {name: [(step, value), ...]}.

    Аргументы:
        tracker: MetricTracker.

    Возвращает:
        Словарь списков пар (step, value).
    """
    result: Dict[str, List[Tuple[int, float]]] = {}
    for name in tracker.metric_names():
        result[name] = [(r.step, r.value) for r in tracker.history(name)]
    return result
