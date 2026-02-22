"""Трекер прогресса пайплайна обработки фрагментов.

Модуль позволяет регистрировать шаги пайплайна, отслеживать их выполнение,
измерять время и формировать итоговые отчёты.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


_STATUSES = {"pending", "running", "done", "failed", "skipped"}


# ─── StepRecord ───────────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    """Запись об одном шаге пайплайна.

    Атрибуты:
        name:       Название шага (непустое).
        status:     'pending' | 'running' | 'done' | 'failed' | 'skipped'.
        started_at: Unix-время начала (None если не стартовал).
        ended_at:   Unix-время завершения (None если не завершён).
        error:      Сообщение об ошибке (None если нет).
        meta:       Произвольные метаданные.
    """

    name: str
    status: str = "pending"
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name не может быть пустым")
        if self.status not in _STATUSES:
            raise ValueError(
                f"status должен быть одним из {_STATUSES}, "
                f"получено '{self.status}'"
            )

    @property
    def elapsed(self) -> Optional[float]:
        """Время выполнения в секундах (None если шаг не завершён)."""
        if self.started_at is None or self.ended_at is None:
            return None
        return self.ended_at - self.started_at

    @property
    def is_complete(self) -> bool:
        """True если шаг завершён (done/failed/skipped)."""
        return self.status in ("done", "failed", "skipped")


# ─── PipelineReport ───────────────────────────────────────────────────────────

@dataclass
class PipelineReport:
    """Итоговый отчёт о выполнении пайплайна.

    Атрибуты:
        total_steps:  Число зарегистрированных шагов.
        done:         Число успешно завершённых.
        failed:       Число упавших.
        skipped:      Число пропущенных.
        pending:      Число ожидающих.
        total_elapsed: Суммарное время всех завершённых шагов (с).
    """

    total_steps: int = 0
    done: int = 0
    failed: int = 0
    skipped: int = 0
    pending: int = 0
    total_elapsed: float = 0.0

    def __post_init__(self) -> None:
        for name, val in (
            ("total_steps", self.total_steps),
            ("done", self.done),
            ("failed", self.failed),
            ("skipped", self.skipped),
            ("pending", self.pending),
        ):
            if val < 0:
                raise ValueError(f"{name} должен быть >= 0, получено {val}")
        if self.total_elapsed < 0.0:
            raise ValueError(
                f"total_elapsed должен быть >= 0, получено {self.total_elapsed}"
            )

    @property
    def success_rate(self) -> float:
        """Доля успешных шагов (float в [0, 1])."""
        completed = self.done + self.failed + self.skipped
        if completed == 0:
            return 0.0
        return self.done / completed


# ─── ProgressTracker ──────────────────────────────────────────────────────────

class ProgressTracker:
    """Трекер прогресса для шагов пайплайна.

    Аргументы:
        name:     Название пайплайна (непустое).
        callback: Функция, вызываемая при каждом изменении статуса.

    Исключения:
        ValueError: Если name пустое.
    """

    def __init__(
        self,
        name: str = "pipeline",
        callback: Optional[Callable[[StepRecord], None]] = None,
    ) -> None:
        if not name.strip():
            raise ValueError("name пайплайна не может быть пустым")
        self.name = name
        self._callback = callback
        self._steps: Dict[str, StepRecord] = {}
        self._order: List[str] = []

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, step_name: str, meta: Optional[Dict] = None) -> None:
        """Зарегистрировать шаг.

        Аргументы:
            step_name: Название шага (непустое, уникальное).
            meta:      Произвольные метаданные.

        Исключения:
            ValueError: Если шаг уже зарегистрирован или имя пустое.
        """
        if not step_name.strip():
            raise ValueError("step_name не может быть пустым")
        if step_name in self._steps:
            raise ValueError(f"Шаг '{step_name}' уже зарегистрирован")
        rec = StepRecord(name=step_name, meta=meta or {})
        self._steps[step_name] = rec
        self._order.append(step_name)

    # ── Status transitions ─────────────────────────────────────────────────────

    def start(self, step_name: str) -> None:
        """Отметить шаг как запущенный.

        Исключения:
            KeyError: Если шаг не зарегистрирован.
        """
        rec = self._get(step_name)
        rec.status = "running"
        rec.started_at = time.time()
        self._notify(rec)

    def done(self, step_name: str) -> None:
        """Отметить шаг как успешно завершённый."""
        rec = self._get(step_name)
        rec.status = "done"
        rec.ended_at = time.time()
        self._notify(rec)

    def fail(self, step_name: str, error: str = "") -> None:
        """Отметить шаг как упавший.

        Аргументы:
            step_name: Название шага.
            error:     Сообщение об ошибке.
        """
        rec = self._get(step_name)
        rec.status = "failed"
        rec.ended_at = time.time()
        rec.error = error
        self._notify(rec)

    def skip(self, step_name: str) -> None:
        """Пропустить шаг."""
        rec = self._get(step_name)
        rec.status = "skipped"
        rec.ended_at = time.time()
        self._notify(rec)

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_step(self, step_name: str) -> StepRecord:
        """Получить запись о шаге.

        Исключения:
            KeyError: Если шаг не зарегистрирован.
        """
        return self._get(step_name)

    def steps(self) -> List[StepRecord]:
        """Список всех шагов в порядке регистрации."""
        return [self._steps[n] for n in self._order]

    def pending_steps(self) -> List[StepRecord]:
        """Список незавершённых шагов."""
        return [r for r in self.steps() if not r.is_complete]

    def failed_steps(self) -> List[StepRecord]:
        """Список упавших шагов."""
        return [r for r in self.steps() if r.status == "failed"]

    def is_done(self) -> bool:
        """True если все зарегистрированные шаги завершены."""
        return all(r.is_complete for r in self._steps.values())

    def progress(self) -> float:
        """Доля завершённых шагов (float в [0, 1])."""
        if not self._steps:
            return 0.0
        completed = sum(1 for r in self._steps.values() if r.is_complete)
        return completed / len(self._steps)

    def report(self) -> PipelineReport:
        """Сформировать итоговый отчёт."""
        recs = list(self._steps.values())
        elapsed = sum(
            r.elapsed for r in recs if r.elapsed is not None
        )
        return PipelineReport(
            total_steps=len(recs),
            done=sum(1 for r in recs if r.status == "done"),
            failed=sum(1 for r in recs if r.status == "failed"),
            skipped=sum(1 for r in recs if r.status == "skipped"),
            pending=sum(1 for r in recs if r.status == "pending"),
            total_elapsed=elapsed,
        )

    def reset(self) -> None:
        """Сбросить все шаги в состояние 'pending'."""
        for rec in self._steps.values():
            rec.status = "pending"
            rec.started_at = None
            rec.ended_at = None
            rec.error = None

    # ── Private ───────────────────────────────────────────────────────────────

    def _get(self, step_name: str) -> StepRecord:
        if step_name not in self._steps:
            raise KeyError(f"Шаг '{step_name}' не зарегистрирован")
        return self._steps[step_name]

    def _notify(self, rec: StepRecord) -> None:
        if self._callback is not None:
            self._callback(rec)


# ─── Convenience functions ────────────────────────────────────────────────────

def make_tracker(
    name: str = "pipeline",
    steps: Optional[List[str]] = None,
    callback: Optional[Callable[[StepRecord], None]] = None,
) -> ProgressTracker:
    """Создать трекер и зарегистрировать шаги.

    Аргументы:
        name:     Название пайплайна.
        steps:    Список имён шагов для регистрации.
        callback: Функция обратного вызова.

    Возвращает:
        ProgressTracker.
    """
    tracker = ProgressTracker(name=name, callback=callback)
    for s in (steps or []):
        tracker.register(s)
    return tracker


def run_step(
    tracker: ProgressTracker,
    step_name: str,
    fn: Callable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Выполнить шаг пайплайна с автоматическим отслеживанием.

    Регистрирует статус 'running' перед вызовом fn и 'done'/'failed' после.

    Аргументы:
        tracker:   ProgressTracker.
        step_name: Название шага.
        fn:        Вызываемая функция.
        *args, **kwargs: Аргументы для fn.

    Возвращает:
        Результат fn(*args, **kwargs).

    Исключения:
        Повторно бросает исключение из fn (после установки статуса 'failed').
    """
    tracker.start(step_name)
    try:
        result = fn(*args, **kwargs)
        tracker.done(step_name)
        return result
    except Exception as exc:
        tracker.fail(step_name, error=str(exc))
        raise


def summarize_tracker(tracker: ProgressTracker) -> Dict[str, Any]:
    """Получить краткую сводку трекера в виде словаря.

    Возвращает:
        Словарь с ключами: name, progress, total, done, failed, skipped,
        pending, elapsed.
    """
    rep = tracker.report()
    return {
        "name": tracker.name,
        "progress": tracker.progress(),
        "total": rep.total_steps,
        "done": rep.done,
        "failed": rep.failed,
        "skipped": rep.skipped,
        "pending": rep.pending,
        "elapsed": rep.total_elapsed,
    }
