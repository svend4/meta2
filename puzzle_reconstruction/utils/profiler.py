"""
Инструментарий для профилирования производительности пайплайна.

Позволяет измерять время и (опционально) потребление памяти каждого шага
пайплайна. Поддерживает как контекстный менеджер, так и декоратор.

Классы:
    StepProfile    — результат одного шага (имя, время, память, n_calls)
    PipelineProfiler — коллектор профилей; start/stop/record + аналитика

Функции:
    profile_function  — профилирует один вызов функции
    timed             — декоратор, записывающий время в глобальный профилировщик
    format_duration   — форматирует миллисекунды в читаемую строку

Декоратор::

    profiler = PipelineProfiler()

    @timed(profiler, name="preprocess")
    def preprocess(img):
        ...

Контекстный менеджер::

    profiler = PipelineProfiler()
    with profiler.step("matching"):
        result = run_matching(frags, entries)
    print(profiler.summary_table())
"""
from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np


# ─── StepProfile ──────────────────────────────────────────────────────────────

@dataclass
class StepProfile:
    """
    Профиль одного именованного шага.

    Attributes:
        name:       Имя шага.
        elapsed_ms: Суммарное время в миллисекундах.
        n_calls:    Число вызовов.
        memory_mb:  Пиковая дельта памяти в МиБ (0.0 если не замерялась).
        children:   Список вложенных профилей.
    """
    name:       str
    elapsed_ms: float          = 0.0
    n_calls:    int            = 0
    memory_mb:  float          = 0.0
    children:   List["StepProfile"] = field(default_factory=list)

    @property
    def elapsed_s(self) -> float:
        return self.elapsed_ms / 1000.0

    @property
    def avg_ms(self) -> float:
        return self.elapsed_ms / self.n_calls if self.n_calls > 0 else 0.0

    def __repr__(self) -> str:
        return (f"StepProfile(name={self.name!r}, "
                f"elapsed={self.elapsed_ms:.2f}ms, "
                f"n_calls={self.n_calls})")


# ─── PipelineProfiler ─────────────────────────────────────────────────────────

class PipelineProfiler:
    """
    Коллектор профилей шагов пайплайна.

    Usage::

        p = PipelineProfiler()
        with p.step("preprocess"):
            ...
        with p.step("matching"):
            ...
        print(p.summary_table())
    """

    def __init__(self, track_memory: bool = False) -> None:
        """
        Args:
            track_memory: Если True — пытается замерить потребление памяти
                          через psutil (если недоступен, молча игнорируется).
        """
        self._profiles:     Dict[str, StepProfile] = {}
        self._active:       Dict[str, float]        = {}
        self.track_memory:  bool                    = track_memory

    # ── Публичный интерфейс ───────────────────────────────────────────────

    def start(self, name: str) -> None:
        """Начинает измерение шага `name`."""
        self._active[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """
        Останавливает шаг `name`.

        Returns:
            Время в мс, затраченное на этот вызов.

        Raises:
            KeyError: Если start() для `name` не вызывался.
        """
        t0 = self._active.pop(name)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.record(name, elapsed_ms)
        return elapsed_ms

    def record(self, name: str, elapsed_ms: float,
                memory_mb: float = 0.0) -> None:
        """
        Накапливает замер для шага `name` (без start/stop).

        Args:
            name:       Имя шага.
            elapsed_ms: Длительность в мс.
            memory_mb:  Потребление памяти в МиБ.
        """
        if name not in self._profiles:
            self._profiles[name] = StepProfile(name=name)
        p = self._profiles[name]
        p.elapsed_ms += elapsed_ms
        p.n_calls    += 1
        p.memory_mb   = max(p.memory_mb, memory_mb)

    @contextmanager
    def step(self, name: str) -> Generator[None, None, None]:
        """
        Контекстный менеджер для измерения шага.

        Usage::

            with profiler.step("matching"):
                do_matching()
        """
        mem_before = self._get_rss_mb()
        self.start(name)
        try:
            yield
        finally:
            elapsed_ms = self.stop(name)
            if self.track_memory:
                mem_after = self._get_rss_mb()
                delta = max(0.0, mem_after - mem_before)
                if name in self._profiles:
                    self._profiles[name].memory_mb = max(
                        self._profiles[name].memory_mb, delta)

    def reset(self) -> None:
        """Сбрасывает все профили."""
        self._profiles.clear()
        self._active.clear()

    # ── Аналитика ─────────────────────────────────────────────────────────

    def profiles(self) -> List[StepProfile]:
        """Список профилей, отсортированных по суммарному времени (убывание)."""
        return sorted(self._profiles.values(),
                       key=lambda p: p.elapsed_ms, reverse=True)

    def total_elapsed_ms(self) -> float:
        """Суммарное время всех шагов (мс)."""
        return sum(p.elapsed_ms for p in self._profiles.values())

    def slowest_step(self) -> Optional[str]:
        """Имя самого медленного шага или None если профилей нет."""
        if not self._profiles:
            return None
        return max(self._profiles, key=lambda k: self._profiles[k].elapsed_ms)

    def fastest_step(self) -> Optional[str]:
        """Имя самого быстрого шага."""
        if not self._profiles:
            return None
        return min(self._profiles, key=lambda k: self._profiles[k].elapsed_ms)

    def get(self, name: str) -> Optional[StepProfile]:
        """Возвращает профиль шага по имени (None если не найден)."""
        return self._profiles.get(name)

    def step_names(self) -> List[str]:
        """Список имён шагов в порядке убывания времени."""
        return [p.name for p in self.profiles()]

    def summary_table(self, top_n: Optional[int] = None) -> str:
        """
        Форматирует таблицу профилей в строку (Markdown).

        Args:
            top_n: Показывать только top_n самых медленных шагов.

        Example::

            | Step        | Calls | Total(ms) | Avg(ms) | Mem(MB) |
            |-------------|-------|-----------|---------|---------|
            | matching    |     1 |   1234.56 |  1234.6 |    0.00 |
        """
        profs = self.profiles()
        if top_n is not None:
            profs = profs[:top_n]

        header  = "| Step              | Calls | Total(ms) | Avg(ms) | Mem(MB) |"
        sep     = "|-------------------|-------|-----------|---------|---------|"
        total   = self.total_elapsed_ms()
        lines   = [header, sep]

        for p in profs:
            pct = (p.elapsed_ms / total * 100) if total > 0 else 0.0
            lines.append(
                f"| {p.name:<17} | {p.n_calls:>5} | "
                f"{p.elapsed_ms:>9.2f} | {p.avg_ms:>7.2f} | "
                f"{p.memory_mb:>7.2f} |  {pct:.1f}%"
            )

        lines.append(sep)
        lines.append(f"| {'TOTAL':<17} | {sum(p.n_calls for p in profs):>5} | "
                      f"{total:>9.2f} | {'':>7} | {'':>7} |")
        return "\n".join(lines)

    def __repr__(self) -> str:
        n     = len(self._profiles)
        total = self.total_elapsed_ms()
        return f"PipelineProfiler(steps={n}, total={total:.2f}ms)"

    # ── Внутренние методы ─────────────────────────────────────────────────

    def _get_rss_mb(self) -> float:
        try:
            import psutil, os
            proc = psutil.Process(os.getpid())
            return proc.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0


# ─── Утилиты ──────────────────────────────────────────────────────────────────

def format_duration(ms: float) -> str:
    """
    Форматирует продолжительность в миллисекундах в читаемую строку.

    Examples::

        format_duration(0.3)    → "0.30ms"
        format_duration(999)    → "999.00ms"
        format_duration(1500)   → "1.50s"
        format_duration(90000)  → "1m 30.00s"
    """
    if ms < 1000:
        return f"{ms:.2f}ms"
    s = ms / 1000.0
    if s < 60:
        return f"{s:.2f}s"
    m = int(s // 60)
    s = s - m * 60
    return f"{m}m {s:.2f}s"


def profile_function(fn:        Callable,
                      *args,
                      profiler:  Optional[PipelineProfiler] = None,
                      step_name: Optional[str] = None,
                      **kwargs) -> Tuple[Any, StepProfile]:
    """
    Профилирует один вызов функции.

    Args:
        fn:        Функция для профилирования.
        *args:     Позиционные аргументы.
        profiler:  Если передан, записывает результат в него.
        step_name: Имя шага (по умолчанию fn.__name__).
        **kwargs:  Именованные аргументы.

    Returns:
        (result, StepProfile) — результат функции и профиль.
    """
    name = step_name or getattr(fn, "__name__", "unknown")
    t0   = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    sp = StepProfile(name=name, elapsed_ms=elapsed_ms, n_calls=1)

    if profiler is not None:
        profiler.record(name, elapsed_ms)

    return result, sp


def timed(profiler: PipelineProfiler,
           name:    Optional[str] = None) -> Callable:
    """
    Декоратор для автоматического профилирования функции.

    Args:
        profiler: PipelineProfiler для записи.
        name:     Имя шага (по умолчанию fn.__name__).

    Usage::

        @timed(my_profiler)
        def my_step(x):
            return x * 2
    """
    def decorator(fn: Callable) -> Callable:
        step_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            profiler.record(step_name, elapsed_ms)
            return result

        return wrapper

    return decorator


def compare_profilers(p1:     PipelineProfiler,
                       p2:     PipelineProfiler,
                       label1: str = "run_1",
                       label2: str = "run_2") -> str:
    """
    Сравнивает два профиля и возвращает таблицу разницы (Markdown).

    Args:
        p1, p2:         Профили для сравнения.
        label1, label2: Метки в заголовке.

    Returns:
        Строка с Markdown-таблицей.
    """
    all_names = sorted(
        set(p1._profiles) | set(p2._profiles),
        key=lambda n: p1._profiles.get(n, StepProfile(n)).elapsed_ms
        + p2._profiles.get(n, StepProfile(n)).elapsed_ms,
        reverse=True,
    )

    header = (f"| Step              | {label1:<10}(ms) | "
               f"{label2:<10}(ms) | Δ(ms)    | Speedup |")
    sep    = "|-------------------|-----------------|-----------------|----------|---------|"
    lines  = [header, sep]

    for name in all_names:
        t1 = p1._profiles[name].elapsed_ms if name in p1._profiles else 0.0
        t2 = p2._profiles[name].elapsed_ms if name in p2._profiles else 0.0
        delta   = t2 - t1
        speedup = (t1 / t2) if t2 > 0 else float("inf")
        lines.append(
            f"| {name:<17} | {t1:>15.2f} | {t2:>15.2f} | "
            f"{delta:>+8.2f} | {speedup:>6.2f}x |"
        )

    return "\n".join(lines)
