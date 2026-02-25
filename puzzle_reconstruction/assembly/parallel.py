"""
Параллельный запуск нескольких методов сборки с выбором лучшего результата.

Позволяет запустить любое подмножество из 8 доступных методов, задать
таймаут для каждого метода и автоматически выбрать сборку с наивысшим
total_score.

Дополнительно поддерживает «гонку» (AssemblyRace): методы запускаются
конкурентно, первый успешный результат возвращается немедленно.

Функции:
    run_all_methods    — последовательный запуск с опциональным таймаутом
    run_selected       — выбор подмножества методов по имени
    pick_best          — выбор лучшей Assembly по total_score
    AssemblyRace       — контекст-менеджер для параллельного запуска

Классы:
    MethodResult       — результат одного метода (имя, Assembly, время, ошибка)
    AssemblyRacer      — запускает методы в ThreadPoolExecutor

Константы:
    ALL_METHODS        — полный список имён методов
    DEFAULT_METHODS    — методы, запускаемые по умолчанию
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..models import Assembly, CompatEntry, Fragment

logger = logging.getLogger(__name__)

# ─── Константы ────────────────────────────────────────────────────────────────

ALL_METHODS = [
    "greedy",
    "sa",
    "beam",
    "gamma",
    "genetic",
    "exhaustive",
    "ant_colony",
    "mcts",
]

DEFAULT_METHODS = ["greedy", "sa", "beam", "genetic"]


# ─── MethodResult ─────────────────────────────────────────────────────────────

@dataclass
class MethodResult:
    """
    Результат одного метода сборки.

    Attributes:
        name:     Имя метода ('greedy', 'sa', …).
        assembly: Assembly или None если метод упал/таймаут.
        elapsed:  Время выполнения в секундах.
        error:    Текст ошибки (None если успех).
        timed_out: True если метод превысил таймаут.
    """
    name:      str
    assembly:  Optional[Assembly] = None
    elapsed:   float              = 0.0
    error:     Optional[str]      = None
    timed_out: bool               = False

    @property
    def success(self) -> bool:
        return self.assembly is not None and not self.timed_out

    @property
    def score(self) -> float:
        return self.assembly.total_score if self.assembly else 0.0

    @property
    def method(self) -> str:
        """Псевдоним для name (обратная совместимость)."""
        return self.name

    def __repr__(self) -> str:
        status = "OK" if self.success else ("TIMEOUT" if self.timed_out else "ERR")
        return (f"MethodResult(name={self.name!r}, status={status}, "
                f"score={self.score:.4f}, elapsed={self.elapsed:.2f}s)")


# ─── Основные функции ─────────────────────────────────────────────────────────

def run_all_methods(fragments:    List[Fragment],
                    entries:      List[CompatEntry],
                    methods:      Optional[Sequence[str]] = None,
                    timeout:      float = 60.0,
                    seed:         Optional[int] = None,
                    n_workers:    int = 1,
                    **method_kwargs) -> List[MethodResult]:
    """
    Запускает выбранные методы сборки и собирает результаты.

    Args:
        fragments:     Список Fragment.
        entries:       Список CompatEntry.
        methods:       Список имён методов (None → DEFAULT_METHODS).
        timeout:       Максимальное время на метод в секундах (0 → без ограничения).
        seed:          Random seed для воспроизводимости.
        n_workers:     Число потоков (1 → последовательно, N → параллельно).
        **method_kwargs: Дополнительные параметры методов (n_iterations, …).

    Returns:
        Список MethodResult (один per метод).
    """
    selected = list(methods or DEFAULT_METHODS)
    callers  = _build_callers(fragments, entries, seed, method_kwargs)

    results: List[MethodResult] = []

    if n_workers <= 1:
        for name in selected:
            result = _run_one(name, callers.get(name), timeout)
            results.append(result)
            logger.info("  [%s] score=%.4f elapsed=%.2fs %s",
                        name, result.score, result.elapsed,
                        "TIMEOUT" if result.timed_out else "")
    else:
        results = _run_parallel(selected, callers, timeout, n_workers)

    return results


def run_selected(fragments:  List[Fragment],
                  entries:    List[CompatEntry],
                  methods:    Sequence[str],
                  **kwargs) -> List[MethodResult]:
    """
    Удобная обёртка для запуска явно указанных методов.

    Проверяет, что все имена есть в ALL_METHODS.

    Raises:
        ValueError: Если имя метода не найдено.
    """
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown:
        raise ValueError(f"Неизвестные методы: {unknown}. "
                          f"Доступные: {ALL_METHODS}")
    return run_all_methods(fragments, entries, methods=methods, **kwargs)


def pick_best(results: List[MethodResult]) -> Optional[Assembly]:
    """
    Выбирает Assembly с наивысшим total_score из списка результатов.

    Args:
        results: Список MethodResult.

    Returns:
        Лучшая Assembly или None если все методы упали.
    """
    successful = [r for r in results if r.success]
    if not successful:
        return None
    best = max(successful, key=lambda r: r.score)
    logger.info("Лучший метод: %s (score=%.4f)", best.name, best.score)
    return best.assembly


def pick_best_k(results: List[MethodResult], k: int) -> List[Assembly]:
    """
    Возвращает топ-k Assembly по total_score.

    Args:
        results: Список MethodResult.
        k:       Число лучших сборок.

    Returns:
        Список Assembly длиной ≤ k (меньше если успешных меньше k).
    """
    successful = [r for r in results if r.success]
    successful.sort(key=lambda r: r.score, reverse=True)
    return [r.assembly for r in successful[:k]]


def summary_table(results: List[MethodResult]) -> str:
    """
    Форматирует таблицу результатов в строку (Markdown).

    Example::

        | Method    | Score  | Time(s) | Status |
        |-----------|--------|---------|--------|
        | greedy    | 0.7234 | 0.02    | OK     |
        | sa        | 0.8112 | 4.56    | OK     |
    """
    header = "| Method     | Score  | Time(s) | Status  |"
    sep    = "|------------|--------|---------|---------|"
    lines  = [header, sep]
    for r in sorted(results, key=lambda x: x.score, reverse=True):
        status = "TIMEOUT" if r.timed_out else ("OK" if r.success else "ERROR")
        lines.append(f"| {r.name:<10} | {r.score:.4f} | {r.elapsed:.2f}    | {status:<7} |")
    return "\n".join(lines)


# ─── Параллельный раннер ──────────────────────────────────────────────────────

class AssemblyRacer:
    """
    Запускает методы конкурентно в ThreadPoolExecutor.

    Usage::

        racer = AssemblyRacer(fragments, entries, seed=42)
        results = racer.race(methods=["greedy", "sa", "genetic"],
                              timeout=30.0, first_only=False)
        best = pick_best(results)
    """

    def __init__(self,
                 fragments:  List[Fragment],
                 entries:    List[CompatEntry],
                 seed:       Optional[int] = None,
                 **method_kwargs) -> None:
        self.fragments     = fragments
        self.entries       = entries
        self.seed          = seed
        self.method_kwargs = method_kwargs

    def race(self,
              methods:     Sequence[str],
              timeout:     float = 60.0,
              first_only:  bool  = False,
              max_workers: int   = 4) -> List[MethodResult]:
        """
        Запускает методы конкурентно.

        Args:
            methods:     Список имён методов.
            timeout:     Таймаут на каждый метод.
            first_only:  True → вернуть как только первый метод завершится.
            max_workers: Число потоков.

        Returns:
            Список MethodResult (неполный если first_only=True).
        """
        callers = _build_callers(self.fragments, self.entries,
                                  self.seed, self.method_kwargs)
        results: List[MethodResult] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_name = {
                executor.submit(_run_callable, callers.get(name), timeout): name
                for name in methods
                if name in callers
            }

            for future in as_completed(future_to_name):
                name   = future_to_name[future]
                try:
                    mr = future.result(timeout=timeout + 5)
                except FutureTimeoutError:
                    mr = MethodResult(name=name, timed_out=True)
                except Exception as exc:
                    mr = MethodResult(name=name, error=str(exc))
                mr.name = name
                results.append(mr)
                if first_only and mr.success:
                    # Отменяем оставшиеся
                    for f in future_to_name:
                        f.cancel()
                    break

        return results


# ─── Внутренние функции ───────────────────────────────────────────────────────

def _build_callers(fragments:  List[Fragment],
                    entries:    List[CompatEntry],
                    seed:       Optional[int],
                    kwargs:     dict) -> Dict[str, Callable[[], Assembly]]:
    """Строит словарь {method_name: callable} с замыканиями."""
    from .greedy      import greedy_assembly
    from .annealing   import simulated_annealing
    from .beam_search import beam_search
    from .gamma_optimizer import gamma_optimizer
    from .exhaustive  import exhaustive_assembly
    from .genetic     import genetic_assembly
    from .ant_colony  import ant_colony_assembly
    from .mcts        import mcts_assembly

    n_iter    = kwargs.get("n_iterations", 200)
    n_sim     = kwargs.get("n_simulations", 50)
    beam_w    = kwargs.get("beam_width", 5)

    return {
        "greedy":     lambda: greedy_assembly(fragments, entries),
        "sa":         lambda: simulated_annealing(fragments, entries,
                                                   n_iterations=n_iter, seed=seed),
        "beam":       lambda: beam_search(fragments, entries,
                                           beam_width=beam_w),
        "gamma":      lambda: gamma_optimizer(fragments, entries, seed=seed),
        "genetic":    lambda: genetic_assembly(fragments, entries,
                                               n_generations=max(n_iter // 10, 20),
                                               seed=seed),
        "exhaustive": lambda: exhaustive_assembly(fragments, entries),
        "ant_colony": lambda: ant_colony_assembly(fragments, entries,
                                                   n_iterations=max(n_iter // 5, 20),
                                                   seed=seed),
        "mcts":       lambda: mcts_assembly(fragments, entries,
                                             n_simulations=n_sim, seed=seed),
    }


def _run_one(name:    str,
              caller:  Optional[Callable[[], Assembly]],
              timeout: float) -> MethodResult:
    """Выполняет один метод последовательно с таймаутом."""
    if caller is None:
        return MethodResult(name=name, error=f"Метод {name!r} не найден")

    t0 = time.perf_counter()
    if timeout > 0:
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(caller)
            try:
                asm = future.result(timeout=timeout)
            except FutureTimeoutError:
                elapsed = time.perf_counter() - t0
                return MethodResult(name=name, elapsed=elapsed, timed_out=True)
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                return MethodResult(name=name, elapsed=elapsed, error=str(exc))
    else:
        try:
            asm = caller()
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            return MethodResult(name=name, elapsed=elapsed, error=str(exc))

    elapsed = time.perf_counter() - t0
    return MethodResult(name=name, assembly=asm, elapsed=elapsed)


def _run_callable(caller: Optional[Callable], timeout: float) -> MethodResult:
    """Обёртка для ThreadPoolExecutor."""
    return _run_one("_", caller, timeout=0)  # Таймаут управляется снаружи


def _run_parallel(selected:  List[str],
                   callers:   Dict[str, Callable[[], Assembly]],
                   timeout:   float,
                   n_workers: int) -> List[MethodResult]:
    """Параллельный запуск через ThreadPoolExecutor."""
    results: List[MethodResult] = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_map = {
            executor.submit(callers.get(name)): name
            for name in selected
            if name in callers
        }

        for future in as_completed(future_map, timeout=timeout + 5):
            name = future_map[future]
            t0   = time.perf_counter()
            try:
                asm     = future.result(timeout=timeout)
                elapsed = time.perf_counter() - t0
                results.append(MethodResult(name=name, assembly=asm,
                                             elapsed=elapsed))
            except FutureTimeoutError:
                results.append(MethodResult(name=name, timed_out=True))
            except Exception as exc:
                results.append(MethodResult(name=name, error=str(exc)))

        # Отменяем не завершённые
        for f in future_map:
            f.cancel()

    return results
