"""Запуск многоэтапного пайплайна обработки.

Модуль предоставляет структуры и функции для последовательного
выполнения именованных шагов пайплайна с фиксацией результатов,
ошибок и метрик каждого этапа.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ─── RunnerConfig ─────────────────────────────────────────────────────────────

@dataclass
class RunnerConfig:
    """Параметры исполнителя пайплайна.

    Атрибуты:
        stop_on_error: Остановить пайплайн при первой ошибке.
        measure_time:  Измерять время выполнения каждого шага.
        verbose:       Выводить отладочную информацию.
        max_steps:     Максимальное число шагов (0 = без лимита, >= 0).
    """

    stop_on_error: bool = True
    measure_time: bool = True
    verbose: bool = False
    max_steps: int = 0

    def __post_init__(self) -> None:
        if self.max_steps < 0:
            raise ValueError(
                f"max_steps должен быть >= 0, получено {self.max_steps}"
            )


# ─── StepResult ───────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Результат выполнения одного шага пайплайна.

    Атрибуты:
        name:      Имя шага (непустая строка).
        index:     Порядковый номер (>= 0).
        success:   True если шаг завершился без исключений.
        output:    Результат функции шага (None при ошибке).
        error:     Сообщение об ошибке (None при успехе).
        elapsed_s: Время выполнения в секундах (>= 0).
    """

    name: str
    index: int
    success: bool
    output: Any = None
    error: Optional[str] = None
    elapsed_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name не должен быть пустой строкой")
        if self.index < 0:
            raise ValueError(
                f"index должен быть >= 0, получено {self.index}"
            )
        if self.elapsed_s < 0:
            raise ValueError(
                f"elapsed_s должен быть >= 0, получено {self.elapsed_s}"
            )

    @property
    def is_slow(self) -> bool:
        """True если выполнение заняло более 1 секунды."""
        return self.elapsed_s > 1.0


# ─── PipelineResult ───────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Итоговый результат выполнения пайплайна.

    Атрибуты:
        step_results: Список StepResult в порядке выполнения.
        n_steps:      Общее число шагов (>= 0).
        n_success:    Число успешных шагов (>= 0).
        n_failed:     Число шагов с ошибкой (>= 0).
        total_time_s: Суммарное время выполнения (>= 0).
        aborted:      True если пайплайн прерван досрочно.
    """

    step_results: List[StepResult]
    n_steps: int
    n_success: int
    n_failed: int
    total_time_s: float
    aborted: bool = False

    def __post_init__(self) -> None:
        for name, val in (
            ("n_steps", self.n_steps),
            ("n_success", self.n_success),
            ("n_failed", self.n_failed),
        ):
            if val < 0:
                raise ValueError(f"{name} должен быть >= 0, получено {val}")
        if self.total_time_s < 0:
            raise ValueError(
                f"total_time_s должен быть >= 0, получено {self.total_time_s}"
            )

    @property
    def success_ratio(self) -> float:
        """Доля успешных шагов (0 если n_steps == 0)."""
        if self.n_steps == 0:
            return 0.0
        return float(self.n_success) / float(self.n_steps)

    @property
    def outputs(self) -> Dict[str, Any]:
        """Словарь {имя шага: output} для успешных шагов."""
        return {r.name: r.output for r in self.step_results if r.success}

    @property
    def errors(self) -> Dict[str, str]:
        """Словарь {имя шага: error} для неудачных шагов."""
        return {r.name: r.error for r in self.step_results
                if not r.success and r.error}

    @property
    def slowest_step(self) -> Optional[str]:
        """Имя самого медленного шага или None."""
        if not self.step_results:
            return None
        return max(self.step_results, key=lambda r: r.elapsed_s).name


# ─── PipelineStep ─────────────────────────────────────────────────────────────

@dataclass
class PipelineStep:
    """Описание одного шага пайплайна.

    Атрибуты:
        name: Уникальное имя шага (непустая строка).
        fn:   Функция шага; принимает предыдущий output или начальный input.
    """

    name: str
    fn: Callable[[Any], Any]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name не должен быть пустой строкой")


# ─── make_step ────────────────────────────────────────────────────────────────

def make_step(name: str, fn: Callable[[Any], Any]) -> PipelineStep:
    """Создать PipelineStep.

    Аргументы:
        name: Имя шага.
        fn:   Функция шага.

    Возвращает:
        PipelineStep.
    """
    return PipelineStep(name=name, fn=fn)


# ─── run_pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    steps: List[PipelineStep],
    initial_input: Any = None,
    cfg: Optional[RunnerConfig] = None,
) -> PipelineResult:
    """Выполнить пайплайн последовательно.

    Каждый шаг принимает output предыдущего (или initial_input для первого).
    При ошибке шага передаётся None следующему (если пайплайн не прерван).

    Аргументы:
        steps:         Список PipelineStep.
        initial_input: Начальные данные.
        cfg:           Параметры.

    Возвращает:
        PipelineResult.
    """
    if cfg is None:
        cfg = RunnerConfig()

    step_results: List[StepResult] = []
    current = initial_input
    n_success = 0
    n_failed = 0
    aborted = False
    total_time = 0.0

    effective_steps = steps
    if cfg.max_steps > 0:
        effective_steps = steps[: cfg.max_steps]

    for idx, step in enumerate(effective_steps):
        t_start = time.monotonic() if cfg.measure_time else 0.0
        try:
            output = step.fn(current)
            elapsed = (time.monotonic() - t_start) if cfg.measure_time else 0.0
            sr = StepResult(
                name=step.name,
                index=idx,
                success=True,
                output=output,
                elapsed_s=elapsed,
            )
            current = output
            n_success += 1
        except Exception as exc:
            elapsed = (time.monotonic() - t_start) if cfg.measure_time else 0.0
            sr = StepResult(
                name=step.name,
                index=idx,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_s=elapsed,
            )
            current = None
            n_failed += 1
            step_results.append(sr)
            total_time += elapsed
            if cfg.stop_on_error:
                aborted = True
                break
            continue

        step_results.append(sr)
        total_time += elapsed

    return PipelineResult(
        step_results=step_results,
        n_steps=len(step_results),
        n_success=n_success,
        n_failed=n_failed,
        total_time_s=total_time,
        aborted=aborted,
    )


# ─── get_step_output ──────────────────────────────────────────────────────────

def get_step_output(result: PipelineResult, name: str) -> Any:
    """Получить output конкретного шага по имени.

    Аргументы:
        result: PipelineResult.
        name:   Имя шага.

    Возвращает:
        Output шага или None если шаг не найден или завершился ошибкой.
    """
    for sr in result.step_results:
        if sr.name == name:
            return sr.output if sr.success else None
    return None


# ─── filter_step_results ──────────────────────────────────────────────────────

def filter_step_results(
    result: PipelineResult,
    success_only: bool = True,
) -> List[StepResult]:
    """Отфильтровать результаты шагов.

    Аргументы:
        result:       PipelineResult.
        success_only: True = только успешные, False = только неудачные.

    Возвращает:
        Список StepResult.
    """
    return [sr for sr in result.step_results if sr.success == success_only]


# ─── retry_failed_steps ───────────────────────────────────────────────────────

def retry_failed_steps(
    steps: List[PipelineStep],
    prev_result: PipelineResult,
    initial_input: Any = None,
    cfg: Optional[RunnerConfig] = None,
) -> PipelineResult:
    """Повторно выполнить только неудачные шаги.

    Аргументы:
        steps:         Полный список шагов.
        prev_result:   PipelineResult предыдущего прогона.
        initial_input: Начальный вход.
        cfg:           Параметры.

    Возвращает:
        PipelineResult только для повторно запущенных шагов.
    """
    failed_names = set(prev_result.errors.keys())
    retry = [s for s in steps if s.name in failed_names]
    return run_pipeline(retry, initial_input, cfg)
