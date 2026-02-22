"""Температурные расписания для алгоритма имитации отжига.

Модуль реализует стратегии снижения температуры, применяемые
в алгоритме SA для восстановления пазлов: линейную, геометрическую,
экспоненциальную, косинусную и ступенчатую.

Классы:
    ScheduleConfig    — параметры расписания
    TemperatureRecord — температура на одном шаге

Функции:
    linear_schedule      — линейное убывание температуры
    geometric_schedule   — геометрическое убывание (умножение на α)
    exponential_schedule — экспоненциальное убывание (e^{-kt})
    cosine_schedule      — косинусный отжиг
    stepped_schedule     — ступенчатое убывание
    get_temperature      — температура на произвольном шаге
    estimate_steps       — число шагов до достижения целевой температуры
    batch_temperatures   — температуры сразу для массива шагов
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ─── ScheduleConfig ───────────────────────────────────────────────────────────

@dataclass
class ScheduleConfig:
    """Параметры температурного расписания.

    Атрибуты:
        t_start:    Начальная температура (> 0).
        t_end:      Конечная температура (> 0, < t_start).
        n_steps:    Число шагов (>= 1).
        kind:       Тип расписания: 'linear' | 'geometric' | 'exponential'
                    | 'cosine' | 'stepped'.
        step_size:  Размер ступени для 'stepped' (>= 1).
    """

    t_start: float = 1.0
    t_end: float = 1e-3
    n_steps: int = 1000
    kind: str = "geometric"
    step_size: int = 1

    def __post_init__(self) -> None:
        if self.t_start <= 0.0:
            raise ValueError(
                f"t_start должен быть > 0, получено {self.t_start}"
            )
        if self.t_end <= 0.0:
            raise ValueError(
                f"t_end должен быть > 0, получено {self.t_end}"
            )
        if self.t_end >= self.t_start:
            raise ValueError(
                f"t_end ({self.t_end}) должен быть < t_start ({self.t_start})"
            )
        if self.n_steps < 1:
            raise ValueError(
                f"n_steps должен быть >= 1, получено {self.n_steps}"
            )
        if self.kind not in ("linear", "geometric", "exponential", "cosine", "stepped"):
            raise ValueError(
                f"kind должен быть одним из "
                f"'linear', 'geometric', 'exponential', 'cosine', 'stepped', "
                f"получено '{self.kind}'"
            )
        if self.step_size < 1:
            raise ValueError(
                f"step_size должен быть >= 1, получено {self.step_size}"
            )

    @property
    def cooling_rate(self) -> float:
        """Коэффициент охлаждения α для геометрического расписания."""
        if self.n_steps == 1:
            return self.t_end / self.t_start
        return (self.t_end / self.t_start) ** (1.0 / (self.n_steps - 1))


# ─── TemperatureRecord ────────────────────────────────────────────────────────

@dataclass
class TemperatureRecord:
    """Температура на одном шаге расписания.

    Атрибуты:
        step:        Номер шага (0-based).
        temperature: Значение температуры (> 0).
        progress:    Прогресс ∈ [0, 1].
    """

    step: int
    temperature: float
    progress: float

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError(
                f"step должен быть >= 0, получено {self.step}"
            )
        if self.temperature <= 0.0:
            raise ValueError(
                f"temperature должна быть > 0, получено {self.temperature}"
            )
        if not (0.0 <= self.progress <= 1.0):
            raise ValueError(
                f"progress должен быть в [0, 1], получено {self.progress}"
            )


# ─── linear_schedule ──────────────────────────────────────────────────────────

def linear_schedule(cfg: ScheduleConfig) -> List[TemperatureRecord]:
    """Линейное убывание температуры.

    T(k) = t_start - (t_start - t_end) * k / (n_steps - 1)

    Аргументы:
        cfg: Параметры расписания.

    Возвращает:
        Список TemperatureRecord длиной n_steps.
    """
    result = []
    n = cfg.n_steps
    for k in range(n):
        progress = k / max(n - 1, 1)
        temp = cfg.t_start - (cfg.t_start - cfg.t_end) * progress
        temp = max(temp, cfg.t_end)
        result.append(TemperatureRecord(step=k, temperature=temp, progress=progress))
    return result


# ─── geometric_schedule ───────────────────────────────────────────────────────

def geometric_schedule(cfg: ScheduleConfig) -> List[TemperatureRecord]:
    """Геометрическое убывание температуры.

    T(k) = t_start * α^k,  α = cooling_rate

    Аргументы:
        cfg: Параметры расписания.

    Возвращает:
        Список TemperatureRecord длиной n_steps.
    """
    alpha = cfg.cooling_rate
    result = []
    n = cfg.n_steps
    for k in range(n):
        progress = k / max(n - 1, 1)
        temp = max(cfg.t_start * (alpha ** k), cfg.t_end)
        result.append(TemperatureRecord(step=k, temperature=temp, progress=progress))
    return result


# ─── exponential_schedule ─────────────────────────────────────────────────────

def exponential_schedule(cfg: ScheduleConfig) -> List[TemperatureRecord]:
    """Экспоненциальное убывание температуры.

    T(k) = t_start * exp(-k * λ),  λ = log(t_start/t_end) / (n_steps - 1)

    Аргументы:
        cfg: Параметры расписания.

    Возвращает:
        Список TemperatureRecord длиной n_steps.
    """
    n = cfg.n_steps
    if n == 1:
        lam = 0.0
    else:
        lam = math.log(cfg.t_start / cfg.t_end) / (n - 1)
    result = []
    for k in range(n):
        progress = k / max(n - 1, 1)
        temp = max(cfg.t_start * math.exp(-lam * k), cfg.t_end)
        result.append(TemperatureRecord(step=k, temperature=temp, progress=progress))
    return result


# ─── cosine_schedule ──────────────────────────────────────────────────────────

def cosine_schedule(cfg: ScheduleConfig) -> List[TemperatureRecord]:
    """Косинусный отжиг температуры.

    T(k) = t_end + 0.5 * (t_start - t_end) * (1 + cos(π * k / (n_steps - 1)))

    Аргументы:
        cfg: Параметры расписания.

    Возвращает:
        Список TemperatureRecord длиной n_steps.
    """
    n = cfg.n_steps
    result = []
    for k in range(n):
        progress = k / max(n - 1, 1)
        temp = cfg.t_end + 0.5 * (cfg.t_start - cfg.t_end) * (
            1.0 + math.cos(math.pi * progress)
        )
        temp = max(temp, cfg.t_end)
        result.append(TemperatureRecord(step=k, temperature=temp, progress=progress))
    return result


# ─── stepped_schedule ─────────────────────────────────────────────────────────

def stepped_schedule(cfg: ScheduleConfig) -> List[TemperatureRecord]:
    """Ступенчатое убывание температуры.

    Внутри каждой ступени температура постоянна; между ступенями —
    линейный переход.

    Аргументы:
        cfg: Параметры расписания (step_size задаёт ширину ступени).

    Возвращает:
        Список TemperatureRecord длиной n_steps.
    """
    n = cfg.n_steps
    s = cfg.step_size
    n_plateaus = max(1, (n + s - 1) // s)

    result = []
    for k in range(n):
        progress = k / max(n - 1, 1)
        plateau = k // s
        plateau_progress = plateau / max(n_plateaus - 1, 1)
        plateau_progress = min(plateau_progress, 1.0)
        temp = cfg.t_start - (cfg.t_start - cfg.t_end) * plateau_progress
        temp = max(temp, cfg.t_end)
        result.append(TemperatureRecord(step=k, temperature=temp, progress=progress))
    return result


# ─── get_temperature ──────────────────────────────────────────────────────────

def get_temperature(step: int, cfg: ScheduleConfig) -> float:
    """Вычислить температуру на произвольном шаге.

    Аргументы:
        step: Номер шага (0 <= step < n_steps).
        cfg:  Параметры расписания.

    Возвращает:
        Температура (float > 0).

    Исключения:
        ValueError: Если step выходит за пределы [0, n_steps).
    """
    if step < 0 or step >= cfg.n_steps:
        raise ValueError(
            f"step должен быть в [0, {cfg.n_steps}), получено {step}"
        )

    dispatch = {
        "linear": linear_schedule,
        "geometric": geometric_schedule,
        "exponential": exponential_schedule,
        "cosine": cosine_schedule,
        "stepped": stepped_schedule,
    }
    records = dispatch[cfg.kind](cfg)
    return records[step].temperature


# ─── estimate_steps ───────────────────────────────────────────────────────────

def estimate_steps(
    t_start: float,
    t_target: float,
    alpha: float,
) -> int:
    """Оценить число шагов геометрического расписания до целевой температуры.

    T(k) = t_start * alpha^k >= t_target  →  k = ceil(log(t_target/t_start) / log(alpha))

    Аргументы:
        t_start:  Начальная температура (> 0).
        t_target: Целевая температура (0 < t_target < t_start).
        alpha:    Коэффициент охлаждения (0 < alpha < 1).

    Возвращает:
        Минимальное число шагов (int >= 1).

    Исключения:
        ValueError: При нарушении ограничений на параметры.
    """
    if t_start <= 0.0:
        raise ValueError(f"t_start должен быть > 0, получено {t_start}")
    if t_target <= 0.0 or t_target >= t_start:
        raise ValueError(
            f"t_target должен быть в (0, t_start), получено {t_target}"
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha должен быть в (0, 1), получено {alpha}")

    k = math.log(t_target / t_start) / math.log(alpha)
    return max(1, int(math.ceil(k)))


# ─── batch_temperatures ───────────────────────────────────────────────────────

def batch_temperatures(
    steps: np.ndarray,
    cfg: ScheduleConfig,
) -> np.ndarray:
    """Вычислить температуры для массива шагов.

    Аргументы:
        steps: 1-D массив шагов (int, каждый в [0, n_steps)).
        cfg:   Параметры расписания.

    Возвращает:
        1-D float64 массив температур той же длины.

    Исключения:
        ValueError: Если steps не 1-D или содержит выход за пределы.
    """
    steps = np.asarray(steps)
    if steps.ndim != 1:
        raise ValueError(
            f"steps должен быть 1-D, получено ndim={steps.ndim}"
        )
    if steps.size == 0:
        return np.empty(0, dtype=np.float64)

    out_of_bounds = np.any((steps < 0) | (steps >= cfg.n_steps))
    if out_of_bounds:
        raise ValueError(
            f"Все шаги должны быть в [0, {cfg.n_steps})"
        )

    dispatch = {
        "linear": linear_schedule,
        "geometric": geometric_schedule,
        "exponential": exponential_schedule,
        "cosine": cosine_schedule,
        "stepped": stepped_schedule,
    }
    records = dispatch[cfg.kind](cfg)
    temps = np.array([records[int(k)].temperature for k in steps], dtype=np.float64)
    return temps
