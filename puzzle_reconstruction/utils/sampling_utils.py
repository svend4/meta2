"""
Утилиты случайной выборки для оптимизации сборки пазла.

Предоставляет функции равномерной, взвешенной и сеточной выборки позиций,
углов поворота и перестановок фрагментов. Используется в алгоритмах
имитации отжига и случайного поиска при сборке пазла.

Экспортирует:
    SamplingConfig      — параметры случайной выборки
    uniform_sample      — равномерная выборка вещественного числа из [lo, hi]
    sample_angle        — случайный угол поворота из набора допустимых
    sample_position     — случайная позиция (x, y) в прямоугольной области
    sample_positions_grid — позиции на регулярной сетке со случайным шагом
    sample_permutation  — случайная перестановка индексов
    weighted_sample     — выборка индекса по весовому вектору
    acceptance_probability — вероятность принятия шага в SA
    sample_swap_pair    — случайная пара индексов для перестановки
    batch_uniform_sample — пакетная равномерная выборка
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ─── SamplingConfig ───────────────────────────────────────────────────────────

@dataclass
class SamplingConfig:
    """Параметры случайной выборки.

    Атрибуты:
        seed:       Зерно генератора случайных чисел (None → случайное).
        angles_deg: Допустимые углы поворота в градусах.
        grid_step:  Шаг сетки позиций (>= 1).
    """
    seed:       Optional[int]  = None
    angles_deg: List[float]    = field(default_factory=lambda: [0.0, 90.0, 180.0, 270.0])
    grid_step:  int            = 10

    def __post_init__(self) -> None:
        if self.grid_step < 1:
            raise ValueError(
                f"grid_step должен быть >= 1, получено {self.grid_step}"
            )
        if not self.angles_deg:
            raise ValueError("angles_deg не должен быть пустым")

    def make_rng(self) -> np.random.Generator:
        """Создать генератор случайных чисел с заданным зерном."""
        return np.random.default_rng(self.seed)


# ─── uniform_sample ───────────────────────────────────────────────────────────

def uniform_sample(
    lo: float,
    hi: float,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Равномерная выборка вещественного числа из [lo, hi].

    Аргументы:
        lo:  Нижняя граница диапазона.
        hi:  Верхняя граница диапазона (>= lo).
        rng: Генератор случайных чисел (None → np.random.default_rng()).

    Возвращает:
        Случайное число float из [lo, hi].

    Исключения:
        ValueError: Если lo > hi.
    """
    if lo > hi:
        raise ValueError(
            f"lo ({lo}) должен быть <= hi ({hi})"
        )
    if rng is None:
        rng = np.random.default_rng()
    if lo == hi:
        return float(lo)
    return float(rng.uniform(lo, hi))


# ─── sample_angle ─────────────────────────────────────────────────────────────

def sample_angle(
    cfg: Optional[SamplingConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Выбрать случайный угол поворота из допустимого набора.

    Аргументы:
        cfg: Параметры выборки (None → SamplingConfig()).
        rng: Генератор случайных чисел.

    Возвращает:
        Угол в радианах (float).
    """
    if cfg is None:
        cfg = SamplingConfig()
    if rng is None:
        rng = np.random.default_rng()
    deg = float(rng.choice(cfg.angles_deg))
    return math.radians(deg)


# ─── sample_position ──────────────────────────────────────────────────────────

def sample_position(
    width: int,
    height: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Выбрать случайную позицию (x, y) внутри прямоугольной области.

    Аргументы:
        width:  Ширина области (>= 1).
        height: Высота области (>= 1).
        rng:    Генератор случайных чисел.

    Возвращает:
        Кортеж (x, y) с float координатами ∈ [0, width) × [0, height).

    Исключения:
        ValueError: Если width < 1 или height < 1.
    """
    if width < 1:
        raise ValueError(f"width должен быть >= 1, получено {width}")
    if height < 1:
        raise ValueError(f"height должен быть >= 1, получено {height}")
    if rng is None:
        rng = np.random.default_rng()
    x = float(rng.uniform(0.0, float(width)))
    y = float(rng.uniform(0.0, float(height)))
    return (x, y)


# ─── sample_positions_grid ────────────────────────────────────────────────────

def sample_positions_grid(
    width: int,
    height: int,
    cfg: Optional[SamplingConfig] = None,
) -> List[Tuple[int, int]]:
    """Вернуть узлы регулярной сетки в прямоугольной области.

    Аргументы:
        width:  Ширина области (>= 1).
        height: Высота области (>= 1).
        cfg:    Параметры выборки (определяет grid_step).

    Возвращает:
        Список кортежей (x, y) — целые координаты узлов сетки.

    Исключения:
        ValueError: Если width < 1 или height < 1.
    """
    if width < 1:
        raise ValueError(f"width должен быть >= 1, получено {width}")
    if height < 1:
        raise ValueError(f"height должен быть >= 1, получено {height}")
    if cfg is None:
        cfg = SamplingConfig()
    step = cfg.grid_step
    positions: List[Tuple[int, int]] = []
    for y in range(0, height, step):
        for x in range(0, width, step):
            positions.append((x, y))
    return positions


# ─── sample_permutation ───────────────────────────────────────────────────────

def sample_permutation(
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """Случайная перестановка целых чисел [0, n).

    Аргументы:
        n:   Число элементов (>= 1).
        rng: Генератор случайных чисел.

    Возвращает:
        Список длины n с перемешанными индексами.

    Исключения:
        ValueError: Если n < 1.
    """
    if n < 1:
        raise ValueError(f"n должен быть >= 1, получено {n}")
    if rng is None:
        rng = np.random.default_rng()
    perm = list(range(n))
    rng.shuffle(perm)
    return perm


# ─── weighted_sample ──────────────────────────────────────────────────────────

def weighted_sample(
    weights: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Выбрать индекс по весовому вектору (пропорционально весам).

    Аргументы:
        weights: 1D массив неотрицательных весов (не пустой).
        rng:     Генератор случайных чисел.

    Возвращает:
        Целочисленный индекс (int).

    Исключения:
        ValueError: Если weights пуст, содержит отрицательные значения
                    или сумма равна 0.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("weights должен быть непустым 1D массивом")
    if np.any(w < 0.0):
        raise ValueError("weights не должен содержать отрицательных значений")
    w_sum = float(w.sum())
    if w_sum <= 0.0:
        raise ValueError("Сумма весов должна быть > 0")
    if rng is None:
        rng = np.random.default_rng()
    probs = w / w_sum
    return int(rng.choice(len(w), p=probs))


# ─── acceptance_probability ───────────────────────────────────────────────────

def acceptance_probability(
    delta: float,
    temperature: float,
) -> float:
    """Вычислить вероятность принятия худшего шага в алгоритме SA.

    Стандартная формула: P = exp(-delta / T).
    Если delta <= 0 (улучшение), возвращает 1.0.

    Аргументы:
        delta:       Разность оценок (новая − текущая).
        temperature: Текущая температура SA (> 0).

    Возвращает:
        Вероятность ∈ [0, 1].

    Исключения:
        ValueError: Если temperature <= 0.
    """
    if temperature <= 0.0:
        raise ValueError(
            f"temperature должна быть > 0, получено {temperature}"
        )
    if delta <= 0.0:
        return 1.0
    prob = math.exp(-delta / temperature)
    return float(min(1.0, prob))


# ─── sample_swap_pair ─────────────────────────────────────────────────────────

def sample_swap_pair(
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, int]:
    """Выбрать случайную пару различных индексов из [0, n).

    Аргументы:
        n:   Число элементов (>= 2).
        rng: Генератор случайных чисел.

    Возвращает:
        Кортеж (i, j), i != j, оба ∈ [0, n).

    Исключения:
        ValueError: Если n < 2.
    """
    if n < 2:
        raise ValueError(f"n должен быть >= 2, получено {n}")
    if rng is None:
        rng = np.random.default_rng()
    i = int(rng.integers(0, n))
    j = int(rng.integers(0, n - 1))
    if j >= i:
        j += 1
    return (i, j)


# ─── batch_uniform_sample ─────────────────────────────────────────────────────

def batch_uniform_sample(
    lo: float,
    hi: float,
    size: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Пакетная равномерная выборка float-чисел из [lo, hi].

    Аргументы:
        lo:   Нижняя граница.
        hi:   Верхняя граница (>= lo).
        size: Число выборок (>= 1).
        rng:  Генератор случайных чисел.

    Возвращает:
        float64 массив длины size.

    Исключения:
        ValueError: Если lo > hi или size < 1.
    """
    if lo > hi:
        raise ValueError(f"lo ({lo}) должен быть <= hi ({hi})")
    if size < 1:
        raise ValueError(f"size должен быть >= 1, получено {size}")
    if rng is None:
        rng = np.random.default_rng()
    if lo == hi:
        return np.full(size, lo, dtype=np.float64)
    return rng.uniform(lo, hi, size=size).astype(np.float64)
