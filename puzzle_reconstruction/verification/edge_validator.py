"""Валидация совместимости краёв между соседними фрагментами.

Модуль предоставляет структуры и функции для проверки того,
насколько хорошо края двух фрагментов «подходят» друг к другу
по нескольким критериям: интенсивность, разрыв и согласованность
нормалей к контуру.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── EdgeValidConfig ──────────────────────────────────────────────────────────

@dataclass
class EdgeValidConfig:
    """Параметры валидатора краёв.

    Атрибуты:
        intensity_tol:  Максимально допустимая разница интенсивностей [0, 1].
        gap_tol:        Максимально допустимый пространственный разрыв (px, >= 0).
        normal_tol_deg: Максимально допустимое расхождение нормалей (градусы, >= 0).
        require_all:    Все критерии должны быть выполнены (иначе — хотя бы один).
    """

    intensity_tol: float = 0.15
    gap_tol: float = 2.0
    normal_tol_deg: float = 30.0
    require_all: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.intensity_tol <= 1.0):
            raise ValueError(
                f"intensity_tol должен быть в [0, 1], получено {self.intensity_tol}"
            )
        if self.gap_tol < 0:
            raise ValueError(
                f"gap_tol должен быть >= 0, получено {self.gap_tol}"
            )
        if self.normal_tol_deg < 0:
            raise ValueError(
                f"normal_tol_deg должен быть >= 0, "
                f"получено {self.normal_tol_deg}"
            )


# ─── EdgeCheck ────────────────────────────────────────────────────────────────

@dataclass
class EdgeCheck:
    """Результат одного критерия валидации края.

    Атрибуты:
        name:    Название критерия (непустая строка).
        passed:  True если критерий выполнен.
        value:   Измеренное значение.
        limit:   Пороговое значение.
    """

    name: str
    passed: bool
    value: float
    limit: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name не должен быть пустой строкой")

    @property
    def margin(self) -> float:
        """Запас до порога (положительный = в норме)."""
        return float(self.limit - self.value)


# ─── EdgeValidResult ──────────────────────────────────────────────────────────

@dataclass
class EdgeValidResult:
    """Результат валидации пары краёв.

    Атрибуты:
        pair:    (fragment_id_a, fragment_id_b).
        checks:  Список EdgeCheck.
        valid:   Итоговый вывод о валидности.
    """

    pair: Tuple[int, int]
    checks: List[EdgeCheck]
    valid: bool

    @property
    def fragment_a(self) -> int:
        """Первый фрагмент."""
        return self.pair[0]

    @property
    def fragment_b(self) -> int:
        """Второй фрагмент."""
        return self.pair[1]

    @property
    def n_passed(self) -> int:
        """Число выполненных критериев."""
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_failed(self) -> int:
        """Число невыполненных критериев."""
        return sum(1 for c in self.checks if not c.passed)

    @property
    def check_names(self) -> List[str]:
        """Имена всех критериев."""
        return [c.name for c in self.checks]

    def get_check(self, name: str) -> Optional[EdgeCheck]:
        """Вернуть EdgeCheck по имени или None."""
        for c in self.checks:
            if c.name == name:
                return c
        return None


# ─── _intensity_diff ──────────────────────────────────────────────────────────

def _intensity_diff(profile_a: np.ndarray, profile_b: np.ndarray) -> float:
    """Средняя абсолютная разница нормированных профилей [0, 1]."""
    a = profile_a.astype(float)
    b = profile_b.astype(float)
    # Нормировать в [0, 1]
    for arr in (a, b):
        rng = arr.max() - arr.min()
        if rng > 1e-12:
            arr[:] = (arr - arr.min()) / rng
    return float(np.mean(np.abs(a - b)))


# ─── check_intensity ──────────────────────────────────────────────────────────

def check_intensity(
    profile_a: np.ndarray,
    profile_b: np.ndarray,
    cfg: Optional[EdgeValidConfig] = None,
) -> EdgeCheck:
    """Проверить совместимость по интенсивности краёв.

    Аргументы:
        profile_a: Профиль интенсивности первого края (1D).
        profile_b: Профиль интенсивности второго края (1D).
        cfg:       Параметры.

    Возвращает:
        EdgeCheck.
    """
    if cfg is None:
        cfg = EdgeValidConfig()
    diff = _intensity_diff(profile_a, profile_b)
    return EdgeCheck(
        name="intensity",
        passed=diff <= cfg.intensity_tol,
        value=diff,
        limit=cfg.intensity_tol,
    )


# ─── check_gap ────────────────────────────────────────────────────────────────

def check_gap(
    points_a: np.ndarray,
    points_b: np.ndarray,
    cfg: Optional[EdgeValidConfig] = None,
) -> EdgeCheck:
    """Проверить пространственный разрыв между краями.

    Аргументы:
        points_a: Координаты контура первого края (N×2).
        points_b: Координаты контура второго края (M×2).
        cfg:      Параметры.

    Возвращает:
        EdgeCheck.
    """
    if cfg is None:
        cfg = EdgeValidConfig()

    if len(points_a) == 0 or len(points_b) == 0:
        return EdgeCheck(name="gap", passed=False, value=float("inf"),
                         limit=cfg.gap_tol)

    # Минимальное расстояние между двумя наборами точек (упрощённо)
    a = np.array(points_a, dtype=float)
    b = np.array(points_b, dtype=float)
    dists = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    min_gap = float(dists.min())

    return EdgeCheck(
        name="gap",
        passed=min_gap <= cfg.gap_tol,
        value=min_gap,
        limit=cfg.gap_tol,
    )


# ─── check_normals ────────────────────────────────────────────────────────────

def check_normals(
    normals_a: np.ndarray,
    normals_b: np.ndarray,
    cfg: Optional[EdgeValidConfig] = None,
) -> EdgeCheck:
    """Проверить согласованность нормалей двух краёв.

    Аргументы:
        normals_a: Нормали первого края (N×2), каждая нормирована.
        normals_b: Нормали второго края (M×2).
        cfg:       Параметры.

    Возвращает:
        EdgeCheck.
    """
    if cfg is None:
        cfg = EdgeValidConfig()

    if len(normals_a) == 0 or len(normals_b) == 0:
        return EdgeCheck(name="normals", passed=False,
                         value=float("inf"), limit=cfg.normal_tol_deg)

    n = min(len(normals_a), len(normals_b))
    a = np.array(normals_a[:n], dtype=float)
    b = np.array(normals_b[:n], dtype=float)

    # Нормировать
    norm_a = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    norm_b = np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    a = a / norm_a
    b = b / norm_b

    dots = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    # Нормали соседних краёв должны быть антипараллельны → cos ≈ -1
    angles_deg = np.degrees(np.arccos(np.abs(dots)))  # [0, 90]
    mean_angle = float(np.mean(angles_deg))

    return EdgeCheck(
        name="normals",
        passed=mean_angle <= cfg.normal_tol_deg,
        value=mean_angle,
        limit=cfg.normal_tol_deg,
    )


# ─── validate_edge_pair ───────────────────────────────────────────────────────

def validate_edge_pair(
    fragment_id_a: int,
    fragment_id_b: int,
    intensity_a: np.ndarray,
    intensity_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    normals_a: np.ndarray,
    normals_b: np.ndarray,
    cfg: Optional[EdgeValidConfig] = None,
) -> EdgeValidResult:
    """Выполнить полную валидацию пары краёв.

    Аргументы:
        fragment_id_a/b: Идентификаторы фрагментов.
        intensity_a/b:   Профили интенсивности краёв (1D массивы).
        points_a/b:      Координаты контуров (N×2).
        normals_a/b:     Нормали к контурам (N×2).
        cfg:             Параметры.

    Возвращает:
        EdgeValidResult.
    """
    if cfg is None:
        cfg = EdgeValidConfig()

    checks: List[EdgeCheck] = [
        check_intensity(intensity_a, intensity_b, cfg),
        check_gap(points_a, points_b, cfg),
        check_normals(normals_a, normals_b, cfg),
    ]

    if cfg.require_all:
        valid = all(c.passed for c in checks)
    else:
        valid = any(c.passed for c in checks)

    return EdgeValidResult(
        pair=(fragment_id_a, fragment_id_b),
        checks=checks,
        valid=valid,
    )


# ─── summarise_validations ────────────────────────────────────────────────────

def summarise_validations(
    results: List[EdgeValidResult],
) -> Dict[str, float]:
    """Подсчитать статистику по набору результатов.

    Аргументы:
        results: Список EdgeValidResult.

    Возвращает:
        Словарь {"valid_ratio", "mean_passed_checks", "n_results"}.
    """
    if not results:
        return {"valid_ratio": 0.0, "mean_passed_checks": 0.0, "n_results": 0}

    valid_ratio = sum(1 for r in results if r.valid) / len(results)
    mean_passed = sum(r.n_passed for r in results) / len(results)

    return {
        "valid_ratio": float(valid_ratio),
        "mean_passed_checks": float(mean_passed),
        "n_results": len(results),
    }


# ─── batch_validate_edges ─────────────────────────────────────────────────────

def batch_validate_edges(
    pairs: List[Tuple[int, int]],
    intensity_map: Dict[int, np.ndarray],
    points_map: Dict[int, np.ndarray],
    normals_map: Dict[int, np.ndarray],
    cfg: Optional[EdgeValidConfig] = None,
) -> List[EdgeValidResult]:
    """Валидировать несколько пар краёв.

    Аргументы:
        pairs:         Список пар (fragment_id_a, fragment_id_b).
        intensity_map: {fragment_id: профиль интенсивности}.
        points_map:    {fragment_id: координаты контура (N×2)}.
        normals_map:   {fragment_id: нормали (N×2)}.
        cfg:           Параметры.

    Возвращает:
        Список EdgeValidResult.
    """
    results: List[EdgeValidResult] = []
    empty = np.zeros((0, 2))
    empty_1d = np.zeros(0)

    for a_id, b_id in pairs:
        ia = intensity_map.get(a_id, empty_1d)
        ib = intensity_map.get(b_id, empty_1d)
        pa = points_map.get(a_id, empty)
        pb = points_map.get(b_id, empty)
        na = normals_map.get(a_id, empty)
        nb = normals_map.get(b_id, empty)
        results.append(validate_edge_pair(
            a_id, b_id, ia, ib, pa, pb, na, nb, cfg
        ))
    return results
