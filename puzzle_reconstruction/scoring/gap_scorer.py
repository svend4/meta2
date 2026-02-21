"""Оценка качества зазоров между фрагментами в компоновке.

Модуль анализирует зазоры (расстояния) между соседними фрагментами
и вычисляет интегральную оценку качества их расположения.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── GapConfig ────────────────────────────────────────────────────────────────

@dataclass
class GapConfig:
    """Параметры оценки зазоров.

    Атрибуты:
        target_gap:   Целевой зазор между фрагментами (>= 0).
        tolerance:    Допустимое отклонение от target_gap (>= 0).
        penalty_scale: Масштаб штрафа за отклонение (> 0).
        max_gap:      Максимально допустимый зазор (> target_gap).
    """

    target_gap: float = 5.0
    tolerance: float = 1.0
    penalty_scale: float = 1.0
    max_gap: float = 20.0

    def __post_init__(self) -> None:
        if self.target_gap < 0.0:
            raise ValueError(
                f"target_gap должен быть >= 0, получено {self.target_gap}"
            )
        if self.tolerance < 0.0:
            raise ValueError(
                f"tolerance должна быть >= 0, получено {self.tolerance}"
            )
        if self.penalty_scale <= 0.0:
            raise ValueError(
                f"penalty_scale должен быть > 0, получено {self.penalty_scale}"
            )
        if self.max_gap <= self.target_gap:
            raise ValueError(
                f"max_gap ({self.max_gap}) должен быть > target_gap "
                f"({self.target_gap})"
            )


# ─── GapMeasure ───────────────────────────────────────────────────────────────

@dataclass
class GapMeasure:
    """Измерение зазора между двумя фрагментами.

    Атрибуты:
        id_a:     ID первого фрагмента.
        id_b:     ID второго фрагмента.
        distance: Измеренное расстояние (>= 0).
        score:    Оценка [0, 1] (1 = идеально).
        penalty:  Штраф (>= 0).
    """

    id_a: int
    id_b: int
    distance: float
    score: float
    penalty: float

    def __post_init__(self) -> None:
        if self.distance < 0.0:
            raise ValueError(
                f"distance должно быть >= 0, получено {self.distance}"
            )
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"score должен быть в [0, 1], получено {self.score}"
            )
        if self.penalty < 0.0:
            raise ValueError(
                f"penalty должен быть >= 0, получено {self.penalty}"
            )

    @property
    def pair_key(self) -> Tuple[int, int]:
        """Упорядоченная пара (min, max)."""
        return (min(self.id_a, self.id_b), max(self.id_a, self.id_b))

    @property
    def is_acceptable(self) -> bool:
        """True если score > 0.5."""
        return self.score > 0.5


# ─── GapReport ────────────────────────────────────────────────────────────────

@dataclass
class GapReport:
    """Итоговый отчёт об оценке зазоров.

    Атрибуты:
        measures:     Список GapMeasure.
        mean_score:   Средняя оценка [0, 1].
        total_penalty: Суммарный штраф.
        n_acceptable: Число допустимых пар.
    """

    measures: List[GapMeasure]
    mean_score: float
    total_penalty: float
    n_acceptable: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.mean_score <= 1.0):
            raise ValueError(
                f"mean_score должен быть в [0, 1], получено {self.mean_score}"
            )
        if self.total_penalty < 0.0:
            raise ValueError(
                f"total_penalty должен быть >= 0, получено {self.total_penalty}"
            )

    @property
    def n_pairs(self) -> int:
        """Число измеренных пар."""
        return len(self.measures)

    @property
    def acceptance_rate(self) -> float:
        """Доля допустимых пар [0, 1]."""
        if self.n_pairs == 0:
            return 0.0
        return float(self.n_acceptable / self.n_pairs)

    @property
    def mean_distance(self) -> float:
        """Среднее расстояние."""
        if not self.measures:
            return 0.0
        return float(sum(m.distance for m in self.measures) / len(self.measures))

    def get_measure(self, id_a: int, id_b: int) -> Optional[GapMeasure]:
        """Найти GapMeasure по паре ID или None."""
        key = (min(id_a, id_b), max(id_a, id_b))
        for m in self.measures:
            if m.pair_key == key:
                return m
        return None


# ─── score_gap ────────────────────────────────────────────────────────────────

def score_gap(
    distance: float,
    cfg: Optional[GapConfig] = None,
) -> Tuple[float, float]:
    """Вычислить оценку и штраф для одного зазора.

    Аргументы:
        distance: Измеренный зазор (>= 0).
        cfg:      Параметры.

    Возвращает:
        Кортеж (score, penalty).

    Исключения:
        ValueError: если distance < 0.
    """
    if distance < 0.0:
        raise ValueError(f"distance должно быть >= 0, получено {distance}")

    if cfg is None:
        cfg = GapConfig()

    deviation = abs(distance - cfg.target_gap)

    # В пределах tolerance: score = 1, penalty = 0
    if deviation <= cfg.tolerance:
        return 1.0, 0.0

    # Слишком большой зазор: жёсткий штраф
    if distance > cfg.max_gap:
        return 0.0, float(cfg.penalty_scale * (distance - cfg.max_gap + deviation))

    # Линейный штраф за отклонение от [target - tol, target + tol]
    excess = deviation - cfg.tolerance
    penalty = float(cfg.penalty_scale * excess)
    span = cfg.max_gap - cfg.target_gap
    if span < 1e-12:
        score = 0.0
    else:
        score = float(max(0.0, 1.0 - excess / span))

    return score, penalty


# ─── measure_gap ──────────────────────────────────────────────────────────────

def measure_gap(
    id_a: int,
    id_b: int,
    distance: float,
    cfg: Optional[GapConfig] = None,
) -> GapMeasure:
    """Создать GapMeasure для пары фрагментов.

    Аргументы:
        id_a:     ID первого фрагмента.
        id_b:     ID второго фрагмента.
        distance: Измеренное расстояние.
        cfg:      Параметры.

    Возвращает:
        GapMeasure.
    """
    if cfg is None:
        cfg = GapConfig()
    score, penalty = score_gap(distance, cfg)
    return GapMeasure(id_a=id_a, id_b=id_b, distance=distance,
                      score=score, penalty=penalty)


# ─── build_gap_report ─────────────────────────────────────────────────────────

def build_gap_report(
    distances: Dict[Tuple[int, int], float],
    cfg: Optional[GapConfig] = None,
) -> GapReport:
    """Построить отчёт об оценке зазоров для набора пар.

    Аргументы:
        distances: Словарь {(id_a, id_b): distance}.
        cfg:       Параметры.

    Возвращает:
        GapReport.
    """
    if cfg is None:
        cfg = GapConfig()

    measures: List[GapMeasure] = []
    for (a, b), dist in distances.items():
        measures.append(measure_gap(a, b, dist, cfg))

    if not measures:
        return GapReport(measures=[], mean_score=0.0,
                         total_penalty=0.0, n_acceptable=0)

    mean_score = float(sum(m.score for m in measures) / len(measures))
    total_penalty = float(sum(m.penalty for m in measures))
    n_acceptable = sum(1 for m in measures if m.is_acceptable)

    return GapReport(measures=measures, mean_score=mean_score,
                     total_penalty=total_penalty, n_acceptable=n_acceptable)


# ─── filter_gap_measures ──────────────────────────────────────────────────────

def filter_gap_measures(
    report: GapReport,
    min_score: float = 0.0,
) -> List[GapMeasure]:
    """Вернуть GapMeasure с score >= min_score.

    Аргументы:
        report:    GapReport.
        min_score: Минимальная оценка [0, 1].

    Возвращает:
        Отфильтрованный список GapMeasure.

    Исключения:
        ValueError: если min_score вне [0, 1].
    """
    if not (0.0 <= min_score <= 1.0):
        raise ValueError(
            f"min_score должен быть в [0, 1], получено {min_score}"
        )
    return [m for m in report.measures if m.score >= min_score]


# ─── worst_gap_pairs ──────────────────────────────────────────────────────────

def worst_gap_pairs(
    report: GapReport,
    top_k: int = 5,
) -> List[GapMeasure]:
    """Вернуть top_k пар с наихудшей оценкой (наибольшим штрафом).

    Аргументы:
        report: GapReport.
        top_k:  Число пар (>= 1).

    Возвращает:
        Список GapMeasure, отсортированный по убыванию penalty.

    Исключения:
        ValueError: если top_k < 1.
    """
    if top_k < 1:
        raise ValueError(f"top_k должен быть >= 1, получено {top_k}")
    sorted_m = sorted(report.measures, key=lambda m: m.penalty, reverse=True)
    return sorted_m[:top_k]


# ─── gap_score_matrix ─────────────────────────────────────────────────────────

def gap_score_matrix(
    fragment_ids: List[int],
    distances: Dict[Tuple[int, int], float],
    cfg: Optional[GapConfig] = None,
) -> Dict[Tuple[int, int], float]:
    """Построить матрицу оценок зазоров.

    Аргументы:
        fragment_ids: Список ID фрагментов.
        distances:    Словарь {(id_a, id_b): distance}.
        cfg:          Параметры.

    Возвращает:
        Словарь {(min_id, max_id): score}.
    """
    if cfg is None:
        cfg = GapConfig()

    result: Dict[Tuple[int, int], float] = {}
    for a in fragment_ids:
        for b in fragment_ids:
            if a >= b:
                continue
            key = (a, b)
            rev = (b, a)
            dist = distances.get(key, distances.get(rev, None))
            if dist is not None:
                score, _ = score_gap(dist, cfg)
                result[key] = score
    return result
