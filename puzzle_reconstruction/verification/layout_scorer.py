"""Оценка качества итоговой компоновки фрагментов.

Модуль вычисляет набор метрик для готовой сборки: заполненность холста,
равномерность распределения, согласованность ориентаций, уровень
перекрытий и итоговый составной балл.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ─── PlacedFragment ───────────────────────────────────────────────────────────

@dataclass
class PlacedFragment:
    """Описание размещённого фрагмента на холсте.

    Атрибуты:
        fragment_id: Уникальный ID (>= 0).
        x:           Левый край (>= 0).
        y:           Верхний край (>= 0).
        w:           Ширина (>= 1).
        h:           Высота (>= 1).
        angle:       Угол поворота в градусах.
        score:       Оценка совместимости при размещении (>= 0).
    """

    fragment_id: int
    x: int
    y: int
    w: int
    h: int
    angle: float = 0.0
    score: float = 0.0

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        if self.x < 0:
            raise ValueError(f"x должен быть >= 0, получено {self.x}")
        if self.y < 0:
            raise ValueError(f"y должен быть >= 0, получено {self.y}")
        if self.w < 1:
            raise ValueError(f"w должен быть >= 1, получено {self.w}")
        if self.h < 1:
            raise ValueError(f"h должен быть >= 1, получено {self.h}")
        if self.score < 0.0:
            raise ValueError(
                f"score должен быть >= 0, получено {self.score}"
            )

    @property
    def x2(self) -> int:
        """Правый край (включительно)."""
        return self.x + self.w

    @property
    def y2(self) -> int:
        """Нижний край (включительно)."""
        return self.y + self.h

    @property
    def area(self) -> int:
        """Площадь фрагмента."""
        return self.w * self.h

    @property
    def center(self) -> Tuple[float, float]:
        """Центр фрагмента (cx, cy)."""
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


# ─── LayoutScorerConfig ───────────────────────────────────────────────────────

@dataclass
class LayoutScorerConfig:
    """Параметры оценки компоновки.

    Атрибуты:
        canvas_w:           Ширина холста (>= 1).
        canvas_h:           Высота холста (>= 1).
        overlap_penalty:    Вес штрафа за перекрытия (>= 0).
        coverage_weight:    Вес метрики заполненности (>= 0).
        uniformity_weight:  Вес равномерности (>= 0).
        score_weight:       Вес совокупной оценки размещения (>= 0).
    """

    canvas_w: int = 512
    canvas_h: int = 512
    overlap_penalty: float = 1.0
    coverage_weight: float = 1.0
    uniformity_weight: float = 0.5
    score_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.canvas_w < 1:
            raise ValueError(
                f"canvas_w должен быть >= 1, получено {self.canvas_w}"
            )
        if self.canvas_h < 1:
            raise ValueError(
                f"canvas_h должен быть >= 1, получено {self.canvas_h}"
            )
        for name, val in (
            ("overlap_penalty", self.overlap_penalty),
            ("coverage_weight", self.coverage_weight),
            ("uniformity_weight", self.uniformity_weight),
            ("score_weight", self.score_weight),
        ):
            if val < 0.0:
                raise ValueError(f"{name} должен быть >= 0, получено {val}")


# ─── LayoutScoreResult ────────────────────────────────────────────────────────

@dataclass
class LayoutScoreResult:
    """Результат оценки компоновки.

    Атрибуты:
        total_score:     Итоговый балл (>= 0).
        coverage:        Доля площади холста, заполненная фрагментами (0–1).
        overlap_ratio:   Доля суммарной площади, занятая перекрытиями (0–1).
        uniformity:      Равномерность распределения центров (0–1).
        mean_frag_score: Средняя оценка совместимости фрагментов (>= 0).
        n_fragments:     Число размещённых фрагментов (>= 0).
    """

    total_score: float
    coverage: float
    overlap_ratio: float
    uniformity: float
    mean_frag_score: float
    n_fragments: int

    def __post_init__(self) -> None:
        if self.total_score < 0.0:
            raise ValueError(
                f"total_score должен быть >= 0, получено {self.total_score}"
            )
        for name, val in (
            ("coverage", self.coverage),
            ("overlap_ratio", self.overlap_ratio),
            ("uniformity", self.uniformity),
        ):
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"{name} должен быть в [0, 1], получено {val}"
                )
        if self.mean_frag_score < 0.0:
            raise ValueError(
                f"mean_frag_score должен быть >= 0, "
                f"получено {self.mean_frag_score}"
            )
        if self.n_fragments < 0:
            raise ValueError(
                f"n_fragments должен быть >= 0, получено {self.n_fragments}"
            )

    @property
    def quality_level(self) -> str:
        """Качественный уровень: 'poor' | 'fair' | 'good' | 'excellent'."""
        if self.total_score < 0.25:
            return "poor"
        elif self.total_score < 0.5:
            return "fair"
        elif self.total_score < 0.75:
            return "good"
        else:
            return "excellent"


# ─── compute_coverage ────────────────────────────────────────────────────────

def compute_coverage(
    fragments: List[PlacedFragment],
    canvas_w: int,
    canvas_h: int,
) -> float:
    """Доля площади холста, покрытая фрагментами (без двойного счёта).

    Аргументы:
        fragments: Список фрагментов.
        canvas_w:  Ширина холста (>= 1).
        canvas_h:  Высота холста (>= 1).

    Возвращает:
        Покрытие в [0, 1].

    Исключения:
        ValueError: Если canvas_w или canvas_h < 1.
    """
    if canvas_w < 1 or canvas_h < 1:
        raise ValueError("canvas_w и canvas_h должны быть >= 1")
    if not fragments:
        return 0.0
    mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    for f in fragments:
        x1 = max(0, f.x)
        y1 = max(0, f.y)
        x2 = min(canvas_w, f.x2)
        y2 = min(canvas_h, f.y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return float(mask.sum()) / float(canvas_w * canvas_h)


# ─── compute_overlap_ratio ───────────────────────────────────────────────────

def compute_overlap_ratio(fragments: List[PlacedFragment]) -> float:
    """Доля суммарной площади фрагментов, приходящаяся на перекрытия.

    Аргументы:
        fragments: Список фрагментов.

    Возвращает:
        Отношение площади перекрытий к суммарной площади (0 если нет).
    """
    if len(fragments) < 2:
        return 0.0
    total_area = sum(f.area for f in fragments)
    if total_area == 0:
        return 0.0

    overlap_area = 0
    for i in range(len(fragments)):
        for j in range(i + 1, len(fragments)):
            a, b = fragments[i], fragments[j]
            ix1 = max(a.x, b.x)
            iy1 = max(a.y, b.y)
            ix2 = min(a.x2, b.x2)
            iy2 = min(a.y2, b.y2)
            if ix2 > ix1 and iy2 > iy1:
                overlap_area += (ix2 - ix1) * (iy2 - iy1)

    return float(min(1.0, overlap_area / total_area))


# ─── compute_uniformity ───────────────────────────────────────────────────────

def compute_uniformity(
    fragments: List[PlacedFragment],
    canvas_w: int,
    canvas_h: int,
) -> float:
    """Равномерность распределения центров фрагментов по холсту.

    Измеряется как 1 - нормированное стандартное отклонение расстояний
    от центров до центра холста. Возвращает 1.0 для <= 1 фрагмента.

    Аргументы:
        fragments: Список фрагментов.
        canvas_w:  Ширина холста.
        canvas_h:  Высота холста.

    Возвращает:
        Равномерность в [0, 1].
    """
    if len(fragments) <= 1:
        return 1.0
    cx, cy = canvas_w / 2.0, canvas_h / 2.0
    max_dist = np.sqrt(cx ** 2 + cy ** 2) + 1e-12
    dists = np.array([
        np.sqrt((f.center[0] - cx) ** 2 + (f.center[1] - cy) ** 2) / max_dist
        for f in fragments
    ])
    std = float(np.std(dists))
    return float(np.clip(1.0 - std, 0.0, 1.0))


# ─── score_layout ─────────────────────────────────────────────────────────────

def score_layout(
    fragments: List[PlacedFragment],
    cfg: Optional[LayoutScorerConfig] = None,
) -> LayoutScoreResult:
    """Вычислить итоговый балл компоновки.

    Аргументы:
        fragments: Список размещённых фрагментов.
        cfg:       Параметры (None → LayoutScorerConfig()).

    Возвращает:
        LayoutScoreResult.
    """
    if cfg is None:
        cfg = LayoutScorerConfig()

    if not fragments:
        return LayoutScoreResult(
            total_score=0.0,
            coverage=0.0,
            overlap_ratio=0.0,
            uniformity=1.0,
            mean_frag_score=0.0,
            n_fragments=0,
        )

    coverage = compute_coverage(fragments, cfg.canvas_w, cfg.canvas_h)
    overlap = compute_overlap_ratio(fragments)
    uniformity = compute_uniformity(fragments, cfg.canvas_w, cfg.canvas_h)
    mean_score = float(np.mean([f.score for f in fragments]))

    w_total = (
        cfg.coverage_weight
        + cfg.uniformity_weight
        + cfg.score_weight
        + cfg.overlap_penalty
        + 1e-12
    )
    total = (
        cfg.coverage_weight * coverage
        + cfg.uniformity_weight * uniformity
        + cfg.score_weight * mean_score
        - cfg.overlap_penalty * overlap
    ) / w_total

    total = float(np.clip(total, 0.0, 1.0))

    return LayoutScoreResult(
        total_score=total,
        coverage=coverage,
        overlap_ratio=overlap,
        uniformity=uniformity,
        mean_frag_score=mean_score,
        n_fragments=len(fragments),
    )


# ─── rank_layouts ─────────────────────────────────────────────────────────────

def rank_layouts(
    fragment_lists: List[List[PlacedFragment]],
    cfg: Optional[LayoutScorerConfig] = None,
) -> List[Tuple[int, LayoutScoreResult]]:
    """Ранжировать несколько вариантов компоновки по убыванию total_score.

    Аргументы:
        fragment_lists: Список вариантов (каждый — список PlacedFragment).
        cfg:            Параметры.

    Возвращает:
        Список (исходный_индекс, LayoutScoreResult) по убыванию оценки.
    """
    results = [
        (i, score_layout(frags, cfg))
        for i, frags in enumerate(fragment_lists)
    ]
    results.sort(key=lambda x: x[1].total_score, reverse=True)
    return results


# ─── batch_score_layouts ──────────────────────────────────────────────────────

def batch_score_layouts(
    fragment_lists: List[List[PlacedFragment]],
    cfg: Optional[LayoutScorerConfig] = None,
) -> List[LayoutScoreResult]:
    """Оценить список вариантов компоновки.

    Аргументы:
        fragment_lists: Список вариантов.
        cfg:            Параметры.

    Возвращает:
        Список LayoutScoreResult в том же порядке.
    """
    return [score_layout(frags, cfg) for frags in fragment_lists]
