"""placement_metrics_utils — утилиты метрик качества размещения фрагментов.

Предоставляет функции для вычисления, нормализации и сравнения метрик
качества конфигураций размещения фрагментов пазла.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ─── Конфигурация ─────────────────────────────────────────────────────────────

@dataclass
class PlacementMetricsConfig:
    """Параметры вычисления метрик размещения."""
    w_density:  float = 0.4
    w_coverage: float = 0.4
    w_overlap:  float = 0.2
    canvas_size: Tuple[int, int] = (512, 512)


# ─── Результат метрик размещения ──────────────────────────────────────────────

@dataclass
class PlacementMetrics:
    """Набор метрик качества одной конфигурации размещения.

    Attributes
    ----------
    n_placed:
        Количество размещённых фрагментов.
    n_total:
        Общее количество фрагментов.
    density:
        Доля размещённых фрагментов (0..1).
    coverage:
        Доля площади холста, покрытой фрагментами (0..1).
    pairwise_overlap:
        Суммарное перекрытие пар (>= 0; меньше — лучше).
    quality_score:
        Интегральная оценка качества (0..1; больше — лучше).
    meta:
        Произвольные дополнительные поля.
    """
    n_placed:         int
    n_total:          int
    density:          float
    coverage:         float
    pairwise_overlap: float
    quality_score:    float = 0.0
    meta:             Dict   = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"PlacementMetrics(n={self.n_placed}/{self.n_total}, "
            f"density={self.density:.3f}, coverage={self.coverage:.3f}, "
            f"overlap={self.pairwise_overlap:.3f}, "
            f"quality={self.quality_score:.4f})"
        )


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def placement_density(n_placed: int, n_total: int) -> float:
    """Вычислить долю размещённых фрагментов.

    Parameters
    ----------
    n_placed:
        Число размещённых фрагментов.
    n_total:
        Общее число фрагментов.

    Returns
    -------
    float в [0, 1].

    Raises
    ------
    ValueError
        Если n_total < 0 или n_placed < 0.
    """
    if n_total < 0 or n_placed < 0:
        raise ValueError(
            f"n_placed and n_total must be non-negative, "
            f"got {n_placed}, {n_total}"
        )
    if n_total == 0:
        return 0.0
    return float(min(n_placed, n_total)) / float(n_total)


def bbox_of_contour(
    contour: np.ndarray,
    position: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[float, float, float, float]:
    """Вычислить ограничивающий прямоугольник контура с учётом смещения.

    Parameters
    ----------
    contour:
        Массив точек контура shape (N, 2), float.
    position:
        (dx, dy) — смещение фрагмента на холсте.

    Returns
    -------
    (x_min, y_min, x_max, y_max) — float.
    """
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        dx, dy = position
        return (dx, dy, dx, dy)
    dx, dy = float(position[0]), float(position[1])
    xs = pts[:, 0] + dx
    ys = pts[:, 1] + dy
    return (float(xs.min()), float(ys.min()),
            float(xs.max()), float(ys.max()))


def bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    """Площадь ограничивающего прямоугольника.

    Parameters
    ----------
    bbox:
        (x_min, y_min, x_max, y_max).

    Returns
    -------
    float >= 0.
    """
    x_min, y_min, x_max, y_max = bbox
    return max(0.0, float(x_max - x_min)) * max(0.0, float(y_max - y_min))


def bbox_intersection_area(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Площадь пересечения двух прямоугольников.

    Returns
    -------
    float >= 0.
    """
    ix_min = max(a[0], b[0])
    iy_min = max(a[1], b[1])
    ix_max = min(a[2], b[2])
    iy_max = min(a[3], b[3])
    return max(0.0, ix_max - ix_min) * max(0.0, iy_max - iy_min)


def compute_coverage(
    positions: Sequence[Tuple[float, float]],
    contours:  Sequence[np.ndarray],
    canvas_size: Tuple[int, int] = (512, 512),
) -> float:
    """Вычислить долю площади холста, покрытой фрагментами.

    Аппроксимируется через объединение ограничивающих прямоугольников
    (быстро, без точного полигонального объединения).

    Parameters
    ----------
    positions:
        Список позиций (x, y) размещённых фрагментов.
    contours:
        Соответствующие контуры фрагментов (shape N_i×2).
    canvas_size:
        (width, height) холста.

    Returns
    -------
    float в [0, 1].
    """
    if len(positions) == 0 or len(contours) == 0:
        return 0.0

    canvas_w, canvas_h = canvas_size
    canvas_area = float(canvas_w * canvas_h)
    if canvas_area <= 0.0:
        return 0.0

    # Растеризуем bbox каждого фрагмента в булев массив
    mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    n = min(len(positions), len(contours))
    for i in range(n):
        x_min, y_min, x_max, y_max = bbox_of_contour(contours[i], positions[i])
        # Клип к холсту
        x0 = int(max(0, int(np.floor(x_min))))
        y0 = int(max(0, int(np.floor(y_min))))
        x1 = int(min(canvas_w, int(np.ceil(x_max))))
        y1 = int(min(canvas_h, int(np.ceil(y_max))))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True

    return float(mask.sum()) / canvas_area


def compute_pairwise_overlap(
    positions: Sequence[Tuple[float, float]],
    contours:  Sequence[np.ndarray],
) -> float:
    """Вычислить суммарное попарное перекрытие bbox'ов фрагментов.

    Parameters
    ----------
    positions:
        Позиции размещённых фрагментов.
    contours:
        Контуры фрагментов.

    Returns
    -------
    float >= 0. Сумма площадей пересечений всех пар bbox'ов.
    """
    n = min(len(positions), len(contours))
    if n < 2:
        return 0.0

    bboxes = [bbox_of_contour(contours[i], positions[i]) for i in range(n)]
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += bbox_intersection_area(bboxes[i], bboxes[j])
    return total


def quality_score(
    density:          float,
    coverage:         float,
    pairwise_overlap: float,
    w_density:        float = 0.4,
    w_coverage:       float = 0.4,
    w_overlap:        float = 0.2,
    overlap_scale:    float = 1e4,
) -> float:
    """Интегральная оценка качества размещения.

    Parameters
    ----------
    density:
        Доля размещённых фрагментов (0..1).
    coverage:
        Покрытие холста (0..1).
    pairwise_overlap:
        Суммарное перекрытие bbox (>= 0; нормализуется через overlap_scale).
    w_density, w_coverage, w_overlap:
        Веса компонент (должны быть >= 0).
    overlap_scale:
        Нормировочный масштаб для перекрытия.

    Returns
    -------
    float в [0, 1].
    """
    overlap_penalty = float(min(1.0, pairwise_overlap / max(overlap_scale, 1.0)))
    overlap_score = 1.0 - overlap_penalty

    total_w = w_density + w_coverage + w_overlap
    if total_w < 1e-10:
        return 0.0

    score = (w_density * density + w_coverage * coverage + w_overlap * overlap_score)
    return float(max(0.0, min(1.0, score / total_w)))


def assess_placement(
    positions:       Sequence[Tuple[float, float]],
    contours:        Sequence[np.ndarray],
    n_total:         int,
    canvas_size:     Tuple[int, int] = (512, 512),
    cfg:             Optional[PlacementMetricsConfig] = None,
) -> PlacementMetrics:
    """Полная оценка конфигурации размещения.

    Parameters
    ----------
    positions:
        Позиции размещённых фрагментов.
    contours:
        Контуры фрагментов (по одному на фрагмент).
    n_total:
        Общее число фрагментов.
    canvas_size:
        Размер холста (w, h).
    cfg:
        Конфигурация (если None — используются умолчания).

    Returns
    -------
    PlacementMetrics
    """
    if cfg is None:
        cfg = PlacementMetricsConfig(canvas_size=canvas_size)

    n_placed = min(len(positions), len(contours))
    dens = placement_density(n_placed, n_total)
    cov  = compute_coverage(positions, contours, cfg.canvas_size)
    ovlp = compute_pairwise_overlap(positions, contours)
    qs   = quality_score(
        dens, cov, ovlp,
        w_density=cfg.w_density,
        w_coverage=cfg.w_coverage,
        w_overlap=cfg.w_overlap,
    )
    return PlacementMetrics(
        n_placed=n_placed,
        n_total=n_total,
        density=dens,
        coverage=cov,
        pairwise_overlap=ovlp,
        quality_score=qs,
    )


def compare_metrics(a: PlacementMetrics, b: PlacementMetrics) -> Dict:
    """Сравнить две метрики размещения.

    Returns
    -------
    dict с ключами 'density_diff', 'coverage_diff', 'overlap_diff',
    'quality_diff', 'better' ('a' или 'b').
    """
    density_diff  = a.density  - b.density
    coverage_diff = a.coverage - b.coverage
    overlap_diff  = a.pairwise_overlap - b.pairwise_overlap
    quality_diff  = a.quality_score - b.quality_score
    better = "a" if a.quality_score >= b.quality_score else "b"
    return {
        "density_diff":  density_diff,
        "coverage_diff": coverage_diff,
        "overlap_diff":  overlap_diff,
        "quality_diff":  quality_diff,
        "better":        better,
    }


def best_of(metrics_list: List[PlacementMetrics]) -> int:
    """Вернуть индекс конфигурации с наибольшим quality_score.

    Parameters
    ----------
    metrics_list:
        Список PlacementMetrics.

    Returns
    -------
    int — индекс лучшей конфигурации.

    Raises
    ------
    ValueError
        Если список пуст.
    """
    if not metrics_list:
        raise ValueError("metrics_list must not be empty")
    return int(np.argmax([m.quality_score for m in metrics_list]))


def normalize_metrics(
    metrics_list: List[PlacementMetrics],
    eps: float = 1e-8,
) -> List[PlacementMetrics]:
    """Нормализовать quality_score по всему списку метрик.

    Пересчитывает quality_score в [0, 1] относительно min/max значений.
    Константные значения нормализуются в 1.0.

    Parameters
    ----------
    metrics_list:
        Входной список PlacementMetrics.
    eps:
        Защита от деления на ноль.

    Returns
    -------
    Новый список PlacementMetrics с обновлёнными quality_score.
    """
    if not metrics_list:
        return []

    scores = np.array([m.quality_score for m in metrics_list], dtype=np.float64)
    mn, mx = scores.min(), scores.max()
    rng = mx - mn
    if rng < eps:
        normalized = np.ones_like(scores)
    else:
        normalized = (scores - mn) / rng

    result = []
    for m, ns in zip(metrics_list, normalized):
        result.append(PlacementMetrics(
            n_placed=m.n_placed,
            n_total=m.n_total,
            density=m.density,
            coverage=m.coverage,
            pairwise_overlap=m.pairwise_overlap,
            quality_score=float(ns),
            meta=dict(m.meta),
        ))
    return result


def batch_quality_scores(
    metrics_list: List[PlacementMetrics],
    w_density:    float = 0.4,
    w_coverage:   float = 0.4,
    w_overlap:    float = 0.2,
) -> List[float]:
    """Пересчитать quality_score для всего списка метрик с заданными весами.

    Parameters
    ----------
    metrics_list:
        Список PlacementMetrics.
    w_density, w_coverage, w_overlap:
        Новые веса компонент.

    Returns
    -------
    Список float одинаковой длины.
    """
    return [
        quality_score(
            m.density, m.coverage, m.pairwise_overlap,
            w_density=w_density, w_coverage=w_coverage, w_overlap=w_overlap,
        )
        for m in metrics_list
    ]
