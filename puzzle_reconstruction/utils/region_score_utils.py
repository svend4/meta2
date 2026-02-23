"""region_score_utils — утилиты оценки и фильтрации регионов.

Предоставляет функции для вычисления оценок качества, ранжирования
и пакетной обработки регионов маски на основе геометрических свойств.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Конфигурация ─────────────────────────────────────────────────────────────

@dataclass
class RegionScoreConfig:
    """Параметры вычисления оценок регионов."""
    w_area:       float = 0.4
    w_compactness: float = 0.3
    w_extent:     float = 0.3
    min_area:     int   = 1
    max_area:     int   = 10_000_000


# ─── Результат оценки региона ─────────────────────────────────────────────────

@dataclass
class RegionScore:
    """Оценка одного региона маски.

    Attributes
    ----------
    label:
        Метка региона.
    area:
        Площадь (пикселей).
    compactness:
        Компактность: 4π·area / perimeter². В [0, 1]; круг = 1.
    extent:
        Заполнение bounding box: area / bbox_area. В (0, 1].
    score:
        Интегральная оценка (0..1).
    meta:
        Дополнительные данные.
    """
    label:       int
    area:        int
    compactness: float
    extent:      float
    score:       float = 0.0
    meta:        Dict  = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"RegionScore(label={self.label}, area={self.area}, "
                f"compactness={self.compactness:.3f}, "
                f"extent={self.extent:.3f}, score={self.score:.4f})")


# ─── Вычисление геометрических характеристик ──────────────────────────────────

def region_compactness(area: int, perimeter: float) -> float:
    """Вычислить компактность региона.

    compactness = 4π·area / perimeter²

    Parameters
    ----------
    area:
        Площадь региона (пикс.).
    perimeter:
        Периметр региона (пикс.).

    Returns
    -------
    float в [0, 1]; 0 если perimeter == 0.
    """
    if perimeter < 1e-9:
        return 0.0
    return float(min(1.0, 4.0 * np.pi * area / (perimeter ** 2)))


def region_extent(area: int, bbox: Tuple[int, int, int, int]) -> float:
    """Вычислить заполнение bounding box.

    extent = area / (bbox_width * bbox_height)

    Parameters
    ----------
    area:
        Площадь региона.
    bbox:
        (row_min, col_min, row_max, col_max).

    Returns
    -------
    float в (0, 1]. 1 если bbox_area == 0.
    """
    r_min, c_min, r_max, c_max = bbox
    bbox_area = max(1, (r_max - r_min) * (c_max - c_min))
    return float(min(1.0, area / bbox_area))


def mask_perimeter(mask: np.ndarray) -> float:
    """Приближённый периметр бинарной маски (подсчёт граничных пикселей).

    Parameters
    ----------
    mask:
        2-D массив uint8 (ненулевые пиксели = регион).

    Returns
    -------
    float — число граничных пикселей.
    """
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return 0.0
    # Сдвиги по всем 4 направлениям
    up    = np.pad(m, ((1, 0), (0, 0)), mode="constant")[:-1, :]
    down  = np.pad(m, ((0, 1), (0, 0)), mode="constant")[1:,  :]
    left  = np.pad(m, ((0, 0), (1, 0)), mode="constant")[:, :-1]
    right = np.pad(m, ((0, 0), (0, 1)), mode="constant")[:, 1:]
    boundary = m & ~(up & down & left & right)
    return float(boundary.sum())


# ─── Оценка регионов ──────────────────────────────────────────────────────────

def score_region(
    area:     int,
    bbox:     Tuple[int, int, int, int],
    mask:     np.ndarray,
    cfg:      Optional[RegionScoreConfig] = None,
) -> float:
    """Вычислить интегральную оценку одного региона.

    Parameters
    ----------
    area:
        Площадь региона.
    bbox:
        (row_min, col_min, row_max, col_max).
    mask:
        2-D маска региона (uint8).
    cfg:
        Конфигурация весов.

    Returns
    -------
    float в [0, 1].
    """
    if cfg is None:
        cfg = RegionScoreConfig()

    perim = mask_perimeter(mask)
    comp  = region_compactness(area, perim)
    ext   = region_extent(area, bbox)

    # Нормализованная площадь (логарифмическая шкала)
    norm_area = float(
        np.clip(np.log1p(area) / np.log1p(cfg.max_area), 0.0, 1.0)
    )

    total_w = cfg.w_area + cfg.w_compactness + cfg.w_extent
    if total_w < 1e-10:
        return 0.0
    raw = (cfg.w_area * norm_area
           + cfg.w_compactness * comp
           + cfg.w_extent * ext)
    return float(max(0.0, min(1.0, raw / total_w)))


def evaluate_region(
    label: int,
    area:  int,
    bbox:  Tuple[int, int, int, int],
    mask:  np.ndarray,
    cfg:   Optional[RegionScoreConfig] = None,
) -> RegionScore:
    """Создать RegionScore для одного региона.

    Parameters
    ----------
    label:
        Метка региона.
    area, bbox, mask:
        Геометрические свойства региона.
    cfg:
        Конфигурация.

    Returns
    -------
    RegionScore
    """
    if cfg is None:
        cfg = RegionScoreConfig()

    perim = mask_perimeter(mask)
    comp  = region_compactness(area, perim)
    ext   = region_extent(area, bbox)
    sc    = score_region(area, bbox, mask, cfg)

    return RegionScore(
        label=label,
        area=area,
        compactness=comp,
        extent=ext,
        score=sc,
    )


def filter_by_score(
    scores: List[RegionScore],
    threshold: float = 0.0,
) -> List[RegionScore]:
    """Отфильтровать регионы с оценкой ниже порога.

    Parameters
    ----------
    scores:
        Список RegionScore.
    threshold:
        Минимальный score (включительно).

    Returns
    -------
    Отфильтрованный список (порядок сохранён).
    """
    return [s for s in scores if s.score >= threshold]


def rank_regions(
    scores: List[RegionScore],
    reverse: bool = True,
) -> List[RegionScore]:
    """Отсортировать регионы по убыванию (или возрастанию) оценки.

    Parameters
    ----------
    scores:
        Список RegionScore.
    reverse:
        True → убывание (лучшие первые).

    Returns
    -------
    Новый отсортированный список.
    """
    return sorted(scores, key=lambda r: r.score, reverse=reverse)


def top_k_regions(
    scores: List[RegionScore],
    k: int,
) -> List[RegionScore]:
    """Вернуть top-k регионов по убыванию score.

    Parameters
    ----------
    scores:
        Список RegionScore.
    k:
        Количество лучших регионов.

    Returns
    -------
    Список из не более k RegionScore.
    """
    return rank_regions(scores)[:k]


def region_score_stats(scores: List[RegionScore]) -> Dict:
    """Вычислить статистику по списку RegionScore.

    Returns
    -------
    dict с ключами 'n', 'mean_score', 'max_score', 'min_score',
    'mean_area', 'total_area'.
    """
    if not scores:
        return {
            "n": 0, "mean_score": 0.0, "max_score": 0.0,
            "min_score": 0.0, "mean_area": 0.0, "total_area": 0,
        }
    sc   = [r.score for r in scores]
    areas = [r.area for r in scores]
    return {
        "n":           len(scores),
        "mean_score":  float(np.mean(sc)),
        "max_score":   float(np.max(sc)),
        "min_score":   float(np.min(sc)),
        "mean_area":   float(np.mean(areas)),
        "total_area":  int(sum(areas)),
    }


def batch_evaluate_regions(
    regions: List[Dict],
    cfg:     Optional[RegionScoreConfig] = None,
) -> List[RegionScore]:
    """Пакетно оценить список регионов.

    Parameters
    ----------
    regions:
        Список dict с ключами 'label', 'area', 'bbox', 'mask'.
    cfg:
        Конфигурация.

    Returns
    -------
    Список RegionScore.
    """
    return [
        evaluate_region(
            label=r["label"],
            area=r["area"],
            bbox=r["bbox"],
            mask=r["mask"],
            cfg=cfg,
        )
        for r in regions
    ]


def normalize_scores(
    scores: List[RegionScore],
    eps: float = 1e-8,
) -> List[RegionScore]:
    """Нормализовать score всех регионов в [0, 1] относительно min/max.

    Parameters
    ----------
    scores:
        Входной список RegionScore.
    eps:
        Защита от деления на ноль.

    Returns
    -------
    Новый список RegionScore с обновлёнными score.
    """
    if not scores:
        return []
    vals = np.array([r.score for r in scores], dtype=np.float64)
    mn, mx = vals.min(), vals.max()
    rng = mx - mn
    if rng < eps:
        normalized = np.ones_like(vals)
    else:
        normalized = (vals - mn) / rng

    result = []
    for r, ns in zip(scores, normalized):
        result.append(RegionScore(
            label=r.label,
            area=r.area,
            compactness=r.compactness,
            extent=r.extent,
            score=float(ns),
            meta=dict(r.meta),
        ))
    return result
