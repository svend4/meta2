"""Оценка совместимости регионов фрагментов документа.

Вычисляет попарную совместимость двух регионов (прямоугольных
областей изображения) по нескольким каналам: цвет, текстура, форма
и граничная близость.

Публичный API:
    RegionScorerConfig  — параметры оценки
    RegionScore         — результат попарной оценки двух регионов
    color_similarity    — схожесть по средней яркости ∈ [0, 1]
    texture_similarity  — схожесть по стандартному отклонению ∈ [0, 1]
    shape_similarity    — схожесть по форме bbox (aspect ratio) ∈ [0, 1]
    boundary_proximity  — близость граничных точек ∈ [0, 1]
    score_region_pair   — итоговая взвешенная оценка пары регионов
    batch_score_regions — пакетная оценка списка пар
    rank_region_pairs   — ранжирование пар по убыванию оценки
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── RegionScorerConfig ───────────────────────────────────────────────────────

@dataclass
class RegionScorerConfig:
    """Параметры оценки совместимости регионов.

    Атрибуты:
        w_color:    Вес канала цвета (>= 0).
        w_texture:  Вес канала текстуры (>= 0).
        w_shape:    Вес канала формы (>= 0).
        w_boundary: Вес канала граничной близости (>= 0).
        max_distance: Максимальное расстояние для нормализации близости (> 0).
    """

    w_color:     float = 0.35
    w_texture:   float = 0.25
    w_shape:     float = 0.20
    w_boundary:  float = 0.20
    max_distance: float = 100.0

    def __post_init__(self) -> None:
        for name, val in (
            ("w_color",    self.w_color),
            ("w_texture",  self.w_texture),
            ("w_shape",    self.w_shape),
            ("w_boundary", self.w_boundary),
        ):
            if val < 0.0:
                raise ValueError(
                    f"{name} должен быть >= 0, получено {val}"
                )
        if self.max_distance <= 0.0:
            raise ValueError(
                f"max_distance должен быть > 0, получено {self.max_distance}"
            )

    @property
    def total_weight(self) -> float:
        """Суммарный вес всех каналов."""
        return self.w_color + self.w_texture + self.w_shape + self.w_boundary


# ─── RegionScore ──────────────────────────────────────────────────────────────

@dataclass
class RegionScore:
    """Результат оценки совместимости пары регионов.

    Атрибуты:
        score:        Итоговая взвешенная оценка ∈ [0, 1].
        color_score:  Оценка по цвету ∈ [0, 1].
        texture_score: Оценка по текстуре ∈ [0, 1].
        shape_score:  Оценка по форме ∈ [0, 1].
        boundary_score: Оценка близости границ ∈ [0, 1].
        params:       Дополнительные данные.
    """

    score:          float
    color_score:    float
    texture_score:  float
    shape_score:    float
    boundary_score: float
    params:         Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, val in (
            ("score",          self.score),
            ("color_score",    self.color_score),
            ("texture_score",  self.texture_score),
            ("shape_score",    self.shape_score),
            ("boundary_score", self.boundary_score),
        ):
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"{name} должен быть в [0, 1], получено {val}"
                )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RegionScore(score={self.score:.3f}, "
            f"color={self.color_score:.3f}, "
            f"texture={self.texture_score:.3f}, "
            f"shape={self.shape_score:.3f}, "
            f"boundary={self.boundary_score:.3f})"
        )


# ─── color_similarity ─────────────────────────────────────────────────────────

def color_similarity(
    patch_a: np.ndarray,
    patch_b: np.ndarray,
) -> float:
    """Схожесть по средней яркости двух патчей ∈ [0, 1].

    Использует нормализованную разность средних: 1 − |μ_a − μ_b| / 255.

    Аргументы:
        patch_a: Grayscale или RGB uint8 патч.
        patch_b: Grayscale или RGB uint8 патч.

    Возвращает:
        Схожесть ∈ [0, 1].
    """
    a = np.asarray(patch_a, dtype=np.float32)
    b = np.asarray(patch_b, dtype=np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    if b.ndim == 3:
        b = b.mean(axis=2)

    if a.size == 0 or b.size == 0:
        return 1.0

    mu_a = float(a.mean())
    mu_b = float(b.mean())
    diff = abs(mu_a - mu_b) / 255.0
    return float(np.clip(1.0 - diff, 0.0, 1.0))


# ─── texture_similarity ───────────────────────────────────────────────────────

def texture_similarity(
    patch_a: np.ndarray,
    patch_b: np.ndarray,
) -> float:
    """Схожесть по стандартному отклонению яркости ∈ [0, 1].

    Использует отношение min(σ_a, σ_b) / max(σ_a, σ_b).
    При нулевом std обоих патчей возвращает 1.0.

    Аргументы:
        patch_a: Grayscale или RGB uint8 патч.
        patch_b: Grayscale или RGB uint8 патч.

    Возвращает:
        Схожесть ∈ [0, 1].
    """
    a = np.asarray(patch_a, dtype=np.float32)
    b = np.asarray(patch_b, dtype=np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    if b.ndim == 3:
        b = b.mean(axis=2)

    sa = float(a.std())
    sb = float(b.std())
    mx = max(sa, sb)
    if mx < 1e-9:
        return 1.0
    return float(np.clip(min(sa, sb) / mx, 0.0, 1.0))


# ─── shape_similarity ─────────────────────────────────────────────────────────

def shape_similarity(
    bbox_a: Tuple[int, int, int, int],
    bbox_b: Tuple[int, int, int, int],
) -> float:
    """Схожесть формы двух bbox по соотношению сторон ∈ [0, 1].

    Каждый bbox задан как (x, y, w, h).
    Соотношение сторон: max(w,h) / max(min(w,h), 1).

    Аргументы:
        bbox_a: (x, y, w, h) первого региона.
        bbox_b: (x, y, w, h) второго региона.

    Возвращает:
        Схожесть ∈ [0, 1].
    """
    _, _, wa, ha = bbox_a
    _, _, wb, hb = bbox_b

    def _aspect(w: int, h: int) -> float:
        return float(max(w, h)) / float(max(min(w, h), 1))

    ar_a = _aspect(wa, ha)
    ar_b = _aspect(wb, hb)
    denom = max(ar_a, ar_b, 1e-9)
    return float(np.clip(1.0 - abs(ar_a - ar_b) / denom, 0.0, 1.0))


# ─── boundary_proximity ───────────────────────────────────────────────────────

def boundary_proximity(
    centroid_a: Tuple[float, float],
    centroid_b: Tuple[float, float],
    max_distance: float = 100.0,
) -> float:
    """Близость двух регионов по расстоянию центроидов ∈ [0, 1].

    Оценка = max(0, 1 − dist / max_distance).

    Аргументы:
        centroid_a:   (x, y) центроида первого региона.
        centroid_b:   (x, y) центроида второго региона.
        max_distance: Нормирующее расстояние (> 0).

    Возвращает:
        Близость ∈ [0, 1].

    Исключения:
        ValueError: Если max_distance <= 0.
    """
    if max_distance <= 0.0:
        raise ValueError(
            f"max_distance должен быть > 0, получено {max_distance}"
        )
    dx = centroid_a[0] - centroid_b[0]
    dy = centroid_a[1] - centroid_b[1]
    dist = float(np.sqrt(dx ** 2 + dy ** 2))
    return float(np.clip(1.0 - dist / max_distance, 0.0, 1.0))


# ─── score_region_pair ────────────────────────────────────────────────────────

def score_region_pair(
    patch_a: np.ndarray,
    bbox_a:  Tuple[int, int, int, int],
    patch_b: np.ndarray,
    bbox_b:  Tuple[int, int, int, int],
    cfg:     Optional[RegionScorerConfig] = None,
) -> RegionScore:
    """Итоговая взвешенная оценка совместимости пары регионов.

    Аргументы:
        patch_a: Патч первого региона (uint8, gray или RGB).
        bbox_a:  (x, y, w, h) первого региона.
        patch_b: Патч второго региона.
        bbox_b:  (x, y, w, h) второго региона.
        cfg:     Параметры (None → RegionScorerConfig()).

    Возвращает:
        :class:`RegionScore` с итоговой оценкой и компонентами.
    """
    if cfg is None:
        cfg = RegionScorerConfig()

    c_score = color_similarity(patch_a, patch_b)
    t_score = texture_similarity(patch_a, patch_b)
    s_score = shape_similarity(bbox_a, bbox_b)

    xa, ya, wa, ha = bbox_a
    xb, yb, wb, hb = bbox_b
    cen_a = (xa + wa / 2.0, ya + ha / 2.0)
    cen_b = (xb + wb / 2.0, yb + hb / 2.0)
    b_score = boundary_proximity(cen_a, cen_b, max_distance=cfg.max_distance)

    total = cfg.total_weight + 1e-12
    score = (
        cfg.w_color    * c_score
        + cfg.w_texture  * t_score
        + cfg.w_shape    * s_score
        + cfg.w_boundary * b_score
    ) / total

    score = float(np.clip(score, 0.0, 1.0))

    return RegionScore(
        score=score,
        color_score=c_score,
        texture_score=t_score,
        shape_score=s_score,
        boundary_score=b_score,
        params={
            "w_color":    cfg.w_color,
            "w_texture":  cfg.w_texture,
            "w_shape":    cfg.w_shape,
            "w_boundary": cfg.w_boundary,
        },
    )


# ─── batch_score_regions ──────────────────────────────────────────────────────

def batch_score_regions(
    pairs: List[Tuple[np.ndarray, Tuple[int, int, int, int],
                       np.ndarray, Tuple[int, int, int, int]]],
    cfg:   Optional[RegionScorerConfig] = None,
) -> List[RegionScore]:
    """Пакетная оценка списка пар регионов.

    Аргументы:
        pairs: Список кортежей (patch_a, bbox_a, patch_b, bbox_b).
        cfg:   Параметры оценки.

    Возвращает:
        Список :class:`RegionScore` той же длины.
    """
    if cfg is None:
        cfg = RegionScorerConfig()
    return [
        score_region_pair(pa, ba, pb, bb, cfg)
        for pa, ba, pb, bb in pairs
    ]


# ─── rank_region_pairs ────────────────────────────────────────────────────────

def rank_region_pairs(
    scores: List[RegionScore],
    indices: Optional[List[int]] = None,
) -> List[Tuple[int, float]]:
    """Ранжировать пары регионов по убыванию оценки.

    Аргументы:
        scores:  Список :class:`RegionScore`.
        indices: Идентификаторы пар (None → 0, 1, 2, …).

    Возвращает:
        Список кортежей (pair_id, score), отсортированный по убыванию score.
    """
    if indices is None:
        indices = list(range(len(scores)))
    paired = list(zip(indices, [rs.score for rs in scores]))
    return sorted(paired, key=lambda x: x[1], reverse=True)
