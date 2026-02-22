"""
Геометрическое сопоставление фрагментов документа.

Сравнивает фрагменты по геометрическим характеристикам, вычисленным из
бинарных масок: площадь, периметр, отношение сторон, выпуклая оболочка,
инвариантные моменты Ху и длина сопрягаемого края.

Классы:
    FragmentGeometry     — геометрический профиль одного фрагмента
    GeometricMatchResult — результат сопоставления двух фрагментов

Функции:
    compute_fragment_geometry — вычисление профиля по бинарной маске
    aspect_ratio_similarity   — сходство по отношению сторон ∈ [0,1]
    area_ratio_similarity     — сходство по площади ∈ [0,1]
    hu_moments_similarity     — сходство по моментам Ху ∈ [0,1]
    edge_length_similarity    — сходство по длине края ∈ [0,1]
    match_geometry            — взвешенная итоговая оценка
    batch_geometry_match      — пакетная обработка пар фрагментов
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── FragmentGeometry ─────────────────────────────────────────────────────────

@dataclass
class FragmentGeometry:
    """
    Геометрический профиль одного фрагмента.

    Attributes:
        area:         Площадь контура (пикс²).
        perimeter:    Длина контура (пикс).
        aspect_ratio: Отношение сторон описывающего прямоугольника (w/h ≥ 1).
        hull_area:    Площадь выпуклой оболочки (пикс²).
        solidity:     Плотность = area / hull_area ∈ (0,1].
        hu_moments:   7 инвариантных моментов Ху (log-нормализованные).
        bbox:         (x, y, w, h) описывающего прямоугольника.
        params:       Дополнительные поля.
    """
    area:         float
    perimeter:    float
    aspect_ratio: float
    hull_area:    float
    solidity:     float
    hu_moments:   np.ndarray   # shape (7,), dtype float64
    bbox:         Tuple[int, int, int, int]
    params:       Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"FragmentGeometry(area={self.area:.1f}, "
                f"ar={self.aspect_ratio:.2f}, "
                f"solidity={self.solidity:.2f})")


# ─── GeometricMatchResult ────────────────────────────────────────────────────

@dataclass
class GeometricMatchResult:
    """
    Результат геометрического сопоставления двух фрагментов.

    Attributes:
        score:        Взвешенная итоговая оценка ∈ [0,1].
        aspect_score: Сходство по отношению сторон ∈ [0,1].
        area_score:   Сходство по площади ∈ [0,1].
        hu_score:     Сходство по моментам Ху ∈ [0,1].
        method:       Всегда 'geometric'.
        params:       Веса и вспомогательная информация.
    """
    score:        float
    aspect_score: float
    area_score:   float
    hu_score:     float
    method:       str  = "geometric"
    params:       Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"GeometricMatchResult(score={self.score:.3f}, "
                f"aspect={self.aspect_score:.3f}, "
                f"area={self.area_score:.3f}, "
                f"hu={self.hu_score:.3f})")


# ─── compute_fragment_geometry ───────────────────────────────────────────────

def compute_fragment_geometry(mask: np.ndarray,
                               epsilon_frac: float = 0.02) -> FragmentGeometry:
    """
    Вычисляет геометрический профиль по бинарной маске фрагмента.

    Args:
        mask:         Бинарное изображение (uint8, пиксели 0 или 255) или
                      изображение в оттенках серого (будет бинаризовано > 127).
        epsilon_frac: Доля периметра для аппроксимации контура (не используется
                      в вычислении, но сохраняется в params).

    Returns:
        FragmentGeometry с вычисленными характеристиками.
        Если контур не найден, возвращает нулевой профиль.
    """
    # Бинаризация
    if mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask

    bw = (gray > 127).astype(np.uint8) * 255

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    _zero7 = np.zeros(7, dtype=np.float64)

    if not contours:
        return FragmentGeometry(
            area=0.0, perimeter=0.0, aspect_ratio=1.0,
            hull_area=0.0, solidity=0.0,
            hu_moments=_zero7, bbox=(0, 0, 0, 0),
            params={"epsilon_frac": epsilon_frac, "n_contours": 0},
        )

    # Берём наибольший контур
    cnt = max(contours, key=cv2.contourArea)

    area      = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))

    x, y, w, h = cv2.boundingRect(cnt)
    ar = float(max(w, h)) / float(max(min(w, h), 1))

    hull      = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity  = float(area / hull_area) if hull_area > 0 else 0.0

    # Инвариантные моменты Ху (log-нормализация по Belongie et al.)
    raw_moments = cv2.moments(cnt)
    hu_raw      = cv2.HuMoments(raw_moments).flatten()
    # log-scale: sign(h) * log10(|h|+ε)
    hu_log = np.sign(hu_raw) * np.log10(np.abs(hu_raw) + 1e-10)

    return FragmentGeometry(
        area=area, perimeter=perimeter, aspect_ratio=ar,
        hull_area=hull_area, solidity=solidity,
        hu_moments=hu_log.astype(np.float64),
        bbox=(int(x), int(y), int(w), int(h)),
        params={"epsilon_frac": epsilon_frac, "n_contours": len(contours)},
    )


# ─── Попарные меры сходства ──────────────────────────────────────────────────

def aspect_ratio_similarity(g1: FragmentGeometry,
                              g2: FragmentGeometry) -> float:
    """
    Сходство по отношению сторон ∈ [0,1].

    Оценка = 1 − |AR1 − AR2| / max(AR1, AR2).
    """
    a1, a2 = g1.aspect_ratio, g2.aspect_ratio
    denom  = max(a1, a2, 1e-9)
    return float(np.clip(1.0 - abs(a1 - a2) / denom, 0.0, 1.0))


def area_ratio_similarity(g1: FragmentGeometry,
                           g2: FragmentGeometry) -> float:
    """
    Сходство по площади ∈ [0,1].

    Оценка = min(A1,A2) / max(A1,A2) или 1.0 при нулевых площадях.
    """
    a1, a2 = g1.area, g2.area
    mx = max(a1, a2)
    if mx < 1e-9:
        return 1.0
    return float(np.clip(min(a1, a2) / mx, 0.0, 1.0))


def hu_moments_similarity(g1: FragmentGeometry,
                           g2: FragmentGeometry) -> float:
    """
    Сходство по инвариантным моментам Ху ∈ [0,1].

    Использует L2-расстояние между log-нормализованными векторами моментов.
    Нормированное расстояние → оценка через экспоненциальный спад.
    """
    hu1 = g1.hu_moments
    hu2 = g2.hu_moments
    dist = float(np.linalg.norm(hu1 - hu2))
    # Масштаб σ подобран эмпирически для log-моментов (диапазон ~0–20)
    return float(np.exp(-dist / 10.0))


def edge_length_similarity(len1: float, len2: float) -> float:
    """
    Сходство по длине сопрягаемого края ∈ [0,1].

    Оценка = min(len1,len2) / max(len1,len2) или 1.0 при нулях.
    """
    mx = max(len1, len2)
    if mx < 1e-9:
        return 1.0
    return float(np.clip(min(len1, len2) / mx, 0.0, 1.0))


# ─── match_geometry ───────────────────────────────────────────────────────────

def match_geometry(g1:        FragmentGeometry,
                   g2:        FragmentGeometry,
                   w_aspect:  float = 0.3,
                   w_area:    float = 0.4,
                   w_hu:      float = 0.3,
                   edge_len1: Optional[float] = None,
                   edge_len2: Optional[float] = None) -> GeometricMatchResult:
    """
    Итоговая взвешенная оценка геометрического сопоставления.

    Args:
        g1, g2:    Профили двух фрагментов.
        w_aspect:  Вес оценки по отношению сторон.
        w_area:    Вес оценки по площади.
        w_hu:      Вес оценки по моментам Ху.
        edge_len1: Длина сопрягаемого края первого фрагмента (опц.).
        edge_len2: Длина сопрягаемого края второго фрагмента (опц.).

    Returns:
        GeometricMatchResult со взвешенной оценкой.
    """
    # Нормировка весов
    total = w_aspect + w_area + w_hu
    if total < 1e-9:
        total = 1.0
    wa, wA, wh = w_aspect / total, w_area / total, w_hu / total

    s_aspect = aspect_ratio_similarity(g1, g2)
    s_area   = area_ratio_similarity(g1, g2)
    s_hu     = hu_moments_similarity(g1, g2)

    score = float(wa * s_aspect + wA * s_area + wh * s_hu)
    score = float(np.clip(score, 0.0, 1.0))

    params: Dict = {
        "w_aspect": w_aspect, "w_area": w_area, "w_hu": w_hu,
    }
    if edge_len1 is not None and edge_len2 is not None:
        el_score = edge_length_similarity(edge_len1, edge_len2)
        score    = float(np.clip((score + el_score) / 2.0, 0.0, 1.0))
        params["edge_len_score"] = el_score

    return GeometricMatchResult(
        score=score,
        aspect_score=s_aspect,
        area_score=s_area,
        hu_score=s_hu,
        method="geometric",
        params=params,
    )


# ─── batch_geometry_match ─────────────────────────────────────────────────────

def batch_geometry_match(geometries: List[FragmentGeometry],
                          pairs:      List[Tuple[int, int]],
                          w_aspect:   float = 0.3,
                          w_area:     float = 0.4,
                          w_hu:       float = 0.3) -> List[GeometricMatchResult]:
    """
    Пакетная геометрическая оценка для списка пар индексов.

    Args:
        geometries: Список FragmentGeometry.
        pairs:      Список пар (i, j) — индексы в geometries.
        w_aspect:   Вес для отношения сторон.
        w_area:     Вес для площади.
        w_hu:       Вес для моментов Ху.

    Returns:
        Список GeometricMatchResult (по одному на пару).

    Raises:
        IndexError: Если индексы выходят за пределы geometries.
    """
    return [
        match_geometry(geometries[i], geometries[j],
                       w_aspect=w_aspect, w_area=w_area, w_hu=w_hu)
        for i, j in pairs
    ]
