"""
Детальный анализ качества швов (stitch seams) между фрагментами.

Оценивает насколько хорошо два фрагмента «стыкуются» по своим краям,
анализируя непрерывность яркости, градиентов и текстуры вдоль шва.

Классы:
    SeamAnalysis — результат анализа одного шва

Функции:
    extract_seam_profiles   — строит попарные 1D-профили вдоль шва
    brightness_continuity   — оценка непрерывности яркости по шву
    gradient_continuity     — оценка непрерывности градиента по шву
    texture_continuity      — оценка непрерывности текстуры по шву
    analyze_seam            — полный анализ одного шва
    score_seam_quality      — скалярная оценка качества шва
    batch_analyze_seams     — пакетный анализ списка пар
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── SeamAnalysis ─────────────────────────────────────────────────────────────

@dataclass
class SeamAnalysis:
    """
    Результат анализа одного шва между двумя фрагментами.

    Attributes:
        idx1:                Индекс первого фрагмента.
        idx2:                Индекс второго фрагмента.
        side1:               Сторона первого фрагмента (0=верх,1=право,2=низ,3=лево).
        side2:               Сторона второго фрагмента.
        brightness_score:    Непрерывность яркости ∈ [0, 1].
        gradient_score:      Непрерывность градиента ∈ [0, 1].
        texture_score:       Непрерывность текстуры ∈ [0, 1].
        quality_score:       Взвешенное среднее всех трёх ∈ [0, 1].
        profile_length:      Длина профиля (пикселей вдоль шва).
        params:              Параметры анализа.
    """
    idx1:             int
    idx2:             int
    side1:            int
    side2:            int
    brightness_score: float
    gradient_score:   float
    texture_score:    float
    quality_score:    float
    profile_length:   int
    params:           Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"SeamAnalysis(idx1={self.idx1}, idx2={self.idx2}, "
                f"side1={self.side1}, side2={self.side2}, "
                f"quality={self.quality_score:.3f})")


# ─── _to_gray ─────────────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ─── _edge_profile ────────────────────────────────────────────────────────────

def _edge_profile(img: np.ndarray, side: int, border_px: int) -> np.ndarray:
    """Средний 1D-профиль яркости вдоль указанного края. Возвращает float64."""
    gray = _to_gray(img).astype(np.float64)
    bp   = max(1, border_px)
    if side == 0:
        return gray[:bp, :].mean(axis=0)
    elif side == 1:
        return gray[:, -bp:].mean(axis=1)
    elif side == 2:
        return gray[-bp:, :].mean(axis=0)
    else:  # side == 3
        return gray[:, :bp].mean(axis=1)


# ─── extract_seam_profiles ────────────────────────────────────────────────────

def extract_seam_profiles(
    img1:      np.ndarray,
    img2:      np.ndarray,
    side1:     int = 2,
    side2:     int = 0,
    border_px: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Извлекает усреднённые 1D-профили яркости вдоль указанных краёв.

    Args:
        img1:      Первый фрагмент (BGR или grayscale).
        img2:      Второй фрагмент (BGR или grayscale).
        side1:     Край первого фрагмента (0–3).
        side2:     Край второго фрагмента (0–3).
        border_px: Ширина усредняемой полосы.

    Returns:
        (profile1, profile2) — два массива float64 одинаковой длины (усечены
        до min(len(p1), len(p2))).

    Raises:
        ValueError: Если side1 или side2 не в [0, 3].
    """
    for s in (side1, side2):
        if s not in (0, 1, 2, 3):
            raise ValueError(f"side must be 0, 1, 2, or 3; got {s!r}.")

    p1 = _edge_profile(img1, side1, border_px)
    p2 = _edge_profile(img2, side2, border_px)
    L  = min(len(p1), len(p2))
    return p1[:L], p2[:L]


# ─── brightness_continuity ────────────────────────────────────────────────────

def brightness_continuity(
    profile1: np.ndarray,
    profile2: np.ndarray,
    max_diff: float = 255.0,
) -> float:
    """
    Оценивает непрерывность яркости между двумя профилями.

    score = 1 - mean(|p1 - p2|) / max_diff.

    Args:
        profile1: float64 1D-профиль первого края.
        profile2: float64 1D-профиль второго края.
        max_diff: Нормировочная константа (максимально возможная разница).

    Returns:
        Оценка ∈ [0.0, 1.0].
    """
    if profile1.size == 0 or profile2.size == 0:
        return 0.0
    L    = min(len(profile1), len(profile2))
    diff = np.abs(profile1[:L] - profile2[:L])
    return float(np.clip(1.0 - diff.mean() / max(max_diff, 1e-9), 0.0, 1.0))


# ─── gradient_continuity ──────────────────────────────────────────────────────

def gradient_continuity(
    profile1: np.ndarray,
    profile2: np.ndarray,
) -> float:
    """
    Оценивает непрерывность производной (градиента) профилей.

    Сравнивает градиенты (diff) двух профилей через нормированную
    кросс-корреляцию. Если оба градиента плоские → 1.0.

    Args:
        profile1: float64 1D-профиль первого края.
        profile2: float64 1D-профиль второго края.

    Returns:
        Оценка ∈ [0.0, 1.0].
    """
    if profile1.size < 2 or profile2.size < 2:
        return 0.0
    L    = min(len(profile1), len(profile2))
    g1   = np.diff(profile1[:L])
    g2   = np.diff(profile2[:L])
    s1, s2 = g1.std(), g2.std()
    if s1 < 1e-6 and s2 < 1e-6:
        # Both constant: check if same direction
        m1, m2 = float(g1.mean()), float(g2.mean())
        if abs(m1) < 1e-6 and abs(m2) < 1e-6:
            return 1.0  # Both near-zero → continuous
        return 1.0 if m1 * m2 >= 0 else 0.0  # Same direction → good, opposite → bad
    if s1 < 1e-6 or s2 < 1e-6:
        return 0.5
    corr = float(np.corrcoef(g1, g2)[0, 1])
    return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


# ─── texture_continuity ───────────────────────────────────────────────────────

def texture_continuity(
    profile1: np.ndarray,
    profile2: np.ndarray,
) -> float:
    """
    Оценивает схожесть текстуры (разброса) двух профилей.

    Использует отношение минимального std к максимальному.
    Если оба профиля однородны → 1.0.

    Args:
        profile1: float64 1D-профиль первого края.
        profile2: float64 1D-профиль второго края.

    Returns:
        Оценка ∈ [0.0, 1.0].
    """
    if profile1.size == 0 or profile2.size == 0:
        return 0.0
    L    = min(len(profile1), len(profile2))
    s1   = profile1[:L].std()
    s2   = profile2[:L].std()
    if s1 < 1e-6 and s2 < 1e-6:
        return 1.0
    mn   = min(s1, s2)
    mx   = max(s1, s2)
    return float(np.clip(mn / max(mx, 1e-9), 0.0, 1.0))


# ─── analyze_seam ─────────────────────────────────────────────────────────────

def analyze_seam(
    img1:      np.ndarray,
    img2:      np.ndarray,
    idx1:      int   = 0,
    idx2:      int   = 1,
    side1:     int   = 2,
    side2:     int   = 0,
    border_px: int   = 8,
    weights:   Optional[Tuple[float, float, float]] = None,
) -> SeamAnalysis:
    """
    Проводит полный анализ качества шва между двумя фрагментами.

    Args:
        img1:      Первый фрагмент.
        img2:      Второй фрагмент.
        idx1:      Индекс первого фрагмента.
        idx2:      Индекс второго фрагмента.
        side1:     Край первого фрагмента (0–3).
        side2:     Край второго фрагмента (0–3).
        border_px: Ширина полосы анализа.
        weights:   (w_brightness, w_gradient, w_texture). None → равные.

    Returns:
        SeamAnalysis с полными метриками.
    """
    p1, p2 = extract_seam_profiles(img1, img2, side1, side2, border_px)

    b_score = brightness_continuity(p1, p2)
    g_score = gradient_continuity(p1, p2)
    t_score = texture_continuity(p1, p2)

    if weights is None:
        wb, wg, wt = 1.0 / 3, 1.0 / 3, 1.0 / 3
    else:
        wb, wg, wt = weights
        s = wb + wg + wt
        if s > 1e-9:
            wb /= s; wg /= s; wt /= s

    quality = float(wb * b_score + wg * g_score + wt * t_score)

    return SeamAnalysis(
        idx1=idx1, idx2=idx2,
        side1=side1, side2=side2,
        brightness_score=b_score,
        gradient_score=g_score,
        texture_score=t_score,
        quality_score=float(np.clip(quality, 0.0, 1.0)),
        profile_length=len(p1),
        params={"border_px": border_px, "weights": weights},
    )


# ─── score_seam_quality ───────────────────────────────────────────────────────

def score_seam_quality(analysis: SeamAnalysis) -> float:
    """
    Возвращает scalar quality_score из SeamAnalysis.

    Args:
        analysis: Результат analyze_seam.

    Returns:
        quality_score ∈ [0.0, 1.0].
    """
    return float(np.clip(analysis.quality_score, 0.0, 1.0))


# ─── batch_analyze_seams ──────────────────────────────────────────────────────

def batch_analyze_seams(
    images:     List[np.ndarray],
    pairs:      List[Tuple[int, int]],
    side_pairs: Optional[List[Tuple[int, int]]] = None,
    border_px:  int = 8,
    weights:    Optional[Tuple[float, float, float]] = None,
) -> List[SeamAnalysis]:
    """
    Анализирует качество швов для списка пар фрагментов.

    Args:
        images:     Список изображений (BGR или grayscale).
        pairs:      Список пар индексов [(idx1, idx2), ...].
        side_pairs: [(side1, side2), ...] или None → (2, 0) для всех.
        border_px:  Ширина полосы анализа.
        weights:    Веса метрик.

    Returns:
        Список SeamAnalysis длиной len(pairs).
    """
    if not pairs:
        return []

    if side_pairs is None:
        side_pairs = [(2, 0)] * len(pairs)

    return [
        analyze_seam(
            images[i1], images[i2],
            idx1=i1, idx2=i2,
            side1=s1, side2=s2,
            border_px=border_px,
            weights=weights,
        )
        for (i1, i2), (s1, s2) in zip(pairs, side_pairs)
    ]
