"""
Попиксельное (патч-based) сопоставление краёв фрагментов.

Сравнивает узкие полосы пикселей вдоль краёв двух фрагментов,
используя скользящее окно и несколько метрик схожести:
нормированную кросс-корреляцию (NCC), сумму квадратов разностей (SSD)
и структурное подобие (SSIM-упрощённое).

Классы:
    PatchMatch — результат сопоставления одной пары краёв

Функции:
    extract_edge_strip      — вырезает полосу пикселей вдоль края
    ncc_score               — нормированная кросс-корреляция двух полос
    ssd_score               — нормированная SSD (→ [0, 1])
    ssim_score              — упрощённый SSIM для 1D/2D полос
    match_edge_strips       — сравнивает две полосы всеми метриками
    match_patch_pair        — полное сопоставление двух фрагментов по краю
    batch_patch_match       — пакетное сопоставление списка пар
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── PatchMatch ───────────────────────────────────────────────────────────────

@dataclass
class PatchMatch:
    """
    Результат патч-based сопоставления двух краёв.

    Attributes:
        idx1:        Индекс первого фрагмента.
        idx2:        Индекс второго фрагмента.
        side1:       Сторона первого фрагмента (0=верх,1=право,2=низ,3=лево).
        side2:       Сторона второго фрагмента.
        ncc:         Нормированная кросс-корреляция ∈ [-1, 1].
        ssd:         Нормированная SSD → совместимость ∈ [0, 1].
        ssim:        Упрощённый SSIM ∈ [0, 1].
        total_score: Взвешенное среднее ncc_norm, ssd, ssim.
        params:      Параметры сопоставления.
    """
    idx1:        int
    idx2:        int
    side1:       int
    side2:       int
    ncc:         float
    ssd:         float
    ssim:        float
    total_score: float
    params:      Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"PatchMatch(idx1={self.idx1}, idx2={self.idx2}, "
                f"side1={self.side1}, side2={self.side2}, "
                f"total={self.total_score:.3f})")


# ─── extract_edge_strip ───────────────────────────────────────────────────────

def extract_edge_strip(
    img:       np.ndarray,
    side:      int,
    border_px: int = 10,
) -> np.ndarray:
    """
    Вырезает полосу пикселей вдоль указанного края изображения.

    Args:
        img:       Grayscale или BGR изображение uint8.
        side:      0 = верхний, 1 = правый, 2 = нижний, 3 = левый.
        border_px: Ширина полосы в пикселях.

    Returns:
        Массив float32, форма (border_px, W) для гор. краёв
        или (H, border_px) для вертикальных.

    Raises:
        ValueError: Если side не в [0, 3].
    """
    if side not in (0, 1, 2, 3):
        raise ValueError(f"side must be 0, 1, 2, or 3; got {side!r}.")

    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bp   = max(1, border_px)

    if side == 0:
        strip = gray[:bp, :]
    elif side == 1:
        strip = gray[:, -bp:]
    elif side == 2:
        strip = gray[-bp:, :]
    else:  # side == 3
        strip = gray[:, :bp]

    return strip.astype(np.float32)


# ─── ncc_score ────────────────────────────────────────────────────────────────

def ncc_score(strip1: np.ndarray, strip2: np.ndarray) -> float:
    """
    Нормированная кросс-корреляция (NCC) двух полос одинакового размера.

    Возвращает 0.0, если одна из полос вырождена (std < 1e-6).

    Args:
        strip1: float32-массив.
        strip2: float32-массив той же формы.

    Returns:
        NCC ∈ [-1.0, 1.0].
    """
    a = strip1.ravel().astype(np.float64)
    b = strip2.ravel().astype(np.float64)
    if a.size == 0:
        return 0.0
    sa, sb = a.std(), b.std()
    if sa < 1e-6 or sb < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ─── ssd_score ────────────────────────────────────────────────────────────────

def ssd_score(strip1: np.ndarray, strip2: np.ndarray) -> float:
    """
    Нормированная SSD двух полос → совместимость ∈ [0, 1].

    score = 1 / (1 + mean_sq_diff / 255²).

    Args:
        strip1: float32-массив.
        strip2: float32-массив той же формы.

    Returns:
        SSD-совместимость ∈ (0, 1].
    """
    if strip1.size == 0:
        return 0.0
    diff = (strip1.astype(np.float64) - strip2.astype(np.float64))
    msd  = float(np.mean(diff ** 2))
    return float(1.0 / (1.0 + msd / (255.0 ** 2)))


# ─── ssim_score ───────────────────────────────────────────────────────────────

def ssim_score(
    strip1: np.ndarray,
    strip2: np.ndarray,
    c1:     float = 6.5025,
    c2:     float = 58.5225,
) -> float:
    """
    Упрощённый SSIM для двух 1D/2D полос (без скользящего окна).

    SSIM = (2μ₁μ₂+C₁)(2σ₁₂+C₂) / ((μ₁²+μ₂²+C₁)(σ₁²+σ₂²+C₂))

    Args:
        strip1: float32-массив.
        strip2: float32-массив той же формы.
        c1:     Константа стабилизации (по умолчанию (0.01*255)²).
        c2:     Константа стабилизации (по умолчанию (0.03*255)²).

    Returns:
        SSIM ∈ [-1.0, 1.0], зажатый в [0.0, 1.0].
    """
    if strip1.size == 0:
        return 0.0
    a  = strip1.astype(np.float64).ravel()
    b  = strip2.astype(np.float64).ravel()
    mu1, mu2   = a.mean(), b.mean()
    s1, s2     = a.var(), b.var()
    s12        = float(np.mean((a - mu1) * (b - mu2)))
    numerator   = (2.0 * mu1 * mu2 + c1) * (2.0 * s12 + c2)
    denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2)
    raw = numerator / denominator if abs(denominator) > 1e-12 else 0.0
    return float(np.clip(raw, 0.0, 1.0))


# ─── match_edge_strips ────────────────────────────────────────────────────────

def match_edge_strips(
    strip1:  np.ndarray,
    strip2:  np.ndarray,
    weights: Optional[Tuple[float, float, float]] = None,
) -> Tuple[float, float, float, float]:
    """
    Сравнивает две полосы тремя метриками и возвращает взвешенное среднее.

    Args:
        strip1:  float32-массив (полоса края 1).
        strip2:  float32-массив (полоса края 2; должна иметь ту же форму).
        weights: (w_ncc, w_ssd, w_ssim). None → (1/3, 1/3, 1/3).

    Returns:
        Кортеж (ncc, ssd, ssim, total_score).
    """
    # Привести к одинаковому размеру (обрезать по меньшему)
    if strip1.shape != strip2.shape:
        rows = min(strip1.shape[0], strip2.shape[0])
        cols = min(strip1.shape[1], strip2.shape[1])
        strip1 = strip1[:rows, :cols]
        strip2 = strip2[:rows, :cols]

    ncc  = ncc_score(strip1, strip2)
    ssd  = ssd_score(strip1, strip2)
    ssim = ssim_score(strip1, strip2)

    # Нормируем NCC в [0, 1]
    ncc_norm = float(np.clip((ncc + 1.0) / 2.0, 0.0, 1.0))

    if weights is None:
        w_ncc, w_ssd, w_ssim = 1.0 / 3, 1.0 / 3, 1.0 / 3
    else:
        w_ncc, w_ssd, w_ssim = weights
        s = w_ncc + w_ssd + w_ssim
        if s > 1e-9:
            w_ncc /= s; w_ssd /= s; w_ssim /= s

    total = w_ncc * ncc_norm + w_ssd * ssd + w_ssim * ssim
    return ncc, ssd, ssim, float(total)


# ─── match_patch_pair ─────────────────────────────────────────────────────────

def match_patch_pair(
    img1:      np.ndarray,
    img2:      np.ndarray,
    idx1:      int   = 0,
    idx2:      int   = 1,
    side1:     int   = 2,
    side2:     int   = 0,
    border_px: int   = 10,
    weights:   Optional[Tuple[float, float, float]] = None,
) -> PatchMatch:
    """
    Сопоставляет два фрагмента по указанным краям.

    Args:
        img1:      Первый фрагмент (BGR или grayscale).
        img2:      Второй фрагмент (BGR или grayscale).
        idx1:      Индекс первого фрагмента.
        idx2:      Индекс второго фрагмента.
        side1:     Край первого фрагмента (0-3).
        side2:     Край второго фрагмента (0-3).
        border_px: Ширина полосы сравнения.
        weights:   Веса метрик (ncc, ssd, ssim). None → равные.

    Returns:
        PatchMatch с полными метриками.
    """
    s1 = extract_edge_strip(img1, side1, border_px)
    s2 = extract_edge_strip(img2, side2, border_px)
    ncc, ssd, ssim, total = match_edge_strips(s1, s2, weights)

    return PatchMatch(
        idx1=idx1, idx2=idx2,
        side1=side1, side2=side2,
        ncc=ncc, ssd=ssd, ssim=ssim,
        total_score=total,
        params={
            "border_px": border_px,
            "weights": weights,
        },
    )


# ─── batch_patch_match ────────────────────────────────────────────────────────

def batch_patch_match(
    images:     List[np.ndarray],
    pairs:      List[Tuple[int, int]],
    side_pairs: Optional[List[Tuple[int, int]]] = None,
    border_px:  int = 10,
    weights:    Optional[Tuple[float, float, float]] = None,
) -> List[PatchMatch]:
    """
    Пакетное сопоставление списка пар изображений.

    Args:
        images:     Список изображений (BGR или grayscale).
        pairs:      Список пар индексов [(idx1, idx2), ...].
        side_pairs: [(side1, side2), ...] или None → (2, 0) для всех.
        border_px:  Ширина полосы сравнения.
        weights:    Веса метрик.

    Returns:
        Список PatchMatch длиной len(pairs).
    """
    if not pairs:
        return []

    if side_pairs is None:
        side_pairs = [(2, 0)] * len(pairs)

    results: List[PatchMatch] = []
    for (i1, i2), (s1, s2) in zip(pairs, side_pairs):
        pm = match_patch_pair(
            images[i1], images[i2],
            idx1=i1, idx2=i2,
            side1=s1, side2=s2,
            border_px=border_px,
            weights=weights,
        )
        results.append(pm)

    return results
