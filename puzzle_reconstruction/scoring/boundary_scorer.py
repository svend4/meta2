"""Оценка совместимости фрагментов по граничным пикселям.

Модуль вычисляет меры совместимости двух соседних краёв фрагментов:
разность интенсивностей, градиентная согласованность, цветовая совместимость,
текстурное сходство граничных полос, а также агрегированную оценку.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── EdgeSide (перечисление сторон) ───────────────────────────────────────────

class BoundarySide(str, Enum):
    """Сторона фрагмента для оценки границы."""
    TOP    = "top"
    BOTTOM = "bottom"
    LEFT   = "left"
    RIGHT  = "right"

    def opposite(self) -> "BoundarySide":
        """Противоположная сторона."""
        opp = {
            BoundarySide.TOP: BoundarySide.BOTTOM,
            BoundarySide.BOTTOM: BoundarySide.TOP,
            BoundarySide.LEFT: BoundarySide.RIGHT,
            BoundarySide.RIGHT: BoundarySide.LEFT,
        }
        return opp[self]


# ─── BoundaryScore ────────────────────────────────────────────────────────────

@dataclass
class BoundaryScore:
    """Оценка совместимости двух граничных полос.

    Атрибуты:
        intensity_diff:  Средняя абсолютная разность интенсивности в [0, 1].
        gradient_score:  Согласованность градиентов в [0, 1].
        color_score:     Цветовая совместимость в [0, 1].
        aggregate:       Агрегированная оценка в [0, 1] (выше → лучше).
        side1:           Сторона первого фрагмента.
        side2:           Сторона второго фрагмента.
    """

    intensity_diff: float
    gradient_score: float
    color_score: float
    aggregate: float
    side1: BoundarySide = BoundarySide.RIGHT
    side2: BoundarySide = BoundarySide.LEFT

    def __post_init__(self) -> None:
        for name, val in [
            ("intensity_diff", self.intensity_diff),
            ("gradient_score", self.gradient_score),
            ("color_score", self.color_score),
            ("aggregate", self.aggregate),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"{name} должен быть в [0, 1], получено {val}"
                )


# ─── ScoringConfig ────────────────────────────────────────────────────────────

@dataclass
class ScoringConfig:
    """Параметры оценки граничной совместимости.

    Атрибуты:
        strip_width:       Ширина граничной полосы в пикселях (>= 1).
        w_intensity:       Вес разности интенсивности (>= 0).
        w_gradient:        Вес градиентной согласованности (>= 0).
        w_color:           Вес цветовой совместимости (>= 0).
        normalize_weights: Нормировать веса (сумма → 1).
    """

    strip_width: int = 4
    w_intensity: float = 0.4
    w_gradient: float = 0.3
    w_color: float = 0.3
    normalize_weights: bool = True

    def __post_init__(self) -> None:
        if self.strip_width < 1:
            raise ValueError(
                f"strip_width должен быть >= 1, получено {self.strip_width}"
            )
        for name, val in [
            ("w_intensity", self.w_intensity),
            ("w_gradient", self.w_gradient),
            ("w_color", self.w_color),
        ]:
            if val < 0.0:
                raise ValueError(f"{name} должен быть >= 0, получено {val}")

    @property
    def weights(self) -> Tuple[float, float, float]:
        """Нормированные (или исходные) веса (w_i, w_g, w_c)."""
        wi, wg, wc = self.w_intensity, self.w_gradient, self.w_color
        total = wi + wg + wc
        if self.normalize_weights and total > 1e-12:
            return wi / total, wg / total, wc / total
        return wi, wg, wc


# ─── _extract_strip ────────────────────────────────────────────────────────────

def _extract_strip(img: np.ndarray, side: BoundarySide, width: int) -> np.ndarray:
    """Извлечь граничную полосу шириной `width` пикселей с заданной стороны."""
    h, w = img.shape[:2]
    width = min(width, min(h, w))
    if side == BoundarySide.TOP:
        return img[:width, :]
    if side == BoundarySide.BOTTOM:
        return img[h - width:, :]
    if side == BoundarySide.LEFT:
        return img[:, :width]
    # RIGHT
    return img[:, w - width:]


def _to_float(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float64) / 255.0


# ─── intensity_compatibility ─────────────────────────────────────────────────

def intensity_compatibility(
    strip1: np.ndarray, strip2: np.ndarray
) -> float:
    """Средняя абсолютная разность интенсивностей (нормирована к [0, 1]).

    Возвращает 1 − MAE (выше → лучше).

    Аргументы:
        strip1: Граничная полоса первого фрагмента.
        strip2: Граничная полоса второго фрагмента (та же форма).

    Возвращает:
        float в [0, 1].

    Исключения:
        ValueError: Если формы не совпадают.
    """
    s1 = _to_float(np.asarray(strip1))
    s2 = _to_float(np.asarray(strip2))
    if s1.shape != s2.shape:
        raise ValueError(
            f"Формы полос не совпадают: {s1.shape} != {s2.shape}"
        )
    mae = float(np.abs(s1 - s2).mean())
    return float(np.clip(1.0 - mae, 0.0, 1.0))


# ─── gradient_compatibility ──────────────────────────────────────────────────

def gradient_compatibility(
    strip1: np.ndarray, strip2: np.ndarray
) -> float:
    """Согласованность градиентов на граничной полосе.

    Вычисляет нормализованный ZNCC между картами краёв Собеля.

    Аргументы:
        strip1: Полоса первого фрагмента (uint8, 2-D или 3-D).
        strip2: Полоса второго фрагмента.

    Возвращает:
        float в [0, 1] (1 = идеальное совпадение).
    """
    def _edge_map(arr: np.ndarray) -> np.ndarray:
        gray = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1)
        return np.sqrt(gx ** 2 + gy ** 2)

    e1 = _edge_map(np.asarray(strip1, dtype=np.uint8)).ravel()
    e2 = _edge_map(np.asarray(strip2, dtype=np.uint8)).ravel()

    if e1.size == 0:
        return 1.0

    mu1, mu2 = e1.mean(), e2.mean()
    std1 = e1.std() + 1e-8
    std2 = e2.std() + 1e-8
    zncc = float(((e1 - mu1) * (e2 - mu2)).mean() / (std1 * std2))
    return float(np.clip((zncc + 1.0) / 2.0, 0.0, 1.0))


# ─── color_compatibility ─────────────────────────────────────────────────────

def color_compatibility(
    strip1: np.ndarray, strip2: np.ndarray
) -> float:
    """Цветовая совместимость граничных полос (по гистограммам LAB).

    Аргументы:
        strip1: Полоса первого фрагмента (uint8, 2-D или 3-D).
        strip2: Полоса второго фрагмента.

    Возвращает:
        float в [0, 1] (пересечение нормированных гистограмм).
    """
    def _hist_lab(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.uint8)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
        hists = []
        for c in range(3):
            h = cv2.calcHist([lab], [c], None, [32], [0, 256])
            h = h.ravel().astype(np.float64)
            s = h.sum()
            hists.append(h / s if s > 0 else h)
        return np.concatenate(hists)

    h1 = _hist_lab(strip1)
    h2 = _hist_lab(strip2)
    intersection = float(np.minimum(h1, h2).sum())
    return float(np.clip(intersection, 0.0, 1.0))


# ─── score_boundary ───────────────────────────────────────────────────────────

def score_boundary(
    img1: np.ndarray,
    img2: np.ndarray,
    side1: BoundarySide = BoundarySide.RIGHT,
    side2: Optional[BoundarySide] = None,
    cfg: Optional[ScoringConfig] = None,
) -> BoundaryScore:
    """Оценить совместимость двух соседних фрагментов по граничным полосам.

    Аргументы:
        img1:  Первый фрагмент (uint8, 2-D или 3-D).
        img2:  Второй фрагмент.
        side1: Сторона первого фрагмента.
        side2: Сторона второго фрагмента (None → противоположная side1).
        cfg:   Параметры оценки (None → ScoringConfig()).

    Возвращает:
        BoundaryScore с компонентными и агрегированной оценками.
    """
    if cfg is None:
        cfg = ScoringConfig()
    if side2 is None:
        side2 = side1.opposite()

    strip1 = _extract_strip(img1, side1, cfg.strip_width)
    strip2 = _extract_strip(img2, side2, cfg.strip_width)

    # Приводим размеры к одинаковым (crop по меньшей)
    min_rows = min(strip1.shape[0], strip2.shape[0])
    min_cols = min(strip1.shape[1], strip2.shape[1])
    s1 = strip1[:min_rows, :min_cols]
    s2 = strip2[:min_rows, :min_cols]

    i_score = intensity_compatibility(s1, s2)
    g_score = gradient_compatibility(s1, s2)
    c_score = color_compatibility(s1, s2)

    wi, wg, wc = cfg.weights
    agg = float(np.clip(wi * i_score + wg * g_score + wc * c_score, 0.0, 1.0))

    return BoundaryScore(
        intensity_diff=float(np.clip(i_score, 0.0, 1.0)),
        gradient_score=float(np.clip(g_score, 0.0, 1.0)),
        color_score=float(np.clip(c_score, 0.0, 1.0)),
        aggregate=agg,
        side1=side1,
        side2=side2,
    )


# ─── score_matrix ─────────────────────────────────────────────────────────────

def score_matrix(
    images: List[np.ndarray],
    side1: BoundarySide = BoundarySide.RIGHT,
    side2: Optional[BoundarySide] = None,
    cfg: Optional[ScoringConfig] = None,
) -> np.ndarray:
    """Вычислить матрицу граничных оценок для всех пар фрагментов.

    Аргументы:
        images: Список изображений (uint8).
        side1:  Сторона источника.
        side2:  Сторона цели (None → противоположная side1).
        cfg:    Параметры оценки.

    Возвращает:
        Матрица (N×N, float64): result[i, j] = aggregate score(i→j).
        Диагональ = 0.
    """
    n = len(images)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                bs = score_boundary(images[i], images[j], side1, side2, cfg)
                mat[i, j] = bs.aggregate
    return mat


# ─── batch_score_boundaries ──────────────────────────────────────────────────

def batch_score_boundaries(
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    side1: BoundarySide = BoundarySide.RIGHT,
    side2: Optional[BoundarySide] = None,
    cfg: Optional[ScoringConfig] = None,
) -> List[BoundaryScore]:
    """Оценить список пар (img1, img2).

    Аргументы:
        pairs: Список кортежей (img1, img2).
        side1: Сторона первого фрагмента.
        side2: Сторона второго фрагмента.
        cfg:   Параметры оценки.

    Возвращает:
        Список BoundaryScore, по одному на каждую пару.
    """
    return [score_boundary(a, b, side1, side2, cfg) for a, b in pairs]
