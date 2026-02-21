"""
Текстурное сопоставление краёв фрагментов.

Использует Local Binary Patterns (LBP), банк Gabor-фильтров и гистограммы
ориентаций градиентов для оценки совместимости текстур.

Классы:
    TextureMatchResult — результат сравнения текстур двух краёв

Функции:
    compute_lbp_histogram     — нормированная гистограмма LBP
    lbp_distance              — схожесть двух LBP-гистограмм ∈ [0,1]
    compute_gabor_features    — вектор признаков банка Gabor-фильтров
    gabor_distance            — нормированное L2 → схожесть ∈ [0,1]
    gradient_orientation_hist — гистограмма ориентаций градиентов
    texture_match_pair        — итоговая взвешенная оценка совместимости
    texture_compatibility_matrix — матрица совместимости N×N
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np


# ─── TextureMatchResult ───────────────────────────────────────────────────────

@dataclass
class TextureMatchResult:
    """
    Результат текстурного сравнения двух краёв.

    Attributes:
        score:          Итоговая взвешенная оценка совместимости ∈ [0,1].
        lbp_score:      Вклад LBP-гистограммы ∈ [0,1].
        gabor_score:    Вклад Gabor-фильтров ∈ [0,1].
        gradient_score: Вклад гистограммы ориентаций градиентов ∈ [0,1].
        method:         Идентификатор метода (всегда 'texture').
        params:         Словарь параметров.
    """
    score:          float
    lbp_score:      float
    gabor_score:    float
    gradient_score: float
    method:         str = "texture"
    params:         Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"TextureMatchResult(score={self.score:.3f}, "
                f"lbp={self.lbp_score:.3f}, gabor={self.gabor_score:.3f}, "
                f"grad={self.gradient_score:.3f})")


# ─── Вспомогательные ──────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _extract_border_strip(img: np.ndarray, side: int,
                           border_frac: float = 0.10) -> np.ndarray:
    """Извлекает полосу вдоль заданной стороны (0=верх,1=право,2=низ,3=лево)."""
    h, w = img.shape[:2]
    n = max(1, int(min(h, w) * border_frac))
    if side == 0:
        return img[:n, :]
    elif side == 1:
        return img[:, w - n:]
    elif side == 2:
        return img[h - n:, :]
    else:
        return img[:, :n]


# ─── compute_lbp_histogram ────────────────────────────────────────────────────

def compute_lbp_histogram(gray: np.ndarray,
                           radius: int = 1,
                           n_points: int = 8,
                           bins: int = 64) -> np.ndarray:
    """
    Нормированная гистограмма Local Binary Pattern.

    Каждый пиксель кодируется числом соседей (на окружности радиуса ``radius``),
    чья яркость ≥ яркости центра. Используется билинейная интерполяция.

    Args:
        gray:     Grayscale изображение (uint8 или float).
        radius:   Радиус окрестности.
        n_points: Число точек-соседей на окружности.
        bins:     Число бинов гистограммы.

    Returns:
        float32 массив длиной ``bins``, нормирован (сумма = 1 или нулевой).
    """
    src = gray.astype(np.float32)
    h, w = src.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    yy = np.arange(h, dtype=np.float32)[:, None] * np.ones((1, w), dtype=np.float32)
    xx = np.ones((h, 1), dtype=np.float32) * np.arange(w, dtype=np.float32)[None, :]

    for i in range(n_points):
        angle  = 2.0 * np.pi * i / n_points
        ny     = (yy + radius * np.sin(angle)).astype(np.float32)
        nx     = (xx + radius * np.cos(angle)).astype(np.float32)
        neighbor = cv2.remap(src, nx, ny, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        lbp += (neighbor >= src).astype(np.uint8)

    hist, _ = np.histogram(lbp, bins=bins, range=(0, n_points + 1))
    hist = hist.astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def lbp_distance(hist1: np.ndarray, hist2: np.ndarray,
                  metric: str = "chi2") -> float:
    """
    Схожесть двух LBP-гистограмм ∈ [0,1].

    Args:
        hist1, hist2: Нормированные LBP-гистограммы.
        metric:       'chi2' | 'bhatt' | 'corr'.

    Returns:
        float ∈ [0,1]; 1 = идентичные.

    Raises:
        ValueError: Если ``metric`` неизвестен.
    """
    h1 = hist1.reshape(-1, 1).astype(np.float32)
    h2 = hist2.reshape(-1, 1).astype(np.float32)

    if metric == "chi2":
        d = cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR)
        return float(np.exp(-0.5 * d))
    elif metric == "bhatt":
        d = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
        return float(1.0 - min(d, 1.0))
    elif metric == "corr":
        r = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        return float((r + 1.0) / 2.0)
    else:
        raise ValueError(f"Unknown metric: {metric!r}. Use 'chi2', 'bhatt', or 'corr'.")


# ─── compute_gabor_features ───────────────────────────────────────────────────

def compute_gabor_features(gray: np.ndarray,
                            frequencies: Tuple[float, ...] = (0.1, 0.3),
                            n_orientations: int = 4) -> np.ndarray:
    """
    Вектор признаков банка Gabor-фильтров.

    Для каждой пары (частота, ориентация) вычисляет среднее и σ отклика.
    Итого 2 × len(frequencies) × n_orientations элементов.

    Args:
        gray:           Grayscale изображение.
        frequencies:    Нормированные пространственные частоты (0, 0.5].
        n_orientations: Число равномерно распределённых ориентаций.

    Returns:
        float32 массив длиной 2 · |frequencies| · n_orientations.
    """
    src = gray.astype(np.float32)
    feats: List[float] = []
    ksize  = 31
    sigma  = 3.0
    gamma  = 0.5

    for freq in frequencies:
        lambda_ = 1.0 / max(freq, 1e-6)
        for k in range(n_orientations):
            theta   = np.pi * k / n_orientations
            kernel  = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lambda_, gamma, 0,
                ktype=cv2.CV_32F,
            )
            response = cv2.filter2D(src, cv2.CV_32F, kernel)
            feats.append(float(response.mean()))
            feats.append(float(response.std()))

    return np.array(feats, dtype=np.float32)


def gabor_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """
    Нормированное L2 → схожесть ∈ [0,1].

    Args:
        feat1, feat2: Gabor-вектора одинаковой длины.

    Returns:
        float ∈ [0,1]; 1 = идентичные.
    """
    if len(feat1) == 0 or len(feat2) == 0:
        return 0.0
    scale = max(float(np.linalg.norm(feat1)),
                float(np.linalg.norm(feat2)), 1e-6)
    d = float(np.linalg.norm(feat1 - feat2))
    return float(np.exp(-d / scale))


# ─── gradient_orientation_hist ────────────────────────────────────────────────

def gradient_orientation_hist(gray: np.ndarray, bins: int = 8) -> np.ndarray:
    """
    Нормированная гистограмма ориентаций градиентов (weighted by magnitude).

    Args:
        gray: Grayscale изображение.
        bins: Число бинов (равномерно в [0, π)).

    Returns:
        float32 массив длиной ``bins``.
    """
    gx  = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy  = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = np.arctan2(gy, gx) % np.pi      # ∈ [0, π)

    hist, _ = np.histogram(ang.flatten(), bins=bins, range=(0.0, np.pi),
                            weights=mag.flatten())
    hist = hist.astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def _orientation_hist_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """Chi2-схожесть двух гистограмм ориентаций → ∈ [0,1]."""
    a = h1.reshape(-1, 1).astype(np.float32)
    b = h2.reshape(-1, 1).astype(np.float32)
    d = cv2.compareHist(a, b, cv2.HISTCMP_CHISQR)
    return float(np.exp(-0.5 * d))


# ─── texture_match_pair ───────────────────────────────────────────────────────

_BORDER_FRAC = 0.10


def texture_match_pair(img1: np.ndarray,
                        img2: np.ndarray,
                        side1: int = 1,
                        side2: int = 3,
                        border_frac: float = _BORDER_FRAC,
                        w_lbp: float = 0.4,
                        w_gabor: float = 0.3,
                        w_gradient: float = 0.3) -> TextureMatchResult:
    """
    Оценивает совместимость текстур двух краёв.

    Args:
        img1, img2:   BGR или grayscale изображения.
        side1:        Сторона первого фрагмента (0=верх,1=право,2=низ,3=лево).
        side2:        Сторона второго фрагмента.
        border_frac:  Доля min(h,w) для ширины полосы края.
        w_lbp:        Вес LBP-компоненты.
        w_gabor:      Вес Gabor-компоненты.
        w_gradient:   Вес компоненты ориентаций градиентов.

    Returns:
        TextureMatchResult с score ∈ [0,1].
    """
    g1 = _to_gray(_extract_border_strip(img1, side1, border_frac))
    g2 = _to_gray(_extract_border_strip(img2, side2, border_frac))

    # LBP
    lbp1   = compute_lbp_histogram(g1)
    lbp2   = compute_lbp_histogram(g2)
    lbp_s  = lbp_distance(lbp1, lbp2, metric="chi2")

    # Gabor
    gab1   = compute_gabor_features(g1)
    gab2   = compute_gabor_features(g2)
    gab_s  = gabor_distance(gab1, gab2)

    # Gradient orientation
    oh1    = gradient_orientation_hist(g1)
    oh2    = gradient_orientation_hist(g2)
    grad_s = _orientation_hist_similarity(oh1, oh2)

    score = float(w_lbp * lbp_s + w_gabor * gab_s + w_gradient * grad_s)

    return TextureMatchResult(
        score=float(np.clip(score, 0.0, 1.0)),
        lbp_score=lbp_s,
        gabor_score=gab_s,
        gradient_score=grad_s,
        method="texture",
        params={
            "side1": side1, "side2": side2,
            "border_frac": border_frac,
            "w_lbp": w_lbp, "w_gabor": w_gabor, "w_gradient": w_gradient,
        },
    )


# ─── texture_compatibility_matrix ────────────────────────────────────────────

def texture_compatibility_matrix(images: List[np.ndarray],
                                  border_frac: float = _BORDER_FRAC,
                                  w_lbp: float = 0.4,
                                  w_gabor: float = 0.3,
                                  w_gradient: float = 0.3) -> np.ndarray:
    """
    Строит матрицу текстурной совместимости N×N.

    Оценка (i, j) — максимум по всем парам сторон (4×4).
    Диагональ = 1.0. Матрица симметричная.

    Args:
        images:      Список BGR или grayscale изображений.
        border_frac: Доля ширины/высоты для полосы края.
        w_lbp / w_gabor / w_gradient: Веса подметрик.

    Returns:
        float32 матрица N×N ∈ [0,1].
    """
    n = len(images)
    mat = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(mat, 1.0)

    for i in range(n):
        for j in range(i + 1, n):
            best = 0.0
            for si in range(4):
                for sj in range(4):
                    r = texture_match_pair(
                        images[i], images[j],
                        side1=si, side2=sj,
                        border_frac=border_frac,
                        w_lbp=w_lbp, w_gabor=w_gabor, w_gradient=w_gradient,
                    )
                    if r.score > best:
                        best = r.score
            mat[i, j] = best
            mat[j, i] = best

    return mat
