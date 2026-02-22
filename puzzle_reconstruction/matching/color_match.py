"""
Цветовое сопоставление фрагментов документов.

Сравнивает фрагменты на основе цветовых гистограмм, цветовых моментов
и профилей цвета вдоль граничных полос.

Классы:
    ColorMatchResult — результат цветового сопоставления двух изображений

Функции:
    compute_color_histogram   — нормированная гистограмма (HSV/BGR/gray)
    histogram_distance        — расстояние между гистограммами (chi2/bhatt/corr)
    compute_color_moments     — первые 3 статистических момента по каналам
    moments_distance          — L2-расстояние по цветовым моментам
    edge_color_profile        — профиль цвета вдоль краевой полосы фрагмента
    color_match_pair          — итоговое сопоставление двух изображений
    color_compatibility_matrix — матрица совместимости N × N
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── ColorMatchResult ─────────────────────────────────────────────────────────

@dataclass
class ColorMatchResult:
    """
    Результат цветового сопоставления двух изображений.

    Attributes:
        score:          Итоговая оценка совместимости ∈ [0, 1]
                        (1 = идеально совпадают).
        hist_score:     Оценка по гистограммам ∈ [0, 1].
        moment_score:   Оценка по цветовым моментам ∈ [0, 1].
        profile_score:  Оценка по граничным профилям ∈ [0, 1].
        method:         Пространство цветов / метод гистограммы.
        params:         Дополнительные параметры.
    """
    score:         float
    hist_score:    float
    moment_score:  float
    profile_score: float
    method:        str
    params:        Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"ColorMatchResult(score={self.score:.4f}, "
                f"hist={self.hist_score:.3f}, "
                f"moment={self.moment_score:.3f}, "
                f"profile={self.profile_score:.3f})")


# ─── Вспомогательные ──────────────────────────────────────────────────────────

def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


_COLORSPACE_CONV = {
    "hsv": cv2.COLOR_BGR2HSV,
    "lab": cv2.COLOR_BGR2LAB,
    "yuv": cv2.COLOR_BGR2YUV,
}


# ─── compute_color_histogram ──────────────────────────────────────────────────

def compute_color_histogram(img:        np.ndarray,
                              bins:       int = 32,
                              colorspace: str = "hsv",
                              mask:       Optional[np.ndarray] = None,
                              ) -> np.ndarray:
    """
    Вычисляет нормированную гистограмму изображения.

    Для grayscale — 1D гистограмма.
    Для BGR — 3-канальная конкатенация гистограмм в указанном цветовом
    пространстве (hsv / lab / yuv / bgr).

    Args:
        img:        BGR или grayscale изображение.
        bins:       Число бинов на канал.
        colorspace: 'hsv' | 'lab' | 'yuv' | 'bgr'.
        mask:       Бинарная маска (uint8, 255 = учитывать пиксель).

    Returns:
        Нормированный вектор гистограммы (float32).
    """
    if img.ndim == 2:
        hist = cv2.calcHist([img], [0], mask, [bins], [0, 256])
        hist = hist.flatten().astype(np.float32)
        total = hist.sum()
        return hist / max(1.0, total)

    bgr = _to_bgr(img)
    if colorspace in _COLORSPACE_CONV:
        converted = cv2.cvtColor(bgr, _COLORSPACE_CONV[colorspace])
    elif colorspace == "bgr":
        converted = bgr
    else:
        raise ValueError(
            f"Неизвестное цветовое пространство: {colorspace!r}. "
            f"Допустимые: hsv, lab, yuv, bgr."
        )

    hists = []
    for ch in range(3):
        h = cv2.calcHist([converted], [ch], mask, [bins], [0, 256])
        hists.append(h.flatten().astype(np.float32))

    combined = np.concatenate(hists)
    total = combined.sum()
    return combined / max(1.0, total)


# ─── histogram_distance ───────────────────────────────────────────────────────

_HIST_METRICS = {
    "chi2":    cv2.HISTCMP_CHISQR_ALT,
    "bhatt":   cv2.HISTCMP_BHATTACHARYYA,
    "corr":    cv2.HISTCMP_CORREL,
    "inter":   cv2.HISTCMP_INTERSECT,
}


def histogram_distance(hist1: np.ndarray,
                        hist2: np.ndarray,
                        metric: str = "bhatt") -> float:
    """
    Вычисляет расстояние (или сходство) между двумя гистограммами.

    Args:
        hist1:  Нормированная гистограмма.
        hist2:  Нормированная гистограмма.
        metric: 'chi2' | 'bhatt' | 'corr' | 'inter'.
                chi2/bhatt → расстояние (меньше = лучше), конвертируется в [0,1].
                corr → сходство (больше = лучше, −1…1), конвертируется в [0,1].
                inter → пересечение (больше = лучше), конвертируется в [0,1].

    Returns:
        Нормированное сходство ∈ [0, 1].

    Raises:
        ValueError: Если metric не из допустимого набора.
    """
    if metric not in _HIST_METRICS:
        raise ValueError(
            f"Неизвестная метрика: {metric!r}. Допустимые: {list(_HIST_METRICS)}"
        )

    h1 = hist1.astype(np.float32).reshape(-1)
    h2 = hist2.astype(np.float32).reshape(-1)

    raw = cv2.compareHist(h1, h2, _HIST_METRICS[metric])

    if metric == "chi2":
        # chi2 ∈ [0, +∞), 0=идеал → similarity = exp(-raw/2)
        similarity = float(np.exp(-0.5 * max(0.0, raw)))
    elif metric == "bhatt":
        # Bhattacharyya ∈ [0, 1], 0=идеал
        similarity = float(1.0 - np.clip(raw, 0.0, 1.0))
    elif metric == "corr":
        # Корреляция ∈ [−1, 1]
        similarity = float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))
    else:  # inter
        # Пересечение ∈ [0, 1] для нормированных гистограмм
        similarity = float(np.clip(raw, 0.0, 1.0))

    return similarity


# ─── compute_color_moments ────────────────────────────────────────────────────

def compute_color_moments(img: np.ndarray,
                            colorspace: str = "bgr") -> np.ndarray:
    """
    Вычисляет цветовые моменты: среднее, стандартное отклонение,
    ненормированную асимметрию по каждому каналу.

    Args:
        img:        BGR или grayscale изображение.
        colorspace: 'bgr' | 'hsv' | 'lab' | 'yuv' (для BGR-входа).

    Returns:
        Вектор моментов длиной 3 (gray) или 9 (3-канальный), float32.
        Порядок: [μ₁, σ₁, skew₁, μ₂, σ₂, skew₂, μ₃, σ₃, skew₃].
    """
    if img.ndim == 2:
        ch = img.astype(np.float32)
        mu   = float(ch.mean())
        sig  = float(ch.std())
        cube = float(np.mean((ch - mu) ** 3))
        skew = float(np.cbrt(cube)) if cube >= 0 else -float(np.cbrt(-cube))
        return np.array([mu, sig, skew], dtype=np.float32)

    bgr = _to_bgr(img)
    if colorspace in _COLORSPACE_CONV:
        converted = cv2.cvtColor(bgr, _COLORSPACE_CONV[colorspace])
    elif colorspace == "bgr":
        converted = bgr
    else:
        raise ValueError(f"Неизвестное цветовое пространство: {colorspace!r}.")

    moments = []
    for c in range(3):
        ch = converted[:, :, c].astype(np.float32)
        mu  = float(ch.mean())
        sig = float(ch.std())
        cube = float(np.mean((ch - mu) ** 3))
        skew = float(np.cbrt(cube)) if cube >= 0 else -float(np.cbrt(-cube))
        moments.extend([mu, sig, skew])

    return np.array(moments, dtype=np.float32)


# ─── moments_distance ─────────────────────────────────────────────────────────

def moments_distance(m1: np.ndarray,
                      m2: np.ndarray) -> float:
    """
    Нормированное L2-расстояние между векторами цветовых моментов.

    Returns:
        Сходство ∈ [0, 1]: 1 = идентичные, 0 = максимально разные.
    """
    if m1.size == 0 or m2.size == 0:
        return 0.0
    diff = (m1.astype(np.float64) - m2.astype(np.float64))
    dist = float(np.linalg.norm(diff))
    # Нормировка: диапазон каждого μ ≈ 255, σ ≈ 128, skew ≈ 128
    n_channels = m1.size // 3 if m1.size >= 3 else 1
    scale = np.sqrt(n_channels) * 200.0
    similarity = float(np.exp(-dist / max(1.0, scale)))
    return float(np.clip(similarity, 0.0, 1.0))


# ─── edge_color_profile ───────────────────────────────────────────────────────

_SIDE_NAMES = {0: "top", 1: "right", 2: "bottom", 3: "left"}
_BORDER_FRAC = 0.10


def edge_color_profile(img:  np.ndarray,
                        side: int,
                        bins: int = 16,
                        border_frac: float = _BORDER_FRAC,
                        ) -> np.ndarray:
    """
    Вычисляет цветовой профиль вдоль заданной стороны фрагмента.

    Вырезает полосу шириной border_frac × размер, возвращает
    нормированную гистограмму.

    Args:
        img:         BGR или grayscale изображение.
        side:        0=top, 1=right, 2=bottom, 3=left.
        bins:        Число бинов.
        border_frac: Доля размера для ширины полосы.

    Returns:
        Нормированный вектор гистограммы (float32).
    """
    h, w = img.shape[:2]
    bh = max(1, int(h * border_frac))
    bw = max(1, int(w * border_frac))

    if side == 0:
        strip = img[:bh, :]
    elif side == 1:
        strip = img[:, w - bw:]
    elif side == 2:
        strip = img[h - bh:, :]
    else:
        strip = img[:, :bw]

    return compute_color_histogram(strip, bins=bins, colorspace="bgr")


# ─── color_match_pair ─────────────────────────────────────────────────────────

def color_match_pair(img1:       np.ndarray,
                      img2:       np.ndarray,
                      colorspace: str   = "hsv",
                      metric:     str   = "bhatt",
                      bins:       int   = 32,
                      side1:      int   = 1,
                      side2:      int   = 3,
                      w_hist:     float = 0.4,
                      w_moment:   float = 0.3,
                      w_profile:  float = 0.3,
                      ) -> ColorMatchResult:
    """
    Полное цветовое сопоставление двух изображений-фрагментов.

    Объединяет:
    1. Гистограммное сходство (глобальное).
    2. Сходство по цветовым моментам.
    3. Сходство граничных профилей (side1 img1 vs side2 img2).

    Args:
        img1, img2:  Изображения для сравнения (BGR или gray).
        colorspace:  Пространство для гистограммы ('hsv'/'lab'/'yuv'/'bgr').
        metric:      Метрика гистограммы ('bhatt'/'chi2'/'corr'/'inter').
        bins:        Число бинов гистограммы.
        side1:       Сторона img1 для профиля (0=top,1=right,2=bot,3=left).
        side2:       Сторона img2 для профиля.
        w_hist:      Вес гистограммы.
        w_moment:    Вес цветовых моментов.
        w_profile:   Вес граничных профилей.

    Returns:
        ColorMatchResult.
    """
    # Гистограммное сходство
    h1 = compute_color_histogram(img1, bins=bins, colorspace=colorspace)
    h2 = compute_color_histogram(img2, bins=bins, colorspace=colorspace)
    hist_score = histogram_distance(h1, h2, metric=metric)

    # Цветовые моменты
    m1 = compute_color_moments(img1)
    m2 = compute_color_moments(img2)
    moment_score = moments_distance(m1, m2)

    # Граничные профили
    p1 = edge_color_profile(img1, side=side1, bins=bins)
    p2 = edge_color_profile(img2, side=side2, bins=bins)
    profile_score = histogram_distance(p1, p2, metric=metric)

    # Взвешенная сумма
    total = w_hist + w_moment + w_profile
    if total <= 0.0:
        score = 0.0
    else:
        score = (w_hist * hist_score
                 + w_moment * moment_score
                 + w_profile * profile_score) / total

    return ColorMatchResult(
        score=float(np.clip(score, 0.0, 1.0)),
        hist_score=hist_score,
        moment_score=moment_score,
        profile_score=profile_score,
        method=f"{colorspace}_{metric}",
        params={
            "bins": bins, "side1": side1, "side2": side2,
            "w_hist": w_hist, "w_moment": w_moment, "w_profile": w_profile,
        },
    )


# ─── color_compatibility_matrix ───────────────────────────────────────────────

def color_compatibility_matrix(images:     List[np.ndarray],
                                 colorspace: str = "hsv",
                                 metric:     str = "bhatt",
                                 bins:       int = 32,
                                 ) -> np.ndarray:
    """
    Строит матрицу цветовой совместимости N × N.

    Элемент [i, j] — итоговая оценка color_match_pair(images[i], images[j]).
    Диагональ = 1.0.

    Args:
        images:     Список изображений.
        colorspace: Пространство для гистограммы.
        metric:     Метрика гистограммы.
        bins:       Число бинов.

    Returns:
        Симметричная матрица float32 формата (N, N).
    """
    n = len(images)
    mat = np.eye(n, dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            res = color_match_pair(
                images[i], images[j],
                colorspace=colorspace,
                metric=metric,
                bins=bins,
            )
            mat[i, j] = res.score
            mat[j, i] = res.score

    return mat
