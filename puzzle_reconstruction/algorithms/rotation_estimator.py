"""
Оценка угла поворота для выравнивания пары фрагментов.

Экспортирует:
    RotationEstimate    — результат оценки угла поворота
    estimate_by_pca     — оценка через главные оси (PCA) контура
    estimate_by_moments — оценка через центральные моменты изображения
    estimate_by_gradient — оценка через доминирующую ориентацию градиентов
    refine_rotation     — уточнение угла минимизацией функции потерь (NCC)
    estimate_rotation_pair — согласование ориентаций двух фрагментов
    batch_estimate_rotations — оценка для списка фрагментов
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class RotationEstimate:
    """Результат оценки угла поворота фрагмента.

    Attributes:
        angle_deg:   Оценённый угол поворота в градусах
                     (положительный — по часовой стрелке).
        confidence:  Достоверность оценки [0, 1].
        method:      Метод оценки: ``"pca"``, ``"moments"``,
                     ``"gradient"``, ``"refine"``.
        params:      Дополнительные параметры и промежуточные результаты.
    """
    angle_deg: float
    confidence: float
    method: str
    params: Dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RotationEstimate(angle={self.angle_deg:.2f}°, "
            f"conf={self.confidence:.3f}, method={self.method!r})"
        )


# ─── Приватные утилиты ────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    """Привести изображение к одноканальному uint8."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.uint8)


def _normalize_angle(angle_deg: float) -> float:
    """Нормализовать угол в диапазон (-90, 90]."""
    a = float(angle_deg) % 180.0
    if a > 90.0:
        a -= 180.0
    return a


# ─── Публичные функции ────────────────────────────────────────────────────────

def estimate_by_pca(contour: np.ndarray) -> RotationEstimate:
    """Оценить угол поворота через главные оси контура (PCA).

    Первая главная компонента точек контура даёт направление
    доминирующей ориентации фрагмента.

    Args:
        contour: Массив точек контура (N, 2) float или int.

    Returns:
        :class:`RotationEstimate` с методом ``"pca"``.

    Raises:
        ValueError: Если контур содержит менее 2 точек.
    """
    pts = np.asarray(contour, dtype=np.float64)
    if pts.ndim == 3:
        pts = pts.reshape(-1, 2)
    if len(pts) < 2:
        raise ValueError(f"Contour must have >= 2 points, got {len(pts)}")

    centered = pts - pts.mean(axis=0)
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Largest eigenvalue → principal axis
    principal = eigenvectors[:, -1]
    angle_deg = float(np.degrees(np.arctan2(principal[1], principal[0])))
    angle_deg = _normalize_angle(angle_deg)

    ratio = float(eigenvalues[-1] / (eigenvalues[0] + 1e-12))
    confidence = float(np.clip(1.0 - 1.0 / (1.0 + ratio), 0.0, 1.0))

    return RotationEstimate(
        angle_deg=angle_deg,
        confidence=confidence,
        method="pca",
        params={"eigenvalues": eigenvalues.tolist()},
    )


def estimate_by_moments(img: np.ndarray) -> RotationEstimate:
    """Оценить угол поворота через центральные моменты изображения.

    Использует центральный момент второго порядка μ₁₁, μ₂₀, μ₀₂
    для нахождения главной оси яркостного распределения.

    Args:
        img: Изображение (H, W) или (H, W, 3) uint8.

    Returns:
        :class:`RotationEstimate` с методом ``"moments"``.

    Raises:
        ValueError: Если изображение пустое или все пиксели одного цвета.
    """
    gray = _to_gray(img)
    if gray.size == 0:
        raise ValueError("Image must not be empty")

    gray_f = gray.astype(np.float64)
    # Normalise so that very dark images still work
    if gray_f.max() > gray_f.min():
        gray_f = (gray_f - gray_f.min()) / (gray_f.max() - gray_f.min())
    else:
        raise ValueError("All pixels have the same intensity; cannot estimate rotation")

    h, w = gray_f.shape
    ys, xs = np.mgrid[0:h, 0:w]
    total = gray_f.sum() + 1e-12
    cx = (xs * gray_f).sum() / total
    cy = (ys * gray_f).sum() / total
    dx = xs - cx
    dy = ys - cy

    mu20 = (dx ** 2 * gray_f).sum() / total
    mu02 = (dy ** 2 * gray_f).sum() / total
    mu11 = (dx * dy * gray_f).sum() / total

    angle_rad = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
    angle_deg = _normalize_angle(float(np.degrees(angle_rad)))

    # Confidence from eccentricity
    lam1 = 0.5 * ((mu20 + mu02) + np.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2))
    lam2 = 0.5 * ((mu20 + mu02) - np.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2))
    lam2 = max(lam2, 0.0)
    eccentricity = float(np.sqrt(1.0 - lam2 / (lam1 + 1e-12)))
    confidence = float(np.clip(eccentricity, 0.0, 1.0))

    return RotationEstimate(
        angle_deg=angle_deg,
        confidence=confidence,
        method="moments",
        params={"mu20": mu20, "mu02": mu02, "mu11": mu11},
    )


def estimate_by_gradient(
    img: np.ndarray,
    n_bins: int = 180,
) -> RotationEstimate:
    """Оценить угол поворота через доминирующую ориентацию градиентов.

    Строит гистограмму ориентаций градиентов (0°–180°) и находит
    пик, соответствующий основному направлению штрихов/линий.

    Args:
        img:    Изображение (H, W) или (H, W, 3) uint8.
        n_bins: Число корзин гистограммы ориентаций.

    Returns:
        :class:`RotationEstimate` с методом ``"gradient"``.

    Raises:
        ValueError: Если ``n_bins`` < 2.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    gray = _to_gray(img)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.hypot(gx, gy)
    orientation = (np.degrees(np.arctan2(gy, gx)) % 180.0)

    weights = magnitude.ravel()
    angles = orientation.ravel()
    hist, bin_edges = np.histogram(angles, bins=n_bins, range=(0, 180), weights=weights)

    peak_bin = int(np.argmax(hist))
    dominant_angle = float(0.5 * (bin_edges[peak_bin] + bin_edges[peak_bin + 1]))
    angle_deg = _normalize_angle(dominant_angle)

    total_weight = float(weights.sum()) + 1e-12
    peak_weight = float(hist[peak_bin])
    confidence = float(np.clip(peak_weight / total_weight * n_bins / 10.0, 0.0, 1.0))

    return RotationEstimate(
        angle_deg=angle_deg,
        confidence=confidence,
        method="gradient",
        params={"n_bins": n_bins, "peak_bin": peak_bin},
    )


def refine_rotation(
    img: np.ndarray,
    initial_angle: float,
    search_range: float = 5.0,
    n_steps: int = 20,
) -> RotationEstimate:
    """Уточнить угол поворота перебором в окрестности начального значения.

    Вращает изображение на кандидатные углы и выбирает тот,
    при котором дисперсия горизонтальных проекций максимальна
    (документные строки горизонтальны → проекции хорошо разделены).

    Args:
        img:          Изображение (H, W) или (H, W, 3) uint8.
        initial_angle: Начальный угол (градусы).
        search_range:  Диапазон поиска ±search_range (градусы).
        n_steps:       Количество шагов в диапазоне (≥ 2).

    Returns:
        :class:`RotationEstimate` с методом ``"refine"``.

    Raises:
        ValueError: Если ``n_steps`` < 2 или ``search_range`` ≤ 0.
    """
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2, got {n_steps}")
    if search_range <= 0:
        raise ValueError(f"search_range must be > 0, got {search_range}")

    gray = _to_gray(img)
    h, w = gray.shape
    cx, cy = w / 2.0, h / 2.0

    candidates = np.linspace(
        initial_angle - search_range,
        initial_angle + search_range,
        n_steps,
    )
    best_angle = float(initial_angle)
    best_score = -np.inf

    for angle in candidates:
        M = cv2.getRotationMatrix2D((cx, cy), -float(angle), 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR)
        proj = rotated.sum(axis=1).astype(np.float64)
        score = float(proj.var())
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    confidence = float(np.clip(best_score / (best_score + 1e-3), 0.0, 1.0))

    return RotationEstimate(
        angle_deg=_normalize_angle(best_angle),
        confidence=confidence,
        method="refine",
        params={
            "initial_angle": initial_angle,
            "search_range": search_range,
            "n_steps": n_steps,
            "best_score": best_score,
        },
    )


def estimate_rotation_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    method: str = "gradient",
) -> Tuple[RotationEstimate, RotationEstimate]:
    """Оценить углы поворота для двух фрагментов одним методом.

    Args:
        img1:   Первый фрагмент.
        img2:   Второй фрагмент.
        method: Метод оценки: ``"pca"`` (требует контур), ``"moments"``,
                ``"gradient"``.

    Returns:
        Пара (:class:`RotationEstimate`, :class:`RotationEstimate`).

    Raises:
        ValueError: Если ``method`` неизвестен.
    """
    _methods = {
        "moments": estimate_by_moments,
        "gradient": estimate_by_gradient,
    }
    if method not in _methods:
        raise ValueError(
            f"Unknown method {method!r}. Choose from {sorted(_methods.keys())}"
        )
    fn = _methods[method]
    return fn(img1), fn(img2)


def batch_estimate_rotations(
    images: List[np.ndarray],
    method: str = "gradient",
) -> List[RotationEstimate]:
    """Оценить угол поворота для списка изображений.

    Args:
        images: Список изображений.
        method: Метод оценки: ``"moments"`` или ``"gradient"``.

    Returns:
        Список :class:`RotationEstimate` той же длины.
        Для пустого списка — пустой список.

    Raises:
        ValueError: Если ``method`` неизвестен.
    """
    _methods = {
        "moments": estimate_by_moments,
        "gradient": estimate_by_gradient,
    }
    if method not in _methods:
        raise ValueError(
            f"Unknown method {method!r}. Choose from {sorted(_methods.keys())}"
        )
    fn = _methods[method]
    return [fn(img) for img in images]
