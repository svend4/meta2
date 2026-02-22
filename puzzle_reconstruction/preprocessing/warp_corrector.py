"""Коррекция перспективных и аффинных искажений фрагментов.

Модуль предоставляет структуры и функции для оценки и устранения
искажений формы изображения фрагмента: перспективного сдвига,
аффинных деформаций и общего «выравнивания» прямоугольника.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ─── WarpConfig ───────────────────────────────────────────────────────────────

@dataclass
class WarpConfig:
    """Параметры коррекции искажений.

    Атрибуты:
        output_size:    Размер выходного изображения (ширина, высота) (>= 1).
        border_mode:    Режим заполнения границы: "zero", "replicate".
        max_iter:       Максимальное число итераций оценки (>= 1).
        convergence_eps: Порог сходимости (>= 0).
    """

    output_size: Tuple[int, int] = (128, 128)
    border_mode: str = "zero"
    max_iter: int = 10
    convergence_eps: float = 1e-4

    def __post_init__(self) -> None:
        w, h = self.output_size
        if w < 1 or h < 1:
            raise ValueError(
                f"output_size должен быть >= (1, 1), получено {self.output_size}"
            )
        if self.border_mode not in ("zero", "replicate"):
            raise ValueError(
                f"border_mode должен быть 'zero' или 'replicate', "
                f"получено '{self.border_mode}'"
            )
        if self.max_iter < 1:
            raise ValueError(
                f"max_iter должен быть >= 1, получено {self.max_iter}"
            )
        if self.convergence_eps < 0:
            raise ValueError(
                f"convergence_eps должен быть >= 0, получено {self.convergence_eps}"
            )


# ─── WarpEstimate ─────────────────────────────────────────────────────────────

@dataclass
class WarpEstimate:
    """Оценка аффинного преобразования для коррекции искажения.

    Атрибуты:
        matrix:      Матрица 2×3 аффинного преобразования.
        skew_angle:  Угол перекоса в градусах.
        scale_x:     Масштаб по X (> 0).
        scale_y:     Масштаб по Y (> 0).
        translation: Вектор сдвига (tx, ty).
        confidence:  Уверенность оценки [0, 1].
    """

    matrix: np.ndarray
    skew_angle: float
    scale_x: float
    scale_y: float
    translation: Tuple[float, float]
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.matrix.shape != (2, 3):
            raise ValueError(
                f"matrix должен иметь форму (2, 3), получено {self.matrix.shape}"
            )
        if self.scale_x <= 0:
            raise ValueError(
                f"scale_x должен быть > 0, получено {self.scale_x}"
            )
        if self.scale_y <= 0:
            raise ValueError(
                f"scale_y должен быть > 0, получено {self.scale_y}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence должен быть в [0, 1], получено {self.confidence}"
            )

    @property
    def is_identity(self) -> bool:
        """True если матрица близка к единичной."""
        identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        return bool(np.allclose(self.matrix, identity, atol=1e-6))

    @property
    def rotation_deg(self) -> float:
        """Угол вращения в градусах (из 2×3 матрицы)."""
        return float(np.degrees(np.arctan2(self.matrix[1, 0], self.matrix[0, 0])))


# ─── WarpResult ───────────────────────────────────────────────────────────────

@dataclass
class WarpResult:
    """Результат коррекции изображения.

    Атрибуты:
        corrected:   Выправленное изображение (H×W или H×W×C).
        estimate:    WarpEstimate, использованное для коррекции.
        n_iter:      Число выполненных итераций (>= 0).
        converged:   True если алгоритм сошёлся.
    """

    corrected: np.ndarray
    estimate: WarpEstimate
    n_iter: int
    converged: bool

    def __post_init__(self) -> None:
        if self.n_iter < 0:
            raise ValueError(
                f"n_iter должен быть >= 0, получено {self.n_iter}"
            )

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """Форма выходного изображения."""
        return tuple(self.corrected.shape)

    @property
    def was_modified(self) -> bool:
        """True если применена ненулевая трансформация."""
        return not self.estimate.is_identity


# ─── _identity_estimate ───────────────────────────────────────────────────────

def _identity_estimate() -> WarpEstimate:
    """Создать единичное аффинное преобразование."""
    return WarpEstimate(
        matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        skew_angle=0.0,
        scale_x=1.0,
        scale_y=1.0,
        translation=(0.0, 0.0),
    )


# ─── estimate_warp ────────────────────────────────────────────────────────────

def estimate_warp(
    image: np.ndarray,
    cfg: Optional[WarpConfig] = None,
) -> WarpEstimate:
    """Оценить аффинное преобразование для выравнивания изображения.

    Аргументы:
        image: 2D или 3D изображение фрагмента.
        cfg:   Параметры.

    Возвращает:
        WarpEstimate.

    Исключения:
        ValueError: Если image имеет неподдерживаемую размерность.
    """
    if cfg is None:
        cfg = WarpConfig()
    if image.ndim not in (2, 3):
        raise ValueError("image должен быть 2D или 3D массивом")

    if image.ndim == 3:
        gray = np.mean(image, axis=2).astype(float)
    else:
        gray = image.astype(float)

    h, w = gray.shape
    if h == 0 or w == 0:
        return _identity_estimate()

    # Оценить перекос через угол наклона первого момента
    ys, xs = np.mgrid[0:h, 0:w]
    total = float(gray.sum()) + 1e-12
    cx = float((xs * gray).sum()) / total
    cy = float((ys * gray).sum()) / total

    # Вычислить центральные моменты 2-го порядка
    mu20 = float(((xs - cx) ** 2 * gray).sum()) / total
    mu02 = float(((ys - cy) ** 2 * gray).sum()) / total
    mu11 = float(((xs - cx) * (ys - cy) * gray).sum()) / total

    skew_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    skew_deg = float(np.degrees(skew_rad))

    cos_a, sin_a = float(np.cos(skew_rad)), float(np.sin(skew_rad))
    scale_x = float(np.sqrt(max(mu20, 1e-12)) / max(w / 4.0, 1e-12))
    scale_y = float(np.sqrt(max(mu02, 1e-12)) / max(h / 4.0, 1e-12))
    scale_x = float(np.clip(scale_x, 0.1, 10.0))
    scale_y = float(np.clip(scale_y, 0.1, 10.0))

    tx = cx - w / 2.0
    ty = cy - h / 2.0

    matrix = np.array([
        [cos_a * scale_x, -sin_a * scale_y, tx],
        [sin_a * scale_x,  cos_a * scale_y, ty],
    ])

    confidence = float(np.clip(1.0 / (1.0 + abs(skew_deg) / 45.0), 0.0, 1.0))

    return WarpEstimate(
        matrix=matrix,
        skew_angle=skew_deg,
        scale_x=scale_x,
        scale_y=scale_y,
        translation=(tx, ty),
        confidence=confidence,
    )


# ─── apply_warp ───────────────────────────────────────────────────────────────

def apply_warp(
    image: np.ndarray,
    estimate: WarpEstimate,
    cfg: Optional[WarpConfig] = None,
) -> np.ndarray:
    """Применить аффинное преобразование к изображению.

    Аргументы:
        image:    Входное изображение (2D или 3D).
        estimate: WarpEstimate.
        cfg:      Параметры (используется output_size, border_mode).

    Возвращает:
        Преобразованное изображение заданного размера.
    """
    if cfg is None:
        cfg = WarpConfig()

    out_w, out_h = cfg.output_size
    m = estimate.matrix  # 2×3

    if image.ndim == 3:
        channels = image.shape[2]
        out = np.zeros((out_h, out_w, channels), dtype=image.dtype)
    else:
        out = np.zeros((out_h, out_w), dtype=image.dtype)

    h_in, w_in = image.shape[:2]

    ys_dst, xs_dst = np.mgrid[0:out_h, 0:out_w]
    # Обратное отображение: src = M^{-1} * dst
    try:
        m3 = np.vstack([m, [0, 0, 1]])
        m3_inv = np.linalg.inv(m3)[:2]
    except np.linalg.LinAlgError:
        m3_inv = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

    xs_src = m3_inv[0, 0] * xs_dst + m3_inv[0, 1] * ys_dst + m3_inv[0, 2]
    ys_src = m3_inv[1, 0] * xs_dst + m3_inv[1, 1] * ys_dst + m3_inv[1, 2]

    xs_src_i = xs_src.astype(int)
    ys_src_i = ys_src.astype(int)

    if cfg.border_mode == "replicate":
        xs_src_i = np.clip(xs_src_i, 0, w_in - 1)
        ys_src_i = np.clip(ys_src_i, 0, h_in - 1)
        valid = np.ones((out_h, out_w), dtype=bool)
    else:
        valid = (
            (xs_src_i >= 0) & (xs_src_i < w_in) &
            (ys_src_i >= 0) & (ys_src_i < h_in)
        )

    if image.ndim == 3:
        out[valid] = image[ys_src_i[valid], xs_src_i[valid]]
    else:
        out[valid] = image[ys_src_i[valid], xs_src_i[valid]]

    return out


# ─── correct_warp ─────────────────────────────────────────────────────────────

def correct_warp(
    image: np.ndarray,
    cfg: Optional[WarpConfig] = None,
) -> WarpResult:
    """Оценить и применить коррекцию искажений.

    Аргументы:
        image: Входное изображение (2D или 3D).
        cfg:   Параметры.

    Возвращает:
        WarpResult.
    """
    if cfg is None:
        cfg = WarpConfig()

    estimate = estimate_warp(image, cfg)
    corrected = apply_warp(image, estimate, cfg)

    return WarpResult(
        corrected=corrected,
        estimate=estimate,
        n_iter=1,
        converged=True,
    )


# ─── warp_score ───────────────────────────────────────────────────────────────

def warp_score(estimate: WarpEstimate) -> float:
    """Оценить качество коррекции [0, 1].

    Малый угол перекоса и близкие к 1 масштабы дают высокую оценку.

    Аргументы:
        estimate: WarpEstimate.

    Возвращает:
        Оценка [0, 1].
    """
    angle_penalty = min(abs(estimate.skew_angle) / 45.0, 1.0)
    scale_penalty = (abs(estimate.scale_x - 1.0) + abs(estimate.scale_y - 1.0)) / 2.0
    score = estimate.confidence * (1.0 - 0.5 * angle_penalty - 0.5 * min(scale_penalty, 1.0))
    return float(np.clip(score, 0.0, 1.0))


# ─── batch_correct_warp ───────────────────────────────────────────────────────

def batch_correct_warp(
    images: List[np.ndarray],
    cfg: Optional[WarpConfig] = None,
) -> List[WarpResult]:
    """Применить коррекцию ко всем изображениям списка.

    Аргументы:
        images: Список изображений.
        cfg:    Параметры.

    Возвращает:
        Список WarpResult.
    """
    return [correct_warp(img, cfg) for img in images]
