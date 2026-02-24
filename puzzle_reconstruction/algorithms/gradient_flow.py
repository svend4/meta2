"""
Анализ градиентного поля для сопоставления фрагментов.

Вычисляет поля градиентов, векторы потока, дивергенцию, ротор и
граничный поток для анализа и сравнения фрагментов документа.

Экспортирует:
    GradientField       — структура данных: поле градиента (gx, gy)
    compute_gradient    — вычислить поле градиента изображения (Sobel)
    compute_magnitude   — карта магнитуд градиента
    compute_orientation — карта ориентаций (углы в радианах)
    compute_divergence  — дивергенция векторного поля
    compute_curl        — ротор (curl) векторного поля
    flow_along_boundary — поток вдоль границы (контура)
    compare_gradient_fields — сходство двух полей градиентов
    batch_gradient_fields   — пакетное вычисление полей
    GradientStats       — статистика поля градиента
    compute_gradient_stats  — сводная статистика поля
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─── Структуры данных ────────────────────────────────────────────────────────

@dataclass
class GradientField:
    """Поле градиента изображения.

    Attributes:
        gx:     Горизонтальная составляющая градиента (float32, H×W).
        gy:     Вертикальная составляющая градиента (float32, H×W).
        params: Параметры вычисления.
    """
    gx: np.ndarray
    gy: np.ndarray
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.gx.shape != self.gy.shape:
            raise ValueError(
                f"gx and gy must have the same shape, got {self.gx.shape} vs {self.gy.shape}"
            )

    @property
    def shape(self) -> Tuple[int, int]:
        return self.gx.shape[:2]

    def __repr__(self) -> str:  # pragma: no cover
        return f"GradientField(shape={self.shape})"


@dataclass
class GradientStats:
    """Сводная статистика поля градиента.

    Attributes:
        mean_magnitude:    Среднее значение магнитуды.
        std_magnitude:     СКО магнитуды.
        mean_orientation:  Средняя ориентация (рад).
        dominant_angle:    Доминирующий угол по гистограмме (рад).
        edge_density:      Доля пикселей с магнитудой выше порога (0–1).
        params:            Параметры, использованные при вычислении.
    """
    mean_magnitude: float
    std_magnitude: float
    mean_orientation: float
    dominant_angle: float
    edge_density: float
    params: dict = field(default_factory=dict)


# ─── Публичные функции ───────────────────────────────────────────────────────

def compute_gradient(
    img: np.ndarray,
    ksize: int = 3,
    normalize: bool = False,
) -> GradientField:
    """Вычислить поле градиента изображения методом Собела.

    Args:
        img:       Изображение uint8 (2D) или BGR (3D; автоматически переводится в оттенки серого).
        ksize:     Размер ядра Собела (1, 3, 5 или 7).
        normalize: Если ``True``, нормировать (gx, gy) к диапазону [−1, 1].

    Returns:
        :class:`GradientField` с составляющими float32.

    Raises:
        ValueError: Если ``ksize`` не входит в {1, 3, 5, 7}.
    """
    if ksize not in (1, 3, 5, 7):
        raise ValueError(f"ksize must be one of {{1, 3, 5, 7}}, got {ksize}")
    gray = _to_gray(img)
    gray_f = gray.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=ksize)
    if normalize:
        mag = np.sqrt(gx ** 2 + gy ** 2)
        max_mag = mag.max()
        if max_mag > 1e-8:
            gx = gx / max_mag
            gy = gy / max_mag
    return GradientField(gx=gx, gy=gy, params={"ksize": ksize, "normalize": normalize})


def compute_magnitude(field: GradientField) -> np.ndarray:
    """Карта магнитуд: sqrt(gx² + gy²).

    Args:
        field: :class:`GradientField`.

    Returns:
        Массив float32 (H, W).
    """
    return np.sqrt(field.gx ** 2 + field.gy ** 2).astype(np.float32)


def compute_orientation(field: GradientField) -> np.ndarray:
    """Карта ориентаций: arctan2(gy, gx) ∈ [−π, π].

    Args:
        field: :class:`GradientField`.

    Returns:
        Массив float32 (H, W) в радианах.
    """
    return np.arctan2(field.gy, field.gx).astype(np.float32)


def compute_divergence(field: GradientField) -> np.ndarray:
    """Дивергенция векторного поля: ∂gx/∂x + ∂gy/∂y.

    Args:
        field: :class:`GradientField`.

    Returns:
        Массив float32 (H, W).
    """
    dgx_dx = cv2.Sobel(field.gx, cv2.CV_32F, 1, 0, ksize=3)
    dgy_dy = cv2.Sobel(field.gy, cv2.CV_32F, 0, 1, ksize=3)
    return (dgx_dx + dgy_dy).astype(np.float32)


def compute_curl(field: GradientField) -> np.ndarray:
    """Ротор (2D curl) векторного поля: ∂gy/∂x − ∂gx/∂y.

    Args:
        field: :class:`GradientField`.

    Returns:
        Массив float32 (H, W).
    """
    dgy_dx = cv2.Sobel(field.gy, cv2.CV_32F, 1, 0, ksize=3)
    dgx_dy = cv2.Sobel(field.gx, cv2.CV_32F, 0, 1, ksize=3)
    return (dgy_dx - dgx_dy).astype(np.float32)


def flow_along_boundary(
    field: GradientField,
    contour: np.ndarray,
    window: int = 1,
) -> np.ndarray:
    """Вычислить поток (dot-product с касательной) вдоль контура.

    Для каждой точки контура вычисляется скалярное произведение вектора
    градиента на единичную касательную к контуру.

    Args:
        field:   :class:`GradientField`.
        contour: Массив формы (N, 2) или (N, 1, 2) с координатами (x, y).
        window:  Радиус сглаживания касательной (кол-во соседних точек).

    Returns:
        Массив float32 (N,) — значения потока в каждой точке контура.
        Для пустого контура — пустой массив.

    Raises:
        ValueError: Если ``window`` < 1.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    pts = contour.reshape(-1, 2).astype(np.float32)
    n = len(pts)
    if n == 0:
        return np.empty(0, dtype=np.float32)

    h, w = field.shape
    flows = np.zeros(n, dtype=np.float32)
    for i in range(n):
        prev_idx = (i - window) % n
        next_idx = (i + window) % n
        tangent = pts[next_idx] - pts[prev_idx]
        norm = np.linalg.norm(tangent)
        if norm < 1e-8:
            continue
        tangent = tangent / norm
        xi = int(np.clip(round(pts[i, 0]), 0, w - 1))
        yi = int(np.clip(round(pts[i, 1]), 0, h - 1))
        gvec = np.array([field.gx[yi, xi], field.gy[yi, xi]], dtype=np.float32)
        flows[i] = float(np.dot(gvec, tangent))
    return flows


def compare_gradient_fields(
    field1: GradientField,
    field2: GradientField,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Вычислить сходство двух полей градиентов (нормированное скалярное произведение).

    Метрика: среднее cosine-сходство покоординатно совпадающих векторов.
    Поля должны иметь одинаковый размер.

    Args:
        field1: Первое поле.
        field2: Второе поле.
        mask:   Бинарная маска (H, W); если задана, учитываются только
                пиксели маски.

    Returns:
        Значение от −1.0 до 1.0 (1.0 = идентичные поля).

    Raises:
        ValueError: Если размеры полей не совпадают.
    """
    if field1.shape != field2.shape:
        raise ValueError(
            f"Fields must have the same shape, got {field1.shape} vs {field2.shape}"
        )
    dot = field1.gx * field2.gx + field1.gy * field2.gy
    mag1 = np.sqrt(field1.gx ** 2 + field1.gy ** 2) + 1e-8
    mag2 = np.sqrt(field2.gx ** 2 + field2.gy ** 2) + 1e-8
    cosine = dot / (mag1 * mag2)

    if mask is not None:
        m = mask.astype(bool)
        if m.sum() == 0:
            return 0.0
        return float(cosine[m].mean())
    return float(cosine.mean())


def batch_gradient_fields(
    images: List[np.ndarray],
    ksize: int = 3,
    normalize: bool = False,
) -> List[GradientField]:
    """Пакетное вычисление полей градиентов.

    Args:
        images:    Список изображений uint8.
        ksize:     Размер ядра Собела.
        normalize: Нормализация полей.

    Returns:
        Список :class:`GradientField` той же длины.
    """
    return [compute_gradient(img, ksize=ksize, normalize=normalize) for img in images]


def compute_gradient_stats(
    field: GradientField,
    threshold: float = 10.0,
    n_orientation_bins: int = 36,
) -> GradientStats:
    """Вычислить сводную статистику поля градиента.

    Args:
        field:               :class:`GradientField`.
        threshold:           Порог магнитуды для вычисления ``edge_density``.
        n_orientation_bins:  Число корзин гистограммы ориентаций.

    Returns:
        :class:`GradientStats`.

    Raises:
        ValueError: Если ``threshold`` < 0 или ``n_orientation_bins`` < 1.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    if n_orientation_bins < 1:
        raise ValueError(f"n_orientation_bins must be >= 1, got {n_orientation_bins}")

    mag = compute_magnitude(field)
    orient = compute_orientation(field)

    mean_mag = float(mag.mean())
    std_mag = float(mag.std())
    mean_orient = float(orient.mean())
    edge_density = float((mag >= threshold).mean())

    # Доминирующий угол по взвешенной гистограмме ориентаций
    weights = mag.ravel()
    angles = orient.ravel()
    hist, bin_edges = np.histogram(
        angles,
        bins=n_orientation_bins,
        range=(-np.pi, np.pi),
        weights=weights,
    )
    dominant_bin = int(np.argmax(hist))
    dominant_angle = float(0.5 * (bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]))

    return GradientStats(
        mean_magnitude=mean_mag,
        std_magnitude=std_mag,
        mean_orientation=mean_orient,
        dominant_angle=dominant_angle,
        edge_density=edge_density,
        params={"threshold": threshold, "n_orientation_bins": n_orientation_bins},
    )


# ─── Приватные ───────────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
