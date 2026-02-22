"""Анализ текстурных признаков фрагментов пазла.

Модуль вычисляет статистические и градиентные характеристики текстуры
изображения (энтропия, контраст, градиентные моменты) и предоставляет
пайплайн извлечения признаков для последующего сравнения фрагментов.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np


_VALID_PATCH_SIZES = {3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25}


# ─── TextureConfig ────────────────────────────────────────────────────────────

@dataclass
class TextureConfig:
    """Параметры извлечения текстурных признаков.

    Атрибуты:
        n_bins:      Число бинов гистограммы (>= 2).
        patch_size:  Размер окна для локальных операций (нечётное >= 3).
        use_gradient: Вычислять градиентные статистики.
        use_stats:    Вычислять статистические моменты (mean/std/skew).
    """

    n_bins: int = 32
    patch_size: int = 5
    use_gradient: bool = True
    use_stats: bool = True

    def __post_init__(self) -> None:
        if self.n_bins < 2:
            raise ValueError(
                f"n_bins должен быть >= 2, получено {self.n_bins}"
            )
        if self.patch_size % 2 == 0 or self.patch_size < 3:
            raise ValueError(
                f"patch_size должен быть нечётным числом >= 3, "
                f"получено {self.patch_size}"
            )


# ─── GradientStats ────────────────────────────────────────────────────────────

@dataclass
class GradientStats:
    """Градиентные статистики изображения.

    Атрибуты:
        mean:    Среднее абсолютное значение градиента (>= 0).
        std:     Стандартное отклонение градиента (>= 0).
        max_val: Максимальное абсолютное значение (>= 0).
        energy:  Сумма квадратов градиентов (>= 0).
    """

    mean: float
    std: float
    max_val: float
    energy: float

    def __post_init__(self) -> None:
        for name, val in (
            ("mean", self.mean),
            ("std", self.std),
            ("max_val", self.max_val),
            ("energy", self.energy),
        ):
            if val < 0.0:
                raise ValueError(
                    f"{name} должен быть >= 0, получено {val}"
                )


# ─── TextureFeatures ──────────────────────────────────────────────────────────

@dataclass
class TextureFeatures:
    """Вектор текстурных признаков фрагмента.

    Атрибуты:
        fragment_id:    Идентификатор фрагмента (>= 0).
        gradient_mean:  Среднее абс. значение градиента (>= 0).
        gradient_std:   СКО градиента (>= 0).
        entropy:        Текстурная энтропия (>= 0).
        contrast:       Локальный контраст (>= 0).
        n_bins:         Число бинов (>= 2).
        histogram:      Нормированная гистограмма (sum ≈ 1).
    """

    fragment_id: int
    gradient_mean: float = 0.0
    gradient_std: float = 0.0
    entropy: float = 0.0
    contrast: float = 0.0
    n_bins: int = 32
    histogram: Optional[np.ndarray] = field(default=None)

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        for name, val in (
            ("gradient_mean", self.gradient_mean),
            ("gradient_std", self.gradient_std),
            ("entropy", self.entropy),
            ("contrast", self.contrast),
        ):
            if val < 0.0:
                raise ValueError(
                    f"{name} должен быть >= 0, получено {val}"
                )
        if self.n_bins < 2:
            raise ValueError(
                f"n_bins должен быть >= 2, получено {self.n_bins}"
            )

    @property
    def feature_vector(self) -> np.ndarray:
        """Компактный вектор числовых признаков (без гистограммы)."""
        return np.array(
            [self.gradient_mean, self.gradient_std,
             self.entropy, self.contrast],
            dtype=np.float64,
        )


# ─── compute_gradient_stats ───────────────────────────────────────────────────

def compute_gradient_stats(image: np.ndarray) -> GradientStats:
    """Вычислить градиентные статистики изображения.

    Аргументы:
        image: Серое или цветное изображение (uint8 или float).

    Возвращает:
        GradientStats.

    Исключения:
        ValueError: Если image не 2D или 3D.
    """
    img = np.asarray(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    elif img.ndim != 2:
        raise ValueError(
            f"image должно быть 2D или 3D, получено ndim={img.ndim}"
        )
    gray = img.astype(np.float64)

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    return GradientStats(
        mean=float(magnitude.mean()),
        std=float(magnitude.std()),
        max_val=float(magnitude.max()),
        energy=float((magnitude ** 2).sum()),
    )


# ─── compute_texture_entropy ──────────────────────────────────────────────────

def compute_texture_entropy(
    image: np.ndarray,
    n_bins: int = 32,
) -> float:
    """Вычислить текстурную энтропию через гистограмму яркостей.

    Аргументы:
        image:  Серое или цветное изображение.
        n_bins: Число бинов гистограммы (>= 2).

    Возвращает:
        Энтропия Шеннона (bits, >= 0).

    Исключения:
        ValueError: Если n_bins < 2.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins должен быть >= 2, получено {n_bins}")

    img = np.asarray(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    hist, _ = np.histogram(img.ravel(), bins=n_bins, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total < 1.0:
        return 0.0

    probs = hist / total
    probs = probs[probs > 0.0]
    return float(-np.sum(probs * np.log2(probs)))


# ─── compute_texture_contrast ─────────────────────────────────────────────────

def compute_texture_contrast(
    image: np.ndarray,
    patch_size: int = 5,
) -> float:
    """Вычислить средний локальный контраст (СКО в окне).

    Аргументы:
        image:      Серое или цветное изображение.
        patch_size: Размер окна (нечётное >= 3).

    Возвращает:
        Среднее значение локального СКО (>= 0).

    Исключения:
        ValueError: Если patch_size чётный или < 3.
    """
    if patch_size % 2 == 0 or patch_size < 3:
        raise ValueError(
            f"patch_size должен быть нечётным >= 3, получено {patch_size}"
        )

    img = np.asarray(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    gray = img.astype(np.float64)

    ksize = (patch_size, patch_size)
    mean_img = cv2.blur(gray, ksize)
    mean_sq = cv2.blur(gray ** 2, ksize)
    variance = np.clip(mean_sq - mean_img ** 2, 0.0, None)
    local_std = np.sqrt(variance)
    return float(local_std.mean())


# ─── extract_texture_features ─────────────────────────────────────────────────

def extract_texture_features(
    image: np.ndarray,
    fragment_id: int = 0,
    cfg: Optional[TextureConfig] = None,
) -> TextureFeatures:
    """Извлечь полный вектор текстурных признаков из изображения фрагмента.

    Аргументы:
        image:       Изображение фрагмента (uint8, серое или BGR).
        fragment_id: Идентификатор фрагмента (>= 0).
        cfg:         Параметры (None → TextureConfig()).

    Возвращает:
        TextureFeatures.

    Исключения:
        ValueError: Если fragment_id < 0.
    """
    if fragment_id < 0:
        raise ValueError(
            f"fragment_id должен быть >= 0, получено {fragment_id}"
        )
    if cfg is None:
        cfg = TextureConfig()

    img = np.asarray(image)

    # Вычисляем гистограмму
    if img.ndim == 3:
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = img.astype(np.uint8)

    hist, _ = np.histogram(gray.ravel(), bins=cfg.n_bins, range=(0, 256))
    hist_norm = hist.astype(np.float64)
    s = hist_norm.sum()
    if s > 0:
        hist_norm /= s

    # Энтропия
    entropy = compute_texture_entropy(img, cfg.n_bins)

    # Контраст
    contrast = compute_texture_contrast(img, cfg.patch_size)

    # Градиент
    grad_mean, grad_std = 0.0, 0.0
    if cfg.use_gradient:
        gs = compute_gradient_stats(img)
        grad_mean = gs.mean
        grad_std = gs.std

    return TextureFeatures(
        fragment_id=fragment_id,
        gradient_mean=grad_mean,
        gradient_std=grad_std,
        entropy=entropy,
        contrast=contrast,
        n_bins=cfg.n_bins,
        histogram=hist_norm,
    )


# ─── compare_texture_features ────────────────────────────────────────────────

def compare_texture_features(
    a: TextureFeatures,
    b: TextureFeatures,
) -> float:
    """Сравнить два вектора текстурных признаков (косинусное сходство).

    Аргументы:
        a: Первый вектор.
        b: Второй вектор.

    Возвращает:
        Косинусное сходство в [-1, 1]. 1.0 — идентичные векторы.
    """
    va = a.feature_vector.astype(np.float64)
    vb = b.feature_vector.astype(np.float64)
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


# ─── batch_extract_texture ────────────────────────────────────────────────────

def batch_extract_texture(
    images: List[np.ndarray],
    cfg: Optional[TextureConfig] = None,
) -> List[TextureFeatures]:
    """Извлечь текстурные признаки для списка изображений.

    Аргументы:
        images: Список изображений.
        cfg:    Параметры (None → TextureConfig()).

    Возвращает:
        Список TextureFeatures (fragment_id = порядковый индекс).
    """
    return [
        extract_texture_features(img, fragment_id=i, cfg=cfg)
        for i, img in enumerate(images)
    ]
