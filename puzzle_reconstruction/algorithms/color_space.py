"""Анализ цветового пространства фрагментов документа.

Модуль предоставляет инструменты для работы с цветовыми пространствами:
вычисление цветовых гистограмм, сравнение гистограмм и конвертация
между цветовыми пространствами (BGR, HSV, LAB, Gray).

Классы:
    ColorSpaceConfig — параметры вычисления гистограммы
    ColorHistogram   — цветовая гистограмма изображения

Функции:
    bgr_to_space             — конвертировать BGR-изображение в заданное пространство
    compute_channel_hist     — гистограмма одного канала
    compute_color_histogram  — полная цветовая гистограмма
    histogram_intersection   — сходство по пересечению гистограмм ∈ [0, 1]
    histogram_chi2           — χ²-сходство гистограмм ∈ [0, 1]
    batch_compute_histograms — пакетное вычисление гистограмм
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


_VALID_SPACES = {"bgr", "hsv", "lab", "gray"}


# ─── ColorSpaceConfig ─────────────────────────────────────────────────────────

@dataclass
class ColorSpaceConfig:
    """Параметры вычисления цветовой гистограммы.

    Атрибуты:
        target_space: Цветовое пространство ('bgr', 'hsv', 'lab', 'gray').
        n_bins:       Число бинов гистограммы (>= 4).
        normalize:    Нормализовать гистограмму до суммы 1.
    """
    target_space: str = "hsv"
    n_bins:       int  = 32
    normalize:    bool = True

    def __post_init__(self) -> None:
        if self.target_space not in _VALID_SPACES:
            raise ValueError(
                f"target_space должен быть одним из {sorted(_VALID_SPACES)}, "
                f"получено '{self.target_space}'"
            )
        if self.n_bins < 4:
            raise ValueError(
                f"n_bins должен быть >= 4, получено {self.n_bins}"
            )


# ─── ColorHistogram ───────────────────────────────────────────────────────────

@dataclass
class ColorHistogram:
    """Цветовая гистограмма изображения.

    Атрибуты:
        fragment_id: Идентификатор фрагмента (>= 0).
        space:       Цветовое пространство.
        hist:        Нормализованная гистограмма (float32, 1-D).
        n_bins:      Число бинов на канал (>= 4).
        params:      Дополнительные параметры.
    """
    fragment_id: int
    space:       str
    hist:        np.ndarray
    n_bins:      int
    params:      Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        if self.n_bins < 4:
            raise ValueError(
                f"n_bins должен быть >= 4, получено {self.n_bins}"
            )
        self.hist = np.asarray(self.hist, dtype=np.float32)
        if self.hist.ndim != 1:
            raise ValueError(
                f"hist должен быть 1-D, получено ndim={self.hist.ndim}"
            )

    @property
    def dim(self) -> int:
        """Длина вектора гистограммы."""
        return len(self.hist)


# ─── bgr_to_space ─────────────────────────────────────────────────────────────

def bgr_to_space(img: np.ndarray, space: str) -> np.ndarray:
    """Конвертировать BGR-изображение в заданное цветовое пространство.

    Аргументы:
        img:   BGR или grayscale изображение (uint8, 2-D или 3-D).
        space: 'bgr' | 'hsv' | 'lab' | 'gray'.

    Возвращает:
        Изображение в целевом пространстве (uint8).

    Исключения:
        ValueError: Если space неизвестно или img некорректен.
    """
    if space not in _VALID_SPACES:
        raise ValueError(
            f"space должен быть одним из {sorted(_VALID_SPACES)}, "
            f"получено '{space}'"
        )

    img = np.asarray(img)
    if img.ndim not in (2, 3):
        raise ValueError(
            f"img должен быть 2-D или 3-D, получено ndim={img.ndim}"
        )

    # Привести к uint8 BGR (3-D) для конвертации
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        bgr = img.copy()

    if space == "bgr":
        return bgr
    if space == "gray":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if space == "hsv":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if space == "lab":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    # Unreachable — already validated
    raise ValueError(f"Неизвестное пространство: {space}")  # pragma: no cover


# ─── compute_channel_hist ─────────────────────────────────────────────────────

def compute_channel_hist(
    channel: np.ndarray,
    n_bins:  int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """Вычислить гистограмму одного канала изображения.

    Аргументы:
        channel:   2-D массив (uint8).
        n_bins:    Число бинов (>= 4).
        normalize: Нормализовать до суммы 1.

    Возвращает:
        Гистограмма (float32, shape=(n_bins,)).

    Исключения:
        ValueError: Если n_bins < 4.
    """
    if n_bins < 4:
        raise ValueError(f"n_bins должен быть >= 4, получено {n_bins}")

    hist, _ = np.histogram(channel.ravel(), bins=n_bins, range=(0, 256))
    hist = hist.astype(np.float32)
    if normalize:
        total = hist.sum()
        if total > 0:
            hist /= total
    return hist


# ─── compute_color_histogram ──────────────────────────────────────────────────

def compute_color_histogram(
    img:         np.ndarray,
    cfg:         Optional[ColorSpaceConfig] = None,
    fragment_id: int = 0,
) -> ColorHistogram:
    """Вычислить цветовую гистограмму изображения.

    Для grayscale-пространства гистограмма 1-канальная.
    Для BGR/HSV/LAB — конкатенация гистограмм по каналам.

    Аргументы:
        img:         Изображение (uint8, 2-D или 3-D).
        cfg:         Конфигурация (None → ColorSpaceConfig()).
        fragment_id: Идентификатор фрагмента (>= 0).

    Возвращает:
        ColorHistogram.
    """
    if cfg is None:
        cfg = ColorSpaceConfig()

    converted = bgr_to_space(img, cfg.target_space)
    parts: List[np.ndarray] = []

    if converted.ndim == 2:
        parts.append(compute_channel_hist(converted, cfg.n_bins, cfg.normalize))
    else:
        for c in range(converted.shape[2]):
            parts.append(
                compute_channel_hist(converted[:, :, c], cfg.n_bins,
                                     cfg.normalize)
            )

    hist = np.concatenate(parts).astype(np.float32)

    return ColorHistogram(
        fragment_id=fragment_id,
        space=cfg.target_space,
        hist=hist,
        n_bins=cfg.n_bins,
        params={"n_channels": len(parts), "normalize": cfg.normalize},
    )


# ─── histogram_intersection ───────────────────────────────────────────────────

def histogram_intersection(a: ColorHistogram, b: ColorHistogram) -> float:
    """Сходство двух гистограмм по пересечению.

    Σ min(a_i, b_i) — для нормализованных гистограмм ∈ [0, 1].

    Аргументы:
        a: Первая гистограмма.
        b: Вторая гистограмма.

    Возвращает:
        Оценка ∈ [0, 1].

    Исключения:
        ValueError: Если длины гистограмм не совпадают.
    """
    if a.dim != b.dim:
        raise ValueError(
            f"Длины гистограмм не совпадают: {a.dim} vs {b.dim}"
        )
    return float(np.minimum(a.hist, b.hist).sum())


# ─── histogram_chi2 ───────────────────────────────────────────────────────────

def histogram_chi2(a: ColorHistogram, b: ColorHistogram) -> float:
    """Сходство двух гистограмм на основе χ²-расстояния.

    χ²(a, b) = Σ (a_i - b_i)² / (a_i + b_i + ε).
    Возвращает 1 / (1 + χ²) ∈ (0, 1].

    Аргументы:
        a: Первая гистограмма.
        b: Вторая гистограмма.

    Возвращает:
        Оценка ∈ (0, 1].

    Исключения:
        ValueError: Если длины гистограмм не совпадают.
    """
    if a.dim != b.dim:
        raise ValueError(
            f"Длины гистограмм не совпадают: {a.dim} vs {b.dim}"
        )
    h_a = a.hist.astype(np.float64)
    h_b = b.hist.astype(np.float64)
    num = (h_a - h_b) ** 2
    denom = h_a + h_b + 1e-10
    chi2 = float((num / denom).sum())
    return float(1.0 / (1.0 + chi2))


# ─── batch_compute_histograms ─────────────────────────────────────────────────

def batch_compute_histograms(
    images: List[np.ndarray],
    cfg:    Optional[ColorSpaceConfig] = None,
) -> List[ColorHistogram]:
    """Вычислить гистограммы для списка изображений.

    Аргументы:
        images: Список изображений (uint8).
        cfg:    Конфигурация (None → ColorSpaceConfig()).

    Возвращает:
        Список ColorHistogram; fragment_id = индекс в списке.
    """
    if cfg is None:
        cfg = ColorSpaceConfig()
    return [
        compute_color_histogram(img, cfg, fragment_id=i)
        for i, img in enumerate(images)
    ]
