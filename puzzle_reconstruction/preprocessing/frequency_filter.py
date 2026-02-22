"""
Фильтрация изображений в частотной области (FFT).

Предоставляет инструменты для подавления и усиления частотных компонент
изображений методами быстрого преобразования Фурье.

Экспортирует:
    FrequencyFilterParams — параметры частотного фильтра
    fft_image             — прямое FFT изображения
    ifft_image            — обратное FFT с нормализацией в uint8
    gaussian_low_pass     — Гауссов фильтр низких частот
    gaussian_high_pass    — Гауссов фильтр высоких частот
    band_pass_filter      — полосовой фильтр
    notch_filter          — режекторный фильтр (notch)
    apply_frequency_filter — применение фильтра по параметрам
    batch_frequency_filter — пакетная обработка изображений
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class FrequencyFilterParams:
    """Параметры частотного фильтра.

    Attributes:
        filter_type:  Тип фильтра: ``'low_pass'``, ``'high_pass'``,
                      ``'band_pass'``, ``'notch'``.
        sigma_low:    Сигма для нижней границы полосы (σ > 0).
        sigma_high:   Сигма для верхней границы (``'band_pass'`` только; σ > sigma_low).
        notch_radius: Радиус режекторного фильтра (``'notch'`` только; > 0).
    """
    filter_type: str = "low_pass"
    sigma_low: float = 10.0
    sigma_high: float = 30.0
    notch_radius: float = 5.0

    _VALID_TYPES = frozenset({"low_pass", "high_pass", "band_pass", "notch"})

    def __post_init__(self) -> None:
        if self.filter_type not in self._VALID_TYPES:
            raise ValueError(
                f"filter_type must be one of {sorted(self._VALID_TYPES)}, "
                f"got {self.filter_type!r}"
            )
        if self.sigma_low <= 0:
            raise ValueError(f"sigma_low must be > 0, got {self.sigma_low}")
        if self.filter_type == "band_pass":
            if self.sigma_high <= self.sigma_low:
                raise ValueError(
                    f"sigma_high ({self.sigma_high}) must be > sigma_low "
                    f"({self.sigma_low}) for band_pass"
                )
        if self.notch_radius <= 0:
            raise ValueError(
                f"notch_radius must be > 0, got {self.notch_radius}"
            )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FrequencyFilterParams(type={self.filter_type!r}, "
            f"sigma_low={self.sigma_low}, sigma_high={self.sigma_high})"
        )


# ─── Публичные функции ────────────────────────────────────────────────────────

def fft_image(img: np.ndarray) -> np.ndarray:
    """Вычислить FFT изображения.

    Args:
        img: Изображение uint8 (H, W) или (H, W, C); при C>1 конвертируется
             в оттенки серого.

    Returns:
        Комплексный массив float64 (H, W) — сдвинутый спектр
        (numpy.fft.fftshift применён, нулевая частота — в центре).

    Raises:
        ValueError: Если изображение не двух- и не трёхмерное.
    """
    gray = _to_gray_float(img)
    spectrum = np.fft.fft2(gray)
    return np.fft.fftshift(spectrum)


def ifft_image(fft_arr: np.ndarray) -> np.ndarray:
    """Восстановить изображение из FFT-спектра.

    Args:
        fft_arr: Комплексный сдвинутый спектр (H, W), как из :func:`fft_image`.

    Returns:
        Изображение uint8 (H, W) с нормализацией по [0, 255].
    """
    shifted_back = np.fft.ifftshift(fft_arr)
    restored = np.fft.ifft2(shifted_back)
    real = np.abs(restored)
    return _normalize_to_uint8(real)


def gaussian_low_pass(img: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Применить Гауссов фильтр низких частот.

    Args:
        img:   Изображение uint8 (H, W) или (H, W, C).
        sigma: Среднеквадратическое отклонение Гауссова окна (> 0).

    Returns:
        Отфильтрованное изображение uint8 (H, W).

    Raises:
        ValueError: Если ``sigma`` ≤ 0.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    spectrum = fft_image(img)
    h, w = spectrum.shape
    mask = _gaussian_mask(h, w, sigma)
    return ifft_image(spectrum * mask)


def gaussian_high_pass(img: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Применить Гауссов фильтр высоких частот.

    Args:
        img:   Изображение uint8 (H, W) или (H, W, C).
        sigma: Среднеквадратическое отклонение (> 0).

    Returns:
        Отфильтрованное изображение uint8 (H, W).

    Raises:
        ValueError: Если ``sigma`` ≤ 0.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    spectrum = fft_image(img)
    h, w = spectrum.shape
    mask = 1.0 - _gaussian_mask(h, w, sigma)
    return ifft_image(spectrum * mask)


def band_pass_filter(
    img: np.ndarray,
    sigma_low: float = 5.0,
    sigma_high: float = 20.0,
) -> np.ndarray:
    """Применить полосовой фильтр (сохранить полосу между sigma_low и sigma_high).

    Args:
        img:        Изображение uint8 (H, W) или (H, W, C).
        sigma_low:  Нижняя граница (> 0).
        sigma_high: Верхняя граница (> sigma_low).

    Returns:
        Отфильтрованное изображение uint8 (H, W).

    Raises:
        ValueError: Если параметры сигма некорректны.
    """
    if sigma_low <= 0:
        raise ValueError(f"sigma_low must be > 0, got {sigma_low}")
    if sigma_high <= sigma_low:
        raise ValueError(
            f"sigma_high ({sigma_high}) must be > sigma_low ({sigma_low})"
        )
    spectrum = fft_image(img)
    h, w = spectrum.shape
    lp = _gaussian_mask(h, w, sigma_high)
    hp = 1.0 - _gaussian_mask(h, w, sigma_low)
    mask = lp * hp
    return ifft_image(spectrum * mask)


def notch_filter(
    img: np.ndarray,
    notch_freqs: Optional[List[Tuple[int, int]]] = None,
    radius: float = 5.0,
) -> np.ndarray:
    """Применить режекторный фильтр (подавить заданные частоты).

    Args:
        img:         Изображение uint8 (H, W) или (H, W, C).
        notch_freqs: Список пар (u, v) — координаты в сдвинутом спектре
                     относительно центра. Если ``None`` — возвращает исходное.
        radius:      Радиус режекторного окна (> 0).

    Returns:
        Отфильтрованное изображение uint8 (H, W).

    Raises:
        ValueError: Если ``radius`` ≤ 0.
    """
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")
    spectrum = fft_image(img)
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2

    if notch_freqs:
        ys, xs = np.ogrid[:h, :w]
        for du, dv in notch_freqs:
            # Подавить окрестность (cx+du, cy+dv)
            dist = np.sqrt((xs - (cx + du)) ** 2 + (ys - (cy + dv)) ** 2)
            spectrum[dist <= radius] = 0.0
            # Симметричная точка
            dist_sym = np.sqrt((xs - (cx - du)) ** 2 + (ys - (cy - dv)) ** 2)
            spectrum[dist_sym <= radius] = 0.0

    return ifft_image(spectrum)


def apply_frequency_filter(
    img: np.ndarray,
    params: FrequencyFilterParams,
) -> np.ndarray:
    """Применить частотный фильтр согласно параметрам.

    Args:
        img:    Изображение uint8 (H, W) или (H, W, C).
        params: Параметры фильтра.

    Returns:
        Отфильтрованное изображение uint8 (H, W).
    """
    if params.filter_type == "low_pass":
        return gaussian_low_pass(img, sigma=params.sigma_low)
    if params.filter_type == "high_pass":
        return gaussian_high_pass(img, sigma=params.sigma_low)
    if params.filter_type == "band_pass":
        return band_pass_filter(img, params.sigma_low, params.sigma_high)
    # notch — без явных частот применяем identity (notch_freqs=None)
    return notch_filter(img, notch_freqs=None, radius=params.notch_radius)


def batch_frequency_filter(
    images: List[np.ndarray],
    params: Optional[FrequencyFilterParams] = None,
) -> List[np.ndarray]:
    """Пакетное применение частотного фильтра.

    Args:
        images: Список изображений uint8.
        params: Параметры фильтра. Если ``None`` — используются значения
                по умолчанию (low_pass, sigma=10).

    Returns:
        Список отфильтрованных изображений того же размера.
    """
    if params is None:
        params = FrequencyFilterParams()
    return [apply_frequency_filter(img, params) for img in images]


# ─── Приватные ───────────────────────────────────────────────────────────────

def _to_gray_float(img: np.ndarray) -> np.ndarray:
    """Конвертировать изображение в float64 оттенки серого."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        gray = img
    else:
        raise ValueError(
            f"img must be 2-D or 3-D, got ndim={img.ndim}"
        )
    return gray.astype(np.float64)


def _gaussian_mask(h: int, w: int, sigma: float) -> np.ndarray:
    """Создать Гауссов маску (H, W) float64 с нулевой частотой в центре."""
    cy, cx = h // 2, w // 2
    ys, xs = np.ogrid[:h, :w]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    return np.exp(-d2 / (2.0 * sigma ** 2))


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Нормализовать вещественный массив в uint8 [0, 255]."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.zeros(arr.shape, dtype=np.uint8)
    normalized = (arr - mn) / (mx - mn) * 255.0
    return normalized.astype(np.uint8)
