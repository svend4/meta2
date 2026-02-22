"""
Утилиты частотного анализа изображений для реконструкции пазла.

Предоставляет функции вычисления спектров, фильтрации в частотной области
и сравнения частотных признаков фрагментов.

Классы:
    FrequencyConfig  — параметры частотного анализа

Функции:
    compute_fft_magnitude    — спектр амплитуд (логарифмическая шкала)
    radial_power_spectrum    — радиально усреднённый спектр мощности
    frequency_band_energy    — энергия в частотной полосе
    high_frequency_ratio     — доля высокочастотной энергии
    low_pass_filter          — фильтрация нижних частот (НЧ)
    high_pass_filter         — фильтрация высоких частот (ВЧ)
    compare_frequency_spectra — схожесть двух спектров ∈ [0,1]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


# ─── FrequencyConfig ──────────────────────────────────────────────────────────

@dataclass
class FrequencyConfig:
    """Параметры частотного анализа.

    Attributes:
        log_scale:    Применять логарифмическую шкалу к спектру.
        center_zero:  Центрировать нулевую частоту в центре спектра.
        normalize:    Нормировать выход в [0, 1].
        n_bands:      Число радиальных полос для радиального спектра (>= 2).
    """
    log_scale: bool = True
    center_zero: bool = True
    normalize: bool = True
    n_bands: int = 32

    def __post_init__(self) -> None:
        if self.n_bands < 2:
            raise ValueError(f"n_bands must be >= 2, got {self.n_bands}")


# ─── Вспомогательные ──────────────────────────────────────────────────────────

def _to_gray_float(img: np.ndarray) -> np.ndarray:
    """Преобразовать изображение в float32 grayscale."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def _radial_indices(h: int, w: int) -> np.ndarray:
    """Вернуть матрицу радиальных расстояний от центра."""
    cy, cx = h / 2.0, w / 2.0
    y = np.arange(h, dtype=np.float32) - cy
    x = np.arange(w, dtype=np.float32) - cx
    return np.sqrt(y[:, None] ** 2 + x[None, :] ** 2)


# ─── Публичные функции ────────────────────────────────────────────────────────

def compute_fft_magnitude(
    img: np.ndarray,
    cfg: FrequencyConfig | None = None,
) -> np.ndarray:
    """Вычислить спектр амплитуд изображения через 2D FFT.

    Args:
        img: Grayscale или BGR изображение.
        cfg: Параметры; None → FrequencyConfig().

    Returns:
        float32 матрица той же формы, что и входное изображение.

    Raises:
        ValueError: Если изображение не 2D/3D.
    """
    if img.ndim not in (2, 3):
        raise ValueError(f"img must be 2D or 3D, got ndim={img.ndim}")
    if cfg is None:
        cfg = FrequencyConfig()

    gray = _to_gray_float(img)
    fft = np.fft.fft2(gray)
    if cfg.center_zero:
        fft = np.fft.fftshift(fft)
    mag = np.abs(fft).astype(np.float32)
    if cfg.log_scale:
        mag = np.log1p(mag)
    if cfg.normalize:
        mx = float(mag.max())
        if mx > 0:
            mag = mag / mx
    return mag


def radial_power_spectrum(
    img: np.ndarray,
    cfg: FrequencyConfig | None = None,
) -> np.ndarray:
    """Вычислить радиально усреднённый спектр мощности.

    Усредняет мощность FFT по кольцевым полосам, создавая 1D профиль.

    Args:
        img: Grayscale или BGR изображение.
        cfg: Параметры (cfg.n_bands задаёт число полос); None → FrequencyConfig().

    Returns:
        float32 массив длиной cfg.n_bands.

    Raises:
        ValueError: Если изображение не 2D/3D.
    """
    if img.ndim not in (2, 3):
        raise ValueError(f"img must be 2D or 3D, got ndim={img.ndim}")
    if cfg is None:
        cfg = FrequencyConfig()

    gray = _to_gray_float(img)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    power = (np.abs(fft) ** 2).astype(np.float64)
    h, w = power.shape
    radii = _radial_indices(h, w)
    max_r = float(radii.max())
    band_width = max_r / cfg.n_bands if max_r > 0 else 1.0

    spectrum = np.zeros(cfg.n_bands, dtype=np.float32)
    for k in range(cfg.n_bands):
        r_min = k * band_width
        r_max = (k + 1) * band_width
        mask = (radii >= r_min) & (radii < r_max)
        if mask.any():
            spectrum[k] = float(power[mask].mean())

    if cfg.normalize:
        mx = float(spectrum.max())
        if mx > 0:
            spectrum = spectrum / mx
    return spectrum


def frequency_band_energy(
    img: np.ndarray,
    low_frac: float = 0.0,
    high_frac: float = 0.5,
) -> float:
    """Вычислить суммарную энергию в радиальной частотной полосе.

    Args:
        img:      Grayscale или BGR изображение.
        low_frac: Нижняя граница полосы как доля максимального радиуса [0, 1).
        high_frac: Верхняя граница полосы [0, 1].

    Returns:
        Неотрицательная суммарная энергия в полосе.

    Raises:
        ValueError: Если low_frac >= high_frac или они вне [0, 1].
    """
    if not (0.0 <= low_frac < high_frac <= 1.0):
        raise ValueError(
            f"Require 0 <= low_frac < high_frac <= 1, "
            f"got low_frac={low_frac}, high_frac={high_frac}"
        )
    gray = _to_gray_float(img)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    power = (np.abs(fft) ** 2).astype(np.float64)
    h, w = power.shape
    radii = _radial_indices(h, w)
    max_r = float(radii.max())
    r_low = low_frac * max_r
    r_high = high_frac * max_r
    mask = (radii >= r_low) & (radii <= r_high)
    return float(power[mask].sum())


def high_frequency_ratio(
    img: np.ndarray,
    threshold_frac: float = 0.5,
) -> float:
    """Вычислить долю высокочастотной энергии ∈ [0, 1].

    Args:
        img:            Grayscale или BGR изображение.
        threshold_frac: Относительный радиус разделения [0, 1].

    Returns:
        float ∈ [0, 1]; 1 = вся энергия в высоких частотах.

    Raises:
        ValueError: Если threshold_frac вне (0, 1).
    """
    if not (0.0 < threshold_frac < 1.0):
        raise ValueError(
            f"threshold_frac must be in (0, 1), got {threshold_frac}"
        )
    low_energy = frequency_band_energy(img, 0.0, threshold_frac)
    high_energy = frequency_band_energy(img, threshold_frac, 1.0)
    total = low_energy + high_energy
    if total < 1e-12:
        return 0.0
    return float(np.clip(high_energy / total, 0.0, 1.0))


def low_pass_filter(
    img: np.ndarray,
    cutoff_frac: float = 0.3,
) -> np.ndarray:
    """Применить НЧ-фильтр в частотной области.

    Args:
        img:         Grayscale или BGR изображение (uint8).
        cutoff_frac: Граница среза как доля максимального радиуса (0, 1].

    Returns:
        uint8 изображение того же размера и числа каналов.

    Raises:
        ValueError: Если cutoff_frac вне (0, 1].
    """
    if not (0.0 < cutoff_frac <= 1.0):
        raise ValueError(
            f"cutoff_frac must be in (0, 1], got {cutoff_frac}"
        )
    is_color = img.ndim == 3
    gray = _to_gray_float(img)
    h, w = gray.shape
    fft = np.fft.fftshift(np.fft.fft2(gray))
    radii = _radial_indices(h, w)
    mask = radii <= (cutoff_frac * float(radii.max()))
    fft_filtered = fft * mask
    result = np.real(np.fft.ifft2(np.fft.ifftshift(fft_filtered)))
    result = np.clip(result, 0, 255).astype(np.uint8)
    if is_color:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result


def high_pass_filter(
    img: np.ndarray,
    cutoff_frac: float = 0.3,
) -> np.ndarray:
    """Применить ВЧ-фильтр в частотной области.

    Args:
        img:         Grayscale или BGR изображение (uint8).
        cutoff_frac: Граница среза как доля максимального радиуса [0, 1).

    Returns:
        uint8 изображение того же размера и числа каналов.

    Raises:
        ValueError: Если cutoff_frac вне [0, 1).
    """
    if not (0.0 <= cutoff_frac < 1.0):
        raise ValueError(
            f"cutoff_frac must be in [0, 1), got {cutoff_frac}"
        )
    is_color = img.ndim == 3
    gray = _to_gray_float(img)
    h, w = gray.shape
    fft = np.fft.fftshift(np.fft.fft2(gray))
    radii = _radial_indices(h, w)
    mask = radii > (cutoff_frac * float(radii.max()))
    fft_filtered = fft * mask
    result = np.real(np.fft.ifft2(np.fft.ifftshift(fft_filtered)))
    result = np.clip(result, 0, 255).astype(np.uint8)
    if is_color:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result


def compare_frequency_spectra(
    img1: np.ndarray,
    img2: np.ndarray,
    cfg: FrequencyConfig | None = None,
) -> float:
    """Сравнить частотные спектры двух изображений.

    Вычисляет радиальные спектры мощности и возвращает их косинусное
    сходство ∈ [0, 1].

    Args:
        img1, img2: Grayscale или BGR изображения.
        cfg:        Параметры; None → FrequencyConfig().

    Returns:
        float ∈ [0, 1]; 1 = идентичные спектры.
    """
    if cfg is None:
        cfg = FrequencyConfig()
    s1 = radial_power_spectrum(img1, cfg).astype(np.float64)
    s2 = radial_power_spectrum(img2, cfg).astype(np.float64)
    n1 = float(np.linalg.norm(s1))
    n2 = float(np.linalg.norm(s2))
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_sim = float(np.dot(s1, s2) / (n1 * n2))
    return float(np.clip(cos_sim, 0.0, 1.0))
