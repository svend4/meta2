"""Частотный анализ изображений фрагментов.

Модуль предоставляет функции для вычисления спектра мощности, извлечения
частотных дескрипторов (энергия полос, доминирующие частоты, спектральный
центроид) и сравнения частотных профилей.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ─── FreqConfig ───────────────────────────────────────────────────────────────

@dataclass
class FreqConfig:
    """Параметры частотного анализа.

    Атрибуты:
        n_bands:      Число частотных полос (>= 2).
        log_scale:    Использовать логарифмическую шкалу спектра.
        normalize:    Нормировать спектр мощности в [0, 1].
        n_top_freqs:  Число доминирующих частот для извлечения (>= 1).
    """

    n_bands: int = 8
    log_scale: bool = True
    normalize: bool = True
    n_top_freqs: int = 5

    def __post_init__(self) -> None:
        if self.n_bands < 2:
            raise ValueError(
                f"n_bands должен быть >= 2, получено {self.n_bands}"
            )
        if self.n_top_freqs < 1:
            raise ValueError(
                f"n_top_freqs должен быть >= 1, получено {self.n_top_freqs}"
            )


# ─── FreqSpectrum ─────────────────────────────────────────────────────────────

@dataclass
class FreqSpectrum:
    """Спектр мощности изображения.

    Атрибуты:
        magnitude:    2D-массив амплитуд (h x w).
        power:        2D-массив мощности (h x w).
        total_power:  Суммарная мощность (>= 0).
    """

    magnitude: np.ndarray
    power: np.ndarray
    total_power: float

    def __post_init__(self) -> None:
        if self.total_power < 0.0:
            raise ValueError(
                f"total_power должен быть >= 0, получено {self.total_power}"
            )

    @property
    def shape(self) -> Tuple[int, int]:
        """Размер спектра (h, w)."""
        return self.magnitude.shape[:2]

    @property
    def dc_component(self) -> float:
        """DC-компонента (нулевая частота)."""
        h, w = self.shape
        return float(self.power[h // 2, w // 2])


# ─── FreqDescriptor ───────────────────────────────────────────────────────────

@dataclass
class FreqDescriptor:
    """Частотный дескриптор фрагмента.

    Атрибуты:
        fragment_id:   ID фрагмента (>= 0).
        band_energies: Энергия в каждой полосе (len == n_bands).
        centroid:      Спектральный центроид (>= 0).
        top_freqs:     Доминирующие частоты (нормированные, [0, 1]).
        entropy:       Спектральная энтропия (>= 0).
    """

    fragment_id: int
    band_energies: List[float]
    centroid: float
    top_freqs: List[float]
    entropy: float

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        if not self.band_energies:
            raise ValueError("band_energies не должен быть пустым")
        if self.centroid < 0.0:
            raise ValueError(
                f"centroid должен быть >= 0, получено {self.centroid}"
            )
        if self.entropy < 0.0:
            raise ValueError(
                f"entropy должен быть >= 0, получено {self.entropy}"
            )

    @property
    def n_bands(self) -> int:
        """Число частотных полос."""
        return len(self.band_energies)

    @property
    def dominant_band(self) -> int:
        """Индекс наиболее энергетичной полосы."""
        return int(np.argmax(self.band_energies))

    @property
    def high_freq_ratio(self) -> float:
        """Доля энергии в верхней половине полос."""
        mid = len(self.band_energies) // 2
        total = sum(self.band_energies) + 1e-12
        return float(sum(self.band_energies[mid:])) / total


# ─── compute_power_spectrum ───────────────────────────────────────────────────

def compute_power_spectrum(
    image: np.ndarray,
    log_scale: bool = True,
    normalize: bool = True,
) -> FreqSpectrum:
    """Вычислить спектр мощности изображения.

    Аргументы:
        image:      Изображение (2D или 3D; float или uint8).
        log_scale:  Логарифмическая шкала амплитуд.
        normalize:  Нормировать амплитуды в [0, 1].

    Возвращает:
        FreqSpectrum.

    Исключения:
        ValueError: Если изображение имеет некорректные размеры.
    """
    arr = np.asarray(image, dtype=float)
    if arr.ndim == 3:
        arr = np.mean(arr, axis=2)
    if arr.ndim != 2:
        raise ValueError(
            f"Изображение должно быть 2D или 3D, получено ndim={arr.ndim}"
        )
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError(
            f"Изображение должно быть не менее 2x2, "
            f"получено {arr.shape}"
        )

    fshift = np.fft.fftshift(np.fft.fft2(arr))
    magnitude = np.abs(fshift)
    power = magnitude ** 2

    if log_scale:
        magnitude = np.log1p(magnitude)

    if normalize:
        m_max = magnitude.max()
        magnitude = magnitude / (m_max + 1e-12)

    return FreqSpectrum(
        magnitude=magnitude,
        power=power,
        total_power=float(power.sum()),
    )


# ─── compute_band_energies ────────────────────────────────────────────────────

def compute_band_energies(
    spectrum: FreqSpectrum,
    n_bands: int = 8,
) -> List[float]:
    """Разбить спектр на кольцевые полосы и вычислить энергию каждой.

    Аргументы:
        spectrum: FreqSpectrum.
        n_bands:  Число полос (>= 2).

    Возвращает:
        Список энергий.

    Исключения:
        ValueError: Если n_bands < 2.
    """
    if n_bands < 2:
        raise ValueError(f"n_bands должен быть >= 2, получено {n_bands}")

    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx)

    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    band_energies: List[float] = []
    for i in range(n_bands):
        r_low = max_r * i / n_bands
        r_high = max_r * (i + 1) / n_bands
        mask = (rr >= r_low) & (rr < r_high)
        energy = float(spectrum.power[mask].sum())
        band_energies.append(energy)

    return band_energies


# ─── compute_spectral_centroid ────────────────────────────────────────────────

def compute_spectral_centroid(spectrum: FreqSpectrum) -> float:
    """Вычислить спектральный центроид (средневзвешенное расстояние от DC).

    Аргументы:
        spectrum: FreqSpectrum.

    Возвращает:
        Нормированный центроид в [0, 1].
    """
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    max_r = float(min(cy, cx)) + 1e-12

    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    total_power = spectrum.power.sum() + 1e-12
    centroid = float((rr * spectrum.power).sum() / total_power)
    return float(np.clip(centroid / max_r, 0.0, 1.0))


# ─── compute_spectral_entropy ─────────────────────────────────────────────────

def compute_spectral_entropy(spectrum: FreqSpectrum) -> float:
    """Вычислить спектральную энтропию (мера равномерности распределения).

    Аргументы:
        spectrum: FreqSpectrum.

    Возвращает:
        Энтропия (>= 0).
    """
    power = spectrum.power.ravel()
    total = power.sum() + 1e-12
    p = power / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p + 1e-12)))


# ─── extract_top_frequencies ──────────────────────────────────────────────────

def extract_top_frequencies(
    spectrum: FreqSpectrum,
    n_top: int = 5,
) -> List[float]:
    """Извлечь нормированные частоты с наибольшей мощностью.

    Аргументы:
        spectrum: FreqSpectrum.
        n_top:    Число частот (>= 1).

    Возвращает:
        Список нормированных частот в порядке убывания мощности.

    Исключения:
        ValueError: Если n_top < 1.
    """
    if n_top < 1:
        raise ValueError(f"n_top должен быть >= 1, получено {n_top}")

    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    max_r = float(min(cy, cx)) + 1e-12

    power_flat = spectrum.power.ravel()
    top_idx = np.argsort(power_flat)[::-1][:n_top]

    freqs: List[float] = []
    for idx in top_idx:
        iy, ix = np.unravel_index(idx, spectrum.shape)
        r = float(np.sqrt((iy - cy) ** 2 + (ix - cx) ** 2))
        freqs.append(float(np.clip(r / max_r, 0.0, 1.0)))
    return freqs


# ─── extract_freq_descriptor ──────────────────────────────────────────────────

def extract_freq_descriptor(
    image: np.ndarray,
    fragment_id: int = 0,
    cfg: Optional[FreqConfig] = None,
) -> FreqDescriptor:
    """Извлечь частотный дескриптор из изображения фрагмента.

    Аргументы:
        image:       Изображение (2D или 3D).
        fragment_id: ID фрагмента (>= 0).
        cfg:         Параметры (None → FreqConfig()).

    Возвращает:
        FreqDescriptor.

    Исключения:
        ValueError: Если fragment_id < 0.
    """
    if cfg is None:
        cfg = FreqConfig()
    if fragment_id < 0:
        raise ValueError(
            f"fragment_id должен быть >= 0, получено {fragment_id}"
        )

    spectrum = compute_power_spectrum(image, cfg.log_scale, cfg.normalize)
    band_energies = compute_band_energies(spectrum, cfg.n_bands)
    centroid = compute_spectral_centroid(spectrum)
    entropy = compute_spectral_entropy(spectrum)
    top_freqs = extract_top_frequencies(spectrum, cfg.n_top_freqs)

    return FreqDescriptor(
        fragment_id=fragment_id,
        band_energies=band_energies,
        centroid=centroid,
        top_freqs=top_freqs,
        entropy=entropy,
    )


# ─── compare_freq_descriptors ─────────────────────────────────────────────────

def compare_freq_descriptors(
    a: FreqDescriptor,
    b: FreqDescriptor,
) -> float:
    """Сравнить два частотных дескриптора (косинусное сходство полос).

    Аргументы:
        a: Первый дескриптор.
        b: Второй дескриптор.

    Возвращает:
        Сходство в [0, 1].

    Исключения:
        ValueError: Если n_bands у дескрипторов различаются.
    """
    if a.n_bands != b.n_bands:
        raise ValueError(
            f"Число полос должно совпадать: {a.n_bands} != {b.n_bands}"
        )
    va = np.array(a.band_energies, dtype=float)
    vb = np.array(b.band_energies, dtype=float)
    na = np.linalg.norm(va) + 1e-12
    nb = np.linalg.norm(vb) + 1e-12
    return float(np.clip(np.dot(va / na, vb / nb), 0.0, 1.0))


# ─── batch_extract_freq_descriptors ──────────────────────────────────────────

def batch_extract_freq_descriptors(
    images: List[np.ndarray],
    cfg: Optional[FreqConfig] = None,
) -> List[FreqDescriptor]:
    """Извлечь частотные дескрипторы для списка изображений.

    Аргументы:
        images: Список изображений.
        cfg:    Параметры.

    Возвращает:
        Список FreqDescriptor.
    """
    return [extract_freq_descriptor(img, fid, cfg)
            for fid, img in enumerate(images)]
