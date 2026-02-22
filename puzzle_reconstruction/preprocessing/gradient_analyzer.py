"""Анализ градиентов изображений фрагментов пазла.

Модуль вычисляет магнитуду, направление и ориентационные гистограммы
градиентов, позволяя оценивать резкость краёв и ориентацию доминирующих
структур в изображении.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


_VALID_KERNELS = {"sobel", "scharr", "prewitt"}


# ─── GradientConfig ───────────────────────────────────────────────────────────

@dataclass
class GradientConfig:
    """Параметры анализа градиентов.

    Атрибуты:
        kernel:      'sobel' | 'scharr' | 'prewitt'.
        ksize:       Размер ядра Собеля (3, 5 или 7; игнорируется для scharr/prewitt).
        n_bins:      Число бинов ориентационной гистограммы (>= 2).
        normalize:   Нормировать гистограмму в [0, 1].
    """

    kernel: str = "sobel"
    ksize: int = 3
    n_bins: int = 8
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.kernel not in _VALID_KERNELS:
            raise ValueError(
                f"kernel должен быть одним из {_VALID_KERNELS}, "
                f"получено '{self.kernel}'"
            )
        if self.ksize not in (3, 5, 7):
            raise ValueError(
                f"ksize должен быть 3, 5 или 7, получено {self.ksize}"
            )
        if self.n_bins < 2:
            raise ValueError(
                f"n_bins должен быть >= 2, получено {self.n_bins}"
            )


# ─── GradientMap ─────────────────────────────────────────────────────────────

@dataclass
class GradientMap:
    """Результат вычисления градиентов изображения.

    Атрибуты:
        magnitude:  2D-массив магнитуд (float64, >= 0).
        angle:      2D-массив углов в градусах [0, 180).
        mean_mag:   Среднее значение магнитуды (>= 0).
        max_mag:    Максимум магнитуды (>= 0).
        kernel:     Использованное ядро.
    """

    magnitude: np.ndarray
    angle: np.ndarray
    mean_mag: float
    max_mag: float
    kernel: str

    def __post_init__(self) -> None:
        if self.mean_mag < 0.0:
            raise ValueError(
                f"mean_mag должен быть >= 0, получено {self.mean_mag}"
            )
        if self.max_mag < 0.0:
            raise ValueError(
                f"max_mag должен быть >= 0, получено {self.max_mag}"
            )

    @property
    def shape(self) -> Tuple[int, ...]:
        """Форма массива магнитуд."""
        return self.magnitude.shape


# ─── GradientProfile ─────────────────────────────────────────────────────────

@dataclass
class GradientProfile:
    """Агрегированный профиль градиентов фрагмента.

    Атрибуты:
        fragment_id:      Идентификатор фрагмента (>= 0).
        mean_magnitude:   Средняя магнитуда (>= 0).
        std_magnitude:    СКО магнитуды (>= 0).
        max_magnitude:    Максимальная магнитуда (>= 0).
        energy:           Сумма квадратов магнитуд (>= 0).
        dominant_angle:   Доминирующий угол ориентации [0, 180).
        orientation_hist: Нормированная гистограмма ориентаций.
    """

    fragment_id: int
    mean_magnitude: float = 0.0
    std_magnitude: float = 0.0
    max_magnitude: float = 0.0
    energy: float = 0.0
    dominant_angle: float = 0.0
    orientation_hist: Optional[np.ndarray] = field(default=None)

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        for name, val in (
            ("mean_magnitude", self.mean_magnitude),
            ("std_magnitude", self.std_magnitude),
            ("max_magnitude", self.max_magnitude),
            ("energy", self.energy),
        ):
            if val < 0.0:
                raise ValueError(
                    f"{name} должен быть >= 0, получено {val}"
                )

    @property
    def sharpness_score(self) -> float:
        """Оценка резкости на основе средней магнитуды (0–1, нормировано на 255)."""
        return float(np.clip(self.mean_magnitude / 255.0, 0.0, 1.0))


# ─── compute_gradient_map ─────────────────────────────────────────────────────

def compute_gradient_map(
    image: np.ndarray,
    cfg: Optional[GradientConfig] = None,
) -> GradientMap:
    """Вычислить карту градиентов изображения.

    Аргументы:
        image: Серое или цветное изображение (uint8).
        cfg:   Параметры (None → GradientConfig()).

    Возвращает:
        GradientMap с magnitude, angle, mean_mag, max_mag.

    Исключения:
        ValueError: Если image не 2D или 3D.
    """
    if cfg is None:
        cfg = GradientConfig()

    img = np.asarray(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    elif img.ndim != 2:
        raise ValueError(
            f"image должно быть 2D или 3D, получено ndim={img.ndim}"
        )

    gray = img.astype(np.float64)

    if cfg.kernel == "scharr":
        gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    elif cfg.kernel == "prewitt":
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
        ky = kx.T
        gx = cv2.filter2D(gray, cv2.CV_64F, kx)
        gy = cv2.filter2D(gray, cv2.CV_64F, ky)
    else:  # sobel
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=cfg.ksize)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=cfg.ksize)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.degrees(np.arctan2(np.abs(gy), np.abs(gx))) % 180.0

    return GradientMap(
        magnitude=magnitude,
        angle=angle,
        mean_mag=float(magnitude.mean()),
        max_mag=float(magnitude.max()),
        kernel=cfg.kernel,
    )


# ─── compute_orientation_histogram ────────────────────────────────────────────

def compute_orientation_histogram(
    gmap: GradientMap,
    n_bins: int = 8,
    normalize: bool = True,
) -> np.ndarray:
    """Вычислить взвешенную гистограмму ориентаций градиентов.

    Аргументы:
        gmap:      Карта градиентов.
        n_bins:    Число бинов (>= 2).
        normalize: Нормировать на суммарную магнитуду.

    Возвращает:
        1D-массив (n_bins,) float64.

    Исключения:
        ValueError: Если n_bins < 2.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins должен быть >= 2, получено {n_bins}")

    hist, _ = np.histogram(
        gmap.angle.ravel(),
        bins=n_bins,
        range=(0.0, 180.0),
        weights=gmap.magnitude.ravel(),
    )
    hist = hist.astype(np.float64)
    if normalize:
        total = hist.sum()
        if total > 1e-12:
            hist /= total
    return hist


# ─── extract_gradient_profile ─────────────────────────────────────────────────

def extract_gradient_profile(
    image: np.ndarray,
    fragment_id: int = 0,
    cfg: Optional[GradientConfig] = None,
) -> GradientProfile:
    """Извлечь агрегированный профиль градиентов фрагмента.

    Аргументы:
        image:       Изображение фрагмента.
        fragment_id: Идентификатор (>= 0).
        cfg:         Параметры.

    Возвращает:
        GradientProfile.

    Исключения:
        ValueError: Если fragment_id < 0.
    """
    if fragment_id < 0:
        raise ValueError(
            f"fragment_id должен быть >= 0, получено {fragment_id}"
        )
    if cfg is None:
        cfg = GradientConfig()

    gmap = compute_gradient_map(image, cfg)
    hist = compute_orientation_histogram(gmap, cfg.n_bins, cfg.normalize)
    dominant_bin = int(np.argmax(hist))
    bin_width = 180.0 / cfg.n_bins
    dominant_angle = dominant_bin * bin_width + bin_width / 2.0

    mag = gmap.magnitude
    return GradientProfile(
        fragment_id=fragment_id,
        mean_magnitude=float(mag.mean()),
        std_magnitude=float(mag.std()),
        max_magnitude=float(mag.max()),
        energy=float((mag ** 2).sum()),
        dominant_angle=dominant_angle,
        orientation_hist=hist,
    )


# ─── compare_gradient_profiles ────────────────────────────────────────────────

def compare_gradient_profiles(
    a: GradientProfile,
    b: GradientProfile,
) -> float:
    """Сравнить профили по гистограммам ориентаций (косинусное сходство).

    Аргументы:
        a, b: Профили для сравнения.

    Возвращает:
        Косинусное сходство в [-1, 1].
    """
    if a.orientation_hist is None or b.orientation_hist is None:
        return 0.0
    ha = a.orientation_hist.astype(np.float64)
    hb = b.orientation_hist.astype(np.float64)
    na = float(np.linalg.norm(ha))
    nb = float(np.linalg.norm(hb))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(ha, hb) / (na * nb))


# ─── batch_extract_gradient_profiles ─────────────────────────────────────────

def batch_extract_gradient_profiles(
    images: List[np.ndarray],
    cfg: Optional[GradientConfig] = None,
) -> List[GradientProfile]:
    """Извлечь профили градиентов для списка изображений.

    Аргументы:
        images: Список изображений.
        cfg:    Параметры.

    Возвращает:
        Список GradientProfile (fragment_id = порядковый индекс).
    """
    return [
        extract_gradient_profile(img, fragment_id=i, cfg=cfg)
        for i, img in enumerate(images)
    ]
