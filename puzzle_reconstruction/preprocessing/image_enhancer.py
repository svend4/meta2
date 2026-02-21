"""Улучшение качества изображений фрагментов пазла.

Модуль реализует pipeline улучшения: повышение резкости (unsharp mask),
подавление шума (Gaussian, Median, Bilateral) и улучшение контраста
(minmax-stretch, CLAHE) с поддержкой батч-обработки.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np


_SHARPNESS_MODES = {"none", "mild", "strong"}
_DENOISE_MODES = {"none", "gaussian", "median", "bilateral"}
_CONTRAST_MODES = {"none", "stretch", "clahe"}


# ─── EnhanceConfig ────────────────────────────────────────────────────────────

@dataclass
class EnhanceConfig:
    """Параметры улучшения изображения.

    Атрибуты:
        sharpness:   Режим повышения резкости: 'none' | 'mild' | 'strong'.
        denoise:     Режим подавления шума: 'none' | 'gaussian' | 'median' | 'bilateral'.
        contrast:    Режим улучшения контраста: 'none' | 'stretch' | 'clahe'.
        kernel_size: Размер ядра для фильтров (нечётное, >= 3).
    """

    sharpness: str = "none"
    denoise: str = "none"
    contrast: str = "none"
    kernel_size: int = 3

    def __post_init__(self) -> None:
        if self.sharpness not in _SHARPNESS_MODES:
            raise ValueError(
                f"sharpness должен быть одним из {_SHARPNESS_MODES}, "
                f"получено '{self.sharpness}'"
            )
        if self.denoise not in _DENOISE_MODES:
            raise ValueError(
                f"denoise должен быть одним из {_DENOISE_MODES}, "
                f"получено '{self.denoise}'"
            )
        if self.contrast not in _CONTRAST_MODES:
            raise ValueError(
                f"contrast должен быть одним из {_CONTRAST_MODES}, "
                f"получено '{self.contrast}'"
            )
        if self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size должен быть нечётным и >= 3, "
                f"получено {self.kernel_size}"
            )


# ─── EnhanceResult ────────────────────────────────────────────────────────────

@dataclass
class EnhanceResult:
    """Результат улучшения изображения.

    Атрибуты:
        image:       Обработанное изображение (uint8).
        operations:  Список применённых операций.
        mean_before: Средняя яркость до обработки (>= 0).
        mean_after:  Средняя яркость после обработки (>= 0).
    """

    image: np.ndarray
    operations: List[str]
    mean_before: float
    mean_after: float

    def __post_init__(self) -> None:
        if self.mean_before < 0.0:
            raise ValueError(
                f"mean_before должен быть >= 0, получено {self.mean_before}"
            )
        if self.mean_after < 0.0:
            raise ValueError(
                f"mean_after должен быть >= 0, получено {self.mean_after}"
            )

    @property
    def delta_mean(self) -> float:
        """Изменение средней яркости (mean_after - mean_before)."""
        return self.mean_after - self.mean_before


# ─── sharpen_image ────────────────────────────────────────────────────────────

def sharpen_image(
    image: np.ndarray,
    mode: str = "mild",
    kernel_size: int = 3,
) -> np.ndarray:
    """Повысить резкость изображения методом unsharp mask.

    Аргументы:
        image:       Входное изображение (uint8, grayscale или BGR).
        mode:        'mild' или 'strong'.
        kernel_size: Размер ядра размытия (нечётное >= 3).

    Возвращает:
        Изображение uint8 с повышенной резкостью.

    Исключения:
        ValueError: Если mode неверный или kernel_size < 3 / чётный.
    """
    if mode not in ("mild", "strong"):
        raise ValueError(f"mode должен быть 'mild' или 'strong', получено '{mode}'")
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size должен быть нечётным и >= 3, получено {kernel_size}"
        )

    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    strength = 1.5 if mode == "mild" else 2.5
    sharpened = cv2.addWeighted(image, strength, blurred, -(strength - 1.0), 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ─── denoise_image ────────────────────────────────────────────────────────────

def denoise_image(
    image: np.ndarray,
    mode: str = "gaussian",
    kernel_size: int = 3,
) -> np.ndarray:
    """Подавить шум на изображении.

    Аргументы:
        image:       Входное изображение (uint8).
        mode:        'gaussian' | 'median' | 'bilateral'.
        kernel_size: Размер ядра (нечётное >= 3).

    Возвращает:
        Сглаженное изображение uint8.

    Исключения:
        ValueError: Если mode неверный или kernel_size невалиден.
    """
    if mode not in ("gaussian", "median", "bilateral"):
        raise ValueError(
            f"mode должен быть 'gaussian', 'median' или 'bilateral', "
            f"получено '{mode}'"
        )
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size должен быть нечётным и >= 3, получено {kernel_size}"
        )

    if mode == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    if mode == "median":
        return cv2.medianBlur(image, kernel_size)
    # bilateral
    sigma = kernel_size * 5
    return cv2.bilateralFilter(image, kernel_size, sigma, sigma)


# ─── enhance_contrast ─────────────────────────────────────────────────────────

def enhance_contrast(
    image: np.ndarray,
    mode: str = "stretch",
    kernel_size: int = 3,
) -> np.ndarray:
    """Улучшить контраст изображения.

    Аргументы:
        image:       Входное изображение (uint8).
        mode:        'stretch' (minmax) | 'clahe'.
        kernel_size: Размер тайла для CLAHE (нечётное >= 3).

    Возвращает:
        Изображение uint8 с улучшенным контрастом.

    Исключения:
        ValueError: Если mode неверный.
    """
    if mode not in ("stretch", "clahe"):
        raise ValueError(
            f"mode должен быть 'stretch' или 'clahe', получено '{mode}'"
        )
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size должен быть нечётным и >= 3, получено {kernel_size}"
        )

    if mode == "stretch":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        min_val = float(gray.min())
        max_val = float(gray.max())
        if max_val - min_val < 1e-6:
            return image.copy()
        stretched = ((gray.astype(np.float32) - min_val) /
                     (max_val - min_val) * 255.0).astype(np.uint8)
        if image.ndim == 3:
            return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)
        return stretched

    # clahe
    tile = kernel_size * 8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile, tile))
    if image.ndim == 3:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return clahe.apply(image)


# ─── enhance_image ────────────────────────────────────────────────────────────

def enhance_image(
    image: np.ndarray,
    cfg: Optional[EnhanceConfig] = None,
) -> EnhanceResult:
    """Применить pipeline улучшения изображения.

    Аргументы:
        image: Входное изображение uint8.
        cfg:   Параметры (None → EnhanceConfig()).

    Возвращает:
        EnhanceResult.
    """
    if cfg is None:
        cfg = EnhanceConfig()

    img = image.astype(np.uint8)
    mean_before = float(img.mean())
    ops: list = []

    if cfg.denoise != "none":
        img = denoise_image(img, cfg.denoise, cfg.kernel_size)
        ops.append(f"denoise:{cfg.denoise}")

    if cfg.sharpness != "none":
        img = sharpen_image(img, cfg.sharpness, cfg.kernel_size)
        ops.append(f"sharpen:{cfg.sharpness}")

    if cfg.contrast != "none":
        img = enhance_contrast(img, cfg.contrast, cfg.kernel_size)
        ops.append(f"contrast:{cfg.contrast}")

    mean_after = float(img.mean())
    return EnhanceResult(
        image=img,
        operations=ops,
        mean_before=mean_before,
        mean_after=mean_after,
    )


# ─── batch_enhance ────────────────────────────────────────────────────────────

def batch_enhance(
    images: List[np.ndarray],
    cfg: Optional[EnhanceConfig] = None,
) -> List[EnhanceResult]:
    """Улучшить несколько изображений.

    Аргументы:
        images: Список изображений uint8.
        cfg:    Параметры.

    Возвращает:
        Список EnhanceResult.
    """
    return [enhance_image(img, cfg) for img in images]
