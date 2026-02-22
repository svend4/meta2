"""Нормализация освещения на фрагментах пазла.

Модуль компенсирует неравномерное освещение изображения фрагмента:
вычитание оцениваемого фона (background subtraction), нормализация
среднего и стандартного отклонения яркости, а также CLAHE-подобная
коррекция.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# ─── IllumConfig ──────────────────────────────────────────────────────────────

@dataclass
class IllumConfig:
    """Параметры нормализации освещения.

    Атрибуты:
        blur_ksize:      Размер ядра гауссового размытия для оценки фона
                         (нечётное >= 3).
        target_mean:     Целевое среднее яркости [0, 255].
        target_std:      Целевое стандартное отклонение > 0.
        clip_limit:      Порог ограничения CLAHE (> 0).
        tile_grid_size:  Размер ячейки CLAHE (оба > 0).
    """

    blur_ksize: int = 51
    target_mean: float = 128.0
    target_std: float = 60.0
    clip_limit: float = 2.0
    tile_grid_size: Tuple[int, int] = (8, 8)

    def __post_init__(self) -> None:
        if self.blur_ksize < 3 or self.blur_ksize % 2 == 0:
            raise ValueError(
                f"blur_ksize должен быть нечётным >= 3, получено {self.blur_ksize}"
            )
        if not (0.0 <= self.target_mean <= 255.0):
            raise ValueError(
                f"target_mean должен быть в [0, 255], получено {self.target_mean}"
            )
        if self.target_std <= 0.0:
            raise ValueError(
                f"target_std должен быть > 0, получено {self.target_std}"
            )
        if self.clip_limit <= 0.0:
            raise ValueError(
                f"clip_limit должен быть > 0, получено {self.clip_limit}"
            )
        if self.tile_grid_size[0] <= 0 or self.tile_grid_size[1] <= 0:
            raise ValueError(
                f"tile_grid_size должен быть положительным, "
                f"получено {self.tile_grid_size}"
            )


# ─── IllumResult ──────────────────────────────────────────────────────────────

@dataclass
class IllumResult:
    """Результат нормализации освещения.

    Атрибуты:
        image:        Нормализованное изображение (uint8).
        original_mean: Среднее яркости исходного.
        original_std:  СКО яркости исходного.
        method:        Имя применённого метода.
    """

    image: np.ndarray
    original_mean: float
    original_std: float
    method: str

    def __post_init__(self) -> None:
        if self.image.ndim not in (2, 3):
            raise ValueError(
                f"image должно быть 2D или 3D, получено ndim={self.image.ndim}"
            )
        if self.original_mean < 0.0 or self.original_mean > 255.0:
            raise ValueError(
                f"original_mean должен быть в [0, 255], "
                f"получено {self.original_mean}"
            )
        if self.original_std < 0.0:
            raise ValueError(
                f"original_std должен быть >= 0, получено {self.original_std}"
            )
        if not self.method:
            raise ValueError("method не должен быть пустым")

    @property
    def shape(self) -> Tuple[int, ...]:
        """Форма нормализованного изображения."""
        return self.image.shape


# ─── _to_gray_float ───────────────────────────────────────────────────────────

def _to_gray_float(image: np.ndarray) -> np.ndarray:
    """Привести изображение к серому float64 в [0, 255]."""
    img = np.asarray(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return img.astype(np.float64)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


# ─── estimate_illumination ────────────────────────────────────────────────────

def estimate_illumination(
    image: np.ndarray,
    blur_ksize: int = 51,
) -> np.ndarray:
    """Оценить поверхность освещения (background) гауссовым размытием.

    Аргументы:
        image:      Серое или RGB изображение.
        blur_ksize: Размер ядра гауссового фильтра (нечётное >= 3).

    Возвращает:
        Background float64 (h, w).

    Исключения:
        ValueError: Если blur_ksize < 3 или чётный.
    """
    if blur_ksize < 3 or blur_ksize % 2 == 0:
        raise ValueError(
            f"blur_ksize должен быть нечётным >= 3, получено {blur_ksize}"
        )
    gray = _to_gray_float(image)
    ksize = (blur_ksize, blur_ksize)
    background = cv2.GaussianBlur(gray, ksize, sigmaX=0)
    return background


# ─── subtract_background ──────────────────────────────────────────────────────

def subtract_background(
    image: np.ndarray,
    blur_ksize: int = 51,
    offset: float = 128.0,
) -> np.ndarray:
    """Вычесть оценённый фон и добавить смещение (оставить яркость нейтральной).

    Аргументы:
        image:      Серое или RGB изображение.
        blur_ksize: Размер ядра для оценки фона (нечётное >= 3).
        offset:     Смещение после вычитания (0–255).

    Возвращает:
        uint8-изображение с вычтенным фоном.

    Исключения:
        ValueError: Если blur_ksize некорректен.
    """
    gray = _to_gray_float(image)
    bg = estimate_illumination(image, blur_ksize)
    corrected = gray - bg + offset
    return _to_uint8(corrected)


# ─── normalize_mean_std ───────────────────────────────────────────────────────

def normalize_mean_std(
    image: np.ndarray,
    target_mean: float = 128.0,
    target_std: float = 60.0,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Нормализовать среднее и стандартное отклонение яркости.

    Аргументы:
        image:        Серое или RGB изображение.
        target_mean:  Целевое среднее [0, 255].
        target_std:   Целевое СКО > 0.
        mask:         Опциональная маска фрагмента (uint8); статистики
                      вычисляются только по маске.

    Возвращает:
        uint8-изображение.

    Исключения:
        ValueError: При некорректных параметрах.
    """
    if not (0.0 <= target_mean <= 255.0):
        raise ValueError(
            f"target_mean должен быть в [0, 255], получено {target_mean}"
        )
    if target_std <= 0.0:
        raise ValueError(
            f"target_std должен быть > 0, получено {target_std}"
        )

    gray = _to_gray_float(image)

    if mask is not None:
        pixels = gray[mask > 0]
    else:
        pixels = gray.ravel()

    if len(pixels) == 0:
        return _to_uint8(gray)

    src_mean = float(pixels.mean())
    src_std = float(pixels.std()) + 1e-12
    normalized = (gray - src_mean) * (target_std / src_std) + target_mean
    return _to_uint8(normalized)


# ─── apply_clahe ──────────────────────────────────────────────────────────────

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Применить CLAHE для выравнивания локального контраста.

    Аргументы:
        image:          Серое или RGB изображение (uint8).
        clip_limit:     Порог ограничения усиления (> 0).
        tile_grid_size: Размер ячейки (width, height), оба > 0.

    Возвращает:
        uint8-изображение.

    Исключения:
        ValueError: При некорректных параметрах.
    """
    if clip_limit <= 0.0:
        raise ValueError(
            f"clip_limit должен быть > 0, получено {clip_limit}"
        )
    if tile_grid_size[0] <= 0 or tile_grid_size[1] <= 0:
        raise ValueError(
            f"tile_grid_size должен быть положительным, "
            f"получено {tile_grid_size}"
        )

    img = np.asarray(image)
    is_color = img.ndim == 3

    if is_color:
        lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
    else:
        l_ch = img.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l_ch)

    if is_color:
        lab_eq = cv2.merge([l_eq, a_ch, b_ch])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return l_eq


# ─── normalize_illumination ───────────────────────────────────────────────────

def normalize_illumination(
    image: np.ndarray,
    method: str = "mean_std",
    cfg: Optional[IllumConfig] = None,
    mask: Optional[np.ndarray] = None,
) -> IllumResult:
    """Нормализовать освещение изображения фрагмента.

    Аргументы:
        image:  Серое или RGB изображение.
        method: Метод нормализации — "mean_std" | "background" | "clahe".
        cfg:    Параметры (None → IllumConfig()).
        mask:   Маска фрагмента (uint8; для "mean_std").

    Возвращает:
        IllumResult.

    Исключения:
        ValueError: При неизвестном методе или некорректных данных.
    """
    if cfg is None:
        cfg = IllumConfig()

    gray = _to_gray_float(image)
    original_mean = float(gray.mean())
    original_std = float(gray.std())

    if method == "mean_std":
        result = normalize_mean_std(
            image, cfg.target_mean, cfg.target_std, mask
        )
    elif method == "background":
        result = subtract_background(image, cfg.blur_ksize)
    elif method == "clahe":
        result = apply_clahe(image, cfg.clip_limit, cfg.tile_grid_size)
    else:
        raise ValueError(
            f"Неизвестный метод нормализации: {method!r}. "
            f"Допустимые: 'mean_std', 'background', 'clahe'"
        )

    return IllumResult(
        image=result,
        original_mean=original_mean,
        original_std=original_std,
        method=method,
    )


# ─── batch_normalize_illumination ─────────────────────────────────────────────

def batch_normalize_illumination(
    images: list,
    method: str = "mean_std",
    cfg: Optional[IllumConfig] = None,
) -> list:
    """Нормализовать освещение для списка изображений.

    Аргументы:
        images: Список изображений.
        method: Метод нормализации.
        cfg:    Параметры.

    Возвращает:
        Список IllumResult.
    """
    return [normalize_illumination(img, method, cfg) for img in images]
