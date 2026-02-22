"""Улучшение и усиление краёв на изображениях фрагментов.

Модуль предоставляет функции для усиления краёв: нерезкое маскирование
(unsharp mask), усиление с помощью Лапласиана, гибридный подход
(sharp + edge overlay), масштабирование градиента,
а также пакетную обработку и вычисление меры резкости.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np


# ─── EdgeEnhanceParams ────────────────────────────────────────────────────────

_VALID_ENHANCE_METHODS = frozenset(
    {"unsharp", "laplacian", "hybrid", "gradient_scale"}
)


@dataclass
class EdgeEnhanceParams:
    """Параметры усиления краёв.

    Атрибуты:
        method:     Метод усиления краёв.
        strength:   Коэффициент усиления (> 0).
        blur_sigma: Ст. отклонение размытия для unsharp/hybrid (> 0).
        kernel_size: Размер ядра (нечётное >= 3).
        clip:       Ограничить значения до [0, 255] после усиления.
        params:     Дополнительные параметры.
    """

    method: str = "unsharp"
    strength: float = 1.5
    blur_sigma: float = 1.0
    kernel_size: int = 5
    clip: bool = True
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in _VALID_ENHANCE_METHODS:
            raise ValueError(
                f"Неизвестный метод '{self.method}'. "
                f"Допустимые: {sorted(_VALID_ENHANCE_METHODS)}"
            )
        if self.strength <= 0.0:
            raise ValueError(
                f"strength должен быть > 0, получено {self.strength}"
            )
        if self.blur_sigma <= 0.0:
            raise ValueError(
                f"blur_sigma должна быть > 0, получено {self.blur_sigma}"
            )
        if self.kernel_size < 3:
            raise ValueError(
                f"kernel_size должен быть >= 3, получено {self.kernel_size}"
            )
        if self.kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size должен быть нечётным, получено {self.kernel_size}"
            )


# ─── _ensure_uint8 ────────────────────────────────────────────────────────────

def _check_image(img: np.ndarray) -> np.ndarray:
    if img.ndim not in (2, 3):
        raise ValueError(
            f"img должен быть 2-D или 3-D, получено ndim={img.ndim}"
        )
    return np.asarray(img, dtype=np.uint8)


# ─── unsharp_mask ─────────────────────────────────────────────────────────────

def unsharp_mask(
    img: np.ndarray,
    strength: float = 1.5,
    blur_sigma: float = 1.0,
    kernel_size: int = 5,
    clip: bool = True,
) -> np.ndarray:
    """Нерезкое маскирование (unsharp masking).

    enhanced = img + strength * (img − blur(img))

    Аргументы:
        img:         Изображение (uint8, 2-D или 3-D).
        strength:    Коэффициент усиления (> 0).
        blur_sigma:  Ст. отклонение размытия (> 0).
        kernel_size: Размер ядра Гаусса (нечётное >= 3).
        clip:        Ограничить результат до [0, 255].

    Возвращает:
        Изображение (uint8) той же формы.

    Исключения:
        ValueError: Если параметры некорректны.
    """
    img = _check_image(img)
    if strength <= 0.0:
        raise ValueError(f"strength должен быть > 0, получено {strength}")
    if blur_sigma <= 0.0:
        raise ValueError(f"blur_sigma должна быть > 0, получено {blur_sigma}")
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size должен быть нечётным >= 3, получено {kernel_size}"
        )

    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), blur_sigma)
    mask = img.astype(np.float32) - blurred.astype(np.float32)
    enhanced = img.astype(np.float32) + strength * mask

    if clip:
        enhanced = np.clip(enhanced, 0, 255)
    return enhanced.astype(np.uint8)


# ─── laplacian_enhance ────────────────────────────────────────────────────────

def laplacian_enhance(
    img: np.ndarray,
    strength: float = 1.0,
    clip: bool = True,
) -> np.ndarray:
    """Усиление краёв с помощью Лапласиана.

    enhanced = img − strength * Laplacian(img)

    Аргументы:
        img:      Изображение (uint8, 2-D или 3-D).
        strength: Коэффициент усиления (> 0).
        clip:     Ограничить результат до [0, 255].

    Возвращает:
        Изображение (uint8) той же формы.

    Исключения:
        ValueError: Если параметры некорректны.
    """
    img = _check_image(img)
    if strength <= 0.0:
        raise ValueError(f"strength должен быть > 0, получено {strength}")

    if img.ndim == 2:
        lap = cv2.Laplacian(img, cv2.CV_64F)
    else:
        channels = [cv2.Laplacian(img[:, :, c], cv2.CV_64F) for c in range(img.shape[2])]
        lap = np.stack(channels, axis=2)

    enhanced = img.astype(np.float64) - strength * lap
    if clip:
        enhanced = np.clip(enhanced, 0, 255)
    return enhanced.astype(np.uint8)


# ─── hybrid_enhance ───────────────────────────────────────────────────────────

def hybrid_enhance(
    img: np.ndarray,
    strength: float = 1.0,
    blur_sigma: float = 1.0,
    kernel_size: int = 5,
    clip: bool = True,
) -> np.ndarray:
    """Гибридное усиление: unsharp mask + Лапласиан.

    Аргументы:
        img:         Изображение (uint8, 2-D или 3-D).
        strength:    Суммарный коэффициент усиления (> 0).
        blur_sigma:  Ст. отклонение для unsharp (> 0).
        kernel_size: Размер ядра (нечётное >= 3).
        clip:        Ограничить до [0, 255].

    Возвращает:
        Изображение (uint8).
    """
    img = _check_image(img)
    us = unsharp_mask(img, strength=strength * 0.5,
                      blur_sigma=blur_sigma, kernel_size=kernel_size, clip=False)
    lap = laplacian_enhance(img, strength=strength * 0.5, clip=False)
    combined = (us.astype(np.float32) + lap.astype(np.float32)) / 2.0
    if clip:
        combined = np.clip(combined, 0, 255)
    return combined.astype(np.uint8)


# ─── gradient_scale_enhance ───────────────────────────────────────────────────

def gradient_scale_enhance(
    img: np.ndarray,
    strength: float = 1.5,
    clip: bool = True,
) -> np.ndarray:
    """Усиление через масштабирование градиента.

    Вычисляет градиент Собеля и добавляет масштабированную карту краёв.

    Аргументы:
        img:      Изображение (uint8, 2-D или 3-D).
        strength: Коэффициент усиления (> 0).
        clip:     Ограничить до [0, 255].

    Возвращает:
        Изображение (uint8).
    """
    img = _check_image(img)
    if strength <= 0.0:
        raise ValueError(f"strength должен быть > 0, получено {strength}")

    def _process_channel(ch: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        return ch.astype(np.float64) + strength * mag

    if img.ndim == 2:
        enhanced = _process_channel(img)
    else:
        channels = [_process_channel(img[:, :, c]) for c in range(img.shape[2])]
        enhanced = np.stack(channels, axis=2)

    if clip:
        enhanced = np.clip(enhanced, 0, 255)
    return enhanced.astype(np.uint8)


# ─── sharpness_measure ────────────────────────────────────────────────────────

def sharpness_measure(img: np.ndarray) -> float:
    """Оценить резкость изображения через дисперсию Лапласиана.

    Аргументы:
        img: Изображение (uint8, 2-D или 3-D).

    Возвращает:
        Скалярная оценка резкости (float >= 0; больше → резче).

    Исключения:
        ValueError: Если img не 2-D или 3-D.
    """
    img = _check_image(img)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


# ─── apply_edge_enhance ───────────────────────────────────────────────────────

def apply_edge_enhance(
    img: np.ndarray, params: EdgeEnhanceParams
) -> np.ndarray:
    """Применить усиление краёв согласно параметрам.

    Аргументы:
        img:    Изображение (uint8).
        params: Параметры усиления.

    Возвращает:
        Изображение (uint8) той же формы.
    """
    dispatch = {
        "unsharp": lambda i: unsharp_mask(
            i, params.strength, params.blur_sigma, params.kernel_size, params.clip
        ),
        "laplacian": lambda i: laplacian_enhance(i, params.strength, params.clip),
        "hybrid": lambda i: hybrid_enhance(
            i, params.strength, params.blur_sigma, params.kernel_size, params.clip
        ),
        "gradient_scale": lambda i: gradient_scale_enhance(
            i, params.strength, params.clip
        ),
    }
    return dispatch[params.method](img)


# ─── batch_edge_enhance ───────────────────────────────────────────────────────

def batch_edge_enhance(
    images: List[np.ndarray], params: EdgeEnhanceParams
) -> List[np.ndarray]:
    """Применить усиление краёв к списку изображений.

    Аргументы:
        images: Список изображений (uint8).
        params: Параметры усиления.

    Возвращает:
        Список обработанных изображений (uint8).
    """
    return [apply_edge_enhance(img, params) for img in images]
