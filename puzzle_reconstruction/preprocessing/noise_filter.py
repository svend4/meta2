"""Фильтрация шума на изображениях фрагментов.

Модуль предоставляет функции для удаления различных типов шума:
Гауссов (усредняющий и гауссов фильтр), соль-и-перец (медианный фильтр),
билатеральный (сохраняет края), нелокальные средние (NLM), а также
пакетную обработку списка изображений.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

import cv2
import numpy as np


# ─── NoiseFilterParams ────────────────────────────────────────────────────────

_VALID_FILTER_METHODS = frozenset(
    {"average", "gaussian", "median", "bilateral", "nlm"}
)


@dataclass
class NoiseFilterParams:
    """Параметры фильтрации шума.

    Атрибуты:
        method:      Алгоритм фильтрации.
        kernel_size: Размер ядра фильтра (нечётное целое >= 3).
        sigma_color: Диапазон цветов для билатерального фильтра (> 0).
        sigma_space: Пространственный диапазон для билатерального фильтра (> 0).
        h:           Сила фильтрации для NLM (> 0).
        template_window: Размер шаблонного окна NLM (нечётное >= 3).
        search_window:   Размер окна поиска NLM (нечётное >= 3).
        params:      Дополнительные параметры.
    """

    method: str = "gaussian"
    kernel_size: int = 5
    sigma_color: float = 75.0
    sigma_space: float = 75.0
    h: float = 10.0
    template_window: int = 7
    search_window: int = 21
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in _VALID_FILTER_METHODS:
            raise ValueError(
                f"Неизвестный метод '{self.method}'. "
                f"Допустимые: {sorted(_VALID_FILTER_METHODS)}"
            )
        if self.kernel_size < 3:
            raise ValueError(
                f"kernel_size должен быть >= 3, получено {self.kernel_size}"
            )
        if self.kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size должен быть нечётным, получено {self.kernel_size}"
            )
        if self.sigma_color <= 0.0:
            raise ValueError(
                f"sigma_color должна быть > 0, получено {self.sigma_color}"
            )
        if self.sigma_space <= 0.0:
            raise ValueError(
                f"sigma_space должна быть > 0, получено {self.sigma_space}"
            )
        if self.h <= 0.0:
            raise ValueError(
                f"h должна быть > 0, получено {self.h}"
            )
        if self.template_window < 3 or self.template_window % 2 == 0:
            raise ValueError(
                f"template_window должен быть нечётным >= 3, получено {self.template_window}"
            )
        if self.search_window < 3 or self.search_window % 2 == 0:
            raise ValueError(
                f"search_window должен быть нечётным >= 3, получено {self.search_window}"
            )


# ─── _ensure_uint8 ────────────────────────────────────────────────────────────

def _ensure_uint8_2d_or_3d(img: np.ndarray) -> np.ndarray:
    """Проверяет, что img — 2-D или 3-D массив uint8."""
    if img.ndim not in (2, 3):
        raise ValueError(
            f"Изображение должно быть 2-D или 3-D, получено ndim={img.ndim}"
        )
    return np.asarray(img, dtype=np.uint8)


# ─── average_filter ──────────────────────────────────────────────────────────

def average_filter(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Усредняющий (box) фильтр.

    Аргументы:
        img:         Изображение (uint8, 2-D или 3-D).
        kernel_size: Размер ядра (нечётное >= 3).

    Возвращает:
        Отфильтрованное изображение (uint8) той же формы.

    Исключения:
        ValueError: Если img некорректен или kernel_size невалиден.
    """
    img = _ensure_uint8_2d_or_3d(img)
    if kernel_size < 3:
        raise ValueError(f"kernel_size должен быть >= 3, получено {kernel_size}")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size должен быть нечётным, получено {kernel_size}")
    return cv2.blur(img, (kernel_size, kernel_size))


# ─── gaussian_filter ─────────────────────────────────────────────────────────

def gaussian_filter(
    img: np.ndarray, kernel_size: int = 5, sigma: float = 0.0
) -> np.ndarray:
    """Гауссов фильтр сглаживания.

    Аргументы:
        img:         Изображение (uint8).
        kernel_size: Размер ядра (нечётное >= 3).
        sigma:       Ст. отклонение (0 — автовычисление по kernel_size).

    Возвращает:
        Отфильтрованное изображение (uint8) той же формы.

    Исключения:
        ValueError: Если img некорректен или kernel_size невалиден.
    """
    img = _ensure_uint8_2d_or_3d(img)
    if kernel_size < 3:
        raise ValueError(f"kernel_size должен быть >= 3, получено {kernel_size}")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size должен быть нечётным, получено {kernel_size}")
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


# ─── median_filter ───────────────────────────────────────────────────────────

def median_filter(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Медианный фильтр (эффективен против шума «соль-и-перец»).

    Аргументы:
        img:         Изображение (uint8).
        kernel_size: Размер ядра (нечётное >= 3).

    Возвращает:
        Отфильтрованное изображение (uint8) той же формы.

    Исключения:
        ValueError: Если img некорректен или kernel_size невалиден.
    """
    img = _ensure_uint8_2d_or_3d(img)
    if kernel_size < 3:
        raise ValueError(f"kernel_size должен быть >= 3, получено {kernel_size}")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size должен быть нечётным, получено {kernel_size}")
    return cv2.medianBlur(img, kernel_size)


# ─── bilateral_filter ─────────────────────────────────────────────────────────

def bilateral_filter(
    img: np.ndarray,
    kernel_size: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Билатеральный фильтр (сохраняет края).

    Аргументы:
        img:         Изображение (uint8).
        kernel_size: Диаметр окрестности (нечётное >= 3).
        sigma_color: Ст. отклонение в пространстве цветов.
        sigma_space: Ст. отклонение в координатном пространстве.

    Возвращает:
        Отфильтрованное изображение (uint8) той же формы.

    Исключения:
        ValueError: Если параметры некорректны.
    """
    img = _ensure_uint8_2d_or_3d(img)
    if kernel_size < 3:
        raise ValueError(f"kernel_size должен быть >= 3, получено {kernel_size}")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size должен быть нечётным, получено {kernel_size}")
    if sigma_color <= 0.0:
        raise ValueError(f"sigma_color должна быть > 0, получено {sigma_color}")
    if sigma_space <= 0.0:
        raise ValueError(f"sigma_space должна быть > 0, получено {sigma_space}")
    return cv2.bilateralFilter(img, kernel_size, sigma_color, sigma_space)


# ─── nlm_filter ───────────────────────────────────────────────────────────────

def nlm_filter(
    img: np.ndarray,
    h: float = 10.0,
    template_window: int = 7,
    search_window: int = 21,
) -> np.ndarray:
    """Нелокальные средние (Non-Local Means).

    Аргументы:
        img:             Изображение (uint8, градации серого или BGR).
        h:               Сила фильтрации (больше — сильнее, > 0).
        template_window: Размер шаблонного окна (нечётное >= 3).
        search_window:   Размер окна поиска (нечётное >= 3).

    Возвращает:
        Отфильтрованное изображение (uint8) той же формы.

    Исключения:
        ValueError: Если параметры некорректны.
    """
    img = _ensure_uint8_2d_or_3d(img)
    if h <= 0.0:
        raise ValueError(f"h должна быть > 0, получено {h}")
    if template_window < 3 or template_window % 2 == 0:
        raise ValueError(
            f"template_window должен быть нечётным >= 3, получено {template_window}"
        )
    if search_window < 3 or search_window % 2 == 0:
        raise ValueError(
            f"search_window должен быть нечётным >= 3, получено {search_window}"
        )

    if img.ndim == 2:
        return cv2.fastNlMeansDenoising(
            img, h=h,
            templateWindowSize=template_window,
            searchWindowSize=search_window,
        )
    else:
        return cv2.fastNlMeansDenoisingColored(
            img, h=h, hColor=h,
            templateWindowSize=template_window,
            searchWindowSize=search_window,
        )


# ─── apply_noise_filter ───────────────────────────────────────────────────────

def apply_noise_filter(
    img: np.ndarray, params: NoiseFilterParams
) -> np.ndarray:
    """Применить фильтрацию шума согласно параметрам.

    Аргументы:
        img:    Изображение (uint8).
        params: Параметры фильтрации.

    Возвращает:
        Отфильтрованное изображение (uint8) той же формы.
    """
    dispatch = {
        "average": lambda i: average_filter(i, params.kernel_size),
        "gaussian": lambda i: gaussian_filter(i, params.kernel_size),
        "median": lambda i: median_filter(i, params.kernel_size),
        "bilateral": lambda i: bilateral_filter(
            i, params.kernel_size, params.sigma_color, params.sigma_space
        ),
        "nlm": lambda i: nlm_filter(
            i, params.h, params.template_window, params.search_window
        ),
    }
    return dispatch[params.method](img)


# ─── batch_noise_filter ───────────────────────────────────────────────────────

def batch_noise_filter(
    images: List[np.ndarray], params: NoiseFilterParams
) -> List[np.ndarray]:
    """Применить фильтрацию шума к списку изображений.

    Аргументы:
        images: Список изображений (uint8).
        params: Параметры фильтрации.

    Возвращает:
        Список отфильтрованных изображений (uint8).
    """
    return [apply_noise_filter(img, params) for img in images]
