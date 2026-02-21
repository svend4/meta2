"""
Морфологические операции для фрагментов документов.

Экспортирует:
    MorphParams       — параметры морфологической операции
    erode             — эрозия
    dilate            — расширение
    open_morph        — морфологическое открытие (эрозия → расширение)
    close_morph       — морфологическое закрытие (расширение → эрозия)
    tophat            — оператор «белой шляпы» (top-hat)
    blackhat          — оператор «чёрной шляпы» (black-hat)
    morphological_gradient — морфологический градиент
    skeleton          — скелет бинарного изображения
    remove_small_blobs — удаление мелких связных компонент
    fill_holes        — заполнение замкнутых дыр
    batch_morph       — пакетная обработка
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── Параметры ────────────────────────────────────────────────────────────────

_VALID_KERNELS = {"rect", "ellipse", "cross"}
_VALID_OPS = {"erode", "dilate", "open", "close", "tophat", "blackhat", "gradient"}


@dataclass
class MorphParams:
    """Параметры морфологической операции.

    Attributes:
        op:          Операция: ``'erode'``, ``'dilate'``, ``'open'``, ``'close'``,
                     ``'tophat'``, ``'blackhat'``, ``'gradient'``.
        kernel_type: Форма ядра: ``'rect'``, ``'ellipse'``, ``'cross'``.
        ksize:       Размер ядра (нечётное число ≥ 3).
        iterations:  Число итераций (≥ 1).
        params:      Дополнительные параметры.
    """
    op: str = "open"
    kernel_type: str = "rect"
    ksize: int = 3
    iterations: int = 1
    params: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.op not in _VALID_OPS:
            raise ValueError(
                f"op must be one of {sorted(_VALID_OPS)}, got {self.op!r}"
            )
        if self.kernel_type not in _VALID_KERNELS:
            raise ValueError(
                f"kernel_type must be one of {sorted(_VALID_KERNELS)}, got {self.kernel_type!r}"
            )
        if self.ksize < 3:
            raise ValueError(f"ksize must be >= 3, got {self.ksize}")
        if self.ksize % 2 == 0:
            raise ValueError(f"ksize must be odd, got {self.ksize}")
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")


# ─── Вспомогательная функция ────────────────────────────────────────────────

def _make_kernel(kernel_type: str, ksize: int) -> np.ndarray:
    shapes = {
        "rect":    cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross":   cv2.MORPH_CROSS,
    }
    return cv2.getStructuringElement(shapes[kernel_type], (ksize, ksize))


# ─── Публичные функции ────────────────────────────────────────────────────────

def erode(
    img: np.ndarray,
    ksize: int = 3,
    kernel_type: str = "rect",
    iterations: int = 1,
) -> np.ndarray:
    """Морфологическая эрозия.

    Args:
        img:         Изображение uint8 (2D или 3D).
        ksize:       Размер ядра (нечётное, ≥ 3).
        kernel_type: Форма ядра: ``'rect'``, ``'ellipse'``, ``'cross'``.
        iterations:  Число итераций.

    Returns:
        Изображение uint8 той же формы.

    Raises:
        ValueError: Если ``ksize`` < 3, чётное, или ``kernel_type`` недопустим.
    """
    _validate_ksize(ksize)
    _validate_kernel_type(kernel_type)
    k = _make_kernel(kernel_type, ksize)
    return cv2.erode(img.astype(np.uint8), k, iterations=iterations)


def dilate(
    img: np.ndarray,
    ksize: int = 3,
    kernel_type: str = "rect",
    iterations: int = 1,
) -> np.ndarray:
    """Морфологическое расширение.

    Args:
        img:         Изображение uint8 (2D или 3D).
        ksize:       Размер ядра (нечётное, ≥ 3).
        kernel_type: Форма ядра.
        iterations:  Число итераций.

    Returns:
        Изображение uint8 той же формы.
    """
    _validate_ksize(ksize)
    _validate_kernel_type(kernel_type)
    k = _make_kernel(kernel_type, ksize)
    return cv2.dilate(img.astype(np.uint8), k, iterations=iterations)


def open_morph(
    img: np.ndarray,
    ksize: int = 3,
    kernel_type: str = "rect",
) -> np.ndarray:
    """Морфологическое открытие (эрозия затем расширение).

    Убирает мелкие светлые объекты при сохранении крупных структур.

    Args:
        img:         Изображение uint8.
        ksize:       Размер ядра (нечётное, ≥ 3).
        kernel_type: Форма ядра.

    Returns:
        Изображение uint8 той же формы.
    """
    _validate_ksize(ksize)
    _validate_kernel_type(kernel_type)
    k = _make_kernel(kernel_type, ksize)
    return cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_OPEN, k)


def close_morph(
    img: np.ndarray,
    ksize: int = 3,
    kernel_type: str = "rect",
) -> np.ndarray:
    """Морфологическое закрытие (расширение затем эрозия).

    Заполняет небольшие тёмные промежутки внутри светлых регионов.

    Args:
        img:         Изображение uint8.
        ksize:       Размер ядра (нечётное, ≥ 3).
        kernel_type: Форма ядра.

    Returns:
        Изображение uint8 той же формы.
    """
    _validate_ksize(ksize)
    _validate_kernel_type(kernel_type)
    k = _make_kernel(kernel_type, ksize)
    return cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_CLOSE, k)


def tophat(
    img: np.ndarray,
    ksize: int = 5,
    kernel_type: str = "rect",
) -> np.ndarray:
    """Оператор «белой шляпы» (top-hat): img − open(img).

    Выделяет мелкие яркие структуры на тёмном фоне.

    Args:
        img:         Изображение uint8.
        ksize:       Размер ядра (нечётное, ≥ 3).
        kernel_type: Форма ядра.

    Returns:
        Изображение uint8 той же формы.
    """
    _validate_ksize(ksize)
    _validate_kernel_type(kernel_type)
    k = _make_kernel(kernel_type, ksize)
    return cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_TOPHAT, k)


def blackhat(
    img: np.ndarray,
    ksize: int = 5,
    kernel_type: str = "rect",
) -> np.ndarray:
    """Оператор «чёрной шляпы» (black-hat): close(img) − img.

    Выделяет мелкие тёмные структуры на ярком фоне.

    Args:
        img:         Изображение uint8.
        ksize:       Размер ядра (нечётное, ≥ 3).
        kernel_type: Форма ядра.

    Returns:
        Изображение uint8 той же формы.
    """
    _validate_ksize(ksize)
    _validate_kernel_type(kernel_type)
    k = _make_kernel(kernel_type, ksize)
    return cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_BLACKHAT, k)


def morphological_gradient(
    img: np.ndarray,
    ksize: int = 3,
    kernel_type: str = "rect",
) -> np.ndarray:
    """Морфологический градиент: dilate(img) − erode(img).

    Выделяет границы объектов.

    Args:
        img:         Изображение uint8.
        ksize:       Размер ядра (нечётное, ≥ 3).
        kernel_type: Форма ядра.

    Returns:
        Изображение uint8 той же формы.
    """
    _validate_ksize(ksize)
    _validate_kernel_type(kernel_type)
    k = _make_kernel(kernel_type, ksize)
    return cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_GRADIENT, k)


def skeleton(img: np.ndarray) -> np.ndarray:
    """Вычислить скелет бинарного изображения методом итеративных открытий.

    Алгоритм Zhang-Suen (через последовательные открытия), работает с бинарными
    изображениями uint8 (0 / 255).

    Args:
        img: Бинарное изображение uint8 (2D).

    Returns:
        Скелет — бинарное изображение uint8 (2D).

    Raises:
        ValueError: Если изображение многоканальное.
    """
    if img.ndim != 2:
        raise ValueError(
            f"skeleton expects a 2D binary image, got ndim={img.ndim}"
        )
    src = img.copy().astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = np.zeros_like(src)
    while True:
        opened = cv2.morphologyEx(src, cv2.MORPH_OPEN, k)
        temp = cv2.subtract(src, opened)
        skel = cv2.bitwise_or(skel, temp)
        eroded = cv2.erode(src, k)
        if cv2.countNonZero(eroded) == 0:
            break
        src = eroded
    return skel


def remove_small_blobs(
    img: np.ndarray,
    min_area: int = 50,
) -> np.ndarray:
    """Удалить связные компоненты меньше заданной площади из бинарного изображения.

    Args:
        img:      Бинарное изображение uint8 (2D; ненулевые пиксели — объект).
        min_area: Минимальная допустимая площадь (пиксели, ≥ 0).

    Returns:
        Бинарное изображение uint8 (0 / 255) той же формы.

    Raises:
        ValueError: Если изображение многоканальное.
    """
    if img.ndim != 2:
        raise ValueError(
            f"remove_small_blobs expects a 2D binary image, got ndim={img.ndim}"
        )
    if min_area < 0:
        raise ValueError(f"min_area must be >= 0, got {min_area}")
    binary = np.where(img > 0, np.uint8(255), np.uint8(0))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    result = np.zeros_like(binary)
    for comp in range(1, n):
        area = stats[comp, cv2.CC_STAT_AREA]
        if area >= min_area:
            result[labels == comp] = 255
    return result


def fill_holes(img: np.ndarray) -> np.ndarray:
    """Заполнить замкнутые дыры в бинарном изображении методом заливки.

    Args:
        img: Бинарное изображение uint8 (2D).

    Returns:
        Изображение uint8 (2D) с заполненными дырами.

    Raises:
        ValueError: Если изображение многоканальное.
    """
    if img.ndim != 2:
        raise ValueError(
            f"fill_holes expects a 2D binary image, got ndim={img.ndim}"
        )
    binary = np.where(img > 0, np.uint8(255), np.uint8(0))
    h, w = binary.shape
    flood = binary.copy()
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    inverted = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary, inverted)


def apply_morph(img: np.ndarray, params: MorphParams) -> np.ndarray:
    """Применить морфологическую операцию по параметрам.

    Args:
        img:    Изображение uint8.
        params: :class:`MorphParams` с описанием операции.

    Returns:
        Обработанное изображение uint8.
    """
    kw = dict(ksize=params.ksize, kernel_type=params.kernel_type)
    dispatch = {
        "erode":    lambda: erode(img, **kw, iterations=params.iterations),
        "dilate":   lambda: dilate(img, **kw, iterations=params.iterations),
        "open":     lambda: open_morph(img, **kw),
        "close":    lambda: close_morph(img, **kw),
        "tophat":   lambda: tophat(img, **kw),
        "blackhat": lambda: blackhat(img, **kw),
        "gradient": lambda: morphological_gradient(img, **kw),
    }
    return dispatch[params.op]()


def batch_morph(
    images: List[np.ndarray],
    params: Optional[MorphParams] = None,
) -> List[np.ndarray]:
    """Пакетная морфологическая обработка.

    Args:
        images: Список изображений uint8.
        params: Параметры (по умолчанию ``MorphParams()``).

    Returns:
        Список обработанных изображений uint8.
    """
    if params is None:
        params = MorphParams()
    return [apply_morph(img, params) for img in images]


# ─── Вспомогательные валидаторы ───────────────────────────────────────────────

def _validate_ksize(ksize: int) -> None:
    if ksize < 3:
        raise ValueError(f"ksize must be >= 3, got {ksize}")
    if ksize % 2 == 0:
        raise ValueError(f"ksize must be odd, got {ksize}")


def _validate_kernel_type(kernel_type: str) -> None:
    if kernel_type not in _VALID_KERNELS:
        raise ValueError(
            f"kernel_type must be one of {sorted(_VALID_KERNELS)}, got {kernel_type!r}"
        )
