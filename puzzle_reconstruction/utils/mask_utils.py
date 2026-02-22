"""
Утилиты для работы с масками изображений фрагментов.

Создание, применение, морфологическая обработка и комбинирование
бинарных масок (uint8, значения 0 и 255). Используются для
сегментации фрагментов, обрезки и создания альфа-каналов.

Функции:
    create_alpha_mask  — создать прямоугольную маску размера h×w
    apply_mask         — применить маску к изображению (вне маски → fill)
    erode_mask         — эрозия маски (сужение области)
    dilate_mask        — дилатация маски (расширение области)
    mask_from_contour  — создать маску из контура (заливка fillPoly)
    combine_masks      — объединить маски ('and', 'or', 'xor')
    crop_to_mask       — вырезать ограничивающий прямоугольник маски
    invert_mask        — инвертировать маску (0↔255)
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# ─── create_alpha_mask ────────────────────────────────────────────────────────

def create_alpha_mask(h:    int,
                       w:    int,
                       fill: int = 255) -> np.ndarray:
    """
    Создаёт прямоугольную маску h×w, заполненную значением fill.

    Args:
        h:    Высота маски.
        w:    Ширина маски.
        fill: Значение заполнения (0 = чёрная маска, 255 = белая маска).

    Returns:
        Маска (h, w) dtype uint8.

    Raises:
        ValueError: Если h ≤ 0 или w ≤ 0.
    """
    if h <= 0 or w <= 0:
        raise ValueError(f"Mask dimensions must be positive, got ({h}, {w}).")
    mask = np.full((h, w), int(np.clip(fill, 0, 255)), dtype=np.uint8)
    return mask


# ─── apply_mask ───────────────────────────────────────────────────────────────

def apply_mask(img:  np.ndarray,
                mask: np.ndarray,
                fill: int = 0) -> np.ndarray:
    """
    Применяет бинарную маску к изображению.

    Пиксели, где mask == 0, заполняются значением fill.
    Пиксели, где mask > 0, сохраняются.

    Args:
        img:  BGR или grayscale изображение uint8.
        mask: Маска (h, w) uint8; ненулевые пиксели — «активная область».
        fill: Значение заполнения для «неактивных» пикселей (0..255).

    Returns:
        Маскированное изображение той же формы и dtype.

    Raises:
        ValueError: Если пространственные размеры img и mask не совпадают.
    """
    if img.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image shape {img.shape[:2]} != mask shape {mask.shape[:2]}."
        )
    result = img.copy()
    binary = mask > 0
    if img.ndim == 2:
        result[~binary] = int(np.clip(fill, 0, 255))
    else:
        result[~binary] = int(np.clip(fill, 0, 255))
    return result


# ─── erode_mask ───────────────────────────────────────────────────────────────

def erode_mask(mask:       np.ndarray,
                ksize:      int = 3,
                iterations: int = 1) -> np.ndarray:
    """
    Применяет морфологическую эрозию к маске.

    Уменьшает область маски (убирает краевые пиксели).

    Args:
        mask:       Маска (h, w) uint8.
        ksize:      Размер квадратного ядра (нечётное, ≥ 1).
        iterations: Количество итераций эрозии.

    Returns:
        Эродированная маска (h, w) uint8.
    """
    ksize  = max(1, ksize)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (ksize, ksize)
    )
    return cv2.erode(mask, kernel, iterations=iterations)


# ─── dilate_mask ──────────────────────────────────────────────────────────────

def dilate_mask(mask:       np.ndarray,
                 ksize:      int = 3,
                 iterations: int = 1) -> np.ndarray:
    """
    Применяет морфологическую дилатацию к маске.

    Расширяет область маски (добавляет краевые пиксели).

    Args:
        mask:       Маска (h, w) uint8.
        ksize:      Размер квадратного ядра (нечётное, ≥ 1).
        iterations: Количество итераций дилатации.

    Returns:
        Дилатированная маска (h, w) uint8.
    """
    ksize  = max(1, ksize)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (ksize, ksize)
    )
    return cv2.dilate(mask, kernel, iterations=iterations)


# ─── mask_from_contour ────────────────────────────────────────────────────────

def mask_from_contour(contour: np.ndarray,
                       h:       int,
                       w:       int) -> np.ndarray:
    """
    Создаёт маску путём закрашивания области внутри контура.

    Args:
        contour: Массив точек формы (N,2) или (N,1,2) int32/float32.
        h:       Высота маски.
        w:       Ширина маски.

    Returns:
        Маска (h, w) uint8 (255 внутри, 0 снаружи).

    Raises:
        ValueError: Если h ≤ 0 или w ≤ 0.
    """
    if h <= 0 or w <= 0:
        raise ValueError(f"Mask dimensions must be positive, got ({h}, {w}).")

    pts = np.asarray(contour, dtype=np.int32)
    if pts.ndim == 2:
        pts = pts.reshape(-1, 1, 2)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


# ─── combine_masks ────────────────────────────────────────────────────────────

def combine_masks(mask1: np.ndarray,
                   mask2: np.ndarray,
                   mode:  str = "and") -> np.ndarray:
    """
    Комбинирует две маски по логической операции.

    Args:
        mask1: Первая маска (h, w) uint8.
        mask2: Вторая маска (h, w) uint8.
        mode:  'and' | 'or' | 'xor'.

    Returns:
        Результирующая маска (h, w) uint8 (0 или 255).

    Raises:
        ValueError: Несовпадение форм или неизвестный mode.
    """
    if mask1.shape != mask2.shape:
        raise ValueError(
            f"Mask shapes must match: {mask1.shape} vs {mask2.shape}."
        )
    if mode == "and":
        result = cv2.bitwise_and(mask1, mask2)
    elif mode == "or":
        result = cv2.bitwise_or(mask1, mask2)
    elif mode == "xor":
        result = cv2.bitwise_xor(mask1, mask2)
    else:
        raise ValueError(
            f"Unknown combine mode {mode!r}. Choose 'and', 'or', or 'xor'."
        )
    return result


# ─── crop_to_mask ─────────────────────────────────────────────────────────────

def crop_to_mask(img:  np.ndarray,
                  mask: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Вырезает минимальный ограничивающий прямоугольник активной области маски.

    Args:
        img:  BGR или grayscale изображение uint8.
        mask: Маска (h, w) uint8.

    Returns:
        (cropped_img, bbox) где bbox = (x, y, w, h).
        Если маска пустая, возвращает исходное изображение и (0,0,w,h).

    Raises:
        ValueError: Если пространственные размеры не совпадают.
    """
    if img.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image shape {img.shape[:2]} != mask shape {mask.shape[:2]}."
        )

    coords = cv2.findNonZero(mask)
    if coords is None:
        h, w = img.shape[:2]
        return img.copy(), (0, 0, w, h)

    x, y, bw, bh = cv2.boundingRect(coords)
    cropped = img[y:y + bh, x:x + bw]
    return cropped, (x, y, bw, bh)


# ─── invert_mask ──────────────────────────────────────────────────────────────

def invert_mask(mask: np.ndarray) -> np.ndarray:
    """
    Инвертирует маску (0 ↔ 255).

    Args:
        mask: Маска (h, w) uint8.

    Returns:
        Инвертированная маска (h, w) uint8.
    """
    return cv2.bitwise_not(mask)
