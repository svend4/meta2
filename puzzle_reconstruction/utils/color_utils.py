"""
Утилиты для работы с цветом изображений фрагментов.

Конвертация цветовых пространств, сравнение гистограмм, извлечение
доминирующих цветов, вычисление цветового расстояния ΔE и построение
гистограмм полос вдоль краёв. Используются при оценке совместимости
краёв по цвету.

Функции:
    to_gray            — BGR или grayscale → grayscale
    to_lab             — BGR → CIE-Lab (float32, L∈[0,100], a/b∈[-127,127])
    to_hsv             — BGR → HSV (H∈[0,180], S∈[0,255], V∈[0,255])
    from_lab           — CIE-Lab → BGR (uint8)
    compute_histogram  — нормализованная гистограмма (1D или объединённая)
    compare_histograms — сравнение двух гистограмм (correlation/chi/bhattacharyya)
    dominant_colors    — K доминирующих цветов через K-means (BGR uint8)
    color_distance     — цветовое расстояние ΔE76 в пространстве CIE-Lab
    strip_histogram    — гистограмма полосы вдоль указанного края изображения
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


# ─── Конвертация цветовых пространств ─────────────────────────────────────────

def to_gray(img: np.ndarray) -> np.ndarray:
    """
    Конвертирует BGR или grayscale изображение в grayscale.

    Args:
        img: Изображение uint8 (BGR или grayscale).

    Returns:
        Одноканальное изображение uint8.
    """
    if img.ndim == 2:
        return img.copy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_lab(img: np.ndarray) -> np.ndarray:
    """
    Конвертирует BGR-изображение в CIE-Lab.

    Args:
        img: BGR изображение uint8.

    Returns:
        Lab изображение float32 (L∈[0,100], a/b∈[-127,127]).
    """
    bgr   = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bgrf  = bgr.astype(np.float32) / 255.0
    lab   = cv2.cvtColor(bgrf, cv2.COLOR_BGR2Lab)
    return lab


def to_hsv(img: np.ndarray) -> np.ndarray:
    """
    Конвертирует BGR-изображение в HSV.

    Args:
        img: BGR изображение uint8.

    Returns:
        HSV изображение uint8 (H∈[0,180], S∈[0,255], V∈[0,255]).
    """
    bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def from_lab(lab: np.ndarray) -> np.ndarray:
    """
    Конвертирует CIE-Lab изображение обратно в BGR.

    Args:
        lab: Lab изображение float32 (как возвращает to_lab).

    Returns:
        BGR изображение uint8.
    """
    bgrf = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2BGR)
    return np.clip(bgrf * 255.0, 0, 255).astype(np.uint8)


# ─── compute_histogram ────────────────────────────────────────────────────────

def compute_histogram(img:       np.ndarray,
                       bins:      int  = 256,
                       channel:   int  = 0,
                       normalize: bool = True) -> np.ndarray:
    """
    Вычисляет одномерную гистограмму заданного канала изображения.

    Для grayscale изображений channel игнорируется (используется единственный канал).

    Args:
        img:       BGR или grayscale изображение uint8.
        bins:      Количество бинов гистограммы (1..256).
        channel:   Индекс канала для BGR-изображения (0=B, 1=G, 2=R).
        normalize: Нормировать ли гистограмму (sum=1).

    Returns:
        Одномерный массив float32 длины bins.
    """
    if img.ndim == 2:
        src = [img]
        ch  = [0]
    else:
        src = [img]
        ch  = [channel]

    hist = cv2.calcHist(src, ch, None, [bins], [0, 256])
    hist = hist.flatten().astype(np.float32)

    if normalize:
        total = hist.sum()
        if total > 0:
            hist /= total

    return hist


# ─── compare_histograms ───────────────────────────────────────────────────────

# Отображение названий методов на константы OpenCV
_HIST_METHODS = {
    "correlation":    cv2.HISTCMP_CORREL,
    "chi":            cv2.HISTCMP_CHISQR,
    "bhattacharyya":  cv2.HISTCMP_BHATTACHARYYA,
    "intersection":   cv2.HISTCMP_INTERSECT,
}


def compare_histograms(h1:     np.ndarray,
                        h2:     np.ndarray,
                        method: str = "correlation") -> float:
    """
    Сравнивает две одномерные нормализованные гистограммы.

    Args:
        h1:     Первая гистограмма (float32, 1D).
        h2:     Вторая гистограмма (float32, 1D, той же длины).
        method: 'correlation' (1=идентичны, −1=противоположны) |
                'chi'         (0=идентичны, ↑=расходятся) |
                'bhattacharyya' (0=идентичны, ↑=расходятся) |
                'intersection' (чем больше, тем ближе).

    Returns:
        Скалярное значение float.

    Raises:
        ValueError: Неизвестный метод.
    """
    if method not in _HIST_METHODS:
        raise ValueError(
            f"Unknown histogram comparison method {method!r}. "
            f"Choose from: {list(_HIST_METHODS)}."
        )
    a = h1.reshape(-1, 1).astype(np.float32)
    b = h2.reshape(-1, 1).astype(np.float32)
    return float(cv2.compareHist(a, b, _HIST_METHODS[method]))


# ─── dominant_colors ──────────────────────────────────────────────────────────

def dominant_colors(img:  np.ndarray,
                     k:    int = 3,
                     seed: int = 42) -> np.ndarray:
    """
    Извлекает K доминирующих цветов через кластеризацию K-means.

    Args:
        img:  BGR или grayscale изображение uint8.
        k:    Количество кластеров (≥ 1).
        seed: Seed для воспроизводимости K-means.

    Returns:
        Массив (k, 3) uint8 с BGR-цветами центроидов, отсортированный
        по убыванию размера кластера.
    """
    k = max(1, k)

    # Привести к BGR
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        bgr = img

    pixels = bgr.reshape(-1, 3).astype(np.float32)

    n_pixels = pixels.shape[0]
    k        = min(k, n_pixels)   # K не может превышать число пикселей

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100, 0.2,
    )
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    labels  = labels.flatten()
    centers = centers.astype(np.uint8)

    # Сортировка центроидов по убыванию размера кластера
    counts = np.bincount(labels, minlength=k)
    order  = np.argsort(-counts)
    return centers[order]


# ─── color_distance ───────────────────────────────────────────────────────────

def color_distance(color1: np.ndarray,
                    color2: np.ndarray,
                    space:  str = "lab") -> float:
    """
    Вычисляет цветовое расстояние ΔE76 между двумя цветами.

    Args:
        color1: Цвет в пространстве BGR (массив длины 3, uint8).
        color2: Второй цвет (BGR, uint8).
        space:  'lab' → ΔE76 в CIE-Lab; 'rgb' → Евклид в RGB.

    Returns:
        Расстояние float ≥ 0.

    Raises:
        ValueError: Неизвестное пространство.
    """
    c1 = np.asarray(color1, dtype=np.uint8).reshape(1, 1, 3)
    c2 = np.asarray(color2, dtype=np.uint8).reshape(1, 1, 3)

    if space == "lab":
        l1 = to_lab(c1)[0, 0]
        l2 = to_lab(c2)[0, 0]
        return float(np.linalg.norm(l1 - l2))
    elif space == "rgb":
        r1 = c1[0, 0].astype(np.float32)
        r2 = c2[0, 0].astype(np.float32)
        return float(np.linalg.norm(r1 - r2))
    else:
        raise ValueError(
            f"Unknown color space {space!r}. Choose 'lab' or 'rgb'."
        )


# ─── strip_histogram ──────────────────────────────────────────────────────────

def strip_histogram(img:       np.ndarray,
                     side:      int  = 0,
                     border_px: int  = 10,
                     bins:      int  = 64,
                     channel:   int  = 0) -> np.ndarray:
    """
    Вычисляет нормализованную гистограмму полосы вдоль указанного края.

    Args:
        img:       BGR или grayscale изображение uint8.
        side:      0=верх, 1=право, 2=низ, 3=лево.
        border_px: Ширина полосы в пикселях.
        bins:      Количество бинов гистограммы.
        channel:   Канал для BGR-изображения (0=B, 1=G, 2=R).

    Returns:
        Нормализованная гистограмма float32 длины bins.

    Raises:
        ValueError: Неизвестная сторона.
    """
    h, w = img.shape[:2]
    bp   = max(1, border_px)

    if side == 0:         # верх
        strip = img[:bp, :]
    elif side == 1:       # право
        strip = img[:, w - bp:]
    elif side == 2:       # низ
        strip = img[h - bp:, :]
    elif side == 3:       # лево
        strip = img[:, :bp]
    else:
        raise ValueError(
            f"Unknown side {side!r}. Must be 0 (top), 1 (right), "
            f"2 (bottom), or 3 (left)."
        )

    return compute_histogram(strip, bins=bins, channel=channel, normalize=True)
