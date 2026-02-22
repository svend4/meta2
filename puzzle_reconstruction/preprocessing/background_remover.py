"""
Удаление фона с изображений фрагментов документа.

Реализует три стратегии:
  - порогово-оценочную (Otsu) для светлых/тёмных фонов,
  - контурно-заливочную (Canny + flood-fill) для однородных фонов,
  - GrabCut для сложных фонов с явным прямоугольником ROI.

Классы:
    BackgroundRemovalResult — результат удаления фона одного изображения

Функции:
    remove_background_thresh   — удаление фона по порогу яркости
    remove_background_edges    — удаление фона через контуры + заливку
    remove_background_grabcut  — удаление фона методом GrabCut
    auto_remove_background     — автоматический выбор метода
    batch_remove_background    — пакетная обработка списка изображений
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import cv2
import numpy as np


# ─── BackgroundRemovalResult ──────────────────────────────────────────────────

@dataclass
class BackgroundRemovalResult:
    """
    Результат удаления фона одного изображения.

    Attributes:
        foreground: Изображение с удалённым фоном (uint8, та же форма).
                    Пиксели фона обнулены (или залиты bg_fill).
        mask:       Бинарная маска переднего плана (uint8, 0 или 255).
        method:     Использованный метод: 'thresh' | 'edges' | 'grabcut'.
        params:     Параметры, переданные в функцию.
    """
    foreground: np.ndarray
    mask:       np.ndarray
    method:     str
    params:     Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        h, w = self.foreground.shape[:2]
        coverage = float(self.mask.astype(np.float32).mean() / 255.0)
        return (f"BackgroundRemovalResult(method={self.method!r}, "
                f"shape=({h},{w}), fg_coverage={coverage:.2f})")


# ─── _to_gray ─────────────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    """Переводит BGR или grayscale в grayscale uint8."""
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ─── remove_background_thresh ─────────────────────────────────────────────────

def remove_background_thresh(
    img:        np.ndarray,
    bg_thresh:  int  = 240,
    invert:     bool = False,
    bg_fill:    int  = 0,
) -> BackgroundRemovalResult:
    """
    Удаляет фон по порогу яркости (предполагается светлый фон).

    Пиксели ярче bg_thresh считаются фоном и заменяются на bg_fill.
    Флаг invert меняет логику на «тёмный фон».

    Args:
        img:       BGR или grayscale изображение uint8.
        bg_thresh: Порог яркости [0, 255].
        invert:    True → тёмный фон (пиксели темнее порога = фон).
        bg_fill:   Значение для заполнения фона в foreground.

    Returns:
        BackgroundRemovalResult с полями foreground, mask, method, params.
    """
    gray = _to_gray(img)

    if invert:
        _, mask = cv2.threshold(gray, bg_thresh, 255, cv2.THRESH_BINARY)
    else:
        _, mask = cv2.threshold(gray, bg_thresh, 255, cv2.THRESH_BINARY_INV)

    foreground = img.copy()
    if img.ndim == 2:
        foreground[mask == 0] = bg_fill
    else:
        foreground[mask == 0] = bg_fill

    return BackgroundRemovalResult(
        foreground=foreground,
        mask=mask,
        method="thresh",
        params={"bg_thresh": bg_thresh, "invert": invert, "bg_fill": bg_fill},
    )


# ─── remove_background_edges ──────────────────────────────────────────────────

def remove_background_edges(
    img:           np.ndarray,
    low_thresh:    int = 50,
    high_thresh:   int = 150,
    dilate_ksize:  int = 5,
    bg_fill:       int = 0,
) -> BackgroundRemovalResult:
    """
    Удаляет фон через детектирование краёв и заливку от углов.

    Алгоритм:
    1. Canny-края → дилатация для замыкания контуров.
    2. Flood-fill от каждого угла — помечает фон.
    3. Инверсия → маска переднего плана.

    Args:
        img:          BGR или grayscale изображение uint8.
        low_thresh:   Нижний порог Canny.
        high_thresh:  Верхний порог Canny.
        dilate_ksize: Размер ядра дилатации для замыкания краёв.
        bg_fill:      Значение для заполнения фона в foreground.

    Returns:
        BackgroundRemovalResult с полями foreground, mask, method, params.
    """
    gray  = _to_gray(img)
    edges = cv2.Canny(gray, low_thresh, high_thresh)

    # Дилатация для замыкания незначительных разрывов в контуре
    if dilate_ksize > 1:
        k     = cv2.getStructuringElement(cv2.MORPH_RECT,
                                          (dilate_ksize, dilate_ksize))
        edges = cv2.dilate(edges, k, iterations=1)

    # Flood-fill от углов (фон = 128)
    h, w       = gray.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    fill_img   = edges.copy()
    corners    = [(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)]

    for cx, cy in corners:
        if fill_img[cy, cx] == 0:   # не граница
            cv2.floodFill(fill_img, flood_mask, (cx, cy), 128)

    # Пиксели, залитые (==128) — фон; всё остальное — передний план
    bg_region = (fill_img == 128)
    mask      = np.where(bg_region, 0, 255).astype(np.uint8)

    foreground = img.copy()
    if img.ndim == 2:
        foreground[mask == 0] = bg_fill
    else:
        foreground[mask == 0] = bg_fill

    return BackgroundRemovalResult(
        foreground=foreground,
        mask=mask,
        method="edges",
        params={
            "low_thresh": low_thresh,
            "high_thresh": high_thresh,
            "dilate_ksize": dilate_ksize,
            "bg_fill": bg_fill,
        },
    )


# ─── remove_background_grabcut ────────────────────────────────────────────────

def remove_background_grabcut(
    img:    np.ndarray,
    margin: int = 10,
    n_iter: int = 5,
    bg_fill: int = 0,
) -> BackgroundRemovalResult:
    """
    Удаляет фон методом GrabCut.

    Прямоугольник ROI строится с отступом margin от краёв изображения.
    Для grayscale входа конвертирует в BGR перед применением GrabCut.

    Args:
        img:     BGR или grayscale изображение uint8.
        margin:  Отступ от краёв изображения для прямоугольника ROI (пикс.).
        n_iter:  Число итераций GrabCut.
        bg_fill: Значение для заполнения фона в foreground.

    Returns:
        BackgroundRemovalResult с полями foreground, mask, method, params.
    """
    h, w = img.shape[:2]
    is_gray = img.ndim == 2

    # GrabCut требует BGR
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if is_gray else img.copy()

    m  = max(1, margin)
    x1 = min(m, w - 2)
    y1 = min(m, h - 2)
    x2 = max(x1 + 1, w - m - 1)
    y2 = max(y1 + 1, h - m - 1)
    rect = (x1, y1, x2 - x1, y2 - y1)

    gc_mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    cv2.grabCut(bgr, gc_mask, rect, bgd_model, fgd_model, n_iter,
                cv2.GC_INIT_WITH_RECT)

    # GC_FGD=1, GC_PR_FGD=3 → передний план
    mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                    255, 0).astype(np.uint8)

    foreground = img.copy()
    if is_gray:
        foreground[mask == 0] = bg_fill
    else:
        foreground[mask == 0] = bg_fill

    return BackgroundRemovalResult(
        foreground=foreground,
        mask=mask,
        method="grabcut",
        params={"margin": margin, "n_iter": n_iter, "bg_fill": bg_fill},
    )


# ─── auto_remove_background ───────────────────────────────────────────────────

def auto_remove_background(
    img:    np.ndarray,
    method: str = "thresh",
    **kwargs,
) -> BackgroundRemovalResult:
    """
    Автоматически удаляет фон выбранным методом.

    Args:
        img:     BGR или grayscale изображение uint8.
        method:  'thresh' | 'edges' | 'grabcut'.
        **kwargs: Параметры конкретного метода.

    Returns:
        BackgroundRemovalResult.

    Raises:
        ValueError: Неизвестный метод.
    """
    if method == "thresh":
        return remove_background_thresh(img, **kwargs)
    elif method == "edges":
        return remove_background_edges(img, **kwargs)
    elif method == "grabcut":
        return remove_background_grabcut(img, **kwargs)
    else:
        raise ValueError(
            f"Unknown background removal method {method!r}. "
            f"Choose 'thresh', 'edges', or 'grabcut'."
        )


# ─── batch_remove_background ──────────────────────────────────────────────────

def batch_remove_background(
    images: List[np.ndarray],
    method: str = "thresh",
    **kwargs,
) -> List[BackgroundRemovalResult]:
    """
    Применяет auto_remove_background ко всем изображениям в списке.

    Args:
        images: Список BGR или grayscale изображений uint8.
        method: 'thresh' | 'edges' | 'grabcut'.
        **kwargs: Параметры auto_remove_background.

    Returns:
        Список BackgroundRemovalResult той же длины.

    Raises:
        ValueError: Неизвестный метод (проверяется заранее).
    """
    if method not in ("thresh", "edges", "grabcut"):
        raise ValueError(
            f"Unknown background removal method {method!r}. "
            f"Choose 'thresh', 'edges', or 'grabcut'."
        )
    return [auto_remove_background(img, method=method, **kwargs)
            for img in images]
