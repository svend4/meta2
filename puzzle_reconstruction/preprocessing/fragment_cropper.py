"""
Обрезка фрагментов к содержательной области.

Удаляет пустые (светлые) края вокруг фрагмента, находит минимальный
ограничивающий прямоугольник содержательной области и возвращает
обрезанное изображение с настраиваемым отступом (padding).

Классы:
    CropResult — результат обрезки (изображение, bbox, метрики)

Функции:
    find_content_bbox — поиск bbox содержательной области по маске
    crop_to_content   — обрезка с padding и проверкой минимального размера
    pad_image         — добавление равномерного поля заливки
    auto_crop         — авто-обрезка с пороговой бинаризацией
    batch_crop        — пакетная обрезка списка изображений
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── CropResult ───────────────────────────────────────────────────────────────

@dataclass
class CropResult:
    """
    Результат обрезки фрагмента.

    Attributes:
        cropped:        Обрезанное изображение (uint8).
        bbox:           (x, y, w, h) — bbox в координатах исходного изображения.
        padding:        Применённый отступ (пикс).
        original_shape: (h, w) исходного изображения.
        method:         Метод обрезки ('content' | 'auto').
        params:         Пороги и вспомогательные параметры.
    """
    cropped:        np.ndarray
    bbox:           Tuple[int, int, int, int]
    padding:        int
    original_shape: Tuple[int, int]
    method:         str
    params:         Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        x, y, w, h = self.bbox
        ch, cw = self.cropped.shape[:2]
        return (f"CropResult(cropped={cw}×{ch}, "
                f"bbox=({x},{y},{w},{h}), "
                f"pad={self.padding}, "
                f"method={self.method!r})")


# ─── Вспомогательные ──────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ─── find_content_bbox ────────────────────────────────────────────────────────

def find_content_bbox(img:       np.ndarray,
                       bg_thresh: int = 240,
                       margin:    int = 0) -> Tuple[int, int, int, int]:
    """
    Находит ограничивающий прямоугольник содержательной (тёмной) области.

    Пикселями «содержания» считаются точки с яркостью ≤ bg_thresh.

    Args:
        img:       BGR или grayscale изображение (uint8).
        bg_thresh: Порог яркости для отделения фона (включительно).
        margin:    Дополнительный отступ внутри исходного bbox (пикс).

    Returns:
        (x, y, w, h) — bbox содержательной области или (0,0,w,h)
        для всего изображения при отсутствии контента.
    """
    gray = _to_gray(img)
    h, w = gray.shape

    mask = (gray <= bg_thresh).astype(np.uint8)

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        # Нет контента → весь кадр
        return (0, 0, w, h)

    y0 = max(0, int(rows[0]) - margin)
    y1 = min(h, int(rows[-1]) + 1 + margin)
    x0 = max(0, int(cols[0]) - margin)
    x1 = min(w, int(cols[-1]) + 1 + margin)

    return (x0, y0, x1 - x0, y1 - y0)


# ─── pad_image ────────────────────────────────────────────────────────────────

def pad_image(img:     np.ndarray,
               padding: int,
               fill:    int = 255) -> np.ndarray:
    """
    Добавляет равномерное поле вокруг изображения.

    Args:
        img:     BGR или grayscale изображение (uint8).
        padding: Ширина поля в пикселях (со всех сторон).
        fill:    Значение заполнения (0..255).

    Returns:
        Расширенное изображение.
    """
    p = max(0, padding)
    if p == 0:
        return img.copy()
    return cv2.copyMakeBorder(img, p, p, p, p,
                               cv2.BORDER_CONSTANT, value=fill)


# ─── crop_to_content ──────────────────────────────────────────────────────────

def crop_to_content(img:       np.ndarray,
                     padding:   int = 4,
                     bg_thresh: int = 240,
                     min_size:  int = 4,
                     fill:      int = 255) -> CropResult:
    """
    Обрезает изображение к содержательной области с отступом.

    Алгоритм:
      1. Находит bbox содержательной области (find_content_bbox).
      2. Обрезает с учётом padding (расширение bbox).
      3. Гарантирует минимальный размер min_size × min_size.

    Args:
        img:       BGR или grayscale изображение (uint8).
        padding:   Отступ от края контента (пикс).
        bg_thresh: Порог яркости фона.
        min_size:  Минимальный размер результата (пикс).
        fill:      Цвет фона при добавлении padding.

    Returns:
        CropResult с method='content'.
    """
    h_orig, w_orig = img.shape[:2]
    x, y, w, h     = find_content_bbox(img, bg_thresh=bg_thresh)

    # Расширяем bbox на padding
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(w_orig, x + w + padding)
    y1 = min(h_orig, y + h + padding)

    # Обрезка
    cropped = img[y0:y1, x0:x1]

    # Гарантируем минимальный размер
    ch, cw = cropped.shape[:2]
    if ch < min_size or cw < min_size:
        need_h = max(0, min_size - ch)
        need_w = max(0, min_size - cw)
        cropped = cv2.copyMakeBorder(
            cropped,
            need_h // 2, need_h - need_h // 2,
            need_w // 2, need_w - need_w // 2,
            cv2.BORDER_CONSTANT, value=fill,
        )

    return CropResult(
        cropped=cropped,
        bbox=(x0, y0, x1 - x0, y1 - y0),
        padding=padding,
        original_shape=(h_orig, w_orig),
        method="content",
        params={"bg_thresh": bg_thresh, "min_size": min_size, "fill": fill},
    )


# ─── auto_crop ────────────────────────────────────────────────────────────────

def auto_crop(img:       np.ndarray,
               padding:   int = 4,
               bg_thresh: int = 240,
               min_size:  int = 8) -> CropResult:
    """
    Авто-обрезка фрагмента с адаптивным порогом.

    Если заданный порог bg_thresh не даёт контента (изображение однотонное),
    автоматически снижает порог до медианного значения яркости.

    Args:
        img:       BGR или grayscale изображение.
        padding:   Отступ (пикс).
        bg_thresh: Начальный порог фона.
        min_size:  Минимальный размер результата.

    Returns:
        CropResult с method='auto'.
    """
    gray    = _to_gray(img)
    thresh  = bg_thresh
    x, y, w, h = find_content_bbox(img, bg_thresh=thresh)

    if w <= 0 or h <= 0:
        # Порог не сработал → адаптируем
        median  = float(np.median(gray))
        thresh  = max(1, int(median * 0.9))
        x, y, w, h = find_content_bbox(img, bg_thresh=thresh)

    h_orig, w_orig = img.shape[:2]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(w_orig, x + w + padding)
    y1 = min(h_orig, y + h + padding)

    cropped = img[y0:y1, x0:x1]
    ch, cw  = cropped.shape[:2]

    if ch < min_size or cw < min_size:
        need_h = max(0, min_size - ch)
        need_w = max(0, min_size - cw)
        cropped = cv2.copyMakeBorder(
            cropped,
            need_h // 2, need_h - need_h // 2,
            need_w // 2, need_w - need_w // 2,
            cv2.BORDER_CONSTANT, value=255,
        )

    return CropResult(
        cropped=cropped,
        bbox=(x0, y0, x1 - x0, y1 - y0),
        padding=padding,
        original_shape=(h_orig, w_orig),
        method="auto",
        params={"bg_thresh": thresh, "min_size": min_size},
    )


# ─── batch_crop ───────────────────────────────────────────────────────────────

def batch_crop(images:    List[np.ndarray],
               padding:   int = 4,
               bg_thresh: int = 240,
               method:    str = "auto") -> List[CropResult]:
    """
    Пакетная обрезка списка изображений.

    Args:
        images:    Список BGR или grayscale изображений.
        padding:   Отступ (пикс).
        bg_thresh: Порог яркости фона.
        method:    'content' | 'auto'.

    Returns:
        Список CropResult (по одному на изображение).

    Raises:
        ValueError: Если метод неизвестен.
    """
    if method == "content":
        fn = lambda img: crop_to_content(img, padding=padding,
                                           bg_thresh=bg_thresh)
    elif method == "auto":
        fn = lambda img: auto_crop(img, padding=padding, bg_thresh=bg_thresh)
    else:
        raise ValueError(
            f"Unknown method {method!r}. Use 'content' or 'auto'."
        )
    return [fn(img) for img in images]
