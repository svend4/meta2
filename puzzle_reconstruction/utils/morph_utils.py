"""
Утилиты морфологической обработки изображений для реконструкции пазла.

Предоставляет функции применения морфологических операций к бинарным
и полутоновым изображениям, скелетизации, маркировки регионов
и пакетной обработки.

Классы:
    MorphConfig  — параметры морфологических операций

Функции:
    apply_erosion          — эрозия
    apply_dilation         — дилатация
    apply_opening          — открытие (эрозия → дилатация)
    apply_closing          — закрытие (дилатация → эрозия)
    get_skeleton           — скелетизация бинарного изображения
    label_regions          — маркировка связных компонент
    filter_regions_by_size — фильтрация компонент по площади
    compute_region_stats   — статистика связных компонент
    batch_morphology       — пакетное применение морфологических операций
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── MorphConfig ──────────────────────────────────────────────────────────────

@dataclass
class MorphConfig:
    """Параметры морфологических операций.

    Attributes:
        kernel_size:  Размер ядра (нечётное, >= 1).
        kernel_shape: Форма ядра: ``'rect'``, ``'ellipse'``, ``'cross'``.
        iterations:   Число итераций применения операции (>= 1).
    """
    kernel_size: int = 3
    kernel_shape: str = "rect"
    iterations: int = 1

    def __post_init__(self) -> None:
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {self.kernel_size}"
            )
        valid = {"rect", "ellipse", "cross"}
        if self.kernel_shape not in valid:
            raise ValueError(
                f"kernel_shape must be one of {valid}, got {self.kernel_shape!r}"
            )
        if self.iterations < 1:
            raise ValueError(
                f"iterations must be >= 1, got {self.iterations}"
            )

    def build_kernel(self) -> np.ndarray:
        """Построить ядро OpenCV согласно конфигурации."""
        shape_map = {
            "rect": cv2.MORPH_RECT,
            "ellipse": cv2.MORPH_ELLIPSE,
            "cross": cv2.MORPH_CROSS,
        }
        return cv2.getStructuringElement(
            shape_map[self.kernel_shape],
            (self.kernel_size, self.kernel_size),
        )


# ─── Вспомогательные ──────────────────────────────────────────────────────────

def _ensure_gray_uint8(img: np.ndarray) -> np.ndarray:
    """Привести изображение к grayscale uint8."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _ensure_binary(img: np.ndarray) -> np.ndarray:
    """Привести grayscale-изображение к бинарному uint8 (0/255)."""
    gray = _ensure_gray_uint8(img)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


# ─── Публичные функции ────────────────────────────────────────────────────────

def apply_erosion(
    img: np.ndarray,
    cfg: MorphConfig | None = None,
) -> np.ndarray:
    """Применить эрозию к изображению.

    Args:
        img: Grayscale или BGR изображение uint8.
        cfg: Параметры; None → MorphConfig().

    Returns:
        uint8 изображение той же формы.
    """
    if cfg is None:
        cfg = MorphConfig()
    kernel = cfg.build_kernel()
    return cv2.erode(img, kernel, iterations=cfg.iterations)


def apply_dilation(
    img: np.ndarray,
    cfg: MorphConfig | None = None,
) -> np.ndarray:
    """Применить дилатацию к изображению.

    Args:
        img: Grayscale или BGR изображение uint8.
        cfg: Параметры; None → MorphConfig().

    Returns:
        uint8 изображение той же формы.
    """
    if cfg is None:
        cfg = MorphConfig()
    kernel = cfg.build_kernel()
    return cv2.dilate(img, kernel, iterations=cfg.iterations)


def apply_opening(
    img: np.ndarray,
    cfg: MorphConfig | None = None,
) -> np.ndarray:
    """Применить морфологическое открытие (эрозия + дилатация).

    Args:
        img: Grayscale или BGR изображение uint8.
        cfg: Параметры; None → MorphConfig().

    Returns:
        uint8 изображение той же формы.
    """
    if cfg is None:
        cfg = MorphConfig()
    kernel = cfg.build_kernel()
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=cfg.iterations)


def apply_closing(
    img: np.ndarray,
    cfg: MorphConfig | None = None,
) -> np.ndarray:
    """Применить морфологическое закрытие (дилатация + эрозия).

    Args:
        img: Grayscale или BGR изображение uint8.
        cfg: Параметры; None → MorphConfig().

    Returns:
        uint8 изображение той же формы.
    """
    if cfg is None:
        cfg = MorphConfig()
    kernel = cfg.build_kernel()
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=cfg.iterations)


def get_skeleton(img: np.ndarray) -> np.ndarray:
    """Вычислить скелет бинарного изображения.

    Изображение автоматически бинаризуется через Otsu. Скелет вычисляется
    итеративно методом «erosion + opening» по алгоритму Zhang–Suen.

    Args:
        img: Grayscale или BGR изображение uint8.

    Returns:
        uint8 бинарное изображение (0 или 255) той же высоты и ширины.
    """
    binary = _ensure_binary(img)
    skeleton = np.zeros_like(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    tmp = binary.copy()
    # Use BORDER_CONSTANT so pixels outside the image are treated as
    # background (0), ensuring the loop always terminates.
    while True:
        eroded = cv2.erode(tmp, kernel,
                           borderType=cv2.BORDER_CONSTANT, borderValue=0)
        opened = cv2.dilate(eroded, kernel,
                            borderType=cv2.BORDER_CONSTANT, borderValue=0)
        diff = cv2.subtract(tmp, opened)
        skeleton = cv2.bitwise_or(skeleton, diff)
        tmp = eroded.copy()
        if cv2.countNonZero(tmp) == 0:
            break
    return skeleton


def label_regions(
    img: np.ndarray,
    connectivity: int = 8,
) -> Tuple[int, np.ndarray]:
    """Пометить связные компоненты в бинарном изображении.

    Args:
        img:          Grayscale или BGR изображение uint8.
        connectivity: 4 или 8 (связность соседних пикселей).

    Returns:
        ``(n_labels, label_map)`` — число меток и карта меток int32.

    Raises:
        ValueError: Если connectivity не 4 и не 8.
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    binary = _ensure_binary(img)
    n_labels, label_map = cv2.connectedComponents(binary, connectivity=connectivity)
    return int(n_labels - 1), label_map.astype(np.int32)


def filter_regions_by_size(
    img: np.ndarray,
    min_area: int = 0,
    max_area: int = 0,
    connectivity: int = 8,
) -> np.ndarray:
    """Оставить только компоненты заданного диапазона площадей.

    Args:
        img:          Grayscale или BGR изображение uint8.
        min_area:     Минимальная площадь (включительно, >= 0).
        max_area:     Максимальная площадь (0 = нет ограничения сверху).
        connectivity: 4 или 8.

    Returns:
        uint8 бинарное изображение той же формы с отфильтрованными компонентами.

    Raises:
        ValueError: Если min_area < 0.
    """
    if min_area < 0:
        raise ValueError(f"min_area must be >= 0, got {min_area}")
    binary = _ensure_binary(img)
    n, label_map = cv2.connectedComponents(binary, connectivity=connectivity)
    result = np.zeros_like(binary)
    for lbl in range(1, n):
        mask = label_map == lbl
        area = int(mask.sum())
        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        result[mask] = 255
    return result


def compute_region_stats(
    img: np.ndarray,
    connectivity: int = 8,
) -> List[Dict[str, float]]:
    """Вычислить статистику каждой связной компоненты.

    Args:
        img:          Grayscale или BGR изображение uint8.
        connectivity: 4 или 8.

    Returns:
        Список словарей с ключами:
        ``label``, ``area``, ``cx``, ``cy``, ``bbox_x``, ``bbox_y``,
        ``bbox_w``, ``bbox_h``, ``aspect_ratio``.
        Пустой список, если нет компонент.
    """
    binary = _ensure_binary(img)
    n, label_map, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=connectivity
    )
    result = []
    for lbl in range(1, n):  # skip background (0)
        x, y, w, h, area = (int(stats[lbl, k]) for k in (
            cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
            cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT,
            cv2.CC_STAT_AREA,
        ))
        cx, cy = float(centroids[lbl, 0]), float(centroids[lbl, 1])
        aspect = float(w) / float(h) if h > 0 else 0.0
        result.append({
            "label": float(lbl),
            "area": float(area),
            "cx": cx,
            "cy": cy,
            "bbox_x": float(x),
            "bbox_y": float(y),
            "bbox_w": float(w),
            "bbox_h": float(h),
            "aspect_ratio": aspect,
        })
    return result


def batch_morphology(
    images: List[np.ndarray],
    operation: str = "opening",
    cfg: MorphConfig | None = None,
) -> List[np.ndarray]:
    """Применить морфологическую операцию ко всем изображениям пакета.

    Args:
        images:    Список изображений uint8.
        operation: Одна из ``'erosion'``, ``'dilation'``, ``'opening'``,
                   ``'closing'``.
        cfg:       Параметры; None → MorphConfig().

    Returns:
        Список обработанных изображений той же длины.

    Raises:
        ValueError: Если operation неизвестна.
    """
    ops = {
        "erosion":  apply_erosion,
        "dilation": apply_dilation,
        "opening":  apply_opening,
        "closing":  apply_closing,
    }
    if operation not in ops:
        raise ValueError(
            f"operation must be one of {list(ops)}, got {operation!r}"
        )
    fn = ops[operation]
    return [fn(img, cfg) for img in images]
