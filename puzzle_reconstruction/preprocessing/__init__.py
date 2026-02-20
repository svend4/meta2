"""
Предобработка фрагментов документа.

Модули:
    segmentation  — выделение маски фрагмента (Otsu / Adaptive / GrabCut)
    contour       — извлечение и нормализация внешнего контура
    orientation   — оценка и коррекция угла поворота
    color_norm    — нормализация цвета (CLAHE, Gray World, гамма)
"""
from .segmentation import segment_fragment
from .contour import extract_contour
from .orientation import estimate_orientation, rotate_to_upright
from .color_norm import (
    normalize_color,
    clahe_normalize,
    white_balance,
    gamma_correction,
    normalize_brightness,
    batch_normalize,
)

__all__ = [
    "segment_fragment",
    "extract_contour",
    "estimate_orientation",
    "rotate_to_upright",
    "normalize_color",
    "clahe_normalize",
    "white_balance",
    "gamma_correction",
    "normalize_brightness",
    "batch_normalize",
]
