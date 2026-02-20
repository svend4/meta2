"""
Предобработка фрагментов документа.

Модули:
    segmentation  — выделение маски фрагмента (Otsu / Adaptive / GrabCut)
    contour       — извлечение и нормализация внешнего контура
    orientation   — оценка и коррекция угла поворота
    color_norm    — нормализация цвета (CLAHE, Gray World, гамма)
    denoise       — шумоподавление (Gaussian, Median, Bilateral, NLM, auto)
    augment       — аугментация данных (crop, rotate, noise, jitter, JPEG)
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
from .denoise import (
    gaussian_denoise,
    median_denoise,
    bilateral_denoise,
    nlmeans_denoise,
    auto_denoise,
    denoise_batch,
)
from .augment import (
    random_crop,
    random_rotate,
    add_gaussian_noise,
    add_salt_pepper,
    brightness_jitter,
    simulate_scan_noise,
    jpeg_compress,
    augment_batch,
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
    "gaussian_denoise",
    "median_denoise",
    "bilateral_denoise",
    "nlmeans_denoise",
    "auto_denoise",
    "denoise_batch",
    "random_crop",
    "random_rotate",
    "add_gaussian_noise",
    "add_salt_pepper",
    "brightness_jitter",
    "simulate_scan_noise",
    "jpeg_compress",
    "augment_batch",
]
