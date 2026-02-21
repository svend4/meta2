"""
Предобработка фрагментов документа.

Модули:
    segmentation    — выделение маски фрагмента (Otsu / Adaptive / GrabCut)
    contour         — извлечение и нормализация внешнего контура
    orientation     — оценка и коррекция угла поворота
    color_norm      — нормализация цвета (CLAHE, Gray World, гамма)
    denoise         — шумоподавление (Gaussian, Median, Bilateral, NLM, auto)
    augment         — аугментация данных (crop, rotate, noise, jitter, JPEG)
    edge_detector   — специализированное детектирование краёв (Canny, Sobel, LoG)
    skew_correction — коррекция наклона (Hough, projection, FFT)
    perspective     — коррекция перспективных искажений (contour, Hough)
    noise_reduction — расширенное шумоподавление (DenoiseResult, NLM, bilateral,
                      morphological, smart_denoise, batch_denoise)
    contrast        — улучшение контраста (ContrastResult, CLAHE, histeq,
                      gamma, stretch, retinex, auto_enhance, batch_enhance)
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
from .edge_detector import (
    detect_edges,
    adaptive_canny,
    sobel_edges,
    laplacian_edges,
    refine_edge_contour,
    edge_density,
    edge_orientation_hist,
    EdgeDetectionResult,
)
from .skew_correction import (
    SkewResult,
    detect_skew_hough,
    detect_skew_projection,
    detect_skew_fft,
    correct_skew,
    auto_correct_skew,
    skew_confidence,
    batch_correct_skew,
)
from .perspective import (
    PerspectiveResult,
    order_corners,
    four_point_transform,
    detect_corners_contour,
    detect_corners_hough,
    correct_perspective,
    auto_correct_perspective,
    batch_correct_perspective,
)
from .noise_reduction import (
    DenoiseResult,
    estimate_noise_level,
    denoise_gaussian,
    denoise_median,
    denoise_nlm,
    denoise_bilateral,
    denoise_morphological,
    smart_denoise,
    batch_denoise,
)
from .contrast import (
    ContrastResult,
    measure_contrast,
    enhance_clahe,
    enhance_histeq,
    enhance_gamma,
    enhance_stretch,
    enhance_retinex,
    auto_enhance,
    batch_enhance,
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
    "detect_edges",
    "adaptive_canny",
    "sobel_edges",
    "laplacian_edges",
    "refine_edge_contour",
    "edge_density",
    "edge_orientation_hist",
    "EdgeDetectionResult",
    "SkewResult",
    "detect_skew_hough",
    "detect_skew_projection",
    "detect_skew_fft",
    "correct_skew",
    "auto_correct_skew",
    "skew_confidence",
    "batch_correct_skew",
    "PerspectiveResult",
    "order_corners",
    "four_point_transform",
    "detect_corners_contour",
    "detect_corners_hough",
    "correct_perspective",
    "auto_correct_perspective",
    "batch_correct_perspective",
    # Расширенное шумоподавление
    "DenoiseResult",
    "estimate_noise_level",
    "denoise_gaussian",
    "denoise_median",
    "denoise_nlm",
    "denoise_bilateral",
    "denoise_morphological",
    "smart_denoise",
    "batch_denoise",
    # Улучшение контраста
    "ContrastResult",
    "measure_contrast",
    "enhance_clahe",
    "enhance_histeq",
    "enhance_gamma",
    "enhance_stretch",
    "enhance_retinex",
    "auto_enhance",
    "batch_enhance",
]
