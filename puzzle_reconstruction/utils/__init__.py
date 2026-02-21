"""
Утилиты: логирование, геометрия, кэш дескрипторов, ввод-вывод.

Модули:
    logger   — структурированное логирование с цветами (get_logger, stage, PipelineTimer)
    geometry — геометрические операции (rotation_matrix_2d, polygon_area, poly_iou, …)
    cache    — LRU-кэш и дисковый кэш дескрипторов (DescriptorCache, DiskCache, @cached)
    io       — загрузка/сохранение фрагментов и Assembly (load_image_dir, save_assembly_json, …)
    profiler   — профилировщик шагов пайплайна (StepProfile, PipelineProfiler, @timed)
    visualizer — утилиты визуализации (word boxes, contours, matches, confidence bar)
"""
from .logger import (
    get_logger,
    stage,
    ProgressBar,
    PipelineTimer,
    log,
)
from .geometry import (
    rotation_matrix_2d,
    rotate_points,
    polygon_area,
    polygon_centroid,
    bbox_from_points,
    resample_curve,
    align_centroids,
    poly_iou,
    point_in_polygon,
    normalize_contour,
    smooth_contour,
    curvature,
)
from .cache import (
    DescriptorCache,
    DiskCache,
    descriptor_key,
    cached,
    get_default_cache,
    clear_default_cache,
)
from .io import (
    load_image_dir,
    fragments_from_images,
    save_assembly_json,
    load_assembly_json,
    save_fragments_npz,
    load_fragments_npz,
    FragmentSetInfo,
)
from .profiler import (
    StepProfile,
    PipelineProfiler,
    profile_function,
    timed,
    format_duration,
    compare_profilers,
)
from .visualizer import (
    VisConfig,
    draw_word_boxes,
    draw_fragment_boxes,
    draw_edge_matches,
    draw_contour,
    draw_assembly_layout,
    draw_skew_angle,
    draw_confidence_bar,
    tile_images,
)

__all__ = [
    # Логирование
    "get_logger",
    "stage",
    "ProgressBar",
    "PipelineTimer",
    "log",
    # Геометрия
    "rotation_matrix_2d",
    "rotate_points",
    "polygon_area",
    "polygon_centroid",
    "bbox_from_points",
    "resample_curve",
    "align_centroids",
    "poly_iou",
    "point_in_polygon",
    "normalize_contour",
    "smooth_contour",
    "curvature",
    # Кэш
    "DescriptorCache",
    "DiskCache",
    "descriptor_key",
    "cached",
    "get_default_cache",
    "clear_default_cache",
    # Ввод-вывод
    "load_image_dir",
    "fragments_from_images",
    "save_assembly_json",
    "load_assembly_json",
    "save_fragments_npz",
    "load_fragments_npz",
    "FragmentSetInfo",
    # Профилировщик
    "StepProfile",
    "PipelineProfiler",
    "profile_function",
    "timed",
    "format_duration",
    "compare_profilers",
    # Визуализация
    "VisConfig",
    "draw_word_boxes",
    "draw_fragment_boxes",
    "draw_edge_matches",
    "draw_contour",
    "draw_assembly_layout",
    "draw_skew_angle",
    "draw_confidence_bar",
    "tile_images",
]
