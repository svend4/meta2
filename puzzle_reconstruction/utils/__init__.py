"""
Утилиты: логирование, геометрия.

Модули:
    logger   — структурированное логирование с цветами (get_logger, stage, PipelineTimer)
    geometry — геометрические операции (rotation_matrix_2d, polygon_area, poly_iou, …)
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
]
