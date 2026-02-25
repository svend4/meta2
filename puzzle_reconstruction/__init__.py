"""
puzzle_reconstruction — система восстановления разорванных документов.

Публичный API верхнего уровня:

    from puzzle_reconstruction import (
        Fragment, Assembly, Config,
        Pipeline, PipelineResult,
        cluster_fragments, ClusteringResult,
    )
    from puzzle_reconstruction.export import (
        render_canvas, render_heatmap, render_mosaic,
        save_png, save_pdf, comparison_strip,
    )
    from puzzle_reconstruction.matching import icp_align, contour_icp
    from puzzle_reconstruction.utils import (
        polygon_area, poly_iou, resample_curve, normalize_contour,
    )
"""
from .models import (
    Fragment,
    Assembly,
    CompatEntry,
    EdgeSignature,
    FractalSignature,
    TangramSignature,
    ShapeClass,
    EdgeSide,
)
from .config import Config
from .clustering import cluster_fragments, ClusteringResult, split_by_cluster
from .pipeline import Pipeline, PipelineResult

__version__ = "0.4.0b1"
__all__ = [
    # Модели данных
    "Fragment",
    "Assembly",
    "CompatEntry",
    "EdgeSignature",
    "FractalSignature",
    "TangramSignature",
    "ShapeClass",
    "EdgeSide",
    # Конфиг
    "Config",
    # Кластеризация
    "cluster_fragments",
    "ClusteringResult",
    "split_by_cluster",
    # Пайплайн
    "Pipeline",
    "PipelineResult",
    # Версия
    "__version__",
]
