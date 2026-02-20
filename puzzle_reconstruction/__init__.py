"""
puzzle_reconstruction — система восстановления разорванных документов.

Публичный API верхнего уровня:

    from puzzle_reconstruction import (
        Fragment, Assembly, Config,
        cluster_fragments, ClusteringResult,
    )
    from puzzle_reconstruction.export import (
        render_canvas, render_heatmap, render_mosaic,
        save_png, save_pdf, comparison_strip,
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

__version__ = "0.2.0"
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
    # Версия
    "__version__",
]
