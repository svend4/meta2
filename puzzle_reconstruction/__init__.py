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

Модуль scoring — оценка и ранжирование совместимости:

    from puzzle_reconstruction import (
        normalize_score_matrix, NormMethod,
        select_threshold, apply_threshold, ThresholdConfig,
        filter_pairs, FilterConfig,
        run_consistency_check, ConsistencyReport, ConsistencyIssue,
        aggregate_evidence, EvidenceConfig,
        evaluate_match, MatchEval,
    )

Модуль io — загрузка и экспорт:

    from puzzle_reconstruction import (
        load_from_directory, LoadConfig, LoadedImage,
        export_result, ExportConfig, AssemblyResult,
        write_json, write_csv, WriterConfig, MetadataRecord,
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

# ── scoring ──────────────────────────────────────────────────────────────────
from .scoring.score_normalizer import normalize_score_matrix, NormMethod, NormalizedMatrix
from .scoring.threshold_selector import (
    select_threshold, apply_threshold,
    ThresholdConfig, ThresholdResult,
)
from .scoring.pair_filter import filter_pairs, FilterConfig, FilterReport
from .scoring.consistency_checker import (
    run_consistency_check,
    ConsistencyReport,
    ConsistencyIssue,
)
from .scoring.evidence_aggregator import aggregate_evidence, EvidenceConfig, EvidenceScore
from .scoring.match_evaluator import evaluate_match, MatchEval, EvalReport

# ── io ────────────────────────────────────────────────────────────────────────
from .io.image_loader import load_from_directory, LoadConfig, LoadedImage
from .io.result_exporter import export_result, ExportConfig, AssemblyResult
from .io.metadata_writer import (
    write_json, write_csv, write_summary,
    WriterConfig, MetadataRecord, MetadataCollection,
)

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
    # scoring — нормализация
    "normalize_score_matrix",
    "NormMethod",
    "NormalizedMatrix",
    # scoring — порог
    "select_threshold",
    "apply_threshold",
    "ThresholdConfig",
    "ThresholdResult",
    # scoring — фильтрация пар
    "filter_pairs",
    "FilterConfig",
    "FilterReport",
    # scoring — согласованность
    "run_consistency_check",
    "ConsistencyReport",
    "ConsistencyIssue",
    # scoring — агрегация свидетельств
    "aggregate_evidence",
    "EvidenceConfig",
    "EvidenceScore",
    # scoring — оценка матчей
    "evaluate_match",
    "MatchEval",
    "EvalReport",
    # io — загрузка
    "load_from_directory",
    "LoadConfig",
    "LoadedImage",
    # io — экспорт
    "export_result",
    "ExportConfig",
    "AssemblyResult",
    # io — метаданные
    "write_json",
    "write_csv",
    "write_summary",
    "WriterConfig",
    "MetadataRecord",
    "MetadataCollection",
]
