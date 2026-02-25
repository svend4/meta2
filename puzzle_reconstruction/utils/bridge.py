"""
Мост интеграции утилит (Bridge #6) — реестр 130 sleeping utility modules.

Подключает модули puzzle_reconstruction/utils/, которые экспортированы в
__init__.py, но не использовались в основном пайплайне напрямую.

Архитектура:
    Утилиты разделены на категории по функциональному назначению:

    core        — инфраструктура (logger, profiler, event_bus, metric_tracker,
                  batch_processor, pipeline_runner, progress_tracker, event_log,
                  result_cache, cache, cache_manager, tracker_utils)

    geometry    — пространственные вычисления (geometry, geometry_utils,
                  bbox_utils, polygon_utils, polygon_ops_utils, spatial_index,
                  alignment_utils, icp_utils, transform_utils, rotation_utils,
                  image_transform_utils)

    image       — обработка изображений (image_io, image_stats, patch_extractor,
                  patch_utils, blend_utils, render_utils, morph_utils, mask_utils,
                  tile_utils, threshold_utils, visualizer, image_cluster_utils,
                  image_pipeline_utils)

    signal      — 1D обработка сигналов (signal_utils, smoothing_utils,
                  window_utils, interpolation_utils, frequency_utils, gradient_utils)

    metrics     — качество/оценка (metrics, stats_utils, score_aggregator,
                  distance_utils, distance_matrix, clustering_utils,
                  normalization_utils, feature_selector, score_matrix_utils,
                  score_norm_utils, noise_stats_utils, normalize_noise_utils)

    graph       — граф/пути (graph_utils, path_plan_utils, sparse_utils,
                  topology_utils)

    contour     — контуры (contour_utils, contour_profile_utils, contour_sampler,
                  shape_match_utils, segment_utils, curvature_utils, curve_metrics)

    color       — цвет/гистограммы (color_utils, histogram_utils, color_hist_utils,
                  freq_metric_utils)

    keypoint    — ключевые точки (keypoint_utils, descriptor_utils)

    fragment    — уровень фрагмента (fragment_stats, fragment_filter_utils,
                  edge_profile_utils, edge_profiler, edge_scorer)

    scoring     — оценка пар/патчей (pair_score_utils, patch_score_utils,
                  region_score_utils, rotation_score_utils, score_seam_utils,
                  voting_utils, sequence_utils, match_rank_utils, quality_score_utils,
                  consensus_score_utils, annealing_score_utils, overlap_score_utils,
                  seq_gap_utils, scoring_pipeline_utils, sampling_utils)

    assembly    — уровень сборки (assembly_score_utils, canvas_build_utils,
                  placement_score_utils, placement_metrics_utils,
                  position_tracking_utils, rank_result_utils, candidate_rank_utils,
                  ranking_layout_utils, ranking_validation_utils)

    records     — типы данных/записи (assembly_records, assembly_config_utils,
                  classification_freq_records, color_edge_export_utils,
                  contour_contrast_records, contour_curvature_records,
                  descriptor_edge_records, edge_fragment_records,
                  event_affine_utils, gap_geometry_records, gradient_graph_records,
                  illum_layout_records, matching_consistency_records,
                  region_seam_records, scorer_state_records, texture_pipeline_utils,
                  orient_topology_utils, orient_skew_utils, window_tile_records)

    io          — ввод/вывод и конфиг (io, config_manager, config_utils,
                  array_utils)

    annealing   — имитация отжига (annealing_schedule, annealing_score_utils)

Использование:
    from puzzle_reconstruction.utils.bridge import (
        build_util_registry,
        list_utils,
        get_util,
        UTIL_CATEGORIES,
    )

    registry = build_util_registry()
    fn = registry.get("compute_image_stats")
    if fn:
        stats = fn(image)

    # Получить все geometry-утилиты:
    names = UTIL_CATEGORIES["geometry"]
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── Категории утилит ────────────────────────────────────────────────────────

UTIL_CATEGORIES: Dict[str, List[str]] = {
    # Инфраструктурные утилиты (мониторинг, кэш, события)
    "core": [
        "get_logger",
        "PipelineProfiler",
        "make_event_bus",
        "make_event_log",
        "make_metric_tracker",
        "process_items",
        "run_pipeline",
        "make_progress_tracker",
        "ResultCache",
        "DescriptorCache",
        "LRUCache",
        "TrackerConfig",
    ],
    # Пространственные и геометрические вычисления
    "geometry": [
        "rotation_matrix_2d",
        "BoundingBox",
        "BBox",
        "polygon_area",
        "PolygonOpsConfig",
        "build_spatial_index",
        "AlignmentConfig",
        "icp_centroid",
        "rotate_image",
        "rotate_image_angle",
        "ImageTransformConfig",
    ],
    # Обработка изображений
    "image": [
        "load_image",
        "compute_image_stats",
        "PatchSet",
        "extract_patch",
        "alpha_blend",
        "CanvasConfig",
        "MorphConfig",
        "create_alpha_mask",
        "TileConfig",
        "apply_threshold",
        "VisConfig",
        "ImageStatsAnalysisConfig",
        "FrequencyMatchSummary",
    ],
    # 1D обработка сигналов
    "signal": [
        "smooth_signal",
        "moving_average",
        "apply_window_function",
        "lerp",
        "FrequencyConfig",
        "compute_gradient_magnitude",
    ],
    # Метрики и оценка качества
    "metrics": [
        "ReconstructionMetrics",
        "describe_stats",
        "AggregationResult",
        "euclidean_distance",
        "euclidean_distance_matrix",
        "kmeans_cluster",
        "l2_normalize",
        "variance_selection",
        "ScoreMatrixConfig",
        "ScoreNormConfig",
        "NoiseStatsConfig",
        "NormResultConfig",
    ],
    # Граф и пути
    "graph": [
        "build_graph",
        "PathPlanConfig",
        "to_sparse_entries",
        "TopologyConfig",
    ],
    # Контуры
    "contour": [
        "simplify_contour",
        "sample_profile_along_contour",
        "SampledContour",
        "ShapeMatchConfig",
        "SegmentConfig",
        "compute_curvature",
        "curve_l2",
    ],
    # Цвет и гистограммы
    "color": [
        "to_gray",
        "compute_1d_histogram",
        "ColorHistConfig",
        "BandEnergyRecord",
    ],
    # Ключевые точки и дескрипторы
    "keypoint": [
        "detect_keypoints",
        "DescriptorConfig",
    ],
    # Уровень фрагмента
    "fragment": [
        "FragmentMetrics",
        "FragmentFilterConfig",
        "EdgeProfileConfig",
        "EdgeScoreConfig",
    ],
    # Оценка пар и патчей
    "scoring": [
        "PairScoreConfig",
        "PatchScoreConfig",
        "RegionScoreConfig",
        "RotationScoreConfig",
        "SeamNormalizationRecord",
        "cast_pair_votes",
        "rank_sequence",
        "RankingConfig",
        "QualityScoreConfig",
        "ConsensusScoreConfig",
        "AnnealingScoreConfig",
        "OverlapScoreConfig",
        "SequenceScoreConfig",
        "ScoringPipelineReport",
        "SamplingConfig",
    ],
    # Уровень сборки
    "assembly": [
        "AssemblySummary",
        "CanvasBuildConfig",
        "PlacementScoreConfig",
        "PlacementMetricsConfig",
        "PositionQualityRecord",
        "RankResultConfig",
        "CandidateRankConfig",
        "GlobalRankingConfig",
        "RankingRunRecord",
    ],
    # Типы данных и записи
    "records": [
        "CollisionRecord",
        "AssemblyStateRecord",
        "FragmentClassifyRecord",
        "ColorMatchAnalysisConfig",
        "ContourProcessRecord",
        "AnnealingRunRecord",
        "DescriptorCombineRecord",
        "EdgeCompareRecord",
        "EventRecordEntry",
        "GapScoringRecord",
        "GradientFlowRecord",
        "IllumNormRecord",
        "BoundaryMatchRecord",
        "RegionPairRecord",
        "AssemblyScoringRecord",
        "TextureMatchRecord",
        "OrientMatchRecord",
        "OrientMatchEntry",
        "WindowOpRecord",
    ],
    # Ввод/вывод и конфигурация
    "io": [
        "load_image_dir",
        "ConfigSpec",
        "validate_section",
        "normalize_array",
    ],
    # Имитация отжига
    "annealing": [
        "linear_schedule",
        "AnnealingScoreEntry",
    ],
}


# ─── Реестр утилит ───────────────────────────────────────────────────────────

def build_util_registry() -> Dict[str, Any]:
    """
    Строит словарь {util_name: callable_or_class}.

    Все import-ы выполняются внутри try/except, поэтому недоступные
    зависимости не останавливают инициализацию.

    Returns:
        Словарь зарегистрированных утилит.
    """
    registry: Dict[str, Any] = {}

    # ── CORE ───────────────────────────────────────────────────────────────────

    try:
        from .logger import get_logger
        registry["get_logger"] = get_logger
    except Exception:
        pass

    try:
        from .profiler import PipelineProfiler
        registry["PipelineProfiler"] = PipelineProfiler
    except Exception:
        pass

    try:
        from .event_bus import make_event_bus
        registry["make_event_bus"] = make_event_bus
    except Exception:
        pass

    try:
        from .event_log import make_event_log
        registry["make_event_log"] = make_event_log
    except Exception:
        pass

    try:
        from .metric_tracker import make_tracker as make_metric_tracker
        registry["make_metric_tracker"] = make_metric_tracker
    except Exception:
        pass

    try:
        from .batch_processor import process_items
        registry["process_items"] = process_items
    except Exception:
        pass

    try:
        from .pipeline_runner import run_pipeline
        registry["run_pipeline"] = run_pipeline
    except Exception:
        pass

    try:
        from .progress_tracker import make_tracker as make_progress_tracker
        registry["make_progress_tracker"] = make_progress_tracker
    except Exception:
        pass

    try:
        from .result_cache import ResultCache
        registry["ResultCache"] = ResultCache
    except Exception:
        pass

    try:
        from .cache import DescriptorCache
        registry["DescriptorCache"] = DescriptorCache
    except Exception:
        pass

    try:
        from .cache_manager import LRUCache
        registry["LRUCache"] = LRUCache
    except Exception:
        pass

    try:
        from .tracker_utils import TrackerConfig
        registry["TrackerConfig"] = TrackerConfig
    except Exception:
        pass

    # ── GEOMETRY ───────────────────────────────────────────────────────────────

    try:
        from .geometry import rotation_matrix_2d
        registry["rotation_matrix_2d"] = rotation_matrix_2d
    except Exception:
        pass

    try:
        from .geometry_utils import BoundingBox
        registry["BoundingBox"] = BoundingBox
    except Exception:
        pass

    try:
        from .bbox_utils import BBox
        registry["BBox"] = BBox
    except Exception:
        pass

    try:
        from .polygon_utils import polygon_area
        registry["polygon_area"] = polygon_area
    except Exception:
        pass

    try:
        from .polygon_ops_utils import PolygonOpsConfig
        registry["PolygonOpsConfig"] = PolygonOpsConfig
    except Exception:
        pass

    try:
        from .spatial_index import build_spatial_index
        registry["build_spatial_index"] = build_spatial_index
    except Exception:
        pass

    try:
        from .alignment_utils import AlignmentConfig
        registry["AlignmentConfig"] = AlignmentConfig
    except Exception:
        pass

    try:
        from .icp_utils import centroid as icp_centroid
        registry["icp_centroid"] = icp_centroid
    except Exception:
        pass

    try:
        from .transform_utils import rotate_image
        registry["rotate_image"] = rotate_image
    except Exception:
        pass

    try:
        from .rotation_utils import rotate_image_angle
        registry["rotate_image_angle"] = rotate_image_angle
    except Exception:
        pass

    try:
        from .image_transform_utils import ImageTransformConfig
        registry["ImageTransformConfig"] = ImageTransformConfig
    except Exception:
        pass

    # ── IMAGE ──────────────────────────────────────────────────────────────────

    try:
        from .image_io import load_image
        registry["load_image"] = load_image
    except Exception:
        pass

    try:
        from .image_stats import compute_image_stats
        registry["compute_image_stats"] = compute_image_stats
    except Exception:
        pass

    try:
        from .patch_extractor import PatchSet
        registry["PatchSet"] = PatchSet
    except Exception:
        pass

    try:
        from .patch_utils import extract_patch
        registry["extract_patch"] = extract_patch
    except Exception:
        pass

    try:
        from .blend_utils import alpha_blend
        registry["alpha_blend"] = alpha_blend
    except Exception:
        pass

    try:
        from .render_utils import CanvasConfig
        registry["CanvasConfig"] = CanvasConfig
    except Exception:
        pass

    try:
        from .morph_utils import MorphConfig
        registry["MorphConfig"] = MorphConfig
    except Exception:
        pass

    try:
        from .mask_utils import create_alpha_mask
        registry["create_alpha_mask"] = create_alpha_mask
    except Exception:
        pass

    try:
        from .tile_utils import TileConfig
        registry["TileConfig"] = TileConfig
    except Exception:
        pass

    try:
        from .threshold_utils import apply_threshold
        registry["apply_threshold"] = apply_threshold
    except Exception:
        pass

    try:
        from .visualizer import VisConfig
        registry["VisConfig"] = VisConfig
    except Exception:
        pass

    try:
        from .image_cluster_utils import ImageStatsAnalysisConfig
        registry["ImageStatsAnalysisConfig"] = ImageStatsAnalysisConfig
    except Exception:
        pass

    try:
        from .image_pipeline_utils import FrequencyMatchSummary
        registry["FrequencyMatchSummary"] = FrequencyMatchSummary
    except Exception:
        pass

    # ── SIGNAL ─────────────────────────────────────────────────────────────────

    try:
        from .signal_utils import smooth_signal
        registry["smooth_signal"] = smooth_signal
    except Exception:
        pass

    try:
        from .smoothing_utils import moving_average
        registry["moving_average"] = moving_average
    except Exception:
        pass

    try:
        from .window_utils import apply_window_function
        registry["apply_window_function"] = apply_window_function
    except Exception:
        pass

    try:
        from .interpolation_utils import lerp
        registry["lerp"] = lerp
    except Exception:
        pass

    try:
        from .frequency_utils import FrequencyConfig
        registry["FrequencyConfig"] = FrequencyConfig
    except Exception:
        pass

    try:
        from .gradient_utils import compute_gradient_magnitude
        registry["compute_gradient_magnitude"] = compute_gradient_magnitude
    except Exception:
        pass

    # ── METRICS ────────────────────────────────────────────────────────────────

    try:
        from .metrics import ReconstructionMetrics
        registry["ReconstructionMetrics"] = ReconstructionMetrics
    except Exception:
        pass

    try:
        from .stats_utils import describe
        registry["describe_stats"] = describe
    except Exception:
        pass

    try:
        from .score_aggregator import AggregationResult
        registry["AggregationResult"] = AggregationResult
    except Exception:
        pass

    try:
        from .distance_utils import euclidean_distance
        registry["euclidean_distance"] = euclidean_distance
    except Exception:
        pass

    try:
        from .distance_matrix import euclidean_distance_matrix
        registry["euclidean_distance_matrix"] = euclidean_distance_matrix
    except Exception:
        pass

    try:
        from .clustering_utils import kmeans_cluster
        registry["kmeans_cluster"] = kmeans_cluster
    except Exception:
        pass

    try:
        from .normalization_utils import l2_normalize
        registry["l2_normalize"] = l2_normalize
    except Exception:
        pass

    try:
        from .feature_selector import variance_selection
        registry["variance_selection"] = variance_selection
    except Exception:
        pass

    try:
        from .score_matrix_utils import ScoreMatrixConfig
        registry["ScoreMatrixConfig"] = ScoreMatrixConfig
    except Exception:
        pass

    try:
        from .score_norm_utils import ScoreNormConfig
        registry["ScoreNormConfig"] = ScoreNormConfig
    except Exception:
        pass

    try:
        from .noise_stats_utils import NoiseStatsConfig
        registry["NoiseStatsConfig"] = NoiseStatsConfig
    except Exception:
        pass

    try:
        from .normalize_noise_utils import NormResultConfig
        registry["NormResultConfig"] = NormResultConfig
    except Exception:
        pass

    # ── GRAPH ──────────────────────────────────────────────────────────────────

    try:
        from .graph_utils import build_graph
        registry["build_graph"] = build_graph
    except Exception:
        pass

    try:
        from .path_plan_utils import PathPlanConfig
        registry["PathPlanConfig"] = PathPlanConfig
    except Exception:
        pass

    try:
        from .sparse_utils import to_sparse_entries
        registry["to_sparse_entries"] = to_sparse_entries
    except Exception:
        pass

    try:
        from .topology_utils import TopologyConfig
        registry["TopologyConfig"] = TopologyConfig
    except Exception:
        pass

    # ── CONTOUR ────────────────────────────────────────────────────────────────

    try:
        from .contour_utils import simplify_contour
        registry["simplify_contour"] = simplify_contour
    except Exception:
        pass

    try:
        from .contour_profile_utils import sample_profile_along_contour
        registry["sample_profile_along_contour"] = sample_profile_along_contour
    except Exception:
        pass

    try:
        from .contour_sampler import SampledContour
        registry["SampledContour"] = SampledContour
    except Exception:
        pass

    try:
        from .shape_match_utils import ShapeMatchConfig
        registry["ShapeMatchConfig"] = ShapeMatchConfig
    except Exception:
        pass

    try:
        from .segment_utils import SegmentConfig
        registry["SegmentConfig"] = SegmentConfig
    except Exception:
        pass

    try:
        from .curvature_utils import compute_curvature
        registry["compute_curvature"] = compute_curvature
    except Exception:
        pass

    try:
        from .curve_metrics import curve_l2
        registry["curve_l2"] = curve_l2
    except Exception:
        pass

    # ── COLOR ──────────────────────────────────────────────────────────────────

    try:
        from .color_utils import to_gray
        registry["to_gray"] = to_gray
    except Exception:
        pass

    try:
        from .histogram_utils import compute_1d_histogram
        registry["compute_1d_histogram"] = compute_1d_histogram
    except Exception:
        pass

    try:
        from .color_hist_utils import ColorHistConfig
        registry["ColorHistConfig"] = ColorHistConfig
    except Exception:
        pass

    try:
        from .freq_metric_utils import BandEnergyRecord
        registry["BandEnergyRecord"] = BandEnergyRecord
    except Exception:
        pass

    # ── KEYPOINT ───────────────────────────────────────────────────────────────

    try:
        from .keypoint_utils import detect_keypoints
        registry["detect_keypoints"] = detect_keypoints
    except Exception:
        pass

    try:
        from .descriptor_utils import DescriptorConfig
        registry["DescriptorConfig"] = DescriptorConfig
    except Exception:
        pass

    # ── FRAGMENT ───────────────────────────────────────────────────────────────

    try:
        from .fragment_stats import FragmentMetrics
        registry["FragmentMetrics"] = FragmentMetrics
    except Exception:
        pass

    try:
        from .fragment_filter_utils import FragmentFilterConfig
        registry["FragmentFilterConfig"] = FragmentFilterConfig
    except Exception:
        pass

    try:
        from .edge_profile_utils import EdgeProfileConfig
        registry["EdgeProfileConfig"] = EdgeProfileConfig
    except Exception:
        pass

    try:
        from .edge_scorer import EdgeScoreConfig
        registry["EdgeScoreConfig"] = EdgeScoreConfig
    except Exception:
        pass

    # ── SCORING ────────────────────────────────────────────────────────────────

    try:
        from .pair_score_utils import PairScoreConfig
        registry["PairScoreConfig"] = PairScoreConfig
    except Exception:
        pass

    try:
        from .patch_score_utils import PatchScoreConfig
        registry["PatchScoreConfig"] = PatchScoreConfig
    except Exception:
        pass

    try:
        from .region_score_utils import RegionScoreConfig
        registry["RegionScoreConfig"] = RegionScoreConfig
    except Exception:
        pass

    try:
        from .rotation_score_utils import RotationScoreConfig
        registry["RotationScoreConfig"] = RotationScoreConfig
    except Exception:
        pass

    try:
        from .score_seam_utils import NormalizationRecord as SeamNormalizationRecord
        registry["SeamNormalizationRecord"] = SeamNormalizationRecord
    except Exception:
        pass

    try:
        from .voting_utils import cast_pair_votes
        registry["cast_pair_votes"] = cast_pair_votes
    except Exception:
        pass

    try:
        from .sequence_utils import rank_sequence
        registry["rank_sequence"] = rank_sequence
    except Exception:
        pass

    try:
        from .match_rank_utils import RankingConfig
        registry["RankingConfig"] = RankingConfig
    except Exception:
        pass

    try:
        from .quality_score_utils import QualityScoreConfig
        registry["QualityScoreConfig"] = QualityScoreConfig
    except Exception:
        pass

    try:
        from .consensus_score_utils import ConsensusScoreConfig
        registry["ConsensusScoreConfig"] = ConsensusScoreConfig
    except Exception:
        pass

    try:
        from .annealing_score_utils import AnnealingScoreConfig
        registry["AnnealingScoreConfig"] = AnnealingScoreConfig
    except Exception:
        pass

    try:
        from .overlap_score_utils import OverlapScoreConfig
        registry["OverlapScoreConfig"] = OverlapScoreConfig
    except Exception:
        pass

    try:
        from .seq_gap_utils import SequenceScoreConfig
        registry["SequenceScoreConfig"] = SequenceScoreConfig
    except Exception:
        pass

    try:
        from .scoring_pipeline_utils import PipelineReport as ScoringPipelineReport
        registry["ScoringPipelineReport"] = ScoringPipelineReport
    except Exception:
        pass

    try:
        from .sampling_utils import SamplingConfig
        registry["SamplingConfig"] = SamplingConfig
    except Exception:
        pass

    # ── ASSEMBLY ───────────────────────────────────────────────────────────────

    try:
        from .assembly_score_utils import AssemblySummary
        registry["AssemblySummary"] = AssemblySummary
    except Exception:
        pass

    try:
        from .canvas_build_utils import CanvasBuildConfig
        registry["CanvasBuildConfig"] = CanvasBuildConfig
    except Exception:
        pass

    try:
        from .placement_score_utils import PlacementScoreConfig
        registry["PlacementScoreConfig"] = PlacementScoreConfig
    except Exception:
        pass

    try:
        from .placement_metrics_utils import PlacementMetricsConfig
        registry["PlacementMetricsConfig"] = PlacementMetricsConfig
    except Exception:
        pass

    try:
        from .position_tracking_utils import PositionQualityRecord
        registry["PositionQualityRecord"] = PositionQualityRecord
    except Exception:
        pass

    try:
        from .rank_result_utils import RankResultConfig
        registry["RankResultConfig"] = RankResultConfig
    except Exception:
        pass

    try:
        from .candidate_rank_utils import CandidateRankConfig
        registry["CandidateRankConfig"] = CandidateRankConfig
    except Exception:
        pass

    try:
        from .ranking_layout_utils import GlobalRankingConfig
        registry["GlobalRankingConfig"] = GlobalRankingConfig
    except Exception:
        pass

    try:
        from .ranking_validation_utils import RankingRunRecord
        registry["RankingRunRecord"] = RankingRunRecord
    except Exception:
        pass

    # ── RECORDS ────────────────────────────────────────────────────────────────

    try:
        from .assembly_records import CollisionRecord
        registry["CollisionRecord"] = CollisionRecord
    except Exception:
        pass

    try:
        from .assembly_config_utils import AssemblyStateRecord
        registry["AssemblyStateRecord"] = AssemblyStateRecord
    except Exception:
        pass

    try:
        from .classification_freq_records import FragmentClassifyRecord
        registry["FragmentClassifyRecord"] = FragmentClassifyRecord
    except Exception:
        pass

    try:
        from .color_edge_export_utils import ColorMatchAnalysisConfig
        registry["ColorMatchAnalysisConfig"] = ColorMatchAnalysisConfig
    except Exception:
        pass

    try:
        from .contour_contrast_records import ContourProcessRecord
        registry["ContourProcessRecord"] = ContourProcessRecord
    except Exception:
        pass

    try:
        from .contour_curvature_records import AnnealingRunRecord
        registry["AnnealingRunRecord"] = AnnealingRunRecord
    except Exception:
        pass

    try:
        from .descriptor_edge_records import DescriptorCombineRecord
        registry["DescriptorCombineRecord"] = DescriptorCombineRecord
    except Exception:
        pass

    try:
        from .edge_fragment_records import EdgeCompareRecord
        registry["EdgeCompareRecord"] = EdgeCompareRecord
    except Exception:
        pass

    try:
        from .event_affine_utils import EventRecordEntry
        registry["EventRecordEntry"] = EventRecordEntry
    except Exception:
        pass

    try:
        from .gap_geometry_records import GapScoringRecord
        registry["GapScoringRecord"] = GapScoringRecord
    except Exception:
        pass

    try:
        from .gradient_graph_records import GradientFlowRecord
        registry["GradientFlowRecord"] = GradientFlowRecord
    except Exception:
        pass

    try:
        from .illum_layout_records import IllumNormRecord
        registry["IllumNormRecord"] = IllumNormRecord
    except Exception:
        pass

    try:
        from .matching_consistency_records import BoundaryMatchRecord
        registry["BoundaryMatchRecord"] = BoundaryMatchRecord
    except Exception:
        pass

    try:
        from .region_seam_records import RegionPairRecord
        registry["RegionPairRecord"] = RegionPairRecord
    except Exception:
        pass

    try:
        from .scorer_state_records import AssemblyScoringRecord
        registry["AssemblyScoringRecord"] = AssemblyScoringRecord
    except Exception:
        pass

    try:
        from .texture_pipeline_utils import TextureMatchRecord
        registry["TextureMatchRecord"] = TextureMatchRecord
    except Exception:
        pass

    try:
        from .orient_topology_utils import OrientMatchRecord
        registry["OrientMatchRecord"] = OrientMatchRecord
    except Exception:
        pass

    try:
        from .orient_skew_utils import OrientMatchEntry
        registry["OrientMatchEntry"] = OrientMatchEntry
    except Exception:
        pass

    try:
        from .window_tile_records import WindowOpRecord
        registry["WindowOpRecord"] = WindowOpRecord
    except Exception:
        pass

    # ── IO / CONFIG ────────────────────────────────────────────────────────────

    try:
        from .io import load_image_dir
        registry["load_image_dir"] = load_image_dir
    except Exception:
        pass

    try:
        from .config_manager import ConfigSpec
        registry["ConfigSpec"] = ConfigSpec
    except Exception:
        pass

    try:
        from .config_utils import validate_section
        registry["validate_section"] = validate_section
    except Exception:
        pass

    try:
        from .array_utils import normalize_array
        registry["normalize_array"] = normalize_array
    except Exception:
        pass

    # ── ANNEALING ──────────────────────────────────────────────────────────────

    try:
        from .annealing_schedule import linear_schedule
        registry["linear_schedule"] = linear_schedule
    except Exception:
        pass

    try:
        from .annealing_score_utils import AnnealingScoreEntry
        registry["AnnealingScoreEntry"] = AnnealingScoreEntry
    except Exception:
        pass

    return registry


# ─── Глобальный реестр ───────────────────────────────────────────────────────

UTIL_REGISTRY: Dict[str, Any] = {}


def _ensure_registry() -> None:
    global UTIL_REGISTRY
    if not UTIL_REGISTRY:
        UTIL_REGISTRY = build_util_registry()


def list_utils(category: Optional[str] = None) -> List[str]:
    """
    Список всех зарегистрированных утилит.

    Args:
        category: Фильтр по категории ('core', 'geometry', 'image', ...).
                  None → все категории.

    Returns:
        Отсортированный список имён утилит.
    """
    _ensure_registry()
    if category is not None:
        names = set(UTIL_CATEGORIES.get(category, []))
        return sorted(n for n in UTIL_REGISTRY if n in names)
    return sorted(UTIL_REGISTRY.keys())


def get_util(name: str) -> Optional[Any]:
    """
    Возвращает callable/class для утилиты по имени.

    Args:
        name: Имя утилиты (из UTIL_CATEGORIES).

    Returns:
        Callable/class или None если утилита недоступна.
    """
    _ensure_registry()
    obj = UTIL_REGISTRY.get(name)
    if obj is None:
        logger.debug("util %r not available", name)
    return obj


def get_util_category(name: str) -> Optional[str]:
    """Возвращает категорию утилиты или None."""
    for cat, names in UTIL_CATEGORIES.items():
        if name in names:
            return cat
    return None
