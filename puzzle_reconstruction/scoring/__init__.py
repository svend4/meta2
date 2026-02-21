"""Модули оценки качества сборки пазла.

Доступные модули:
    consistency_checker — проверка согласованности сборки (ConsistencyIssue,
                          ConsistencyReport, check_unique_ids, check_all_present,
                          check_canvas_bounds, check_score_threshold,
                          check_gap_uniformity, run_consistency_check,
                          batch_consistency_check)
    boundary_scorer     — оценка совместимости по граничным пикселям (BoundarySide,
                          BoundaryScore, ScoringConfig, intensity_compatibility,
                          gradient_compatibility, color_compatibility,
                          score_boundary, score_matrix, batch_score_boundaries)
    global_ranker       — глобальное ранжирование фрагментов (RankedPair, RankingConfig,
                          normalize_matrix, aggregate_score_matrices, rank_pairs,
                          top_k_candidates, global_rank, score_vector, batch_global_rank)
"""
from .consistency_checker import (
    ConsistencyIssue,
    ConsistencyReport,
    check_unique_ids,
    check_all_present,
    check_canvas_bounds,
    check_score_threshold,
    check_gap_uniformity,
    run_consistency_check,
    batch_consistency_check,
)
from .boundary_scorer import (
    BoundarySide,
    BoundaryScore,
    ScoringConfig,
    intensity_compatibility,
    gradient_compatibility,
    color_compatibility,
    score_boundary,
    score_matrix,
    batch_score_boundaries,
)
from .global_ranker import (
    RankedPair,
    RankingConfig,
    normalize_matrix,
    aggregate_score_matrices,
    rank_pairs,
    top_k_candidates,
    global_rank,
    score_vector,
    batch_global_rank,
)

__all__ = [
    # Проверка согласованности
    "ConsistencyIssue",
    "ConsistencyReport",
    "check_unique_ids",
    "check_all_present",
    "check_canvas_bounds",
    "check_score_threshold",
    "check_gap_uniformity",
    "run_consistency_check",
    "batch_consistency_check",
    # Оценка граничной совместимости
    "BoundarySide",
    "BoundaryScore",
    "ScoringConfig",
    "intensity_compatibility",
    "gradient_compatibility",
    "color_compatibility",
    "score_boundary",
    "score_matrix",
    "batch_score_boundaries",
    # Глобальное ранжирование
    "RankedPair",
    "RankingConfig",
    "normalize_matrix",
    "aggregate_score_matrices",
    "rank_pairs",
    "top_k_candidates",
    "global_rank",
    "score_vector",
    "batch_global_rank",
]
