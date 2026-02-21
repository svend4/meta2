"""Модули оценки качества сборки пазла.

Доступные модули:
    consistency_checker — проверка согласованности сборки (ConsistencyIssue,
                          ConsistencyReport, check_unique_ids, check_all_present,
                          check_canvas_bounds, check_score_threshold,
                          check_gap_uniformity, run_consistency_check,
                          batch_consistency_check)
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

__all__ = [
    "ConsistencyIssue",
    "ConsistencyReport",
    "check_unique_ids",
    "check_all_present",
    "check_canvas_bounds",
    "check_score_threshold",
    "check_gap_uniformity",
    "run_consistency_check",
    "batch_consistency_check",
]
