"""Tests for puzzle_reconstruction.utils.scoring_pipeline_utils."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.scoring_pipeline_utils import (
    StageResult,
    PipelineReport,
    BoundaryScoreRecord,
    PatchComparisonRecord,
    build_pipeline_report,
    summarize_boundary_scores,
    rank_stage_results,
)

np.random.seed(55)


# ─── StageResult ─────────────────────────────────────────────────────────────

def test_stage_result_creation():
    s = StageResult(stage_name="color", score=0.8, weight=1.0)
    assert s.stage_name == "color"
    assert s.score == pytest.approx(0.8)
    assert s.weight == pytest.approx(1.0)


def test_stage_result_empty_name_raises():
    with pytest.raises(ValueError):
        StageResult(stage_name="", score=0.5)


def test_stage_result_score_out_of_range_raises():
    with pytest.raises(ValueError):
        StageResult(stage_name="test", score=1.5)


def test_stage_result_score_negative_raises():
    with pytest.raises(ValueError):
        StageResult(stage_name="test", score=-0.1)


def test_stage_result_negative_weight_raises():
    with pytest.raises(ValueError):
        StageResult(stage_name="test", score=0.5, weight=-1.0)


def test_stage_result_weighted_score():
    s = StageResult(stage_name="geo", score=0.8, weight=2.0)
    assert s.weighted_score == pytest.approx(1.6)


def test_stage_result_zero_weight():
    s = StageResult(stage_name="geo", score=0.9, weight=0.0)
    assert s.weighted_score == pytest.approx(0.0)


def test_stage_result_boundary_score():
    s = StageResult(stage_name="s", score=0.0)
    assert s.score == pytest.approx(0.0)
    s2 = StageResult(stage_name="s", score=1.0)
    assert s2.score == pytest.approx(1.0)


# ─── PipelineReport ──────────────────────────────────────────────────────────

def test_pipeline_report_empty():
    r = PipelineReport()
    assert r.n_stages == 0
    assert r.total_weight == pytest.approx(0.0)
    assert r.weighted_score == pytest.approx(0.0)


def test_pipeline_report_add_stage():
    r = PipelineReport()
    r.add_stage(StageResult("color", 0.8, 1.0))
    r.add_stage(StageResult("shape", 0.6, 2.0))
    assert r.n_stages == 2


def test_pipeline_report_weighted_score():
    r = PipelineReport()
    r.add_stage(StageResult("a", 1.0, 1.0))
    r.add_stage(StageResult("b", 0.0, 1.0))
    assert r.weighted_score == pytest.approx(0.5)


def test_pipeline_report_total_weight():
    r = PipelineReport()
    r.add_stage(StageResult("a", 0.5, 2.0))
    r.add_stage(StageResult("b", 0.5, 3.0))
    assert r.total_weight == pytest.approx(5.0)


def test_pipeline_report_stage_by_name_found():
    r = PipelineReport()
    r.add_stage(StageResult("color", 0.8, 1.0))
    r.add_stage(StageResult("shape", 0.6, 1.0))
    s = r.stage_by_name("color")
    assert s is not None
    assert s.score == pytest.approx(0.8)


def test_pipeline_report_stage_by_name_not_found():
    r = PipelineReport()
    r.add_stage(StageResult("color", 0.8, 1.0))
    s = r.stage_by_name("nonexistent")
    assert s is None


def test_pipeline_report_to_dict():
    r = PipelineReport()
    r.add_stage(StageResult("color", 0.7, 1.0))
    d = r.to_dict()
    assert "n_stages" in d
    assert "weighted_score" in d
    assert "stages" in d
    assert d["n_stages"] == 1


# ─── BoundaryScoreRecord ─────────────────────────────────────────────────────

def test_boundary_score_record_creation():
    b = BoundaryScoreRecord(idx1=0, idx2=1, hausdorff_score=0.8,
                             chamfer_score=0.7, frechet_score=0.9,
                             total_score=0.8)
    assert b.idx1 == 0
    assert b.total_score == pytest.approx(0.8)


def test_boundary_score_record_negative_idx():
    with pytest.raises(ValueError):
        BoundaryScoreRecord(idx1=-1, idx2=0, hausdorff_score=0.5,
                             chamfer_score=0.5, frechet_score=0.5,
                             total_score=0.5)


def test_boundary_score_record_score_out_of_range():
    with pytest.raises(ValueError):
        BoundaryScoreRecord(idx1=0, idx2=1, hausdorff_score=1.5,
                             chamfer_score=0.5, frechet_score=0.5,
                             total_score=0.5)


def test_boundary_score_record_boundary_values():
    b = BoundaryScoreRecord(idx1=0, idx2=1, hausdorff_score=0.0,
                             chamfer_score=1.0, frechet_score=0.0,
                             total_score=1.0)
    assert b.hausdorff_score == pytest.approx(0.0)
    assert b.total_score == pytest.approx(1.0)


# ─── PatchComparisonRecord ───────────────────────────────────────────────────

def test_patch_comparison_record_creation():
    p = PatchComparisonRecord(row=1, col=2, method="ncc", value=0.75)
    assert p.row == 1
    assert p.col == 2
    assert p.method == "ncc"
    assert p.value == pytest.approx(0.75)


def test_patch_comparison_record_empty_method():
    with pytest.raises(ValueError):
        PatchComparisonRecord(row=0, col=0, method="", value=0.5)


def test_patch_comparison_record_negative_row():
    with pytest.raises(ValueError):
        PatchComparisonRecord(row=-1, col=0, method="ncc", value=0.5)


def test_patch_comparison_record_negative_col():
    with pytest.raises(ValueError):
        PatchComparisonRecord(row=0, col=-1, method="ncc", value=0.5)


# ─── build_pipeline_report ───────────────────────────────────────────────────

def test_build_pipeline_report_basic():
    stages = [
        StageResult("color", 0.8, 1.0),
        StageResult("shape", 0.6, 2.0),
    ]
    r = build_pipeline_report(stages)
    assert r.n_stages == 2


def test_build_pipeline_report_empty():
    r = build_pipeline_report([])
    assert r.n_stages == 0


def test_build_pipeline_report_weighted_score():
    stages = [
        StageResult("a", 0.8, 1.0),
        StageResult("b", 0.4, 1.0),
    ]
    r = build_pipeline_report(stages)
    assert r.weighted_score == pytest.approx(0.6)


# ─── summarize_boundary_scores ───────────────────────────────────────────────

def test_summarize_boundary_scores_basic():
    records = [
        BoundaryScoreRecord(0, 1, 0.8, 0.7, 0.9, 0.8),
        BoundaryScoreRecord(1, 2, 0.6, 0.5, 0.7, 0.6),
    ]
    result = summarize_boundary_scores(records)
    assert result["n_pairs"] == 2
    assert result["mean_total"] == pytest.approx(0.7)
    assert result["max_total"] == pytest.approx(0.8)


def test_summarize_boundary_scores_empty():
    result = summarize_boundary_scores([])
    assert result["n_pairs"] == 0
    assert result["mean_total"] == pytest.approx(0.0)


def test_summarize_boundary_scores_single():
    records = [BoundaryScoreRecord(0, 1, 0.5, 0.5, 0.5, 0.9)]
    result = summarize_boundary_scores(records)
    assert result["max_total"] == pytest.approx(0.9)


# ─── rank_stage_results ──────────────────────────────────────────────────────

def test_rank_stage_results_order():
    stages = [
        StageResult("a", 0.3, 1.0),
        StageResult("b", 0.9, 1.0),
        StageResult("c", 0.6, 1.0),
    ]
    ranked = rank_stage_results(stages)
    assert ranked[0][0] == 1  # rank 1
    assert ranked[0][1].stage_name == "b"  # highest score


def test_rank_stage_results_ranks():
    stages = [
        StageResult("a", 0.5, 1.0),
        StageResult("b", 0.8, 1.0),
    ]
    ranked = rank_stage_results(stages)
    ranks = [r[0] for r in ranked]
    assert ranks == [1, 2]


def test_rank_stage_results_empty():
    ranked = rank_stage_results([])
    assert ranked == []


def test_rank_stage_results_single():
    stages = [StageResult("only", 0.7, 1.0)]
    ranked = rank_stage_results(stages)
    assert len(ranked) == 1
    assert ranked[0][0] == 1
