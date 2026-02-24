"""Extra tests for puzzle_reconstruction/utils/scoring_pipeline_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.scoring_pipeline_utils import (
    StageResult,
    PipelineReport,
    BoundaryScoreRecord,
    PatchComparisonRecord,
    build_pipeline_report,
    summarize_boundary_scores,
    rank_stage_results,
)


# ─── StageResult ──────────────────────────────────────────────────────────────

class TestStageResultExtra:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            StageResult(stage_name="", score=0.5)

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            StageResult(stage_name="test", score=1.5)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            StageResult(stage_name="test", score=0.5, weight=-1.0)

    def test_weighted_score(self):
        r = StageResult(stage_name="a", score=0.6, weight=2.0)
        assert r.weighted_score == pytest.approx(1.2)

    def test_default_weight(self):
        r = StageResult(stage_name="a", score=0.5)
        assert r.weight == pytest.approx(1.0)


# ─── PipelineReport ──────────────────────────────────────────────────────────

class TestPipelineReportExtra:
    def test_empty_report(self):
        p = PipelineReport()
        assert p.n_stages == 0 and p.weighted_score == pytest.approx(0.0)

    def test_add_stage(self):
        p = PipelineReport()
        p.add_stage(StageResult(stage_name="a", score=0.5))
        assert p.n_stages == 1

    def test_weighted_score_uniform(self):
        p = PipelineReport(stages=[
            StageResult(stage_name="a", score=0.4),
            StageResult(stage_name="b", score=0.8),
        ])
        assert p.weighted_score == pytest.approx(0.6)

    def test_weighted_score_different_weights(self):
        p = PipelineReport(stages=[
            StageResult(stage_name="a", score=1.0, weight=3.0),
            StageResult(stage_name="b", score=0.0, weight=1.0),
        ])
        assert p.weighted_score == pytest.approx(0.75)

    def test_stage_by_name_found(self):
        s = StageResult(stage_name="target", score=0.7)
        p = PipelineReport(stages=[s])
        assert p.stage_by_name("target") is s

    def test_stage_by_name_missing(self):
        p = PipelineReport()
        assert p.stage_by_name("missing") is None

    def test_to_dict_keys(self):
        p = PipelineReport(stages=[StageResult(stage_name="a", score=0.5)])
        d = p.to_dict()
        assert "n_stages" in d and "weighted_score" in d and "stages" in d

    def test_total_weight(self):
        p = PipelineReport(stages=[
            StageResult(stage_name="a", score=0.5, weight=2.0),
            StageResult(stage_name="b", score=0.3, weight=3.0),
        ])
        assert p.total_weight == pytest.approx(5.0)


# ─── BoundaryScoreRecord ─────────────────────────────────────────────────────

class TestBoundaryScoreRecordExtra:
    def test_negative_idx_raises(self):
        with pytest.raises(ValueError):
            BoundaryScoreRecord(idx1=-1, idx2=0, hausdorff_score=0.5,
                                 chamfer_score=0.5, frechet_score=0.5,
                                 total_score=0.5)

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            BoundaryScoreRecord(idx1=0, idx2=1, hausdorff_score=1.5,
                                 chamfer_score=0.5, frechet_score=0.5,
                                 total_score=0.5)

    def test_fields_stored(self):
        r = BoundaryScoreRecord(idx1=0, idx2=1, hausdorff_score=0.8,
                                 chamfer_score=0.7, frechet_score=0.6,
                                 total_score=0.5)
        assert r.hausdorff_score == pytest.approx(0.8)


# ─── PatchComparisonRecord ────────────────────────────────────────────────────

class TestPatchComparisonRecordExtra:
    def test_empty_method_raises(self):
        with pytest.raises(ValueError):
            PatchComparisonRecord(row=0, col=0, method="", value=0.5)

    def test_negative_row_raises(self):
        with pytest.raises(ValueError):
            PatchComparisonRecord(row=-1, col=0, method="ncc", value=0.5)

    def test_fields_stored(self):
        r = PatchComparisonRecord(row=1, col=2, method="ncc", value=0.9)
        assert r.method == "ncc" and r.value == pytest.approx(0.9)


# ─── build_pipeline_report ────────────────────────────────────────────────────

class TestBuildPipelineReportExtra:
    def test_returns_report(self):
        stages = [StageResult(stage_name="a", score=0.5)]
        r = build_pipeline_report(stages)
        assert isinstance(r, PipelineReport) and r.n_stages == 1

    def test_empty_stages(self):
        r = build_pipeline_report([])
        assert r.n_stages == 0


# ─── summarize_boundary_scores ────────────────────────────────────────────────

class TestSummarizeBoundaryScoresExtra:
    def test_empty_returns_zeros(self):
        s = summarize_boundary_scores([])
        assert s["n_pairs"] == 0

    def test_mean_total(self):
        records = [
            BoundaryScoreRecord(0, 1, 0.5, 0.5, 0.5, 0.4),
            BoundaryScoreRecord(2, 3, 0.7, 0.7, 0.7, 0.8),
        ]
        s = summarize_boundary_scores(records)
        assert s["mean_total"] == pytest.approx(0.6)
        assert s["max_total"] == pytest.approx(0.8)


# ─── rank_stage_results ───────────────────────────────────────────────────────

class TestRankStageResultsExtra:
    def test_descending_order(self):
        stages = [StageResult(stage_name="a", score=0.3),
                  StageResult(stage_name="b", score=0.9)]
        ranked = rank_stage_results(stages)
        assert ranked[0][0] == 1 and ranked[0][1].stage_name == "b"

    def test_empty(self):
        assert rank_stage_results([]) == []
