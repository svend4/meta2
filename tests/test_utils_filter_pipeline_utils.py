"""Tests for puzzle_reconstruction.utils.filter_pipeline_utils"""
import pytest
import numpy as np

from puzzle_reconstruction.utils.filter_pipeline_utils import (
    FilterStepConfig,
    FilterStepResult,
    FilterPipelineSummary,
    make_filter_step,
    steps_from_log,
    summarise_pipeline,
    filter_effective_steps,
    filter_by_removal_rate,
    most_aggressive_step,
    least_aggressive_step,
    pipeline_stats,
    compare_pipelines,
    batch_summarise_pipelines,
)


# ── FilterStepConfig ────────────────────────────────────────────────────────

def test_filter_step_config_defaults():
    cfg = FilterStepConfig()
    assert cfg.name == "threshold"
    assert cfg.threshold == 0.5
    assert cfg.top_k == 0
    assert cfg.deduplicate is False


def test_filter_step_config_custom():
    cfg = FilterStepConfig(name="myfilter", threshold=0.3, top_k=10, deduplicate=True)
    assert cfg.name == "myfilter"
    assert cfg.threshold == 0.3
    assert cfg.top_k == 10
    assert cfg.deduplicate is True


def test_filter_step_config_empty_name_raises():
    with pytest.raises(ValueError):
        FilterStepConfig(name="")


def test_filter_step_config_threshold_out_of_range_raises():
    with pytest.raises(ValueError):
        FilterStepConfig(threshold=1.5)
    with pytest.raises(ValueError):
        FilterStepConfig(threshold=-0.1)


def test_filter_step_config_negative_top_k_raises():
    with pytest.raises(ValueError):
        FilterStepConfig(top_k=-1)


def test_filter_step_config_boundary_thresholds():
    cfg_low = FilterStepConfig(threshold=0.0)
    cfg_high = FilterStepConfig(threshold=1.0)
    assert cfg_low.threshold == 0.0
    assert cfg_high.threshold == 1.0


# ── FilterStepResult ────────────────────────────────────────────────────────

def test_make_filter_step_basic():
    step = make_filter_step("dedupe", n_input=100, n_output=80)
    assert step.step_name == "dedupe"
    assert step.n_input == 100
    assert step.n_output == 80
    assert step.n_removed == 20


def test_filter_step_result_removal_rate():
    step = make_filter_step("s", n_input=50, n_output=25)
    assert step.removal_rate == pytest.approx(0.5)


def test_filter_step_result_removal_rate_zero_input():
    step = make_filter_step("s", n_input=0, n_output=0)
    assert step.removal_rate == 0.0


def test_filter_step_result_repr_contains_name():
    step = make_filter_step("myStep", n_input=10, n_output=5)
    assert "myStep" in repr(step)


def test_make_filter_step_with_meta():
    step = make_filter_step("s", n_input=10, n_output=8, meta={"key": "val"})
    assert step.meta == {"key": "val"}


def test_make_filter_step_meta_default_empty():
    step = make_filter_step("s", n_input=5, n_output=5)
    assert step.meta == {}


# ── steps_from_log ──────────────────────────────────────────────────────────

def test_steps_from_log_basic():
    log = [{"step_name": "a", "n_input": 100, "n_output": 80}]
    steps = steps_from_log(log)
    assert len(steps) == 1
    assert steps[0].step_name == "a"
    assert steps[0].n_removed == 20


def test_steps_from_log_extra_keys_become_meta():
    log = [{"step_name": "x", "n_input": 10, "n_output": 5, "extra": "data"}]
    steps = steps_from_log(log)
    assert steps[0].meta.get("extra") == "data"


def test_steps_from_log_empty():
    assert steps_from_log([]) == []


def test_steps_from_log_missing_step_name_defaults_unknown():
    log = [{"n_input": 5, "n_output": 3}]
    steps = steps_from_log(log)
    assert steps[0].step_name == "unknown"


# ── summarise_pipeline ──────────────────────────────────────────────────────

def test_summarise_pipeline_empty():
    summary = summarise_pipeline([])
    assert summary.n_initial == 0
    assert summary.n_final == 0
    assert summary.total_removed == 0
    assert summary.overall_removal_rate == 0.0


def test_summarise_pipeline_single_step():
    step = make_filter_step("s", 100, 60)
    summary = summarise_pipeline([step])
    assert summary.n_initial == 100
    assert summary.n_final == 60
    assert summary.total_removed == 40
    assert summary.overall_removal_rate == pytest.approx(0.4)


def test_summarise_pipeline_two_steps():
    s1 = make_filter_step("a", 100, 70)
    s2 = make_filter_step("b", 70, 50)
    summary = summarise_pipeline([s1, s2])
    assert summary.n_initial == 100
    assert summary.n_final == 50
    assert summary.total_removed == 50
    assert summary.overall_removal_rate == pytest.approx(0.5)


def test_summarise_pipeline_repr_contains_steps():
    s = make_filter_step("x", 10, 5)
    summary = summarise_pipeline([s])
    assert "steps=1" in repr(summary)


# ── filter_effective_steps ──────────────────────────────────────────────────

def test_filter_effective_steps_removes_zero_removal():
    s1 = make_filter_step("no_remove", 10, 10)
    s2 = make_filter_step("remove", 10, 5)
    effective = filter_effective_steps([s1, s2])
    assert len(effective) == 1
    assert effective[0].step_name == "remove"


def test_filter_effective_steps_all_effective():
    steps = [make_filter_step(f"s{i}", 100, 50) for i in range(3)]
    assert len(filter_effective_steps(steps)) == 3


def test_filter_effective_steps_empty_input():
    assert filter_effective_steps([]) == []


# ── filter_by_removal_rate ──────────────────────────────────────────────────

def test_filter_by_removal_rate_basic():
    s1 = make_filter_step("high", 100, 10)   # 90%
    s2 = make_filter_step("low", 100, 90)    # 10%
    result = filter_by_removal_rate([s1, s2], min_rate=0.5)
    assert len(result) == 1
    assert result[0].step_name == "high"


def test_filter_by_removal_rate_zero_includes_all():
    steps = [make_filter_step(f"s{i}", 10, 10) for i in range(3)]
    assert len(filter_by_removal_rate(steps, min_rate=0.0)) == 3


# ── most_aggressive_step / least_aggressive_step ────────────────────────────

def test_most_aggressive_step():
    s1 = make_filter_step("big", 100, 10)   # removes 90
    s2 = make_filter_step("small", 100, 95)  # removes 5
    result = most_aggressive_step([s1, s2])
    assert result.step_name == "big"


def test_most_aggressive_step_empty():
    assert most_aggressive_step([]) is None


def test_least_aggressive_step():
    s1 = make_filter_step("big", 100, 10)
    s2 = make_filter_step("small", 100, 95)
    result = least_aggressive_step([s1, s2])
    assert result.step_name == "small"


def test_least_aggressive_step_empty():
    assert least_aggressive_step([]) is None


# ── pipeline_stats ──────────────────────────────────────────────────────────

def test_pipeline_stats_empty():
    stats = pipeline_stats([])
    assert stats["n_steps"] == 0
    assert stats["total_removed"] == 0
    assert stats["mean_removal_rate"] == 0.0


def test_pipeline_stats_basic():
    s1 = make_filter_step("a", 100, 50)
    s2 = make_filter_step("b", 50, 25)
    stats = pipeline_stats([s1, s2])
    assert stats["n_steps"] == 2
    assert stats["total_removed"] == 75
    assert 0.0 <= stats["mean_removal_rate"] <= 1.0


def test_pipeline_stats_max_min_rate():
    s1 = make_filter_step("a", 100, 0)   # rate 1.0
    s2 = make_filter_step("b", 100, 100) # rate 0.0
    stats = pipeline_stats([s1, s2])
    assert stats["max_removal_rate"] == pytest.approx(1.0)
    assert stats["min_removal_rate"] == pytest.approx(0.0)


# ── compare_pipelines ───────────────────────────────────────────────────────

def test_compare_pipelines_delta():
    s1 = make_filter_step("a", 100, 50)
    s2 = make_filter_step("b", 100, 30)
    sum_a = summarise_pipeline([s1])
    sum_b = summarise_pipeline([s2])
    comp = compare_pipelines(sum_a, sum_b)
    assert "n_final_delta" in comp
    assert comp["n_final_delta"] == 50 - 30


# ── batch_summarise_pipelines ───────────────────────────────────────────────

def test_batch_summarise_pipelines():
    log_a = [{"step_name": "s1", "n_input": 100, "n_output": 60}]
    log_b = [{"step_name": "s2", "n_input": 200, "n_output": 100}]
    summaries = batch_summarise_pipelines([log_a, log_b])
    assert len(summaries) == 2
    assert summaries[0].n_initial == 100
    assert summaries[1].n_initial == 200


def test_batch_summarise_pipelines_empty():
    summaries = batch_summarise_pipelines([])
    assert summaries == []
