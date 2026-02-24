"""Extra tests for puzzle_reconstruction/utils/filter_pipeline_utils.py."""
from __future__ import annotations

import pytest

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _step(name="thresh", n_in=100, n_out=80) -> FilterStepResult:
    return make_filter_step(name, n_in, n_out)


def _pipeline(steps=3) -> list:
    n = 100
    result = []
    for i in range(steps):
        out = n - 10
        result.append(_step(f"step_{i}", n_in=n, n_out=out))
        n = out
    return result


# ─── FilterStepConfig ─────────────────────────────────────────────────────────

class TestFilterStepConfigExtra:
    def test_default_name(self):
        assert FilterStepConfig().name == "threshold"

    def test_default_threshold(self):
        assert FilterStepConfig().threshold == pytest.approx(0.5)

    def test_default_top_k(self):
        assert FilterStepConfig().top_k == 0

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            FilterStepConfig(name="")

    def test_threshold_above_one_raises(self):
        with pytest.raises(ValueError):
            FilterStepConfig(threshold=1.1)

    def test_threshold_below_zero_raises(self):
        with pytest.raises(ValueError):
            FilterStepConfig(threshold=-0.1)

    def test_negative_top_k_raises(self):
        with pytest.raises(ValueError):
            FilterStepConfig(top_k=-1)

    def test_custom_values(self):
        cfg = FilterStepConfig(name="nms", threshold=0.3, top_k=10)
        assert cfg.name == "nms"
        assert cfg.threshold == pytest.approx(0.3)
        assert cfg.top_k == 10


# ─── FilterStepResult ─────────────────────────────────────────────────────────

class TestFilterStepResultExtra:
    def test_stores_step_name(self):
        assert _step(name="nms").step_name == "nms"

    def test_stores_n_input(self):
        assert _step(n_in=50).n_input == 50

    def test_stores_n_output(self):
        assert _step(n_out=30).n_output == 30

    def test_n_removed_computed(self):
        s = _step(n_in=100, n_out=60)
        assert s.n_removed == 40

    def test_removal_rate(self):
        s = _step(n_in=100, n_out=75)
        assert s.removal_rate == pytest.approx(0.25)

    def test_removal_rate_zero_input(self):
        s = make_filter_step("x", 0, 0)
        assert s.removal_rate == pytest.approx(0.0)

    def test_repr_contains_name(self):
        r = repr(_step(name="myfilter"))
        assert "myfilter" in r


# ─── FilterPipelineSummary ────────────────────────────────────────────────────

class TestFilterPipelineSummaryExtra:
    def test_n_initial_from_first_step(self):
        steps = _pipeline(3)
        summary = summarise_pipeline(steps)
        assert summary.n_initial == 100

    def test_n_final_from_last_step(self):
        steps = _pipeline(3)
        summary = summarise_pipeline(steps)
        assert summary.n_final == steps[-1].n_output

    def test_total_removed_correct(self):
        steps = _pipeline(2)
        summary = summarise_pipeline(steps)
        assert summary.total_removed == sum(s.n_removed for s in steps)

    def test_repr_contains_steps(self):
        summary = summarise_pipeline(_pipeline(2))
        assert "2" in repr(summary)

    def test_empty_pipeline(self):
        summary = summarise_pipeline([])
        assert summary.n_initial == 0 and summary.n_final == 0


# ─── make_filter_step ─────────────────────────────────────────────────────────

class TestMakeFilterStepExtra:
    def test_returns_result(self):
        assert isinstance(make_filter_step("t", 10, 8), FilterStepResult)

    def test_n_removed_set(self):
        s = make_filter_step("t", 10, 3)
        assert s.n_removed == 7

    def test_meta_stored(self):
        s = make_filter_step("t", 10, 5, meta={"key": "val"})
        assert s.meta.get("key") == "val"


# ─── steps_from_log ───────────────────────────────────────────────────────────

class TestStepsFromLogExtra:
    def test_returns_list(self):
        log = [{"step_name": "a", "n_input": 10, "n_output": 8}]
        assert isinstance(steps_from_log(log), list)

    def test_length_matches(self):
        log = [
            {"step_name": "a", "n_input": 10, "n_output": 8},
            {"step_name": "b", "n_input": 8, "n_output": 5},
        ]
        assert len(steps_from_log(log)) == 2

    def test_empty_log(self):
        assert steps_from_log([]) == []

    def test_values_correct(self):
        log = [{"step_name": "s1", "n_input": 20, "n_output": 15}]
        result = steps_from_log(log)
        assert result[0].step_name == "s1" and result[0].n_removed == 5


# ─── filter_effective_steps ───────────────────────────────────────────────────

class TestFilterEffectiveStepsExtra:
    def test_removes_zero_removal_steps(self):
        steps = [_step(n_in=10, n_out=10), _step(n_in=10, n_out=8)]
        result = filter_effective_steps(steps)
        assert all(s.n_removed > 0 for s in result)

    def test_empty_input(self):
        assert filter_effective_steps([]) == []

    def test_all_effective(self):
        steps = [_step(n_in=10, n_out=8), _step(n_in=8, n_out=5)]
        assert len(filter_effective_steps(steps)) == 2


# ─── filter_by_removal_rate ───────────────────────────────────────────────────

class TestFilterByRemovalRateExtra:
    def test_keeps_high_rate_steps(self):
        steps = [_step(n_in=100, n_out=90), _step(n_in=100, n_out=50)]
        result = filter_by_removal_rate(steps, min_rate=0.4)
        assert all(s.removal_rate >= 0.4 for s in result)

    def test_zero_rate_keeps_all(self):
        steps = _pipeline(3)
        assert len(filter_by_removal_rate(steps, min_rate=0.0)) == 3

    def test_empty_input(self):
        assert filter_by_removal_rate([]) == []


# ─── most/least aggressive ────────────────────────────────────────────────────

class TestAggressiveStepsExtra:
    def test_most_aggressive_max_removed(self):
        steps = [_step(n_in=100, n_out=90), _step(n_in=100, n_out=40)]
        result = most_aggressive_step(steps)
        assert result.n_removed == 60

    def test_least_aggressive_min_removed(self):
        steps = [_step(n_in=100, n_out=90), _step(n_in=100, n_out=40)]
        result = least_aggressive_step(steps)
        assert result.n_removed == 10

    def test_most_empty_is_none(self):
        assert most_aggressive_step([]) is None

    def test_least_empty_is_none(self):
        assert least_aggressive_step([]) is None


# ─── pipeline_stats ───────────────────────────────────────────────────────────

class TestPipelineStatsExtra:
    def test_returns_dict(self):
        assert isinstance(pipeline_stats(_pipeline(2)), dict)

    def test_keys_present(self):
        stats = pipeline_stats(_pipeline(2))
        for k in ("n_steps", "total_removed", "mean_removal_rate",
                  "max_removal_rate", "min_removal_rate"):
            assert k in stats

    def test_empty_returns_zero(self):
        assert pipeline_stats([])["n_steps"] == 0

    def test_n_steps_correct(self):
        assert pipeline_stats(_pipeline(3))["n_steps"] == 3


# ─── compare_pipelines ────────────────────────────────────────────────────────

class TestComparePipelinesExtra:
    def test_returns_dict(self):
        s = summarise_pipeline(_pipeline(2))
        assert isinstance(compare_pipelines(s, s), dict)

    def test_identical_zero_deltas(self):
        s = summarise_pipeline(_pipeline(2))
        d = compare_pipelines(s, s)
        assert d["removal_rate_delta"] == pytest.approx(0.0)
        assert d["n_steps_delta"] == 0

    def test_keys_present(self):
        s = summarise_pipeline(_pipeline(2))
        d = compare_pipelines(s, s)
        for k in ("n_steps_delta", "n_final_delta",
                  "total_removed_delta", "removal_rate_delta"):
            assert k in d


# ─── batch_summarise_pipelines ────────────────────────────────────────────────

class TestBatchSummarisePipelinesExtra:
    def test_returns_list(self):
        logs = [[{"step_name": "a", "n_input": 10, "n_output": 8}]]
        assert isinstance(batch_summarise_pipelines(logs), list)

    def test_length_matches(self):
        logs = [
            [{"step_name": "a", "n_input": 10, "n_output": 8}],
            [{"step_name": "b", "n_input": 20, "n_output": 15}],
        ]
        assert len(batch_summarise_pipelines(logs)) == 2

    def test_each_is_summary(self):
        logs = [[{"step_name": "a", "n_input": 5, "n_output": 3}]]
        for s in batch_summarise_pipelines(logs):
            assert isinstance(s, FilterPipelineSummary)

    def test_empty_logs(self):
        assert batch_summarise_pipelines([]) == []
