"""Extra tests for puzzle_reconstruction/utils/pipeline_runner.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.pipeline_runner import (
    RunnerConfig,
    StepResult,
    PipelineResult,
    PipelineStep,
    make_step,
    run_pipeline,
    get_step_output,
    filter_step_results,
    retry_failed_steps,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _step(name="step1", fn=None) -> PipelineStep:
    if fn is None:
        fn = lambda x: x
    return PipelineStep(name=name, fn=fn)


def _sr(name="s", index=0, success=True, output=None, error=None,
        elapsed=0.0) -> StepResult:
    return StepResult(name=name, index=index, success=success,
                      output=output, error=error, elapsed_s=elapsed)


def _pr(step_results=None, n_steps=1, n_success=1, n_failed=0,
        total_time=0.0, aborted=False) -> PipelineResult:
    if step_results is None:
        step_results = [_sr()]
    return PipelineResult(step_results=step_results, n_steps=n_steps,
                           n_success=n_success, n_failed=n_failed,
                           total_time_s=total_time, aborted=aborted)


# ─── RunnerConfig ─────────────────────────────────────────────────────────────

class TestRunnerConfigExtra:
    def test_default_stop_on_error(self):
        assert RunnerConfig().stop_on_error is True

    def test_default_measure_time(self):
        assert RunnerConfig().measure_time is True

    def test_negative_max_steps_raises(self):
        with pytest.raises(ValueError):
            RunnerConfig(max_steps=-1)

    def test_zero_max_steps_ok(self):
        cfg = RunnerConfig(max_steps=0)
        assert cfg.max_steps == 0


# ─── StepResult ───────────────────────────────────────────────────────────────

class TestStepResultExtra:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            StepResult(name="", index=0, success=True)

    def test_negative_index_raises(self):
        with pytest.raises(ValueError):
            StepResult(name="s", index=-1, success=True)

    def test_negative_elapsed_raises(self):
        with pytest.raises(ValueError):
            StepResult(name="s", index=0, success=True, elapsed_s=-0.1)

    def test_is_slow_false(self):
        sr = _sr(elapsed=0.5)
        assert sr.is_slow is False

    def test_is_slow_true(self):
        sr = _sr(elapsed=2.0)
        assert sr.is_slow is True


# ─── PipelineResult ───────────────────────────────────────────────────────────

class TestPipelineResultExtra:
    def test_success_ratio(self):
        pr = _pr(n_steps=4, n_success=3, n_failed=1)
        assert pr.success_ratio == pytest.approx(0.75)

    def test_success_ratio_zero_steps(self):
        pr = PipelineResult(step_results=[], n_steps=0, n_success=0,
                             n_failed=0, total_time_s=0.0)
        assert pr.success_ratio == pytest.approx(0.0)

    def test_negative_n_steps_raises(self):
        with pytest.raises(ValueError):
            PipelineResult(step_results=[], n_steps=-1, n_success=0,
                            n_failed=0, total_time_s=0.0)

    def test_negative_total_time_raises(self):
        with pytest.raises(ValueError):
            PipelineResult(step_results=[], n_steps=0, n_success=0,
                            n_failed=0, total_time_s=-1.0)

    def test_outputs_only_success(self):
        steps = [_sr("a", success=True, output=42),
                 _sr("b", index=1, success=False, error="oops")]
        pr = PipelineResult(step_results=steps, n_steps=2, n_success=1,
                             n_failed=1, total_time_s=0.0)
        assert "a" in pr.outputs and "b" not in pr.outputs

    def test_errors_only_failed(self):
        steps = [_sr("a", success=False, error="err")]
        pr = PipelineResult(step_results=steps, n_steps=1, n_success=0,
                             n_failed=1, total_time_s=0.0)
        assert "a" in pr.errors

    def test_slowest_step_none_empty(self):
        pr = PipelineResult(step_results=[], n_steps=0, n_success=0,
                             n_failed=0, total_time_s=0.0)
        assert pr.slowest_step is None

    def test_slowest_step(self):
        steps = [_sr("fast", elapsed=0.1), _sr("slow", index=1, elapsed=3.0)]
        pr = PipelineResult(step_results=steps, n_steps=2, n_success=2,
                             n_failed=0, total_time_s=3.1)
        assert pr.slowest_step == "slow"


# ─── PipelineStep / make_step ─────────────────────────────────────────────────

class TestPipelineStepExtra:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            PipelineStep(name="", fn=lambda x: x)

    def test_make_step_returns_step(self):
        s = make_step("double", lambda x: x * 2)
        assert isinstance(s, PipelineStep)
        assert s.name == "double"


# ─── run_pipeline ─────────────────────────────────────────────────────────────

class TestRunPipelineExtra:
    def test_empty_steps_returns_result(self):
        pr = run_pipeline([])
        assert pr.n_steps == 0

    def test_single_success_step(self):
        steps = [make_step("add1", lambda x: (x or 0) + 1)]
        pr = run_pipeline(steps, initial_input=0)
        assert pr.n_success == 1
        assert pr.aborted is False

    def test_passes_output_to_next_step(self):
        steps = [
            make_step("a", lambda x: 10),
            make_step("b", lambda x: x * 2),
        ]
        pr = run_pipeline(steps, initial_input=0)
        assert get_step_output(pr, "b") == 20

    def test_stop_on_error_aborts(self):
        def fail(x):
            raise RuntimeError("boom")
        steps = [make_step("bad", fail), make_step("ok", lambda x: x)]
        pr = run_pipeline(steps, cfg=RunnerConfig(stop_on_error=True))
        assert pr.aborted is True
        assert pr.n_steps == 1

    def test_continue_on_error(self):
        def fail(x):
            raise RuntimeError("boom")
        steps = [make_step("bad", fail), make_step("ok", lambda x: "done")]
        pr = run_pipeline(steps, cfg=RunnerConfig(stop_on_error=False))
        assert pr.aborted is False
        assert pr.n_failed == 1

    def test_max_steps_limits(self):
        steps = [make_step(f"s{i}", lambda x: x) for i in range(5)]
        pr = run_pipeline(steps, cfg=RunnerConfig(max_steps=2))
        assert pr.n_steps == 2


# ─── get_step_output ──────────────────────────────────────────────────────────

class TestGetStepOutputExtra:
    def test_returns_output(self):
        steps = [make_step("compute", lambda x: 42)]
        pr = run_pipeline(steps, initial_input=None)
        assert get_step_output(pr, "compute") == 42

    def test_missing_step_returns_none(self):
        pr = _pr()
        assert get_step_output(pr, "nonexistent") is None

    def test_failed_step_returns_none(self):
        sr = _sr("bad", success=False, error="err")
        pr = PipelineResult(step_results=[sr], n_steps=1,
                             n_success=0, n_failed=1, total_time_s=0.0)
        assert get_step_output(pr, "bad") is None


# ─── filter_step_results ──────────────────────────────────────────────────────

class TestFilterStepResultsExtra:
    def test_success_only(self):
        steps = [make_step("ok", lambda x: x),
                 make_step("bad", lambda x: (_ for _ in ()).throw(ValueError()))]
        pr = run_pipeline(steps, cfg=RunnerConfig(stop_on_error=False))
        success = filter_step_results(pr, success_only=True)
        assert all(sr.success for sr in success)

    def test_failed_only(self):
        def fail(x): raise RuntimeError("x")
        steps = [make_step("bad", fail), make_step("ok", lambda x: 1)]
        pr = run_pipeline(steps, cfg=RunnerConfig(stop_on_error=False))
        failed = filter_step_results(pr, success_only=False)
        assert all(not sr.success for sr in failed)


# ─── retry_failed_steps ───────────────────────────────────────────────────────

class TestRetryFailedStepsExtra:
    def test_retries_only_failed(self):
        def fail(x): raise ValueError("fail")
        steps = [make_step("ok", lambda x: x), make_step("bad", fail)]
        pr = run_pipeline(steps, cfg=RunnerConfig(stop_on_error=False))
        # Now fix "bad" and retry
        fixed_steps = [make_step("ok", lambda x: x),
                       make_step("bad", lambda x: "fixed")]
        retry_result = retry_failed_steps(fixed_steps, pr)
        assert retry_result.n_success == 1
