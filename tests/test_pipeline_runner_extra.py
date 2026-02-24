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

def _step_result(name="s", index=0, success=True, output=None,
                 error=None, elapsed=0.0) -> StepResult:
    return StepResult(name=name, index=index, success=success,
                      output=output, error=error, elapsed_s=elapsed)


def _pipeline_result(steps=None, n=0, ns=0, nf=0,
                     total=0.0, aborted=False) -> PipelineResult:
    return PipelineResult(
        step_results=steps or [],
        n_steps=n, n_success=ns, n_failed=nf,
        total_time_s=total, aborted=aborted,
    )


def _id_step(name: str) -> PipelineStep:
    return make_step(name, lambda x: x)


def _fail_step(name: str) -> PipelineStep:
    def _fail(x):
        raise RuntimeError("deliberate failure")
    return make_step(name, _fail)


def _double_step(name: str) -> PipelineStep:
    return make_step(name, lambda x: (x or 0) * 2)


# ─── RunnerConfig ─────────────────────────────────────────────────────────────

class TestRunnerConfigExtra:
    def test_default_stop_on_error(self):
        assert RunnerConfig().stop_on_error is True

    def test_default_measure_time(self):
        assert RunnerConfig().measure_time is True

    def test_default_verbose(self):
        assert RunnerConfig().verbose is False

    def test_default_max_steps(self):
        assert RunnerConfig().max_steps == 0

    def test_negative_max_steps_raises(self):
        with pytest.raises(ValueError):
            RunnerConfig(max_steps=-1)

    def test_custom_max_steps(self):
        cfg = RunnerConfig(max_steps=5)
        assert cfg.max_steps == 5

    def test_stop_on_error_false(self):
        cfg = RunnerConfig(stop_on_error=False)
        assert cfg.stop_on_error is False

    def test_verbose_true(self):
        cfg = RunnerConfig(verbose=True)
        assert cfg.verbose is True


# ─── StepResult ───────────────────────────────────────────────────────────────

class TestStepResultExtra:
    def test_name_stored(self):
        sr = _step_result(name="step1")
        assert sr.name == "step1"

    def test_index_stored(self):
        sr = _step_result(index=3)
        assert sr.index == 3

    def test_success_stored(self):
        sr = _step_result(success=False)
        assert sr.success is False

    def test_output_stored(self):
        sr = _step_result(output=42)
        assert sr.output == 42

    def test_error_stored(self):
        sr = _step_result(error="oops", success=False)
        assert sr.error == "oops"

    def test_elapsed_stored(self):
        sr = _step_result(elapsed=1.5)
        assert sr.elapsed_s == pytest.approx(1.5)

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            StepResult(name="", index=0, success=True)

    def test_negative_index_raises(self):
        with pytest.raises(ValueError):
            StepResult(name="s", index=-1, success=True)

    def test_negative_elapsed_raises(self):
        with pytest.raises(ValueError):
            StepResult(name="s", index=0, success=True, elapsed_s=-0.1)

    def test_is_slow_true(self):
        sr = _step_result(elapsed=2.0)
        assert sr.is_slow is True

    def test_is_slow_false(self):
        sr = _step_result(elapsed=0.5)
        assert sr.is_slow is False

    def test_is_slow_boundary(self):
        sr = _step_result(elapsed=1.0)
        assert sr.is_slow is False


# ─── PipelineResult ───────────────────────────────────────────────────────────

class TestPipelineResultExtra:
    def test_n_steps_stored(self):
        pr = _pipeline_result(n=3)
        assert pr.n_steps == 3

    def test_n_success_stored(self):
        pr = _pipeline_result(ns=2)
        assert pr.n_success == 2

    def test_n_failed_stored(self):
        pr = _pipeline_result(nf=1)
        assert pr.n_failed == 1

    def test_total_time_stored(self):
        pr = _pipeline_result(total=3.14)
        assert pr.total_time_s == pytest.approx(3.14)

    def test_aborted_stored(self):
        pr = _pipeline_result(aborted=True)
        assert pr.aborted is True

    def test_negative_n_steps_raises(self):
        with pytest.raises(ValueError):
            PipelineResult(step_results=[], n_steps=-1, n_success=0,
                           n_failed=0, total_time_s=0.0)

    def test_negative_total_time_raises(self):
        with pytest.raises(ValueError):
            PipelineResult(step_results=[], n_steps=0, n_success=0,
                           n_failed=0, total_time_s=-1.0)

    def test_success_ratio_zero_steps(self):
        pr = _pipeline_result(n=0)
        assert pr.success_ratio == pytest.approx(0.0)

    def test_success_ratio_full(self):
        pr = _pipeline_result(n=3, ns=3)
        assert pr.success_ratio == pytest.approx(1.0)

    def test_success_ratio_partial(self):
        pr = _pipeline_result(n=4, ns=2)
        assert pr.success_ratio == pytest.approx(0.5)

    def test_outputs_only_successful(self):
        steps = [
            _step_result("a", 0, True, output="out_a"),
            _step_result("b", 1, False, output=None),
        ]
        pr = _pipeline_result(steps=steps)
        assert "a" in pr.outputs
        assert "b" not in pr.outputs

    def test_errors_only_failed(self):
        steps = [
            _step_result("a", 0, True),
            _step_result("b", 1, False, error="boom"),
        ]
        pr = _pipeline_result(steps=steps)
        assert "b" in pr.errors
        assert "a" not in pr.errors

    def test_slowest_step_none_when_empty(self):
        pr = _pipeline_result()
        assert pr.slowest_step is None

    def test_slowest_step_identified(self):
        steps = [
            _step_result("fast", 0, elapsed=0.1),
            _step_result("slow", 1, elapsed=2.0),
        ]
        pr = _pipeline_result(steps=steps)
        assert pr.slowest_step == "slow"


# ─── PipelineStep ─────────────────────────────────────────────────────────────

class TestPipelineStepExtra:
    def test_name_stored(self):
        step = PipelineStep(name="preprocess", fn=lambda x: x)
        assert step.name == "preprocess"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            PipelineStep(name="", fn=lambda x: x)

    def test_fn_callable(self):
        fn = lambda x: x + 1
        step = PipelineStep(name="s", fn=fn)
        assert step.fn(1) == 2


# ─── make_step ────────────────────────────────────────────────────────────────

class TestMakeStepExtra:
    def test_returns_pipeline_step(self):
        step = make_step("s", lambda x: x)
        assert isinstance(step, PipelineStep)

    def test_name_assigned(self):
        step = make_step("my_step", lambda x: x)
        assert step.name == "my_step"

    def test_fn_works(self):
        step = make_step("double", lambda x: x * 2)
        assert step.fn(5) == 10


# ─── run_pipeline ─────────────────────────────────────────────────────────────

class TestRunPipelineExtra:
    def test_returns_pipeline_result(self):
        result = run_pipeline([_id_step("s")])
        assert isinstance(result, PipelineResult)

    def test_empty_steps(self):
        result = run_pipeline([])
        assert result.n_steps == 0

    def test_single_step_success(self):
        result = run_pipeline([_id_step("s")], initial_input=5)
        assert result.n_success == 1
        assert result.n_failed == 0

    def test_step_output_propagates(self):
        steps = [_double_step("d1"), _double_step("d2")]
        result = run_pipeline(steps, initial_input=3)
        assert result.outputs["d2"] == 12  # 3*2*2

    def test_stop_on_error_aborts(self):
        steps = [_fail_step("f"), _id_step("s")]
        cfg = RunnerConfig(stop_on_error=True)
        result = run_pipeline(steps, cfg=cfg)
        assert result.aborted is True
        assert result.n_steps == 1  # only one step recorded

    def test_continue_on_error(self):
        steps = [_fail_step("f"), _id_step("s")]
        cfg = RunnerConfig(stop_on_error=False)
        result = run_pipeline(steps, cfg=cfg)
        assert result.aborted is False
        assert result.n_failed == 1
        assert result.n_success == 1

    def test_max_steps_limits(self):
        steps = [_id_step("a"), _id_step("b"), _id_step("c")]
        cfg = RunnerConfig(max_steps=2)
        result = run_pipeline(steps, cfg=cfg)
        assert result.n_steps == 2

    def test_none_cfg_uses_defaults(self):
        result = run_pipeline([_id_step("s")], cfg=None)
        assert isinstance(result, PipelineResult)

    def test_error_message_captured(self):
        steps = [_fail_step("f")]
        cfg = RunnerConfig(stop_on_error=False)
        result = run_pipeline(steps, cfg=cfg)
        assert "f" in result.errors

    def test_total_time_nonnegative(self):
        result = run_pipeline([_id_step("s")])
        assert result.total_time_s >= 0.0


# ─── get_step_output ──────────────────────────────────────────────────────────

class TestGetStepOutputExtra:
    def _pr_with_steps(self, names_outputs):
        steps = [
            _step_result(name=n, index=i, success=True, output=o)
            for i, (n, o) in enumerate(names_outputs)
        ]
        return _pipeline_result(steps=steps)

    def test_returns_output_for_existing(self):
        pr = self._pr_with_steps([("load", 42)])
        assert get_step_output(pr, "load") == 42

    def test_returns_none_for_missing(self):
        pr = self._pr_with_steps([("load", 42)])
        assert get_step_output(pr, "nonexistent") is None

    def test_returns_none_for_failed_step(self):
        steps = [_step_result("f", 0, success=False, output=None)]
        pr = _pipeline_result(steps=steps)
        assert get_step_output(pr, "f") is None

    def test_multiple_steps(self):
        pr = self._pr_with_steps([("a", 1), ("b", 2)])
        assert get_step_output(pr, "b") == 2


# ─── filter_step_results ──────────────────────────────────────────────────────

class TestFilterStepResultsExtra:
    def _pr(self):
        steps = [
            _step_result("ok1", 0, success=True),
            _step_result("fail1", 1, success=False),
            _step_result("ok2", 2, success=True),
        ]
        return _pipeline_result(steps=steps)

    def test_success_only_default(self):
        filtered = filter_step_results(self._pr())
        assert all(sr.success for sr in filtered)

    def test_success_only_count(self):
        assert len(filter_step_results(self._pr())) == 2

    def test_failure_only(self):
        filtered = filter_step_results(self._pr(), success_only=False)
        assert all(not sr.success for sr in filtered)
        assert len(filtered) == 1

    def test_empty_pipeline(self):
        pr = _pipeline_result()
        assert filter_step_results(pr) == []


# ─── retry_failed_steps ───────────────────────────────────────────────────────

class TestRetryFailedStepsExtra:
    def test_returns_pipeline_result(self):
        prev = _pipeline_result(
            steps=[_step_result("f", 0, False, error="err")]
        )
        steps = [_id_step("f")]
        result = retry_failed_steps(steps, prev)
        assert isinstance(result, PipelineResult)

    def test_only_failed_retried(self):
        prev = _pipeline_result(
            steps=[
                _step_result("ok", 0, True),
                _step_result("bad", 1, False, error="err"),
            ]
        )
        steps = [_id_step("ok"), _id_step("bad")]
        result = retry_failed_steps(steps, prev)
        # Only 'bad' should be retried
        assert all(sr.name == "bad" for sr in result.step_results)

    def test_no_failed_empty_result(self):
        prev = _pipeline_result(
            steps=[_step_result("ok", 0, True)]
        )
        steps = [_id_step("ok")]
        result = retry_failed_steps(steps, prev)
        assert result.n_steps == 0

    def test_retry_can_succeed(self):
        prev = _pipeline_result(
            steps=[_step_result("s", 0, False, error="err")]
        )
        steps = [_id_step("s")]
        result = retry_failed_steps(steps, prev, initial_input=7)
        assert result.n_success == 1
