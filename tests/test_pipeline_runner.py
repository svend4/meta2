"""Тесты для puzzle_reconstruction.utils.pipeline_runner."""
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

def _identity(x):
    return x


def _add_one(x):
    return (x or 0) + 1


def _raise(x):
    raise RuntimeError("deliberate error")


def _steps(*fns_and_names):
    """Create PipelineStep list from (name, fn) tuples."""
    return [make_step(name, fn) for name, fn in fns_and_names]


def _simple_steps(n: int = 3):
    return _steps(*[(f"step_{i}", _add_one) for i in range(n)])


# ─── TestRunnerConfig ─────────────────────────────────────────────────────────

class TestRunnerConfig:
    def test_defaults(self):
        cfg = RunnerConfig()
        assert cfg.stop_on_error is True
        assert cfg.measure_time is True
        assert cfg.verbose is False
        assert cfg.max_steps == 0

    def test_valid_custom(self):
        cfg = RunnerConfig(stop_on_error=False, verbose=True, max_steps=5)
        assert cfg.stop_on_error is False
        assert cfg.verbose is True
        assert cfg.max_steps == 5

    def test_max_steps_zero_ok(self):
        cfg = RunnerConfig(max_steps=0)
        assert cfg.max_steps == 0

    def test_max_steps_neg_raises(self):
        with pytest.raises(ValueError):
            RunnerConfig(max_steps=-1)

    def test_measure_time_false(self):
        cfg = RunnerConfig(measure_time=False)
        assert cfg.measure_time is False


# ─── TestStepResult ───────────────────────────────────────────────────────────

class TestStepResult:
    def test_basic_success(self):
        sr = StepResult(name="s0", index=0, success=True, output=42,
                        elapsed_s=0.01)
        assert sr.name == "s0"
        assert sr.index == 0
        assert sr.success is True
        assert sr.output == 42

    def test_basic_failure(self):
        sr = StepResult(name="s1", index=1, success=False,
                        error="RuntimeError: oops", elapsed_s=0.0)
        assert sr.success is False
        assert sr.error == "RuntimeError: oops"
        assert sr.output is None

    def test_is_slow_false(self):
        sr = StepResult(name="s", index=0, success=True, elapsed_s=0.5)
        assert sr.is_slow is False

    def test_is_slow_true(self):
        sr = StepResult(name="s", index=0, success=True, elapsed_s=1.5)
        assert sr.is_slow is True

    def test_is_slow_boundary(self):
        sr = StepResult(name="s", index=0, success=True, elapsed_s=1.0)
        assert sr.is_slow is False  # strictly >

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            StepResult(name="", index=0, success=True, elapsed_s=0.0)

    def test_neg_index_raises(self):
        with pytest.raises(ValueError):
            StepResult(name="s", index=-1, success=True, elapsed_s=0.0)

    def test_neg_elapsed_raises(self):
        with pytest.raises(ValueError):
            StepResult(name="s", index=0, success=True, elapsed_s=-0.1)


# ─── TestPipelineResult ───────────────────────────────────────────────────────

class TestPipelineResult:
    def _make(self, n_success=2, n_failed=1, aborted=False):
        steps = [
            StepResult(name=f"ok_{i}", index=i, success=True, output=i * 10,
                       elapsed_s=0.01)
            for i in range(n_success)
        ]
        steps += [
            StepResult(name=f"fail_{i}", index=n_success + i, success=False,
                       error="Err: boom", elapsed_s=0.0)
            for i in range(n_failed)
        ]
        return PipelineResult(
            step_results=steps,
            n_steps=n_success + n_failed,
            n_success=n_success,
            n_failed=n_failed,
            total_time_s=0.1,
            aborted=aborted,
        )

    def test_success_ratio_full(self):
        pr = self._make(n_success=3, n_failed=0)
        assert pr.success_ratio == pytest.approx(1.0)

    def test_success_ratio_partial(self):
        pr = self._make(n_success=2, n_failed=2)
        assert pr.success_ratio == pytest.approx(0.5)

    def test_success_ratio_zero_steps(self):
        pr = PipelineResult(step_results=[], n_steps=0, n_success=0,
                            n_failed=0, total_time_s=0.0)
        assert pr.success_ratio == pytest.approx(0.0)

    def test_outputs_keys_are_successful(self):
        pr = self._make(n_success=2, n_failed=1)
        assert "ok_0" in pr.outputs
        assert "ok_1" in pr.outputs
        assert "fail_0" not in pr.outputs

    def test_errors_keys_are_failed(self):
        pr = self._make(n_success=1, n_failed=2)
        assert "fail_0" in pr.errors
        assert "fail_1" in pr.errors
        assert "ok_0" not in pr.errors

    def test_slowest_step_none_when_empty(self):
        pr = PipelineResult(step_results=[], n_steps=0, n_success=0,
                            n_failed=0, total_time_s=0.0)
        assert pr.slowest_step is None

    def test_slowest_step_name(self):
        steps = [
            StepResult(name="fast", index=0, success=True, elapsed_s=0.01),
            StepResult(name="slow", index=1, success=True, elapsed_s=2.5),
        ]
        pr = PipelineResult(step_results=steps, n_steps=2, n_success=2,
                            n_failed=0, total_time_s=2.51)
        assert pr.slowest_step == "slow"

    def test_aborted_flag(self):
        pr = self._make(aborted=True)
        assert pr.aborted is True

    def test_invalid_n_steps_neg(self):
        with pytest.raises(ValueError):
            PipelineResult(step_results=[], n_steps=-1, n_success=0,
                           n_failed=0, total_time_s=0.0)

    def test_invalid_n_success_neg(self):
        with pytest.raises(ValueError):
            PipelineResult(step_results=[], n_steps=0, n_success=-1,
                           n_failed=0, total_time_s=0.0)

    def test_invalid_n_failed_neg(self):
        with pytest.raises(ValueError):
            PipelineResult(step_results=[], n_steps=0, n_success=0,
                           n_failed=-1, total_time_s=0.0)

    def test_invalid_total_time_neg(self):
        with pytest.raises(ValueError):
            PipelineResult(step_results=[], n_steps=0, n_success=0,
                           n_failed=0, total_time_s=-0.1)


# ─── TestPipelineStep ─────────────────────────────────────────────────────────

class TestPipelineStep:
    def test_basic(self):
        ps = PipelineStep(name="double", fn=lambda x: x * 2)
        assert ps.name == "double"
        assert ps.fn(5) == 10

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            PipelineStep(name="", fn=_identity)


# ─── TestMakeStep ─────────────────────────────────────────────────────────────

class TestMakeStep:
    def test_returns_pipeline_step(self):
        s = make_step("inc", _add_one)
        assert isinstance(s, PipelineStep)

    def test_name_preserved(self):
        s = make_step("my_step", _identity)
        assert s.name == "my_step"

    def test_fn_callable(self):
        s = make_step("inc", _add_one)
        assert s.fn(0) == 1

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            make_step("", _identity)


# ─── TestRunPipeline ──────────────────────────────────────────────────────────

class TestRunPipeline:
    def test_basic_chain(self):
        steps = _simple_steps(3)
        pr = run_pipeline(steps, initial_input=0)
        assert pr.n_success == 3
        assert pr.n_failed == 0

    def test_output_accumulates(self):
        steps = _steps(("inc", _add_one), ("inc2", _add_one), ("inc3", _add_one))
        pr = run_pipeline(steps, initial_input=10)
        assert pr.outputs["inc3"] == 13

    def test_empty_steps(self):
        pr = run_pipeline([])
        assert pr.n_steps == 0
        assert pr.n_success == 0
        assert pr.n_failed == 0

    def test_stop_on_error_true(self):
        steps = _steps(
            ("ok", _add_one),
            ("fail", _raise),
            ("after", _add_one),
        )
        cfg = RunnerConfig(stop_on_error=True)
        pr = run_pipeline(steps, initial_input=0, cfg=cfg)
        assert pr.aborted is True
        assert pr.n_failed == 1
        assert pr.n_steps == 2  # "after" not executed

    def test_stop_on_error_false_continues(self):
        steps = _steps(
            ("ok", _add_one),
            ("fail", _raise),
            ("after", _add_one),
        )
        cfg = RunnerConfig(stop_on_error=False)
        pr = run_pipeline(steps, initial_input=0, cfg=cfg)
        assert pr.aborted is False
        assert pr.n_failed == 1
        assert pr.n_success == 2

    def test_max_steps_limits_execution(self):
        steps = _simple_steps(5)
        cfg = RunnerConfig(max_steps=2)
        pr = run_pipeline(steps, initial_input=0, cfg=cfg)
        assert pr.n_steps == 2

    def test_max_steps_zero_runs_all(self):
        steps = _simple_steps(4)
        cfg = RunnerConfig(max_steps=0)
        pr = run_pipeline(steps, initial_input=0, cfg=cfg)
        assert pr.n_steps == 4

    def test_step_indices_sequential(self):
        steps = _simple_steps(4)
        pr = run_pipeline(steps, initial_input=0)
        for i, sr in enumerate(pr.step_results):
            assert sr.index == i

    def test_total_time_nonneg(self):
        steps = _simple_steps(3)
        pr = run_pipeline(steps, initial_input=0)
        assert pr.total_time_s >= 0.0

    def test_measure_time_false_elapsed_zero(self):
        steps = _simple_steps(2)
        cfg = RunnerConfig(measure_time=False)
        pr = run_pipeline(steps, initial_input=0, cfg=cfg)
        for sr in pr.step_results:
            assert sr.elapsed_s == pytest.approx(0.0)

    def test_error_message_stored(self):
        steps = _steps(("boom", _raise))
        cfg = RunnerConfig(stop_on_error=True)
        pr = run_pipeline(steps, initial_input=None, cfg=cfg)
        assert "RuntimeError" in pr.errors.get("boom", "")

    def test_initial_input_none(self):
        pr = run_pipeline(_simple_steps(1), initial_input=None)
        assert pr.n_success == 1

    def test_default_config(self):
        pr = run_pipeline(_simple_steps(2), initial_input=0)
        assert pr.n_success == 2


# ─── TestGetStepOutput ────────────────────────────────────────────────────────

class TestGetStepOutput:
    def _run(self):
        steps = _steps(("a", _add_one), ("b", _add_one))
        return run_pipeline(steps, initial_input=5)

    def test_found_success(self):
        pr = self._run()
        out = get_step_output(pr, "a")
        assert out == 6

    def test_found_second_step(self):
        pr = self._run()
        out = get_step_output(pr, "b")
        assert out == 7

    def test_not_found_returns_none(self):
        pr = self._run()
        assert get_step_output(pr, "nonexistent") is None

    def test_failed_step_returns_none(self):
        steps = _steps(("boom", _raise))
        cfg = RunnerConfig(stop_on_error=False)
        pr = run_pipeline(steps, initial_input=0, cfg=cfg)
        assert get_step_output(pr, "boom") is None


# ─── TestFilterStepResults ────────────────────────────────────────────────────

class TestFilterStepResults:
    def _run_mixed(self):
        steps = _steps(("ok", _add_one), ("fail", _raise), ("ok2", _add_one))
        cfg = RunnerConfig(stop_on_error=False)
        return run_pipeline(steps, initial_input=0, cfg=cfg)

    def test_success_only(self):
        pr = self._run_mixed()
        filtered = filter_step_results(pr, success_only=True)
        assert all(sr.success for sr in filtered)

    def test_failed_only(self):
        pr = self._run_mixed()
        filtered = filter_step_results(pr, success_only=False)
        assert all(not sr.success for sr in filtered)

    def test_success_count(self):
        pr = self._run_mixed()
        assert len(filter_step_results(pr, True)) == 2

    def test_failed_count(self):
        pr = self._run_mixed()
        assert len(filter_step_results(pr, False)) == 1

    def test_all_success_no_failures(self):
        pr = run_pipeline(_simple_steps(3), initial_input=0)
        assert filter_step_results(pr, False) == []


# ─── TestRetryFailedSteps ─────────────────────────────────────────────────────

class TestRetryFailedSteps:
    def _run_with_failures(self):
        steps = _steps(
            ("ok_a", _add_one),
            ("fail_b", _raise),
            ("ok_c", _add_one),
        )
        cfg = RunnerConfig(stop_on_error=False)
        return steps, run_pipeline(steps, initial_input=0, cfg=cfg)

    def test_retries_failed_only(self):
        steps, prev = self._run_with_failures()
        # Replace fail_b with a working step for retry
        fixed_steps = [
            make_step("ok_a", _add_one),
            make_step("fail_b", _add_one),  # fixed
            make_step("ok_c", _add_one),
        ]
        retry_pr = retry_failed_steps(fixed_steps, prev, initial_input=0)
        assert retry_pr.n_success >= 1
        assert "ok_a" not in [sr.name for sr in retry_pr.step_results]
        assert "ok_c" not in [sr.name for sr in retry_pr.step_results]

    def test_no_failures_returns_empty(self):
        steps = _simple_steps(3)
        prev = run_pipeline(steps, initial_input=0)
        retry_pr = retry_failed_steps(steps, prev, initial_input=0)
        assert retry_pr.n_steps == 0

    def test_returns_pipeline_result(self):
        steps, prev = self._run_with_failures()
        retry_pr = retry_failed_steps(steps, prev, initial_input=0)
        assert isinstance(retry_pr, PipelineResult)
