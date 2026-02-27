"""Tests for puzzle_reconstruction.utils.pipeline_runner."""
import pytest
import numpy as np

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

np.random.seed(42)


# ── RunnerConfig ──────────────────────────────────────────────────────────────

def test_runner_config_defaults():
    cfg = RunnerConfig()
    assert cfg.stop_on_error is True
    assert cfg.measure_time is True
    assert cfg.verbose is False
    assert cfg.max_steps == 0


def test_runner_config_custom():
    cfg = RunnerConfig(stop_on_error=False, measure_time=False, verbose=True, max_steps=5)
    assert cfg.stop_on_error is False
    assert cfg.max_steps == 5


def test_runner_config_negative_max_steps():
    with pytest.raises(ValueError):
        RunnerConfig(max_steps=-1)


# ── StepResult ────────────────────────────────────────────────────────────────

def test_step_result_success():
    sr = StepResult(name="step1", index=0, success=True, output=42)
    assert sr.name == "step1"
    assert sr.success is True
    assert sr.output == 42


def test_step_result_empty_name_raises():
    with pytest.raises(ValueError):
        StepResult(name="", index=0, success=True)


def test_step_result_negative_index_raises():
    with pytest.raises(ValueError):
        StepResult(name="s", index=-1, success=True)


def test_step_result_negative_elapsed_raises():
    with pytest.raises(ValueError):
        StepResult(name="s", index=0, success=True, elapsed_s=-0.1)


def test_step_result_is_slow_false():
    sr = StepResult(name="s", index=0, success=True, elapsed_s=0.5)
    assert sr.is_slow is False


def test_step_result_is_slow_true():
    sr = StepResult(name="s", index=0, success=True, elapsed_s=1.5)
    assert sr.is_slow is True


# ── PipelineResult ────────────────────────────────────────────────────────────

def test_pipeline_result_success_ratio_empty():
    pr = PipelineResult(step_results=[], n_steps=0, n_success=0, n_failed=0, total_time_s=0.0)
    assert pr.success_ratio == 0.0


def test_pipeline_result_success_ratio():
    sr1 = StepResult(name="a", index=0, success=True, output=1)
    sr2 = StepResult(name="b", index=1, success=False, error="err")
    pr = PipelineResult(step_results=[sr1, sr2], n_steps=2, n_success=1, n_failed=1, total_time_s=0.1)
    assert pr.success_ratio == 0.5


def test_pipeline_result_outputs():
    sr1 = StepResult(name="x", index=0, success=True, output=99)
    sr2 = StepResult(name="y", index=1, success=False, error="oops")
    pr = PipelineResult(step_results=[sr1, sr2], n_steps=2, n_success=1, n_failed=1, total_time_s=0.0)
    assert pr.outputs == {"x": 99}


def test_pipeline_result_errors():
    sr = StepResult(name="z", index=0, success=False, error="bad")
    pr = PipelineResult(step_results=[sr], n_steps=1, n_success=0, n_failed=1, total_time_s=0.0)
    assert pr.errors == {"z": "bad"}


def test_pipeline_result_slowest_step():
    sr1 = StepResult(name="a", index=0, success=True, elapsed_s=0.1)
    sr2 = StepResult(name="b", index=1, success=True, elapsed_s=0.9)
    pr = PipelineResult(step_results=[sr1, sr2], n_steps=2, n_success=2, n_failed=0, total_time_s=1.0)
    assert pr.slowest_step == "b"


def test_pipeline_result_slowest_step_none():
    pr = PipelineResult(step_results=[], n_steps=0, n_success=0, n_failed=0, total_time_s=0.0)
    assert pr.slowest_step is None


# ── make_step / PipelineStep ──────────────────────────────────────────────────

def test_make_step_creates_pipeline_step():
    ps = make_step("double", lambda x: x * 2)
    assert isinstance(ps, PipelineStep)
    assert ps.name == "double"
    assert ps.fn(5) == 10


def test_pipeline_step_empty_name_raises():
    with pytest.raises(ValueError):
        PipelineStep(name="", fn=lambda x: x)


# ── run_pipeline ──────────────────────────────────────────────────────────────

def test_run_pipeline_basic():
    steps = [
        make_step("add1", lambda x: x + 1),
        make_step("mul2", lambda x: x * 2),
    ]
    result = run_pipeline(steps, initial_input=3)
    assert result.n_success == 2
    assert result.n_failed == 0
    assert result.outputs["mul2"] == 8


def test_run_pipeline_stop_on_error():
    def fail(x):
        raise RuntimeError("boom")

    steps = [
        make_step("ok", lambda x: 1),
        make_step("fail", fail),
        make_step("never", lambda x: 2),
    ]
    result = run_pipeline(steps, cfg=RunnerConfig(stop_on_error=True))
    assert result.aborted is True
    assert result.n_failed == 1
    assert len(result.step_results) == 2


def test_run_pipeline_no_stop_on_error():
    def fail(x):
        raise ValueError("err")

    steps = [
        make_step("ok", lambda x: 42),
        make_step("fail", fail),
        make_step("ok2", lambda x: 99 if x is None else x),
    ]
    result = run_pipeline(steps, cfg=RunnerConfig(stop_on_error=False))
    assert result.aborted is False
    assert result.n_failed == 1
    assert result.n_success == 2


def test_run_pipeline_max_steps():
    steps = [make_step(f"s{i}", lambda x, i=i: i) for i in range(5)]
    result = run_pipeline(steps, cfg=RunnerConfig(max_steps=3))
    assert result.n_steps == 3


def test_run_pipeline_empty():
    result = run_pipeline([])
    assert result.n_steps == 0
    assert result.n_success == 0


# ── get_step_output ───────────────────────────────────────────────────────────

def test_get_step_output_found():
    steps = [make_step("compute", lambda x: 7)]
    result = run_pipeline(steps, initial_input=0)
    assert get_step_output(result, "compute") == 7


def test_get_step_output_not_found():
    result = run_pipeline([])
    assert get_step_output(result, "missing") is None


# ── filter_step_results ───────────────────────────────────────────────────────

def test_filter_step_results_success_only():
    steps = [
        make_step("a", lambda x: 1),
        make_step("b", lambda x: (_ for _ in ()).throw(ValueError("err"))),
    ]
    result = run_pipeline(steps, cfg=RunnerConfig(stop_on_error=False))
    ok = filter_step_results(result, success_only=True)
    assert all(r.success for r in ok)


def test_filter_step_results_failed_only():
    steps = [
        make_step("a", lambda x: 1),
        make_step("b", lambda x: (_ for _ in ()).throw(ValueError("err"))),
    ]
    result = run_pipeline(steps, cfg=RunnerConfig(stop_on_error=False))
    failed = filter_step_results(result, success_only=False)
    assert all(not r.success for r in failed)


# ── retry_failed_steps ────────────────────────────────────────────────────────

def test_retry_failed_steps():
    call_count = {"n": 0}

    def flaky(x):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("first attempt fails")
        return 99

    steps = [make_step("flaky", flaky)]
    first = run_pipeline(steps, cfg=RunnerConfig(stop_on_error=False))
    assert first.n_failed == 1

    retry = retry_failed_steps(steps, first)
    assert retry.n_success == 1
