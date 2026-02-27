"""Tests for puzzle_reconstruction.utils.profiler"""
import time
import pytest
import numpy as np

from puzzle_reconstruction.utils.profiler import (
    StepProfile,
    PipelineProfiler,
    format_duration,
    profile_function,
    timed,
    compare_profilers,
)


# ── StepProfile ──────────────────────────────────────────────────────────────

def test_step_profile_defaults():
    sp = StepProfile(name="step1")
    assert sp.name == "step1"
    assert sp.elapsed_ms == 0.0
    assert sp.n_calls == 0
    assert sp.memory_mb == 0.0
    assert sp.children == []


def test_step_profile_elapsed_s():
    sp = StepProfile(name="x", elapsed_ms=2500.0, n_calls=1)
    assert sp.elapsed_s == pytest.approx(2.5)


def test_step_profile_avg_ms_no_calls():
    sp = StepProfile(name="x", elapsed_ms=100.0, n_calls=0)
    assert sp.avg_ms == 0.0


def test_step_profile_avg_ms_with_calls():
    sp = StepProfile(name="x", elapsed_ms=300.0, n_calls=3)
    assert sp.avg_ms == pytest.approx(100.0)


def test_step_profile_repr_contains_name():
    sp = StepProfile(name="mySte")
    assert "mySte" in repr(sp)


# ── PipelineProfiler ─────────────────────────────────────────────────────────

def test_profiler_empty():
    p = PipelineProfiler()
    assert p.total_elapsed_ms() == 0.0
    assert p.slowest_step() is None
    assert p.fastest_step() is None


def test_profiler_record():
    p = PipelineProfiler()
    p.record("step1", 100.0)
    assert p.get("step1") is not None
    assert p.get("step1").elapsed_ms == pytest.approx(100.0)
    assert p.get("step1").n_calls == 1


def test_profiler_record_accumulates():
    p = PipelineProfiler()
    p.record("step1", 50.0)
    p.record("step1", 50.0)
    assert p.get("step1").elapsed_ms == pytest.approx(100.0)
    assert p.get("step1").n_calls == 2


def test_profiler_start_stop():
    p = PipelineProfiler()
    p.start("step2")
    time.sleep(0.01)
    elapsed = p.stop("step2")
    assert elapsed >= 0.0
    assert p.get("step2").elapsed_ms >= 0.0


def test_profiler_stop_unstarted_raises():
    p = PipelineProfiler()
    with pytest.raises(KeyError):
        p.stop("nonexistent")


def test_profiler_step_context_manager():
    p = PipelineProfiler()
    with p.step("ctx_step"):
        time.sleep(0.005)
    assert p.get("ctx_step") is not None
    assert p.get("ctx_step").n_calls == 1


def test_profiler_total_elapsed():
    p = PipelineProfiler()
    p.record("a", 100.0)
    p.record("b", 200.0)
    assert p.total_elapsed_ms() == pytest.approx(300.0)


def test_profiler_slowest_step():
    p = PipelineProfiler()
    p.record("fast", 10.0)
    p.record("slow", 500.0)
    assert p.slowest_step() == "slow"


def test_profiler_fastest_step():
    p = PipelineProfiler()
    p.record("fast", 10.0)
    p.record("slow", 500.0)
    assert p.fastest_step() == "fast"


def test_profiler_get_none_for_missing():
    p = PipelineProfiler()
    assert p.get("nonexistent") is None


def test_profiler_step_names_sorted_by_time():
    p = PipelineProfiler()
    p.record("slow", 300.0)
    p.record("medium", 200.0)
    p.record("fast", 100.0)
    names = p.step_names()
    assert names[0] == "slow"
    assert names[-1] == "fast"


def test_profiler_reset_clears():
    p = PipelineProfiler()
    p.record("step", 100.0)
    p.reset()
    assert p.get("step") is None
    assert p.total_elapsed_ms() == 0.0


def test_profiler_profiles_returns_list():
    p = PipelineProfiler()
    p.record("a", 10.0)
    profs = p.profiles()
    assert isinstance(profs, list)
    assert len(profs) == 1


def test_profiler_summary_table_contains_step_name():
    p = PipelineProfiler()
    p.record("my_step", 100.0)
    table = p.summary_table()
    assert "my_step" in table


def test_profiler_summary_table_top_n():
    p = PipelineProfiler()
    for i in range(5):
        p.record(f"step{i}", float(i * 10))
    table = p.summary_table(top_n=2)
    assert isinstance(table, str)


def test_profiler_repr():
    p = PipelineProfiler()
    p.record("x", 50.0)
    r = repr(p)
    assert "steps=1" in r


# ── format_duration ──────────────────────────────────────────────────────────

def test_format_duration_ms():
    assert "ms" in format_duration(500.0)
    assert "500.00ms" == format_duration(500.0)


def test_format_duration_seconds():
    result = format_duration(1500.0)
    assert "s" in result
    assert "1.50s" == result


def test_format_duration_minutes():
    result = format_duration(90000.0)
    assert "m" in result


def test_format_duration_zero():
    result = format_duration(0.0)
    assert "ms" in result


def test_format_duration_sub_second():
    result = format_duration(0.5)
    assert "ms" in result


# ── profile_function ─────────────────────────────────────────────────────────

def test_profile_function_returns_result():
    result, sp = profile_function(lambda x: x * 2, 21)
    assert result == 42
    assert sp.elapsed_ms >= 0.0
    assert sp.n_calls == 1


def test_profile_function_records_in_profiler():
    p = PipelineProfiler()
    profile_function(lambda: None, profiler=p, step_name="my_fn")
    assert p.get("my_fn") is not None


def test_profile_function_uses_fn_name():
    def my_named_fn():
        return 1
    _, sp = profile_function(my_named_fn)
    assert sp.name == "my_named_fn"


# ── timed decorator ──────────────────────────────────────────────────────────

def test_timed_decorator_records():
    p = PipelineProfiler()

    @timed(p, name="decorated")
    def my_fn(x):
        return x + 1

    result = my_fn(5)
    assert result == 6
    assert p.get("decorated") is not None
    assert p.get("decorated").n_calls == 1


def test_timed_decorator_uses_fn_name_by_default():
    p = PipelineProfiler()

    @timed(p)
    def auto_name():
        return True

    auto_name()
    assert p.get("auto_name") is not None


def test_timed_decorator_accumulates():
    p = PipelineProfiler()

    @timed(p, name="step")
    def fn():
        pass

    fn()
    fn()
    fn()
    assert p.get("step").n_calls == 3


# ── compare_profilers ────────────────────────────────────────────────────────

def test_compare_profilers_returns_string():
    p1 = PipelineProfiler()
    p2 = PipelineProfiler()
    p1.record("a", 100.0)
    p2.record("a", 50.0)
    result = compare_profilers(p1, p2)
    assert isinstance(result, str)
    assert "a" in result


def test_compare_profilers_empty():
    p1 = PipelineProfiler()
    p2 = PipelineProfiler()
    result = compare_profilers(p1, p2)
    assert isinstance(result, str)
