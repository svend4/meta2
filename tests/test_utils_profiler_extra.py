"""Extra tests for puzzle_reconstruction/utils/profiler.py."""
from __future__ import annotations

import time
import pytest

from puzzle_reconstruction.utils.profiler import (
    StepProfile,
    PipelineProfiler,
    format_duration,
    profile_function,
    timed,
    compare_profilers,
)


# ─── StepProfile ──────────────────────────────────────────────────────────────

class TestStepProfileExtra:
    def test_elapsed_s(self):
        sp = StepProfile(name="s", elapsed_ms=500.0, n_calls=1)
        assert sp.elapsed_s == pytest.approx(0.5)

    def test_avg_ms_zero_calls(self):
        sp = StepProfile(name="s", elapsed_ms=0.0, n_calls=0)
        assert sp.avg_ms == pytest.approx(0.0)

    def test_avg_ms_computed(self):
        sp = StepProfile(name="s", elapsed_ms=300.0, n_calls=3)
        assert sp.avg_ms == pytest.approx(100.0)

    def test_repr_contains_name(self):
        sp = StepProfile(name="myStep", elapsed_ms=10.0, n_calls=1)
        assert "myStep" in repr(sp)


# ─── PipelineProfiler ─────────────────────────────────────────────────────────

class TestPipelineProfilerExtra:
    def test_record_and_get(self):
        p = PipelineProfiler()
        p.record("step1", 100.0)
        sp = p.get("step1")
        assert sp is not None and sp.elapsed_ms == pytest.approx(100.0)

    def test_get_missing_returns_none(self):
        p = PipelineProfiler()
        assert p.get("nonexistent") is None

    def test_n_calls_accumulated(self):
        p = PipelineProfiler()
        p.record("x", 10.0)
        p.record("x", 20.0)
        assert p.get("x").n_calls == 2
        assert p.get("x").elapsed_ms == pytest.approx(30.0)

    def test_total_elapsed_ms(self):
        p = PipelineProfiler()
        p.record("a", 100.0)
        p.record("b", 200.0)
        assert p.total_elapsed_ms() == pytest.approx(300.0)

    def test_slowest_step(self):
        p = PipelineProfiler()
        p.record("fast", 10.0)
        p.record("slow", 500.0)
        assert p.slowest_step() == "slow"

    def test_fastest_step(self):
        p = PipelineProfiler()
        p.record("fast", 10.0)
        p.record("slow", 500.0)
        assert p.fastest_step() == "fast"

    def test_slowest_none_empty(self):
        p = PipelineProfiler()
        assert p.slowest_step() is None

    def test_step_names_sorted_by_time(self):
        p = PipelineProfiler()
        p.record("quick", 10.0)
        p.record("heavy", 500.0)
        names = p.step_names()
        assert names[0] == "heavy"

    def test_reset_clears(self):
        p = PipelineProfiler()
        p.record("a", 100.0)
        p.reset()
        assert p.get("a") is None
        assert p.total_elapsed_ms() == pytest.approx(0.0)

    def test_start_stop(self):
        p = PipelineProfiler()
        p.start("s")
        time.sleep(0.001)
        elapsed = p.stop("s")
        assert elapsed > 0
        assert p.get("s").n_calls == 1

    def test_context_manager(self):
        p = PipelineProfiler()
        with p.step("ctx"):
            time.sleep(0.001)
        assert p.get("ctx").n_calls == 1

    def test_repr_contains_steps(self):
        p = PipelineProfiler()
        p.record("x", 50.0)
        assert "steps=1" in repr(p)

    def test_summary_table_returns_string(self):
        p = PipelineProfiler()
        p.record("step", 100.0)
        table = p.summary_table()
        assert isinstance(table, str) and "step" in table

    def test_profiles_sorted_desc(self):
        p = PipelineProfiler()
        p.record("fast", 10.0)
        p.record("slow", 300.0)
        profs = p.profiles()
        assert profs[0].name == "slow"


# ─── format_duration ──────────────────────────────────────────────────────────

class TestFormatDurationExtra:
    def test_sub_second(self):
        s = format_duration(250.0)
        assert "ms" in s

    def test_seconds(self):
        s = format_duration(2000.0)
        assert "s" in s and "m" not in s

    def test_minutes(self):
        s = format_duration(90_000.0)
        assert "m" in s

    def test_zero(self):
        s = format_duration(0.0)
        assert "0.00ms" in s


# ─── profile_function ─────────────────────────────────────────────────────────

class TestProfileFunctionExtra:
    def test_returns_result_and_profile(self):
        result, sp = profile_function(lambda: 42)
        assert result == 42
        assert isinstance(sp, StepProfile)

    def test_elapsed_positive(self):
        _, sp = profile_function(lambda: time.sleep(0.001))
        assert sp.elapsed_ms > 0

    def test_records_in_profiler(self):
        p = PipelineProfiler()
        profile_function(lambda: 1, profiler=p, step_name="fn")
        assert p.get("fn") is not None

    def test_step_name_default_is_function_name(self):
        def my_func():
            return 0
        _, sp = profile_function(my_func)
        assert sp.name == "my_func"


# ─── timed decorator ──────────────────────────────────────────────────────────

class TestTimedDecoratorExtra:
    def test_decorated_function_returns_value(self):
        p = PipelineProfiler()

        @timed(p, name="double")
        def double(x):
            return x * 2

        assert double(3) == 6

    def test_records_in_profiler(self):
        p = PipelineProfiler()

        @timed(p, name="work")
        def work():
            return None

        work()
        assert p.get("work") is not None
        assert p.get("work").n_calls == 1

    def test_name_defaults_to_function_name(self):
        p = PipelineProfiler()

        @timed(p)
        def compute():
            return 0

        compute()
        assert p.get("compute") is not None


# ─── compare_profilers ────────────────────────────────────────────────────────

class TestCompareProfilersExtra:
    def test_returns_string(self):
        p1 = PipelineProfiler()
        p2 = PipelineProfiler()
        p1.record("step", 100.0)
        p2.record("step", 200.0)
        result = compare_profilers(p1, p2)
        assert isinstance(result, str)

    def test_contains_step_name(self):
        p1 = PipelineProfiler()
        p2 = PipelineProfiler()
        p1.record("preprocess", 50.0)
        p2.record("preprocess", 80.0)
        result = compare_profilers(p1, p2)
        assert "preprocess" in result
