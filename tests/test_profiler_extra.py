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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _profile(name="step", elapsed_ms=100.0, n_calls=1) -> StepProfile:
    return StepProfile(name=name, elapsed_ms=elapsed_ms, n_calls=n_calls)


def _profiler_with(*steps) -> PipelineProfiler:
    """Build a PipelineProfiler with pre-recorded steps: (name, ms) pairs."""
    p = PipelineProfiler()
    for name, ms in steps:
        p.record(name, ms)
    return p


# ─── StepProfile ──────────────────────────────────────────────────────────────

class TestStepProfileExtra:
    def test_name_stored(self):
        sp = _profile(name="preprocess")
        assert sp.name == "preprocess"

    def test_elapsed_ms_stored(self):
        sp = _profile(elapsed_ms=250.0)
        assert sp.elapsed_ms == pytest.approx(250.0)

    def test_n_calls_stored(self):
        sp = _profile(n_calls=5)
        assert sp.n_calls == 5

    def test_memory_mb_default(self):
        sp = _profile()
        assert sp.memory_mb == pytest.approx(0.0)

    def test_children_default_empty(self):
        assert _profile().children == []

    def test_elapsed_s_conversion(self):
        sp = _profile(elapsed_ms=2000.0)
        assert sp.elapsed_s == pytest.approx(2.0)

    def test_avg_ms_single_call(self):
        sp = _profile(elapsed_ms=300.0, n_calls=1)
        assert sp.avg_ms == pytest.approx(300.0)

    def test_avg_ms_multiple_calls(self):
        sp = _profile(elapsed_ms=600.0, n_calls=3)
        assert sp.avg_ms == pytest.approx(200.0)

    def test_avg_ms_zero_calls(self):
        sp = StepProfile(name="x", elapsed_ms=0.0, n_calls=0)
        assert sp.avg_ms == pytest.approx(0.0)

    def test_repr_contains_name(self):
        sp = _profile(name="matching")
        assert "matching" in repr(sp)


# ─── PipelineProfiler ─────────────────────────────────────────────────────────

class TestPipelineProfilerExtra:
    def test_initial_no_profiles(self):
        p = PipelineProfiler()
        assert len(p.profiles()) == 0

    def test_record_creates_profile(self):
        p = PipelineProfiler()
        p.record("step1", 100.0)
        assert p.get("step1") is not None

    def test_record_accumulates(self):
        p = PipelineProfiler()
        p.record("s", 100.0)
        p.record("s", 200.0)
        assert p.get("s").elapsed_ms == pytest.approx(300.0)
        assert p.get("s").n_calls == 2

    def test_record_max_memory(self):
        p = PipelineProfiler()
        p.record("s", 100.0, memory_mb=5.0)
        p.record("s", 100.0, memory_mb=3.0)
        assert p.get("s").memory_mb == pytest.approx(5.0)

    def test_start_stop(self):
        p = PipelineProfiler()
        p.start("work")
        time.sleep(0.001)
        elapsed = p.stop("work")
        assert elapsed > 0.0
        assert p.get("work") is not None

    def test_stop_unknown_raises(self):
        p = PipelineProfiler()
        with pytest.raises(KeyError):
            p.stop("nonexistent")

    def test_step_context_manager(self):
        p = PipelineProfiler()
        with p.step("ctx"):
            pass
        assert p.get("ctx") is not None

    def test_step_elapsed_positive(self):
        p = PipelineProfiler()
        with p.step("s"):
            time.sleep(0.001)
        assert p.get("s").elapsed_ms > 0.0

    def test_reset_clears(self):
        p = _profiler_with(("a", 100.0), ("b", 200.0))
        p.reset()
        assert len(p.profiles()) == 0

    def test_total_elapsed_ms(self):
        p = _profiler_with(("a", 100.0), ("b", 200.0))
        assert p.total_elapsed_ms() == pytest.approx(300.0)

    def test_total_elapsed_ms_empty(self):
        assert PipelineProfiler().total_elapsed_ms() == pytest.approx(0.0)

    def test_slowest_step(self):
        p = _profiler_with(("fast", 50.0), ("slow", 500.0))
        assert p.slowest_step() == "slow"

    def test_fastest_step(self):
        p = _profiler_with(("fast", 50.0), ("slow", 500.0))
        assert p.fastest_step() == "fast"

    def test_slowest_step_empty(self):
        assert PipelineProfiler().slowest_step() is None

    def test_fastest_step_empty(self):
        assert PipelineProfiler().fastest_step() is None

    def test_get_existing(self):
        p = _profiler_with(("s", 100.0))
        assert isinstance(p.get("s"), StepProfile)

    def test_get_missing_none(self):
        assert PipelineProfiler().get("missing") is None

    def test_step_names_sorted_by_elapsed(self):
        p = _profiler_with(("fast", 50.0), ("slow", 500.0))
        names = p.step_names()
        assert names[0] == "slow"

    def test_profiles_sorted_descending(self):
        p = _profiler_with(("a", 100.0), ("b", 400.0), ("c", 200.0))
        elapseds = [sp.elapsed_ms for sp in p.profiles()]
        assert elapseds == sorted(elapseds, reverse=True)

    def test_summary_table_returns_string(self):
        p = _profiler_with(("s", 100.0))
        assert isinstance(p.summary_table(), str)

    def test_summary_table_contains_step_name(self):
        p = _profiler_with(("matching", 100.0))
        assert "matching" in p.summary_table()

    def test_summary_table_top_n(self):
        p = _profiler_with(("slowstep", 300.0), ("midstep", 200.0), ("zzzfastest", 100.0))
        table = p.summary_table(top_n=2)
        assert "zzzfastest" not in table  # fastest step excluded from top_n=2

    def test_repr_contains_steps(self):
        p = _profiler_with(("x", 100.0))
        assert "steps=1" in repr(p)


# ─── format_duration ──────────────────────────────────────────────────────────

class TestFormatDurationExtra:
    def test_returns_string(self):
        assert isinstance(format_duration(100.0), str)

    def test_sub_second(self):
        s = format_duration(500.0)
        assert "ms" in s

    def test_sub_second_value(self):
        s = format_duration(0.3)
        assert "0.30ms" == s

    def test_seconds_range(self):
        s = format_duration(1500.0)
        assert "s" in s
        assert "1.50s" == s

    def test_minutes_range(self):
        s = format_duration(90000.0)
        assert "m" in s
        assert "1m" in s

    def test_exact_second(self):
        s = format_duration(1000.0)
        assert "1.00s" == s

    def test_large_ms(self):
        s = format_duration(999.0)
        assert "ms" in s


# ─── profile_function ─────────────────────────────────────────────────────────

class TestProfileFunctionExtra:
    def test_returns_tuple(self):
        result, sp = profile_function(lambda: 42)
        assert result == 42
        assert isinstance(sp, StepProfile)

    def test_elapsed_positive(self):
        _, sp = profile_function(time.sleep, 0.001)
        assert sp.elapsed_ms > 0.0

    def test_n_calls_one(self):
        _, sp = profile_function(lambda: None)
        assert sp.n_calls == 1

    def test_step_name_default_uses_fn_name(self):
        def my_fn():
            return 1
        _, sp = profile_function(my_fn)
        assert sp.name == "my_fn"

    def test_custom_step_name(self):
        _, sp = profile_function(lambda: 1, step_name="custom")
        assert sp.name == "custom"

    def test_records_to_profiler(self):
        p = PipelineProfiler()
        profile_function(lambda: 1, profiler=p, step_name="fn")
        assert p.get("fn") is not None

    def test_kwargs_passed(self):
        def add(a, b):
            return a + b
        result, _ = profile_function(add, 3, b=4)
        assert result == 7


# ─── timed decorator ──────────────────────────────────────────────────────────

class TestTimedDecoratorExtra:
    def test_returns_same_result(self):
        p = PipelineProfiler()

        @timed(p)
        def double(x):
            return x * 2

        assert double(5) == 10

    def test_records_elapsed(self):
        p = PipelineProfiler()

        @timed(p)
        def noop():
            pass

        noop()
        assert p.get("noop") is not None
        assert p.get("noop").elapsed_ms >= 0.0

    def test_n_calls_accumulates(self):
        p = PipelineProfiler()

        @timed(p)
        def work():
            pass

        work()
        work()
        assert p.get("work").n_calls == 2

    def test_custom_name(self):
        p = PipelineProfiler()

        @timed(p, name="custom_step")
        def fn():
            pass

        fn()
        assert p.get("custom_step") is not None

    def test_wraps_preserves_name(self):
        p = PipelineProfiler()

        @timed(p)
        def my_function():
            pass

        assert my_function.__name__ == "my_function"


# ─── compare_profilers ────────────────────────────────────────────────────────

class TestCompareProfilersExtra:
    def test_returns_string(self):
        p1 = _profiler_with(("s", 100.0))
        p2 = _profiler_with(("s", 200.0))
        assert isinstance(compare_profilers(p1, p2), str)

    def test_contains_step_name(self):
        p1 = _profiler_with(("matching", 100.0))
        p2 = _profiler_with(("matching", 200.0))
        assert "matching" in compare_profilers(p1, p2)

    def test_contains_labels(self):
        p1 = _profiler_with(("s", 100.0))
        p2 = _profiler_with(("s", 200.0))
        result = compare_profilers(p1, p2, label1="run_A", label2="run_B")
        assert "run_A" in result
        assert "run_B" in result

    def test_handles_missing_step_in_p2(self):
        p1 = _profiler_with(("only_in_p1", 100.0))
        p2 = PipelineProfiler()
        result = compare_profilers(p1, p2)
        assert "only_in_p1" in result

    def test_handles_missing_step_in_p1(self):
        p1 = PipelineProfiler()
        p2 = _profiler_with(("only_in_p2", 100.0))
        result = compare_profilers(p1, p2)
        assert "only_in_p2" in result

    def test_both_empty(self):
        result = compare_profilers(PipelineProfiler(), PipelineProfiler())
        assert isinstance(result, str)
