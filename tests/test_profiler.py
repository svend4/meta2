"""
Тесты для puzzle_reconstruction/utils/profiler.py

Покрытие:
    StepProfile    — elapsed_s, avg_ms, repr, children, defaults
    PipelineProfiler:
        start/stop — запись elapsed_ms > 0, накопление n_calls
        record     — накопление, memory_mb = max
        step()     — context manager, без ошибок при вложении
        reset      — очистка профилей и активных таймеров
        profiles() — сортировка по убыванию elapsed_ms
        total_elapsed_ms — сумма всех шагов
        slowest/fastest_step — корректность
        get        — по имени / None для неизвестных
        step_names — список имён в порядке убывания
        summary_table — строка, заголовок, TOTAL, имена шагов
        repr       — содержит steps= и total=
    format_duration — 0.3ms, 1500ms, 90000ms
    profile_function — возвращает (result, StepProfile), n_calls=1, записывает в profiler
    timed()       — декоратор, functools.wraps, записывает в profiler
    compare_profilers — строка, содержит оба label, имена шагов
"""
import math
import time

import pytest

from puzzle_reconstruction.utils.profiler import (
    PipelineProfiler,
    StepProfile,
    compare_profilers,
    format_duration,
    profile_function,
    timed,
)


# ─── StepProfile ──────────────────────────────────────────────────────────────

class TestStepProfile:
    def test_elapsed_s(self):
        sp = StepProfile(name="x", elapsed_ms=2500.0, n_calls=1)
        assert sp.elapsed_s == pytest.approx(2.5)

    def test_avg_ms_one_call(self):
        sp = StepProfile(name="x", elapsed_ms=120.0, n_calls=1)
        assert sp.avg_ms == pytest.approx(120.0)

    def test_avg_ms_multiple_calls(self):
        sp = StepProfile(name="x", elapsed_ms=300.0, n_calls=3)
        assert sp.avg_ms == pytest.approx(100.0)

    def test_avg_ms_zero_calls(self):
        sp = StepProfile(name="x", elapsed_ms=0.0, n_calls=0)
        assert sp.avg_ms == pytest.approx(0.0)

    def test_repr_contains_name(self):
        sp = StepProfile(name="matching", elapsed_ms=50.0, n_calls=2)
        assert "matching" in repr(sp)

    def test_repr_contains_elapsed(self):
        sp = StepProfile(name="x", elapsed_ms=50.25, n_calls=1)
        assert "50.25" in repr(sp)

    def test_repr_contains_n_calls(self):
        sp = StepProfile(name="x", elapsed_ms=10.0, n_calls=5)
        assert "5" in repr(sp)

    def test_defaults(self):
        sp = StepProfile(name="step")
        assert sp.elapsed_ms == pytest.approx(0.0)
        assert sp.n_calls    == 0
        assert sp.memory_mb  == pytest.approx(0.0)
        assert sp.children   == []


# ─── PipelineProfiler — start/stop/record ─────────────────────────────────────

class TestPipelineProfilerBasic:
    def test_start_stop_records(self):
        p = PipelineProfiler()
        p.start("step_a")
        time.sleep(0.005)
        elapsed = p.stop("step_a")
        assert elapsed > 0.0
        assert p.get("step_a") is not None
        assert p.get("step_a").elapsed_ms == pytest.approx(elapsed, rel=0.1)

    def test_start_stop_n_calls(self):
        p = PipelineProfiler()
        for _ in range(3):
            p.start("s")
            p.stop("s")
        assert p.get("s").n_calls == 3

    def test_record_accumulates(self):
        p = PipelineProfiler()
        p.record("x", 100.0)
        p.record("x", 200.0)
        assert p.get("x").elapsed_ms == pytest.approx(300.0)
        assert p.get("x").n_calls    == 2

    def test_record_memory_max(self):
        p = PipelineProfiler()
        p.record("x", 10.0, memory_mb=5.0)
        p.record("x", 10.0, memory_mb=3.0)
        assert p.get("x").memory_mb == pytest.approx(5.0)

    def test_stop_missing_raises(self):
        p = PipelineProfiler()
        with pytest.raises(KeyError):
            p.stop("nonexistent")

    def test_get_unknown_returns_none(self):
        p = PipelineProfiler()
        assert p.get("xyz") is None

    def test_reset_clears(self):
        p = PipelineProfiler()
        p.record("a", 50.0)
        p.reset()
        assert p.get("a") is None
        assert p.total_elapsed_ms() == pytest.approx(0.0)


# ─── PipelineProfiler — step() context manager ────────────────────────────────

class TestPipelineProfilerStep:
    def test_step_records(self):
        p = PipelineProfiler()
        with p.step("process"):
            time.sleep(0.002)
        sp = p.get("process")
        assert sp is not None
        assert sp.elapsed_ms > 0.0

    def test_step_n_calls(self):
        p = PipelineProfiler()
        for _ in range(4):
            with p.step("loop"):
                pass
        assert p.get("loop").n_calls == 4

    def test_step_exception_still_records(self):
        """Исключение внутри step() не нарушает запись профиля."""
        p = PipelineProfiler()
        try:
            with p.step("failing"):
                raise ValueError("oops")
        except ValueError:
            pass
        # Профиль должен быть записан даже при ошибке
        sp = p.get("failing")
        assert sp is not None

    def test_step_no_memory_tracking_by_default(self):
        p = PipelineProfiler(track_memory=False)
        with p.step("x"):
            _ = list(range(1000))
        assert p.get("x").memory_mb == pytest.approx(0.0)

    def test_step_multiple_different(self):
        p = PipelineProfiler()
        with p.step("a"):
            pass
        with p.step("b"):
            pass
        assert p.get("a") is not None
        assert p.get("b") is not None


# ─── PipelineProfiler — аналитика ─────────────────────────────────────────────

class TestPipelineProfilerAnalytics:
    def _filled(self) -> PipelineProfiler:
        p = PipelineProfiler()
        p.record("fast",   10.0)
        p.record("medium", 50.0)
        p.record("slow",  200.0)
        return p

    def test_profiles_sorted_descending(self):
        p      = self._filled()
        profs  = p.profiles()
        times  = [sp.elapsed_ms for sp in profs]
        assert times == sorted(times, reverse=True)

    def test_total_elapsed_ms(self):
        p = self._filled()
        assert p.total_elapsed_ms() == pytest.approx(260.0)

    def test_total_elapsed_empty(self):
        p = PipelineProfiler()
        assert p.total_elapsed_ms() == pytest.approx(0.0)

    def test_slowest_step(self):
        p = self._filled()
        assert p.slowest_step() == "slow"

    def test_fastest_step(self):
        p = self._filled()
        assert p.fastest_step() == "fast"

    def test_slowest_empty_none(self):
        p = PipelineProfiler()
        assert p.slowest_step() is None

    def test_fastest_empty_none(self):
        p = PipelineProfiler()
        assert p.fastest_step() is None

    def test_step_names_sorted(self):
        p     = self._filled()
        names = p.step_names()
        assert names[0] == "slow"
        assert names[-1] == "fast"

    def test_repr_contains_steps(self):
        p = self._filled()
        assert "steps=3" in repr(p)

    def test_repr_contains_total(self):
        p = self._filled()
        assert "total=" in repr(p)

    def test_summary_table_string(self):
        p     = self._filled()
        table = p.summary_table()
        assert isinstance(table, str)

    def test_summary_table_has_header(self):
        p     = self._filled()
        table = p.summary_table()
        assert "Step" in table
        assert "Calls" in table

    def test_summary_table_has_total(self):
        p     = self._filled()
        table = p.summary_table()
        assert "TOTAL" in table

    def test_summary_table_has_step_names(self):
        p     = self._filled()
        table = p.summary_table()
        assert "slow"   in table
        assert "medium" in table
        assert "fast"   in table

    def test_summary_table_top_n(self):
        p     = self._filled()
        table = p.summary_table(top_n=2)
        # Только 2 самых медленных: slow + medium
        assert "slow"   in table
        assert "medium" in table

    def test_summary_table_empty(self):
        p     = PipelineProfiler()
        table = p.summary_table()
        assert isinstance(table, str)


# ─── format_duration ──────────────────────────────────────────────────────────

class TestFormatDuration:
    def test_sub_second_ms(self):
        assert "ms" in format_duration(0.3)
        assert "ms" in format_duration(999.0)

    def test_seconds(self):
        s = format_duration(1500.0)
        assert "s" in s
        assert "1.50" in s

    def test_minutes(self):
        s = format_duration(90000.0)
        assert "m" in s
        assert "30" in s

    def test_zero(self):
        s = format_duration(0.0)
        assert "0.00ms" in s

    def test_exact_second(self):
        s = format_duration(1000.0)
        assert "1.00s" in s

    def test_returns_string(self):
        assert isinstance(format_duration(42.5), str)


# ─── profile_function ─────────────────────────────────────────────────────────

class TestProfileFunction:
    def test_returns_tuple(self):
        result, sp = profile_function(lambda: 42)
        assert result == 42
        assert isinstance(sp, StepProfile)

    def test_elapsed_ms_positive(self):
        _, sp = profile_function(time.sleep, 0.005)
        assert sp.elapsed_ms > 0.0

    def test_n_calls_one(self):
        _, sp = profile_function(lambda: None)
        assert sp.n_calls == 1

    def test_records_to_profiler(self):
        p = PipelineProfiler()
        profile_function(lambda: None, profiler=p, step_name="fn_step")
        assert p.get("fn_step") is not None

    def test_step_name_default(self):
        def my_fn():
            return "ok"
        _, sp = profile_function(my_fn)
        assert sp.name == "my_fn"

    def test_step_name_custom(self):
        _, sp = profile_function(lambda: None, step_name="custom")
        assert sp.name == "custom"

    def test_args_passed(self):
        result, _ = profile_function(max, 3, 7)
        assert result == 7

    def test_kwargs_passed(self):
        def fn(x, y=10):
            return x + y
        result, _ = profile_function(fn, 5, y=20)
        assert result == 25


# ─── timed() декоратор ────────────────────────────────────────────────────────

class TestTimedDecorator:
    def test_records_to_profiler(self):
        p = PipelineProfiler()

        @timed(p)
        def my_func():
            return 99

        my_func()
        assert p.get("my_func") is not None

    def test_return_value_preserved(self):
        p = PipelineProfiler()

        @timed(p)
        def add(a, b):
            return a + b

        assert add(3, 4) == 7

    def test_custom_name(self):
        p = PipelineProfiler()

        @timed(p, name="custom_step")
        def fn():
            pass

        fn()
        assert p.get("custom_step") is not None

    def test_n_calls_increments(self):
        p = PipelineProfiler()

        @timed(p)
        def step():
            pass

        step()
        step()
        step()
        assert p.get("step").n_calls == 3

    def test_functools_wraps(self):
        p = PipelineProfiler()

        @timed(p)
        def documented_fn():
            """My docstring."""
            pass

        assert documented_fn.__name__ == "documented_fn"

    def test_elapsed_ms_positive(self):
        p = PipelineProfiler()

        @timed(p)
        def slow():
            time.sleep(0.003)

        slow()
        assert p.get("slow").elapsed_ms > 0.0


# ─── compare_profilers ────────────────────────────────────────────────────────

class TestCompareProfilers:
    def test_returns_string(self):
        p1 = PipelineProfiler()
        p2 = PipelineProfiler()
        p1.record("a", 100.0)
        p2.record("a", 80.0)
        s  = compare_profilers(p1, p2)
        assert isinstance(s, str)

    def test_contains_labels(self):
        p1 = PipelineProfiler()
        p2 = PipelineProfiler()
        s  = compare_profilers(p1, p2, label1="baseline", label2="optimized")
        assert "baseline"  in s
        assert "optimized" in s

    def test_contains_step_names(self):
        p1 = PipelineProfiler()
        p2 = PipelineProfiler()
        p1.record("matching", 500.0)
        p2.record("matching", 300.0)
        s  = compare_profilers(p1, p2)
        assert "matching" in s

    def test_both_have_different_steps(self):
        p1 = PipelineProfiler()
        p2 = PipelineProfiler()
        p1.record("step_a", 100.0)
        p2.record("step_b", 200.0)
        s  = compare_profilers(p1, p2)
        assert "step_a" in s
        assert "step_b" in s

    def test_empty_profilers(self):
        p1 = PipelineProfiler()
        p2 = PipelineProfiler()
        s  = compare_profilers(p1, p2)
        assert isinstance(s, str)
