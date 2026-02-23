"""Extra tests for puzzle_reconstruction/utils/progress_tracker.py"""
import time
import pytest

from puzzle_reconstruction.utils.progress_tracker import (
    PipelineReport,
    ProgressTracker,
    StepRecord,
    make_tracker,
    run_step,
    summarize_tracker,
)


# ─── TestStepRecordExtra ──────────────────────────────────────────────────────

class TestStepRecordExtra:
    def test_status_running(self):
        r = StepRecord(name="s", status="running")
        assert r.status == "running"

    def test_status_skipped(self):
        r = StepRecord(name="s", status="skipped")
        assert r.status == "skipped"

    def test_status_failed(self):
        r = StepRecord(name="s", status="failed")
        assert r.status == "failed"

    def test_elapsed_large(self):
        t0 = 1000.0
        r = StepRecord(name="s", started_at=t0, ended_at=t0 + 100.0)
        assert r.elapsed == pytest.approx(100.0, abs=1e-6)

    def test_elapsed_very_small(self):
        t0 = time.time()
        r = StepRecord(name="s", started_at=t0, ended_at=t0 + 0.001)
        assert r.elapsed == pytest.approx(0.001, abs=1e-6)

    def test_is_complete_all_terminal(self):
        for st in ("done", "failed", "skipped"):
            r = StepRecord(name="s", status=st)
            assert r.is_complete is True

    def test_is_complete_non_terminal(self):
        for st in ("pending", "running"):
            r = StepRecord(name="s", status=st)
            assert r.is_complete is False

    def test_error_stored(self):
        r = StepRecord(name="s", error="something went wrong")
        assert r.error == "something went wrong"

    def test_meta_stored(self):
        r = StepRecord(name="s", meta={"k": 42, "v": "x"})
        assert r.meta["k"] == 42
        assert r.meta["v"] == "x"

    def test_long_name_valid(self):
        name = "preprocessing_step_01_document_cleaner"
        r = StepRecord(name=name)
        assert r.name == name


# ─── TestPipelineReportExtra ──────────────────────────────────────────────────

class TestPipelineReportExtra:
    def test_all_failed(self):
        r = PipelineReport(total_steps=5, failed=5)
        assert r.success_rate == pytest.approx(0.0)

    def test_all_skipped(self):
        r = PipelineReport(total_steps=4, skipped=4)
        assert r.success_rate == pytest.approx(0.0)

    def test_mixed_done_skipped(self):
        r = PipelineReport(total_steps=6, done=3, skipped=2, failed=1)
        assert r.success_rate == pytest.approx(3.0 / 6.0)

    def test_total_elapsed_stored(self):
        r = PipelineReport(total_elapsed=12.5)
        assert r.total_elapsed == pytest.approx(12.5)

    def test_pending_count_stored(self):
        r = PipelineReport(total_steps=5, pending=3)
        assert r.pending == 3

    def test_large_step_count(self):
        r = PipelineReport(total_steps=1000, done=800, failed=100, skipped=50)
        assert r.done == 800
        assert r.failed == 100


# ─── TestProgressTrackerCreationExtra ────────────────────────────────────────

class TestProgressTrackerCreationExtra:
    def test_long_name(self):
        t = ProgressTracker(name="my_long_pipeline_name_01")
        assert t.name == "my_long_pipeline_name_01"

    def test_underscore_in_name(self):
        t = ProgressTracker(name="pipe_1")
        assert t.name == "pipe_1"

    def test_steps_empty_initially(self):
        t = ProgressTracker()
        assert t.steps() == []

    def test_single_char_name(self):
        t = ProgressTracker(name="p")
        assert t.name == "p"


# ─── TestProgressTrackerRegisterExtra ────────────────────────────────────────

class TestProgressTrackerRegisterExtra:
    def test_ten_steps(self):
        t = ProgressTracker()
        for i in range(10):
            t.register(f"step_{i:02d}")
        assert len(t.steps()) == 10

    def test_order_ten_preserved(self):
        t = ProgressTracker()
        names = [f"step_{i}" for i in range(5)]
        for n in names:
            t.register(n)
        assert [r.name for r in t.steps()] == names

    def test_meta_multiple_keys(self):
        t = ProgressTracker()
        t.register("s", meta={"a": 1, "b": 2.0, "c": "x"})
        m = t.get_step("s").meta
        assert m["a"] == 1
        assert m["b"] == pytest.approx(2.0)
        assert m["c"] == "x"

    def test_different_names_all_registered(self):
        t = ProgressTracker()
        for name in ("alpha", "beta", "gamma", "delta"):
            t.register(name)
        names = [r.name for r in t.steps()]
        assert "alpha" in names
        assert "delta" in names


# ─── TestProgressTrackerTransitionsExtra ─────────────────────────────────────

class TestProgressTrackerTransitionsExtra:
    def _t(self, *steps):
        t = ProgressTracker()
        for s in steps:
            t.register(s)
        return t

    def test_done_without_start_sets_done(self):
        t = self._t("s")
        t.done("s")
        assert t.get_step("s").status == "done"

    def test_fail_without_start_sets_failed(self):
        t = self._t("s")
        t.fail("s")
        assert t.get_step("s").status == "failed"

    def test_skip_without_start_sets_skipped(self):
        t = self._t("s")
        t.skip("s")
        assert t.get_step("s").status == "skipped"

    def test_start_multiple_steps(self):
        t = self._t("a", "b", "c")
        for s in ("a", "b", "c"):
            t.start(s)
        for s in ("a", "b", "c"):
            assert t.get_step(s).status == "running"

    def test_done_then_fail_different_steps(self):
        t = self._t("a", "b")
        t.done("a")
        t.fail("b", error="oops")
        assert t.get_step("a").status == "done"
        assert t.get_step("b").status == "failed"
        assert t.get_step("b").error == "oops"

    def test_all_steps_skipped(self):
        t = self._t("a", "b", "c")
        for s in ("a", "b", "c"):
            t.skip(s)
        assert t.is_done() is True

    def test_elapsed_computed_after_start_done(self):
        t = self._t("s")
        t.start("s")
        time.sleep(0.01)
        t.done("s")
        assert t.get_step("s").elapsed is not None
        assert t.get_step("s").elapsed >= 0.0


# ─── TestProgressTrackerQueryExtra ───────────────────────────────────────────

class TestProgressTrackerQueryExtra:
    def _full(self, n=5):
        t = ProgressTracker()
        for i in range(n):
            t.register(f"step_{i}")
        return t

    def test_pending_count_all_five(self):
        t = self._full(5)
        assert len(t.pending_steps()) == 5

    def test_pending_drops_after_done(self):
        t = self._full(4)
        t.done("step_0")
        t.done("step_1")
        assert len(t.pending_steps()) == 2

    def test_failed_count_three(self):
        t = self._full(5)
        for s in ("step_1", "step_3", "step_4"):
            t.fail(s)
        assert len(t.failed_steps()) == 3

    def test_is_done_with_mix(self):
        t = self._full(3)
        t.done("step_0")
        t.fail("step_1")
        t.skip("step_2")
        assert t.is_done() is True

    def test_progress_two_thirds(self):
        t = self._full(3)
        t.done("step_0")
        t.done("step_1")
        assert t.progress() == pytest.approx(2.0 / 3.0)

    def test_empty_tracker_progress_zero(self):
        t = ProgressTracker()
        assert t.progress() == pytest.approx(0.0)

    def test_failed_steps_names(self):
        t = self._full(3)
        t.fail("step_2")
        failed = t.failed_steps()
        assert failed[0].name == "step_2"


# ─── TestProgressTrackerReportExtra ──────────────────────────────────────────

class TestProgressTrackerReportExtra:
    def test_report_all_done(self):
        t = make_tracker(steps=["a", "b", "c"])
        for s in ("a", "b", "c"):
            t.done(s)
        r = t.report()
        assert r.done == 3
        assert r.failed == 0
        assert r.skipped == 0

    def test_report_mix(self):
        t = make_tracker(steps=["a", "b", "c", "d"])
        t.done("a")
        t.fail("b")
        t.skip("c")
        r = t.report()
        assert r.done == 1
        assert r.failed == 1
        assert r.skipped == 1

    def test_reset_clears_all(self):
        t = make_tracker(steps=["a", "b", "c"])
        t.done("a")
        t.fail("b")
        t.skip("c")
        t.reset()
        r = t.report()
        assert r.done == 0
        assert r.failed == 0
        assert r.skipped == 0

    def test_total_steps_large(self):
        steps = [f"s_{i}" for i in range(10)]
        t = make_tracker(steps=steps)
        r = t.report()
        assert r.total_steps == 10

    def test_report_is_pipeline_report(self):
        t = make_tracker(steps=["a"])
        assert isinstance(t.report(), PipelineReport)


# ─── TestMakeTrackerExtra ─────────────────────────────────────────────────────

class TestMakeTrackerExtra:
    def test_ten_steps(self):
        steps = [f"step_{i}" for i in range(10)]
        t = make_tracker(steps=steps)
        assert len(t.steps()) == 10

    def test_name_with_underscore(self):
        t = make_tracker(name="my_pipe_01")
        assert t.name == "my_pipe_01"

    def test_callback_called_on_fail(self):
        called = []
        t = make_tracker(steps=["s"], callback=lambda r: called.append(r.status))
        t.fail("s")
        assert "failed" in called

    def test_callback_called_on_skip(self):
        called = []
        t = make_tracker(steps=["s"], callback=lambda r: called.append(r.name))
        t.skip("s")
        assert "s" in called

    def test_default_name_is_pipeline(self):
        t = make_tracker()
        assert t.name == "pipeline"


# ─── TestRunStepExtra ─────────────────────────────────────────────────────────

class TestRunStepExtra:
    def test_return_value_string(self):
        t = make_tracker(steps=["s"])
        result = run_step(t, "s", lambda: "hello")
        assert result == "hello"

    def test_return_value_none(self):
        t = make_tracker(steps=["s"])
        result = run_step(t, "s", lambda: None)
        assert result is None

    def test_exception_type_stored(self):
        t = make_tracker(steps=["s"])
        with pytest.raises(TypeError):
            run_step(t, "s", lambda: (_ for _ in ()).throw(TypeError("wrong type")))
        assert "wrong type" in t.get_step("s").error

    def test_multiple_steps_sequential(self):
        t = make_tracker(steps=["a", "b", "c"])
        run_step(t, "a", lambda: 1)
        run_step(t, "b", lambda: 2)
        run_step(t, "c", lambda: 3)
        for s in ("a", "b", "c"):
            assert t.get_step(s).status == "done"

    def test_complex_fn_with_kwargs(self):
        t = make_tracker(steps=["s"])
        result = run_step(t, "s", lambda a, b, c=0: a + b + c, 1, 2, c=3)
        assert result == 6

    def test_elapsed_recorded(self):
        t = make_tracker(steps=["s"])
        run_step(t, "s", time.sleep, 0.01)
        assert t.get_step("s").elapsed is not None
        assert t.get_step("s").elapsed >= 0.0


# ─── TestSummarizeTrackerExtra ────────────────────────────────────────────────

class TestSummarizeTrackerExtra:
    def test_progress_after_partial_done(self):
        t = make_tracker(steps=["a", "b", "c", "d"])
        t.done("a")
        t.done("b")
        s = summarize_tracker(t)
        assert s["progress"] == pytest.approx(0.5)

    def test_done_count_in_summary(self):
        t = make_tracker(steps=["a", "b", "c"])
        t.done("a")
        t.done("b")
        s = summarize_tracker(t)
        assert s["done"] == 2

    def test_failed_count_in_summary(self):
        t = make_tracker(steps=["a", "b"])
        t.fail("a")
        s = summarize_tracker(t)
        assert s["failed"] == 1

    def test_skipped_count_in_summary(self):
        t = make_tracker(steps=["a", "b", "c"])
        t.skip("b")
        s = summarize_tracker(t)
        assert s["skipped"] == 1

    def test_pending_count_in_summary(self):
        t = make_tracker(steps=["a", "b", "c"])
        t.done("a")
        s = summarize_tracker(t)
        assert s["pending"] == 2

    def test_total_in_summary(self):
        t = make_tracker(steps=["a", "b", "c", "d", "e"])
        s = summarize_tracker(t)
        assert s["total"] == 5

    def test_elapsed_nonneg(self):
        t = make_tracker(steps=["a"])
        run_step(t, "a", lambda: None)
        s = summarize_tracker(t)
        assert s["elapsed"] >= 0.0
