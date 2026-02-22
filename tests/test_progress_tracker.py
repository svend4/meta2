"""Тесты для puzzle_reconstruction.utils.progress_tracker."""
import time
import pytest

from puzzle_reconstruction.utils.progress_tracker import (
    StepRecord,
    PipelineReport,
    ProgressTracker,
    make_tracker,
    run_step,
    summarize_tracker,
)


# ─── TestStepRecord ───────────────────────────────────────────────────────────

class TestStepRecord:
    def test_basic_creation(self):
        r = StepRecord(name="step1")
        assert r.name == "step1"
        assert r.status == "pending"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            StepRecord(name="")

    def test_whitespace_name_raises(self):
        with pytest.raises(ValueError):
            StepRecord(name="   ")

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError):
            StepRecord(name="s", status="unknown")

    def test_elapsed_none_when_not_started(self):
        r = StepRecord(name="s")
        assert r.elapsed is None

    def test_elapsed_none_when_only_started(self):
        r = StepRecord(name="s", started_at=time.time())
        assert r.elapsed is None

    def test_elapsed_computed(self):
        t0 = time.time()
        r = StepRecord(name="s", started_at=t0, ended_at=t0 + 1.5)
        assert r.elapsed == pytest.approx(1.5, abs=1e-6)

    def test_is_complete_pending(self):
        r = StepRecord(name="s", status="pending")
        assert r.is_complete is False

    def test_is_complete_running(self):
        r = StepRecord(name="s", status="running")
        assert r.is_complete is False

    def test_is_complete_done(self):
        r = StepRecord(name="s", status="done")
        assert r.is_complete is True

    def test_is_complete_failed(self):
        r = StepRecord(name="s", status="failed")
        assert r.is_complete is True

    def test_is_complete_skipped(self):
        r = StepRecord(name="s", status="skipped")
        assert r.is_complete is True


# ─── TestPipelineReport ───────────────────────────────────────────────────────

class TestPipelineReport:
    def test_defaults(self):
        r = PipelineReport()
        assert r.total_steps == 0
        assert r.total_elapsed == pytest.approx(0.0)

    def test_negative_done_raises(self):
        with pytest.raises(ValueError):
            PipelineReport(done=-1)

    def test_negative_elapsed_raises(self):
        with pytest.raises(ValueError):
            PipelineReport(total_elapsed=-0.1)

    def test_success_rate_zero_when_empty(self):
        r = PipelineReport()
        assert r.success_rate == pytest.approx(0.0)

    def test_success_rate_all_done(self):
        r = PipelineReport(total_steps=3, done=3)
        assert r.success_rate == pytest.approx(1.0)

    def test_success_rate_mixed(self):
        r = PipelineReport(total_steps=4, done=2, failed=2)
        assert r.success_rate == pytest.approx(0.5)

    def test_success_rate_skipped_counts_as_completed(self):
        r = PipelineReport(total_steps=3, done=1, skipped=2)
        assert r.success_rate == pytest.approx(1.0 / 3.0)


# ─── TestProgressTrackerCreation ──────────────────────────────────────────────

class TestProgressTrackerCreation:
    def test_default_name(self):
        t = ProgressTracker()
        assert t.name == "pipeline"

    def test_custom_name(self):
        t = ProgressTracker(name="my_pipeline")
        assert t.name == "my_pipeline"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            ProgressTracker(name="")

    def test_whitespace_name_raises(self):
        with pytest.raises(ValueError):
            ProgressTracker(name="  ")

    def test_initial_state_empty(self):
        t = ProgressTracker()
        assert t.steps() == []


# ─── TestProgressTrackerRegister ──────────────────────────────────────────────

class TestProgressTrackerRegister:
    def test_register_step(self):
        t = ProgressTracker()
        t.register("step1")
        assert len(t.steps()) == 1

    def test_duplicate_raises(self):
        t = ProgressTracker()
        t.register("step1")
        with pytest.raises(ValueError):
            t.register("step1")

    def test_empty_step_name_raises(self):
        t = ProgressTracker()
        with pytest.raises(ValueError):
            t.register("")

    def test_multiple_steps(self):
        t = ProgressTracker()
        t.register("a")
        t.register("b")
        t.register("c")
        assert len(t.steps()) == 3

    def test_order_preserved(self):
        t = ProgressTracker()
        t.register("a")
        t.register("b")
        names = [r.name for r in t.steps()]
        assert names == ["a", "b"]

    def test_meta_stored(self):
        t = ProgressTracker()
        t.register("s", meta={"key": "val"})
        assert t.get_step("s").meta["key"] == "val"


# ─── TestProgressTrackerTransitions ───────────────────────────────────────────

class TestProgressTrackerTransitions:
    def _tracker(self, *steps):
        t = ProgressTracker()
        for s in steps:
            t.register(s)
        return t

    def test_start_sets_running(self):
        t = self._tracker("s")
        t.start("s")
        assert t.get_step("s").status == "running"

    def test_start_sets_started_at(self):
        t = self._tracker("s")
        t.start("s")
        assert t.get_step("s").started_at is not None

    def test_done_sets_done(self):
        t = self._tracker("s")
        t.start("s")
        t.done("s")
        assert t.get_step("s").status == "done"

    def test_done_sets_ended_at(self):
        t = self._tracker("s")
        t.start("s")
        t.done("s")
        assert t.get_step("s").ended_at is not None

    def test_fail_sets_failed(self):
        t = self._tracker("s")
        t.start("s")
        t.fail("s", error="boom")
        assert t.get_step("s").status == "failed"

    def test_fail_stores_error(self):
        t = self._tracker("s")
        t.fail("s", error="boom")
        assert t.get_step("s").error == "boom"

    def test_skip_sets_skipped(self):
        t = self._tracker("s")
        t.skip("s")
        assert t.get_step("s").status == "skipped"

    def test_unregistered_start_raises(self):
        t = ProgressTracker()
        with pytest.raises(KeyError):
            t.start("missing")


# ─── TestProgressTrackerQuery ─────────────────────────────────────────────────

class TestProgressTrackerQuery:
    def _tracker_with_steps(self):
        t = ProgressTracker()
        for s in ("a", "b", "c"):
            t.register(s)
        return t

    def test_get_step_returns_record(self):
        t = self._tracker_with_steps()
        r = t.get_step("a")
        assert isinstance(r, StepRecord)

    def test_get_step_missing_raises(self):
        t = ProgressTracker()
        with pytest.raises(KeyError):
            t.get_step("x")

    def test_pending_steps_all_initially(self):
        t = self._tracker_with_steps()
        assert len(t.pending_steps()) == 3

    def test_pending_steps_decreases_after_done(self):
        t = self._tracker_with_steps()
        t.done("a")
        assert len(t.pending_steps()) == 2

    def test_failed_steps_empty_initially(self):
        t = self._tracker_with_steps()
        assert t.failed_steps() == []

    def test_failed_steps_after_fail(self):
        t = self._tracker_with_steps()
        t.fail("b")
        assert len(t.failed_steps()) == 1

    def test_is_done_false_initially(self):
        t = self._tracker_with_steps()
        assert t.is_done() is False

    def test_is_done_true_when_all_complete(self):
        t = self._tracker_with_steps()
        t.done("a")
        t.done("b")
        t.skip("c")
        assert t.is_done() is True

    def test_progress_zero_initially(self):
        t = self._tracker_with_steps()
        assert t.progress() == pytest.approx(0.0)

    def test_progress_partial(self):
        t = self._tracker_with_steps()
        t.done("a")
        assert t.progress() == pytest.approx(1.0 / 3.0)

    def test_progress_one_when_all_done(self):
        t = self._tracker_with_steps()
        for s in ("a", "b", "c"):
            t.done(s)
        assert t.progress() == pytest.approx(1.0)


# ─── TestProgressTrackerReport ────────────────────────────────────────────────

class TestProgressTrackerReport:
    def test_returns_pipeline_report(self):
        t = ProgressTracker()
        assert isinstance(t.report(), PipelineReport)

    def test_total_steps(self):
        t = make_tracker(steps=["a", "b", "c"])
        r = t.report()
        assert r.total_steps == 3

    def test_done_count(self):
        t = make_tracker(steps=["a", "b"])
        t.done("a")
        r = t.report()
        assert r.done == 1

    def test_failed_count(self):
        t = make_tracker(steps=["a", "b"])
        t.fail("a")
        r = t.report()
        assert r.failed == 1

    def test_skipped_count(self):
        t = make_tracker(steps=["a", "b"])
        t.skip("a")
        r = t.report()
        assert r.skipped == 1

    def test_reset_restores_pending(self):
        t = make_tracker(steps=["a", "b"])
        t.done("a")
        t.fail("b")
        t.reset()
        for r in t.steps():
            assert r.status == "pending"


# ─── TestMakeTracker ──────────────────────────────────────────────────────────

class TestMakeTracker:
    def test_returns_tracker(self):
        t = make_tracker()
        assert isinstance(t, ProgressTracker)

    def test_steps_registered(self):
        t = make_tracker(steps=["x", "y", "z"])
        assert len(t.steps()) == 3

    def test_name_set(self):
        t = make_tracker(name="my_pipe")
        assert t.name == "my_pipe"

    def test_callback_set(self):
        called = []
        t = make_tracker(steps=["s"], callback=lambda r: called.append(r.name))
        t.done("s")
        assert "s" in called

    def test_empty_steps(self):
        t = make_tracker(steps=[])
        assert t.steps() == []


# ─── TestRunStep ──────────────────────────────────────────────────────────────

class TestRunStep:
    def test_marks_done_on_success(self):
        t = make_tracker(steps=["s"])
        run_step(t, "s", lambda: 42)
        assert t.get_step("s").status == "done"

    def test_returns_fn_result(self):
        t = make_tracker(steps=["s"])
        result = run_step(t, "s", lambda: 99)
        assert result == 99

    def test_marks_failed_on_exception(self):
        t = make_tracker(steps=["s"])
        with pytest.raises(RuntimeError):
            run_step(t, "s", lambda: (_ for _ in ()).throw(RuntimeError("err")))
        assert t.get_step("s").status == "failed"

    def test_stores_error_message(self):
        t = make_tracker(steps=["s"])
        with pytest.raises(ValueError):
            run_step(t, "s", lambda: (_ for _ in ()).throw(ValueError("bad")))
        assert "bad" in t.get_step("s").error

    def test_args_passed_to_fn(self):
        t = make_tracker(steps=["s"])
        result = run_step(t, "s", lambda x, y: x + y, 3, 4)
        assert result == 7

    def test_kwargs_passed_to_fn(self):
        t = make_tracker(steps=["s"])
        result = run_step(t, "s", lambda x=0, y=0: x * y, x=3, y=5)
        assert result == 15


# ─── TestSummarizeTracker ─────────────────────────────────────────────────────

class TestSummarizeTracker:
    def test_returns_dict(self):
        t = make_tracker()
        s = summarize_tracker(t)
        assert isinstance(s, dict)

    def test_has_name_key(self):
        t = make_tracker(name="pipe")
        s = summarize_tracker(t)
        assert s["name"] == "pipe"

    def test_has_progress_key(self):
        t = make_tracker(steps=["a"])
        s = summarize_tracker(t)
        assert "progress" in s

    def test_has_all_expected_keys(self):
        t = make_tracker()
        s = summarize_tracker(t)
        for k in ("name", "progress", "total", "done", "failed",
                   "skipped", "pending", "elapsed"):
            assert k in s

    def test_progress_zero_initially(self):
        t = make_tracker(steps=["a", "b"])
        s = summarize_tracker(t)
        assert s["progress"] == pytest.approx(0.0)
