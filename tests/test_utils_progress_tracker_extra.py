"""Extra tests for puzzle_reconstruction/utils/progress_tracker.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.progress_tracker import (
    StepRecord,
    PipelineReport,
    ProgressTracker,
    make_tracker,
    run_step,
    summarize_tracker,
)


# ─── StepRecord ───────────────────────────────────────────────────────────────

class TestStepRecordExtra:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            StepRecord(name="")

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError):
            StepRecord(name="s", status="invalid")

    def test_default_status_pending(self):
        r = StepRecord(name="s")
        assert r.status == "pending"

    def test_is_complete_done(self):
        r = StepRecord(name="s", status="done")
        assert r.is_complete is True

    def test_is_complete_pending(self):
        r = StepRecord(name="s", status="pending")
        assert r.is_complete is False

    def test_elapsed_none_when_not_started(self):
        r = StepRecord(name="s")
        assert r.elapsed is None

    def test_elapsed_computed(self):
        r = StepRecord(name="s", started_at=0.0, ended_at=2.5)
        assert r.elapsed == pytest.approx(2.5)


# ─── PipelineReport ───────────────────────────────────────────────────────────

class TestPipelineReportExtra:
    def test_negative_total_steps_raises(self):
        with pytest.raises(ValueError):
            PipelineReport(total_steps=-1)

    def test_negative_total_elapsed_raises(self):
        with pytest.raises(ValueError):
            PipelineReport(total_elapsed=-1.0)

    def test_success_rate_zero_when_no_completed(self):
        r = PipelineReport(total_steps=5, pending=5)
        assert r.success_rate == pytest.approx(0.0)

    def test_success_rate_computed(self):
        r = PipelineReport(total_steps=3, done=2, failed=1)
        assert r.success_rate == pytest.approx(2 / 3)


# ─── ProgressTracker ──────────────────────────────────────────────────────────

class TestProgressTrackerExtra:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            ProgressTracker(name="")

    def test_register_step(self):
        t = ProgressTracker()
        t.register("step1")
        assert t.get_step("step1").status == "pending"

    def test_duplicate_register_raises(self):
        t = ProgressTracker()
        t.register("s")
        with pytest.raises(ValueError):
            t.register("s")

    def test_empty_step_name_raises(self):
        t = ProgressTracker()
        with pytest.raises(ValueError):
            t.register("")

    def test_unregistered_start_raises(self):
        t = ProgressTracker()
        with pytest.raises(KeyError):
            t.start("nonexistent")

    def test_start_sets_running(self):
        t = ProgressTracker()
        t.register("s")
        t.start("s")
        assert t.get_step("s").status == "running"

    def test_done_sets_done(self):
        t = ProgressTracker()
        t.register("s")
        t.start("s")
        t.done("s")
        assert t.get_step("s").status == "done"

    def test_fail_sets_error(self):
        t = ProgressTracker()
        t.register("s")
        t.fail("s", error="something broke")
        assert t.get_step("s").status == "failed"
        assert t.get_step("s").error == "something broke"

    def test_skip_sets_skipped(self):
        t = ProgressTracker()
        t.register("s")
        t.skip("s")
        assert t.get_step("s").status == "skipped"

    def test_is_done_all_complete(self):
        t = make_tracker(steps=["a", "b"])
        t.done("a")
        t.done("b")
        assert t.is_done() is True

    def test_is_done_not_complete(self):
        t = make_tracker(steps=["a", "b"])
        t.done("a")
        assert t.is_done() is False

    def test_progress(self):
        t = make_tracker(steps=["a", "b", "c"])
        t.done("a")
        t.skip("b")
        assert t.progress() == pytest.approx(2 / 3)

    def test_pending_steps(self):
        t = make_tracker(steps=["a", "b"])
        t.done("a")
        pending = t.pending_steps()
        assert len(pending) == 1 and pending[0].name == "b"

    def test_failed_steps(self):
        t = make_tracker(steps=["a", "b"])
        t.fail("a")
        failed = t.failed_steps()
        assert len(failed) == 1

    def test_reset(self):
        t = make_tracker(steps=["a"])
        t.done("a")
        t.reset()
        assert t.get_step("a").status == "pending"

    def test_callback_called(self):
        calls = []
        t = make_tracker(steps=["s"], callback=lambda r: calls.append(r.status))
        t.start("s")
        t.done("s")
        assert "running" in calls and "done" in calls

    def test_report(self):
        t = make_tracker(steps=["a", "b", "c"])
        t.start("a")
        t.done("a")
        t.fail("b")
        rep = t.report()
        assert rep.done == 1 and rep.failed == 1 and rep.pending == 1


# ─── make_tracker ─────────────────────────────────────────────────────────────

class TestMakeTrackerExtra:
    def test_returns_tracker(self):
        t = make_tracker()
        assert isinstance(t, ProgressTracker)

    def test_steps_registered(self):
        t = make_tracker(steps=["a", "b", "c"])
        assert len(t.steps()) == 3

    def test_custom_name(self):
        t = make_tracker(name="myPipeline")
        assert t.name == "myPipeline"


# ─── run_step ─────────────────────────────────────────────────────────────────

class TestRunStepExtra:
    def test_success_sets_done(self):
        t = make_tracker(steps=["compute"])
        result = run_step(t, "compute", lambda: 42)
        assert result == 42
        assert t.get_step("compute").status == "done"

    def test_failure_sets_failed_and_reraises(self):
        t = make_tracker(steps=["bad"])
        with pytest.raises(RuntimeError):
            run_step(t, "bad", lambda: (_ for _ in ()).throw(RuntimeError("x")))
        assert t.get_step("bad").status == "failed"


# ─── summarize_tracker ────────────────────────────────────────────────────────

class TestSummarizeTrackerExtra:
    def test_returns_dict(self):
        t = make_tracker(steps=["a"])
        s = summarize_tracker(t)
        assert isinstance(s, dict)

    def test_keys_present(self):
        t = make_tracker(steps=["a"])
        s = summarize_tracker(t)
        for k in ("name", "progress", "total", "done", "failed", "skipped", "pending"):
            assert k in s

    def test_progress_value(self):
        t = make_tracker(steps=["a", "b"])
        t.done("a")
        s = summarize_tracker(t)
        assert s["progress"] == pytest.approx(0.5)
