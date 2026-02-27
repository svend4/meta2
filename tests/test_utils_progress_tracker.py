"""Tests for puzzle_reconstruction.utils.progress_tracker"""
import time
import pytest
import numpy as np

from puzzle_reconstruction.utils.progress_tracker import (
    StepRecord,
    PipelineReport,
    ProgressTracker,
    make_tracker,
    run_step,
    summarize_tracker,
)


# ── StepRecord ───────────────────────────────────────────────────────────────

def test_step_record_defaults():
    rec = StepRecord(name="step1")
    assert rec.name == "step1"
    assert rec.status == "pending"
    assert rec.started_at is None
    assert rec.ended_at is None
    assert rec.error is None


def test_step_record_empty_name_raises():
    with pytest.raises(ValueError):
        StepRecord(name="")


def test_step_record_whitespace_name_raises():
    with pytest.raises(ValueError):
        StepRecord(name="   ")


def test_step_record_invalid_status_raises():
    with pytest.raises(ValueError):
        StepRecord(name="x", status="invalid")


def test_step_record_elapsed_none_when_not_started():
    rec = StepRecord(name="x")
    assert rec.elapsed is None


def test_step_record_elapsed_computed():
    rec = StepRecord(name="x", started_at=0.0, ended_at=2.5)
    assert rec.elapsed == pytest.approx(2.5)


def test_step_record_is_complete_done():
    rec = StepRecord(name="x", status="done")
    assert rec.is_complete is True


def test_step_record_is_complete_failed():
    rec = StepRecord(name="x", status="failed")
    assert rec.is_complete is True


def test_step_record_is_complete_skipped():
    rec = StepRecord(name="x", status="skipped")
    assert rec.is_complete is True


def test_step_record_is_complete_pending():
    rec = StepRecord(name="x", status="pending")
    assert rec.is_complete is False


def test_step_record_is_complete_running():
    rec = StepRecord(name="x", status="running")
    assert rec.is_complete is False


# ── PipelineReport ────────────────────────────────────────────────────────────

def test_pipeline_report_defaults():
    rep = PipelineReport()
    assert rep.total_steps == 0
    assert rep.done == 0
    assert rep.success_rate == 0.0


def test_pipeline_report_success_rate():
    rep = PipelineReport(total_steps=4, done=3, failed=1)
    assert rep.success_rate == pytest.approx(0.75)


def test_pipeline_report_success_rate_all_done():
    rep = PipelineReport(total_steps=5, done=5)
    assert rep.success_rate == pytest.approx(1.0)


def test_pipeline_report_negative_raises():
    with pytest.raises(ValueError):
        PipelineReport(done=-1)


def test_pipeline_report_negative_elapsed_raises():
    with pytest.raises(ValueError):
        PipelineReport(total_elapsed=-1.0)


# ── ProgressTracker ──────────────────────────────────────────────────────────

def test_tracker_empty_name_raises():
    with pytest.raises(ValueError):
        ProgressTracker(name="")


def test_tracker_register_and_get():
    t = ProgressTracker()
    t.register("preprocess")
    rec = t.get_step("preprocess")
    assert rec.name == "preprocess"
    assert rec.status == "pending"


def test_tracker_register_duplicate_raises():
    t = ProgressTracker()
    t.register("step1")
    with pytest.raises(ValueError):
        t.register("step1")


def test_tracker_get_unregistered_raises():
    t = ProgressTracker()
    with pytest.raises(KeyError):
        t.get_step("nonexistent")


def test_tracker_start():
    t = ProgressTracker()
    t.register("step1")
    t.start("step1")
    rec = t.get_step("step1")
    assert rec.status == "running"
    assert rec.started_at is not None


def test_tracker_done():
    t = ProgressTracker()
    t.register("step1")
    t.start("step1")
    t.done("step1")
    rec = t.get_step("step1")
    assert rec.status == "done"
    assert rec.ended_at is not None


def test_tracker_fail():
    t = ProgressTracker()
    t.register("step1")
    t.start("step1")
    t.fail("step1", error="something broke")
    rec = t.get_step("step1")
    assert rec.status == "failed"
    assert rec.error == "something broke"


def test_tracker_skip():
    t = ProgressTracker()
    t.register("step1")
    t.skip("step1")
    rec = t.get_step("step1")
    assert rec.status == "skipped"


def test_tracker_is_done_all_complete():
    t = make_tracker("p", steps=["a", "b", "c"])
    for s in ["a", "b", "c"]:
        t.start(s)
        t.done(s)
    assert t.is_done() is True


def test_tracker_is_done_partial():
    t = make_tracker("p", steps=["a", "b"])
    t.start("a")
    t.done("a")
    assert t.is_done() is False


def test_tracker_progress_zero():
    t = make_tracker("p", steps=["a", "b"])
    assert t.progress() == 0.0


def test_tracker_progress_partial():
    t = make_tracker("p", steps=["a", "b", "c", "d"])
    t.start("a"); t.done("a")
    t.start("b"); t.done("b")
    assert t.progress() == pytest.approx(0.5)


def test_tracker_progress_empty_tracker():
    t = ProgressTracker()
    assert t.progress() == 0.0


def test_tracker_steps_in_order():
    t = make_tracker("p", steps=["x", "y", "z"])
    names = [s.name for s in t.steps()]
    assert names == ["x", "y", "z"]


def test_tracker_pending_steps():
    t = make_tracker("p", steps=["a", "b", "c"])
    t.start("a"); t.done("a")
    pending = t.pending_steps()
    assert len(pending) == 2


def test_tracker_failed_steps():
    t = make_tracker("p", steps=["a", "b"])
    t.start("a"); t.fail("a", "error")
    failed = t.failed_steps()
    assert len(failed) == 1
    assert failed[0].name == "a"


def test_tracker_report():
    t = make_tracker("p", steps=["a", "b", "c"])
    t.start("a"); t.done("a")
    t.start("b"); t.fail("b", "err")
    t.skip("c")
    rep = t.report()
    assert rep.done == 1
    assert rep.failed == 1
    assert rep.skipped == 1
    assert rep.pending == 0


def test_tracker_reset():
    t = make_tracker("p", steps=["a"])
    t.start("a"); t.done("a")
    t.reset()
    assert t.get_step("a").status == "pending"
    assert t.get_step("a").started_at is None


def test_tracker_callback_called():
    calls = []
    t = make_tracker("p", steps=["a"], callback=lambda r: calls.append(r.status))
    t.start("a")
    t.done("a")
    assert "running" in calls
    assert "done" in calls


# ── make_tracker ─────────────────────────────────────────────────────────────

def test_make_tracker_with_steps():
    t = make_tracker("pipeline", steps=["s1", "s2", "s3"])
    assert len(t.steps()) == 3


def test_make_tracker_empty_steps():
    t = make_tracker("p", steps=[])
    assert len(t.steps()) == 0


# ── run_step ─────────────────────────────────────────────────────────────────

def test_run_step_success():
    t = make_tracker("p", steps=["compute"])
    result = run_step(t, "compute", lambda x: x * 2, 21)
    assert result == 42
    assert t.get_step("compute").status == "done"


def test_run_step_failure():
    def bad_fn():
        raise RuntimeError("fail")

    t = make_tracker("p", steps=["bad"])
    with pytest.raises(RuntimeError):
        run_step(t, "bad", bad_fn)
    assert t.get_step("bad").status == "failed"
    assert "fail" in t.get_step("bad").error


def test_run_step_sets_running_before_execution():
    statuses = []

    def capture_fn(tracker, step_name):
        statuses.append(tracker.get_step(step_name).status)

    t = make_tracker("p", steps=["s"])
    run_step(t, "s", capture_fn, t, "s")
    assert statuses[0] == "running"


# ── summarize_tracker ────────────────────────────────────────────────────────

def test_summarize_tracker_keys():
    t = make_tracker("mypipe", steps=["a"])
    t.start("a"); t.done("a")
    s = summarize_tracker(t)
    for key in ("name", "progress", "total", "done", "failed", "skipped", "pending", "elapsed"):
        assert key in s


def test_summarize_tracker_name():
    t = make_tracker("my_pipeline")
    s = summarize_tracker(t)
    assert s["name"] == "my_pipeline"


def test_summarize_tracker_progress_value():
    t = make_tracker("p", steps=["a", "b"])
    t.start("a"); t.done("a")
    s = summarize_tracker(t)
    assert s["progress"] == pytest.approx(0.5)
