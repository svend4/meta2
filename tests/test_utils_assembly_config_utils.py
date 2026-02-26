"""Tests for puzzle_reconstruction.utils.assembly_config_utils"""
import pytest
from puzzle_reconstruction.utils.assembly_config_utils import (
    AssemblyStateRecord,
    AssemblyStateHistory,
    ConfigChangeRecord,
    ConfigChangeLog,
    CandidateFilterRecord,
    FilterPipelineSummary,
    summarize_assembly_history,
    build_filter_pipeline_summary,
    build_config_change_log,
)


# ─── AssemblyStateRecord ──────────────────────────────────────────────────────

def test_assembly_state_record_basic():
    r = AssemblyStateRecord(step=0, n_placed=3, n_fragments=10, coverage=0.3)
    assert r.step == 0
    assert r.n_placed == 3
    assert r.n_fragments == 10
    assert r.coverage == 0.3


def test_assembly_state_record_is_complete_true():
    r = AssemblyStateRecord(step=1, n_placed=5, n_fragments=5, coverage=1.0)
    assert r.is_complete is True


def test_assembly_state_record_is_complete_false():
    r = AssemblyStateRecord(step=1, n_placed=4, n_fragments=5, coverage=0.8)
    assert r.is_complete is False


def test_assembly_state_record_invalid_step():
    with pytest.raises(ValueError, match="step"):
        AssemblyStateRecord(step=-1, n_placed=0, n_fragments=1, coverage=0.0)


def test_assembly_state_record_invalid_coverage():
    with pytest.raises(ValueError, match="coverage"):
        AssemblyStateRecord(step=0, n_placed=0, n_fragments=1, coverage=1.5)


def test_assembly_state_record_invalid_n_placed():
    with pytest.raises(ValueError, match="n_placed"):
        AssemblyStateRecord(step=0, n_placed=-1, n_fragments=1, coverage=0.0)


def test_assembly_state_record_invalid_n_fragments():
    with pytest.raises(ValueError, match="n_fragments"):
        AssemblyStateRecord(step=0, n_placed=0, n_fragments=0, coverage=0.0)


# ─── AssemblyStateHistory ─────────────────────────────────────────────────────

def test_assembly_state_history_empty():
    h = AssemblyStateHistory()
    assert h.n_steps == 0
    assert h.last_coverage == 0.0
    assert h.is_monotone is True


def test_assembly_state_history_append_and_n_steps():
    h = AssemblyStateHistory()
    h.append(AssemblyStateRecord(0, 1, 5, 0.2))
    h.append(AssemblyStateRecord(1, 2, 5, 0.4))
    assert h.n_steps == 2


def test_assembly_state_history_last_coverage():
    h = AssemblyStateHistory()
    h.append(AssemblyStateRecord(0, 1, 5, 0.2))
    h.append(AssemblyStateRecord(1, 3, 5, 0.6))
    assert h.last_coverage == pytest.approx(0.6)


def test_assembly_state_history_is_monotone_true():
    h = AssemblyStateHistory()
    h.append(AssemblyStateRecord(0, 1, 5, 0.2))
    h.append(AssemblyStateRecord(1, 2, 5, 0.4))
    h.append(AssemblyStateRecord(2, 3, 5, 0.6))
    assert h.is_monotone is True


def test_assembly_state_history_is_monotone_false():
    h = AssemblyStateHistory()
    h.append(AssemblyStateRecord(0, 3, 5, 0.6))
    h.append(AssemblyStateRecord(1, 1, 5, 0.2))
    assert h.is_monotone is False


# ─── ConfigChangeRecord ───────────────────────────────────────────────────────

def test_config_change_record_changed_true():
    r = ConfigChangeRecord(key="lr", old_value=0.01, new_value=0.001)
    assert r.changed is True


def test_config_change_record_changed_false():
    r = ConfigChangeRecord(key="lr", old_value=0.01, new_value=0.01)
    assert r.changed is False


def test_config_change_record_invalid_key():
    with pytest.raises(ValueError, match="key"):
        ConfigChangeRecord(key="", old_value=1, new_value=2)


def test_config_change_record_invalid_step():
    with pytest.raises(ValueError, match="step"):
        ConfigChangeRecord(key="x", old_value=1, new_value=2, step=-1)


# ─── ConfigChangeLog ──────────────────────────────────────────────────────────

def test_config_change_log_n_changes():
    log = ConfigChangeLog()
    log.append(ConfigChangeRecord("a", 1, 2))
    log.append(ConfigChangeRecord("b", 3, 3))
    assert log.n_changes == 1


def test_config_change_log_changed_keys():
    log = ConfigChangeLog()
    log.append(ConfigChangeRecord("z", 0, 1))
    log.append(ConfigChangeRecord("a", 5, 5))
    log.append(ConfigChangeRecord("m", 1, 2))
    assert log.changed_keys == ["m", "z"]


# ─── CandidateFilterRecord ────────────────────────────────────────────────────

def test_candidate_filter_record_keep_ratio():
    r = CandidateFilterRecord("f1", n_input=10, n_kept=7, n_removed=3)
    assert r.keep_ratio == pytest.approx(0.7)


def test_candidate_filter_record_keep_ratio_zero_input():
    r = CandidateFilterRecord("f1", n_input=0, n_kept=0, n_removed=0)
    assert r.keep_ratio == 0.0


def test_candidate_filter_record_invalid_name():
    with pytest.raises(ValueError, match="filter_name"):
        CandidateFilterRecord("", n_input=5, n_kept=3, n_removed=2)


# ─── FilterPipelineSummary ────────────────────────────────────────────────────

def test_filter_pipeline_summary_empty():
    fps = FilterPipelineSummary()
    assert fps.n_stages == 0
    assert fps.total_removed == 0
    assert fps.final_n_kept == 0


def test_filter_pipeline_summary_add_and_query():
    fps = FilterPipelineSummary()
    fps.add_stage(CandidateFilterRecord("s1", 100, 80, 20))
    fps.add_stage(CandidateFilterRecord("s2", 80, 60, 20))
    assert fps.n_stages == 2
    assert fps.total_removed == 40
    assert fps.final_n_kept == 60


# ─── summarize_assembly_history ───────────────────────────────────────────────

def test_summarize_assembly_history_empty():
    h = AssemblyStateHistory()
    d = summarize_assembly_history(h)
    assert d["n_steps"] == 0
    assert d["final_coverage"] == 0.0
    assert d["is_monotone"] is True


def test_summarize_assembly_history_nonempty():
    h = AssemblyStateHistory()
    h.append(AssemblyStateRecord(0, 3, 3, 1.0))
    d = summarize_assembly_history(h)
    assert d["n_steps"] == 1
    assert d["final_coverage"] == pytest.approx(1.0)
    assert d["is_complete"] is True


# ─── build_filter_pipeline_summary ───────────────────────────────────────────

def test_build_filter_pipeline_summary():
    records = [
        CandidateFilterRecord("s1", 50, 40, 10),
        CandidateFilterRecord("s2", 40, 30, 10),
    ]
    fps = build_filter_pipeline_summary(records)
    assert fps.n_stages == 2
    assert fps.final_n_kept == 30


def test_build_filter_pipeline_summary_empty():
    fps = build_filter_pipeline_summary([])
    assert fps.n_stages == 0


# ─── build_config_change_log ─────────────────────────────────────────────────

def test_build_config_change_log_basic():
    diffs = [
        {"lr": (0.01, 0.001), "epochs": (10, 20)},
        {"lr": (0.001, 0.001)},
    ]
    log = build_config_change_log(diffs)
    assert log.n_changes == 2  # lr and epochs changed


def test_build_config_change_log_empty():
    log = build_config_change_log([])
    assert log.n_changes == 0


def test_build_config_change_log_step_assignment():
    diffs = [{"a": (1, 2)}, {"b": (3, 4)}]
    log = build_config_change_log(diffs)
    steps = [r.step for r in log.records]
    assert steps == [0, 1]
