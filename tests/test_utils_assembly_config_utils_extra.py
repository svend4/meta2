"""Extra tests for puzzle_reconstruction/utils/assembly_config_utils.py."""
from __future__ import annotations

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _state(step=0, n_placed=2, n_frags=5, cov=0.4, label="") -> AssemblyStateRecord:
    return AssemblyStateRecord(
        step=step, n_placed=n_placed, n_fragments=n_frags,
        coverage=cov, label=label,
    )


def _change(key="lr", old=0.1, new=0.01, step=0) -> ConfigChangeRecord:
    return ConfigChangeRecord(key=key, old_value=old, new_value=new, step=step)


def _filter(name="score", n_in=100, n_kept=60, n_rem=40) -> CandidateFilterRecord:
    return CandidateFilterRecord(
        filter_name=name, n_input=n_in, n_kept=n_kept, n_removed=n_rem,
    )


# ─── AssemblyStateRecord ──────────────────────────────────────────────────────

class TestAssemblyStateRecordExtra:
    def test_stores_step(self):
        assert _state(step=3).step == 3

    def test_stores_n_placed(self):
        assert _state(n_placed=2).n_placed == 2

    def test_stores_n_fragments(self):
        assert _state(n_frags=10).n_fragments == 10

    def test_stores_coverage(self):
        assert _state(cov=0.7).coverage == pytest.approx(0.7)

    def test_step_negative_raises(self):
        with pytest.raises(ValueError):
            AssemblyStateRecord(step=-1, n_placed=1, n_fragments=2, coverage=0.5)

    def test_n_placed_negative_raises(self):
        with pytest.raises(ValueError):
            AssemblyStateRecord(step=0, n_placed=-1, n_fragments=2, coverage=0.5)

    def test_n_fragments_zero_raises(self):
        with pytest.raises(ValueError):
            AssemblyStateRecord(step=0, n_placed=0, n_fragments=0, coverage=0.0)

    def test_coverage_above_one_raises(self):
        with pytest.raises(ValueError):
            AssemblyStateRecord(step=0, n_placed=1, n_fragments=2, coverage=1.1)

    def test_coverage_below_zero_raises(self):
        with pytest.raises(ValueError):
            AssemblyStateRecord(step=0, n_placed=1, n_fragments=2, coverage=-0.1)

    def test_is_complete_false(self):
        assert _state(n_placed=2, n_frags=5).is_complete is False

    def test_is_complete_true(self):
        assert _state(n_placed=5, n_frags=5).is_complete is True

    def test_default_label_empty(self):
        assert _state().label == ""


# ─── AssemblyStateHistory ─────────────────────────────────────────────────────

class TestAssemblyStateHistoryExtra:
    def test_empty_n_steps(self):
        h = AssemblyStateHistory()
        assert h.n_steps == 0

    def test_append_increases_n_steps(self):
        h = AssemblyStateHistory()
        h.append(_state())
        assert h.n_steps == 1

    def test_last_coverage_empty(self):
        h = AssemblyStateHistory()
        assert h.last_coverage == pytest.approx(0.0)

    def test_last_coverage_after_append(self):
        h = AssemblyStateHistory()
        h.append(_state(cov=0.6))
        assert h.last_coverage == pytest.approx(0.6)

    def test_is_monotone_empty(self):
        h = AssemblyStateHistory()
        assert h.is_monotone is True

    def test_is_monotone_increasing(self):
        h = AssemblyStateHistory()
        h.append(_state(cov=0.2))
        h.append(_state(cov=0.5))
        h.append(_state(cov=0.8))
        assert h.is_monotone is True

    def test_is_monotone_decreasing(self):
        h = AssemblyStateHistory()
        h.append(_state(cov=0.8))
        h.append(_state(cov=0.3))
        assert h.is_monotone is False

    def test_multiple_appends(self):
        h = AssemblyStateHistory()
        for i in range(4):
            h.append(_state(step=i))
        assert h.n_steps == 4


# ─── ConfigChangeRecord ───────────────────────────────────────────────────────

class TestConfigChangeRecordExtra:
    def test_stores_key(self):
        assert _change(key="alpha").key == "alpha"

    def test_stores_old_value(self):
        assert _change(old=0.5).old_value == pytest.approx(0.5)

    def test_stores_new_value(self):
        assert _change(new=0.01).new_value == pytest.approx(0.01)

    def test_empty_key_raises(self):
        with pytest.raises(ValueError):
            ConfigChangeRecord(key="", old_value=1, new_value=2)

    def test_step_negative_raises(self):
        with pytest.raises(ValueError):
            ConfigChangeRecord(key="k", old_value=1, new_value=2, step=-1)

    def test_changed_true(self):
        assert _change(old=0.1, new=0.01).changed is True

    def test_changed_false(self):
        assert _change(old=0.5, new=0.5).changed is False

    def test_default_step_zero(self):
        assert _change().step == 0


# ─── ConfigChangeLog ──────────────────────────────────────────────────────────

class TestConfigChangeLogExtra:
    def test_empty_n_changes(self):
        log = ConfigChangeLog()
        assert log.n_changes == 0

    def test_append_changed(self):
        log = ConfigChangeLog()
        log.append(_change(old=1, new=2))
        assert log.n_changes == 1

    def test_append_unchanged_not_counted(self):
        log = ConfigChangeLog()
        log.append(_change(old=5, new=5))
        assert log.n_changes == 0

    def test_changed_keys(self):
        log = ConfigChangeLog()
        log.append(_change(key="lr", old=0.1, new=0.01))
        log.append(_change(key="bs", old=32, new=64))
        keys = log.changed_keys
        assert "lr" in keys and "bs" in keys

    def test_changed_keys_sorted(self):
        log = ConfigChangeLog()
        log.append(_change(key="z"))
        log.append(_change(key="a"))
        assert log.changed_keys == sorted(log.changed_keys)

    def test_changed_keys_unique(self):
        log = ConfigChangeLog()
        log.append(_change(key="lr"))
        log.append(_change(key="lr"))
        assert log.changed_keys.count("lr") == 1


# ─── CandidateFilterRecord ────────────────────────────────────────────────────

class TestCandidateFilterRecordExtra:
    def test_stores_filter_name(self):
        assert _filter(name="top_k").filter_name == "top_k"

    def test_stores_n_input(self):
        assert _filter(n_in=100).n_input == 100

    def test_stores_n_kept(self):
        assert _filter(n_kept=60).n_kept == 60

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            CandidateFilterRecord(filter_name="", n_input=10, n_kept=5, n_removed=5)

    def test_negative_n_input_raises(self):
        with pytest.raises(ValueError):
            CandidateFilterRecord(filter_name="f", n_input=-1, n_kept=0, n_removed=0)

    def test_negative_n_kept_raises(self):
        with pytest.raises(ValueError):
            CandidateFilterRecord(filter_name="f", n_input=10, n_kept=-1, n_removed=5)

    def test_keep_ratio_normal(self):
        r = _filter(n_in=100, n_kept=60)
        assert r.keep_ratio == pytest.approx(0.6)

    def test_keep_ratio_zero_input(self):
        r = CandidateFilterRecord(filter_name="f", n_input=0, n_kept=0, n_removed=0)
        assert r.keep_ratio == pytest.approx(0.0)

    def test_optional_threshold_none(self):
        assert _filter().threshold is None

    def test_custom_threshold(self):
        r = CandidateFilterRecord(filter_name="f", n_input=10, n_kept=5,
                                   n_removed=5, threshold=0.5)
        assert r.threshold == pytest.approx(0.5)


# ─── FilterPipelineSummary ────────────────────────────────────────────────────

class TestFilterPipelineSummaryExtra:
    def test_empty_n_stages(self):
        assert FilterPipelineSummary().n_stages == 0

    def test_add_stage_increases_n(self):
        fp = FilterPipelineSummary()
        fp.add_stage(_filter())
        assert fp.n_stages == 1

    def test_total_removed_sum(self):
        fp = FilterPipelineSummary()
        fp.add_stage(_filter(n_rem=40))
        fp.add_stage(_filter(n_rem=10))
        assert fp.total_removed == 50

    def test_final_n_kept_empty(self):
        assert FilterPipelineSummary().final_n_kept == 0

    def test_final_n_kept_last_stage(self):
        fp = FilterPipelineSummary()
        fp.add_stage(_filter(n_kept=80))
        fp.add_stage(_filter(n_kept=45))
        assert fp.final_n_kept == 45


# ─── summarize_assembly_history ───────────────────────────────────────────────

class TestSummarizeAssemblyHistoryExtra:
    def test_returns_dict(self):
        assert isinstance(summarize_assembly_history(AssemblyStateHistory()), dict)

    def test_empty_keys(self):
        d = summarize_assembly_history(AssemblyStateHistory())
        assert "n_steps" in d and "final_coverage" in d and "is_monotone" in d

    def test_empty_n_steps_zero(self):
        d = summarize_assembly_history(AssemblyStateHistory())
        assert d["n_steps"] == 0

    def test_nonempty_has_is_complete(self):
        h = AssemblyStateHistory()
        h.append(_state())
        d = summarize_assembly_history(h)
        assert "is_complete" in d

    def test_final_coverage_correct(self):
        h = AssemblyStateHistory()
        h.append(_state(cov=0.7))
        d = summarize_assembly_history(h)
        assert d["final_coverage"] == pytest.approx(0.7)


# ─── build_filter_pipeline_summary ───────────────────────────────────────────

class TestBuildFilterPipelineSummaryExtra:
    def test_returns_summary(self):
        result = build_filter_pipeline_summary([_filter()])
        assert isinstance(result, FilterPipelineSummary)

    def test_n_stages_correct(self):
        result = build_filter_pipeline_summary([_filter(), _filter(name="b")])
        assert result.n_stages == 2

    def test_empty_sequence(self):
        result = build_filter_pipeline_summary([])
        assert result.n_stages == 0

    def test_total_removed(self):
        result = build_filter_pipeline_summary([
            _filter(n_rem=30),
            _filter(n_rem=15),
        ])
        assert result.total_removed == 45


# ─── build_config_change_log ──────────────────────────────────────────────────

class TestBuildConfigChangeLogExtra:
    def test_returns_log(self):
        result = build_config_change_log([{"lr": (0.1, 0.01)}])
        assert isinstance(result, ConfigChangeLog)

    def test_empty_sequence(self):
        result = build_config_change_log([])
        assert result.n_changes == 0

    def test_change_detected(self):
        result = build_config_change_log([{"lr": (0.1, 0.01)}])
        assert result.n_changes >= 1

    def test_no_change_not_counted(self):
        result = build_config_change_log([{"lr": (0.5, 0.5)}])
        assert result.n_changes == 0

    def test_multiple_diffs(self):
        result = build_config_change_log([
            {"lr": (0.1, 0.01)},
            {"bs": (32, 64)},
        ])
        assert result.n_changes == 2
