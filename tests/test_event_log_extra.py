"""Extra tests for puzzle_reconstruction.utils.event_log."""
from __future__ import annotations

import time
import pytest

from puzzle_reconstruction.utils.event_log import (
    DEBUG,
    ERROR,
    INFO,
    WARNING,
    EventLog,
    EventLogConfig,
    EventRecord,
    EventSummary,
    export_event_log,
    filter_events,
    log_event,
    make_event_log,
    merge_event_logs,
    summarize_events,
)


# ─── TestEventLogConfigExtra ─────────────────────────────────────────────────

class TestEventLogConfigExtra:
    def test_default_max_events_zero(self):
        assert EventLogConfig().max_events == 0

    def test_default_level_info(self):
        assert EventLogConfig().default_level == INFO

    def test_default_namespace(self):
        assert EventLogConfig().namespace == "default"

    def test_max_events_positive_ok(self):
        assert EventLogConfig(max_events=50).max_events == 50

    def test_max_events_negative_raises(self):
        with pytest.raises(ValueError):
            EventLogConfig(max_events=-1)

    def test_empty_namespace_raises(self):
        with pytest.raises(ValueError):
            EventLogConfig(namespace="")

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            EventLogConfig(default_level="critical")

    def test_all_valid_levels(self):
        for level in (DEBUG, INFO, WARNING, ERROR):
            cfg = EventLogConfig(default_level=level)
            assert cfg.default_level == level

    def test_warning_level_stored(self):
        assert EventLogConfig(default_level=WARNING).default_level == WARNING

    def test_custom_namespace(self):
        assert EventLogConfig(namespace="pipeline").namespace == "pipeline"


# ─── TestEventRecordExtra ─────────────────────────────────────────────────────

class TestEventRecordExtra:
    def test_name_stored(self):
        rec = EventRecord(event_id=0, name="step", level=INFO, timestamp=1.0)
        assert rec.name == "step"

    def test_event_id_stored(self):
        rec = EventRecord(event_id=5, name="x", level=DEBUG, timestamp=0.0)
        assert rec.event_id == 5

    def test_level_stored(self):
        rec = EventRecord(event_id=0, name="x", level=WARNING, timestamp=0.0)
        assert rec.level == WARNING

    def test_is_error_false_for_warning(self):
        rec = EventRecord(event_id=0, name="x", level=WARNING, timestamp=0.0)
        assert rec.is_error is False

    def test_is_error_true_for_error(self):
        rec = EventRecord(event_id=0, name="x", level=ERROR, timestamp=0.0)
        assert rec.is_error is True

    def test_is_error_false_for_info(self):
        rec = EventRecord(event_id=0, name="x", level=INFO, timestamp=0.0)
        assert rec.is_error is False

    def test_level_order_debug_lowest(self):
        d = EventRecord(event_id=0, name="x", level=DEBUG, timestamp=0.0).level_order
        i = EventRecord(event_id=0, name="x", level=INFO, timestamp=0.0).level_order
        assert d < i

    def test_level_order_warning_lt_error(self):
        w = EventRecord(event_id=0, name="x", level=WARNING, timestamp=0.0).level_order
        e = EventRecord(event_id=0, name="x", level=ERROR, timestamp=0.0).level_order
        assert w < e

    def test_negative_event_id_raises(self):
        with pytest.raises(ValueError):
            EventRecord(event_id=-1, name="x", level=INFO, timestamp=0.0)

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            EventRecord(event_id=0, name="", level=INFO, timestamp=0.0)

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            EventRecord(event_id=0, name="x", level="verbose", timestamp=0.0)

    def test_negative_timestamp_raises(self):
        with pytest.raises(ValueError):
            EventRecord(event_id=0, name="x", level=INFO, timestamp=-0.1)

    def test_meta_default_empty(self):
        rec = EventRecord(event_id=0, name="x", level=DEBUG, timestamp=0.0)
        assert rec.meta == {}

    def test_meta_stored(self):
        rec = EventRecord(event_id=0, name="x", level=INFO, timestamp=0.0,
                          meta={"k": "v"})
        assert rec.meta["k"] == "v"


# ─── TestEventSummaryExtra ────────────────────────────────────────────────────

class TestEventSummaryExtra:
    def test_has_errors_true(self):
        s = EventSummary(total=5, n_debug=1, n_info=2, n_warn=1, n_error=1, names=[])
        assert s.has_errors is True

    def test_has_errors_false(self):
        s = EventSummary(total=3, n_debug=0, n_info=3, n_warn=0, n_error=0, names=[])
        assert s.has_errors is False

    def test_error_ratio_correct(self):
        s = EventSummary(total=4, n_debug=0, n_info=3, n_warn=0, n_error=1, names=[])
        assert s.error_ratio == pytest.approx(0.25)

    def test_error_ratio_zero_total(self):
        s = EventSummary(total=0, n_debug=0, n_info=0, n_warn=0, n_error=0, names=[])
        assert s.error_ratio == pytest.approx(0.0)

    def test_negative_total_raises(self):
        with pytest.raises(ValueError):
            EventSummary(total=-1, n_debug=0, n_info=0, n_warn=0, n_error=0, names=[])

    def test_names_stored(self):
        s = EventSummary(total=2, n_debug=1, n_info=1, n_warn=0, n_error=0,
                         names=["a", "b"])
        assert "a" in s.names and "b" in s.names


# ─── TestEventLogExtra ────────────────────────────────────────────────────────

class TestEventLogExtra:
    def test_log_name_stored(self):
        el = EventLog()
        rec = el.log("test_event", level=INFO)
        assert rec.name == "test_event"

    def test_log_level_stored(self):
        el = EventLog()
        rec = el.log("x", level=WARNING)
        assert rec.level == WARNING

    def test_size_increments(self):
        el = EventLog()
        for i in range(5):
            el.log(f"e{i}")
        assert el.size() == 5

    def test_max_events_capped(self):
        cfg = EventLogConfig(max_events=3)
        el = EventLog(cfg)
        for i in range(7):
            el.log(f"e{i}")
        assert el.size() == 3

    def test_event_ids_monotone(self):
        el = EventLog()
        ids = [el.log(f"e{i}").event_id for i in range(4)]
        assert ids == list(range(4))

    def test_default_level_used(self):
        cfg = EventLogConfig(default_level=ERROR)
        el = EventLog(cfg)
        rec = el.log("x")
        assert rec.level == ERROR

    def test_empty_name_raises(self):
        el = EventLog()
        with pytest.raises(ValueError):
            el.log("")

    def test_meta_stored_in_record(self):
        el = EventLog()
        rec = el.log("x", meta={"score": 0.9})
        assert rec.meta["score"] == pytest.approx(0.9)

    def test_auto_timestamp(self):
        el = EventLog()
        before = time.time()
        rec = el.log("x")
        after = time.time()
        assert before <= rec.timestamp <= after

    def test_namespace_from_config(self):
        cfg = EventLogConfig(namespace="my_ns")
        el = EventLog(cfg)
        assert el.namespace == "my_ns"

    def test_clear_resets_size(self):
        el = EventLog()
        el.log("a")
        el.log("b")
        n = el.clear()
        assert n == 2
        assert el.size() == 0

    def test_to_list_length(self):
        el = EventLog()
        el.log("a")
        el.log("b")
        assert len(el.to_list()) == 2

    def test_filter_by_level_info_includes_warning_error(self):
        el = EventLog()
        el.log("d", level=DEBUG)
        el.log("i", level=INFO)
        el.log("w", level=WARNING)
        el.log("e", level=ERROR)
        result = el.filter_by_level(INFO)
        assert len(result) == 3
        for r in result:
            assert r.level in (INFO, WARNING, ERROR)

    def test_filter_by_level_error_only(self):
        el = EventLog()
        el.log("i", level=INFO)
        el.log("e", level=ERROR)
        result = el.filter_by_level(ERROR)
        assert len(result) == 1

    def test_filter_by_level_invalid_raises(self):
        el = EventLog()
        with pytest.raises(ValueError):
            el.filter_by_level("trace")

    def test_filter_by_name_found(self):
        el = EventLog()
        el.log("start")
        el.log("step")
        el.log("start")
        result = el.filter_by_name("start")
        assert len(result) == 2

    def test_filter_by_name_not_found(self):
        el = EventLog()
        el.log("start")
        assert el.filter_by_name("end") == []

    def test_since_filters_timestamps(self):
        el = EventLog()
        el.log("a", timestamp=100.0)
        el.log("b", timestamp=200.0)
        el.log("c", timestamp=300.0)
        result = el.since(200.0)
        assert len(result) == 2

    def test_since_invalid_raises(self):
        el = EventLog()
        with pytest.raises(ValueError):
            el.since(-1.0)


# ─── TestMakeEventLogExtra ────────────────────────────────────────────────────

class TestMakeEventLogExtra:
    def test_factory_default_ns(self):
        el = make_event_log()
        assert el.namespace == "default"

    def test_factory_custom_ns(self):
        el = make_event_log(namespace="my")
        assert el.namespace == "my"

    def test_factory_max_events(self):
        el = make_event_log(max_events=5)
        for i in range(8):
            el.log(f"e{i}")
        assert el.size() == 5

    def test_factory_default_level(self):
        el = make_event_log(default_level=DEBUG)
        rec = el.log("x")
        assert rec.level == DEBUG


# ─── TestLogEventExtra ────────────────────────────────────────────────────────

class TestLogEventExtra:
    def test_log_event_name(self):
        el = make_event_log()
        rec = log_event(el, "step", level=INFO)
        assert rec.name == "step"

    def test_log_event_meta(self):
        el = make_event_log()
        rec = log_event(el, "x", meta={"val": 7})
        assert rec.meta["val"] == 7

    def test_log_event_increments_size(self):
        el = make_event_log()
        log_event(el, "a")
        log_event(el, "b")
        assert el.size() == 2


# ─── TestFilterEventsExtra ────────────────────────────────────────────────────

class TestFilterEventsExtra:
    def _records(self):
        el = EventLog()
        el.log("a", level=DEBUG)
        el.log("b", level=INFO)
        el.log("a", level=WARNING)
        el.log("c", level=ERROR)
        return el.to_list()

    def test_filter_by_level_warning_plus(self):
        result = filter_events(self._records(), level=WARNING)
        assert all(r.level in (WARNING, ERROR) for r in result)

    def test_filter_by_name(self):
        result = filter_events(self._records(), name="a")
        assert all(r.name == "a" for r in result)
        assert len(result) == 2

    def test_combined_filter(self):
        result = filter_events(self._records(), level=WARNING, name="a")
        assert len(result) == 1
        assert result[0].level == WARNING

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            filter_events(self._records(), level="fatal")

    def test_no_filter_all_returned(self):
        records = self._records()
        result = filter_events(records)
        assert len(result) == len(records)


# ─── TestSummarizeEventsExtra ─────────────────────────────────────────────────

class TestSummarizeEventsExtra:
    def test_total_count(self):
        el = EventLog()
        for _ in range(5):
            el.log("x")
        s = summarize_events(el.to_list())
        assert s.total == 5

    def test_empty_gives_zero_total(self):
        s = summarize_events([])
        assert s.total == 0
        assert not s.has_errors

    def test_names_sorted(self):
        el = EventLog()
        for name in ("z", "a", "m"):
            el.log(name)
        s = summarize_events(el.to_list())
        assert s.names == sorted(s.names)

    def test_counts_correct(self):
        el = EventLog()
        el.log("a", level=DEBUG)
        el.log("b", level=INFO)
        el.log("c", level=WARNING)
        el.log("d", level=ERROR)
        s = summarize_events(el.to_list())
        assert s.n_debug == 1
        assert s.n_info == 1
        assert s.n_warn == 1
        assert s.n_error == 1


# ─── TestMergeEventLogsExtra ──────────────────────────────────────────────────

class TestMergeEventLogsExtra:
    def test_basic_merge_count(self):
        src = make_event_log()
        src.log("a")
        src.log("b")
        tgt = make_event_log()
        added = merge_event_logs(tgt, src)
        assert added == 2
        assert tgt.size() == 2

    def test_names_preserved(self):
        src = make_event_log()
        src.log("step1")
        src.log("step2")
        tgt = make_event_log()
        merge_event_logs(tgt, src)
        names = [r.name for r in tgt.to_list()]
        assert "step1" in names and "step2" in names

    def test_empty_source_zero_added(self):
        src = make_event_log()
        tgt = make_event_log()
        added = merge_event_logs(tgt, src)
        assert added == 0


# ─── TestExportEventLogExtra ──────────────────────────────────────────────────

class TestExportEventLogExtra:
    def test_length_matches_events(self):
        el = make_event_log()
        el.log("a")
        el.log("b")
        assert len(export_event_log(el)) == 2

    def test_empty_log_empty_export(self):
        assert export_event_log(make_event_log()) == []

    def test_required_keys_present(self):
        el = make_event_log()
        el.log("x", level=INFO, meta={"k": 1})
        for rec in export_event_log(el):
            assert "event_id" in rec
            assert "name" in rec
            assert "level" in rec
            assert "timestamp" in rec
            assert "meta" in rec

    def test_meta_copy_independent(self):
        el = make_event_log()
        el.log("ev", meta={"k": 99})
        exported = export_event_log(el)
        exported[0]["meta"]["k"] = 0
        assert el.to_list()[0].meta["k"] == 99
