"""Тесты для puzzle_reconstruction.utils.event_log."""
import pytest
from puzzle_reconstruction.utils.event_log import (
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    EventLogConfig,
    EventRecord,
    EventSummary,
    EventLog,
    make_event_log,
    log_event,
    filter_events,
    summarize_events,
    merge_event_logs,
    export_event_log,
)


class TestEventLogConfig:
    def test_defaults(self):
        cfg = EventLogConfig()
        assert cfg.max_events == 0
        assert cfg.default_level == INFO
        assert cfg.namespace == "default"

    def test_valid_custom(self):
        cfg = EventLogConfig(max_events=100, default_level=WARNING, namespace="test")
        assert cfg.max_events == 100
        assert cfg.default_level == WARNING
        assert cfg.namespace == "test"

    def test_invalid_max_events(self):
        with pytest.raises(ValueError):
            EventLogConfig(max_events=-1)

    def test_invalid_default_level(self):
        with pytest.raises(ValueError):
            EventLogConfig(default_level="critical")

    def test_invalid_namespace(self):
        with pytest.raises(ValueError):
            EventLogConfig(namespace="")


class TestEventRecord:
    def test_basic(self):
        rec = EventRecord(event_id=0, name="start", level=INFO, timestamp=1000.0)
        assert rec.event_id == 0
        assert rec.name == "start"
        assert not rec.is_error

    def test_is_error(self):
        rec = EventRecord(event_id=1, name="fail", level=ERROR, timestamp=1001.0)
        assert rec.is_error

    def test_level_order_ascending(self):
        orders = []
        for level in (DEBUG, INFO, WARNING, ERROR):
            rec = EventRecord(event_id=0, name="x", level=level, timestamp=0.0)
            orders.append(rec.level_order)
        assert orders == sorted(orders)

    def test_invalid_event_id(self):
        with pytest.raises(ValueError):
            EventRecord(event_id=-1, name="x", level=INFO, timestamp=0.0)

    def test_invalid_name(self):
        with pytest.raises(ValueError):
            EventRecord(event_id=0, name="", level=INFO, timestamp=0.0)

    def test_invalid_level(self):
        with pytest.raises(ValueError):
            EventRecord(event_id=0, name="x", level="verbose", timestamp=0.0)

    def test_invalid_timestamp(self):
        with pytest.raises(ValueError):
            EventRecord(event_id=0, name="x", level=INFO, timestamp=-1.0)

    def test_meta_default_empty(self):
        rec = EventRecord(event_id=0, name="x", level=DEBUG, timestamp=0.0)
        assert rec.meta == {}


class TestEventSummary:
    def test_basic(self):
        s = EventSummary(total=5, n_debug=1, n_info=2, n_warn=1, n_error=1,
                         names=["a", "b"])
        assert s.has_errors is True
        assert abs(s.error_ratio - 0.2) < 1e-9

    def test_no_errors(self):
        s = EventSummary(total=3, n_debug=1, n_info=2, n_warn=0, n_error=0, names=[])
        assert s.has_errors is False
        assert s.error_ratio == 0.0

    def test_zero_total_error_ratio(self):
        s = EventSummary(total=0, n_debug=0, n_info=0, n_warn=0, n_error=0, names=[])
        assert s.error_ratio == 0.0

    def test_invalid_counts(self):
        with pytest.raises(ValueError):
            EventSummary(total=-1, n_debug=0, n_info=0, n_warn=0, n_error=0, names=[])


class TestEventLogLog:
    def test_basic_log(self):
        el = EventLog()
        rec = el.log("step", level=INFO)
        assert rec.name == "step"
        assert rec.level == INFO
        assert el.size() == 1

    def test_auto_timestamp(self):
        import time
        before = time.time()
        el = EventLog()
        rec = el.log("x")
        after = time.time()
        assert before <= rec.timestamp <= after

    def test_max_events_enforcement(self):
        cfg = EventLogConfig(max_events=3)
        el = EventLog(cfg)
        for i in range(5):
            el.log(f"e{i}")
        assert el.size() == 3

    def test_event_ids_monotone(self):
        el = EventLog()
        ids = [el.log(f"e{i}").event_id for i in range(5)]
        assert ids == list(range(5))

    def test_default_level_used(self):
        cfg = EventLogConfig(default_level=WARNING)
        el = EventLog(cfg)
        rec = el.log("x")
        assert rec.level == WARNING

    def test_empty_name_raises(self):
        el = EventLog()
        with pytest.raises(ValueError):
            el.log("")

    def test_meta_stored(self):
        el = EventLog()
        rec = el.log("x", meta={"key": 42})
        assert rec.meta["key"] == 42


class TestEventLogFilterByLevel:
    def test_filter_info(self):
        el = EventLog()
        el.log("d", level=DEBUG)
        el.log("i", level=INFO)
        el.log("w", level=WARNING)
        el.log("e", level=ERROR)
        result = el.filter_by_level(INFO)
        assert all(r.level in (INFO, WARNING, ERROR) for r in result)
        assert len(result) == 3

    def test_filter_error(self):
        el = EventLog()
        el.log("i", level=INFO)
        el.log("e", level=ERROR)
        result = el.filter_by_level(ERROR)
        assert len(result) == 1
        assert result[0].level == ERROR

    def test_invalid_level_raises(self):
        el = EventLog()
        with pytest.raises(ValueError):
            el.filter_by_level("trace")


class TestEventLogFilterByName:
    def test_filter_returns_matching(self):
        el = EventLog()
        el.log("start")
        el.log("step")
        el.log("start")
        result = el.filter_by_name("start")
        assert len(result) == 2
        assert all(r.name == "start" for r in result)

    def test_filter_no_match(self):
        el = EventLog()
        el.log("start")
        result = el.filter_by_name("end")
        assert result == []


class TestEventLogSince:
    def test_since_filters(self):
        el = EventLog()
        el.log("a", timestamp=100.0)
        el.log("b", timestamp=200.0)
        el.log("c", timestamp=300.0)
        result = el.since(200.0)
        assert len(result) == 2
        assert all(r.timestamp >= 200.0 for r in result)

    def test_invalid_timestamp(self):
        el = EventLog()
        with pytest.raises(ValueError):
            el.since(-1.0)


class TestEventLogOps:
    def test_to_list(self):
        el = EventLog()
        el.log("a")
        el.log("b")
        lst = el.to_list()
        assert len(lst) == 2

    def test_clear_returns_count(self):
        el = EventLog()
        el.log("a")
        el.log("b")
        n = el.clear()
        assert n == 2
        assert el.size() == 0

    def test_namespace_property(self):
        cfg = EventLogConfig(namespace="pipeline")
        el = EventLog(cfg)
        assert el.namespace == "pipeline"


class TestMakeEventLog:
    def test_factory(self):
        el = make_event_log(max_events=10, default_level=DEBUG, namespace="ns")
        assert el.namespace == "ns"
        el.log("x")
        assert el.size() == 1

    def test_defaults(self):
        el = make_event_log()
        assert el.namespace == "default"


class TestLogEventFunction:
    def test_basic(self):
        el = make_event_log()
        rec = log_event(el, "start", level=INFO, meta={"step": 1})
        assert rec.name == "start"
        assert rec.meta["step"] == 1


class TestFilterEvents:
    def _make_records(self):
        el = EventLog()
        el.log("a", level=DEBUG)
        el.log("b", level=INFO)
        el.log("a", level=WARNING)
        el.log("c", level=ERROR)
        return el.to_list()

    def test_filter_by_level(self):
        records = self._make_records()
        result = filter_events(records, level=WARNING)
        assert all(r.level in (WARNING, ERROR) for r in result)

    def test_filter_by_name(self):
        records = self._make_records()
        result = filter_events(records, name="a")
        assert all(r.name == "a" for r in result)

    def test_combined_filter(self):
        records = self._make_records()
        result = filter_events(records, level=WARNING, name="a")
        assert len(result) == 1
        assert result[0].level == WARNING

    def test_invalid_level_raises(self):
        records = self._make_records()
        with pytest.raises(ValueError):
            filter_events(records, level="fatal")


class TestSummarizeEvents:
    def test_counts(self):
        el = EventLog()
        el.log("a", level=DEBUG)
        el.log("b", level=INFO)
        el.log("b", level=INFO)
        el.log("c", level=WARNING)
        el.log("d", level=ERROR)
        s = summarize_events(el.to_list())
        assert s.total == 5
        assert s.n_debug == 1
        assert s.n_info == 2
        assert s.n_warn == 1
        assert s.n_error == 1

    def test_names_sorted(self):
        el = EventLog()
        el.log("z")
        el.log("a")
        el.log("m")
        s = summarize_events(el.to_list())
        assert s.names == sorted(s.names)

    def test_empty(self):
        s = summarize_events([])
        assert s.total == 0
        assert not s.has_errors


class TestMergeEventLogs:
    def test_basic_merge(self):
        src = make_event_log()
        src.log("a", level=INFO)
        src.log("b", level=WARNING)
        tgt = make_event_log()
        added = merge_event_logs(tgt, src)
        assert added == 2
        assert tgt.size() == 2

    def test_merge_preserves_names(self):
        src = make_event_log()
        src.log("step1")
        src.log("step2")
        tgt = make_event_log()
        merge_event_logs(tgt, src)
        names = [r.name for r in tgt.to_list()]
        assert "step1" in names
        assert "step2" in names


class TestExportEventLog:
    def test_export_structure(self):
        el = make_event_log()
        el.log("start", level=INFO, meta={"x": 1})
        el.log("end", level=WARNING)
        exported = export_event_log(el)
        assert len(exported) == 2
        keys = set(exported[0].keys())
        assert {"event_id", "name", "level", "timestamp", "meta"} == keys

    def test_meta_copied(self):
        el = make_event_log()
        el.log("ev", meta={"k": 99})
        exported = export_event_log(el)
        exported[0]["meta"]["k"] = 0
        # Original should not be modified
        assert el.to_list()[0].meta["k"] == 99

    def test_empty_log(self):
        el = make_event_log()
        assert export_event_log(el) == []
