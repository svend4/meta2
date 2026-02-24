"""Extra tests for puzzle_reconstruction/utils/event_log.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.event_log import (
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rec(eid=0, name="evt", level="info", ts=1.0) -> EventRecord:
    return EventRecord(event_id=eid, name=name, level=level, timestamp=ts)


def _log_with_events(n=4) -> EventLog:
    log = EventLog()
    for i in range(n):
        log.log(f"event_{i}", level="info", timestamp=float(i))
    return log


# ─── EventLogConfig ───────────────────────────────────────────────────────────

class TestEventLogConfigExtra:
    def test_default_max_events(self):
        assert EventLogConfig().max_events == 0

    def test_default_level(self):
        assert EventLogConfig().default_level == "info"

    def test_default_namespace(self):
        assert EventLogConfig().namespace == "default"

    def test_negative_max_events_raises(self):
        with pytest.raises(ValueError):
            EventLogConfig(max_events=-1)

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            EventLogConfig(default_level="verbose")

    def test_empty_namespace_raises(self):
        with pytest.raises(ValueError):
            EventLogConfig(namespace="")

    def test_custom_values(self):
        cfg = EventLogConfig(max_events=50, default_level="warning", namespace="pipe")
        assert cfg.max_events == 50
        assert cfg.default_level == "warning"
        assert cfg.namespace == "pipe"


# ─── EventRecord ──────────────────────────────────────────────────────────────

class TestEventRecordExtra:
    def test_stores_event_id(self):
        assert _rec(eid=7).event_id == 7

    def test_stores_name(self):
        assert _rec(name="fit").name == "fit"

    def test_stores_level(self):
        assert _rec(level="error").level == "error"

    def test_stores_timestamp(self):
        assert _rec(ts=99.0).timestamp == pytest.approx(99.0)

    def test_negative_event_id_raises(self):
        with pytest.raises(ValueError):
            EventRecord(event_id=-1, name="x", level="info", timestamp=0.0)

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            EventRecord(event_id=0, name="", level="info", timestamp=0.0)

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            EventRecord(event_id=0, name="x", level="bad", timestamp=0.0)

    def test_negative_timestamp_raises(self):
        with pytest.raises(ValueError):
            EventRecord(event_id=0, name="x", level="info", timestamp=-1.0)

    def test_is_error_true(self):
        assert _rec(level="error").is_error is True

    def test_is_error_false(self):
        assert _rec(level="info").is_error is False

    def test_level_order_debug(self):
        assert _rec(level="debug").level_order == 0

    def test_level_order_error(self):
        assert _rec(level="error").level_order == 3


# ─── EventSummary ─────────────────────────────────────────────────────────────

class TestEventSummaryExtra:
    def _make(self, total=5, nd=1, ni=2, nw=1, ne=1) -> EventSummary:
        return EventSummary(total=total, n_debug=nd, n_info=ni,
                            n_warn=nw, n_error=ne, names=["a"])

    def test_stores_values(self):
        s = self._make()
        assert s.total == 5 and s.n_error == 1

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            EventSummary(total=-1, n_debug=0, n_info=0, n_warn=0, n_error=0, names=[])

    def test_has_errors_true(self):
        assert self._make(ne=2).has_errors is True

    def test_has_errors_false(self):
        assert self._make(ne=0).has_errors is False

    def test_error_ratio(self):
        s = self._make(total=4, ne=1)
        assert s.error_ratio == pytest.approx(0.25)

    def test_error_ratio_zero_total(self):
        s = EventSummary(total=0, n_debug=0, n_info=0, n_warn=0, n_error=0, names=[])
        assert s.error_ratio == pytest.approx(0.0)


# ─── EventLog ─────────────────────────────────────────────────────────────────

class TestEventLogExtra:
    def test_log_returns_record(self):
        log = EventLog()
        r = log.log("test")
        assert isinstance(r, EventRecord)

    def test_log_empty_name_raises(self):
        log = EventLog()
        with pytest.raises(ValueError):
            log.log("")

    def test_size_grows(self):
        log = EventLog()
        log.log("a")
        log.log("b")
        assert log.size() == 2

    def test_clear_returns_count(self):
        log = _log_with_events(3)
        n = log.clear()
        assert n == 3
        assert log.size() == 0

    def test_to_list_order(self):
        log = _log_with_events(3)
        items = log.to_list()
        assert len(items) == 3

    def test_filter_by_level_error(self):
        log = EventLog()
        log.log("e1", level="error")
        log.log("e2", level="info")
        result = log.filter_by_level("error")
        assert all(r.level == "error" for r in result)

    def test_filter_by_level_invalid_raises(self):
        log = EventLog()
        with pytest.raises(ValueError):
            log.filter_by_level("verbose")

    def test_filter_by_name(self):
        log = EventLog()
        log.log("foo", level="info")
        log.log("bar", level="info")
        result = log.filter_by_name("foo")
        assert all(r.name == "foo" for r in result)

    def test_since(self):
        log = _log_with_events(4)
        result = log.since(2.0)
        assert all(r.timestamp >= 2.0 for r in result)

    def test_namespace_property(self):
        log = EventLog(EventLogConfig(namespace="my_ns"))
        assert log.namespace == "my_ns"

    def test_config_property(self):
        cfg = EventLogConfig(max_events=10)
        log = EventLog(cfg)
        assert log.config is cfg

    def test_max_events_enforced(self):
        log = EventLog(EventLogConfig(max_events=3))
        for i in range(6):
            log.log(f"e{i}")
        assert log.size() == 3


# ─── make_event_log ───────────────────────────────────────────────────────────

class TestMakeEventLogExtra:
    def test_returns_event_log(self):
        assert isinstance(make_event_log(), EventLog)

    def test_namespace_set(self):
        log = make_event_log(namespace="test_ns")
        assert log.namespace == "test_ns"

    def test_max_events_set(self):
        log = make_event_log(max_events=5)
        for i in range(8):
            log.log(f"e{i}")
        assert log.size() == 5


# ─── log_event ────────────────────────────────────────────────────────────────

class TestLogEventExtra:
    def test_returns_record(self):
        log = EventLog()
        r = log_event(log, "test")
        assert isinstance(r, EventRecord)

    def test_level_stored(self):
        log = EventLog()
        r = log_event(log, "err", level="error")
        assert r.level == "error"

    def test_meta_stored(self):
        log = EventLog()
        r = log_event(log, "m", meta={"k": 1})
        assert r.meta.get("k") == 1


# ─── filter_events ────────────────────────────────────────────────────────────

class TestFilterEventsExtra:
    def test_filter_by_level(self):
        recs = [_rec(level="error"), _rec(level="debug")]
        result = filter_events(recs, level="error")
        assert all(r.level == "error" for r in result)

    def test_filter_by_name(self):
        recs = [_rec(name="foo"), _rec(name="bar")]
        result = filter_events(recs, name="foo")
        assert all(r.name == "foo" for r in result)

    def test_no_filter_returns_all(self):
        recs = [_rec(), _rec(eid=1)]
        assert len(filter_events(recs)) == 2

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            filter_events([_rec()], level="bad")


# ─── summarize_events ─────────────────────────────────────────────────────────

class TestSummarizeEventsExtra:
    def test_returns_summary(self):
        recs = [_rec(level="info"), _rec(level="error")]
        assert isinstance(summarize_events(recs), EventSummary)

    def test_counts_correct(self):
        recs = [_rec(level="error"), _rec(level="info"), _rec(level="error")]
        s = summarize_events(recs)
        assert s.n_error == 2 and s.n_info == 1

    def test_empty_list(self):
        s = summarize_events([])
        assert s.total == 0

    def test_unique_names(self):
        recs = [_rec(name="a"), _rec(name="b"), _rec(name="a")]
        s = summarize_events(recs)
        assert sorted(s.names) == ["a", "b"]


# ─── merge_event_logs ─────────────────────────────────────────────────────────

class TestMergeEventLogsExtra:
    def test_returns_count(self):
        target = EventLog()
        source = _log_with_events(3)
        n = merge_event_logs(target, source)
        assert n == 3

    def test_target_grows(self):
        target = EventLog()
        source = _log_with_events(2)
        merge_event_logs(target, source)
        assert target.size() == 2

    def test_empty_source(self):
        target = EventLog()
        source = EventLog()
        assert merge_event_logs(target, source) == 0


# ─── export_event_log ─────────────────────────────────────────────────────────

class TestExportEventLogExtra:
    def test_returns_list(self):
        log = _log_with_events(2)
        assert isinstance(export_event_log(log), list)

    def test_length_matches(self):
        log = _log_with_events(3)
        assert len(export_event_log(log)) == 3

    def test_keys_present(self):
        log = _log_with_events(1)
        d = export_event_log(log)[0]
        for k in ("event_id", "name", "level", "timestamp", "meta"):
            assert k in d

    def test_empty_log(self):
        assert export_event_log(EventLog()) == []
