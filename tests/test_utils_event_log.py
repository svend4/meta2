"""Tests for puzzle_reconstruction.utils.event_log"""
import pytest
from puzzle_reconstruction.utils.event_log import (
    DEBUG, INFO, WARNING, ERROR,
    EventLogConfig, EventRecord, EventSummary, EventLog,
    make_event_log, log_event, filter_events, summarize_events, merge_event_logs,
    export_event_log,
)


# ── EventLogConfig ────────────────────────────────────────────────────────────

def test_event_log_config_defaults():
    cfg = EventLogConfig()
    assert cfg.max_events == 0
    assert cfg.default_level == INFO
    assert cfg.namespace == "default"


def test_event_log_config_negative_max_events_raises():
    with pytest.raises(ValueError):
        EventLogConfig(max_events=-1)


def test_event_log_config_invalid_level_raises():
    with pytest.raises(ValueError):
        EventLogConfig(default_level="critical")


def test_event_log_config_empty_namespace_raises():
    with pytest.raises(ValueError):
        EventLogConfig(namespace="")


# ── EventRecord ───────────────────────────────────────────────────────────────

def test_event_record_is_error_true():
    rec = EventRecord(event_id=0, name="e", level=ERROR, timestamp=1.0)
    assert rec.is_error is True


def test_event_record_is_error_false():
    rec = EventRecord(event_id=0, name="e", level=INFO, timestamp=1.0)
    assert rec.is_error is False


def test_event_record_level_order():
    r_debug = EventRecord(event_id=0, name="e", level=DEBUG, timestamp=1.0)
    r_error = EventRecord(event_id=1, name="e", level=ERROR, timestamp=1.0)
    assert r_debug.level_order == 0
    assert r_error.level_order == 3


def test_event_record_negative_id_raises():
    with pytest.raises(ValueError):
        EventRecord(event_id=-1, name="e", level=INFO, timestamp=1.0)


def test_event_record_empty_name_raises():
    with pytest.raises(ValueError):
        EventRecord(event_id=0, name="", level=INFO, timestamp=1.0)


def test_event_record_negative_timestamp_raises():
    with pytest.raises(ValueError):
        EventRecord(event_id=0, name="e", level=INFO, timestamp=-1.0)


# ── EventSummary ──────────────────────────────────────────────────────────────

def test_event_summary_has_errors_true():
    s = EventSummary(total=5, n_debug=0, n_info=3, n_warn=1, n_error=1, names=[])
    assert s.has_errors is True


def test_event_summary_has_errors_false():
    s = EventSummary(total=3, n_debug=0, n_info=3, n_warn=0, n_error=0, names=[])
    assert s.has_errors is False


def test_event_summary_error_ratio():
    s = EventSummary(total=4, n_debug=0, n_info=2, n_warn=1, n_error=1, names=[])
    assert abs(s.error_ratio - 0.25) < 1e-9


def test_event_summary_zero_total_error_ratio():
    s = EventSummary(total=0, n_debug=0, n_info=0, n_warn=0, n_error=0, names=[])
    assert s.error_ratio == 0.0


# ── EventLog ──────────────────────────────────────────────────────────────────

def test_event_log_log_returns_event_record():
    log = EventLog()
    rec = log.log("test_event", level=INFO)
    assert isinstance(rec, EventRecord)


def test_event_log_size_increments():
    log = EventLog()
    assert log.size() == 0
    log.log("e1")
    log.log("e2")
    assert log.size() == 2


def test_event_log_empty_name_raises():
    log = EventLog()
    with pytest.raises(ValueError):
        log.log("")


def test_event_log_clear_returns_count():
    log = EventLog()
    log.log("a")
    log.log("b")
    n = log.clear()
    assert n == 2
    assert log.size() == 0


def test_event_log_max_events_enforced():
    log = EventLog(EventLogConfig(max_events=3))
    for i in range(10):
        log.log(f"e{i}")
    assert log.size() == 3


def test_event_log_filter_by_level():
    log = EventLog()
    log.log("d", level=DEBUG)
    log.log("i", level=INFO)
    log.log("w", level=WARNING)
    log.log("e", level=ERROR)
    recs = log.filter_by_level(WARNING)
    assert all(r.level_order >= 2 for r in recs)
    assert len(recs) == 2


def test_event_log_filter_by_level_invalid_raises():
    log = EventLog()
    with pytest.raises(ValueError):
        log.filter_by_level("critical")


def test_event_log_filter_by_name():
    log = EventLog()
    log.log("alpha")
    log.log("beta")
    log.log("alpha")
    recs = log.filter_by_name("alpha")
    assert len(recs) == 2


def test_event_log_since():
    log = EventLog()
    log.log("old", timestamp=100.0)
    log.log("new", timestamp=200.0)
    recs = log.since(150.0)
    assert len(recs) == 1
    assert recs[0].name == "new"


def test_event_log_since_negative_raises():
    log = EventLog()
    with pytest.raises(ValueError):
        log.since(-1.0)


def test_event_log_namespace():
    log = EventLog(EventLogConfig(namespace="mynamespace"))
    assert log.namespace == "mynamespace"


def test_event_log_to_list():
    log = EventLog()
    log.log("a")
    log.log("b")
    lst = log.to_list()
    assert len(lst) == 2


# ── make_event_log ────────────────────────────────────────────────────────────

def test_make_event_log_returns_event_log():
    log = make_event_log(max_events=5, namespace="ns")
    assert isinstance(log, EventLog)
    assert log.namespace == "ns"


# ── log_event ─────────────────────────────────────────────────────────────────

def test_log_event_adds_record():
    log = EventLog()
    rec = log_event(log, "my_event", level=WARNING)
    assert rec.level == WARNING
    assert log.size() == 1


# ── filter_events ─────────────────────────────────────────────────────────────

def test_filter_events_by_level():
    log = EventLog()
    log.log("a", level=DEBUG)
    log.log("b", level=ERROR)
    recs = filter_events(log.to_list(), level=ERROR)
    assert all(r.level == ERROR for r in recs)


def test_filter_events_by_name():
    log = EventLog()
    log.log("foo")
    log.log("bar")
    recs = filter_events(log.to_list(), name="foo")
    assert all(r.name == "foo" for r in recs)


def test_filter_events_invalid_level_raises():
    with pytest.raises(ValueError):
        filter_events([], level="bad_level")


# ── summarize_events ──────────────────────────────────────────────────────────

def test_summarize_events_counts():
    log = EventLog()
    log.log("a", level=DEBUG)
    log.log("b", level=INFO)
    log.log("c", level=WARNING)
    log.log("d", level=ERROR)
    log.log("e", level=ERROR)
    s = summarize_events(log.to_list())
    assert s.total == 5
    assert s.n_debug == 1
    assert s.n_info == 1
    assert s.n_warn == 1
    assert s.n_error == 2


def test_summarize_events_empty():
    s = summarize_events([])
    assert s.total == 0
    assert s.has_errors is False


# ── merge_event_logs ──────────────────────────────────────────────────────────

def test_merge_event_logs():
    target = EventLog()
    source = EventLog()
    source.log("s1", level=INFO)
    source.log("s2", level=WARNING)
    n = merge_event_logs(target, source)
    assert n == 2
    assert target.size() == 2


# ── export_event_log ──────────────────────────────────────────────────────────

def test_export_event_log_structure():
    log = EventLog()
    log.log("evt", level=ERROR, meta={"key": "val"})
    exported = export_event_log(log)
    assert len(exported) == 1
    d = exported[0]
    assert "event_id" in d
    assert "name" in d
    assert "level" in d
    assert "timestamp" in d
    assert "meta" in d
    assert d["level"] == ERROR
