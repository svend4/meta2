"""Tests for puzzle_reconstruction.utils.event_bus"""
import pytest
from puzzle_reconstruction.utils.event_bus import (
    BusConfig, EventRecord, BusSummary, EventBus,
    make_event_bus, collect_events, drain_bus,
)


# ── BusConfig ─────────────────────────────────────────────────────────────────

def test_bus_config_defaults():
    cfg = BusConfig()
    assert cfg.max_history == 0
    assert cfg.raise_on_error is False
    assert cfg.allow_wildcard is True


def test_bus_config_negative_history_raises():
    with pytest.raises(ValueError):
        BusConfig(max_history=-1)


def test_bus_config_zero_history_ok():
    cfg = BusConfig(max_history=0)
    assert cfg.max_history == 0


# ── EventRecord ───────────────────────────────────────────────────────────────

def test_event_record_empty_topic_raises():
    with pytest.raises(ValueError):
        EventRecord(topic="", payload=None, seq=0)


def test_event_record_negative_seq_raises():
    with pytest.raises(ValueError):
        EventRecord(topic="t", payload=None, seq=-1)


def test_event_record_has_handlers_false():
    rec = EventRecord(topic="t", payload=42, seq=0, n_handlers=0)
    assert rec.has_handlers is False


def test_event_record_has_handlers_true():
    rec = EventRecord(topic="t", payload=42, seq=0, n_handlers=1)
    assert rec.has_handlers is True


# ── BusSummary ────────────────────────────────────────────────────────────────

def test_bus_summary_error_ratio_zero():
    s = BusSummary(total_published=5, total_handled=5,
                   total_errors=0, n_topics=2, n_subscriptions=2)
    assert s.error_ratio == 0.0


def test_bus_summary_error_ratio_nonzero():
    s = BusSummary(total_published=10, total_handled=8,
                   total_errors=2, n_topics=1, n_subscriptions=1)
    assert abs(s.error_ratio - 2/10) < 1e-9


def test_bus_summary_negative_raises():
    with pytest.raises(ValueError):
        BusSummary(total_published=-1, total_handled=0,
                   total_errors=0, n_topics=0, n_subscriptions=0)


def test_bus_summary_all_zero():
    s = BusSummary(total_published=0, total_handled=0,
                   total_errors=0, n_topics=0, n_subscriptions=0)
    assert s.error_ratio == 0.0


# ── EventBus basic ────────────────────────────────────────────────────────────

def test_event_bus_subscribe_returns_string():
    bus = EventBus()
    sub_id = bus.subscribe("topic", lambda r: None)
    assert isinstance(sub_id, str)


def test_event_bus_publish_returns_event_record():
    bus = EventBus()
    rec = bus.publish("topic", 42)
    assert isinstance(rec, EventRecord)
    assert rec.topic == "topic"
    assert rec.payload == 42


def test_event_bus_publish_increments_seq():
    bus = EventBus()
    r1 = bus.publish("t", 1)
    r2 = bus.publish("t", 2)
    assert r2.seq == r1.seq + 1


def test_event_bus_empty_topic_publish_raises():
    bus = EventBus()
    with pytest.raises(ValueError):
        bus.publish("")


def test_event_bus_handler_called():
    bus = EventBus()
    received = []
    bus.subscribe("t", lambda r: received.append(r.payload))
    bus.publish("t", 99)
    assert received == [99]


def test_event_bus_unsubscribe_returns_true():
    bus = EventBus()
    sub_id = bus.subscribe("t", lambda r: None)
    assert bus.unsubscribe(sub_id) is True


def test_event_bus_unsubscribe_unknown_returns_false():
    bus = EventBus()
    assert bus.unsubscribe("nonexistent") is False


def test_event_bus_unsubscribed_handler_not_called():
    bus = EventBus()
    called = []
    sub_id = bus.subscribe("t", lambda r: called.append(1))
    bus.unsubscribe(sub_id)
    bus.publish("t", None)
    assert called == []


def test_event_bus_wildcard_subscription():
    bus = EventBus()
    received = []
    bus.subscribe("*", lambda r: received.append(r.topic))
    bus.publish("topicA", None)
    bus.publish("topicB", None)
    assert "topicA" in received
    assert "topicB" in received


def test_event_bus_wildcard_disabled_raises():
    bus = EventBus(BusConfig(allow_wildcard=False))
    with pytest.raises(ValueError):
        bus.subscribe("*", lambda r: None)


def test_event_bus_history_filtered_by_topic():
    bus = EventBus()
    bus.publish("a", 1)
    bus.publish("b", 2)
    bus.publish("a", 3)
    hist = bus.history(topic="a")
    assert all(r.topic == "a" for r in hist)
    assert len(hist) == 2


def test_event_bus_clear_history():
    bus = EventBus()
    bus.publish("t", 1)
    bus.publish("t", 2)
    bus.clear_history()
    assert bus.history() == []


def test_event_bus_max_history():
    bus = EventBus(BusConfig(max_history=3))
    for i in range(10):
        bus.publish("t", i)
    assert len(bus.history()) == 3


def test_event_bus_topics():
    bus = EventBus()
    bus.subscribe("alpha", lambda r: None)
    bus.subscribe("beta", lambda r: None)
    topics = bus.topics()
    assert "alpha" in topics
    assert "beta" in topics


def test_event_bus_summary():
    bus = EventBus()
    bus.subscribe("t", lambda r: None)
    bus.publish("t", 1)
    bus.publish("t", 2)
    s = bus.summary()
    assert isinstance(s, BusSummary)
    assert s.total_published == 2
    assert s.total_handled == 2
    assert s.n_subscriptions == 1


def test_event_bus_error_no_raise():
    bus = EventBus(BusConfig(raise_on_error=False))
    bus.subscribe("t", lambda r: 1/0)
    rec = bus.publish("t", None)
    s = bus.summary()
    assert s.total_errors == 1


def test_event_bus_error_raise():
    bus = EventBus(BusConfig(raise_on_error=True))
    bus.subscribe("t", lambda r: 1/0)
    with pytest.raises(RuntimeError):
        bus.publish("t", None)


# ── make_event_bus ────────────────────────────────────────────────────────────

def test_make_event_bus_returns_event_bus():
    bus = make_event_bus(max_history=10)
    assert isinstance(bus, EventBus)


# ── collect_events ────────────────────────────────────────────────────────────

def test_collect_events():
    bus = EventBus()
    bus.publish("x", 1)
    bus.publish("y", 2)
    bus.publish("x", 3)
    evts = collect_events(bus, "x")
    assert len(evts) == 2


# ── drain_bus ─────────────────────────────────────────────────────────────────

def test_drain_bus_returns_all_clears():
    bus = EventBus()
    bus.publish("a", 1)
    bus.publish("b", 2)
    drained = drain_bus(bus)
    assert len(drained) == 2
    assert bus.history() == []
