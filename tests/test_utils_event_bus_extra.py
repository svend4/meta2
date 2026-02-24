"""Extra tests for puzzle_reconstruction/utils/event_bus.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.event_bus import (
    BusConfig,
    EventRecord,
    BusSummary,
    EventBus,
    make_event_bus,
    collect_events,
    drain_bus,
)


# ─── BusConfig ────────────────────────────────────────────────────────────────

class TestBusConfigExtra:
    def test_default_max_history(self):
        assert BusConfig().max_history == 0

    def test_default_raise_on_error(self):
        assert BusConfig().raise_on_error is False

    def test_default_allow_wildcard(self):
        assert BusConfig().allow_wildcard is True

    def test_negative_max_history_raises(self):
        with pytest.raises(ValueError):
            BusConfig(max_history=-1)

    def test_zero_max_history_ok(self):
        cfg = BusConfig(max_history=0)
        assert cfg.max_history == 0

    def test_custom_values(self):
        cfg = BusConfig(max_history=100, raise_on_error=True, allow_wildcard=False)
        assert cfg.max_history == 100
        assert cfg.raise_on_error is True
        assert cfg.allow_wildcard is False


# ─── EventRecord ──────────────────────────────────────────────────────────────

class TestEventRecordBusExtra:
    def test_stores_topic(self):
        r = EventRecord(topic="test", payload=None, seq=0)
        assert r.topic == "test"

    def test_empty_topic_raises(self):
        with pytest.raises(ValueError):
            EventRecord(topic="", payload=None, seq=0)

    def test_negative_seq_raises(self):
        with pytest.raises(ValueError):
            EventRecord(topic="t", payload=None, seq=-1)

    def test_has_handlers_false(self):
        r = EventRecord(topic="t", payload=None, seq=0, n_handlers=0)
        assert r.has_handlers is False

    def test_has_handlers_true(self):
        r = EventRecord(topic="t", payload=None, seq=0, n_handlers=1)
        assert r.has_handlers is True


# ─── BusSummary ───────────────────────────────────────────────────────────────

class TestBusSummaryExtra:
    def _make(self, pub=5, handled=4, errors=1, topics=2, subs=3) -> BusSummary:
        return BusSummary(
            total_published=pub, total_handled=handled, total_errors=errors,
            n_topics=topics, n_subscriptions=subs,
        )

    def test_stores_values(self):
        s = self._make()
        assert s.total_published == 5 and s.total_handled == 4

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            BusSummary(total_published=-1, total_handled=0,
                       total_errors=0, n_topics=0, n_subscriptions=0)

    def test_error_ratio(self):
        s = self._make(handled=4, errors=1)
        assert s.error_ratio == pytest.approx(0.2)

    def test_error_ratio_zero_handled(self):
        s = self._make(handled=0, errors=0)
        assert s.error_ratio == pytest.approx(0.0)


# ─── EventBus ─────────────────────────────────────────────────────────────────

class TestEventBusExtra:
    def test_publish_returns_record(self):
        bus = EventBus()
        r = bus.publish("test", "data")
        assert isinstance(r, EventRecord)

    def test_publish_empty_topic_raises(self):
        bus = EventBus()
        with pytest.raises(ValueError):
            bus.publish("")

    def test_subscribe_returns_id(self):
        bus = EventBus()
        sub_id = bus.subscribe("topic", lambda r: None)
        assert isinstance(sub_id, str)

    def test_handler_called(self):
        bus = EventBus()
        received = []
        bus.subscribe("topic", received.append)
        bus.publish("topic", "data")
        assert len(received) == 1

    def test_unsubscribe_returns_true(self):
        bus = EventBus()
        sub_id = bus.subscribe("t", lambda r: None)
        assert bus.unsubscribe(sub_id) is True

    def test_unsubscribe_unknown_returns_false(self):
        bus = EventBus()
        assert bus.unsubscribe("nonexistent") is False

    def test_handler_not_called_after_unsubscribe(self):
        bus = EventBus()
        received = []
        sub_id = bus.subscribe("t", received.append)
        bus.unsubscribe(sub_id)
        bus.publish("t", "data")
        assert len(received) == 0

    def test_history_grows(self):
        bus = EventBus()
        bus.publish("a", 1)
        bus.publish("b", 2)
        assert len(bus.history()) == 2

    def test_history_filter_by_topic(self):
        bus = EventBus()
        bus.publish("a", 1)
        bus.publish("b", 2)
        hist = bus.history("a")
        assert len(hist) == 1 and hist[0].topic == "a"

    def test_clear_history(self):
        bus = EventBus()
        bus.publish("t", 1)
        bus.clear_history()
        assert bus.history() == []

    def test_topics_returns_list(self):
        bus = EventBus()
        bus.subscribe("x", lambda r: None)
        assert "x" in bus.topics()

    def test_summary_returns_summary(self):
        bus = EventBus()
        bus.publish("t", 1)
        assert isinstance(bus.summary(), BusSummary)

    def test_max_history_enforced(self):
        bus = EventBus(BusConfig(max_history=3))
        for i in range(5):
            bus.publish("t", i)
        assert len(bus.history()) == 3

    def test_wildcard_subscriber(self):
        bus = EventBus(BusConfig(allow_wildcard=True))
        received = []
        bus.subscribe("*", received.append)
        bus.publish("any_topic", 42)
        assert len(received) == 1

    def test_wildcard_disabled_raises(self):
        bus = EventBus(BusConfig(allow_wildcard=False))
        with pytest.raises(ValueError):
            bus.subscribe("*", lambda r: None)

    def test_raise_on_error(self):
        bus = EventBus(BusConfig(raise_on_error=True))
        bus.subscribe("t", lambda r: 1 / 0)
        with pytest.raises(RuntimeError):
            bus.publish("t", None)


# ─── make_event_bus ───────────────────────────────────────────────────────────

class TestMakeEventBusExtra:
    def test_returns_event_bus(self):
        assert isinstance(make_event_bus(), EventBus)

    def test_max_history_set(self):
        bus = make_event_bus(max_history=10)
        for i in range(15):
            bus.publish("t", i)
        assert len(bus.history()) == 10


# ─── collect_events ───────────────────────────────────────────────────────────

class TestCollectEventsExtra:
    def test_returns_list(self):
        bus = EventBus()
        bus.publish("t", 1)
        result = collect_events(bus, "t")
        assert isinstance(result, list)

    def test_collects_published(self):
        bus = EventBus()
        bus.publish("foo", "x")
        bus.publish("foo", "y")
        result = collect_events(bus, "foo")
        assert len(result) == 2

    def test_filters_by_topic(self):
        bus = EventBus()
        bus.publish("a", 1)
        bus.publish("b", 2)
        result = collect_events(bus, "a")
        assert all(r.topic == "a" for r in result)


# ─── drain_bus ────────────────────────────────────────────────────────────────

class TestDrainBusExtra:
    def test_returns_list(self):
        bus = EventBus()
        bus.publish("t", 1)
        result = drain_bus(bus)
        assert isinstance(result, list)

    def test_clears_history(self):
        bus = EventBus()
        bus.publish("t", 1)
        drain_bus(bus)
        assert bus.history() == []

    def test_returns_all_events(self):
        bus = EventBus()
        for i in range(4):
            bus.publish("t", i)
        result = drain_bus(bus)
        assert len(result) == 4
