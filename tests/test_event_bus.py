"""Тесты для puzzle_reconstruction.utils.event_bus."""
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


# ─── TestBusConfig ────────────────────────────────────────────────────────────

class TestBusConfig:
    def test_defaults(self):
        cfg = BusConfig()
        assert cfg.max_history == 0
        assert cfg.raise_on_error is False
        assert cfg.allow_wildcard is True

    def test_valid_custom(self):
        cfg = BusConfig(max_history=50, raise_on_error=True, allow_wildcard=False)
        assert cfg.max_history == 50
        assert cfg.raise_on_error is True
        assert cfg.allow_wildcard is False

    def test_invalid_max_history_neg(self):
        with pytest.raises(ValueError):
            BusConfig(max_history=-1)

    def test_max_history_zero_ok(self):
        cfg = BusConfig(max_history=0)
        assert cfg.max_history == 0


# ─── TestEventRecord ──────────────────────────────────────────────────────────

class TestEventRecord:
    def test_basic(self):
        r = EventRecord(topic="test", payload=42, seq=0)
        assert r.topic == "test"
        assert r.payload == 42
        assert r.seq == 0
        assert r.n_handlers == 0

    def test_has_handlers_false(self):
        r = EventRecord(topic="x", payload=None, seq=0, n_handlers=0)
        assert r.has_handlers is False

    def test_has_handlers_true(self):
        r = EventRecord(topic="x", payload=None, seq=0, n_handlers=2)
        assert r.has_handlers is True

    def test_invalid_topic_empty(self):
        with pytest.raises(ValueError):
            EventRecord(topic="", payload=None, seq=0)

    def test_invalid_seq_neg(self):
        with pytest.raises(ValueError):
            EventRecord(topic="x", payload=None, seq=-1)

    def test_payload_can_be_none(self):
        r = EventRecord(topic="x", payload=None, seq=0)
        assert r.payload is None

    def test_payload_complex(self):
        r = EventRecord(topic="x", payload={"a": [1, 2, 3]}, seq=5)
        assert r.payload["a"] == [1, 2, 3]


# ─── TestBusSummary ───────────────────────────────────────────────────────────

class TestBusSummary:
    def _make(self, published=5, handled=4, errors=1, topics=2, subs=3):
        return BusSummary(
            total_published=published,
            total_handled=handled,
            total_errors=errors,
            n_topics=topics,
            n_subscriptions=subs,
        )

    def test_error_ratio(self):
        s = self._make(handled=4, errors=1)
        assert abs(s.error_ratio - 1 / 5) < 1e-9

    def test_error_ratio_no_errors(self):
        s = self._make(handled=5, errors=0)
        assert s.error_ratio == 0.0

    def test_error_ratio_zero_total(self):
        s = BusSummary(0, 0, 0, 0, 0)
        assert s.error_ratio == 0.0

    def test_invalid_total_published_neg(self):
        with pytest.raises(ValueError):
            BusSummary(total_published=-1, total_handled=0,
                       total_errors=0, n_topics=0, n_subscriptions=0)

    def test_invalid_total_handled_neg(self):
        with pytest.raises(ValueError):
            BusSummary(total_published=0, total_handled=-1,
                       total_errors=0, n_topics=0, n_subscriptions=0)

    def test_invalid_total_errors_neg(self):
        with pytest.raises(ValueError):
            BusSummary(total_published=0, total_handled=0,
                       total_errors=-1, n_topics=0, n_subscriptions=0)

    def test_invalid_n_topics_neg(self):
        with pytest.raises(ValueError):
            BusSummary(total_published=0, total_handled=0,
                       total_errors=0, n_topics=-1, n_subscriptions=0)

    def test_invalid_n_subscriptions_neg(self):
        with pytest.raises(ValueError):
            BusSummary(total_published=0, total_handled=0,
                       total_errors=0, n_topics=0, n_subscriptions=-1)


# ─── TestEventBusSubscribeUnsubscribe ─────────────────────────────────────────

class TestEventBusSubscribeUnsubscribe:
    def test_subscribe_returns_id(self):
        bus = EventBus()
        sid = bus.subscribe("t", lambda r: None)
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_multiple_subscriptions_unique_ids(self):
        bus = EventBus()
        ids = {bus.subscribe("t", lambda r: None) for _ in range(5)}
        assert len(ids) == 5

    def test_unsubscribe_returns_true(self):
        bus = EventBus()
        sid = bus.subscribe("t", lambda r: None)
        assert bus.unsubscribe(sid) is True

    def test_unsubscribe_unknown_returns_false(self):
        bus = EventBus()
        assert bus.unsubscribe("nonexistent") is False

    def test_unsubscribe_twice_second_false(self):
        bus = EventBus()
        sid = bus.subscribe("t", lambda r: None)
        bus.unsubscribe(sid)
        assert bus.unsubscribe(sid) is False

    def test_empty_topic_raises(self):
        bus = EventBus()
        with pytest.raises(ValueError):
            bus.subscribe("", lambda r: None)

    def test_wildcard_disabled_raises(self):
        bus = EventBus(BusConfig(allow_wildcard=False))
        with pytest.raises(ValueError):
            bus.subscribe("*", lambda r: None)

    def test_wildcard_enabled_ok(self):
        bus = EventBus(BusConfig(allow_wildcard=True))
        sid = bus.subscribe("*", lambda r: None)
        assert sid is not None


# ─── TestEventBusPublish ──────────────────────────────────────────────────────

class TestEventBusPublish:
    def test_basic_publish(self):
        bus = EventBus()
        calls = []
        bus.subscribe("topic", lambda r: calls.append(r.payload))
        bus.publish("topic", "hello")
        assert calls == ["hello"]

    def test_empty_topic_raises(self):
        bus = EventBus()
        with pytest.raises(ValueError):
            bus.publish("", "data")

    def test_record_returned(self):
        bus = EventBus()
        rec = bus.publish("t", 123)
        assert isinstance(rec, EventRecord)
        assert rec.topic == "t"
        assert rec.payload == 123

    def test_seq_increments(self):
        bus = EventBus()
        r1 = bus.publish("t", 1)
        r2 = bus.publish("t", 2)
        assert r2.seq == r1.seq + 1

    def test_n_handlers_counted(self):
        bus = EventBus()
        bus.subscribe("t", lambda r: None)
        bus.subscribe("t", lambda r: None)
        rec = bus.publish("t", "x")
        assert rec.n_handlers == 2

    def test_no_subscribers_zero_handlers(self):
        bus = EventBus()
        rec = bus.publish("t", "x")
        assert rec.n_handlers == 0

    def test_wildcard_subscriber_receives_all(self):
        bus = EventBus()
        seen = []
        bus.subscribe("*", lambda r: seen.append(r.topic))
        bus.publish("a", 1)
        bus.publish("b", 2)
        assert "a" in seen
        assert "b" in seen

    def test_wildcard_disabled_no_cross_delivery(self):
        bus = EventBus(BusConfig(allow_wildcard=False))
        calls = []
        bus.subscribe("a", lambda r: calls.append(r.topic))
        bus.publish("b", 1)
        assert calls == []

    def test_error_handler_raise_on_error_true(self):
        bus = EventBus(BusConfig(raise_on_error=True))
        bus.subscribe("t", lambda r: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(RuntimeError):
            bus.publish("t", "x")

    def test_error_handler_raise_on_error_false(self):
        bus = EventBus(BusConfig(raise_on_error=False))
        bus.subscribe("t", lambda r: (_ for _ in ()).throw(RuntimeError("boom")))
        rec = bus.publish("t", "x")  # should not raise
        assert rec is not None

    def test_payload_none(self):
        bus = EventBus()
        calls = []
        bus.subscribe("t", lambda r: calls.append(r.payload))
        bus.publish("t")
        assert calls == [None]


# ─── TestEventBusHistory ──────────────────────────────────────────────────────

class TestEventBusHistory:
    def test_history_grows(self):
        bus = EventBus()
        bus.publish("a", 1)
        bus.publish("b", 2)
        assert len(bus.history()) == 2

    def test_history_filter_by_topic(self):
        bus = EventBus()
        bus.publish("a", 1)
        bus.publish("b", 2)
        bus.publish("a", 3)
        h = bus.history("a")
        assert len(h) == 2
        assert all(r.topic == "a" for r in h)

    def test_clear_history(self):
        bus = EventBus()
        bus.publish("t", 1)
        bus.clear_history()
        assert bus.history() == []

    def test_max_history_enforced(self):
        bus = EventBus(BusConfig(max_history=3))
        for i in range(10):
            bus.publish("t", i)
        assert len(bus.history()) <= 3

    def test_max_history_zero_unlimited(self):
        bus = EventBus(BusConfig(max_history=0))
        for i in range(100):
            bus.publish("t", i)
        assert len(bus.history()) == 100


# ─── TestEventBusTopicsAndSummary ─────────────────────────────────────────────

class TestEventBusTopicsAndSummary:
    def test_topics_empty(self):
        bus = EventBus()
        assert bus.topics() == []

    def test_topics_after_subscribe(self):
        bus = EventBus()
        bus.subscribe("x", lambda r: None)
        bus.subscribe("y", lambda r: None)
        t = bus.topics()
        assert "x" in t
        assert "y" in t

    def test_topics_after_unsubscribe(self):
        bus = EventBus()
        sid = bus.subscribe("z", lambda r: None)
        bus.unsubscribe(sid)
        assert "z" not in bus.topics()

    def test_summary_published_count(self):
        bus = EventBus()
        bus.publish("t", 1)
        bus.publish("t", 2)
        s = bus.summary()
        assert s.total_published == 2

    def test_summary_handled_count(self):
        bus = EventBus()
        bus.subscribe("t", lambda r: None)
        bus.publish("t", 1)
        bus.publish("t", 2)
        s = bus.summary()
        assert s.total_handled == 2

    def test_summary_error_count(self):
        bus = EventBus(BusConfig(raise_on_error=False))
        bus.subscribe("t", lambda r: (_ for _ in ()).throw(ValueError()))
        bus.publish("t", 1)
        s = bus.summary()
        assert s.total_errors == 1

    def test_summary_n_subscriptions(self):
        bus = EventBus()
        bus.subscribe("a", lambda r: None)
        bus.subscribe("b", lambda r: None)
        s = bus.summary()
        assert s.n_subscriptions == 2


# ─── TestMakeEventBus ─────────────────────────────────────────────────────────

class TestMakeEventBus:
    def test_default(self):
        bus = make_event_bus()
        assert isinstance(bus, EventBus)

    def test_custom(self):
        bus = make_event_bus(max_history=10, raise_on_error=True)
        bus.publish("t", 1)
        assert len(bus.history()) == 1

    def test_no_wildcard(self):
        bus = make_event_bus(allow_wildcard=False)
        with pytest.raises(ValueError):
            bus.subscribe("*", lambda r: None)


# ─── TestCollectEvents ────────────────────────────────────────────────────────

class TestCollectEvents:
    def test_basic(self):
        bus = EventBus()
        bus.publish("x", 1)
        bus.publish("y", 2)
        bus.publish("x", 3)
        events = collect_events(bus, "x")
        assert len(events) == 2
        assert all(e.topic == "x" for e in events)

    def test_empty_topic(self):
        bus = EventBus()
        bus.publish("y", 1)
        assert collect_events(bus, "x") == []


# ─── TestDrainBus ─────────────────────────────────────────────────────────────

class TestDrainBus:
    def test_returns_all_events(self):
        bus = EventBus()
        bus.publish("a", 1)
        bus.publish("b", 2)
        events = drain_bus(bus)
        assert len(events) == 2

    def test_clears_history(self):
        bus = EventBus()
        bus.publish("t", 1)
        drain_bus(bus)
        assert bus.history() == []

    def test_empty_bus(self):
        bus = EventBus()
        assert drain_bus(bus) == []

    def test_drain_twice_second_empty(self):
        bus = EventBus()
        bus.publish("t", 1)
        drain_bus(bus)
        assert drain_bus(bus) == []
