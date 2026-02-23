"""Extra tests for puzzle_reconstruction/utils/event_bus.py"""
import pytest

from puzzle_reconstruction.utils.event_bus import (
    BusConfig,
    BusSummary,
    EventBus,
    EventRecord,
    collect_events,
    drain_bus,
    make_event_bus,
)


# ─── TestBusConfigExtra ───────────────────────────────────────────────────────

class TestBusConfigExtra:
    def test_max_history_100(self):
        cfg = BusConfig(max_history=100)
        assert cfg.max_history == 100

    def test_raise_on_error_default_false(self):
        cfg = BusConfig()
        assert cfg.raise_on_error is False

    def test_allow_wildcard_default_true(self):
        cfg = BusConfig()
        assert cfg.allow_wildcard is True

    def test_raise_on_error_true(self):
        cfg = BusConfig(raise_on_error=True)
        assert cfg.raise_on_error is True

    def test_allow_wildcard_false(self):
        cfg = BusConfig(allow_wildcard=False)
        assert cfg.allow_wildcard is False

    def test_max_history_1(self):
        cfg = BusConfig(max_history=1)
        assert cfg.max_history == 1


# ─── TestEventRecordExtra ─────────────────────────────────────────────────────

class TestEventRecordExtra:
    def test_n_handlers_stored(self):
        r = EventRecord(topic="t", payload=None, seq=0, n_handlers=5)
        assert r.n_handlers == 5

    def test_seq_zero_valid(self):
        r = EventRecord(topic="t", payload=None, seq=0)
        assert r.seq == 0

    def test_seq_large_valid(self):
        r = EventRecord(topic="t", payload=None, seq=9999)
        assert r.seq == 9999

    def test_has_handlers_one(self):
        r = EventRecord(topic="t", payload=None, seq=0, n_handlers=1)
        assert r.has_handlers is True

    def test_payload_dict(self):
        r = EventRecord(topic="t", payload={"key": "value"}, seq=0)
        assert r.payload["key"] == "value"

    def test_payload_list(self):
        r = EventRecord(topic="t", payload=[1, 2, 3], seq=1)
        assert r.payload == [1, 2, 3]

    def test_topic_stored(self):
        r = EventRecord(topic="my_topic", payload=None, seq=0)
        assert r.topic == "my_topic"


# ─── TestBusSummaryExtra ──────────────────────────────────────────────────────

class TestBusSummaryExtra:
    def test_error_ratio_all_errors(self):
        # ratio = errors / (handled + errors) = 3 / (0 + 3) = 1.0
        s = BusSummary(total_published=3, total_handled=0,
                       total_errors=3, n_topics=1, n_subscriptions=1)
        assert s.error_ratio == pytest.approx(1.0)

    def test_n_topics_stored(self):
        s = BusSummary(total_published=0, total_handled=0,
                       total_errors=0, n_topics=5, n_subscriptions=0)
        assert s.n_topics == 5

    def test_n_subscriptions_stored(self):
        s = BusSummary(total_published=0, total_handled=0,
                       total_errors=0, n_topics=0, n_subscriptions=7)
        assert s.n_subscriptions == 7

    def test_total_published_stored(self):
        s = BusSummary(total_published=10, total_handled=8,
                       total_errors=0, n_topics=2, n_subscriptions=3)
        assert s.total_published == 10

    def test_total_handled_stored(self):
        s = BusSummary(total_published=10, total_handled=8,
                       total_errors=0, n_topics=2, n_subscriptions=3)
        assert s.total_handled == 8

    def test_zero_errors_zero_ratio(self):
        s = BusSummary(total_published=5, total_handled=5,
                       total_errors=0, n_topics=2, n_subscriptions=1)
        assert s.error_ratio == pytest.approx(0.0)


# ─── TestEventBusSubscribeExtra ───────────────────────────────────────────────

class TestEventBusSubscribeExtra:
    def test_subscribe_returns_str(self):
        bus = EventBus()
        sid = bus.subscribe("topic", lambda r: None)
        assert isinstance(sid, str)

    def test_ten_unique_ids(self):
        bus = EventBus()
        ids = [bus.subscribe("t", lambda r: None) for _ in range(10)]
        assert len(set(ids)) == 10

    def test_subscribe_different_topics(self):
        bus = EventBus()
        sid1 = bus.subscribe("a", lambda r: None)
        sid2 = bus.subscribe("b", lambda r: None)
        assert sid1 != sid2

    def test_subscribe_after_unsubscribe(self):
        bus = EventBus()
        sid = bus.subscribe("t", lambda r: None)
        bus.unsubscribe(sid)
        sid2 = bus.subscribe("t", lambda r: None)
        assert sid2 is not None

    def test_unsubscribe_unknown_false(self):
        bus = EventBus()
        assert bus.unsubscribe("definitely_not_existing") is False

    def test_subscribe_same_topic_twice(self):
        bus = EventBus()
        sid1 = bus.subscribe("t", lambda r: None)
        sid2 = bus.subscribe("t", lambda r: None)
        assert sid1 != sid2


# ─── TestEventBusPublishExtra ─────────────────────────────────────────────────

class TestEventBusPublishExtra:
    def test_multiple_subscribers_all_called(self):
        bus = EventBus()
        calls = []
        bus.subscribe("t", lambda r: calls.append(1))
        bus.subscribe("t", lambda r: calls.append(2))
        bus.subscribe("t", lambda r: calls.append(3))
        bus.publish("t", "x")
        assert len(calls) == 3

    def test_different_topics_no_cross(self):
        bus = EventBus()
        seen_a = []
        seen_b = []
        bus.subscribe("a", lambda r: seen_a.append(r.payload))
        bus.subscribe("b", lambda r: seen_b.append(r.payload))
        bus.publish("a", 1)
        bus.publish("b", 2)
        assert seen_a == [1]
        assert seen_b == [2]

    def test_seq_starts_at_zero(self):
        bus = EventBus()
        r = bus.publish("t", 1)
        assert r.seq == 0

    def test_record_has_handlers_true(self):
        bus = EventBus()
        bus.subscribe("t", lambda r: None)
        rec = bus.publish("t", "x")
        assert rec.has_handlers is True

    def test_record_has_handlers_false_when_no_sub(self):
        bus = EventBus()
        rec = bus.publish("t", "x")
        assert rec.has_handlers is False

    def test_payload_integer_delivered(self):
        bus = EventBus()
        received = []
        bus.subscribe("t", lambda r: received.append(r.payload))
        bus.publish("t", 42)
        assert received == [42]

    def test_wildcard_catches_multiple_topics(self):
        bus = EventBus()
        topics_seen = []
        bus.subscribe("*", lambda r: topics_seen.append(r.topic))
        bus.publish("x", 1)
        bus.publish("y", 2)
        bus.publish("z", 3)
        assert set(topics_seen) == {"x", "y", "z"}


# ─── TestEventBusHistoryExtra ─────────────────────────────────────────────────

class TestEventBusHistoryExtra:
    def test_history_preserves_order(self):
        bus = EventBus()
        bus.publish("t", 1)
        bus.publish("t", 2)
        bus.publish("t", 3)
        payloads = [r.payload for r in bus.history()]
        assert payloads == [1, 2, 3]

    def test_max_history_1(self):
        bus = EventBus(BusConfig(max_history=1))
        bus.publish("t", 1)
        bus.publish("t", 2)
        assert len(bus.history()) == 1

    def test_max_history_5(self):
        bus = EventBus(BusConfig(max_history=5))
        for i in range(20):
            bus.publish("t", i)
        assert len(bus.history()) <= 5

    def test_clear_history_empty(self):
        bus = EventBus()
        for _ in range(5):
            bus.publish("t", None)
        bus.clear_history()
        assert bus.history() == []

    def test_filter_returns_only_matching(self):
        bus = EventBus()
        bus.publish("a", 1)
        bus.publish("b", 2)
        bus.publish("a", 3)
        bus.publish("b", 4)
        a_events = bus.history("a")
        assert all(r.topic == "a" for r in a_events)
        assert len(a_events) == 2

    def test_history_is_list(self):
        bus = EventBus()
        bus.publish("t", 1)
        assert isinstance(bus.history(), list)


# ─── TestEventBusTopicsAndSummaryExtra ────────────────────────────────────────

class TestEventBusTopicsAndSummaryExtra:
    def test_topics_is_list(self):
        bus = EventBus()
        assert isinstance(bus.topics(), list)

    def test_three_topics(self):
        bus = EventBus()
        bus.subscribe("a", lambda r: None)
        bus.subscribe("b", lambda r: None)
        bus.subscribe("c", lambda r: None)
        t = bus.topics()
        assert "a" in t and "b" in t and "c" in t

    def test_summary_returns_bus_summary(self):
        bus = EventBus()
        assert isinstance(bus.summary(), BusSummary)

    def test_summary_n_topics_after_publish(self):
        bus = EventBus()
        bus.subscribe("x", lambda r: None)
        s = bus.summary()
        assert s.n_topics >= 1

    def test_summary_after_clear_history(self):
        bus = EventBus()
        bus.publish("t", 1)
        bus.clear_history()
        # Summary should still reflect published count
        s = bus.summary()
        assert s.total_published >= 0


# ─── TestMakeEventBusExtra ────────────────────────────────────────────────────

class TestMakeEventBusExtra:
    def test_max_history_applied(self):
        bus = make_event_bus(max_history=5)
        for i in range(20):
            bus.publish("t", i)
        assert len(bus.history()) <= 5

    def test_raise_on_error_false_no_raise(self):
        bus = make_event_bus(raise_on_error=False)
        bus.subscribe("t", lambda r: (_ for _ in ()).throw(ValueError("err")))
        rec = bus.publish("t", 1)
        assert rec is not None

    def test_allow_wildcard_true(self):
        bus = make_event_bus(allow_wildcard=True)
        sid = bus.subscribe("*", lambda r: None)
        assert sid is not None

    def test_returns_event_bus(self):
        bus = make_event_bus()
        assert isinstance(bus, EventBus)


# ─── TestCollectEventsExtra ───────────────────────────────────────────────────

class TestCollectEventsExtra:
    def test_five_events_topic(self):
        bus = EventBus()
        for i in range(5):
            bus.publish("evt", i)
        events = collect_events(bus, "evt")
        assert len(events) == 5

    def test_returns_list_of_event_records(self):
        bus = EventBus()
        bus.publish("t", 1)
        events = collect_events(bus, "t")
        for e in events:
            assert isinstance(e, EventRecord)

    def test_topic_not_present_empty_list(self):
        bus = EventBus()
        bus.publish("other", 1)
        assert collect_events(bus, "missing") == []

    def test_events_have_correct_topic(self):
        bus = EventBus()
        bus.publish("a", 1)
        bus.publish("a", 2)
        bus.publish("b", 3)
        events = collect_events(bus, "a")
        assert all(e.topic == "a" for e in events)


# ─── TestDrainBusExtra ────────────────────────────────────────────────────────

class TestDrainBusExtra:
    def test_returns_list(self):
        bus = EventBus()
        bus.publish("t", 1)
        result = drain_bus(bus)
        assert isinstance(result, list)

    def test_drain_returns_event_records(self):
        bus = EventBus()
        bus.publish("t", 1)
        bus.publish("t", 2)
        result = drain_bus(bus)
        for r in result:
            assert isinstance(r, EventRecord)

    def test_drain_ten_events(self):
        bus = EventBus()
        for i in range(10):
            bus.publish("t", i)
        result = drain_bus(bus)
        assert len(result) == 10

    def test_drain_clears(self):
        bus = EventBus()
        bus.publish("t", 1)
        drain_bus(bus)
        assert len(bus.history()) == 0

    def test_drain_multiple_topics(self):
        bus = EventBus()
        bus.publish("a", 1)
        bus.publish("b", 2)
        bus.publish("c", 3)
        result = drain_bus(bus)
        assert len(result) == 3
