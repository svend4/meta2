"""Extra tests for puzzle_reconstruction.utils.cache_manager."""
import time
import pytest
import numpy as np

from puzzle_reconstruction.utils.cache_manager import (
    CacheEntry,
    CacheStats,
    LRUCache,
    make_cache,
    cached_call,
    merge_caches,
)


# ─── TestCacheEntryExtra ─────────────────────────────────────────────────────

class TestCacheEntryExtra:
    def test_key_and_value(self):
        e = CacheEntry(key="mykey", value=99)
        assert e.key == "mykey"
        assert e.value == 99

    def test_hits_default_zero(self):
        e = CacheEntry(key="k", value=1)
        assert e.hits == 0

    def test_ttl_default_none(self):
        e = CacheEntry(key="k", value=1)
        assert e.ttl is None

    def test_not_expired_no_ttl(self):
        e = CacheEntry(key="k", value=1)
        assert e.is_expired() is False

    def test_not_expired_long_ttl(self):
        e = CacheEntry(key="k", value=1, ttl=9999.0)
        assert e.is_expired() is False

    def test_expired_zero_ttl(self):
        e = CacheEntry(key="k", value=1, ttl=0.0)
        time.sleep(0.01)
        assert e.is_expired() is True

    def test_touch_increments(self):
        e = CacheEntry(key="k", value=1)
        e.touch()
        e.touch()
        e.touch()
        assert e.hits == 3

    def test_value_can_be_none(self):
        e = CacheEntry(key="k", value=None)
        assert e.value is None

    def test_value_can_be_list(self):
        e = CacheEntry(key="k", value=[1, 2, 3])
        assert e.value == [1, 2, 3]


# ─── TestCacheStatsExtra ─────────────────────────────────────────────────────

class TestCacheStatsExtra:
    def test_defaults_all_zero(self):
        s = CacheStats()
        assert s.hits == 0
        assert s.misses == 0
        assert s.evictions == 0
        assert s.size == 0
        assert s.capacity == 0

    def test_hit_rate_zero(self):
        s = CacheStats()
        assert s.hit_rate == pytest.approx(0.0)

    def test_hit_rate_all_hits(self):
        s = CacheStats(hits=20, misses=0)
        assert s.hit_rate == pytest.approx(1.0)

    def test_hit_rate_all_misses(self):
        s = CacheStats(hits=0, misses=20)
        assert s.hit_rate == pytest.approx(0.0)

    def test_hit_rate_50_50(self):
        s = CacheStats(hits=5, misses=5)
        assert s.hit_rate == pytest.approx(0.5)

    def test_custom_evictions(self):
        s = CacheStats(evictions=10)
        assert s.evictions == 10


# ─── TestLRUCacheExtra ───────────────────────────────────────────────────────

class TestLRUCacheExtra:
    def test_capacity_zero_raises(self):
        with pytest.raises(ValueError):
            LRUCache(capacity=0)

    def test_capacity_negative_raises(self):
        with pytest.raises(ValueError):
            LRUCache(capacity=-1)

    def test_empty_len(self):
        assert len(LRUCache(capacity=10)) == 0

    def test_put_get(self):
        c = LRUCache(capacity=10)
        c.put("x", 42)
        assert c.get("x") == 42

    def test_get_missing_none(self):
        c = LRUCache(capacity=10)
        assert c.get("no_key") is None

    def test_get_missing_custom_default(self):
        c = LRUCache(capacity=10)
        assert c.get("no_key", default=-1) == -1

    def test_hit_stat(self):
        c = LRUCache(capacity=10)
        c.put("a", 1)
        c.get("a")
        c.get("a")
        assert c.stats.hits == 2

    def test_miss_stat(self):
        c = LRUCache(capacity=10)
        c.get("nope")
        c.get("nope2")
        assert c.stats.misses == 2

    def test_contains(self):
        c = LRUCache(capacity=10)
        c.put("x", 1)
        assert "x" in c
        assert "y" not in c

    def test_delete(self):
        c = LRUCache(capacity=10)
        c.put("a", 1)
        assert c.delete("a") is True
        assert "a" not in c

    def test_delete_missing(self):
        c = LRUCache(capacity=10)
        assert c.delete("nope") is False

    def test_clear(self):
        c = LRUCache(capacity=10)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        c.clear()
        assert len(c) == 0

    def test_eviction_on_overflow(self):
        c = LRUCache(capacity=2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        assert "a" not in c
        assert "c" in c

    def test_eviction_count(self):
        c = LRUCache(capacity=2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        c.put("d", 4)
        assert c.stats.evictions == 2

    def test_lru_order(self):
        c = LRUCache(capacity=2)
        c.put("a", 1)
        c.put("b", 2)
        c.get("a")  # a is now most recent
        c.put("c", 3)  # evicts b
        assert "a" in c
        assert "b" not in c
        assert "c" in c

    def test_update_existing(self):
        c = LRUCache(capacity=5)
        c.put("x", 10)
        c.put("x", 20)
        assert c.get("x") == 20
        assert len(c) == 1

    def test_capacity_property(self):
        c = LRUCache(capacity=32)
        assert c.capacity == 32

    def test_keys(self):
        c = LRUCache(capacity=10)
        c.put("a", 1)
        c.put("b", 2)
        assert set(c.keys()) == {"a", "b"}

    def test_values(self):
        c = LRUCache(capacity=10)
        c.put("a", 1)
        c.put("b", 2)
        assert set(c.values()) == {1, 2}

    def test_resize_larger(self):
        c = LRUCache(capacity=3)
        c.put("a", 1)
        c.resize(10)
        assert c.capacity == 10
        assert "a" in c

    def test_resize_smaller(self):
        c = LRUCache(capacity=5)
        for i in range(5):
            c.put(str(i), i)
        c.resize(2)
        assert len(c) == 2

    def test_resize_zero_raises(self):
        c = LRUCache(capacity=5)
        with pytest.raises(ValueError):
            c.resize(0)

    def test_purge_expired(self):
        c = LRUCache(capacity=10)
        c.put("dead", 1, ttl=0.0)
        c.put("alive", 2, ttl=9999)
        time.sleep(0.01)
        n = c.purge_expired()
        assert n >= 1
        assert "alive" in c

    def test_to_dict(self):
        c = LRUCache(capacity=10)
        c.put("k1", "val1")
        d = c.to_dict()
        assert "k1" in d
        assert d["k1"]["value"] == "val1"

    def test_to_dict_numpy(self):
        c = LRUCache(capacity=10)
        c.put("arr", np.array([1, 2, 3]))
        d = c.to_dict()
        assert isinstance(d["arr"]["value"], list)

    def test_iter(self):
        c = LRUCache(capacity=10)
        c.put("a", 1)
        c.put("b", 2)
        assert set(c) == {"a", "b"}

    def test_stats_size(self):
        c = LRUCache(capacity=10)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        assert c.stats.size == 3

    def test_many_puts(self):
        c = LRUCache(capacity=5)
        for i in range(20):
            c.put(f"key_{i}", i)
        assert len(c) == 5


# ─── TestMakeCacheExtra ──────────────────────────────────────────────────────

class TestMakeCacheExtra:
    def test_returns_lru_cache(self):
        assert isinstance(make_cache(), LRUCache)

    def test_default_capacity(self):
        assert make_cache().capacity == 128

    def test_custom_capacity(self):
        assert make_cache(capacity=64).capacity == 64

    def test_empty_on_create(self):
        assert len(make_cache()) == 0


# ─── TestCachedCallExtra ─────────────────────────────────────────────────────

class TestCachedCallExtra:
    def test_miss_calls_fn(self):
        c = make_cache()
        result = cached_call(c, "k1", lambda: 42)
        assert result == 42

    def test_hit_returns_cached(self):
        c = make_cache()
        cached_call(c, "k1", lambda: 42)
        result = cached_call(c, "k1", lambda: 99)
        assert result == 42

    def test_fn_called_once(self):
        c = make_cache()
        count = [0]
        def fn():
            count[0] += 1
            return "x"
        cached_call(c, "k", fn)
        cached_call(c, "k", fn)
        cached_call(c, "k", fn)
        assert count[0] == 1

    def test_different_keys(self):
        c = make_cache()
        assert cached_call(c, "a", lambda: 10) == 10
        assert cached_call(c, "b", lambda: 20) == 20

    def test_fn_with_args(self):
        c = make_cache()
        result = cached_call(c, "sum", lambda a, b: a + b, 3, 4)
        assert result == 7

    def test_fn_with_kwargs(self):
        c = make_cache()
        result = cached_call(c, "mul", lambda x, y: x * y, x=3, y=5)
        assert result == 15

    def test_cache_stores_value(self):
        c = make_cache()
        cached_call(c, "k", lambda: "stored")
        assert c.get("k") == "stored"


# ─── TestMergeCachesExtra ────────────────────────────────────────────────────

class TestMergeCachesExtra:
    def test_returns_lru_cache(self):
        merged = merge_caches([make_cache(), make_cache()])
        assert isinstance(merged, LRUCache)

    def test_all_keys_present(self):
        c1 = make_cache()
        c2 = make_cache()
        c1.put("a", 1)
        c2.put("b", 2)
        merged = merge_caches([c1, c2])
        assert merged.get("a") == 1
        assert merged.get("b") == 2

    def test_later_overwrites(self):
        c1 = make_cache()
        c2 = make_cache()
        c1.put("x", "old")
        c2.put("x", "new")
        merged = merge_caches([c1, c2])
        assert merged.get("x") == "new"

    def test_empty_input(self):
        assert len(merge_caches([])) == 0

    def test_custom_capacity(self):
        merged = merge_caches([make_cache(), make_cache()], capacity=32)
        assert merged.capacity == 32

    def test_three_caches(self):
        c1 = make_cache()
        c2 = make_cache()
        c3 = make_cache()
        c1.put("a", 1)
        c2.put("b", 2)
        c3.put("c", 3)
        merged = merge_caches([c1, c2, c3])
        assert merged.get("a") == 1
        assert merged.get("b") == 2
        assert merged.get("c") == 3

    def test_single_cache(self):
        c = make_cache()
        c.put("k", "v")
        merged = merge_caches([c])
        assert merged.get("k") == "v"
