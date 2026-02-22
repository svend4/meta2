"""Tests for puzzle_reconstruction/utils/cache_manager.py"""
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


# ─── TestCacheEntry ───────────────────────────────────────────────────────────

class TestCacheEntry:
    def test_construction(self):
        e = CacheEntry(key="k", value=42)
        assert e.key == "k"
        assert e.value == 42
        assert e.hits == 0
        assert e.ttl is None

    def test_not_expired_no_ttl(self):
        e = CacheEntry(key="k", value=1)
        assert e.is_expired() is False

    def test_not_expired_long_ttl(self):
        e = CacheEntry(key="k", value=1, ttl=9999.0)
        assert e.is_expired() is False

    def test_expired_with_zero_ttl(self):
        e = CacheEntry(key="k", value=1, ttl=0.0)
        # TTL=0 → immediately expired
        time.sleep(0.01)
        assert e.is_expired() is True

    def test_touch_increments_hits(self):
        e = CacheEntry(key="k", value=1)
        e.touch()
        assert e.hits == 1
        e.touch()
        assert e.hits == 2


# ─── TestCacheStats ───────────────────────────────────────────────────────────

class TestCacheStats:
    def test_defaults(self):
        s = CacheStats()
        assert s.hits == 0
        assert s.misses == 0
        assert s.evictions == 0
        assert s.size == 0
        assert s.capacity == 0

    def test_hit_rate_zero_total(self):
        s = CacheStats()
        assert s.hit_rate == pytest.approx(0.0)

    def test_hit_rate_all_hits(self):
        s = CacheStats(hits=10, misses=0)
        assert s.hit_rate == pytest.approx(1.0)

    def test_hit_rate_all_misses(self):
        s = CacheStats(hits=0, misses=10)
        assert s.hit_rate == pytest.approx(0.0)

    def test_hit_rate_mixed(self):
        s = CacheStats(hits=3, misses=1)
        assert s.hit_rate == pytest.approx(0.75)


# ─── TestLRUCache ─────────────────────────────────────────────────────────────

class TestLRUCache:
    def test_capacity_zero_raises(self):
        with pytest.raises(ValueError):
            LRUCache(capacity=0)

    def test_capacity_negative_raises(self):
        with pytest.raises(ValueError):
            LRUCache(capacity=-1)

    def test_len_empty(self):
        c = LRUCache(capacity=10)
        assert len(c) == 0

    def test_put_get_basic(self):
        c = LRUCache(capacity=10)
        c.put("a", 42)
        assert c.get("a") == 42

    def test_get_missing_returns_default(self):
        c = LRUCache(capacity=10)
        assert c.get("missing") is None

    def test_get_missing_custom_default(self):
        c = LRUCache(capacity=10)
        assert c.get("x", default="fallback") == "fallback"

    def test_get_hit_increments_stat(self):
        c = LRUCache(capacity=10)
        c.put("a", 1)
        c.get("a")
        assert c.stats.hits == 1

    def test_get_miss_increments_stat(self):
        c = LRUCache(capacity=10)
        c.get("missing")
        assert c.stats.misses == 1

    def test_contains_true(self):
        c = LRUCache(capacity=10)
        c.put("x", 99)
        assert "x" in c

    def test_contains_false(self):
        c = LRUCache(capacity=10)
        assert "y" not in c

    def test_delete_existing(self):
        c = LRUCache(capacity=10)
        c.put("a", 1)
        assert c.delete("a") is True
        assert "a" not in c

    def test_delete_missing_returns_false(self):
        c = LRUCache(capacity=10)
        assert c.delete("nonexistent") is False

    def test_clear(self):
        c = LRUCache(capacity=10)
        c.put("a", 1)
        c.put("b", 2)
        c.clear()
        assert len(c) == 0

    def test_eviction_on_overflow(self):
        c = LRUCache(capacity=3)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        c.put("d", 4)  # should evict "a"
        assert "a" not in c
        assert "d" in c

    def test_eviction_count_increments(self):
        c = LRUCache(capacity=2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)  # evicts "a"
        assert c.stats.evictions == 1

    def test_lru_order_preserves_recent(self):
        c = LRUCache(capacity=2)
        c.put("a", 1)
        c.put("b", 2)
        c.get("a")       # "a" is now most recent
        c.put("c", 3)    # should evict "b"
        assert "a" in c
        assert "c" in c
        assert "b" not in c

    def test_update_existing_key(self):
        c = LRUCache(capacity=5)
        c.put("x", 10)
        c.put("x", 20)
        assert c.get("x") == 20
        assert len(c) == 1

    def test_capacity_property(self):
        c = LRUCache(capacity=64)
        assert c.capacity == 64

    def test_keys_list(self):
        c = LRUCache(capacity=10)
        c.put("a", 1)
        c.put("b", 2)
        keys = c.keys()
        assert "a" in keys
        assert "b" in keys

    def test_values_list(self):
        c = LRUCache(capacity=10)
        c.put("a", 1)
        c.put("b", 2)
        vals = c.values()
        assert 1 in vals
        assert 2 in vals

    def test_resize_larger(self):
        c = LRUCache(capacity=3)
        c.put("a", 1)
        c.resize(10)
        assert c.capacity == 10
        assert "a" in c

    def test_resize_smaller_evicts(self):
        c = LRUCache(capacity=5)
        for i in range(5):
            c.put(i, i)
        c.resize(3)
        assert len(c) == 3

    def test_resize_zero_raises(self):
        c = LRUCache(capacity=5)
        with pytest.raises(ValueError):
            c.resize(0)

    def test_purge_expired(self):
        c = LRUCache(capacity=10)
        c.put("a", 1, ttl=0.0)   # immediately expired
        c.put("b", 2, ttl=9999)  # not expired
        time.sleep(0.01)
        n = c.purge_expired()
        assert n >= 1
        assert "b" in c

    def test_to_dict(self):
        c = LRUCache(capacity=10)
        c.put("k1", "val1")
        d = c.to_dict()
        assert "k1" in d
        assert d["k1"]["value"] == "val1"

    def test_to_dict_numpy_converted(self):
        c = LRUCache(capacity=10)
        arr = np.array([1, 2, 3])
        c.put("arr", arr)
        d = c.to_dict()
        assert isinstance(d["arr"]["value"], list)

    def test_iter(self):
        c = LRUCache(capacity=10)
        c.put("a", 1)
        c.put("b", 2)
        keys = list(c)
        assert set(keys) == {"a", "b"}

    def test_stats_size_correct(self):
        c = LRUCache(capacity=10)
        c.put("a", 1)
        c.put("b", 2)
        assert c.stats.size == 2


# ─── TestMakeCache ────────────────────────────────────────────────────────────

class TestMakeCache:
    def test_returns_lru_cache(self):
        c = make_cache(capacity=16)
        assert isinstance(c, LRUCache)

    def test_capacity_set(self):
        c = make_cache(capacity=32)
        assert c.capacity == 32

    def test_default_capacity(self):
        c = make_cache()
        assert c.capacity == 128


# ─── TestCachedCall ───────────────────────────────────────────────────────────

class TestCachedCall:
    def test_calls_fn_on_miss(self):
        c = make_cache()
        result = cached_call(c, "k1", lambda: 42)
        assert result == 42

    def test_returns_cached_on_hit(self):
        c = make_cache()
        cached_call(c, "k1", lambda: 42)
        result = cached_call(c, "k1", lambda: 99)
        assert result == 42  # cached value, not new fn result

    def test_fn_called_only_once(self):
        c = make_cache()
        call_count = [0]

        def fn():
            call_count[0] += 1
            return "result"

        cached_call(c, "k", fn)
        cached_call(c, "k", fn)
        assert call_count[0] == 1

    def test_different_keys_independent(self):
        c = make_cache()
        r1 = cached_call(c, "k1", lambda: 10)
        r2 = cached_call(c, "k2", lambda: 20)
        assert r1 == 10
        assert r2 == 20

    def test_fn_with_args(self):
        c = make_cache()
        result = cached_call(c, "sum", lambda a, b: a + b, 3, 4)
        assert result == 7

    def test_fn_with_kwargs(self):
        c = make_cache()
        result = cached_call(c, "mul", lambda x, y: x * y, x=3, y=5)
        assert result == 15


# ─── TestMergeCaches ──────────────────────────────────────────────────────────

class TestMergeCaches:
    def test_returns_lru_cache(self):
        c1 = make_cache()
        c2 = make_cache()
        merged = merge_caches([c1, c2])
        assert isinstance(merged, LRUCache)

    def test_merged_has_all_keys(self):
        c1 = make_cache()
        c2 = make_cache()
        c1.put("a", 1)
        c2.put("b", 2)
        merged = merge_caches([c1, c2])
        assert merged.get("a") == 1
        assert merged.get("b") == 2

    def test_later_cache_overwrites(self):
        c1 = make_cache()
        c2 = make_cache()
        c1.put("x", "old")
        c2.put("x", "new")
        merged = merge_caches([c1, c2])
        assert merged.get("x") == "new"

    def test_empty_caches(self):
        merged = merge_caches([])
        assert len(merged) == 0

    def test_capacity_set(self):
        c1 = make_cache()
        c2 = make_cache()
        merged = merge_caches([c1, c2], capacity=64)
        assert merged.capacity == 64
