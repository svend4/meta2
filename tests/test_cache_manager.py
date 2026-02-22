"""Тесты для puzzle_reconstruction.utils.cache_manager."""
import time

import numpy as np
import pytest

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
    def test_basic_creation(self):
        e = CacheEntry(key="k", value=42)
        assert e.key == "k"
        assert e.value == 42
        assert e.hits == 0

    def test_ttl_none_not_expired(self):
        e = CacheEntry(key="k", value=1, ttl=None)
        assert e.is_expired() is False

    def test_ttl_large_not_expired(self):
        e = CacheEntry(key="k", value=1, ttl=9999.0)
        assert e.is_expired() is False

    def test_ttl_zero_expired(self):
        e = CacheEntry(key="k", value=1, ttl=0.0)
        time.sleep(0.01)
        assert e.is_expired() is True

    def test_touch_increments_hits(self):
        e = CacheEntry(key="k", value=1)
        e.touch()
        e.touch()
        assert e.hits == 2

    def test_created_is_float(self):
        e = CacheEntry(key="k", value=1)
        assert isinstance(e.created, float)

    def test_value_can_be_none(self):
        e = CacheEntry(key="k", value=None)
        assert e.value is None


# ─── TestCacheStats ───────────────────────────────────────────────────────────

class TestCacheStats:
    def test_default_zeros(self):
        s = CacheStats()
        assert s.hits == 0 and s.misses == 0 and s.evictions == 0

    def test_hit_rate_zero_when_empty(self):
        s = CacheStats()
        assert s.hit_rate == pytest.approx(0.0)

    def test_hit_rate_all_hits(self):
        s = CacheStats(hits=10, misses=0)
        assert s.hit_rate == pytest.approx(1.0)

    def test_hit_rate_half(self):
        s = CacheStats(hits=5, misses=5)
        assert s.hit_rate == pytest.approx(0.5)

    def test_size_field(self):
        s = CacheStats(size=3, capacity=10)
        assert s.size == 3
        assert s.capacity == 10


# ─── TestLRUCacheCreation ─────────────────────────────────────────────────────

class TestLRUCacheCreation:
    def test_default_capacity(self):
        c = LRUCache()
        assert c.capacity == 128

    def test_custom_capacity(self):
        c = LRUCache(capacity=4)
        assert c.capacity == 4

    def test_zero_capacity_raises(self):
        with pytest.raises(ValueError):
            LRUCache(capacity=0)

    def test_negative_capacity_raises(self):
        with pytest.raises(ValueError):
            LRUCache(capacity=-1)

    def test_initial_len_zero(self):
        c = LRUCache(capacity=4)
        assert len(c) == 0


# ─── TestLRUCacheGetPut ───────────────────────────────────────────────────────

class TestLRUCacheGetPut:
    def test_put_and_get(self):
        c = LRUCache(capacity=4)
        c.put("a", 1)
        assert c.get("a") == 1

    def test_get_missing_returns_default(self):
        c = LRUCache(capacity=4)
        assert c.get("x") is None

    def test_get_missing_custom_default(self):
        c = LRUCache(capacity=4)
        assert c.get("x", default=-1) == -1

    def test_contains_after_put(self):
        c = LRUCache(capacity=4)
        c.put("a", 99)
        assert "a" in c

    def test_not_contains_before_put(self):
        c = LRUCache(capacity=4)
        assert "z" not in c

    def test_overwrite_updates_value(self):
        c = LRUCache(capacity=4)
        c.put("a", 1)
        c.put("a", 2)
        assert c.get("a") == 2

    def test_lru_eviction(self):
        c = LRUCache(capacity=2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)  # evicts "a"
        assert c.get("a") is None
        assert c.get("b") == 2
        assert c.get("c") == 3

    def test_get_promotes_to_end(self):
        c = LRUCache(capacity=2)
        c.put("a", 1)
        c.put("b", 2)
        c.get("a")       # promote "a"
        c.put("c", 3)    # evicts "b" (now LRU)
        assert c.get("a") == 1
        assert c.get("b") is None


# ─── TestLRUCacheDelete ───────────────────────────────────────────────────────

class TestLRUCacheDelete:
    def test_delete_existing(self):
        c = LRUCache(capacity=4)
        c.put("a", 1)
        assert c.delete("a") is True
        assert "a" not in c

    def test_delete_missing(self):
        c = LRUCache(capacity=4)
        assert c.delete("z") is False

    def test_len_decreases_after_delete(self):
        c = LRUCache(capacity=4)
        c.put("a", 1)
        c.put("b", 2)
        c.delete("a")
        assert len(c) == 1

    def test_clear_empties_cache(self):
        c = LRUCache(capacity=4)
        c.put("a", 1)
        c.put("b", 2)
        c.clear()
        assert len(c) == 0


# ─── TestLRUCacheKeysValues ───────────────────────────────────────────────────

class TestLRUCacheKeysValues:
    def test_keys_returns_list(self):
        c = LRUCache(capacity=4)
        c.put("a", 1)
        c.put("b", 2)
        keys = c.keys()
        assert isinstance(keys, list)
        assert set(keys) == {"a", "b"}

    def test_values_returns_list(self):
        c = LRUCache(capacity=4)
        c.put("x", 10)
        c.put("y", 20)
        values = c.values()
        assert set(values) == {10, 20}

    def test_keys_count(self):
        c = LRUCache(capacity=8)
        for i in range(5):
            c.put(i, i * 2)
        assert len(c.keys()) == 5


# ─── TestLRUCacheTTL ──────────────────────────────────────────────────────────

class TestLRUCacheTTL:
    def test_expired_entry_miss(self):
        c = LRUCache(capacity=4, default_ttl=0.0)
        c.put("a", 1)
        time.sleep(0.02)
        assert c.get("a") is None

    def test_non_expired_entry_hit(self):
        c = LRUCache(capacity=4, default_ttl=999.0)
        c.put("a", 42)
        assert c.get("a") == 42

    def test_purge_expired_count(self):
        c = LRUCache(capacity=8, default_ttl=0.0)
        c.put("a", 1)
        c.put("b", 2)
        time.sleep(0.02)
        removed = c.purge_expired()
        assert removed == 2

    def test_purge_no_expired(self):
        c = LRUCache(capacity=4)
        c.put("a", 1)
        removed = c.purge_expired()
        assert removed == 0


# ─── TestLRUCacheResize ───────────────────────────────────────────────────────

class TestLRUCacheResize:
    def test_resize_larger(self):
        c = LRUCache(capacity=4)
        c.resize(8)
        assert c.capacity == 8

    def test_resize_smaller_evicts(self):
        c = LRUCache(capacity=4)
        for i in range(4):
            c.put(i, i)
        c.resize(2)
        assert len(c) == 2

    def test_resize_zero_raises(self):
        c = LRUCache(capacity=4)
        with pytest.raises(ValueError):
            c.resize(0)


# ─── TestLRUCacheStats ────────────────────────────────────────────────────────

class TestLRUCacheStats:
    def test_hit_recorded(self):
        c = LRUCache(capacity=4)
        c.put("a", 1)
        c.get("a")
        assert c.stats.hits == 1

    def test_miss_recorded(self):
        c = LRUCache(capacity=4)
        c.get("missing")
        assert c.stats.misses == 1

    def test_eviction_recorded(self):
        c = LRUCache(capacity=2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        assert c.stats.evictions >= 1

    def test_to_dict_non_expired(self):
        c = LRUCache(capacity=4)
        c.put("a", 1)
        d = c.to_dict()
        assert "a" in d
        assert d["a"]["value"] == 1

    def test_to_dict_ndarray_converted(self):
        c = LRUCache(capacity=4)
        c.put("arr", np.array([1, 2, 3]))
        d = c.to_dict()
        assert isinstance(d["arr"]["value"], list)


# ─── TestMakeCache ────────────────────────────────────────────────────────────

class TestMakeCache:
    def test_returns_lru_cache(self):
        c = make_cache(capacity=16)
        assert isinstance(c, LRUCache)

    def test_capacity_set(self):
        c = make_cache(capacity=32)
        assert c.capacity == 32

    def test_ttl_passed(self):
        c = make_cache(capacity=4, ttl=60.0)
        c.put("k", "v")
        # Still valid after immediate access
        assert c.get("k") == "v"


# ─── TestCachedCall ───────────────────────────────────────────────────────────

class TestCachedCall:
    def test_calls_function_on_miss(self):
        c = LRUCache(capacity=4)
        result = cached_call(c, "k1", lambda: 42)
        assert result == 42

    def test_caches_result(self):
        c = LRUCache(capacity=4)
        call_count = [0]

        def fn():
            call_count[0] += 1
            return 99

        cached_call(c, "k1", fn)
        cached_call(c, "k1", fn)
        assert call_count[0] == 1

    def test_different_keys_call_fn(self):
        c = LRUCache(capacity=4)
        r1 = cached_call(c, "a", lambda: 1)
        r2 = cached_call(c, "b", lambda: 2)
        assert r1 == 1 and r2 == 2

    def test_args_passed(self):
        c = LRUCache(capacity=4)
        result = cached_call(c, "sum", lambda x, y: x + y, 3, 4)
        assert result == 7

    def test_returns_cached_not_recomputed(self):
        c = LRUCache(capacity=4)
        cached_call(c, "k", lambda: "first")
        result = cached_call(c, "k", lambda: "second")
        assert result == "first"


# ─── TestMergeCaches ──────────────────────────────────────────────────────────

class TestMergeCaches:
    def test_returns_lru_cache(self):
        c1 = LRUCache(capacity=4)
        result = merge_caches([c1])
        assert isinstance(result, LRUCache)

    def test_merged_contains_keys(self):
        c1 = LRUCache(capacity=4)
        c1.put("a", 1)
        c2 = LRUCache(capacity=4)
        c2.put("b", 2)
        merged = merge_caches([c1, c2])
        assert merged.get("a") == 1
        assert merged.get("b") == 2

    def test_later_cache_overwrites(self):
        c1 = LRUCache(capacity=4)
        c1.put("k", "old")
        c2 = LRUCache(capacity=4)
        c2.put("k", "new")
        merged = merge_caches([c1, c2])
        assert merged.get("k") == "new"

    def test_capacity_of_merged(self):
        c1 = LRUCache(capacity=4)
        merged = merge_caches([c1], capacity=64)
        assert merged.capacity == 64

    def test_empty_list(self):
        merged = merge_caches([])
        assert len(merged) == 0
