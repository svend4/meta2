"""Extra tests for puzzle_reconstruction/utils/cache_manager.py"""
import time

import numpy as np
import pytest

from puzzle_reconstruction.utils.cache_manager import (
    CacheEntry,
    CacheStats,
    LRUCache,
    cached_call,
    make_cache,
    merge_caches,
)


# ─── TestCacheEntryExtra ──────────────────────────────────────────────────────

class TestCacheEntryExtra:
    def test_value_list(self):
        e = CacheEntry(key="k", value=[1, 2, 3])
        assert e.value == [1, 2, 3]

    def test_value_dict(self):
        e = CacheEntry(key="k", value={"a": 1})
        assert e.value["a"] == 1

    def test_value_ndarray(self):
        arr = np.zeros((3, 3))
        e = CacheEntry(key="k", value=arr)
        assert np.array_equal(e.value, arr)

    def test_large_ttl_not_expired(self):
        e = CacheEntry(key="k", value=1, ttl=1e6)
        assert e.is_expired() is False

    def test_very_small_ttl_expires(self):
        e = CacheEntry(key="k", value=1, ttl=0.001)
        time.sleep(0.02)
        assert e.is_expired() is True

    def test_touch_many_times(self):
        e = CacheEntry(key="k", value=1)
        for _ in range(10):
            e.touch()
        assert e.hits == 10

    def test_created_positive(self):
        e = CacheEntry(key="k", value=1)
        assert e.created > 0.0

    def test_key_can_be_int(self):
        e = CacheEntry(key=42, value="val")
        assert e.key == 42


# ─── TestCacheStatsExtra ──────────────────────────────────────────────────────

class TestCacheStatsExtra:
    def test_hit_rate_large_numbers(self):
        s = CacheStats(hits=1000, misses=1000)
        assert s.hit_rate == pytest.approx(0.5)

    def test_hit_rate_all_misses(self):
        s = CacheStats(hits=0, misses=5)
        assert s.hit_rate == pytest.approx(0.0)

    def test_evictions_tracked(self):
        s = CacheStats(evictions=7)
        assert s.evictions == 7

    def test_capacity_zero_valid(self):
        s = CacheStats(size=0, capacity=0)
        assert s.capacity == 0


# ─── TestLRUCacheCapacityExtra ────────────────────────────────────────────────

class TestLRUCacheCapacityExtra:
    def test_capacity_one_stores_last(self):
        c = LRUCache(capacity=1)
        c.put("a", 1)
        c.put("b", 2)
        assert c.get("a") is None
        assert c.get("b") == 2

    def test_large_capacity(self):
        c = LRUCache(capacity=1000)
        for i in range(500):
            c.put(i, i * 2)
        assert len(c) == 500

    def test_fill_to_capacity(self):
        c = LRUCache(capacity=5)
        for i in range(5):
            c.put(i, i)
        assert len(c) == 5

    def test_overfill_evicts_lru(self):
        c = LRUCache(capacity=3)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        c.put("d", 4)
        assert "a" not in c
        assert len(c) == 3


# ─── TestLRUCacheResizeExtra ──────────────────────────────────────────────────

class TestLRUCacheResizeExtra:
    def test_resize_to_same_size(self):
        c = LRUCache(capacity=4)
        c.put("a", 1)
        c.resize(4)
        assert c.capacity == 4
        assert "a" in c

    def test_resize_to_one_keeps_last(self):
        c = LRUCache(capacity=4)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        c.get("c")  # promote c
        c.resize(1)
        assert len(c) == 1

    def test_resize_negative_raises(self):
        c = LRUCache(capacity=4)
        with pytest.raises(ValueError):
            c.resize(-5)


# ─── TestLRUCacheToDictExtra ──────────────────────────────────────────────────

class TestLRUCacheToDictExtra:
    def test_to_dict_none_value(self):
        c = LRUCache(capacity=4)
        c.put("n", None)
        d = c.to_dict()
        assert "n" in d
        assert d["n"]["value"] is None

    def test_to_dict_multiple_keys(self):
        c = LRUCache(capacity=8)
        for i in range(4):
            c.put(f"k{i}", i * 10)
        d = c.to_dict()
        assert len(d) == 4

    def test_to_dict_hits_recorded(self):
        c = LRUCache(capacity=4)
        c.put("x", 99)
        c.get("x")
        c.get("x")
        d = c.to_dict()
        assert d["x"]["hits"] >= 2

    def test_to_dict_string_value(self):
        c = LRUCache(capacity=4)
        c.put("s", "hello")
        d = c.to_dict()
        assert d["s"]["value"] == "hello"


# ─── TestMakeCacheExtra ───────────────────────────────────────────────────────

class TestMakeCacheExtra:
    def test_default_capacity_128(self):
        c = make_cache()
        assert c.capacity == 128

    def test_capacity_1(self):
        c = make_cache(capacity=1)
        assert c.capacity == 1

    def test_with_long_ttl(self):
        c = make_cache(capacity=4, ttl=3600.0)
        c.put("x", 42)
        assert c.get("x") == 42

    def test_zero_ttl_expires_quickly(self):
        c = make_cache(capacity=4, ttl=0.0)
        c.put("x", 42)
        time.sleep(0.02)
        assert c.get("x") is None


# ─── TestCachedCallExtra ──────────────────────────────────────────────────────

class TestCachedCallExtra:
    def test_none_value_cached(self):
        c = LRUCache(capacity=4)
        calls = [0]

        def fn():
            calls[0] += 1
            return None

        cached_call(c, "k", fn)
        cached_call(c, "k", fn)
        # None result is still cached; fn should be called only once
        assert calls[0] == 1

    def test_large_return_value_cached(self):
        c = LRUCache(capacity=4)
        arr = np.zeros((100, 100))
        result = cached_call(c, "big", lambda: arr)
        assert np.array_equal(result, arr)

    def test_multiple_keys_independent(self):
        c = LRUCache(capacity=8)
        for i in range(5):
            r = cached_call(c, f"k{i}", lambda v=i: v * 3)
            assert r == i * 3

    def test_overwrite_with_new_key(self):
        c = LRUCache(capacity=4)
        cached_call(c, "a", lambda: 1)
        cached_call(c, "b", lambda: 2)
        assert c.get("a") == 1
        assert c.get("b") == 2


# ─── TestMergeCachesExtra ─────────────────────────────────────────────────────

class TestMergeCachesExtra:
    def test_three_caches_merged(self):
        c1, c2, c3 = LRUCache(4), LRUCache(4), LRUCache(4)
        c1.put("a", 1)
        c2.put("b", 2)
        c3.put("c", 3)
        merged = merge_caches([c1, c2, c3])
        assert merged.get("a") == 1
        assert merged.get("b") == 2
        assert merged.get("c") == 3

    def test_default_capacity_of_merged(self):
        c1 = LRUCache(capacity=4)
        c1.put("x", 10)
        merged = merge_caches([c1])
        assert merged.capacity >= 1

    def test_empty_caches_merged(self):
        c1 = LRUCache(capacity=4)
        c2 = LRUCache(capacity=4)
        merged = merge_caches([c1, c2])
        assert len(merged) == 0

    def test_merged_does_not_exceed_capacity(self):
        c1 = LRUCache(capacity=8)
        for i in range(5):
            c1.put(i, i)
        merged = merge_caches([c1], capacity=3)
        assert len(merged) <= 3
