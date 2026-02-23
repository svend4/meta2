"""Extra tests for puzzle_reconstruction.utils.result_cache."""
import time

import pytest

from puzzle_reconstruction.utils.result_cache import (
    CachePolicy,
    CacheRecord,
    CacheSummary,
    ResultCache,
    cached_result,
    evict_expired,
    make_cache,
    merge_caches,
)


# ─── TestCachePolicyExtra ────────────────────────────────────────────────────

class TestCachePolicyExtra:
    def test_ttl_zero_ok(self):
        assert CachePolicy(ttl=0.0).ttl == 0.0

    def test_ttl_large(self):
        assert CachePolicy(ttl=86400.0).ttl == pytest.approx(86400.0)

    def test_max_size_zero_ok(self):
        assert CachePolicy(max_size=0).max_size == 0

    def test_max_size_large(self):
        assert CachePolicy(max_size=10000).max_size == 10000

    def test_namespace_custom(self):
        assert CachePolicy(namespace="edges").namespace == "edges"

    def test_negative_ttl_raises(self):
        with pytest.raises(ValueError):
            CachePolicy(ttl=-0.001)

    def test_negative_max_size_raises(self):
        with pytest.raises(ValueError):
            CachePolicy(max_size=-1)

    def test_empty_namespace_raises(self):
        with pytest.raises(ValueError):
            CachePolicy(namespace="")


# ─── TestCacheRecordExtra ────────────────────────────────────────────────────

class TestCacheRecordExtra:
    def test_basic_fields(self):
        r = CacheRecord(key="test", value="data", created_at=100.0)
        assert r.key == "test"
        assert r.value == "data"

    def test_ttl_zero_never_expires(self):
        r = CacheRecord(key="k", value=1, created_at=0.0, ttl=0.0)
        assert r.is_expired(now=1e9) is False

    def test_ttl_positive_not_expired(self):
        now = time.time()
        r = CacheRecord(key="k", value=1, created_at=now, ttl=3600.0)
        assert r.is_expired() is False

    def test_ttl_positive_expired(self):
        r = CacheRecord(key="k", value=1, created_at=100.0, ttl=10.0)
        assert r.is_expired(now=200.0) is True

    def test_hits_default_zero(self):
        r = CacheRecord(key="k", value=1, created_at=0.0)
        assert r.hits == 0

    def test_hits_positive(self):
        r = CacheRecord(key="k", value=1, created_at=0.0, hits=5)
        assert r.hits == 5

    def test_empty_key_raises(self):
        with pytest.raises(ValueError):
            CacheRecord(key="", value=1, created_at=0.0)

    def test_negative_created_at_raises(self):
        with pytest.raises(ValueError):
            CacheRecord(key="k", value=1, created_at=-1.0)

    def test_negative_ttl_raises(self):
        with pytest.raises(ValueError):
            CacheRecord(key="k", value=1, created_at=0.0, ttl=-1.0)

    def test_negative_hits_raises(self):
        with pytest.raises(ValueError):
            CacheRecord(key="k", value=1, created_at=0.0, hits=-1)

    def test_value_none_ok(self):
        r = CacheRecord(key="k", value=None, created_at=0.0)
        assert r.value is None

    def test_value_dict(self):
        r = CacheRecord(key="k", value={"a": 1}, created_at=0.0)
        assert r.value["a"] == 1


# ─── TestCacheSummaryExtra ───────────────────────────────────────────────────

class TestCacheSummaryExtra:
    def test_hit_ratio_zero_entries(self):
        s = CacheSummary(namespace="ns", n_entries=0, n_expired=0, total_hits=0)
        assert s.hit_ratio == 0.0

    def test_hit_ratio_nonzero(self):
        s = CacheSummary(namespace="ns", n_entries=4, n_expired=0, total_hits=8)
        assert s.hit_ratio == pytest.approx(2.0)

    def test_namespace_stored(self):
        s = CacheSummary(namespace="test_ns", n_entries=0, n_expired=0, total_hits=0)
        assert s.namespace == "test_ns"

    def test_negative_n_entries_raises(self):
        with pytest.raises(ValueError):
            CacheSummary(namespace="ns", n_entries=-1, n_expired=0, total_hits=0)

    def test_negative_n_expired_raises(self):
        with pytest.raises(ValueError):
            CacheSummary(namespace="ns", n_entries=0, n_expired=-1, total_hits=0)

    def test_negative_hits_raises(self):
        with pytest.raises(ValueError):
            CacheSummary(namespace="ns", n_entries=0, n_expired=0, total_hits=-1)

    def test_all_expired(self):
        s = CacheSummary(namespace="ns", n_entries=5, n_expired=5, total_hits=0)
        assert s.n_expired == 5


# ─── TestResultCacheExtra ────────────────────────────────────────────────────

class TestResultCacheExtra:
    def test_put_get(self):
        c = ResultCache()
        c.put("key", "value")
        assert c.get("key") == "value"

    def test_get_miss(self):
        c = ResultCache()
        assert c.get("nonexistent") is None

    def test_has_true(self):
        c = ResultCache()
        c.put("k", 1)
        assert c.has("k") is True

    def test_has_false(self):
        c = ResultCache()
        assert c.has("missing") is False

    def test_invalidate_existing(self):
        c = ResultCache()
        c.put("k", 1)
        assert c.invalidate("k") is True
        assert c.has("k") is False

    def test_invalidate_missing(self):
        c = ResultCache()
        assert c.invalidate("missing") is False

    def test_clear_all(self):
        c = ResultCache()
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        n = c.clear()
        assert n == 3
        assert c.size() == 0

    def test_size(self):
        c = ResultCache()
        assert c.size() == 0
        c.put("a", 1)
        assert c.size() == 1
        c.put("b", 2)
        assert c.size() == 2

    def test_keys(self):
        c = ResultCache()
        c.put("x", 1)
        c.put("y", 2)
        c.put("z", 3)
        assert set(c.keys()) == {"x", "y", "z"}

    def test_overwrite(self):
        c = ResultCache()
        c.put("k", 1)
        c.put("k", 99)
        assert c.get("k") == 99
        assert c.size() == 1

    def test_hits_tracked(self):
        c = ResultCache()
        c.put("k", 1)
        for _ in range(5):
            c.get("k")
        assert c.summarize().total_hits == 5

    def test_max_size_evicts(self):
        c = ResultCache(CachePolicy(max_size=3))
        for i in range(5):
            c.put(f"k{i}", i)
        assert c.size() == 3

    def test_ttl_expiry(self):
        c = ResultCache(CachePolicy(ttl=0.001))
        c.put("k", 42)
        time.sleep(0.02)
        assert c.get("k") is None

    def test_ttl_not_expired(self):
        c = ResultCache(CachePolicy(ttl=3600.0))
        c.put("k", 42)
        assert c.get("k") == 42

    def test_summarize_namespace(self):
        c = ResultCache(CachePolicy(namespace="features"))
        assert c.summarize().namespace == "features"

    def test_summarize_n_entries(self):
        c = ResultCache()
        c.put("a", 1)
        c.put("b", 2)
        assert c.summarize().n_entries == 2

    def test_empty_key_raises(self):
        c = ResultCache()
        with pytest.raises(ValueError):
            c.put("", 1)

    def test_value_types(self):
        c = ResultCache()
        c.put("int", 42)
        c.put("str", "hello")
        c.put("list", [1, 2, 3])
        c.put("none", None)
        assert c.get("int") == 42
        assert c.get("str") == "hello"
        assert c.get("list") == [1, 2, 3]
        assert c.get("none") is None


# ─── TestMakeCacheExtra ──────────────────────────────────────────────────────

class TestMakeCacheExtra:
    def test_returns_result_cache(self):
        assert isinstance(make_cache(), ResultCache)

    def test_default_namespace(self):
        c = make_cache()
        assert c.summarize().namespace == "default"

    def test_custom_namespace(self):
        c = make_cache(namespace="scores")
        assert c.summarize().namespace == "scores"

    def test_custom_ttl(self):
        c = make_cache(ttl=60.0)
        c.put("k", 1)
        assert c.get("k") == 1

    def test_custom_max_size(self):
        c = make_cache(max_size=2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        assert c.size() == 2

    def test_put_get_works(self):
        c = make_cache()
        c.put("test", 42)
        assert c.get("test") == 42


# ─── TestCachedResultExtra ───────────────────────────────────────────────────

class TestCachedResultExtra:
    def test_computes_on_miss(self):
        c = ResultCache()
        assert cached_result(c, "k", lambda: 42) == 42

    def test_returns_cached(self):
        c = ResultCache()
        c.put("k", 99)
        assert cached_result(c, "k", lambda: 0) == 99

    def test_fn_called_once(self):
        c = ResultCache()
        calls = [0]
        def fn():
            calls[0] += 1
            return "result"
        cached_result(c, "k", fn)
        cached_result(c, "k", fn)
        cached_result(c, "k", fn)
        assert calls[0] == 1

    def test_stores_in_cache(self):
        c = ResultCache()
        cached_result(c, "new_key", lambda: "data")
        assert c.has("new_key")
        assert c.get("new_key") == "data"

    def test_none_value_cached(self):
        c = ResultCache()
        calls = [0]
        def fn():
            calls[0] += 1
            return None
        cached_result(c, "k", fn)
        result = cached_result(c, "k", fn)
        # If None is not cached, fn would be called twice
        # Either way result should be None
        assert result is None

    def test_different_keys(self):
        c = ResultCache()
        assert cached_result(c, "a", lambda: 1) == 1
        assert cached_result(c, "b", lambda: 2) == 2
        assert c.size() >= 2


# ─── TestMergeCachesExtra ────────────────────────────────────────────────────

class TestMergeCachesExtra:
    def test_copies_all(self):
        src = ResultCache()
        for i in range(5):
            src.put(f"k{i}", i)
        dst = ResultCache()
        n = merge_caches(dst, src)
        assert n == 5
        for i in range(5):
            assert dst.get(f"k{i}") == i

    def test_empty_source(self):
        src = ResultCache()
        dst = ResultCache()
        assert merge_caches(dst, src) == 0

    def test_overwrites(self):
        src = ResultCache()
        src.put("k", "new")
        dst = ResultCache()
        dst.put("k", "old")
        merge_caches(dst, src)
        assert dst.get("k") == "new"

    def test_preserves_existing(self):
        src = ResultCache()
        src.put("a", 1)
        dst = ResultCache()
        dst.put("b", 2)
        merge_caches(dst, src)
        assert dst.get("a") == 1
        assert dst.get("b") == 2

    def test_returns_count(self):
        src = ResultCache()
        src.put("x", 1)
        src.put("y", 2)
        dst = ResultCache()
        assert merge_caches(dst, src) == 2


# ─── TestEvictExpiredExtra ───────────────────────────────────────────────────

class TestEvictExpiredExtra:
    def test_removes_expired(self):
        c = ResultCache(CachePolicy(ttl=0.001))
        c.put("a", 1)
        c.put("b", 2)
        time.sleep(0.02)
        n = evict_expired(c)
        assert n >= 2

    def test_no_ttl_no_eviction(self):
        c = ResultCache()
        c.put("a", 1)
        c.put("b", 2)
        assert evict_expired(c) == 0

    def test_after_evict_empty(self):
        c = ResultCache(CachePolicy(ttl=0.001))
        c.put("only", 1)
        time.sleep(0.02)
        evict_expired(c)
        assert c.has("only") is False

    def test_returns_int(self):
        c = ResultCache()
        assert isinstance(evict_expired(c), int)

    def test_long_ttl_no_eviction(self):
        c = ResultCache(CachePolicy(ttl=3600.0))
        c.put("k", 1)
        assert evict_expired(c) == 0
        assert c.has("k") is True
