"""Extra tests for puzzle_reconstruction/utils/result_cache.py."""
from __future__ import annotations

import time

import pytest

from puzzle_reconstruction.utils.result_cache import (
    CachePolicy,
    CacheRecord,
    CacheSummary,
    ResultCache,
    make_cache,
    cached_result,
    merge_caches,
    evict_expired,
)


# ─── CachePolicy ─────────────────────────────────────────────────────────────

class TestCachePolicyExtra:
    def test_default_ttl(self):
        assert CachePolicy().ttl == pytest.approx(0.0)

    def test_default_max_size(self):
        assert CachePolicy().max_size == 0

    def test_default_namespace(self):
        assert CachePolicy().namespace == "default"

    def test_negative_ttl_raises(self):
        with pytest.raises(ValueError):
            CachePolicy(ttl=-1.0)

    def test_negative_max_size_raises(self):
        with pytest.raises(ValueError):
            CachePolicy(max_size=-1)

    def test_empty_namespace_raises(self):
        with pytest.raises(ValueError):
            CachePolicy(namespace="")


# ─── CacheRecord ──────────────────────────────────────────────────────────────

class TestCacheRecordExtra:
    def test_empty_key_raises(self):
        with pytest.raises(ValueError):
            CacheRecord(key="", value=1, created_at=0.0)

    def test_negative_created_at_raises(self):
        with pytest.raises(ValueError):
            CacheRecord(key="a", value=1, created_at=-1.0)

    def test_negative_ttl_raises(self):
        with pytest.raises(ValueError):
            CacheRecord(key="a", value=1, created_at=0.0, ttl=-1.0)

    def test_negative_hits_raises(self):
        with pytest.raises(ValueError):
            CacheRecord(key="a", value=1, created_at=0.0, hits=-1)

    def test_not_expired_zero_ttl(self):
        r = CacheRecord(key="a", value=1, created_at=0.0, ttl=0.0)
        assert r.is_expired(now=999999.0) is False

    def test_expired_with_ttl(self):
        r = CacheRecord(key="a", value=1, created_at=100.0, ttl=10.0)
        assert r.is_expired(now=200.0) is True

    def test_not_expired_within_ttl(self):
        r = CacheRecord(key="a", value=1, created_at=100.0, ttl=10.0)
        assert r.is_expired(now=105.0) is False


# ─── CacheSummary ─────────────────────────────────────────────────────────────

class TestCacheSummaryExtra:
    def test_negative_n_entries_raises(self):
        with pytest.raises(ValueError):
            CacheSummary(namespace="x", n_entries=-1, n_expired=0, total_hits=0)

    def test_negative_n_expired_raises(self):
        with pytest.raises(ValueError):
            CacheSummary(namespace="x", n_entries=0, n_expired=-1, total_hits=0)

    def test_negative_total_hits_raises(self):
        with pytest.raises(ValueError):
            CacheSummary(namespace="x", n_entries=0, n_expired=0, total_hits=-1)

    def test_hit_ratio_zero_when_empty(self):
        s = CacheSummary(namespace="x", n_entries=0, n_expired=0, total_hits=0)
        assert s.hit_ratio == pytest.approx(0.0)

    def test_hit_ratio_computed(self):
        s = CacheSummary(namespace="x", n_entries=4, n_expired=0, total_hits=8)
        assert s.hit_ratio == pytest.approx(2.0)


# ─── ResultCache ──────────────────────────────────────────────────────────────

class TestResultCacheExtra:
    def test_put_and_get(self):
        c = ResultCache()
        c.put("k1", 42)
        assert c.get("k1") == 42

    def test_get_missing_returns_none(self):
        c = ResultCache()
        assert c.get("missing") is None

    def test_has_true(self):
        c = ResultCache()
        c.put("k1", 1)
        assert c.has("k1") is True

    def test_has_false(self):
        c = ResultCache()
        assert c.has("missing") is False

    def test_invalidate(self):
        c = ResultCache()
        c.put("k1", 1)
        assert c.invalidate("k1") is True
        assert c.get("k1") is None

    def test_invalidate_missing(self):
        c = ResultCache()
        assert c.invalidate("missing") is False

    def test_clear(self):
        c = ResultCache()
        c.put("a", 1)
        c.put("b", 2)
        n = c.clear()
        assert n == 2 and c.size() == 0

    def test_keys(self):
        c = ResultCache()
        c.put("a", 1)
        c.put("b", 2)
        assert sorted(c.keys()) == ["a", "b"]

    def test_size(self):
        c = ResultCache()
        c.put("a", 1)
        c.put("b", 2)
        assert c.size() == 2

    def test_max_size_eviction(self):
        c = ResultCache(CachePolicy(max_size=2))
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        assert c.size() == 2
        assert c.has("c") is True

    def test_empty_key_raises(self):
        c = ResultCache()
        with pytest.raises(ValueError):
            c.put("", 1)

    def test_summarize(self):
        c = ResultCache(CachePolicy(namespace="test"))
        c.put("k1", 42)
        c.get("k1")
        s = c.summarize()
        assert s.namespace == "test"
        assert s.n_entries == 1
        assert s.total_hits >= 1

    def test_hits_increment(self):
        c = ResultCache()
        c.put("k", 1)
        c.get("k")
        c.get("k")
        s = c.summarize()
        assert s.total_hits == 2


# ─── make_cache ───────────────────────────────────────────────────────────────

class TestMakeCacheExtra:
    def test_returns_result_cache(self):
        c = make_cache()
        assert isinstance(c, ResultCache)

    def test_custom_namespace(self):
        c = make_cache(namespace="custom")
        s = c.summarize()
        assert s.namespace == "custom"


# ─── cached_result ────────────────────────────────────────────────────────────

class TestCachedResultExtra:
    def test_computes_on_miss(self):
        c = ResultCache()
        val = cached_result(c, "k", lambda: 42)
        assert val == 42

    def test_returns_cached_on_hit(self):
        c = ResultCache()
        c.put("k", 99)
        val = cached_result(c, "k", lambda: 42)
        assert val == 99

    def test_stores_after_compute(self):
        c = ResultCache()
        cached_result(c, "k", lambda: 10)
        assert c.get("k") == 10


# ─── merge_caches ─────────────────────────────────────────────────────────────

class TestMergeCachesExtra:
    def test_merge_copies_entries(self):
        src = ResultCache()
        src.put("a", 1)
        src.put("b", 2)
        tgt = ResultCache()
        n = merge_caches(tgt, src)
        assert n == 2 and tgt.get("a") == 1

    def test_merge_empty_source(self):
        tgt = ResultCache()
        n = merge_caches(tgt, ResultCache())
        assert n == 0


# ─── evict_expired ────────────────────────────────────────────────────────────

class TestEvictExpiredExtra:
    def test_no_expired(self):
        c = ResultCache()
        c.put("k", 1)
        n = evict_expired(c)
        assert n == 0

    def test_evicts_expired(self):
        c = ResultCache(CachePolicy(ttl=0.01))
        c.put("k", 1)
        time.sleep(0.05)
        n = evict_expired(c)
        assert n == 1 and c.size() == 0
