"""Тесты для puzzle_reconstruction.utils.result_cache."""
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


# ─── TestCachePolicy ──────────────────────────────────────────────────────────

class TestCachePolicy:
    def test_defaults(self):
        p = CachePolicy()
        assert p.ttl == 0.0
        assert p.max_size == 0
        assert p.namespace == "default"

    def test_negative_ttl_raises(self):
        with pytest.raises(ValueError):
            CachePolicy(ttl=-1.0)

    def test_negative_max_size_raises(self):
        with pytest.raises(ValueError):
            CachePolicy(max_size=-1)

    def test_empty_namespace_raises(self):
        with pytest.raises(ValueError):
            CachePolicy(namespace="")

    def test_zero_ttl_ok(self):
        p = CachePolicy(ttl=0.0)
        assert p.ttl == 0.0

    def test_custom_values(self):
        p = CachePolicy(ttl=60.0, max_size=100, namespace="features")
        assert p.ttl == 60.0
        assert p.max_size == 100
        assert p.namespace == "features"


# ─── TestCacheRecord ──────────────────────────────────────────────────────────

class TestCacheRecord:
    def _make(self, ttl=0.0, hits=0):
        return CacheRecord(key="k1", value=42, created_at=1000.0,
                           ttl=ttl, hits=hits)

    def test_basic_construction(self):
        r = self._make()
        assert r.key == "k1"
        assert r.value == 42

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

    def test_not_expired_ttl_0(self):
        r = self._make(ttl=0.0)
        assert r.is_expired() is False

    def test_not_expired_within_ttl(self):
        r = CacheRecord(key="k", value=1, created_at=time.time(), ttl=3600.0)
        assert r.is_expired() is False

    def test_expired_past_ttl(self):
        r = CacheRecord(key="k", value=1, created_at=1000.0, ttl=1.0)
        # now >> created_at + ttl
        assert r.is_expired(now=2000.0) is True

    def test_expired_custom_now(self):
        r = self._make(ttl=10.0)
        # created_at=1000, ttl=10 → expires at 1010
        assert r.is_expired(now=1011.0) is True
        assert r.is_expired(now=1009.0) is False


# ─── TestCacheSummary ─────────────────────────────────────────────────────────

class TestCacheSummary:
    def _make(self, n=5, expired=1, hits=10):
        return CacheSummary(namespace="ns", n_entries=n,
                            n_expired=expired, total_hits=hits)

    def test_hit_ratio_empty(self):
        s = CacheSummary(namespace="ns", n_entries=0, n_expired=0, total_hits=0)
        assert s.hit_ratio == 0.0

    def test_hit_ratio_nonzero(self):
        s = self._make(n=5, hits=10)
        assert abs(s.hit_ratio - 2.0) < 1e-9

    def test_negative_n_entries_raises(self):
        with pytest.raises(ValueError):
            CacheSummary(namespace="ns", n_entries=-1, n_expired=0, total_hits=0)

    def test_negative_n_expired_raises(self):
        with pytest.raises(ValueError):
            CacheSummary(namespace="ns", n_entries=0, n_expired=-1, total_hits=0)

    def test_negative_total_hits_raises(self):
        with pytest.raises(ValueError):
            CacheSummary(namespace="ns", n_entries=0, n_expired=0, total_hits=-1)


# ─── TestResultCache ──────────────────────────────────────────────────────────

class TestResultCache:
    def test_put_and_get(self):
        c = ResultCache()
        c.put("a", 42)
        assert c.get("a") == 42

    def test_miss_returns_none(self):
        c = ResultCache()
        assert c.get("missing") is None

    def test_empty_key_raises(self):
        c = ResultCache()
        with pytest.raises(ValueError):
            c.put("", 1)

    def test_has_existing_key(self):
        c = ResultCache()
        c.put("x", 99)
        assert c.has("x") is True

    def test_has_missing_key(self):
        c = ResultCache()
        assert c.has("missing") is False

    def test_invalidate_existing(self):
        c = ResultCache()
        c.put("a", 1)
        assert c.invalidate("a") is True
        assert c.get("a") is None

    def test_invalidate_missing_returns_false(self):
        c = ResultCache()
        assert c.invalidate("no") is False

    def test_clear_returns_count(self):
        c = ResultCache()
        c.put("a", 1)
        c.put("b", 2)
        n = c.clear()
        assert n == 2
        assert c.size() == 0

    def test_size_counts_entries(self):
        c = ResultCache()
        c.put("a", 1)
        c.put("b", 2)
        assert c.size() == 2

    def test_keys_returns_list(self):
        c = ResultCache()
        c.put("x", 1)
        c.put("y", 2)
        assert set(c.keys()) == {"x", "y"}

    def test_overwrite_same_key(self):
        c = ResultCache()
        c.put("a", 1)
        c.put("a", 99)
        assert c.get("a") == 99

    def test_hits_incremented_on_get(self):
        c = ResultCache()
        c.put("a", 1)
        c.get("a")
        c.get("a")
        summary = c.summarize()
        assert summary.total_hits == 2

    def test_max_size_eviction(self):
        c = ResultCache(CachePolicy(max_size=2))
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)  # Should evict oldest
        assert c.size() == 2
        assert c.get("c") == 3

    def test_ttl_expiry(self):
        c = ResultCache(CachePolicy(ttl=0.001))
        c.put("a", 1)
        time.sleep(0.01)
        assert c.get("a") is None

    def test_ttl_not_expired(self):
        c = ResultCache(CachePolicy(ttl=3600.0))
        c.put("a", 1)
        assert c.get("a") == 1

    def test_summarize_namespace(self):
        c = ResultCache(CachePolicy(namespace="scores"))
        summary = c.summarize()
        assert summary.namespace == "scores"

    def test_summarize_counts(self):
        c = ResultCache()
        c.put("a", 1)
        c.put("b", 2)
        summary = c.summarize()
        assert summary.n_entries == 2


# ─── TestMakeCache ────────────────────────────────────────────────────────────

class TestMakeCache:
    def test_returns_result_cache(self):
        c = make_cache()
        assert isinstance(c, ResultCache)

    def test_custom_ttl(self):
        c = make_cache(ttl=60.0)
        c.put("a", 1)
        assert c.get("a") == 1

    def test_custom_namespace(self):
        c = make_cache(namespace="test")
        summary = c.summarize()
        assert summary.namespace == "test"

    def test_max_size_respected(self):
        c = make_cache(max_size=1)
        c.put("a", 1)
        c.put("b", 2)
        assert c.size() == 1


# ─── TestCachedResult ─────────────────────────────────────────────────────────

class TestCachedResult:
    def test_computes_on_miss(self):
        c = ResultCache()
        val = cached_result(c, "k", lambda: 42)
        assert val == 42

    def test_returns_cached_on_hit(self):
        c = ResultCache()
        c.put("k", 99)
        val = cached_result(c, "k", lambda: 0)
        assert val == 99

    def test_compute_fn_called_once(self):
        c = ResultCache()
        call_count = [0]
        def fn():
            call_count[0] += 1
            return 1
        cached_result(c, "k", fn)
        cached_result(c, "k", fn)
        assert call_count[0] == 1

    def test_stores_result(self):
        c = ResultCache()
        cached_result(c, "k", lambda: 7)
        assert c.has("k")


# ─── TestMergeCaches ──────────────────────────────────────────────────────────

class TestMergeCaches:
    def test_copies_entries(self):
        src = ResultCache()
        src.put("a", 1)
        src.put("b", 2)
        dst = ResultCache()
        n = merge_caches(dst, src)
        assert n == 2
        assert dst.get("a") == 1
        assert dst.get("b") == 2

    def test_empty_source_copies_nothing(self):
        src = ResultCache()
        dst = ResultCache()
        assert merge_caches(dst, src) == 0

    def test_overwrites_existing_in_target(self):
        src = ResultCache()
        src.put("a", 99)
        dst = ResultCache()
        dst.put("a", 1)
        merge_caches(dst, src)
        assert dst.get("a") == 99


# ─── TestEvictExpired ─────────────────────────────────────────────────────────

class TestEvictExpired:
    def test_removes_expired(self):
        c = ResultCache(CachePolicy(ttl=0.001))
        c.put("a", 1)
        time.sleep(0.01)
        n = evict_expired(c)
        assert n >= 1
        assert c.has("a") is False

    def test_no_expiry_removes_nothing(self):
        c = ResultCache()
        c.put("a", 1)
        n = evict_expired(c)
        assert n == 0
        assert c.has("a") is True
