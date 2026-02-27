"""Tests for puzzle_reconstruction.utils.result_cache."""
import time
import pytest
import numpy as np

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

np.random.seed(99)


# ── CachePolicy ───────────────────────────────────────────────────────────────

def test_cache_policy_defaults():
    p = CachePolicy()
    assert p.ttl == 0.0
    assert p.max_size == 0
    assert p.namespace == "default"


def test_cache_policy_negative_ttl_raises():
    with pytest.raises(ValueError):
        CachePolicy(ttl=-1.0)


def test_cache_policy_negative_max_size_raises():
    with pytest.raises(ValueError):
        CachePolicy(max_size=-1)


def test_cache_policy_empty_namespace_raises():
    with pytest.raises(ValueError):
        CachePolicy(namespace="")


# ── CacheRecord ───────────────────────────────────────────────────────────────

def test_cache_record_not_expired_zero_ttl():
    r = CacheRecord(key="k", value=1, created_at=time.time(), ttl=0.0)
    assert r.is_expired() is False


def test_cache_record_expired():
    r = CacheRecord(key="k", value=1, created_at=0.0, ttl=1.0)
    assert r.is_expired() is True


def test_cache_record_empty_key_raises():
    with pytest.raises(ValueError):
        CacheRecord(key="", value=1, created_at=0.0)


def test_cache_record_negative_created_at_raises():
    with pytest.raises(ValueError):
        CacheRecord(key="k", value=1, created_at=-1.0)


# ── CacheSummary ──────────────────────────────────────────────────────────────

def test_cache_summary_hit_ratio_zero():
    cs = CacheSummary(namespace="ns", n_entries=0, n_expired=0, total_hits=0)
    assert cs.hit_ratio == 0.0


def test_cache_summary_hit_ratio():
    cs = CacheSummary(namespace="ns", n_entries=10, n_expired=0, total_hits=5)
    assert cs.hit_ratio == pytest.approx(0.5)


def test_cache_summary_negative_entries_raises():
    with pytest.raises(ValueError):
        CacheSummary(namespace="ns", n_entries=-1, n_expired=0, total_hits=0)


# ── ResultCache basic ─────────────────────────────────────────────────────────

def test_result_cache_put_get():
    cache = make_cache()
    cache.put("key1", 42)
    assert cache.get("key1") == 42


def test_result_cache_get_missing():
    cache = make_cache()
    assert cache.get("nonexistent") is None


def test_result_cache_has_true():
    cache = make_cache()
    cache.put("x", "hello")
    assert cache.has("x") is True


def test_result_cache_has_false():
    cache = make_cache()
    assert cache.has("missing") is False


def test_result_cache_invalidate_true():
    cache = make_cache()
    cache.put("k", 1)
    assert cache.invalidate("k") is True


def test_result_cache_invalidate_false():
    cache = make_cache()
    assert cache.invalidate("missing") is False


def test_result_cache_clear():
    cache = make_cache()
    cache.put("a", 1)
    cache.put("b", 2)
    n = cache.clear()
    assert n == 2
    assert cache.size() == 0


def test_result_cache_keys():
    cache = make_cache()
    cache.put("a", 1)
    cache.put("b", 2)
    keys = cache.keys()
    assert set(keys) == {"a", "b"}


def test_result_cache_size():
    cache = make_cache()
    cache.put("x", 10)
    assert cache.size() == 1


def test_result_cache_empty_key_put_raises():
    cache = make_cache()
    with pytest.raises(ValueError):
        cache.put("", 1)


# ── max_size eviction ─────────────────────────────────────────────────────────

def test_result_cache_max_size_eviction():
    cache = make_cache(max_size=3)
    for i in range(5):
        cache.put(f"k{i}", i)
    assert cache.size() == 3


# ── TTL expiry ────────────────────────────────────────────────────────────────

def test_result_cache_ttl_expiry():
    cache = make_cache(ttl=0.01)
    cache.put("ephemeral", "data")
    time.sleep(0.02)
    assert cache.get("ephemeral") is None


def test_result_cache_has_expired():
    cache = make_cache(ttl=0.01)
    cache.put("short", "lived")
    time.sleep(0.02)
    assert cache.has("short") is False


# ── summarize ─────────────────────────────────────────────────────────────────

def test_result_cache_summarize():
    cache = make_cache(namespace="test_ns")
    cache.put("a", 1)
    cache.get("a")
    summary = cache.summarize()
    assert summary.namespace == "test_ns"
    assert summary.n_entries >= 1
    assert summary.total_hits >= 1


# ── cached_result ─────────────────────────────────────────────────────────────

def test_cached_result_computes_once():
    cache = make_cache()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return 99

    v1 = cached_result(cache, "k", compute)
    v2 = cached_result(cache, "k", compute)
    assert v1 == 99
    assert v2 == 99
    assert calls["n"] == 1


# ── merge_caches ──────────────────────────────────────────────────────────────

def test_merge_caches():
    src = make_cache()
    dst = make_cache()
    src.put("x", 1)
    src.put("y", 2)
    n = merge_caches(dst, src)
    assert n == 2
    assert dst.get("x") == 1
    assert dst.get("y") == 2


# ── evict_expired ─────────────────────────────────────────────────────────────

def test_evict_expired():
    cache = make_cache(ttl=0.01)
    cache.put("a", 1)
    time.sleep(0.02)
    n = evict_expired(cache)
    assert n >= 1
    assert cache.size() == 0
