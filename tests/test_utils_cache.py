"""Tests for puzzle_reconstruction.utils.cache (DescriptorCache & helpers)."""
import numpy as np
import pytest
import tempfile
import os

from puzzle_reconstruction.utils.cache import (
    DescriptorCache,
    DiskCache,
    descriptor_key,
    cached,
    get_default_cache,
    clear_default_cache,
)
from puzzle_reconstruction.models import Fragment


np.random.seed(0)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_fragment(fid=0):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    contour = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    return Fragment(fragment_id=fid, image=img, contour=contour)


# ── DescriptorCache ────────────────────────────────────────────────────────────

def test_cache_set_get():
    c = DescriptorCache(max_size=10)
    c.set("k1", 42)
    assert c.get("k1") == 42


def test_cache_miss_returns_none():
    c = DescriptorCache(max_size=10)
    assert c.get("missing") is None


def test_cache_len():
    c = DescriptorCache(max_size=10)
    c.set("a", 1)
    c.set("b", 2)
    assert len(c) == 2


def test_cache_contains():
    c = DescriptorCache(max_size=5)
    c.set("x", "hello")
    assert "x" in c
    assert "y" not in c


def test_cache_eviction_lru():
    c = DescriptorCache(max_size=3)
    for i in range(4):
        c.set(str(i), i)
    # The first key "0" should have been evicted
    assert c.get("0") is None
    assert c.get("3") == 3


def test_cache_clear():
    c = DescriptorCache(max_size=10)
    c.set("k", 99)
    c.clear()
    assert len(c) == 0
    assert c.get("k") is None


def test_cache_hit_rate():
    c = DescriptorCache(max_size=10)
    c.set("k", 1)
    c.get("k")   # hit
    c.get("x")   # miss
    assert abs(c.hit_rate - 0.5) < 1e-9


def test_cache_stats_keys():
    c = DescriptorCache(max_size=5)
    s = c.stats()
    assert "size" in s and "hits" in s and "misses" in s and "hit_rate" in s


def test_cache_get_or_compute_caches():
    c = DescriptorCache(max_size=10)
    calls = []
    def fn():
        calls.append(1)
        return "result"
    c.get_or_compute("k", fn)
    c.get_or_compute("k", fn)
    assert len(calls) == 1  # computed only once


def test_cache_repr():
    c = DescriptorCache(max_size=10)
    r = repr(c)
    assert "DescriptorCache" in r


def test_cache_max_size_one():
    c = DescriptorCache(max_size=1)
    c.set("a", 1)
    c.set("b", 2)
    assert len(c) == 1


# ── descriptor_key ─────────────────────────────────────────────────────────────

def test_descriptor_key_length():
    f = _make_fragment(0)
    key = descriptor_key(f)
    assert len(key) == 16


def test_descriptor_key_different_fragments():
    f0 = _make_fragment(0)
    img1 = np.ones((10, 10, 3), dtype=np.uint8) * 128
    f1 = Fragment(fragment_id=1, image=img1, contour=np.array([[0,0],[5,0],[5,5]], dtype=np.float64))
    k0 = descriptor_key(f0)
    k1 = descriptor_key(f1)
    assert k0 != k1


def test_descriptor_key_no_image():
    f = _make_fragment(0)
    key_with = descriptor_key(f, include_image=True)
    key_without = descriptor_key(f, include_image=False)
    # Both are 16 chars
    assert len(key_with) == 16
    assert len(key_without) == 16


def test_descriptor_key_is_string():
    f = _make_fragment(0)
    assert isinstance(descriptor_key(f), str)


# ── @cached decorator ──────────────────────────────────────────────────────────

def test_cached_decorator_caches_result():
    c = DescriptorCache(max_size=10)
    call_count = [0]

    @cached(c)
    def compute(fragment: Fragment):
        call_count[0] += 1
        return "desc"

    f = _make_fragment(0)
    compute(f)
    compute(f)
    assert call_count[0] == 1


def test_cached_decorator_exposes_cache():
    c = DescriptorCache(max_size=10)

    @cached(c)
    def fn(fragment: Fragment):
        return 1

    assert fn.cache is c


# ── get_default_cache / clear_default_cache ────────────────────────────────────

def test_get_default_cache_returns_instance():
    c = get_default_cache()
    assert isinstance(c, DescriptorCache)


def test_get_default_cache_singleton():
    c1 = get_default_cache()
    c2 = get_default_cache()
    assert c1 is c2


def test_clear_default_cache():
    c = get_default_cache()
    c.set("x", 1)
    clear_default_cache()
    assert c.get("x") is None


# ── DiskCache ──────────────────────────────────────────────────────────────────

def test_disk_cache_set_get():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_cache")
        with DiskCache(path) as dc:
            dc["k1"] = "hello"
            assert dc["k1"] == "hello"


def test_disk_cache_contains():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_cache2")
        with DiskCache(path) as dc:
            dc["key"] = 42
            assert "key" in dc
            assert "other" not in dc


def test_disk_cache_len():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_cache3")
        with DiskCache(path) as dc:
            dc["a"] = 1
            dc["b"] = 2
            assert len(dc) == 2


def test_disk_cache_get_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_cache4")
        with DiskCache(path) as dc:
            assert dc.get("missing", "default") == "default"
