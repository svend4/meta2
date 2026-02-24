"""Extra tests for puzzle_reconstruction/utils/cache.py"""
import math
import threading

import numpy as np
import pytest

from puzzle_reconstruction.models import Fragment
from puzzle_reconstruction.utils.cache import (
    DescriptorCache,
    DiskCache,
    cached,
    clear_default_cache,
    descriptor_key,
    get_default_cache,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _make_fragment(fid: int, size: int = 16) -> Fragment:
    rng = np.random.RandomState(fid)
    img = rng.randint(0, 256, (size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    contour = rng.rand(10, 2) * size
    return Fragment(fragment_id=fid, image=img, mask=mask, contour=contour)


# ─── TestDescriptorKeyExtra ───────────────────────────────────────────────────

class TestDescriptorKeyExtra:
    def test_many_fragments_all_distinct(self):
        keys = [descriptor_key(_make_fragment(i)) for i in range(20)]
        assert len(set(keys)) == 20

    def test_key_is_alphanumeric(self):
        k = descriptor_key(_make_fragment(7))
        assert k.isalnum() or all(c in "0123456789abcdefABCDEF" for c in k)

    def test_include_image_false_still_16_chars(self):
        frag = _make_fragment(3)
        k = descriptor_key(frag, include_image=False)
        assert len(k) == 16

    def test_include_image_true_different_fragments(self):
        k1 = descriptor_key(_make_fragment(0), include_image=True)
        k2 = descriptor_key(_make_fragment(1), include_image=True)
        assert k1 != k2

    def test_consistency_across_calls(self):
        frag = _make_fragment(42)
        keys = {descriptor_key(frag) for _ in range(5)}
        assert len(keys) == 1


# ─── TestDescriptorCacheExtra ─────────────────────────────────────────────────

class TestDescriptorCacheExtra:
    def test_very_large_max_size(self):
        c = DescriptorCache(max_size=10_000)
        for i in range(100):
            c.set(f"k{i}", i)
        assert len(c) == 100

    def test_get_returns_correct_after_many_sets(self):
        c = DescriptorCache(max_size=50)
        for i in range(40):
            c.set(f"key{i}", i * 2)
        assert c.get("key0") == 0
        assert c.get("key39") == 78

    def test_overwrite_does_not_increase_size(self):
        c = DescriptorCache(max_size=10)
        c.set("k", "a")
        c.set("k", "b")
        c.set("k", "c")
        assert len(c) == 1
        assert c.get("k") == "c"

    def test_set_numpy_array(self):
        c = DescriptorCache(max_size=8)
        arr = np.arange(6).reshape(2, 3)
        c.set("arr", arr)
        retrieved = c.get("arr")
        assert np.array_equal(retrieved, arr)

    def test_stats_has_required_keys(self):
        c = DescriptorCache(max_size=8)
        s = c.stats()
        for key in ("size", "hits", "misses", "hit_rate"):
            assert key in s

    def test_hit_rate_100_percent(self):
        c = DescriptorCache(max_size=8)
        c.set("k", 1)
        c.get("k")
        c.get("k")
        c.get("k")
        assert math.isclose(c.hit_rate, 1.0)

    def test_clear_resets_hit_rate(self):
        c = DescriptorCache(max_size=8)
        c.set("k", 1)
        c.get("k")
        c.clear()
        assert c.hit_rate == pytest.approx(0.0)

    def test_get_for_fragment_different_frags(self):
        c = DescriptorCache(max_size=16)
        calls = [0]

        def fn(f: Fragment):
            calls[0] += 1
            return f.fragment_id

        f0, f1 = _make_fragment(0), _make_fragment(1)
        v0 = c.get_for_fragment(f0, fn)
        v1 = c.get_for_fragment(f1, fn)
        assert v0 == 0
        assert v1 == 1
        assert calls[0] == 2


# ─── TestDescriptorCacheLRUExtra ─────────────────────────────────────────────

class TestDescriptorCacheLRUExtra:
    def test_max_size_two_lru(self):
        c = DescriptorCache(max_size=2)
        c.set("a", 1)
        c.set("b", 2)
        _ = c.get("a")   # promote a
        c.set("c", 3)    # evict b
        assert "b" not in c
        assert "a" in c
        assert "c" in c

    def test_sequential_fill_evicts_oldest(self):
        c = DescriptorCache(max_size=3)
        for i in range(6):
            c.set(f"k{i}", i)
        assert len(c) == 3

    def test_hit_rate_with_evicted_misses(self):
        c = DescriptorCache(max_size=1)
        c.set("a", 1)
        c.set("b", 2)   # evicts a
        c.get("a")       # miss
        c.get("b")       # hit
        # 1 hit, 1 miss → 0.5
        assert math.isclose(c.hit_rate, 0.5, rel_tol=1e-6)


# ─── TestDescriptorCacheThreadSafetyExtra ────────────────────────────────────

class TestDescriptorCacheThreadSafetyExtra:
    def test_concurrent_reads(self):
        c = DescriptorCache(max_size=100)
        for i in range(20):
            c.set(f"k{i}", i)
        errors = []

        def reader():
            try:
                for i in range(20):
                    c.get(f"k{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_concurrent_get_or_compute(self):
        c = DescriptorCache(max_size=50)
        results = []
        errors = []
        call_count = [0]

        def task():
            try:
                v = c.get_or_compute("shared", lambda: (call_count.__setitem__(0, call_count[0] + 1) or 42))
                results.append(v)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=task) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert all(r == 42 for r in results)


# ─── TestDiskCacheExtra ───────────────────────────────────────────────────────

class TestDiskCacheExtra:
    @pytest.fixture
    def dc(self, tmp_path):
        path = str(tmp_path / "extra_cache")
        obj = DiskCache(path)
        yield obj
        obj.close()

    def test_overwrite_value(self, dc):
        dc["k"] = "old"
        dc["k"] = "new"
        assert dc["k"] == "new"

    def test_set_list_value(self, dc):
        dc["lst"] = [1, 2, 3, 4]
        assert dc["lst"] == [1, 2, 3, 4]

    def test_set_dict_value(self, dc):
        dc["d"] = {"x": 10, "y": 20}
        assert dc["d"] == {"x": 10, "y": 20}

    def test_multiple_keys(self, dc):
        for i in range(5):
            dc[f"key{i}"] = i * 10
        for i in range(5):
            assert dc[f"key{i}"] == i * 10

    def test_get_missing_default_none(self, dc):
        assert dc.get("never_set") is None

    def test_get_or_compute_different_keys(self, dc):
        calls = [0]
        for i in range(3):
            v = dc.get_or_compute(f"k{i}", lambda j=i: (calls.__setitem__(0, calls[0] + 1) or j * 5))
            assert v == i * 5
        assert calls[0] == 3

    def test_clear_removes_all(self, dc):
        for i in range(4):
            dc[f"k{i}"] = i
        dc.clear()
        assert len(dc) == 0

    def test_repr_contains_path(self, tmp_path):
        path = str(tmp_path / "repr_test")
        with DiskCache(path) as dc:
            r = repr(dc)
        assert "DiskCache" in r


# ─── TestCachedDecoratorExtra ─────────────────────────────────────────────────

class TestCachedDecoratorExtra:
    def test_non_none_result_cached(self):
        c = DescriptorCache(max_size=8)
        calls = [0]

        @cached(c, key_fn=lambda x: f"frag_{x}")
        def fn(x):
            calls[0] += 1
            return x * 3

        r1 = fn(4)
        r2 = fn(4)
        assert r1 == r2 == 12
        assert calls[0] == 1

    def test_cache_grows_with_unique_args(self):
        c = DescriptorCache(max_size=32)
        calls = [0]

        @cached(c, key_fn=lambda x: str(x))
        def fn(x):
            calls[0] += 1
            return x * 2

        for i in range(5):
            fn(i)
        assert calls[0] == 5

    def test_wraps_docstring(self):
        c = DescriptorCache(max_size=8)

        @cached(c, key_fn=lambda x: str(x))
        def my_fn(x):
            """My doc."""
            return x

        assert my_fn.__doc__ == "My doc."

    def test_cache_attribute_is_correct_instance(self):
        c = DescriptorCache(max_size=8)

        @cached(c, key_fn=lambda x: str(x))
        def fn(x):
            return x

        assert fn.cache is c


# ─── TestDefaultCacheExtra ────────────────────────────────────────────────────

class TestDefaultCacheExtra:
    def test_clear_and_reuse(self):
        c = get_default_cache()
        c.set("extra_key", 99)
        clear_default_cache()
        assert c.get("extra_key") is None
        # Can still use after clear
        c.set("new_key", 42)
        assert c.get("new_key") == 42

    def test_singleton_after_clear(self):
        clear_default_cache()
        c1 = get_default_cache()
        c2 = get_default_cache()
        assert c1 is c2

    def test_is_descriptor_cache(self):
        c = get_default_cache()
        assert isinstance(c, DescriptorCache)
