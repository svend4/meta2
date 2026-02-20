"""
Тесты для puzzle_reconstruction/utils/cache.py

Покрытие:
    descriptor_key      — разные фрагменты → разные ключи; 16-символьная строка;
                          include_image=True/False
    DescriptorCache     — get/set/contains, LRU вытеснение, max_size=1,
                          get_or_compute (кешируется), get_for_fragment,
                          hit_rate, clear, thread-safety (базовая проверка)
    DiskCache           — get/set/contains, get_or_compute, close/context manager
    cached decorator    — результат кешируется, вычисляется один раз, wraps
    get_default_cache   — синглтон, clear_default_cache
"""
import math
import os
import tempfile
import threading
import numpy as np
import pytest

from puzzle_reconstruction.models import Fragment, EdgeSide
from puzzle_reconstruction.utils.cache import (
    descriptor_key,
    DescriptorCache,
    DiskCache,
    cached,
    get_default_cache,
    clear_default_cache,
)


# ─── Фикстуры ─────────────────────────────────────────────────────────────────

def _make_fragment(fid: int, size: int = 16) -> Fragment:
    rng = np.random.RandomState(fid)
    img = rng.randint(0, 256, (size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    contour = rng.rand(10, 2) * size
    return Fragment(
        fragment_id=fid,
        image=img,
        mask=mask,
        contour=contour,
    )


@pytest.fixture
def frag0():
    return _make_fragment(0)


@pytest.fixture
def frag1():
    return _make_fragment(1)


@pytest.fixture
def cache():
    return DescriptorCache(max_size=8)


# ─── descriptor_key ───────────────────────────────────────────────────────────

class TestDescriptorKey:
    def test_returns_string(self, frag0):
        k = descriptor_key(frag0)
        assert isinstance(k, str)

    def test_length_16(self, frag0):
        k = descriptor_key(frag0)
        assert len(k) == 16

    def test_different_fragments_different_keys(self, frag0, frag1):
        k0 = descriptor_key(frag0)
        k1 = descriptor_key(frag1)
        assert k0 != k1

    def test_same_fragment_same_key(self, frag0):
        k1 = descriptor_key(frag0)
        k2 = descriptor_key(frag0)
        assert k1 == k2

    def test_include_image_false(self, frag0):
        k_with    = descriptor_key(frag0, include_image=True)
        k_without = descriptor_key(frag0, include_image=False)
        # Оба должны быть валидными ключами, могут совпадать или нет
        assert isinstance(k_with, str) and isinstance(k_without, str)
        assert len(k_with) == len(k_without) == 16

    def test_modified_image_changes_key(self):
        """Изменение изображения должно изменить ключ."""
        frag_a = _make_fragment(5)
        frag_b = _make_fragment(5)
        frag_b.image = np.zeros_like(frag_a.image)
        k_a = descriptor_key(frag_a, include_image=True)
        k_b = descriptor_key(frag_b, include_image=True)
        assert k_a != k_b


# ─── DescriptorCache ──────────────────────────────────────────────────────────

class TestDescriptorCacheBasic:
    def test_get_missing_returns_none(self, cache):
        assert cache.get("nonexistent") is None

    def test_set_and_get(self, cache):
        cache.set("k1", [1, 2, 3])
        assert cache.get("k1") == [1, 2, 3]

    def test_contains(self, cache):
        cache.set("k2", "value")
        assert "k2" in cache
        assert "k3" not in cache

    def test_len(self, cache):
        assert len(cache) == 0
        cache.set("a", 1)
        cache.set("b", 2)
        assert len(cache) == 2

    def test_overwrite(self, cache):
        cache.set("k", "first")
        cache.set("k", "second")
        assert cache.get("k") == "second"
        assert len(cache) == 1

    def test_clear(self, cache):
        cache.set("k1", 1)
        cache.set("k2", 2)
        cache.clear()
        assert len(cache) == 0
        assert cache.get("k1") is None


class TestDescriptorCacheLRU:
    def test_eviction_at_max_size(self):
        c = DescriptorCache(max_size=3)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)
        c.set("d", 4)  # Вытесняет "a" (самая старая)
        assert "a" not in c
        assert "b" in c
        assert "d" in c

    def test_access_refreshes_lru(self):
        """Обращение к элементу делает его «новым» → не вытесняется первым."""
        c = DescriptorCache(max_size=3)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)
        _ = c.get("a")    # Обновляем "a"
        c.set("d", 4)     # Вытесняет "b" (самый старый после обновления "a")
        assert "a" in c
        assert "b" not in c

    def test_max_size_one(self):
        c = DescriptorCache(max_size=1)
        c.set("x", 10)
        c.set("y", 20)
        assert "x" not in c
        assert "y" in c


class TestDescriptorCacheGetOrCompute:
    def test_computes_once(self, cache):
        call_count = [0]

        def compute():
            call_count[0] += 1
            return {"data": 42}

        r1 = cache.get_or_compute("k", compute)
        r2 = cache.get_or_compute("k", compute)
        assert r1 == r2 == {"data": 42}
        assert call_count[0] == 1

    def test_different_keys_compute_separately(self, cache):
        call_count = [0]
        def compute():
            call_count[0] += 1
            return call_count[0]

        cache.get_or_compute("k1", compute)
        cache.get_or_compute("k2", compute)
        assert call_count[0] == 2

    def test_get_for_fragment(self, frag0, cache):
        calls = [0]
        def fn(f: Fragment):
            calls[0] += 1
            return f.fragment_id * 10

        v1 = cache.get_for_fragment(frag0, fn)
        v2 = cache.get_for_fragment(frag0, fn)
        assert v1 == v2 == 0
        assert calls[0] == 1


class TestDescriptorCacheStats:
    def test_hit_rate_zero_initially(self, cache):
        assert math.isclose(cache.hit_rate, 0.0)

    def test_hit_rate_after_hits(self, cache):
        cache.set("k", 1)
        cache.get("k")   # hit
        cache.get("k")   # hit
        cache.get("x")   # miss
        assert math.isclose(cache.hit_rate, 2 / 3, rel_tol=1e-6)

    def test_stats_dict(self, cache):
        s = cache.stats()
        assert "size" in s and "hits" in s and "misses" in s and "hit_rate" in s

    def test_repr(self, cache):
        r = repr(cache)
        assert "DescriptorCache" in r


class TestDescriptorCacheThreadSafety:
    def test_concurrent_writes(self):
        c = DescriptorCache(max_size=100)
        errors = []

        def writer(i):
            try:
                for j in range(20):
                    c.set(f"k{i}_{j}", i * 100 + j)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Ошибки при параллельной записи: {errors}"
        assert len(c) <= 100


# ─── DiskCache ────────────────────────────────────────────────────────────────

class TestDiskCache:
    @pytest.fixture
    def disk_cache(self, tmp_path):
        path = str(tmp_path / "test_cache")
        dc   = DiskCache(path)
        yield dc
        dc.close()

    def test_set_and_get(self, disk_cache):
        disk_cache["key1"] = {"value": 42}
        assert disk_cache["key1"] == {"value": 42}

    def test_contains(self, disk_cache):
        disk_cache["k"] = 99
        assert "k" in disk_cache
        assert "z" not in disk_cache

    def test_len(self, disk_cache):
        assert len(disk_cache) == 0
        disk_cache["a"] = 1
        disk_cache["b"] = 2
        assert len(disk_cache) == 2

    def test_get_default(self, disk_cache):
        val = disk_cache.get("missing", default=-1)
        assert val == -1

    def test_get_or_compute(self, disk_cache):
        calls = [0]
        def compute():
            calls[0] += 1
            return "computed"
        v1 = disk_cache.get_or_compute("k", compute)
        v2 = disk_cache.get_or_compute("k", compute)
        assert v1 == v2 == "computed"
        assert calls[0] == 1

    def test_clear(self, disk_cache):
        disk_cache["k1"] = 1
        disk_cache.clear()
        assert len(disk_cache) == 0

    def test_context_manager(self, tmp_path):
        path = str(tmp_path / "ctx_cache")
        with DiskCache(path) as dc:
            dc["x"] = "hello"
            assert "x" in dc
        # После выхода из контекста — кэш закрыт (нет ошибок)

    def test_persistence_across_close(self, tmp_path):
        """Данные сохраняются после close() и повторного открытия."""
        path = str(tmp_path / "persist")
        dc1 = DiskCache(path)
        dc1["persistent"] = [1, 2, 3]
        dc1.close()

        dc2 = DiskCache(path)
        val = dc2.get("persistent")
        dc2.close()
        assert val == [1, 2, 3]

    def test_repr(self, disk_cache):
        assert "DiskCache" in repr(disk_cache)


# ─── @cached декоратор ────────────────────────────────────────────────────────

class TestCachedDecorator:
    def test_result_cached(self, frag0):
        c = DescriptorCache(max_size=16)
        calls = [0]

        @cached(c)
        def compute(fragment: Fragment) -> int:
            calls[0] += 1
            return fragment.fragment_id * 7

        v1 = compute(frag0)
        v2 = compute(frag0)
        assert v1 == v2 == 0
        assert calls[0] == 1

    def test_different_fragments_compute_separately(self, frag0, frag1):
        c = DescriptorCache(max_size=16)
        calls = [0]

        @cached(c)
        def compute(fragment: Fragment) -> int:
            calls[0] += 1
            return fragment.fragment_id

        compute(frag0)
        compute(frag1)
        assert calls[0] == 2

    def test_wraps_preserves_name(self, frag0):
        c = DescriptorCache(16)

        @cached(c)
        def my_function(fragment: Fragment) -> str:
            """My docstring."""
            return "ok"

        assert my_function.__name__ == "my_function"

    def test_cache_attached_to_wrapper(self, frag0):
        c = DescriptorCache(16)

        @cached(c)
        def fn(fragment: Fragment) -> int:
            return 1

        assert hasattr(fn, "cache")
        assert fn.cache is c

    def test_custom_key_fn(self):
        c = DescriptorCache(16)
        calls = [0]

        @cached(c, key_fn=lambda x, y: f"{x}_{y}")
        def add(x, y):
            calls[0] += 1
            return x + y

        r1 = add(2, 3)
        r2 = add(2, 3)
        assert r1 == r2 == 5
        assert calls[0] == 1


# ─── get_default_cache / clear_default_cache ──────────────────────────────────

class TestDefaultCache:
    def test_is_descriptor_cache(self):
        c = get_default_cache()
        assert isinstance(c, DescriptorCache)

    def test_singleton(self):
        c1 = get_default_cache()
        c2 = get_default_cache()
        assert c1 is c2

    def test_clear_default_cache(self):
        c = get_default_cache()
        c.set("test_key_for_clear", 42)
        clear_default_cache()
        assert c.get("test_key_for_clear") is None
