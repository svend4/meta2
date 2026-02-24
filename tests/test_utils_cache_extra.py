"""Extra tests for puzzle_reconstruction/utils/cache.py."""
from __future__ import annotations

import tempfile
import os
import pytest

from puzzle_reconstruction.utils.cache import (
    DescriptorCache,
    DiskCache,
    get_default_cache,
    clear_default_cache,
)


# ─── DescriptorCache ──────────────────────────────────────────────────────────

class TestDescriptorCacheExtra:
    def test_initial_len_zero(self):
        c = DescriptorCache()
        assert len(c) == 0

    def test_set_and_get(self):
        c = DescriptorCache()
        c.set("k", 42)
        assert c.get("k") == 42

    def test_get_missing_none(self):
        c = DescriptorCache()
        assert c.get("missing") is None

    def test_contains_after_set(self):
        c = DescriptorCache()
        c.set("a", 1)
        assert "a" in c

    def test_not_contains_before_set(self):
        c = DescriptorCache()
        assert "z" not in c

    def test_len_increases(self):
        c = DescriptorCache()
        c.set("a", 1)
        c.set("b", 2)
        assert len(c) == 2

    def test_clear_empties(self):
        c = DescriptorCache()
        c.set("a", 1)
        c.clear()
        assert len(c) == 0

    def test_repr_is_str(self):
        c = DescriptorCache()
        assert isinstance(repr(c), str)

    def test_get_or_compute_cache_miss(self):
        c = DescriptorCache()
        val = c.get_or_compute("k", lambda: 99)
        assert val == 99

    def test_get_or_compute_cached(self):
        c = DescriptorCache()
        c.set("k", 10)
        calls = [0]

        def compute():
            calls[0] += 1
            return 99

        val = c.get_or_compute("k", compute)
        assert val == 10
        assert calls[0] == 0

    def test_lru_eviction(self):
        c = DescriptorCache(max_size=2)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)
        assert len(c) <= 2

    def test_hit_rate_zero_initially(self):
        c = DescriptorCache()
        assert c.hit_rate == pytest.approx(0.0)

    def test_hit_rate_increases_after_hit(self):
        c = DescriptorCache()
        c.set("k", 1)
        c.get("k")  # hit
        assert c.hit_rate > 0.0

    def test_stats_keys(self):
        c = DescriptorCache()
        for k in ("size", "max", "hits", "misses", "hit_rate"):
            assert k in c.stats()

    def test_stats_size_correct(self):
        c = DescriptorCache()
        c.set("a", 1)
        assert c.stats()["size"] == 1

    def test_overwrite_key(self):
        c = DescriptorCache()
        c.set("k", 1)
        c.set("k", 99)
        assert c.get("k") == 99

    def test_none_value_stored(self):
        c = DescriptorCache()
        c.set("k", None)
        assert "k" in c

    def test_default_max_size(self):
        c = DescriptorCache()
        assert c.stats()["max"] == 512


# ─── DiskCache ────────────────────────────────────────────────────────────────

class TestDiskCacheExtra:
    def _tmppath(self, tmp_dir) -> str:
        return os.path.join(tmp_dir, "test_cache")

    def test_set_and_get(self, tmp_path):
        c = DiskCache(str(tmp_path / "c"))
        c["k"] = 42
        assert c["k"] == 42
        c.close()

    def test_contains_after_set(self, tmp_path):
        c = DiskCache(str(tmp_path / "c"))
        c["a"] = 1
        assert "a" in c
        c.close()

    def test_not_contains_missing(self, tmp_path):
        c = DiskCache(str(tmp_path / "c"))
        assert "missing" not in c
        c.close()

    def test_get_default(self, tmp_path):
        c = DiskCache(str(tmp_path / "c"))
        assert c.get("x", default=7) == 7
        c.close()

    def test_get_existing(self, tmp_path):
        c = DiskCache(str(tmp_path / "c"))
        c["k"] = 100
        assert c.get("k") == 100
        c.close()

    def test_len_increases(self, tmp_path):
        c = DiskCache(str(tmp_path / "c"))
        c["a"] = 1
        c["b"] = 2
        assert len(c) >= 2
        c.close()

    def test_clear(self, tmp_path):
        c = DiskCache(str(tmp_path / "c"))
        c["a"] = 1
        c.clear()
        assert "a" not in c
        c.close()

    def test_get_or_compute_miss(self, tmp_path):
        c = DiskCache(str(tmp_path / "c"))
        val = c.get_or_compute("k", lambda: 55)
        assert val == 55
        c.close()

    def test_get_or_compute_cached(self, tmp_path):
        c = DiskCache(str(tmp_path / "c"))
        c["k"] = 10
        calls = [0]

        def compute():
            calls[0] += 1
            return 99

        val = c.get_or_compute("k", compute)
        assert val == 10
        assert calls[0] == 0
        c.close()

    def test_context_manager(self, tmp_path):
        with DiskCache(str(tmp_path / "c")) as c:
            c["x"] = 123
        # after __exit__, cache is closed; no exception raised

    def test_repr_is_str(self, tmp_path):
        c = DiskCache(str(tmp_path / "c"))
        assert isinstance(repr(c), str)
        c.close()


# ─── get_default_cache / clear_default_cache ──────────────────────────────────

class TestDefaultCacheExtra:
    def test_returns_descriptor_cache(self):
        c = get_default_cache()
        assert isinstance(c, DescriptorCache)

    def test_singleton(self):
        c1 = get_default_cache()
        c2 = get_default_cache()
        assert c1 is c2

    def test_clear_default_cache(self):
        c = get_default_cache()
        c.set("temp", 1)
        clear_default_cache()
        # After clear, a new call still returns a DescriptorCache
        c2 = get_default_cache()
        assert isinstance(c2, DescriptorCache)

    def test_custom_max_size(self):
        # Different max size creates/returns cache
        c = get_default_cache(max_size=128)
        assert isinstance(c, DescriptorCache)
