"""Extra tests for puzzle_reconstruction/utils/fragment_filter_utils.py."""
from __future__ import annotations

import math
import pytest
import numpy as np

from puzzle_reconstruction.utils.fragment_filter_utils import (
    FragmentFilterConfig,
    FragmentQuality,
    compute_fragment_area,
    compute_aspect_ratio,
    compute_fill_ratio,
    evaluate_fragment,
    deduplicate_fragments,
    filter_fragments,
    sort_by_area,
    top_k_fragments,
    fragment_quality_summary,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _mask(h=10, w=10, fill=True) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    if fill:
        m[:] = 255
    return m


def _rect_mask(h=10, w=5) -> np.ndarray:
    m = np.zeros((20, 20), dtype=np.uint8)
    m[:h, :w] = 255
    return m


def _frag(fid=0, h=10, w=10) -> tuple:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    mask = _mask(h, w)
    return (fid, img, mask)


# ─── FragmentFilterConfig ─────────────────────────────────────────────────────

class TestFragmentFilterConfigExtra:
    def test_default_min_area(self):
        assert FragmentFilterConfig().min_area == pytest.approx(0.0)

    def test_default_max_area(self):
        assert math.isinf(FragmentFilterConfig().max_area)

    def test_negative_min_area_raises(self):
        with pytest.raises(ValueError):
            FragmentFilterConfig(min_area=-1.0)

    def test_max_area_zero_raises(self):
        with pytest.raises(ValueError):
            FragmentFilterConfig(max_area=0.0)

    def test_min_area_gt_max_area_raises(self):
        with pytest.raises(ValueError):
            FragmentFilterConfig(min_area=100.0, max_area=50.0)

    def test_fill_ratio_out_of_range_raises(self):
        with pytest.raises(ValueError):
            FragmentFilterConfig(min_fill_ratio=1.5)

    def test_custom_values(self):
        cfg = FragmentFilterConfig(min_area=10.0, max_area=1000.0, min_fill_ratio=0.5)
        assert cfg.min_area == pytest.approx(10.0)
        assert cfg.min_fill_ratio == pytest.approx(0.5)


# ─── FragmentQuality ──────────────────────────────────────────────────────────

class TestFragmentQualityExtra:
    def test_is_valid_passed(self):
        q = FragmentQuality(fragment_id=0, area=100.0, aspect_ratio=0.8,
                            fill_ratio=0.9, passed=True)
        assert q.is_valid is True

    def test_is_valid_failed(self):
        q = FragmentQuality(fragment_id=0, area=1.0, aspect_ratio=0.1,
                            fill_ratio=0.1, passed=False, reject_reason="area_too_small")
        assert q.is_valid is False

    def test_stores_reject_reason(self):
        q = FragmentQuality(fragment_id=0, area=0.0, aspect_ratio=1.0,
                            fill_ratio=1.0, passed=False, reject_reason="area_too_small")
        assert q.reject_reason == "area_too_small"


# ─── compute_fragment_area ────────────────────────────────────────────────────

class TestComputeFragmentAreaExtra:
    def test_full_mask(self):
        m = _mask(4, 5)
        assert compute_fragment_area(m) == pytest.approx(20.0)

    def test_empty_mask(self):
        m = np.zeros((8, 8), dtype=np.uint8)
        assert compute_fragment_area(m) == pytest.approx(0.0)

    def test_partial_mask(self):
        m = np.zeros((10, 10), dtype=np.uint8)
        m[:3, :3] = 255
        assert compute_fragment_area(m) == pytest.approx(9.0)


# ─── compute_aspect_ratio ─────────────────────────────────────────────────────

class TestComputeAspectRatioExtra:
    def test_square_is_one(self):
        assert compute_aspect_ratio(_mask(10, 10)) == pytest.approx(1.0)

    def test_rect_leq_one(self):
        r = compute_aspect_ratio(_rect_mask(10, 5))
        assert 0.0 < r <= 1.0

    def test_empty_mask_is_one(self):
        m = np.zeros((8, 8), dtype=np.uint8)
        assert compute_aspect_ratio(m) == pytest.approx(1.0)


# ─── compute_fill_ratio ───────────────────────────────────────────────────────

class TestComputeFillRatioExtra:
    def test_full_mask_is_one(self):
        assert compute_fill_ratio(_mask(5, 5)) == pytest.approx(1.0)

    def test_empty_mask_is_one(self):
        m = np.zeros((5, 5), dtype=np.uint8)
        assert compute_fill_ratio(m) == pytest.approx(1.0)

    def test_partial_fill(self):
        m = np.zeros((10, 10), dtype=np.uint8)
        m[0, 0] = 255
        m[9, 9] = 255
        ratio = compute_fill_ratio(m)
        assert 0.0 < ratio <= 1.0


# ─── evaluate_fragment ────────────────────────────────────────────────────────

class TestEvaluateFragmentExtra:
    def test_returns_fragment_quality(self):
        q = evaluate_fragment(0, _mask(), FragmentFilterConfig())
        assert isinstance(q, FragmentQuality)

    def test_passes_default_config(self):
        q = evaluate_fragment(0, _mask(10, 10), FragmentFilterConfig())
        assert q.passed is True

    def test_fails_area_too_small(self):
        cfg = FragmentFilterConfig(min_area=9999.0)
        q = evaluate_fragment(0, _mask(5, 5), cfg)
        assert q.passed is False
        assert "area" in q.reject_reason

    def test_fails_area_too_large(self):
        cfg = FragmentFilterConfig(max_area=1.0)
        q = evaluate_fragment(0, _mask(10, 10), cfg)
        assert q.passed is False

    def test_fragment_id_stored(self):
        q = evaluate_fragment(42, _mask(), FragmentFilterConfig())
        assert q.fragment_id == 42


# ─── deduplicate_fragments ────────────────────────────────────────────────────

class TestDeduplicateFragmentsExtra:
    def test_returns_list(self):
        img = np.zeros((4, 4), dtype=np.uint8)
        result = deduplicate_fragments([(0, img)])
        assert isinstance(result, list)

    def test_unique_kept(self):
        a = np.zeros((4, 4), dtype=np.uint8)
        b = np.ones((4, 4), dtype=np.uint8) * 128
        result = deduplicate_fragments([(0, a), (1, b)])
        assert len(result) == 2

    def test_empty_input(self):
        assert deduplicate_fragments([]) == []


# ─── filter_fragments ─────────────────────────────────────────────────────────

class TestFilterFragmentsExtra:
    def test_returns_tuple(self):
        frags = [_frag(0)]
        kept, quals = filter_fragments(frags)
        assert isinstance(kept, list) and isinstance(quals, list)

    def test_all_pass_default(self):
        frags = [_frag(i) for i in range(3)]
        kept, quals = filter_fragments(frags)
        assert all(q.passed for q in quals)

    def test_fails_min_area(self):
        cfg = FragmentFilterConfig(min_area=9999.0)
        frags = [_frag(0, h=5, w=5)]
        kept, quals = filter_fragments(frags, cfg)
        assert len(kept) == 0 and not quals[0].passed

    def test_empty_input(self):
        kept, quals = filter_fragments([])
        assert kept == [] and quals == []


# ─── sort_by_area ─────────────────────────────────────────────────────────────

class TestSortByAreaExtra:
    def test_descending_default(self):
        frags = [_frag(0, 5, 5), _frag(1, 10, 10)]
        result = sort_by_area(frags)
        areas = [compute_fragment_area(t[2]) for t in result]
        assert areas == sorted(areas, reverse=True)

    def test_ascending(self):
        frags = [_frag(0, 10, 10), _frag(1, 3, 3)]
        result = sort_by_area(frags, descending=False)
        areas = [compute_fragment_area(t[2]) for t in result]
        assert areas == sorted(areas)

    def test_empty(self):
        assert sort_by_area([]) == []


# ─── top_k_fragments ──────────────────────────────────────────────────────────

class TestTopKFragmentsExtra:
    def test_returns_k_items(self):
        frags = [_frag(i) for i in range(5)]
        result = top_k_fragments(frags, 3)
        assert len(result) == 3

    def test_zero_k_raises(self):
        with pytest.raises(ValueError):
            top_k_fragments([_frag(0)], k=0)

    def test_empty_input(self):
        assert top_k_fragments([], k=1) == []


# ─── fragment_quality_summary ─────────────────────────────────────────────────

class TestFragmentQualitySummaryExtra:
    def test_returns_dict(self):
        qs = [FragmentQuality(0, 10.0, 0.5, 0.9, True)]
        assert isinstance(fragment_quality_summary(qs), dict)

    def test_counts_correct(self):
        qs = [
            FragmentQuality(0, 10.0, 0.5, 0.9, True),
            FragmentQuality(1, 1.0, 0.1, 0.1, False, "area_too_small"),
        ]
        s = fragment_quality_summary(qs)
        assert s["total"] == 2 and s["passed"] == 1 and s["rejected"] == 1

    def test_reject_reason_counted(self):
        qs = [FragmentQuality(0, 0.0, 1.0, 1.0, False, "area_too_small")]
        s = fragment_quality_summary(qs)
        assert s.get("area_too_small") == 1

    def test_empty_input(self):
        s = fragment_quality_summary([])
        assert s["total"] == 0
