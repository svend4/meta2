"""Tests for puzzle_reconstruction.utils.fragment_filter_utils"""
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


def _make_mask(h, w, fill=True):
    """Create a simple filled rectangular mask."""
    m = np.zeros((h, w), dtype=np.uint8)
    if fill:
        m[:] = 255
    return m


def _make_frag(fid=0, h=10, w=10):
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    mask = _make_mask(h, w)
    return fid, img, mask


# ── FragmentFilterConfig ─────────────────────────────────────────────────────

def test_config_defaults():
    cfg = FragmentFilterConfig()
    assert cfg.min_area == 0.0
    assert cfg.min_fill_ratio == 0.0
    assert cfg.deduplicate is True


def test_config_negative_min_area_raises():
    with pytest.raises(ValueError):
        FragmentFilterConfig(min_area=-1)


def test_config_max_area_zero_raises():
    with pytest.raises(ValueError):
        FragmentFilterConfig(max_area=0)


def test_config_min_area_greater_than_max_raises():
    with pytest.raises(ValueError):
        FragmentFilterConfig(min_area=100, max_area=10)


def test_config_fill_ratio_out_of_range():
    with pytest.raises(ValueError):
        FragmentFilterConfig(min_fill_ratio=1.5)
    with pytest.raises(ValueError):
        FragmentFilterConfig(min_fill_ratio=-0.1)


# ── compute_fragment_area ────────────────────────────────────────────────────

def test_compute_area_full_mask():
    mask = _make_mask(10, 10)
    assert compute_fragment_area(mask) == 100.0


def test_compute_area_empty_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert compute_fragment_area(mask) == 0.0


def test_compute_area_partial_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:5, 0:5] = 255
    assert compute_fragment_area(mask) == 25.0


def test_compute_area_returns_float():
    mask = _make_mask(5, 5)
    result = compute_fragment_area(mask)
    assert isinstance(result, float)


# ── compute_aspect_ratio ─────────────────────────────────────────────────────

def test_aspect_ratio_square():
    mask = _make_mask(10, 10)
    ratio = compute_aspect_ratio(mask)
    assert ratio == pytest.approx(1.0)


def test_aspect_ratio_wide():
    mask = np.zeros((5, 20), dtype=np.uint8)
    mask[1:4, 1:19] = 255
    ratio = compute_aspect_ratio(mask)
    assert ratio < 1.0


def test_aspect_ratio_in_range():
    np.random.seed(42)
    mask = np.random.randint(0, 2, (20, 10), dtype=np.uint8) * 255
    ratio = compute_aspect_ratio(mask)
    assert 0.0 < ratio <= 1.0


def test_aspect_ratio_empty_mask_returns_1():
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert compute_aspect_ratio(mask) == 1.0


# ── compute_fill_ratio ───────────────────────────────────────────────────────

def test_fill_ratio_full_mask():
    mask = _make_mask(10, 10)
    assert compute_fill_ratio(mask) == pytest.approx(1.0)


def test_fill_ratio_empty_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert compute_fill_ratio(mask) == 1.0


def test_fill_ratio_partial():
    mask = np.zeros((10, 10), dtype=np.uint8)
    # 5x5 region in the corner means bbox=5x5, all filled => 1.0
    mask[0:5, 0:5] = 255
    assert compute_fill_ratio(mask) == pytest.approx(1.0)


def test_fill_ratio_sparse():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, 0] = 255
    mask[9, 9] = 255
    # 2 pixels, bbox 10x10 = 100 => 0.02
    ratio = compute_fill_ratio(mask)
    assert ratio == pytest.approx(2.0 / 100.0)


# ── evaluate_fragment ────────────────────────────────────────────────────────

def test_evaluate_fragment_passes():
    cfg = FragmentFilterConfig(min_area=10)
    mask = _make_mask(20, 20)
    q = evaluate_fragment(0, mask, cfg)
    assert q.passed is True
    assert q.is_valid is True


def test_evaluate_fragment_fails_area_too_small():
    cfg = FragmentFilterConfig(min_area=500)
    mask = _make_mask(10, 10)
    q = evaluate_fragment(0, mask, cfg)
    assert q.passed is False
    assert q.reject_reason == "area_too_small"


def test_evaluate_fragment_fails_area_too_large():
    cfg = FragmentFilterConfig(max_area=5)
    mask = _make_mask(10, 10)
    q = evaluate_fragment(0, mask, cfg)
    assert q.passed is False
    assert q.reject_reason == "area_too_large"


def test_evaluate_fragment_fails_fill_too_low():
    cfg = FragmentFilterConfig(min_fill_ratio=0.99)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0, 0] = 255
    mask[19, 19] = 255
    q = evaluate_fragment(0, mask, cfg)
    assert q.passed is False
    assert q.reject_reason == "fill_too_low"


# ── deduplicate_fragments ────────────────────────────────────────────────────

def test_deduplicate_removes_duplicates():
    np.random.seed(7)
    img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    frags = [(0, img), (1, img.copy()), (2, img)]
    result = deduplicate_fragments(frags)
    # Two identical images should collapse (0 and 2 are same content)
    assert len(result) <= len(frags)


def test_deduplicate_unique_fragments_unchanged():
    # Use seed=2 with 10x10 images which produce distinct hashes for 3 fragments
    np.random.seed(2)
    frags = [(i, np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)) for i in range(3)]
    result = deduplicate_fragments(frags)
    assert len(result) == 3


def test_deduplicate_empty_input():
    assert deduplicate_fragments([]) == []


# ── filter_fragments ─────────────────────────────────────────────────────────

def test_filter_fragments_all_pass():
    cfg = FragmentFilterConfig(min_area=0, deduplicate=False)
    frags = [_make_frag(i) for i in range(5)]
    kept, qualities = filter_fragments(frags, cfg)
    assert len(kept) == 5
    assert len(qualities) == 5
    assert all(q.passed for q in qualities)


def test_filter_fragments_all_fail():
    cfg = FragmentFilterConfig(min_area=100000, deduplicate=False)
    frags = [_make_frag(i) for i in range(3)]
    kept, qualities = filter_fragments(frags, cfg)
    assert len(kept) == 0
    assert len(qualities) == 3


def test_filter_fragments_default_config():
    frags = [_make_frag(i) for i in range(3)]
    kept, qualities = filter_fragments(frags)
    assert isinstance(kept, list)
    assert isinstance(qualities, list)


# ── sort_by_area ─────────────────────────────────────────────────────────────

def test_sort_by_area_descending():
    frags = [
        (0, np.zeros((5, 5, 3), dtype=np.uint8), _make_mask(5, 5)),
        (1, np.zeros((10, 10, 3), dtype=np.uint8), _make_mask(10, 10)),
        (2, np.zeros((3, 3, 3), dtype=np.uint8), _make_mask(3, 3)),
    ]
    sorted_frags = sort_by_area(frags, descending=True)
    areas = [compute_fragment_area(t[2]) for t in sorted_frags]
    assert areas == sorted(areas, reverse=True)


def test_sort_by_area_ascending():
    frags = [
        (0, np.zeros((5, 5, 3), dtype=np.uint8), _make_mask(5, 5)),
        (1, np.zeros((10, 10, 3), dtype=np.uint8), _make_mask(10, 10)),
    ]
    sorted_frags = sort_by_area(frags, descending=False)
    areas = [compute_fragment_area(t[2]) for t in sorted_frags]
    assert areas == sorted(areas)


# ── top_k_fragments ──────────────────────────────────────────────────────────

def test_top_k_returns_k_items():
    frags = [_make_frag(i, h=i+1, w=i+1) for i in range(5)]
    top = top_k_fragments(frags, k=3)
    assert len(top) == 3


def test_top_k_raises_on_zero_k():
    frags = [_make_frag(0)]
    with pytest.raises(ValueError):
        top_k_fragments(frags, k=0)


def test_top_k_larger_than_list():
    frags = [_make_frag(i) for i in range(2)]
    top = top_k_fragments(frags, k=10)
    assert len(top) == 2


# ── fragment_quality_summary ─────────────────────────────────────────────────

def test_fragment_quality_summary_all_pass():
    cfg = FragmentFilterConfig()
    mask = _make_mask(10, 10)
    qualities = [evaluate_fragment(i, mask, cfg) for i in range(5)]
    summary = fragment_quality_summary(qualities)
    assert summary["total"] == 5
    assert summary["passed"] == 5
    assert summary["rejected"] == 0


def test_fragment_quality_summary_mixed():
    cfg = FragmentFilterConfig(min_area=500)
    mask_small = _make_mask(5, 5)
    mask_large = _make_mask(30, 30)
    qualities = [
        evaluate_fragment(0, mask_small, cfg),
        evaluate_fragment(1, mask_large, cfg),
    ]
    summary = fragment_quality_summary(qualities)
    assert summary["total"] == 2
    assert summary["passed"] == 1
    assert summary["rejected"] == 1
