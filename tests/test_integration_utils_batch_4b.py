"""Integration tests for utils batch 4b.

Covers:
- puzzle_reconstruction.utils.normalize_noise_utils
- puzzle_reconstruction.utils.overlap_score_utils
- puzzle_reconstruction.utils.pair_score_utils
- puzzle_reconstruction.utils.patch_score_utils
- puzzle_reconstruction.utils.patch_utils
"""
import pytest
import numpy as np

rng = np.random.default_rng(42)

# ── imports ───────────────────────────────────────────────────────────────────
from puzzle_reconstruction.utils.normalize_noise_utils import (
    NormResultConfig,
    NormResultEntry,
    NormResultSummary,
    make_norm_result_entry,
    summarise_norm_result_entries,
    filter_norm_by_method,
    filter_norm_by_min_spread,
)
from puzzle_reconstruction.utils.overlap_score_utils import (
    OverlapScoreConfig,
    OverlapScoreEntry,
    OverlapSummary,
    make_overlap_entry,
    summarise_overlaps,
    filter_significant_overlaps,
    filter_by_area,
)
from puzzle_reconstruction.utils.pair_score_utils import (
    PairScoreConfig,
    PairScoreEntry,
    PairScoreSummary,
    make_pair_score_entry,
    entries_from_pair_results,
    summarise_pair_scores,
    filter_strong_pair_matches,
    filter_weak_pair_matches,
    filter_pair_by_score_range,
    filter_pair_by_channel,
    filter_pair_by_dominant_channel,
    top_k_pair_entries,
    best_pair_entry,
    pair_score_stats,
    compare_pair_summaries,
)
from puzzle_reconstruction.utils.patch_score_utils import (
    PatchScoreConfig,
    PatchScoreEntry,
    PatchScoreSummary,
    make_patch_entry,
    entries_from_patch_matches,
    summarise_patch_scores,
    filter_good_patch_scores,
    filter_poor_patch_scores,
    filter_patch_by_score_range,
    filter_by_side_pair,
    filter_by_ncc_range,
    top_k_patch_entries,
    best_patch_entry,
    patch_score_stats,
    compare_patch_summaries,
)
from puzzle_reconstruction.utils.patch_utils import (
    PatchConfig,
    extract_patch,
    extract_patches,
    normalize_patch,
    compare_patches,
    patch_ssd,
    patch_ncc,
    patch_mse,
    batch_compare,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. normalize_noise_utils
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalizeNoiseUtils:
    def _make_entries(self):
        return [
            make_norm_result_entry(0, "minmax", 0.0, 10.0, 4, 4),
            make_norm_result_entry(1, "zscore", 2.0, 8.0,  4, 4),
            make_norm_result_entry(2, "minmax", 1.0, 5.0,  4, 4),
        ]

    def test_make_entry_spread(self):
        e = make_norm_result_entry(0, "minmax", 2.0, 7.0, 8, 8)
        assert e.spread == pytest.approx(5.0)

    def test_make_entry_fields(self):
        e = make_norm_result_entry(3, "zscore", -1.0, 4.0, 3, 5, alpha=0.9)
        assert e.run_id == 3
        assert e.method == "zscore"
        assert e.n_rows == 3 and e.n_cols == 5
        assert e.params == {"alpha": 0.9}

    def test_summarise_basic(self):
        entries = self._make_entries()
        s = summarise_norm_result_entries(entries)
        assert s.n_runs == 3
        assert s.mean_spread == pytest.approx((10.0 + 6.0 + 4.0) / 3)
        assert s.best_run_id == 0  # spread 10
        assert s.worst_run_id == 2  # spread 4

    def test_summarise_empty(self):
        s = summarise_norm_result_entries([])
        assert s.n_runs == 0
        assert s.best_run_id is None
        assert s.worst_run_id is None

    def test_summarise_method_counts(self):
        entries = self._make_entries()
        s = summarise_norm_result_entries(entries)
        assert s.method_counts["minmax"] == 2
        assert s.method_counts["zscore"] == 1

    def test_filter_by_method(self):
        entries = self._make_entries()
        result = filter_norm_by_method(entries, "minmax")
        assert len(result) == 2
        assert all(e.method == "minmax" for e in result)

    def test_filter_by_method_unknown(self):
        entries = self._make_entries()
        assert filter_norm_by_method(entries, "unknown") == []

    def test_filter_by_min_spread(self):
        entries = self._make_entries()
        result = filter_norm_by_min_spread(entries, 6.0)
        assert len(result) == 2
        assert all(e.spread >= 6.0 for e in result)

    def test_filter_by_min_spread_zero(self):
        entries = self._make_entries()
        assert len(filter_norm_by_min_spread(entries, 0.0)) == 3

    def test_norm_result_config_defaults(self):
        cfg = NormResultConfig()
        assert cfg.preferred_method == "minmax"
        assert cfg.min_spread == 0.0

    def test_entry_spread_computed_correctly(self):
        e = make_norm_result_entry(0, "minmax", -5.0, 5.0, 2, 2)
        assert e.spread == pytest.approx(10.0)
        assert e.min_val == pytest.approx(-5.0)
        assert e.max_val == pytest.approx(5.0)

    def test_summarise_single_entry(self):
        e = make_norm_result_entry(7, "zscore", 0.0, 3.0, 2, 2)
        s = summarise_norm_result_entries([e])
        assert s.best_run_id == s.worst_run_id == 7
        assert s.mean_spread == pytest.approx(3.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. overlap_score_utils
# ═══════════════════════════════════════════════════════════════════════════════

class TestOverlapScoreUtils:
    def test_config_defaults(self):
        cfg = OverlapScoreConfig()
        assert cfg.iou_threshold == pytest.approx(0.05)
        assert cfg.area_threshold == pytest.approx(1.0)

    def test_config_invalid_iou(self):
        with pytest.raises(ValueError):
            OverlapScoreConfig(iou_threshold=1.5)

    def test_config_invalid_area(self):
        with pytest.raises(ValueError):
            OverlapScoreConfig(area_threshold=-1.0)

    def test_make_overlap_entry_penalty_significant(self):
        e = make_overlap_entry(0, 1, iou=0.3, overlap_area=10.0)
        assert e.penalty == pytest.approx(0.3)

    def test_make_overlap_entry_penalty_insignificant(self):
        # iou below default threshold
        e = make_overlap_entry(0, 1, iou=0.01, overlap_area=10.0)
        assert e.penalty == pytest.approx(0.0)

    def test_entry_pair_property(self):
        e = OverlapScoreEntry(idx1=2, idx2=5, iou=0.1, overlap_area=4.0)
        assert e.pair == (2, 5)

    def test_entry_negative_idx_raises(self):
        with pytest.raises(ValueError):
            OverlapScoreEntry(idx1=-1, idx2=0, iou=0.1, overlap_area=1.0)

    def test_summarise_overlaps_valid(self):
        entries = [make_overlap_entry(0, 1, iou=0.01, overlap_area=0.5)]
        s = summarise_overlaps(entries)
        assert s.is_valid is True
        assert s.n_overlaps == 0

    def test_summarise_overlaps_invalid(self):
        entries = [
            make_overlap_entry(0, 1, iou=0.2, overlap_area=20.0),
            make_overlap_entry(1, 2, iou=0.1, overlap_area=5.0),
        ]
        s = summarise_overlaps(entries)
        assert s.is_valid is False
        assert s.n_overlaps == 2
        assert s.total_area == pytest.approx(25.0)

    def test_filter_significant_overlaps(self):
        entries = [
            make_overlap_entry(0, 1, iou=0.1, overlap_area=5.0),
            make_overlap_entry(2, 3, iou=0.01, overlap_area=5.0),
        ]
        result = filter_significant_overlaps(entries, iou_threshold=0.05)
        assert len(result) == 1
        assert result[0].iou == pytest.approx(0.1)

    def test_overlap_summary_max_iou(self):
        entries = [
            make_overlap_entry(0, 1, iou=0.2, overlap_area=10.0),
            make_overlap_entry(2, 3, iou=0.5, overlap_area=20.0),
        ]
        s = summarise_overlaps(entries)
        assert s.max_iou == pytest.approx(0.5)

    def test_overlap_summary_mean_penalty(self):
        entries = [
            make_overlap_entry(0, 1, iou=0.4, overlap_area=10.0),
            make_overlap_entry(2, 3, iou=0.6, overlap_area=10.0),
        ]
        s = summarise_overlaps(entries)
        assert s.mean_penalty == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. pair_score_utils
# ═══════════════════════════════════════════════════════════════════════════════

class TestPairScoreUtils:
    def _make_entries(self):
        return [
            make_pair_score_entry(0, 1, 0.9, channels={"r": 0.9, "g": 0.7}),
            make_pair_score_entry(1, 2, 0.5, channels={"r": 0.4, "g": 0.6}),
            make_pair_score_entry(2, 3, 0.2, channels={"r": 0.1, "g": 0.3}),
        ]

    def test_pair_key_ordered(self):
        e = make_pair_score_entry(5, 3, 0.8)
        assert e.pair_key == (3, 5)

    def test_dominant_channel(self):
        e = make_pair_score_entry(0, 1, 0.8, channels={"r": 0.9, "g": 0.4})
        assert e.dominant_channel == "r"

    def test_is_strong_match(self):
        assert make_pair_score_entry(0, 1, 0.7).is_strong_match is True
        assert make_pair_score_entry(0, 1, 0.69).is_strong_match is False

    def test_summarise_pair_scores(self):
        entries = self._make_entries()
        s = summarise_pair_scores(entries)
        assert s.n_entries == 3
        assert s.min_score == pytest.approx(0.2)
        assert s.max_score == pytest.approx(0.9)
        assert s.n_strong_matches == 1

    def test_summarise_empty(self):
        s = summarise_pair_scores([])
        assert s.n_entries == 0

    def test_filter_strong(self):
        entries = self._make_entries()
        strong = filter_strong_pair_matches(entries, threshold=0.7)
        assert len(strong) == 1 and strong[0].score == pytest.approx(0.9)

    def test_filter_weak(self):
        entries = self._make_entries()
        weak = filter_weak_pair_matches(entries, threshold=0.3)
        assert len(weak) == 1 and weak[0].score == pytest.approx(0.2)

    def test_filter_score_range(self):
        entries = self._make_entries()
        mid = filter_pair_by_score_range(entries, lo=0.3, hi=0.7)
        assert len(mid) == 1 and mid[0].score == pytest.approx(0.5)

    def test_top_k(self):
        entries = self._make_entries()
        top = top_k_pair_entries(entries, k=2)
        assert len(top) == 2
        assert top[0].score >= top[1].score

    def test_best_entry(self):
        entries = self._make_entries()
        best = best_pair_entry(entries)
        assert best.score == pytest.approx(0.9)

    def test_pair_score_stats(self):
        entries = self._make_entries()
        stats = pair_score_stats(entries)
        assert stats["count"] == 3
        assert "mean" in stats and "std" in stats

    def test_entries_from_pair_results(self):
        pairs = [(0, 1), (1, 2)]
        scores = [0.8, 0.4]
        entries = entries_from_pair_results(pairs, scores)
        assert len(entries) == 2
        assert entries[0].score == pytest.approx(0.8)

    def test_filter_by_channel(self):
        entries = self._make_entries()
        result = filter_pair_by_channel(entries, "r", min_val=0.5)
        assert len(result) == 1
        assert result[0].channels["r"] >= 0.5

    def test_filter_by_dominant_channel(self):
        entries = self._make_entries()
        result = filter_pair_by_dominant_channel(entries, "r")
        assert all(e.dominant_channel == "r" for e in result)

    def test_compare_pair_summaries(self):
        e1 = [make_pair_score_entry(0, 1, 0.8)]
        e2 = [make_pair_score_entry(0, 1, 0.6)]
        s1 = summarise_pair_scores(e1)
        s2 = summarise_pair_scores(e2)
        delta = compare_pair_summaries(s1, s2)
        assert delta["d_mean_score"] == pytest.approx(0.2)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. patch_score_utils
# ═══════════════════════════════════════════════════════════════════════════════

class TestPatchScoreUtils:
    def _make_entries(self):
        return [
            make_patch_entry(0, 0, 1, 0, 1, ncc=0.8, ssd=0.1, ssim=0.9, total_score=0.8),
            make_patch_entry(1, 1, 2, 1, 0, ncc=0.5, ssd=0.3, ssim=0.6, total_score=0.5),
            make_patch_entry(2, 2, 3, 2, 3, ncc=0.2, ssd=0.8, ssim=0.3, total_score=0.3),
        ]

    def test_config_defaults(self):
        cfg = PatchScoreConfig()
        assert cfg.min_score == 0.0
        assert cfg.max_pairs == 1000

    def test_config_invalid_method(self):
        with pytest.raises(ValueError):
            PatchScoreConfig(method="invalid")

    def test_make_patch_entry(self):
        e = make_patch_entry(0, 1, 2, 0, 1, ncc=0.5, ssd=0.2, ssim=0.7, total_score=0.6)
        assert e.pair == (1, 2)
        assert e.is_good is True

    def test_is_good_boundary(self):
        e = make_patch_entry(0, 0, 1, 0, 1, ncc=0.5, ssd=0.5, ssim=0.5, total_score=0.5)
        assert e.is_good is False  # > 0.5 required

    def test_summarise_empty(self):
        s = summarise_patch_scores([])
        assert s.n_total == 0 and s.mean_total == 0.0

    def test_summarise_fields(self):
        entries = self._make_entries()
        s = summarise_patch_scores(entries)
        assert s.n_total == 3
        assert s.n_good == 1
        assert s.n_poor == 2
        assert s.max_total == pytest.approx(0.8)
        assert s.min_total == pytest.approx(0.3)

    def test_filter_good(self):
        entries = self._make_entries()
        good = filter_good_patch_scores(entries)
        assert all(e.total_score > 0.5 for e in good)

    def test_filter_poor(self):
        entries = self._make_entries()
        poor = filter_poor_patch_scores(entries)
        assert all(e.total_score <= 0.5 for e in poor)

    def test_filter_by_score_range(self):
        entries = self._make_entries()
        result = filter_patch_by_score_range(entries, lo=0.4, hi=0.7)
        assert len(result) == 1 and result[0].total_score == pytest.approx(0.5)

    def test_filter_by_side_pair(self):
        entries = self._make_entries()
        result = filter_by_side_pair(entries, side1=0, side2=1)
        assert len(result) == 1

    def test_filter_by_ncc_range(self):
        entries = self._make_entries()
        result = filter_by_ncc_range(entries, lo=0.6, hi=1.0)
        assert len(result) == 1 and result[0].ncc == pytest.approx(0.8)

    def test_top_k(self):
        entries = self._make_entries()
        top = top_k_patch_entries(entries, k=2)
        assert len(top) == 2
        assert top[0].total_score >= top[1].total_score

    def test_best_entry(self):
        entries = self._make_entries()
        best = best_patch_entry(entries)
        assert best.total_score == pytest.approx(0.8)

    def test_compare_summaries(self):
        e1 = [make_patch_entry(0, 0, 1, 0, 1, ncc=0.9, ssd=0.1, ssim=0.9, total_score=0.9)]
        e2 = [make_patch_entry(0, 0, 1, 0, 1, ncc=0.4, ssd=0.5, ssim=0.4, total_score=0.4)]
        s1 = summarise_patch_scores(e1)
        s2 = summarise_patch_scores(e2)
        cmp = compare_patch_summaries(s1, s2)
        assert cmp["a_better"] is True
        assert cmp["delta_mean_total"] == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. patch_utils
# ═══════════════════════════════════════════════════════════════════════════════

class TestPatchUtils:
    def _gray_img(self, h=64, w=64):
        return rng.integers(0, 256, (h, w), dtype=np.uint8)

    def _color_img(self, h=64, w=64):
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    def test_patch_config_defaults(self):
        cfg = PatchConfig()
        assert cfg.patch_h == 32 and cfg.patch_w == 32

    def test_patch_config_invalid_h(self):
        with pytest.raises(ValueError):
            PatchConfig(patch_h=0)

    def test_patch_config_invalid_norm_mode(self):
        with pytest.raises(ValueError):
            PatchConfig(norm_mode="bad")

    def test_extract_patch_shape(self):
        img = self._gray_img()
        cfg = PatchConfig(patch_h=16, patch_w=16)
        patch = extract_patch(img, 32, 32, cfg)
        assert patch.shape == (16, 16)

    def test_extract_patch_color_shape(self):
        img = self._color_img()
        cfg = PatchConfig(patch_h=8, patch_w=8)
        patch = extract_patch(img, 32, 32, cfg)
        assert patch.shape == (8, 8, 3)

    def test_extract_patch_at_boundary(self):
        img = self._gray_img()
        cfg = PatchConfig(patch_h=16, patch_w=16, pad_value=0)
        patch = extract_patch(img, 0, 0, cfg)
        assert patch.shape == (16, 16)

    def test_extract_patches_multiple(self):
        img = self._gray_img()
        cfg = PatchConfig(patch_h=8, patch_w=8)
        centers = [(10, 10), (20, 20), (30, 30)]
        patches = extract_patches(img, centers, cfg)
        assert len(patches) == 3
        assert all(p.shape == (8, 8) for p in patches)

    def test_normalize_patch_minmax(self):
        patch = np.array([[0, 128, 255]], dtype=np.uint8)
        norm = normalize_patch(patch, mode="minmax")
        assert norm.min() == pytest.approx(0.0)
        assert norm.max() == pytest.approx(1.0)

    def test_normalize_patch_zscore(self):
        patch = rng.random((8, 8)).astype(np.float32)
        norm = normalize_patch(patch, mode="zscore")
        assert abs(float(norm.mean())) < 0.1

    def test_normalize_patch_constant_returns_zeros(self):
        patch = np.full((4, 4), 128, dtype=np.uint8)
        norm = normalize_patch(patch, mode="minmax")
        assert np.all(norm == 0.0)

    def test_patch_ssd_identical(self):
        p = rng.random((8, 8)).astype(np.float32)
        assert patch_ssd(p, p) == pytest.approx(0.0)

    def test_patch_ncc_identical(self):
        p = rng.random((8, 8)).astype(np.float32)
        assert patch_ncc(p, p) == pytest.approx(1.0, abs=1e-5)

    def test_patch_mse_identical(self):
        p = rng.random((8, 8)).astype(np.float32)
        assert patch_mse(p, p) == pytest.approx(0.0)

    def test_patch_ssd_shape_mismatch(self):
        a = np.ones((4, 4), dtype=np.float32)
        b = np.ones((5, 5), dtype=np.float32)
        with pytest.raises(ValueError):
            patch_ssd(a, b)

    def test_compare_patches_ncc(self):
        p = rng.random((8, 8)).astype(np.float32)
        val = compare_patches(p, p, method="ncc")
        assert -1.0 <= val <= 1.0

    def test_compare_patches_ssd(self):
        p = rng.random((8, 8)).astype(np.float32)
        val = compare_patches(p, p, method="ssd")
        assert val == pytest.approx(0.0)

    def test_compare_patches_mse(self):
        p = rng.random((8, 8)).astype(np.float32)
        val = compare_patches(p, p, method="mse")
        assert val == pytest.approx(0.0)

    def test_batch_compare_returns_list(self):
        img = self._gray_img()
        cfg = PatchConfig(patch_h=8, patch_w=8)
        p1 = extract_patch(img, 10, 10, cfg)
        p2 = extract_patch(img, 20, 20, cfg)
        p3 = extract_patch(img, 30, 30, cfg)
        results = batch_compare([(p1, p2), (p2, p3)], method="ncc")
        assert len(results) == 2
        assert all(isinstance(r, float) for r in results)
