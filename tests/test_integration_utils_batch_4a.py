"""Integration tests – utils batch 4a:
mask_layout_utils, match_rank_utils, morph_utils,
noise_stats_utils, normalization_utils.
"""
import numpy as np
import pytest

RNG = np.random.default_rng(42)

# ── mask_layout_utils ────────────────────────────────────────────────────────
from puzzle_reconstruction.utils.mask_layout_utils import (
    MaskOpRecord, MaskCoverageRecord, FragmentPlacementRecord,
    LayoutDiffRecord, FeatureSelectionRecord,
    make_mask_coverage_record, make_layout_diff_record,
)

def test_mask_op_area_change():
    assert MaskOpRecord("erode", (100, 100), 500, 400).area_change == -100

def test_mask_op_coverage_ratio():
    assert MaskOpRecord("dilate", (10, 10), 0, 50).coverage_ratio == pytest.approx(0.5)

def test_mask_op_invalid_op():
    with pytest.raises(ValueError, match="Unknown mask operation"):
        MaskOpRecord("bad_op", (10, 10), 0, 0)

def test_mask_op_negative_before():
    with pytest.raises(ValueError):
        MaskOpRecord("crop", (10, 10), -1, 0)

def test_mask_coverage_ratio():
    assert MaskCoverageRecord(3, (100, 100), 2500, 10000).coverage_ratio == pytest.approx(0.25)

def test_mask_coverage_fully_covered():
    assert MaskCoverageRecord(1, (10, 10), 100, 100).is_fully_covered is True

def test_mask_coverage_not_covered():
    assert MaskCoverageRecord(1, (10, 10), 50, 100).is_fully_covered is False

def test_fragment_placement_coverage():
    rec = FragmentPlacementRecord(10, 7)
    assert rec.coverage == pytest.approx(0.7)
    assert rec.n_missing == 3

def test_fragment_placement_invalid():
    with pytest.raises(ValueError):
        FragmentPlacementRecord(5, 6)

def test_make_mask_coverage_record():
    masks = [np.zeros((20, 20), dtype=np.uint8), np.ones((20, 20), dtype=np.uint8) * 255]
    rec = make_mask_coverage_record(masks, (20, 20))
    assert rec.n_masks == 2 and rec.coverage_ratio == pytest.approx(1.0)

def test_make_layout_diff_record():
    rec = make_layout_diff_record({"n_fragments": 5, "mean_shift": 1.5, "max_shift": 3.0, "n_moved": 2})
    assert rec.mean_shift == pytest.approx(1.5) and rec.is_stable is False

def test_layout_diff_is_stable():
    assert LayoutDiffRecord(4, 0.0, 0.0, 0).is_stable is True

# ── match_rank_utils ─────────────────────────────────────────────────────────
from puzzle_reconstruction.utils.match_rank_utils import (
    RankingEntry, make_ranking_entry, summarise_ranking_entries,
    filter_ranking_by_algorithm, filter_ranking_by_min_top_score,
    filter_ranking_by_min_acceptance, top_k_ranking_entries,
    best_ranking_entry, ranking_score_stats, compare_ranking_summaries,
    batch_summarise_ranking_entries,
)

def _entries():
    return [
        make_ranking_entry(0, 100, 80, 0.95, 0.80, "algo_a"),
        make_ranking_entry(1,  50, 10, 0.60, 0.55, "algo_b"),
        make_ranking_entry(2, 200, 150, 0.88, 0.75, "algo_a"),
    ]

def test_ranking_acceptance_rate():
    assert make_ranking_entry(0, 100, 80, 0.9, 0.7, "a").acceptance_rate == pytest.approx(0.8)

def test_ranking_zero_pairs():
    assert RankingEntry(0, 0, 0, 0.0, 0.0, "x").acceptance_rate == pytest.approx(0.0)

def test_summarise_basic():
    s = summarise_ranking_entries(_entries())
    assert s.n_batches == 3 and s.total_pairs == 350 and s.best_batch_id == 0

def test_summarise_empty():
    s = summarise_ranking_entries([])
    assert s.n_batches == 0 and s.best_batch_id is None

def test_filter_by_algorithm():
    filtered = filter_ranking_by_algorithm(_entries(), "algo_a")
    assert len(filtered) == 2 and all(e.algorithm == "algo_a" for e in filtered)

def test_filter_by_min_top_score():
    assert len(filter_ranking_by_min_top_score(_entries(), 0.85)) == 2

def test_filter_by_min_acceptance():
    assert len(filter_ranking_by_min_acceptance(_entries(), 0.7)) == 2

def test_top_k_entries():
    top = top_k_ranking_entries(_entries(), 2)
    assert len(top) == 2 and top[0].top_score >= top[1].top_score

def test_best_entry():
    assert best_ranking_entry(_entries()).top_score == pytest.approx(0.95)

def test_score_stats():
    stats = ranking_score_stats(_entries())
    assert stats["count"] == 3 and stats["max"] == pytest.approx(0.95)

def test_compare_summaries():
    e = _entries()
    cmp = compare_ranking_summaries(summarise_ranking_entries(e[:1]), summarise_ranking_entries(e[1:]))
    assert "delta_mean_top_score" in cmp and isinstance(cmp["same_best"], bool)

def test_batch_summarise():
    e = _entries()
    summaries = batch_summarise_ranking_entries([e[:2], e[2:]])
    assert len(summaries) == 2 and summaries[0].n_batches == 2

# ── morph_utils ──────────────────────────────────────────────────────────────
from puzzle_reconstruction.utils.morph_utils import (
    MorphConfig, apply_erosion, apply_dilation, apply_opening, apply_closing,
    get_skeleton, label_regions, filter_regions_by_size, compute_region_stats,
)

def _bin_img():
    img = np.zeros((50, 50), dtype=np.uint8)
    img[10:40, 10:40] = 255
    return img

def test_morph_config_defaults():
    cfg = MorphConfig()
    assert cfg.kernel_size == 3 and cfg.kernel_shape == "rect" and cfg.iterations == 1

def test_morph_config_even_kernel():
    with pytest.raises(ValueError):
        MorphConfig(kernel_size=4)

def test_morph_config_bad_shape():
    with pytest.raises(ValueError):
        MorphConfig(kernel_shape="diamond")

def test_erosion_reduces():
    img = _bin_img()
    assert np.count_nonzero(apply_erosion(img, MorphConfig(kernel_size=5))) < np.count_nonzero(img)

def test_dilation_grows():
    img = _bin_img()
    assert np.count_nonzero(apply_dilation(img, MorphConfig(kernel_size=5))) > np.count_nonzero(img)

def test_opening_shape():
    img = _bin_img()
    r = apply_opening(img)
    assert r.shape == img.shape and r.dtype == np.uint8

def test_closing_shape():
    assert apply_closing(_bin_img()).shape == (50, 50)

def test_skeleton_binary():
    skel = get_skeleton(_bin_img())
    assert set(np.unique(skel)).issubset({0, 255})

def test_label_regions_two_components():
    img = np.zeros((50, 50), dtype=np.uint8)
    img[5:15, 5:15] = 255
    img[30:40, 30:40] = 255
    assert label_regions(img)[0] == 2

def test_label_regions_bad_connectivity():
    with pytest.raises(ValueError):
        label_regions(_bin_img(), connectivity=6)

def test_filter_regions_by_size():
    img = np.zeros((60, 60), dtype=np.uint8)
    img[5:10, 5:10] = 255   # ~25 px
    img[20:50, 20:50] = 255  # ~900 px
    assert label_regions(filter_regions_by_size(img, min_area=50))[0] == 1

def test_compute_region_stats():
    stats = compute_region_stats(_bin_img())
    assert len(stats) == 1 and stats[0]["area"] > 0

# ── noise_stats_utils ────────────────────────────────────────────────────────
from puzzle_reconstruction.utils.noise_stats_utils import (
    NoiseStatsConfig, make_noise_entry, summarise_noise_stats,
    filter_clean_entries, filter_noisy_entries,
    filter_by_sigma_range, filter_by_snr_range, filter_by_jpeg_threshold,
)

def _noise_entries():
    return [
        make_noise_entry(0,  2.0, 35.0, 0.1, 0.05, "clean"),
        make_noise_entry(1, 15.0, 18.0, 0.4, 0.30, "noisy"),
        make_noise_entry(2, 30.0, 10.0, 0.6, 0.70, "very_noisy"),
    ]

def test_noise_config_defaults():
    assert NoiseStatsConfig().max_sigma == pytest.approx(50.0)

def test_noise_config_invalid_sigma():
    with pytest.raises(ValueError):
        NoiseStatsConfig(max_sigma=0.0)

def test_noise_entry_is_clean():
    e = make_noise_entry(0, 1.0, 40.0, 0.0, 0.0, "clean")
    assert e.is_clean and not e.is_noisy

def test_noise_entry_is_noisy():
    assert make_noise_entry(1, 10.0, 20.0, 0.3, 0.2, "noisy").is_noisy

def test_noise_entry_bad_jpeg():
    with pytest.raises(ValueError):
        make_noise_entry(0, 1.0, 30.0, 1.5, 0.0, "clean")

def test_summarise_noise_basic():
    s = summarise_noise_stats(_noise_entries())
    assert s.n_total == 3 and s.n_clean == 1 and s.max_sigma == pytest.approx(30.0)

def test_summarise_noise_empty():
    s = summarise_noise_stats([])
    assert s.n_total == 0 and s.mean_sigma == pytest.approx(0.0)

def test_filter_clean():
    assert len(filter_clean_entries(_noise_entries())) == 1

def test_filter_noisy():
    assert len(filter_noisy_entries(_noise_entries())) == 2

def test_filter_sigma_range():
    filtered = filter_by_sigma_range(_noise_entries(), lo=5.0, hi=20.0)
    assert len(filtered) == 1 and filtered[0].sigma == pytest.approx(15.0)

def test_filter_snr_range():
    assert len(filter_by_snr_range(_noise_entries(), lo=15.0, hi=40.0)) == 2

def test_filter_jpeg_threshold():
    assert len(filter_by_jpeg_threshold(_noise_entries(), max_jpeg=0.3)) == 1

# ── normalization_utils ───────────────────────────────────────────────────────
from puzzle_reconstruction.utils.normalization_utils import (
    l1_normalize, l2_normalize, minmax_normalize, zscore_normalize,
    softmax, clamp, symmetrize_matrix, zero_diagonal,
    normalize_rows, batch_l2_normalize,
)

def test_l1_sum_one():
    assert np.sum(np.abs(l1_normalize(RNG.random(10)))) == pytest.approx(1.0)

def test_l1_zero_vector():
    assert np.all(l1_normalize(np.zeros(5)) == 0.0)

def test_l2_unit_norm():
    assert np.linalg.norm(l2_normalize(RNG.random(8))) == pytest.approx(1.0)

def test_l2_raises_2d():
    with pytest.raises(ValueError):
        l2_normalize(RNG.random((3, 3)))

def test_minmax_range():
    out = minmax_normalize(np.array([1.0, 3.0, 5.0, 2.0, 4.0]))
    assert out.min() == pytest.approx(0.0) and out.max() == pytest.approx(1.0)

def test_minmax_constant():
    assert np.all(minmax_normalize(np.ones(5) * 3.0) == 0.0)

def test_zscore_mean_std():
    out = zscore_normalize(RNG.random(20) * 10 + 5)
    assert out.mean() == pytest.approx(0.0, abs=1e-10) and out.std() == pytest.approx(1.0, abs=1e-10)

def test_softmax_sums_one():
    out = softmax(RNG.random(6))
    assert out.sum() == pytest.approx(1.0) and np.all(out >= 0)

def test_softmax_bad_temp():
    with pytest.raises(ValueError):
        softmax(np.array([1.0, 2.0, 3.0]), temperature=0.0)

def test_clamp_bounds():
    out = clamp(np.array([-1.0, 0.5, 2.0, 5.0]), 0.0, 1.0)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)

def test_clamp_invalid():
    with pytest.raises(ValueError):
        clamp(np.array([1.0]), lo=1.0, hi=0.0)

def test_symmetrize():
    mat = RNG.random((4, 4))
    sym = symmetrize_matrix(mat)
    assert np.allclose(sym, sym.T)

def test_zero_diagonal():
    result = zero_diagonal(RNG.random((5, 5)))
    assert np.all(np.diag(result) == 0.0)

def test_normalize_rows_l2():
    norms = np.linalg.norm(normalize_rows(RNG.random((4, 6)), method="l2"), axis=1)
    assert np.allclose(norms, 1.0)

def test_batch_l2_normalize():
    for v in batch_l2_normalize([RNG.random(5) for _ in range(4)]):
        assert np.linalg.norm(v) == pytest.approx(1.0)
