"""Integration tests for utils batch 4a:
mask_layout_utils, match_rank_utils, morph_utils,
noise_stats_utils, normalization_utils.
"""
import numpy as np
import pytest

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# mask_layout_utils
# ---------------------------------------------------------------------------
from puzzle_reconstruction.utils.mask_layout_utils import (
    MaskOpRecord,
    MaskCoverageRecord,
    FragmentPlacementRecord,
    LayoutDiffRecord,
    LayoutScoreRecord,
    FeatureSelectionRecord,
    PcaRecord,
    make_mask_coverage_record,
    make_layout_diff_record,
)


def test_mask_op_record_area_change():
    rec = MaskOpRecord("erode", (100, 100), 500, 400)
    assert rec.area_change == -100


def test_mask_op_record_coverage_ratio():
    rec = MaskOpRecord("dilate", (10, 10), 0, 50)
    assert rec.coverage_ratio == pytest.approx(0.5)


def test_mask_op_record_invalid_op():
    with pytest.raises(ValueError, match="Unknown mask operation"):
        MaskOpRecord("unknown_op", (10, 10), 0, 0)


def test_mask_op_record_negative_before():
    with pytest.raises(ValueError):
        MaskOpRecord("crop", (10, 10), -1, 0)


def test_mask_coverage_record_ratio():
    rec = MaskCoverageRecord(3, (100, 100), 2500, 10000)
    assert rec.coverage_ratio == pytest.approx(0.25)


def test_mask_coverage_record_fully_covered():
    rec = MaskCoverageRecord(1, (10, 10), 100, 100)
    assert rec.is_fully_covered is True


def test_mask_coverage_record_not_fully_covered():
    rec = MaskCoverageRecord(1, (10, 10), 50, 100)
    assert rec.is_fully_covered is False


def test_fragment_placement_record_coverage():
    rec = FragmentPlacementRecord(10, 7)
    assert rec.coverage == pytest.approx(0.7)
    assert rec.n_missing == 3


def test_fragment_placement_record_invalid():
    with pytest.raises(ValueError):
        FragmentPlacementRecord(5, 6)


def test_make_mask_coverage_record():
    masks = [
        np.zeros((20, 20), dtype=np.uint8),
        np.ones((20, 20), dtype=np.uint8) * 255,
    ]
    rec = make_mask_coverage_record(masks, (20, 20), label="test")
    assert rec.n_masks == 2
    assert rec.coverage_ratio == pytest.approx(1.0)


def test_make_layout_diff_record():
    d = {"n_fragments": 5, "mean_shift": 1.5, "max_shift": 3.0, "n_moved": 2}
    rec = make_layout_diff_record(d, label="iter1")
    assert rec.n_fragments == 5
    assert rec.mean_shift == pytest.approx(1.5)
    assert rec.is_stable is False


def test_layout_diff_record_is_stable():
    rec = LayoutDiffRecord(4, 0.0, 0.0, 0)
    assert rec.is_stable is True


# ---------------------------------------------------------------------------
# match_rank_utils
# ---------------------------------------------------------------------------
from puzzle_reconstruction.utils.match_rank_utils import (
    RankingConfig,
    RankingEntry,
    RankingSummary,
    make_ranking_entry,
    summarise_ranking_entries,
    filter_ranking_by_algorithm,
    filter_ranking_by_min_top_score,
    filter_ranking_by_min_acceptance,
    top_k_ranking_entries,
    best_ranking_entry,
    ranking_score_stats,
    compare_ranking_summaries,
    batch_summarise_ranking_entries,
)


def _make_entries():
    return [
        make_ranking_entry(0, 100, 80, 0.95, 0.80, "algo_a"),
        make_ranking_entry(1, 50, 10, 0.60, 0.55, "algo_b"),
        make_ranking_entry(2, 200, 150, 0.88, 0.75, "algo_a"),
    ]


def test_ranking_entry_acceptance_rate():
    e = make_ranking_entry(0, 100, 80, 0.9, 0.7, "algo_a")
    assert e.acceptance_rate == pytest.approx(0.8)


def test_ranking_entry_zero_pairs():
    e = RankingEntry(0, 0, 0, 0.0, 0.0, "algo_x")
    assert e.acceptance_rate == pytest.approx(0.0)


def test_summarise_ranking_entries_basic():
    entries = _make_entries()
    s = summarise_ranking_entries(entries)
    assert s.n_batches == 3
    assert s.total_pairs == 350
    assert s.best_batch_id == 0   # top_score 0.95


def test_summarise_ranking_entries_empty():
    s = summarise_ranking_entries([])
    assert s.n_batches == 0
    assert s.best_batch_id is None


def test_filter_ranking_by_algorithm():
    entries = _make_entries()
    filtered = filter_ranking_by_algorithm(entries, "algo_a")
    assert len(filtered) == 2
    assert all(e.algorithm == "algo_a" for e in filtered)


def test_filter_ranking_by_min_top_score():
    entries = _make_entries()
    filtered = filter_ranking_by_min_top_score(entries, 0.85)
    assert len(filtered) == 2


def test_filter_ranking_by_min_acceptance():
    entries = _make_entries()
    filtered = filter_ranking_by_min_acceptance(entries, 0.7)
    assert len(filtered) == 2


def test_top_k_ranking_entries():
    entries = _make_entries()
    top = top_k_ranking_entries(entries, 2)
    assert len(top) == 2
    assert top[0].top_score >= top[1].top_score


def test_best_ranking_entry():
    entries = _make_entries()
    best = best_ranking_entry(entries)
    assert best.top_score == pytest.approx(0.95)


def test_ranking_score_stats():
    entries = _make_entries()
    stats = ranking_score_stats(entries)
    assert stats["count"] == 3
    assert stats["max"] == pytest.approx(0.95)
    assert stats["min"] == pytest.approx(0.60)


def test_compare_ranking_summaries():
    entries = _make_entries()
    s1 = summarise_ranking_entries(entries[:1])
    s2 = summarise_ranking_entries(entries[1:])
    cmp = compare_ranking_summaries(s1, s2)
    assert "delta_mean_top_score" in cmp
    assert isinstance(cmp["same_best"], bool)


def test_batch_summarise_ranking_entries():
    entries = _make_entries()
    summaries = batch_summarise_ranking_entries([entries[:2], entries[2:]])
    assert len(summaries) == 2
    assert summaries[0].n_batches == 2


# ---------------------------------------------------------------------------
# morph_utils
# ---------------------------------------------------------------------------
from puzzle_reconstruction.utils.morph_utils import (
    MorphConfig,
    apply_erosion,
    apply_dilation,
    apply_opening,
    apply_closing,
    get_skeleton,
    label_regions,
    filter_regions_by_size,
    compute_region_stats,
)


def _binary_img():
    img = np.zeros((50, 50), dtype=np.uint8)
    img[10:40, 10:40] = 255
    return img


def test_morph_config_default():
    cfg = MorphConfig()
    assert cfg.kernel_size == 3
    assert cfg.kernel_shape == "rect"
    assert cfg.iterations == 1


def test_morph_config_invalid_kernel_size():
    with pytest.raises(ValueError):
        MorphConfig(kernel_size=4)


def test_morph_config_invalid_shape():
    with pytest.raises(ValueError):
        MorphConfig(kernel_shape="diamond")


def test_apply_erosion_reduces_nonzero():
    img = _binary_img()
    result = apply_erosion(img, MorphConfig(kernel_size=5))
    assert np.count_nonzero(result) < np.count_nonzero(img)


def test_apply_dilation_increases_nonzero():
    img = _binary_img()
    result = apply_dilation(img, MorphConfig(kernel_size=5))
    assert np.count_nonzero(result) > np.count_nonzero(img)


def test_apply_opening_shape_preserved():
    img = _binary_img()
    result = apply_opening(img)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_apply_closing_shape_preserved():
    img = _binary_img()
    result = apply_closing(img)
    assert result.shape == img.shape


def test_get_skeleton_binary_output():
    img = _binary_img()
    skel = get_skeleton(img)
    assert skel.shape == img.shape
    unique = np.unique(skel)
    assert set(unique).issubset({0, 255})


def test_label_regions_count():
    img = np.zeros((50, 50), dtype=np.uint8)
    img[5:15, 5:15] = 255
    img[30:40, 30:40] = 255
    n, label_map = label_regions(img)
    assert n == 2


def test_label_regions_invalid_connectivity():
    img = _binary_img()
    with pytest.raises(ValueError):
        label_regions(img, connectivity=6)


def test_filter_regions_by_size():
    img = np.zeros((60, 60), dtype=np.uint8)
    img[5:10, 5:10] = 255   # small: 25 px
    img[20:50, 20:50] = 255  # large: 900 px
    result = filter_regions_by_size(img, min_area=50)
    n, _ = label_regions(result)
    assert n == 1


def test_compute_region_stats():
    img = _binary_img()
    stats = compute_region_stats(img)
    assert len(stats) == 1
    assert "area" in stats[0]
    assert stats[0]["area"] > 0


# ---------------------------------------------------------------------------
# noise_stats_utils
# ---------------------------------------------------------------------------
from puzzle_reconstruction.utils.noise_stats_utils import (
    NoiseStatsConfig,
    NoiseStatsEntry,
    NoiseStatsSummary,
    make_noise_entry,
    summarise_noise_stats,
    filter_clean_entries,
    filter_noisy_entries,
    filter_by_sigma_range,
    filter_by_snr_range,
    filter_by_jpeg_threshold,
)


def _make_noise_entries():
    return [
        make_noise_entry(0, 2.0, 35.0, 0.1, 0.05, "clean"),
        make_noise_entry(1, 15.0, 18.0, 0.4, 0.3, "noisy"),
        make_noise_entry(2, 30.0, 10.0, 0.6, 0.7, "very_noisy"),
    ]


def test_noise_stats_config_defaults():
    cfg = NoiseStatsConfig()
    assert cfg.max_sigma == pytest.approx(50.0)
    assert cfg.quality_levels == 3


def test_noise_stats_config_invalid_max_sigma():
    with pytest.raises(ValueError):
        NoiseStatsConfig(max_sigma=0.0)


def test_noise_stats_entry_is_clean():
    e = make_noise_entry(0, 1.0, 40.0, 0.0, 0.0, "clean")
    assert e.is_clean is True
    assert e.is_noisy is False


def test_noise_stats_entry_is_noisy():
    e = make_noise_entry(1, 10.0, 20.0, 0.3, 0.2, "noisy")
    assert e.is_noisy is True


def test_noise_stats_entry_invalid_jpeg():
    with pytest.raises(ValueError):
        make_noise_entry(0, 1.0, 30.0, 1.5, 0.0, "clean")


def test_summarise_noise_stats_basic():
    entries = _make_noise_entries()
    s = summarise_noise_stats(entries)
    assert s.n_total == 3
    assert s.n_clean == 1
    assert s.n_noisy == 2
    assert s.max_sigma == pytest.approx(30.0)


def test_summarise_noise_stats_empty():
    s = summarise_noise_stats([])
    assert s.n_total == 0
    assert s.mean_sigma == pytest.approx(0.0)


def test_filter_clean_entries():
    entries = _make_noise_entries()
    clean = filter_clean_entries(entries)
    assert len(clean) == 1
    assert clean[0].quality == "clean"


def test_filter_noisy_entries():
    entries = _make_noise_entries()
    noisy = filter_noisy_entries(entries)
    assert len(noisy) == 2


def test_filter_by_sigma_range():
    entries = _make_noise_entries()
    filtered = filter_by_sigma_range(entries, lo=5.0, hi=20.0)
    assert len(filtered) == 1
    assert filtered[0].sigma == pytest.approx(15.0)


def test_filter_by_snr_range():
    entries = _make_noise_entries()
    filtered = filter_by_snr_range(entries, lo=15.0, hi=40.0)
    assert len(filtered) == 2


def test_filter_by_jpeg_threshold():
    entries = _make_noise_entries()
    filtered = filter_by_jpeg_threshold(entries, max_jpeg=0.3)
    assert len(filtered) == 1


# ---------------------------------------------------------------------------
# normalization_utils
# ---------------------------------------------------------------------------
from puzzle_reconstruction.utils.normalization_utils import (
    l1_normalize,
    l2_normalize,
    minmax_normalize,
    zscore_normalize,
    softmax,
    clamp,
    symmetrize_matrix,
    zero_diagonal,
    normalize_rows,
    batch_l2_normalize,
)


def test_l1_normalize_sum_one():
    arr = RNG.random(10)
    out = l1_normalize(arr)
    assert np.sum(np.abs(out)) == pytest.approx(1.0)


def test_l1_normalize_zero_vector():
    arr = np.zeros(5)
    out = l1_normalize(arr)
    assert np.all(out == 0.0)


def test_l2_normalize_unit_norm():
    arr = RNG.random(8)
    out = l2_normalize(arr)
    assert np.linalg.norm(out) == pytest.approx(1.0)


def test_l2_normalize_raises_2d():
    arr = RNG.random((3, 3))
    with pytest.raises(ValueError):
        l2_normalize(arr)


def test_minmax_normalize_range():
    arr = np.array([1.0, 3.0, 5.0, 2.0, 4.0])
    out = minmax_normalize(arr)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_minmax_normalize_constant():
    arr = np.ones(5) * 3.0
    out = minmax_normalize(arr)
    assert np.all(out == 0.0)


def test_zscore_normalize_mean_std():
    arr = RNG.random(20) * 10 + 5
    out = zscore_normalize(arr)
    assert out.mean() == pytest.approx(0.0, abs=1e-10)
    assert out.std() == pytest.approx(1.0, abs=1e-10)


def test_softmax_sums_to_one():
    arr = RNG.random(6)
    out = softmax(arr)
    assert out.sum() == pytest.approx(1.0)
    assert np.all(out >= 0)


def test_softmax_invalid_temperature():
    arr = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        softmax(arr, temperature=0.0)


def test_clamp_values():
    arr = np.array([-1.0, 0.5, 2.0, 5.0])
    out = clamp(arr, 0.0, 1.0)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


def test_clamp_invalid_bounds():
    with pytest.raises(ValueError):
        clamp(np.array([1.0, 2.0]), lo=1.0, hi=0.0)


def test_symmetrize_matrix():
    mat = RNG.random((4, 4))
    sym = symmetrize_matrix(mat)
    assert np.allclose(sym, sym.T)


def test_zero_diagonal():
    mat = RNG.random((5, 5))
    result = zero_diagonal(mat)
    assert np.all(np.diag(result) == 0.0)


def test_normalize_rows_l2():
    mat = RNG.random((4, 6))
    result = normalize_rows(mat, method="l2")
    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0)


def test_batch_l2_normalize():
    vecs = [RNG.random(5) for _ in range(4)]
    normed = batch_l2_normalize(vecs)
    for v in normed:
        assert np.linalg.norm(v) == pytest.approx(1.0)
