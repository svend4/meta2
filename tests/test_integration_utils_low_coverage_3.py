"""Integration tests for 13 under-tested utility modules (batch 3).

Target: 12+ tests per module = 150+ tests total.
Each section covers one module using numpy.random.default_rng(42) for reproducibility.
"""
from __future__ import annotations

import math
import pytest
import numpy as np
import cv2

# ── Modules under test ─────────────────────────────────────────────────────
from puzzle_reconstruction.utils.freq_metric_utils import (
    BandEnergyRecord, SpectrumComparisonRecord, FreqBatchSummary,
    MetricSnapshot, MetricRunSummary, MovingAverageResult,
    GreedyStepRecord, AssemblyRunRecord,
    make_band_energy_record, make_metric_snapshot, make_greedy_step,
)
from puzzle_reconstruction.utils.frequency_utils import (
    FrequencyConfig, compute_fft_magnitude, radial_power_spectrum,
    frequency_band_energy, high_frequency_ratio,
    low_pass_filter, high_pass_filter, compare_frequency_spectra,
)
from puzzle_reconstruction.utils.gradient_utils import (
    GradientConfig, compute_gradient_magnitude, compute_gradient_direction,
    compute_sobel, compute_laplacian, threshold_gradient,
    suppress_non_maximum, compute_edge_density, batch_compute_gradients,
)
from puzzle_reconstruction.utils.image_cluster_utils import (
    ImageStatsAnalysisConfig, ImageStatsAnalysisEntry, ImageStatsAnalysisSummary,
    make_image_stats_entry, summarise_image_stats_entries,
    filter_by_min_sharpness, filter_by_max_entropy, filter_by_min_contrast,
    top_k_sharpest, best_image_stats_entry, image_stats_score_stats,
    compare_image_stats_summaries, batch_summarise_image_stats_entries,
    ClusteringAnalysisConfig, ClusteringAnalysisEntry, ClusteringAnalysisSummary,
    make_clustering_entry, summarise_clustering_entries,
    filter_clustering_by_min_silhouette, filter_clustering_by_max_inertia,
    filter_clustering_by_algorithm, filter_clustering_by_n_clusters,
    top_k_clustering_entries, best_clustering_entry, clustering_score_stats,
    compare_clustering_summaries, batch_summarise_clustering_entries,
)
from puzzle_reconstruction.utils.image_pipeline_utils import (
    FrequencyMatchRecord, FrequencyMatchSummary,
    PatchMatchRecord, PatchMatchSummary,
    CanvasBuildRecord, CanvasBuildSummary,
    summarize_frequency_matches, filter_frequency_matches,
    summarize_canvas_builds, summarize_patch_matches, top_frequency_matches,
)
from puzzle_reconstruction.utils.image_transform_utils import (
    ImageTransformConfig, TransformResult,
    rotate_image, flip_horizontal, flip_vertical,
    pad_image, crop_image, resize_image, resize_to_max_side,
    apply_affine, rotation_matrix_2x3,
    batch_rotate, batch_pad, batch_resize_to_max,
)
from puzzle_reconstruction.utils.interpolation_utils import (
    InterpolationConfig, lerp, lerp_array,
    bilinear_interpolate, resample_1d, fill_missing,
    interpolate_scores, smooth_interpolate, batch_resample,
)
from puzzle_reconstruction.utils.mask_layout_utils import (
    MaskOpRecord, MaskCoverageRecord, FragmentPlacementRecord,
    LayoutDiffRecord, LayoutScoreRecord,
    FeatureSelectionRecord, PcaRecord,
    make_mask_coverage_record, make_layout_diff_record,
)
from puzzle_reconstruction.utils.match_rank_utils import (
    RankingConfig, RankingEntry, RankingSummary,
    make_ranking_entry, summarise_ranking_entries,
    filter_ranking_by_algorithm, filter_ranking_by_min_top_score,
    filter_ranking_by_min_acceptance, top_k_ranking_entries,
    best_ranking_entry, ranking_score_stats, compare_ranking_summaries,
    batch_summarise_ranking_entries,
    EvalResultConfig, EvalResultEntry, EvalResultSummary,
    make_eval_result_entry, summarise_eval_result_entries,
    filter_eval_by_min_f1, filter_eval_by_algorithm,
    top_k_eval_entries, best_eval_entry, eval_f1_stats,
    compare_eval_summaries, batch_summarise_eval_entries,
)
from puzzle_reconstruction.utils.morph_utils import (
    MorphConfig, apply_erosion, apply_dilation,
    apply_opening, apply_closing, get_skeleton,
    label_regions, filter_regions_by_size, compute_region_stats,
    batch_morphology,
)
from puzzle_reconstruction.utils.noise_stats_utils import (
    NoiseStatsConfig, NoiseStatsEntry, NoiseStatsSummary,
    make_noise_entry, entries_from_analysis_results,
    summarise_noise_stats, filter_clean_entries, filter_noisy_entries,
    filter_by_sigma_range, filter_by_snr_range, filter_by_jpeg_threshold,
    top_k_cleanest, top_k_noisiest, best_snr_entry,
    noise_stats_dict, compare_noise_summaries, batch_summarise_noise_stats,
)
from puzzle_reconstruction.utils.normalization_utils import (
    l1_normalize, l2_normalize, minmax_normalize, zscore_normalize,
    softmax, clamp, symmetrize_matrix, zero_diagonal,
    normalize_rows, batch_l2_normalize,
)
from puzzle_reconstruction.utils.normalize_noise_utils import (
    NormResultConfig, NormResultEntry, NormResultSummary,
    make_norm_result_entry, summarise_norm_result_entries,
    filter_norm_by_method, filter_norm_by_min_spread,
    top_k_norm_by_spread, best_norm_entry, norm_spread_stats,
    compare_norm_summaries, batch_summarise_norm_entries,
    NoiseResultConfig, NoiseResultEntry, NoiseResultSummary,
    make_noise_result_entry, summarise_noise_result_entries,
    filter_noise_by_method, filter_noise_by_max_after,
    filter_noise_by_min_delta, top_k_noise_by_delta,
    best_noise_entry, noise_delta_stats,
    compare_noise_summaries as compare_noise_result_summaries,
    batch_summarise_noise_entries,
)

RNG = np.random.default_rng(42)


# ===========================================================================
# Helpers
# ===========================================================================

def gray_img(h=32, w=32, rng=None):
    """Return a reproducible uint8 grayscale image."""
    if rng is None:
        rng = RNG
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def bgr_img(h=32, w=32, rng=None):
    """Return a reproducible uint8 BGR image."""
    if rng is None:
        rng = RNG
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def binary_img(h=32, w=32, rng=None):
    """Return a reproducible uint8 binary image (0 or 255)."""
    if rng is None:
        rng = RNG
    raw = rng.integers(0, 2, (h, w), dtype=np.uint8)
    return raw * 255

# ===========================================================================
# 1. freq_metric_utils
# ===========================================================================

class TestBandEnergyRecord:
    def test_basic_construction(self):
        r = BandEnergyRecord(fragment_id=1, band_energies=[1.0, 2.0, 3.0])
        assert r.n_bands == 3
        assert r.fragment_id == 1

    def test_dominant_band(self):
        r = BandEnergyRecord(fragment_id=0, band_energies=[0.5, 3.0, 1.0])
        assert r.dominant_band == 1

    def test_total_energy(self):
        r = BandEnergyRecord(fragment_id=0, band_energies=[1.0, 2.0, 3.0])
        assert r.total_energy == pytest.approx(6.0)

    def test_normalized_energies_sum_to_one(self):
        r = BandEnergyRecord(fragment_id=0, band_energies=[1.0, 2.0, 3.0])
        ne = r.normalized_energies
        assert sum(ne) == pytest.approx(1.0)

    def test_normalized_energies_zero_total(self):
        r = BandEnergyRecord(fragment_id=0, band_energies=[0.0, 0.0])
        ne = r.normalized_energies
        assert ne == [0.0, 0.0]

    def test_dominant_band_empty(self):
        r = BandEnergyRecord(fragment_id=0, band_energies=[])
        assert r.dominant_band == 0

    def test_make_band_energy_record(self):
        r = make_band_energy_record(5, [1.0, 2.0])
        assert isinstance(r, BandEnergyRecord)
        assert r.fragment_id == 5
        assert r.n_bands == 2


class TestSpectrumComparisonRecord:
    def test_valid_construction(self):
        r = SpectrumComparisonRecord(0, 1, similarity=0.8)
        assert r.is_match is True

    def test_invalid_similarity_raises(self):
        with pytest.raises(ValueError):
            SpectrumComparisonRecord(0, 1, similarity=1.5)

    def test_similarity_at_boundary(self):
        r = SpectrumComparisonRecord(0, 1, similarity=0.5)
        assert r.is_match is True

    def test_below_threshold_not_match(self):
        r = SpectrumComparisonRecord(0, 1, similarity=0.3)
        assert r.is_match is False

    def test_fields(self):
        r = SpectrumComparisonRecord(2, 3, similarity=0.7, centroid_diff=0.1, entropy_diff=0.2)
        assert r.centroid_diff == pytest.approx(0.1)
        assert r.entropy_diff == pytest.approx(0.2)


class TestFreqBatchSummary:
    def test_is_valid(self):
        s = FreqBatchSummary(n_fragments=5, mean_entropy=3.0, mean_centroid=0.5, n_bands=16)
        assert s.is_valid is True

    def test_is_not_valid_zero_fragments(self):
        s = FreqBatchSummary(n_fragments=0, mean_entropy=3.0, mean_centroid=0.5, n_bands=16)
        assert s.is_valid is False

    def test_is_not_valid_zero_bands(self):
        s = FreqBatchSummary(n_fragments=5, mean_entropy=3.0, mean_centroid=0.5, n_bands=0)
        assert s.is_valid is False


class TestMetricSnapshot:
    def test_construction(self):
        s = make_metric_snapshot(10, {"loss": 0.5, "acc": 0.9}, label="train")
        assert s.step == 10
        assert s.n_metrics == 2

    def test_negative_step_raises(self):
        with pytest.raises(ValueError):
            MetricSnapshot(step=-1, values={})

    def test_get_existing(self):
        s = make_metric_snapshot(0, {"loss": 0.1})
        assert s.get("loss") == pytest.approx(0.1)

    def test_get_default(self):
        s = make_metric_snapshot(0, {"loss": 0.1})
        assert s.get("unknown", 99.0) == pytest.approx(99.0)

    def test_metric_names(self):
        s = make_metric_snapshot(0, {"a": 1.0, "b": 2.0})
        assert set(s.metric_names) == {"a", "b"}


class TestMetricRunSummary:
    def test_tracked_metrics(self):
        r = MetricRunSummary(
            namespace="test", total_steps=100,
            final_values={"loss": 0.1, "acc": 0.9}
        )
        assert set(r.tracked_metrics) == {"loss", "acc"}

    def test_best_and_final(self):
        r = MetricRunSummary(
            namespace="test", total_steps=100,
            best_values={"loss": 0.05},
            final_values={"loss": 0.1}
        )
        assert r.best("loss") == pytest.approx(0.05)
        assert r.final("loss") == pytest.approx(0.1)
        assert r.best("missing") is None


class TestMovingAverageResult:
    def test_length(self):
        m = MovingAverageResult(metric_name="x", window=3, smoothed=[1.0, 2.0, 3.0])
        assert m.length == 3

    def test_at(self):
        m = MovingAverageResult(metric_name="x", window=3, smoothed=[1.0, 2.0, 3.0])
        assert m.at(0) == pytest.approx(1.0)

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            MovingAverageResult(metric_name="x", window=0, smoothed=[1.0])


class TestGreedyStepRecord:
    def test_construction(self):
        s = make_greedy_step(0, fragment_id=1, anchor_id=2, score=0.9)
        assert s.step == 0 and s.score == pytest.approx(0.9)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            GreedyStepRecord(step=0, fragment_id=1, anchor_id=2, score=-0.1)


class TestAssemblyRunRecord:
    def test_placement_rate(self):
        steps = [make_greedy_step(i, i, 0, 1.0) for i in range(3)]
        r = AssemblyRunRecord(n_fragments=6, steps=steps)
        assert r.placement_rate == pytest.approx(0.5)
        assert r.n_placed == 3

    def test_placement_rate_zero_fragments(self):
        r = AssemblyRunRecord(n_fragments=0)
        assert r.placement_rate == pytest.approx(0.0)


# ===========================================================================
# 2. frequency_utils
# ===========================================================================

class TestFrequencyConfig:
    def test_defaults(self):
        cfg = FrequencyConfig()
        assert cfg.n_bands == 32
        assert cfg.log_scale is True
        assert cfg.normalize is True

    def test_invalid_n_bands(self):
        with pytest.raises(ValueError):
            FrequencyConfig(n_bands=1)

    def test_valid_minimal_bands(self):
        cfg = FrequencyConfig(n_bands=2)
        assert cfg.n_bands == 2


class TestComputeFftMagnitude:
    def test_gray_output_shape(self):
        img = gray_img(32, 32)
        mag = compute_fft_magnitude(img)
        assert mag.shape == (32, 32)
        assert mag.dtype == np.float32

    def test_normalized_range(self):
        img = gray_img(32, 32)
        mag = compute_fft_magnitude(img)
        assert mag.min() >= 0.0
        assert mag.max() <= 1.0 + 1e-6

    def test_bgr_output_shape(self):
        img = bgr_img(32, 32)
        mag = compute_fft_magnitude(img)
        assert mag.shape == (32, 32)

    def test_no_log_scale(self):
        img = gray_img(32, 32)
        cfg = FrequencyConfig(log_scale=False)
        mag = compute_fft_magnitude(img, cfg)
        assert mag.shape == (32, 32)

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError):
            compute_fft_magnitude(np.zeros((4, 4, 4, 4), dtype=np.uint8))

    def test_uniform_image(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        mag = compute_fft_magnitude(img)
        assert mag.dtype == np.float32


class TestRadialPowerSpectrum:
    def test_output_length(self):
        img = gray_img(32, 32)
        cfg = FrequencyConfig(n_bands=16)
        spec = radial_power_spectrum(img, cfg)
        assert spec.shape == (16,)

    def test_normalized_max_one(self):
        img = gray_img(32, 32)
        spec = radial_power_spectrum(img)
        assert float(spec.max()) <= 1.0 + 1e-6

    def test_non_negative(self):
        img = gray_img(32, 32)
        spec = radial_power_spectrum(img)
        assert np.all(spec >= 0)

    def test_bgr_image(self):
        img = bgr_img(32, 32)
        spec = radial_power_spectrum(img)
        assert spec.shape == (32,)

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError):
            radial_power_spectrum(np.zeros((4, 4, 4, 4), dtype=np.uint8))


class TestFrequencyBandEnergy:
    def test_returns_float(self):
        img = gray_img(32, 32)
        e = frequency_band_energy(img, 0.0, 0.5)
        assert isinstance(e, float) and e >= 0.0

    def test_invalid_fracs_raises(self):
        img = gray_img(32, 32)
        with pytest.raises(ValueError):
            frequency_band_energy(img, 0.5, 0.3)

    def test_full_band(self):
        img = gray_img(32, 32)
        e = frequency_band_energy(img, 0.0, 1.0)
        assert e > 0.0

    def test_narrow_band(self):
        img = gray_img(32, 32)
        e = frequency_band_energy(img, 0.1, 0.2)
        assert e >= 0.0


class TestHighFrequencyRatio:
    def test_range(self):
        img = gray_img(32, 32)
        r = high_frequency_ratio(img)
        assert 0.0 <= r <= 1.0

    def test_invalid_threshold_raises(self):
        img = gray_img(32, 32)
        with pytest.raises(ValueError):
            high_frequency_ratio(img, 0.0)
        with pytest.raises(ValueError):
            high_frequency_ratio(img, 1.0)

    def test_uniform_image(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        r = high_frequency_ratio(img)
        assert 0.0 <= r <= 1.0


class TestLowPassFilter:
    def test_output_shape_gray(self):
        img = gray_img(32, 32)
        out = low_pass_filter(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_output_shape_bgr(self):
        img = bgr_img(32, 32)
        out = low_pass_filter(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_invalid_cutoff_raises(self):
        img = gray_img(32, 32)
        with pytest.raises(ValueError):
            low_pass_filter(img, cutoff_frac=0.0)


class TestHighPassFilter:
    def test_output_shape_gray(self):
        img = gray_img(32, 32)
        out = high_pass_filter(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_output_shape_bgr(self):
        img = bgr_img(32, 32)
        out = high_pass_filter(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_invalid_cutoff_raises(self):
        img = gray_img(32, 32)
        with pytest.raises(ValueError):
            high_pass_filter(img, cutoff_frac=1.0)


class TestCompareFrequencySpectra:
    def test_identical_images(self):
        img = gray_img(32, 32)
        sim = compare_frequency_spectra(img, img)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_range(self):
        img1 = gray_img(32, 32)
        img2 = gray_img(32, 32, np.random.default_rng(99))
        sim = compare_frequency_spectra(img1, img2)
        assert 0.0 <= sim <= 1.0

    def test_returns_float(self):
        img = gray_img(32, 32)
        sim = compare_frequency_spectra(img, img)
        assert isinstance(sim, float)


# ===========================================================================
# 3. gradient_utils
# ===========================================================================

class TestGradientConfig:
    def test_defaults(self):
        cfg = GradientConfig()
        assert cfg.ksize == 3
        assert cfg.normalize is True

    def test_invalid_ksize_even(self):
        with pytest.raises(ValueError):
            GradientConfig(ksize=2)

    def test_invalid_ksize_zero(self):
        with pytest.raises(ValueError):
            GradientConfig(ksize=0)

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            GradientConfig(threshold=300.0)

    def test_valid_ksize_5(self):
        cfg = GradientConfig(ksize=5)
        assert cfg.ksize == 5


class TestComputeGradientMagnitude:
    def test_shape_gray(self):
        img = gray_img(32, 32)
        mag = compute_gradient_magnitude(img)
        assert mag.shape == (32, 32)
        assert mag.dtype == np.float32

    def test_normalized_range(self):
        img = gray_img(32, 32)
        mag = compute_gradient_magnitude(img)
        assert mag.min() >= 0.0
        assert mag.max() <= 1.0 + 1e-6

    def test_shape_bgr(self):
        img = bgr_img(32, 32)
        mag = compute_gradient_magnitude(img)
        assert mag.shape == (32, 32)

    def test_uniform_image_zero_gradient(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        mag = compute_gradient_magnitude(img)
        assert mag.max() == pytest.approx(0.0, abs=1e-5)

    def test_invalid_ndim(self):
        with pytest.raises(ValueError):
            compute_gradient_magnitude(np.zeros((4, 4, 4, 4), dtype=np.uint8))


class TestComputeGradientDirection:
    def test_shape(self):
        img = gray_img(32, 32)
        d = compute_gradient_direction(img)
        assert d.shape == (32, 32)
        assert d.dtype == np.float32

    def test_range(self):
        img = gray_img(32, 32)
        d = compute_gradient_direction(img)
        assert np.all(d >= -math.pi - 1e-5)
        assert np.all(d <= math.pi + 1e-5)

    def test_bgr(self):
        img = bgr_img(32, 32)
        d = compute_gradient_direction(img)
        assert d.shape == (32, 32)


class TestComputeSobel:
    def test_returns_tuple(self):
        img = gray_img(32, 32)
        mag, dx, dy = compute_sobel(img)
        assert mag.shape == (32, 32)
        assert dx.shape == (32, 32)
        assert dy.shape == (32, 32)

    def test_mag_normalized(self):
        img = gray_img(32, 32)
        mag, _, _ = compute_sobel(img)
        assert mag.max() <= 1.0 + 1e-6

    def test_invalid_ndim(self):
        with pytest.raises(ValueError):
            compute_sobel(np.zeros((4, 4, 4, 4), dtype=np.uint8))


class TestComputeLaplacian:
    def test_shape(self):
        img = gray_img(32, 32)
        lap = compute_laplacian(img)
        assert lap.shape == (32, 32)

    def test_normalized_non_negative(self):
        img = gray_img(32, 32)
        lap = compute_laplacian(img, normalize=True)
        assert lap.min() >= 0.0

    def test_invalid_ksize(self):
        img = gray_img(32, 32)
        with pytest.raises(ValueError):
            compute_laplacian(img, ksize=2)

    def test_ksize_1(self):
        img = gray_img(32, 32)
        lap = compute_laplacian(img, ksize=1)
        assert lap.shape == (32, 32)


class TestThresholdGradient:
    def test_binary_mask(self):
        img = gray_img(32, 32)
        mag = compute_gradient_magnitude(img)
        mask = threshold_gradient(mag)
        assert mask.dtype == bool
        assert mask.shape == (32, 32)

    def test_explicit_threshold(self):
        mag = np.array([[0.1, 0.5], [0.8, 0.2]], dtype=np.float32)
        mask = threshold_gradient(mag, threshold=0.4)
        assert mask[0, 0] == False
        assert mask[0, 1] == True
        assert mask[1, 0] == True
        assert mask[1, 1] == False

    def test_invalid_ndim(self):
        with pytest.raises(ValueError):
            threshold_gradient(np.zeros((4, 4, 3), dtype=np.float32))


class TestSuppressNonMaximum:
    def test_shape(self):
        img = gray_img(16, 16)
        mag = compute_gradient_magnitude(img)
        dirn = compute_gradient_direction(img)
        nms = suppress_non_maximum(mag, dirn)
        assert nms.shape == mag.shape

    def test_non_negative(self):
        img = gray_img(16, 16)
        mag = compute_gradient_magnitude(img)
        dirn = compute_gradient_direction(img)
        nms = suppress_non_maximum(mag, dirn)
        assert nms.min() >= 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            suppress_non_maximum(
                np.zeros((8, 8), dtype=np.float32),
                np.zeros((8, 9), dtype=np.float32)
            )


class TestComputeEdgeDensity:
    def test_range(self):
        img = gray_img(32, 32)
        d = compute_edge_density(img)
        assert 0.0 <= d <= 1.0

    def test_uniform_low_density(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        d = compute_edge_density(img)
        assert d == pytest.approx(0.0, abs=1e-5)

    def test_with_roi(self):
        img = gray_img(32, 32)
        d = compute_edge_density(img, roi=(0, 0, 16, 16))
        assert 0.0 <= d <= 1.0


class TestBatchComputeGradients:
    def test_output_length(self):
        imgs = [gray_img(16, 16) for _ in range(4)]
        mags = batch_compute_gradients(imgs)
        assert len(mags) == 4

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            batch_compute_gradients([])

    def test_shapes_match(self):
        imgs = [gray_img(16, 16) for _ in range(3)]
        mags = batch_compute_gradients(imgs)
        for m in mags:
            assert m.shape == (16, 16)


# ===========================================================================
# 4. image_cluster_utils
# ===========================================================================

def _make_stats_entries(n=5):
    rng = np.random.default_rng(7)
    entries = []
    for i in range(n):
        entries.append(make_image_stats_entry(
            fragment_id=i,
            sharpness=float(rng.uniform(0.1, 1.0)),
            entropy=float(rng.uniform(2.0, 7.0)),
            contrast=float(rng.uniform(5.0, 50.0)),
            mean=float(rng.uniform(100.0, 200.0)),
            n_pixels=1024,
        ))
    return entries


class TestMakeImageStatsEntry:
    def test_creates_entry(self):
        e = make_image_stats_entry(0, 0.5, 3.0, 20.0, 128.0, 1024)
        assert e.fragment_id == 0
        assert e.sharpness == pytest.approx(0.5)

    def test_params_stored(self):
        e = make_image_stats_entry(1, 0.5, 3.0, 20.0, 128.0, 1024, foo="bar")
        assert e.params["foo"] == "bar"


class TestSummariseImageStatsEntries:
    def test_summary_fields(self):
        entries = _make_stats_entries(5)
        s = summarise_image_stats_entries(entries)
        assert s.n_images == 5
        assert isinstance(s.mean_sharpness, float)

    def test_empty_returns_zeros(self):
        s = summarise_image_stats_entries([])
        assert s.n_images == 0
        assert s.sharpest_id is None

    def test_sharpest_blurriest(self):
        entries = _make_stats_entries(5)
        s = summarise_image_stats_entries(entries)
        sharpnesses = [e.sharpness for e in entries]
        assert entries[s.sharpest_id].sharpness == max(sharpnesses)


class TestFilterImageStats:
    def test_filter_by_min_sharpness(self):
        entries = _make_stats_entries(10)
        filtered = filter_by_min_sharpness(entries, 0.5)
        assert all(e.sharpness >= 0.5 for e in filtered)

    def test_filter_by_max_entropy(self):
        entries = _make_stats_entries(10)
        filtered = filter_by_max_entropy(entries, 5.0)
        assert all(e.entropy <= 5.0 for e in filtered)

    def test_filter_by_min_contrast(self):
        entries = _make_stats_entries(10)
        filtered = filter_by_min_contrast(entries, 20.0)
        assert all(e.contrast >= 20.0 for e in filtered)

    def test_top_k_sharpest_count(self):
        entries = _make_stats_entries(10)
        top = top_k_sharpest(entries, 3)
        assert len(top) == 3

    def test_top_k_sharpest_order(self):
        entries = _make_stats_entries(10)
        top = top_k_sharpest(entries, 3)
        sharpnesses = [e.sharpness for e in top]
        assert sharpnesses == sorted(sharpnesses, reverse=True)

    def test_best_image_stats_entry(self):
        entries = _make_stats_entries(5)
        best = best_image_stats_entry(entries)
        assert best is not None
        assert best.sharpness == max(e.sharpness for e in entries)

    def test_best_entry_empty(self):
        assert best_image_stats_entry([]) is None

    def test_score_stats(self):
        entries = _make_stats_entries(5)
        stats = image_stats_score_stats(entries)
        assert "min" in stats and "max" in stats and "mean" in stats and "std" in stats

    def test_score_stats_empty(self):
        stats = image_stats_score_stats([])
        assert stats["count"] == 0

    def test_compare_summaries(self):
        entries1 = _make_stats_entries(3)
        entries2 = _make_stats_entries(6)
        s1 = summarise_image_stats_entries(entries1)
        s2 = summarise_image_stats_entries(entries2)
        cmp = compare_image_stats_summaries(s1, s2)
        assert "delta_mean_sharpness" in cmp
        assert "delta_n_images" in cmp

    def test_batch_summarise(self):
        groups = [_make_stats_entries(3), _make_stats_entries(4)]
        summaries = batch_summarise_image_stats_entries(groups)
        assert len(summaries) == 2


def _make_clustering_entries(n=5):
    rng = np.random.default_rng(13)
    entries = []
    for i in range(n):
        entries.append(make_clustering_entry(
            run_id=i,
            n_clusters=int(rng.integers(2, 8)),
            inertia=float(rng.uniform(100.0, 1000.0)),
            silhouette=float(rng.uniform(-0.2, 0.9)),
            algorithm="kmeans",
            n_samples=100,
        ))
    return entries


class TestMakeClusteringEntry:
    def test_creates_entry(self):
        e = make_clustering_entry(0, 3, 500.0, 0.5, "kmeans", 100)
        assert e.run_id == 0
        assert e.n_clusters == 3

    def test_params_stored(self):
        e = make_clustering_entry(0, 3, 500.0, 0.5, "kmeans", 100, init="random")
        assert e.params["init"] == "random"


class TestSummariseClusteringEntries:
    def test_summary_fields(self):
        entries = _make_clustering_entries(5)
        s = summarise_clustering_entries(entries)
        assert s.n_runs == 5
        assert isinstance(s.mean_inertia, float)

    def test_empty_returns_zeros(self):
        s = summarise_clustering_entries([])
        assert s.n_runs == 0
        assert s.best_run_id is None

    def test_best_worst(self):
        entries = _make_clustering_entries(5)
        s = summarise_clustering_entries(entries)
        sils = [e.silhouette for e in entries]
        assert entries[s.best_run_id].silhouette == max(sils)


class TestFilterClusteringEntries:
    def test_filter_by_min_silhouette(self):
        entries = _make_clustering_entries(10)
        filtered = filter_clustering_by_min_silhouette(entries, 0.3)
        assert all(e.silhouette >= 0.3 for e in filtered)

    def test_filter_by_max_inertia(self):
        entries = _make_clustering_entries(10)
        filtered = filter_clustering_by_max_inertia(entries, 600.0)
        assert all(e.inertia <= 600.0 for e in filtered)

    def test_filter_by_algorithm(self):
        entries = _make_clustering_entries(5)
        entries[0] = make_clustering_entry(0, 3, 500.0, 0.5, "dbscan", 100)
        filtered = filter_clustering_by_algorithm(entries, "kmeans")
        assert all(e.algorithm == "kmeans" for e in filtered)

    def test_filter_by_n_clusters(self):
        entries = _make_clustering_entries(10)
        target_n = entries[0].n_clusters
        filtered = filter_clustering_by_n_clusters(entries, target_n)
        assert all(e.n_clusters == target_n for e in filtered)

    def test_top_k_entries(self):
        entries = _make_clustering_entries(10)
        top = top_k_clustering_entries(entries, 3)
        assert len(top) == 3

    def test_best_clustering_entry(self):
        entries = _make_clustering_entries(5)
        best = best_clustering_entry(entries)
        assert best.silhouette == max(e.silhouette for e in entries)

    def test_best_empty(self):
        assert best_clustering_entry([]) is None

    def test_score_stats(self):
        entries = _make_clustering_entries(5)
        stats = clustering_score_stats(entries)
        assert stats["count"] == 5

    def test_compare_summaries(self):
        s1 = summarise_clustering_entries(_make_clustering_entries(3))
        s2 = summarise_clustering_entries(_make_clustering_entries(5))
        cmp = compare_clustering_summaries(s1, s2)
        assert "delta_mean_silhouette" in cmp

    def test_batch_summarise(self):
        groups = [_make_clustering_entries(3), _make_clustering_entries(4)]
        summaries = batch_summarise_clustering_entries(groups)
        assert len(summaries) == 2


# ===========================================================================
# 5. image_pipeline_utils
# ===========================================================================

class TestFrequencyMatchRecord:
    def test_valid_construction(self):
        r = FrequencyMatchRecord(0, 1, similarity=0.8)
        assert r.pair == (0, 1)
        assert r.is_similar is True

    def test_invalid_similarity(self):
        with pytest.raises(ValueError):
            FrequencyMatchRecord(0, 1, similarity=1.5)

    def test_negative_id_raises(self):
        with pytest.raises(ValueError):
            FrequencyMatchRecord(-1, 1, similarity=0.5)

    def test_below_threshold_not_similar(self):
        r = FrequencyMatchRecord(0, 1, similarity=0.3)
        assert r.is_similar is False

    def test_method_field(self):
        r = FrequencyMatchRecord(0, 1, similarity=0.7, method="cosine")
        assert r.method == "cosine"


class TestFrequencyMatchSummary:
    def test_similar_ratio(self):
        s = FrequencyMatchSummary(total_pairs=10, similar_pairs=4)
        assert s.similar_ratio == pytest.approx(0.4)

    def test_zero_pairs(self):
        s = FrequencyMatchSummary()
        assert s.similar_ratio == pytest.approx(0.0)

    def test_invalid_total_negative(self):
        with pytest.raises(ValueError):
            FrequencyMatchSummary(total_pairs=-1)

    def test_invalid_similar_negative(self):
        with pytest.raises(ValueError):
            FrequencyMatchSummary(total_pairs=5, similar_pairs=-1)


class TestSummarizeFrequencyMatches:
    def test_empty(self):
        s = summarize_frequency_matches([])
        assert s.total_pairs == 0

    def test_non_empty(self):
        records = [FrequencyMatchRecord(i, i+1, similarity=float(i)*0.1) for i in range(5)]
        s = summarize_frequency_matches(records)
        assert s.total_pairs == 5
        assert s.mean_similarity == pytest.approx(sum(float(i)*0.1 for i in range(5)) / 5)

    def test_max_min(self):
        records = [FrequencyMatchRecord(0, 1, similarity=0.2),
                   FrequencyMatchRecord(1, 2, similarity=0.9)]
        s = summarize_frequency_matches(records)
        assert s.max_similarity == pytest.approx(0.9)
        assert s.min_similarity == pytest.approx(0.2)


class TestFilterFrequencyMatches:
    def test_filter(self):
        records = [FrequencyMatchRecord(0, 1, similarity=0.3),
                   FrequencyMatchRecord(1, 2, similarity=0.8)]
        filtered = filter_frequency_matches(records, min_similarity=0.5)
        assert len(filtered) == 1

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            filter_frequency_matches([], min_similarity=-0.1)


class TestTopFrequencyMatches:
    def test_top_k(self):
        records = [FrequencyMatchRecord(i, i+1, similarity=float(i)/10.0) for i in range(10)]
        top = top_frequency_matches(records, 3)
        assert len(top) == 3
        assert all(top[j].similarity >= top[j+1].similarity for j in range(2))

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            top_frequency_matches([], -1)


class TestPatchMatchRecord:
    def test_displacement(self):
        r = PatchMatchRecord(src_row=2, src_col=3, dst_row=5, dst_col=7, score=0.9)
        assert r.displacement == (3, 4)

    def test_negative_row_raises(self):
        with pytest.raises(ValueError):
            PatchMatchRecord(src_row=-1, src_col=0, dst_row=0, dst_col=0, score=0.5)


class TestSummarizeCanvasBuilds:
    def test_empty(self):
        s = summarize_canvas_builds([])
        assert s.n_canvases == 0

    def test_non_empty(self):
        records = [CanvasBuildRecord(n_fragments=5, coverage=0.8, canvas_w=100, canvas_h=100),
                   CanvasBuildRecord(n_fragments=3, coverage=0.6, canvas_w=100, canvas_h=100)]
        s = summarize_canvas_builds(records)
        assert s.n_canvases == 2
        assert s.total_fragments == 8
        assert s.mean_coverage == pytest.approx(0.7)

    def test_well_covered(self):
        records = [CanvasBuildRecord(n_fragments=5, coverage=0.9, canvas_w=100, canvas_h=100)]
        s = summarize_canvas_builds(records)
        assert s.well_covered_count == 1


class TestCanvasBuildRecord:
    def test_canvas_area(self):
        r = CanvasBuildRecord(n_fragments=5, coverage=0.8, canvas_w=100, canvas_h=200)
        assert r.canvas_area == 20000

    def test_is_well_covered(self):
        r = CanvasBuildRecord(n_fragments=5, coverage=0.8, canvas_w=100, canvas_h=100)
        assert r.is_well_covered is True

    def test_invalid_coverage(self):
        with pytest.raises(ValueError):
            CanvasBuildRecord(n_fragments=5, coverage=1.5, canvas_w=100, canvas_h=100)

    def test_invalid_canvas_w(self):
        with pytest.raises(ValueError):
            CanvasBuildRecord(n_fragments=5, coverage=0.5, canvas_w=0, canvas_h=100)


class TestSummarizePatchMatches:
    def test_empty(self):
        s = summarize_patch_matches([])
        assert s.n_pairs == 0

    def test_non_empty(self):
        batch = [
            [PatchMatchRecord(0, 0, 1, 1, 0.5), PatchMatchRecord(0, 1, 1, 2, 0.8)],
            [PatchMatchRecord(0, 2, 1, 3, 0.7)],
        ]
        s = summarize_patch_matches(batch)
        assert s.n_pairs == 2
        assert s.n_total_matches == 3
        assert s.mean_matches_per_pair == pytest.approx(1.5)


# ===========================================================================
# 6. image_transform_utils
# ===========================================================================

class TestImageTransformConfig:
    def test_defaults(self):
        cfg = ImageTransformConfig()
        assert cfg.border_value == 255
        assert cfg.expand is False

    def test_invalid_border_value(self):
        with pytest.raises(ValueError):
            ImageTransformConfig(border_value=300)

    def test_valid_border_zero(self):
        cfg = ImageTransformConfig(border_value=0)
        assert cfg.border_value == 0


class TestRotateImage:
    def test_shape_preserved_gray(self):
        img = gray_img(32, 32)
        out = rotate_image(img, math.pi / 4)
        assert out.shape == img.shape

    def test_shape_preserved_bgr(self):
        img = bgr_img(32, 48)
        out = rotate_image(img, math.pi / 6)
        assert out.shape == img.shape

    def test_zero_rotation_identical(self):
        img = gray_img(32, 32)
        out = rotate_image(img, 0.0)
        assert np.array_equal(out, img)

    def test_returns_uint8(self):
        img = gray_img(32, 32)
        out = rotate_image(img, 0.5)
        assert out.dtype == np.uint8


class TestFlipOperations:
    def test_flip_horizontal(self):
        img = gray_img(16, 16)
        out = flip_horizontal(img)
        assert out.shape == img.shape
        assert np.array_equal(out, np.fliplr(img))

    def test_flip_vertical(self):
        img = gray_img(16, 16)
        out = flip_vertical(img)
        assert out.shape == img.shape
        assert np.array_equal(out, np.flipud(img))

    def test_double_flip_horizontal(self):
        img = gray_img(16, 16)
        assert np.array_equal(flip_horizontal(flip_horizontal(img)), img)

    def test_double_flip_vertical(self):
        img = gray_img(16, 16)
        assert np.array_equal(flip_vertical(flip_vertical(img)), img)


class TestPadImage:
    def test_shape_increase(self):
        img = gray_img(16, 16)
        out = pad_image(img, top=2, bottom=3, left=4, right=5)
        assert out.shape == (16 + 5, 16 + 9)

    def test_fill_value(self):
        img = np.zeros((4, 4), dtype=np.uint8)
        out = pad_image(img, top=1, fill=200)
        assert out[0, 0] == 200

    def test_bgr_padding(self):
        img = bgr_img(16, 16)
        out = pad_image(img, top=2, bottom=2, left=2, right=2)
        assert out.shape == (20, 20, 3)


class TestCropImage:
    def test_basic_crop(self):
        img = gray_img(32, 32)
        out = crop_image(img, 4, 4, 20, 20)
        assert out.shape == (16, 16)

    def test_out_of_bounds_clipped(self):
        img = gray_img(32, 32)
        out = crop_image(img, -5, -5, 100, 100)
        assert out.shape == (32, 32)


class TestResizeImage:
    def test_resize_gray(self):
        img = gray_img(64, 64)
        out = resize_image(img, (32, 32))
        assert out.shape == (32, 32)

    def test_resize_bgr(self):
        img = bgr_img(64, 64)
        out = resize_image(img, (32, 48))
        assert out.shape == (48, 32, 3)


class TestResizeToMaxSide:
    def test_larger_side_clipped(self):
        img = gray_img(64, 32)
        out = resize_to_max_side(img, 32)
        assert max(out.shape[:2]) == 32

    def test_already_small(self):
        img = gray_img(16, 16)
        out = resize_to_max_side(img, 32)
        assert out.shape == img.shape


class TestApplyAffine:
    def test_identity_transform(self):
        img = gray_img(32, 32)
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        out = apply_affine(img, M)
        assert out.shape == img.shape

    def test_bgr_affine(self):
        img = bgr_img(32, 32)
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        out = apply_affine(img, M)
        assert out.shape == img.shape


class TestRotationMatrix2x3:
    def test_shape(self):
        M = rotation_matrix_2x3(math.pi / 4, 16.0, 16.0)
        assert M.shape == (2, 3)

    def test_identity_at_zero(self):
        M = rotation_matrix_2x3(0.0, 0.0, 0.0)
        assert M[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert M[1, 1] == pytest.approx(1.0, abs=1e-5)


class TestBatchOperations:
    def test_batch_rotate(self):
        imgs = [gray_img(16, 16) for _ in range(3)]
        outs = batch_rotate(imgs, math.pi / 6)
        assert len(outs) == 3
        assert all(o.shape == (16, 16) for o in outs)

    def test_batch_pad(self):
        imgs = [gray_img(16, 16) for _ in range(3)]
        outs = batch_pad(imgs, pad=2)
        assert all(o.shape == (20, 20) for o in outs)

    def test_batch_resize_to_max(self):
        imgs = [gray_img(64, 64), gray_img(48, 96)]
        outs = batch_resize_to_max(imgs, 32)
        for o in outs:
            assert max(o.shape[:2]) == 32

    def test_transform_result_to_dict(self):
        img = gray_img(16, 16)
        tr = TransformResult(image=img, angle_rad=0.5, scale=1.0, translation=(0.0, 0.0))
        d = tr.to_dict()
        assert "angle_deg" in d
        assert "scale" in d


# ===========================================================================
# 7. interpolation_utils
# ===========================================================================

class TestInterpolationConfig:
    def test_defaults(self):
        cfg = InterpolationConfig()
        assert cfg.method == "linear"
        assert cfg.clamp is True

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            InterpolationConfig(method="cubic")

    def test_nearest_method(self):
        cfg = InterpolationConfig(method="nearest")
        assert cfg.method == "nearest"


class TestLerp:
    def test_t_zero(self):
        assert lerp(0.0, 10.0, 0.0) == pytest.approx(0.0)

    def test_t_one(self):
        assert lerp(0.0, 10.0, 1.0) == pytest.approx(10.0)

    def test_t_half(self):
        assert lerp(0.0, 10.0, 0.5) == pytest.approx(5.0)

    def test_t_negative_raises(self):
        with pytest.raises(ValueError):
            lerp(0.0, 1.0, -0.1)

    def test_t_greater_one_raises(self):
        with pytest.raises(ValueError):
            lerp(0.0, 1.0, 1.1)

    def test_negative_values(self):
        assert lerp(-5.0, 5.0, 0.5) == pytest.approx(0.0)


class TestLerpArray:
    def test_t_zero(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = lerp_array(a, b, 0.0)
        np.testing.assert_allclose(result, a)

    def test_t_one(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = lerp_array(a, b, 1.0)
        np.testing.assert_allclose(result, b)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            lerp_array(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), 0.5)

    def test_t_invalid_raises(self):
        a = b = np.zeros(3)
        with pytest.raises(ValueError):
            lerp_array(a, b, 1.5)


class TestBilinearInterpolate:
    def test_exact_corner(self):
        grid = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert bilinear_interpolate(grid, 0.0, 0.0) == pytest.approx(1.0)
        assert bilinear_interpolate(grid, 1.0, 1.0) == pytest.approx(4.0)

    def test_midpoint(self):
        grid = np.array([[0.0, 0.0], [0.0, 4.0]])
        val = bilinear_interpolate(grid, 0.5, 0.5)
        assert val == pytest.approx(1.0)

    def test_out_of_range_raises(self):
        grid = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError):
            bilinear_interpolate(grid, 5.0, 0.0)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            bilinear_interpolate(np.zeros((2, 2, 2)), 0.0, 0.0)


class TestResample1d:
    def test_same_length_returns_copy(self):
        arr = np.array([1.0, 2.0, 3.0])
        out = resample_1d(arr, 3)
        np.testing.assert_allclose(out, arr)

    def test_upsample(self):
        arr = np.array([0.0, 10.0])
        out = resample_1d(arr, 5)
        assert len(out) == 5

    def test_downsample(self):
        arr = np.linspace(0, 10, 100)
        out = resample_1d(arr, 10)
        assert len(out) == 10

    def test_nearest_method(self):
        arr = np.array([0.0, 1.0, 2.0])
        cfg = InterpolationConfig(method="nearest")
        out = resample_1d(arr, 5, cfg)
        assert len(out) == 5

    def test_invalid_target_len(self):
        with pytest.raises(ValueError):
            resample_1d(np.array([1.0, 2.0]), 0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            resample_1d(np.array([]), 5)


class TestFillMissing:
    def test_no_nans(self):
        arr = np.array([1.0, 2.0, 3.0])
        out = fill_missing(arr)
        np.testing.assert_allclose(out, arr)

    def test_interior_nans_filled(self):
        arr = np.array([1.0, np.nan, 3.0])
        out = fill_missing(arr)
        assert not np.any(np.isnan(out))
        assert out[1] == pytest.approx(2.0)

    def test_all_nans_returns_zeros(self):
        arr = np.array([np.nan, np.nan, np.nan])
        out = fill_missing(arr)
        np.testing.assert_allclose(out, [0.0, 0.0, 0.0])

    def test_edge_nans_filled(self):
        arr = np.array([np.nan, 2.0, np.nan])
        out = fill_missing(arr)
        assert not np.any(np.isnan(out))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            fill_missing(np.array([]))


class TestInterpolateScores:
    def test_symmetric_result(self):
        m = np.array([[0.0, 1.0], [2.0, 0.0]])
        out = interpolate_scores(m, alpha=0.5)
        np.testing.assert_allclose(out, out.T)

    def test_alpha_one_returns_original(self):
        m = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = interpolate_scores(m, alpha=1.0)
        np.testing.assert_allclose(out, m)

    def test_alpha_zero_returns_transpose(self):
        m = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = interpolate_scores(m, alpha=0.0)
        np.testing.assert_allclose(out, m.T)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            interpolate_scores(np.zeros((2, 3)))

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            interpolate_scores(np.eye(3), alpha=1.5)


class TestSmoothInterpolate:
    def test_length_preserved(self):
        arr = np.linspace(0, 10, 20)
        out = smooth_interpolate(arr, window=5)
        assert len(out) == 20

    def test_window_1_identity(self):
        arr = np.array([1.0, 3.0, 5.0, 2.0])
        out = smooth_interpolate(arr, window=1)
        np.testing.assert_allclose(out, arr)

    def test_smoothing_effect(self):
        arr = np.array([0.0, 10.0, 0.0, 10.0, 0.0])
        out = smooth_interpolate(arr, window=3)
        assert float(np.var(out)) < float(np.var(arr))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            smooth_interpolate(np.array([]))

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            smooth_interpolate(np.array([1.0, 2.0]), window=0)


class TestBatchResample:
    def test_output_length(self):
        arrays = [np.linspace(0, 1, 10), np.linspace(0, 1, 20)]
        out = batch_resample(arrays, 15)
        assert len(out) == 2
        assert all(len(a) == 15 for a in out)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            batch_resample([], 10)

    def test_invalid_target_len_raises(self):
        with pytest.raises(ValueError):
            batch_resample([np.array([1.0, 2.0])], 0)


# ===========================================================================
# 8. mask_layout_utils
# ===========================================================================

class TestMaskOpRecord:
    def test_valid_construction(self):
        r = MaskOpRecord("erode", (32, 32), 500, 400)
        assert r.area_change == -100

    def test_coverage_ratio(self):
        r = MaskOpRecord("dilate", (10, 10), 50, 60)
        assert r.coverage_ratio == pytest.approx(0.6)

    def test_invalid_operation(self):
        with pytest.raises(ValueError):
            MaskOpRecord("unknown_op", (10, 10), 50, 60)

    def test_negative_before_raises(self):
        with pytest.raises(ValueError):
            MaskOpRecord("erode", (10, 10), -1, 50)

    def test_all_valid_ops(self):
        for op in ["erode", "dilate", "invert", "and", "or", "xor", "crop"]:
            r = MaskOpRecord(op, (10, 10), 50, 40)
            assert r.operation == op


class TestMaskCoverageRecord:
    def test_coverage_ratio(self):
        r = MaskCoverageRecord(n_masks=2, canvas_shape=(10, 10),
                               n_covered_pixels=50, n_total_pixels=100)
        assert r.coverage_ratio == pytest.approx(0.5)

    def test_fully_covered(self):
        r = MaskCoverageRecord(n_masks=1, canvas_shape=(10, 10),
                               n_covered_pixels=100, n_total_pixels=100)
        assert r.is_fully_covered is True

    def test_not_fully_covered(self):
        r = MaskCoverageRecord(n_masks=1, canvas_shape=(10, 10),
                               n_covered_pixels=80, n_total_pixels=100)
        assert r.is_fully_covered is False

    def test_zero_total(self):
        r = MaskCoverageRecord(n_masks=0, canvas_shape=(0, 0),
                               n_covered_pixels=0, n_total_pixels=0)
        assert r.coverage_ratio == pytest.approx(0.0)


class TestFragmentPlacementRecord:
    def test_coverage(self):
        r = FragmentPlacementRecord(n_total=10, n_placed=7)
        assert r.coverage == pytest.approx(0.7)
        assert r.n_missing == 3

    def test_all_placed_coverage_one(self):
        r = FragmentPlacementRecord(n_total=5, n_placed=5)
        assert r.coverage == pytest.approx(1.0)

    def test_zero_total(self):
        r = FragmentPlacementRecord(n_total=0, n_placed=0)
        assert r.coverage == pytest.approx(1.0)

    def test_placed_exceeds_total_raises(self):
        with pytest.raises(ValueError):
            FragmentPlacementRecord(n_total=5, n_placed=6)

    def test_missing_ids(self):
        r = FragmentPlacementRecord(n_total=3, n_placed=1, missing_ids=[1, 2])
        assert len(r.missing_ids) == 2


class TestLayoutDiffRecord:
    def test_is_stable(self):
        r = LayoutDiffRecord(n_fragments=5, mean_shift=0.0, max_shift=0.0, n_moved=0)
        assert r.is_stable is True

    def test_not_stable(self):
        r = LayoutDiffRecord(n_fragments=5, mean_shift=1.0, max_shift=3.0, n_moved=2)
        assert r.is_stable is False

    def test_negative_mean_shift_raises(self):
        with pytest.raises(ValueError):
            LayoutDiffRecord(n_fragments=5, mean_shift=-1.0, max_shift=0.0, n_moved=0)


class TestLayoutScoreRecord:
    def test_score_improvement(self):
        r = LayoutScoreRecord(n_pairs=5, initial_score=0.3, final_score=0.7)
        assert r.score_improvement == pytest.approx(0.4)

    def test_converged(self):
        r = LayoutScoreRecord(n_pairs=5, initial_score=0.5, final_score=0.5)
        assert r.converged is True

    def test_not_converged(self):
        r = LayoutScoreRecord(n_pairs=5, initial_score=0.3, final_score=0.9)
        assert r.converged is False


class TestFeatureSelectionRecord:
    def test_selection_ratio(self):
        r = FeatureSelectionRecord(method="variance", n_input_features=100, n_selected_features=50)
        assert r.selection_ratio == pytest.approx(0.5)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            FeatureSelectionRecord(method="random", n_input_features=100, n_selected_features=50)

    def test_selected_exceeds_input_raises(self):
        with pytest.raises(ValueError):
            FeatureSelectionRecord(method="pca", n_input_features=10, n_selected_features=20)

    def test_all_valid_methods(self):
        for m in ["variance", "correlation", "rank", "pca"]:
            r = FeatureSelectionRecord(method=m, n_input_features=20, n_selected_features=10)
            assert r.method == m


class TestPcaRecord:
    def test_total_variance(self):
        r = PcaRecord(n_input_features=10, n_components=3,
                      explained_variance_ratio=[0.5, 0.3, 0.1])
        assert r.total_variance_explained == pytest.approx(0.9)

    def test_dominant_component(self):
        r = PcaRecord(n_input_features=10, n_components=3,
                      explained_variance_ratio=[0.5, 0.3, 0.1])
        assert r.dominant_component_ratio == pytest.approx(0.5)

    def test_empty_ratios(self):
        r = PcaRecord(n_input_features=10, n_components=0)
        assert r.total_variance_explained == pytest.approx(0.0)
        assert r.dominant_component_ratio == pytest.approx(0.0)


class TestMakeMaskCoverageRecord:
    def test_single_mask(self):
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[4:12, 4:12] = 255
        r = make_mask_coverage_record([mask], (16, 16))
        assert r.n_masks == 1
        assert r.n_covered_pixels == 64
        assert r.n_total_pixels == 256

    def test_multiple_masks(self):
        m1 = np.zeros((16, 16), dtype=np.uint8)
        m1[:8, :8] = 255
        m2 = np.zeros((16, 16), dtype=np.uint8)
        m2[8:, 8:] = 255
        r = make_mask_coverage_record([m1, m2], (16, 16))
        assert r.n_masks == 2
        assert r.n_covered_pixels == 128


class TestMakeLayoutDiffRecord:
    def test_from_dict(self):
        d = {"n_fragments": 5, "mean_shift": 1.5, "max_shift": 3.0, "n_moved": 2}
        r = make_layout_diff_record(d, label="test")
        assert r.n_fragments == 5
        assert r.mean_shift == pytest.approx(1.5)
        assert r.label == "test"

    def test_empty_dict_defaults(self):
        r = make_layout_diff_record({})
        assert r.n_fragments == 0
        assert r.mean_shift == pytest.approx(0.0)


# ===========================================================================
# 9. match_rank_utils
# ===========================================================================

def _make_ranking_entries(n=5):
    rng = np.random.default_rng(17)
    entries = []
    for i in range(n):
        entries.append(make_ranking_entry(
            batch_id=i,
            n_pairs=int(rng.integers(5, 20)),
            n_accepted=int(rng.integers(0, 10)),
            top_score=float(rng.uniform(0.3, 1.0)),
            mean_score=float(rng.uniform(0.1, 0.8)),
            algorithm="ncc",
        ))
    return entries


class TestRankingEntry:
    def test_acceptance_rate(self):
        e = make_ranking_entry(0, 10, 5, 0.9, 0.5, "ncc")
        assert e.acceptance_rate == pytest.approx(0.5)

    def test_zero_pairs_acceptance_rate(self):
        e = make_ranking_entry(0, 0, 0, 0.0, 0.0, "ncc")
        assert e.acceptance_rate == pytest.approx(0.0)

    def test_params_stored(self):
        e = make_ranking_entry(0, 10, 5, 0.9, 0.5, "ncc", window=5)
        assert e.params["window"] == 5


class TestSummariseRankingEntries:
    def test_empty(self):
        s = summarise_ranking_entries([])
        assert s.n_batches == 0
        assert s.best_batch_id is None

    def test_summary_fields(self):
        entries = _make_ranking_entries(5)
        s = summarise_ranking_entries(entries)
        assert s.n_batches == 5
        assert s.total_pairs == sum(e.n_pairs for e in entries)

    def test_best_worst_batch(self):
        entries = _make_ranking_entries(5)
        s = summarise_ranking_entries(entries)
        top_scores = [e.top_score for e in entries]
        assert entries[s.best_batch_id].top_score == max(top_scores)
        assert entries[s.worst_batch_id].top_score == min(top_scores)


class TestFilterRankingEntries:
    def test_filter_by_algorithm(self):
        entries = _make_ranking_entries(5)
        entries[0] = make_ranking_entry(0, 10, 5, 0.9, 0.5, "ssd")
        filtered = filter_ranking_by_algorithm(entries, "ncc")
        assert all(e.algorithm == "ncc" for e in filtered)

    def test_filter_by_min_top_score(self):
        entries = _make_ranking_entries(10)
        filtered = filter_ranking_by_min_top_score(entries, 0.7)
        assert all(e.top_score >= 0.7 for e in filtered)

    def test_filter_by_min_acceptance(self):
        entries = _make_ranking_entries(10)
        filtered = filter_ranking_by_min_acceptance(entries, 0.3)
        assert all(e.acceptance_rate >= 0.3 for e in filtered)

    def test_top_k_entries(self):
        entries = _make_ranking_entries(10)
        top = top_k_ranking_entries(entries, 3)
        assert len(top) == 3

    def test_best_ranking_entry(self):
        entries = _make_ranking_entries(5)
        best = best_ranking_entry(entries)
        assert best.top_score == max(e.top_score for e in entries)

    def test_best_empty(self):
        assert best_ranking_entry([]) is None

    def test_score_stats(self):
        entries = _make_ranking_entries(5)
        stats = ranking_score_stats(entries)
        assert stats["count"] == 5
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_score_stats_empty(self):
        stats = ranking_score_stats([])
        assert stats["count"] == 0

    def test_compare_summaries(self):
        s1 = summarise_ranking_entries(_make_ranking_entries(3))
        s2 = summarise_ranking_entries(_make_ranking_entries(5))
        cmp = compare_ranking_summaries(s1, s2)
        assert "delta_mean_top_score" in cmp
        assert "same_best" in cmp

    def test_batch_summarise(self):
        groups = [_make_ranking_entries(3), _make_ranking_entries(4)]
        summaries = batch_summarise_ranking_entries(groups)
        assert len(summaries) == 2


def _make_eval_entries(n=5):
    rng = np.random.default_rng(19)
    entries = []
    for i in range(n):
        entries.append(make_eval_result_entry(
            run_id=i,
            n_pairs=int(rng.integers(5, 20)),
            mean_score=float(rng.uniform(0.3, 0.9)),
            mean_f1=float(rng.uniform(0.2, 0.8)),
            best_f1=float(rng.uniform(0.5, 1.0)),
            algorithm="exact",
        ))
    return entries


class TestEvalResultEntry:
    def test_construction(self):
        e = make_eval_result_entry(0, 10, 0.8, 0.7, 0.9, "exact")
        assert e.run_id == 0
        assert e.mean_f1 == pytest.approx(0.7)


class TestSummariseEvalEntries:
    def test_empty(self):
        s = summarise_eval_result_entries([])
        assert s.n_runs == 0
        assert s.best_run_id is None

    def test_summary_fields(self):
        entries = _make_eval_entries(5)
        s = summarise_eval_result_entries(entries)
        assert s.n_runs == 5
        assert s.total_pairs == sum(e.n_pairs for e in entries)

    def test_best_worst(self):
        entries = _make_eval_entries(5)
        s = summarise_eval_result_entries(entries)
        best_f1s = [e.best_f1 for e in entries]
        assert entries[s.best_run_id].best_f1 == max(best_f1s)


class TestFilterEvalEntries:
    def test_filter_by_min_f1(self):
        entries = _make_eval_entries(10)
        filtered = filter_eval_by_min_f1(entries, 0.5)
        assert all(e.mean_f1 >= 0.5 for e in filtered)

    def test_filter_by_algorithm(self):
        entries = _make_eval_entries(5)
        entries[0] = make_eval_result_entry(0, 10, 0.8, 0.7, 0.9, "approx")
        filtered = filter_eval_by_algorithm(entries, "exact")
        assert all(e.algorithm == "exact" for e in filtered)

    def test_top_k_eval(self):
        entries = _make_eval_entries(10)
        top = top_k_eval_entries(entries, 3)
        assert len(top) == 3

    def test_best_eval_entry(self):
        entries = _make_eval_entries(5)
        best = best_eval_entry(entries)
        assert best.best_f1 == max(e.best_f1 for e in entries)

    def test_best_eval_empty(self):
        assert best_eval_entry([]) is None

    def test_eval_f1_stats(self):
        entries = _make_eval_entries(5)
        stats = eval_f1_stats(entries)
        assert stats["count"] == 5

    def test_compare_eval_summaries(self):
        s1 = summarise_eval_result_entries(_make_eval_entries(3))
        s2 = summarise_eval_result_entries(_make_eval_entries(5))
        cmp = compare_eval_summaries(s1, s2)
        assert "delta_mean_f1" in cmp
        assert "same_best" in cmp

    def test_batch_summarise_eval(self):
        groups = [_make_eval_entries(3), _make_eval_entries(4)]
        summaries = batch_summarise_eval_entries(groups)
        assert len(summaries) == 2


# ===========================================================================
# 10. morph_utils
# ===========================================================================

class TestMorphConfig:
    def test_defaults(self):
        cfg = MorphConfig()
        assert cfg.kernel_size == 3
        assert cfg.kernel_shape == "rect"
        assert cfg.iterations == 1

    def test_invalid_kernel_size_even(self):
        with pytest.raises(ValueError):
            MorphConfig(kernel_size=4)

    def test_invalid_kernel_shape(self):
        with pytest.raises(ValueError):
            MorphConfig(kernel_shape="unknown")

    def test_invalid_iterations(self):
        with pytest.raises(ValueError):
            MorphConfig(iterations=0)

    def test_build_kernel_rect(self):
        cfg = MorphConfig(kernel_size=3, kernel_shape="rect")
        k = cfg.build_kernel()
        assert k.shape == (3, 3)

    def test_build_kernel_ellipse(self):
        cfg = MorphConfig(kernel_size=5, kernel_shape="ellipse")
        k = cfg.build_kernel()
        assert k.shape == (5, 5)

    def test_build_kernel_cross(self):
        cfg = MorphConfig(kernel_size=3, kernel_shape="cross")
        k = cfg.build_kernel()
        assert k.shape == (3, 3)


class TestApplyErosionDilation:
    def test_erosion_shape(self):
        img = binary_img(32, 32)
        out = apply_erosion(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_dilation_shape(self):
        img = binary_img(32, 32)
        out = apply_dilation(img)
        assert out.shape == img.shape

    def test_erosion_reduces_area(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[8:24, 8:24] = 255
        out = apply_erosion(img)
        assert int(out.sum()) <= int(img.sum())

    def test_dilation_increases_area(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[8:24, 8:24] = 255
        out = apply_dilation(img)
        assert int(out.sum()) >= int(img.sum())


class TestApplyOpeningClosing:
    def test_opening_shape(self):
        img = binary_img(32, 32)
        out = apply_opening(img)
        assert out.shape == img.shape

    def test_closing_shape(self):
        img = binary_img(32, 32)
        out = apply_closing(img)
        assert out.shape == img.shape

    def test_opening_removes_small_objects(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[1, 1] = 255
        img[10:22, 10:22] = 255
        out = apply_opening(img, MorphConfig(kernel_size=5))
        assert out[1, 1] == 0

    def test_closing_fills_small_holes(self):
        img = np.full((32, 32), 255, dtype=np.uint8)
        img[15, 15] = 0
        out = apply_closing(img, MorphConfig(kernel_size=5))
        assert out[15, 15] == 255


class TestGetSkeleton:
    def test_skeleton_shape(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[10:22, 14:18] = 255
        skel = get_skeleton(img)
        assert skel.shape == (32, 32)

    def test_skeleton_binary(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[10:22, 14:18] = 255
        skel = get_skeleton(img)
        unique = set(np.unique(skel))
        assert unique.issubset({0, 255})

    def test_empty_image_skeleton(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        skel = get_skeleton(img)
        assert skel.sum() == 0


class TestLabelRegions:
    def test_no_components(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        n, label_map = label_regions(img)
        assert n == 0

    def test_one_component(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[5:20, 5:20] = 255
        n, label_map = label_regions(img)
        assert n == 1

    def test_two_components(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[0:8, 0:8] = 255
        img[24:32, 24:32] = 255
        n, label_map = label_regions(img)
        assert n == 2

    def test_invalid_connectivity(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        with pytest.raises(ValueError):
            label_regions(img, connectivity=6)


class TestFilterRegionsBySize:
    def test_removes_small_regions(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[0, 0] = 255
        img[10:22, 10:22] = 255
        out = filter_regions_by_size(img, min_area=10)
        assert out[0, 0] == 0

    def test_negative_min_area_raises(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        with pytest.raises(ValueError):
            filter_regions_by_size(img, min_area=-1)


class TestComputeRegionStats:
    def test_returns_list(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[5:20, 5:20] = 255
        stats = compute_region_stats(img)
        assert len(stats) == 1
        assert "area" in stats[0]
        assert "cx" in stats[0]

    def test_empty_image(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        stats = compute_region_stats(img)
        assert stats == []

    def test_aspect_ratio(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[10:20, 10:30] = 255
        stats = compute_region_stats(img)
        assert len(stats) == 1
        assert stats[0]["aspect_ratio"] > 1.0


class TestBatchMorphology:
    def test_valid_operations(self):
        imgs = [binary_img(16, 16) for _ in range(3)]
        for op in ["erosion", "dilation", "opening", "closing"]:
            out = batch_morphology(imgs, op)
            assert len(out) == 3

    def test_invalid_operation(self):
        imgs = [binary_img(16, 16)]
        with pytest.raises(ValueError):
            batch_morphology(imgs, "unknown")


# ===========================================================================
# 11. noise_stats_utils
# ===========================================================================

class TestNoiseStatsConfig:
    def test_defaults(self):
        cfg = NoiseStatsConfig()
        assert cfg.max_sigma == 50.0
        assert cfg.snr_threshold == 20.0

    def test_invalid_max_sigma(self):
        with pytest.raises(ValueError):
            NoiseStatsConfig(max_sigma=0.0)

    def test_invalid_snr_threshold(self):
        with pytest.raises(ValueError):
            NoiseStatsConfig(snr_threshold=-1.0)

    def test_invalid_quality_levels(self):
        with pytest.raises(ValueError):
            NoiseStatsConfig(quality_levels=0)


class TestNoiseStatsEntry:
    def test_valid_construction(self):
        e = make_noise_entry(0, 5.0, 30.0, 0.1, 0.2, "clean")
        assert e.is_clean is True
        assert e.is_noisy is False

    def test_noisy_quality(self):
        e = make_noise_entry(1, 15.0, 20.0, 0.3, 0.4, "noisy")
        assert e.is_clean is False
        assert e.is_noisy is True

    def test_very_noisy(self):
        e = make_noise_entry(2, 30.0, 10.0, 0.5, 0.6, "very_noisy")
        assert e.is_noisy is True

    def test_invalid_image_id(self):
        with pytest.raises(ValueError):
            make_noise_entry(-1, 5.0, 30.0, 0.1, 0.2, "clean")

    def test_invalid_sigma(self):
        with pytest.raises(ValueError):
            make_noise_entry(0, -1.0, 30.0, 0.1, 0.2, "clean")

    def test_invalid_jpeg_level(self):
        with pytest.raises(ValueError):
            make_noise_entry(0, 5.0, 30.0, 1.5, 0.2, "clean")

    def test_invalid_quality(self):
        with pytest.raises(ValueError):
            make_noise_entry(0, 5.0, 30.0, 0.1, 0.2, "great")

    def test_meta_stored(self):
        e = make_noise_entry(0, 5.0, 30.0, 0.1, 0.2, "clean", meta={"src": "cam"})
        assert e.meta["src"] == "cam"


def _make_noise_entries():
    return [
        make_noise_entry(0, 2.0, 40.0, 0.05, 0.1, "clean"),
        make_noise_entry(1, 10.0, 25.0, 0.2, 0.3, "noisy"),
        make_noise_entry(2, 25.0, 15.0, 0.4, 0.5, "very_noisy"),
        make_noise_entry(3, 5.0, 35.0, 0.1, 0.2, "clean"),
    ]


class TestSummariseNoiseStats:
    def test_empty(self):
        s = summarise_noise_stats([])
        assert s.n_total == 0

    def test_summary_counts(self):
        entries = _make_noise_entries()
        s = summarise_noise_stats(entries)
        assert s.n_total == 4
        assert s.n_clean == 2
        assert s.n_noisy == 2

    def test_mean_sigma(self):
        entries = _make_noise_entries()
        s = summarise_noise_stats(entries)
        expected_mean = sum([2.0, 10.0, 25.0, 5.0]) / 4
        assert s.mean_sigma == pytest.approx(expected_mean)

    def test_max_min_sigma(self):
        entries = _make_noise_entries()
        s = summarise_noise_stats(entries)
        assert s.max_sigma == pytest.approx(25.0)
        assert s.min_sigma == pytest.approx(2.0)


class TestFilterNoiseEntries:
    def test_filter_clean(self):
        entries = _make_noise_entries()
        clean = filter_clean_entries(entries)
        assert len(clean) == 2
        assert all(e.is_clean for e in clean)

    def test_filter_noisy(self):
        entries = _make_noise_entries()
        noisy = filter_noisy_entries(entries)
        assert len(noisy) == 2
        assert all(e.is_noisy for e in noisy)

    def test_filter_by_sigma_range(self):
        entries = _make_noise_entries()
        filtered = filter_by_sigma_range(entries, lo=3.0, hi=20.0)
        assert all(3.0 <= e.sigma <= 20.0 for e in filtered)

    def test_filter_by_snr_range(self):
        entries = _make_noise_entries()
        filtered = filter_by_snr_range(entries, lo=20.0, hi=50.0)
        assert all(20.0 <= e.snr_db <= 50.0 for e in filtered)

    def test_filter_by_jpeg(self):
        entries = _make_noise_entries()
        filtered = filter_by_jpeg_threshold(entries, max_jpeg=0.15)
        assert all(e.jpeg_level <= 0.15 for e in filtered)

    def test_top_k_cleanest(self):
        entries = _make_noise_entries()
        top = top_k_cleanest(entries, 2)
        assert len(top) == 2
        assert top[0].sigma <= top[1].sigma

    def test_top_k_noisiest(self):
        entries = _make_noise_entries()
        top = top_k_noisiest(entries, 2)
        assert len(top) == 2
        assert top[0].sigma >= top[1].sigma

    def test_top_k_zero(self):
        entries = _make_noise_entries()
        assert top_k_cleanest(entries, 0) == []
        assert top_k_noisiest(entries, 0) == []

    def test_best_snr_entry(self):
        entries = _make_noise_entries()
        best = best_snr_entry(entries)
        assert best is not None
        assert best.snr_db == max(e.snr_db for e in entries)

    def test_best_snr_empty(self):
        assert best_snr_entry([]) is None


class TestNoiseStatsDict:
    def test_empty(self):
        d = noise_stats_dict([])
        assert d["n"] == 0

    def test_non_empty(self):
        entries = _make_noise_entries()
        d = noise_stats_dict(entries)
        assert d["n"] == 4
        assert d["min"] <= d["mean"] <= d["max"]


class TestCompareNoiseSummaries:
    def test_compare(self):
        s1 = summarise_noise_stats(_make_noise_entries()[:2])
        s2 = summarise_noise_stats(_make_noise_entries()[2:])
        cmp = compare_noise_summaries(s1, s2)
        assert "delta_mean_sigma" in cmp
        assert "a_cleaner" in cmp


class TestEntriesFromAnalysisResults:
    def test_from_objects(self):
        class FakeResult:
            noise_level = 5.0
            snr_db = 30.0
            jpeg_artifacts = 0.1
            grain_level = 0.2
            quality = "clean"
        entries = entries_from_analysis_results([FakeResult(), FakeResult()])
        assert len(entries) == 2
        assert entries[0].sigma == pytest.approx(5.0)


class TestBatchSummariseNoiseStats:
    def test_batch(self):
        groups = [_make_noise_entries()[:2], _make_noise_entries()[2:]]
        summaries = batch_summarise_noise_stats(groups)
        assert len(summaries) == 2


# ===========================================================================
# 12. normalization_utils
# ===========================================================================

class TestL1Normalize:
    def test_sum_to_one(self):
        arr = np.array([1.0, 2.0, 3.0])
        out = l1_normalize(arr)
        assert np.sum(np.abs(out)) == pytest.approx(1.0)

    def test_zero_vector(self):
        arr = np.zeros(5)
        out = l1_normalize(arr)
        np.testing.assert_allclose(out, np.zeros(5))

    def test_invalid_ndim(self):
        with pytest.raises(ValueError):
            l1_normalize(np.zeros((3, 3)))

    def test_negative_values(self):
        arr = np.array([-1.0, 2.0, -3.0])
        out = l1_normalize(arr)
        assert np.sum(np.abs(out)) == pytest.approx(1.0)


class TestL2Normalize:
    def test_unit_norm(self):
        arr = np.array([3.0, 4.0])
        out = l2_normalize(arr)
        assert np.linalg.norm(out) == pytest.approx(1.0)

    def test_zero_vector(self):
        arr = np.zeros(5)
        out = l2_normalize(arr)
        np.testing.assert_allclose(out, np.zeros(5))

    def test_invalid_ndim(self):
        with pytest.raises(ValueError):
            l2_normalize(np.zeros((3, 3)))

    def test_large_values(self):
        arr = np.array([1000.0, 2000.0, 3000.0])
        out = l2_normalize(arr)
        assert np.linalg.norm(out) == pytest.approx(1.0)


class TestMinMaxNormalize:
    def test_range_zero_one(self):
        arr = np.array([1.0, 5.0, 3.0])
        out = minmax_normalize(arr)
        assert out.min() == pytest.approx(0.0)
        assert out.max() == pytest.approx(1.0)

    def test_uniform_returns_zeros(self):
        arr = np.array([5.0, 5.0, 5.0])
        out = minmax_normalize(arr)
        np.testing.assert_allclose(out, np.zeros(3))

    def test_invalid_ndim(self):
        with pytest.raises(ValueError):
            minmax_normalize(np.zeros((3, 3)))


class TestZscoreNormalize:
    def test_mean_zero_std_one(self):
        rng = np.random.default_rng(42)
        arr = rng.normal(10.0, 5.0, 100)
        out = zscore_normalize(arr)
        assert np.mean(out) == pytest.approx(0.0, abs=1e-10)
        assert np.std(out) == pytest.approx(1.0, abs=1e-10)

    def test_constant_returns_zeros(self):
        arr = np.array([3.0, 3.0, 3.0])
        out = zscore_normalize(arr)
        np.testing.assert_allclose(out, np.zeros(3))

    def test_invalid_ndim(self):
        with pytest.raises(ValueError):
            zscore_normalize(np.zeros((3, 3)))


class TestSoftmax:
    def test_sums_to_one(self):
        arr = np.array([1.0, 2.0, 3.0])
        out = softmax(arr)
        assert float(out.sum()) == pytest.approx(1.0)

    def test_all_non_negative(self):
        arr = np.array([-5.0, 0.0, 5.0])
        out = softmax(arr)
        assert np.all(out >= 0.0)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError):
            softmax(np.array([1.0, 2.0]), temperature=0.0)

    def test_invalid_ndim(self):
        with pytest.raises(ValueError):
            softmax(np.zeros((3, 3)))

    def test_high_temperature_more_uniform(self):
        arr = np.array([1.0, 2.0, 10.0])
        out_high = softmax(arr, temperature=100.0)
        out_low = softmax(arr, temperature=0.01)
        # High temperature: more uniform (lower max value)
        assert out_high.max() < out_low.max()


class TestClamp:
    def test_clamps_values(self):
        arr = np.array([-5.0, 0.5, 10.0])
        out = clamp(arr, 0.0, 1.0)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_invalid_range(self):
        with pytest.raises(ValueError):
            clamp(np.array([1.0, 2.0]), 5.0, 3.0)

    def test_2d_array(self):
        arr = np.array([[-1.0, 2.0], [0.5, 3.0]])
        out = clamp(arr, 0.0, 1.0)
        assert out.shape == (2, 2)


class TestSymmetrizeMatrix:
    def test_symmetric_result(self):
        m = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = symmetrize_matrix(m)
        np.testing.assert_allclose(out, out.T)

    def test_already_symmetric(self):
        m = np.array([[1.0, 2.0], [2.0, 3.0]])
        out = symmetrize_matrix(m)
        np.testing.assert_allclose(out, m)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            symmetrize_matrix(np.zeros((2, 3)))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            symmetrize_matrix(np.zeros(4))


class TestZeroDiagonal:
    def test_diagonal_zeroed(self):
        m = np.eye(4)
        out = zero_diagonal(m)
        np.testing.assert_allclose(np.diag(out), np.zeros(4))

    def test_off_diagonal_unchanged(self):
        m = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = zero_diagonal(m)
        assert out[0, 1] == pytest.approx(2.0)
        assert out[1, 0] == pytest.approx(3.0)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            zero_diagonal(np.zeros(4))


class TestNormalizeRows:
    def test_l2_rows(self):
        m = np.array([[3.0, 4.0], [5.0, 12.0]])
        out = normalize_rows(m, method="l2")
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0])

    def test_l1_rows(self):
        m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = normalize_rows(m, method="l1")
        np.testing.assert_allclose(np.sum(np.abs(out), axis=1), [1.0, 1.0])

    def test_minmax_rows(self):
        m = np.array([[1.0, 5.0, 3.0], [2.0, 8.0, 4.0]])
        out = normalize_rows(m, method="minmax")
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-9

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            normalize_rows(np.zeros((3, 3)), method="zscore")

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            normalize_rows(np.zeros(4))


class TestBatchL2Normalize:
    def test_all_unit_norm(self):
        vectors = [np.array([3.0, 4.0]), np.array([5.0, 12.0])]
        out = batch_l2_normalize(vectors)
        for v in out:
            assert np.linalg.norm(v) == pytest.approx(1.0)

    def test_output_length(self):
        vectors = [np.array([1.0, 2.0, 3.0]) for _ in range(5)]
        out = batch_l2_normalize(vectors)
        assert len(out) == 5


# ===========================================================================
# 13. normalize_noise_utils
# ===========================================================================

def _make_norm_entries(n=5):
    rng = np.random.default_rng(23)
    entries = []
    for i in range(n):
        min_v = float(rng.uniform(0.0, 0.5))
        max_v = float(rng.uniform(0.6, 1.0))
        entries.append(make_norm_result_entry(
            run_id=i,
            method="minmax",
            min_val=min_v,
            max_val=max_v,
            n_rows=10,
            n_cols=10,
        ))
    return entries


class TestNormResultEntry:
    def test_spread_computed(self):
        e = make_norm_result_entry(0, "minmax", 0.1, 0.9, 10, 10)
        assert e.spread == pytest.approx(0.8)

    def test_fields(self):
        e = make_norm_result_entry(1, "zscore", -1.0, 1.0, 5, 5)
        assert e.method == "zscore"
        assert e.n_rows == 5

    def test_params_stored(self):
        e = make_norm_result_entry(0, "minmax", 0.0, 1.0, 5, 5, eps=1e-6)
        assert e.params["eps"] == pytest.approx(1e-6)


class TestSummariseNormResultEntries:
    def test_empty(self):
        s = summarise_norm_result_entries([])
        assert s.n_runs == 0
        assert s.best_run_id is None

    def test_summary_fields(self):
        entries = _make_norm_entries(5)
        s = summarise_norm_result_entries(entries)
        assert s.n_runs == 5
        assert isinstance(s.mean_spread, float)

    def test_best_worst(self):
        entries = _make_norm_entries(5)
        s = summarise_norm_result_entries(entries)
        spreads = [e.spread for e in entries]
        assert entries[s.best_run_id].spread == max(spreads)
        assert entries[s.worst_run_id].spread == min(spreads)

    def test_method_counts(self):
        entries = _make_norm_entries(3)
        entries.append(make_norm_result_entry(99, "zscore", 0.0, 1.0, 5, 5))
        s = summarise_norm_result_entries(entries)
        assert s.method_counts["minmax"] == 3
        assert s.method_counts["zscore"] == 1


class TestFilterNormEntries:
    def test_filter_by_method(self):
        entries = _make_norm_entries(5)
        entries[0] = make_norm_result_entry(0, "zscore", 0.0, 1.0, 5, 5)
        filtered = filter_norm_by_method(entries, "minmax")
        assert all(e.method == "minmax" for e in filtered)

    def test_filter_by_min_spread(self):
        entries = _make_norm_entries(10)
        filtered = filter_norm_by_min_spread(entries, 0.4)
        assert all(e.spread >= 0.4 for e in filtered)

    def test_top_k_by_spread(self):
        entries = _make_norm_entries(10)
        top = top_k_norm_by_spread(entries, 3)
        assert len(top) == 3
        spreads = [e.spread for e in top]
        assert spreads == sorted(spreads, reverse=True)

    def test_best_norm_entry(self):
        entries = _make_norm_entries(5)
        best = best_norm_entry(entries)
        assert best.spread == max(e.spread for e in entries)

    def test_best_norm_entry_empty(self):
        assert best_norm_entry([]) is None

    def test_norm_spread_stats(self):
        entries = _make_norm_entries(5)
        stats = norm_spread_stats(entries)
        assert stats["count"] == 5
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_norm_spread_stats_empty(self):
        stats = norm_spread_stats([])
        assert stats["count"] == 0

    def test_compare_norm_summaries(self):
        s1 = summarise_norm_result_entries(_make_norm_entries(3))
        s2 = summarise_norm_result_entries(_make_norm_entries(5))
        cmp = compare_norm_summaries(s1, s2)
        assert "delta_mean_spread" in cmp
        assert "same_best" in cmp

    def test_batch_summarise_norm(self):
        groups = [_make_norm_entries(3), _make_norm_entries(4)]
        summaries = batch_summarise_norm_entries(groups)
        assert len(summaries) == 2


def _make_noise_result_entries(n=5):
    rng = np.random.default_rng(29)
    entries = []
    for i in range(n):
        nb = float(rng.uniform(10.0, 30.0))
        na = float(rng.uniform(2.0, 9.0))
        entries.append(make_noise_result_entry(
            image_id=i,
            method="gaussian",
            noise_before=nb,
            noise_after=na,
            n_pixels=1024,
        ))
    return entries


class TestNoiseResultEntry:
    def test_delta_computed(self):
        e = make_noise_result_entry(0, "gaussian", 20.0, 5.0, 1024)
        assert e.noise_delta == pytest.approx(15.0)

    def test_fields(self):
        e = make_noise_result_entry(1, "bilateral", 15.0, 3.0, 512)
        assert e.method == "bilateral"
        assert e.n_pixels == 512

    def test_params_stored(self):
        e = make_noise_result_entry(0, "gaussian", 20.0, 5.0, 1024, sigma=3)
        assert e.params["sigma"] == 3


class TestSummariseNoiseResultEntries:
    def test_empty(self):
        s = summarise_noise_result_entries([])
        assert s.n_images == 0
        assert s.best_image_id is None

    def test_summary_fields(self):
        entries = _make_noise_result_entries(5)
        s = summarise_noise_result_entries(entries)
        assert s.n_images == 5
        assert isinstance(s.mean_noise_before, float)
        assert isinstance(s.mean_noise_after, float)

    def test_best_worst(self):
        entries = _make_noise_result_entries(5)
        s = summarise_noise_result_entries(entries)
        deltas = [e.noise_delta for e in entries]
        assert entries[s.best_image_id].noise_delta == max(deltas)
        assert entries[s.worst_image_id].noise_delta == min(deltas)


class TestFilterNoiseResultEntries:
    def test_filter_by_method(self):
        entries = _make_noise_result_entries(5)
        entries[0] = make_noise_result_entry(0, "bilateral", 20.0, 5.0, 1024)
        filtered = filter_noise_by_method(entries, "gaussian")
        assert all(e.method == "gaussian" for e in filtered)

    def test_filter_by_max_after(self):
        entries = _make_noise_result_entries(10)
        filtered = filter_noise_by_max_after(entries, 7.0)
        assert all(e.noise_after <= 7.0 for e in filtered)

    def test_filter_by_min_delta(self):
        entries = _make_noise_result_entries(10)
        filtered = filter_noise_by_min_delta(entries, 10.0)
        assert all(e.noise_delta >= 10.0 for e in filtered)

    def test_top_k_by_delta(self):
        entries = _make_noise_result_entries(10)
        top = top_k_noise_by_delta(entries, 3)
        assert len(top) == 3
        deltas = [e.noise_delta for e in top]
        assert deltas == sorted(deltas, reverse=True)

    def test_best_noise_entry(self):
        entries = _make_noise_result_entries(5)
        best = best_noise_entry(entries)
        assert best.noise_delta == max(e.noise_delta for e in entries)

    def test_best_noise_entry_empty(self):
        assert best_noise_entry([]) is None

    def test_noise_delta_stats(self):
        entries = _make_noise_result_entries(5)
        stats = noise_delta_stats(entries)
        assert stats["count"] == 5

    def test_noise_delta_stats_empty(self):
        stats = noise_delta_stats([])
        assert stats["count"] == 0

    def test_compare_noise_result_summaries(self):
        s1 = summarise_noise_result_entries(_make_noise_result_entries(3))
        s2 = summarise_noise_result_entries(_make_noise_result_entries(5))
        cmp = compare_noise_result_summaries(s1, s2)
        assert "delta_mean_delta" in cmp
        assert "same_best" in cmp

    def test_batch_summarise_noise(self):
        groups = [_make_noise_result_entries(3), _make_noise_result_entries(4)]
        summaries = batch_summarise_noise_entries(groups)
        assert len(summaries) == 2

