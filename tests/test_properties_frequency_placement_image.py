"""
Property-based invariant tests for:
  - puzzle_reconstruction.utils.frequency_utils
  - puzzle_reconstruction.utils.placement_metrics_utils
  - puzzle_reconstruction.utils.image_stats
"""
from __future__ import annotations

import numpy as np
import pytest

# ─── frequency_utils ──────────────────────────────────────────────────────────
from puzzle_reconstruction.utils.frequency_utils import (
    FrequencyConfig,
    compare_frequency_spectra,
    compute_fft_magnitude,
    frequency_band_energy,
    high_frequency_ratio,
    high_pass_filter,
    low_pass_filter,
    radial_power_spectrum,
)

# ─── placement_metrics_utils ──────────────────────────────────────────────────
from puzzle_reconstruction.utils.placement_metrics_utils import (
    PlacementMetrics,
    PlacementMetricsConfig,
    assess_placement,
    batch_quality_scores,
    bbox_area,
    bbox_intersection_area,
    bbox_of_contour,
    best_of,
    compute_coverage,
    compute_pairwise_overlap,
    normalize_metrics,
    placement_density,
    quality_score,
)

# ─── image_stats ──────────────────────────────────────────────────────────────
from puzzle_reconstruction.utils.image_stats import (
    batch_stats,
    compare_images,
    compute_entropy,
    compute_gradient_stats,
    compute_histogram_stats,
    compute_image_stats,
    compute_sharpness,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _rng_img(h: int = 32, w: int = 32, channels: int = 1, seed: int = 0) -> np.ndarray:
    """Return uint8 image of shape (h, w) or (h, w, channels)."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, (h, w) if channels == 1 else (h, w, channels),
                        dtype=np.uint8)
    return data


def _rect_contour(x: float, y: float, w: float, h: float) -> np.ndarray:
    """Rectangular contour as (4, 2) float array."""
    return np.array([
        [x, y], [x + w, y], [x + w, y + h], [x, y + h]
    ], dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# Tests — frequency_utils
# ══════════════════════════════════════════════════════════════════════════════

class TestFrequencyConfig:
    def test_default_valid(self):
        cfg = FrequencyConfig()
        assert cfg.n_bands >= 2

    def test_n_bands_below_2_raises(self):
        with pytest.raises(ValueError):
            FrequencyConfig(n_bands=1)

    def test_n_bands_exactly_2_ok(self):
        cfg = FrequencyConfig(n_bands=2)
        assert cfg.n_bands == 2


class TestComputeFftMagnitude:
    """compute_fft_magnitude invariants."""

    @pytest.mark.parametrize("h,w,ch", [
        (16, 16, 1), (32, 32, 1), (16, 24, 1),
        (16, 16, 3), (24, 32, 3),
    ])
    def test_output_shape_matches_input(self, h, w, ch):
        img = _rng_img(h, w, ch)
        mag = compute_fft_magnitude(img)
        assert mag.shape == (h, w)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_normalized_output_bounded(self, seed):
        img = _rng_img(32, 32, 1, seed=seed)
        cfg = FrequencyConfig(normalize=True)
        mag = compute_fft_magnitude(img, cfg)
        assert float(mag.max()) <= 1.0 + 1e-7

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_non_negative(self, seed):
        img = _rng_img(32, 32, 1, seed=seed)
        mag = compute_fft_magnitude(img)
        assert float(mag.min()) >= 0.0

    def test_dtype_float32(self):
        img = _rng_img(16, 16, 1)
        mag = compute_fft_magnitude(img)
        assert mag.dtype == np.float32

    def test_constant_image_finite(self):
        img = np.full((16, 16), 128, dtype=np.uint8)
        mag = compute_fft_magnitude(img)
        assert np.all(np.isfinite(mag))

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError):
            compute_fft_magnitude(np.zeros((4, 4, 4, 4), dtype=np.uint8))


class TestRadialPowerSpectrum:
    """radial_power_spectrum invariants."""

    @pytest.mark.parametrize("n_bands", [4, 8, 16, 32])
    def test_output_length_equals_n_bands(self, n_bands):
        img = _rng_img(32, 32)
        cfg = FrequencyConfig(n_bands=n_bands)
        spec = radial_power_spectrum(img, cfg)
        assert len(spec) == n_bands

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_non_negative(self, seed):
        img = _rng_img(32, 32, seed=seed)
        spec = radial_power_spectrum(img)
        assert float(spec.min()) >= 0.0

    def test_normalized_max_le_one(self):
        img = _rng_img(32, 32)
        cfg = FrequencyConfig(normalize=True)
        spec = radial_power_spectrum(img, cfg)
        assert float(spec.max()) <= 1.0 + 1e-7

    def test_dtype_float32(self):
        img = _rng_img(16, 16)
        spec = radial_power_spectrum(img)
        assert spec.dtype == np.float32


class TestFrequencyBandEnergy:
    """frequency_band_energy invariants."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_non_negative(self, seed):
        img = _rng_img(32, 32, seed=seed)
        energy = frequency_band_energy(img, 0.0, 0.5)
        assert energy >= 0.0

    def test_full_band_ge_partial_band(self):
        img = _rng_img(32, 32)
        full = frequency_band_energy(img, 0.0, 1.0)
        partial = frequency_band_energy(img, 0.0, 0.5)
        assert full >= partial - 1e-9

    def test_invalid_fractions_raise(self):
        img = _rng_img(16, 16)
        with pytest.raises(ValueError):
            frequency_band_energy(img, 0.5, 0.3)   # low >= high
        with pytest.raises(ValueError):
            frequency_band_energy(img, -0.1, 0.5)  # low < 0
        with pytest.raises(ValueError):
            frequency_band_energy(img, 0.0, 1.1)   # high > 1


class TestHighFrequencyRatio:
    """high_frequency_ratio invariants."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_in_unit_interval(self, seed):
        img = _rng_img(32, 32, seed=seed)
        r = high_frequency_ratio(img, 0.5)
        assert 0.0 <= r <= 1.0

    @pytest.mark.parametrize("th", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_various_thresholds_in_unit_interval(self, th):
        img = _rng_img(32, 32)
        r = high_frequency_ratio(img, th)
        assert 0.0 <= r <= 1.0

    def test_invalid_threshold_raises(self):
        img = _rng_img(16, 16)
        with pytest.raises(ValueError):
            high_frequency_ratio(img, 0.0)
        with pytest.raises(ValueError):
            high_frequency_ratio(img, 1.0)


class TestLowHighPassFilter:
    """low_pass_filter and high_pass_filter invariants."""

    @pytest.mark.parametrize("h,w,ch", [
        (32, 32, 1), (32, 32, 3),
    ])
    def test_low_pass_same_shape(self, h, w, ch):
        img = _rng_img(h, w, ch)
        result = low_pass_filter(img, 0.5)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("h,w,ch", [
        (32, 32, 1), (32, 32, 3),
    ])
    def test_high_pass_same_shape(self, h, w, ch):
        img = _rng_img(h, w, ch)
        result = high_pass_filter(img, 0.3)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_low_pass_invalid_cutoff_raises(self):
        img = _rng_img(16, 16)
        with pytest.raises(ValueError):
            low_pass_filter(img, 0.0)
        with pytest.raises(ValueError):
            low_pass_filter(img, 1.1)

    def test_high_pass_invalid_cutoff_raises(self):
        img = _rng_img(16, 16)
        with pytest.raises(ValueError):
            high_pass_filter(img, -0.1)
        with pytest.raises(ValueError):
            high_pass_filter(img, 1.0)


class TestCompareFrequencySpectra:
    """compare_frequency_spectra invariants."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_same_image_returns_one(self, seed):
        img = _rng_img(32, 32, seed=seed)
        sim = compare_frequency_spectra(img, img)
        assert abs(sim - 1.0) < 1e-5

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_result_in_unit_interval(self, seed):
        img1 = _rng_img(32, 32, seed=seed)
        img2 = _rng_img(32, 32, seed=seed + 10)
        sim = compare_frequency_spectra(img1, img2)
        assert 0.0 <= sim <= 1.0

    def test_symmetry(self):
        img1 = _rng_img(32, 32, seed=42)
        img2 = _rng_img(32, 32, seed=99)
        assert abs(compare_frequency_spectra(img1, img2) -
                   compare_frequency_spectra(img2, img1)) < 1e-7


# ══════════════════════════════════════════════════════════════════════════════
# Tests — placement_metrics_utils
# ══════════════════════════════════════════════════════════════════════════════

class TestPlacementDensity:
    """placement_density invariants."""

    @pytest.mark.parametrize("n_placed,n_total", [
        (0, 0), (0, 5), (5, 5), (3, 10), (10, 10),
    ])
    def test_in_unit_interval(self, n_placed, n_total):
        d = placement_density(n_placed, n_total)
        assert 0.0 <= d <= 1.0

    def test_full_placement_gives_one(self):
        assert placement_density(5, 5) == 1.0

    def test_no_placement_gives_zero(self):
        assert placement_density(0, 10) == 0.0

    def test_zero_total_gives_zero(self):
        assert placement_density(0, 0) == 0.0

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            placement_density(-1, 5)
        with pytest.raises(ValueError):
            placement_density(3, -1)

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_monotone_in_n_placed(self, n):
        d1 = placement_density(n // 2, n)
        d2 = placement_density(n, n)
        assert d1 <= d2


class TestBboxArea:
    """bbox_area invariants."""

    def test_non_negative(self):
        for x0, y0, x1, y1 in [
            (0, 0, 1, 1), (0, 0, 0, 0), (5, 5, 10, 20),
        ]:
            assert bbox_area((x0, y0, x1, y1)) >= 0.0

    def test_point_has_zero_area(self):
        assert bbox_area((3.0, 4.0, 3.0, 4.0)) == 0.0

    def test_inverted_bbox_has_zero_area(self):
        assert bbox_area((5, 5, 3, 3)) == 0.0

    def test_rectangle_area(self):
        assert abs(bbox_area((0, 0, 4, 3)) - 12.0) < 1e-9


class TestBboxIntersectionArea:
    """bbox_intersection_area invariants."""

    def test_non_negative(self):
        a = (0, 0, 5, 5)
        b = (3, 3, 8, 8)
        assert bbox_intersection_area(a, b) >= 0.0

    def test_disjoint_is_zero(self):
        a = (0, 0, 1, 1)
        b = (5, 5, 6, 6)
        assert bbox_intersection_area(a, b) == 0.0

    def test_contained_equals_inner(self):
        outer = (0.0, 0.0, 10.0, 10.0)
        inner = (2.0, 2.0, 4.0, 4.0)
        inner_area = bbox_area(inner)
        assert abs(bbox_intersection_area(outer, inner) - inner_area) < 1e-9

    def test_symmetric(self):
        a = (0, 0, 5, 5)
        b = (3, 3, 8, 8)
        assert abs(bbox_intersection_area(a, b) -
                   bbox_intersection_area(b, a)) < 1e-9

    def test_intersection_le_min_area(self):
        a = (0, 0, 4, 4)
        b = (2, 2, 6, 6)
        inter = bbox_intersection_area(a, b)
        assert inter <= min(bbox_area(a), bbox_area(b)) + 1e-9


class TestBboxOfContour:
    """bbox_of_contour invariants."""

    def test_contour_within_bbox(self):
        pts = np.array([[1, 2], [3, 4], [5, 1]], dtype=np.float64)
        x_min, y_min, x_max, y_max = bbox_of_contour(pts)
        assert x_min <= 1.0 and y_min <= 1.0
        assert x_max >= 5.0 and y_max >= 4.0

    def test_offset_shifts_bbox(self):
        pts = np.array([[0, 0], [1, 1]], dtype=np.float64)
        b0 = bbox_of_contour(pts, (0.0, 0.0))
        b1 = bbox_of_contour(pts, (10.0, 20.0))
        assert abs(b1[0] - b0[0] - 10.0) < 1e-9
        assert abs(b1[1] - b0[1] - 20.0) < 1e-9


class TestQualityScore:
    """quality_score invariants."""

    @pytest.mark.parametrize("density,coverage,overlap", [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.5, 0.5, 100.0),
        (1.0, 1.0, 1e6),
    ])
    def test_in_unit_interval(self, density, coverage, overlap):
        q = quality_score(density, coverage, overlap)
        assert 0.0 <= q <= 1.0

    def test_no_overlap_perfect_gives_high_score(self):
        q = quality_score(1.0, 1.0, 0.0)
        assert q > 0.9

    def test_large_overlap_reduces_score(self):
        q_low = quality_score(1.0, 1.0, 1e9)
        q_high = quality_score(1.0, 1.0, 0.0)
        assert q_low <= q_high


class TestAssessPlacement:
    """assess_placement invariants."""

    def _simple_setup(self, n: int = 3):
        contours = [_rect_contour(i * 20.0, 0, 15, 15) for i in range(n)]
        positions = [(float(i * 20), 0.0) for i in range(n)]
        return positions, contours

    def test_metrics_in_valid_range(self):
        positions, contours = self._simple_setup(3)
        m = assess_placement(positions, contours, n_total=5)
        assert 0.0 <= m.density <= 1.0
        assert 0.0 <= m.coverage <= 1.0
        assert m.pairwise_overlap >= 0.0
        assert 0.0 <= m.quality_score <= 1.0

    def test_n_placed_equals_min_len(self):
        positions, contours = self._simple_setup(4)
        m = assess_placement(positions, contours, n_total=4)
        assert m.n_placed == 4
        assert m.n_total == 4

    def test_empty_placement_metrics(self):
        m = assess_placement([], [], n_total=5)
        assert m.density == 0.0
        assert m.coverage == 0.0
        assert m.pairwise_overlap == 0.0


class TestNormalizeMetrics:
    """normalize_metrics invariants."""

    def _make_metrics(self, scores):
        return [
            PlacementMetrics(
                n_placed=1, n_total=1,
                density=0.5, coverage=0.5,
                pairwise_overlap=0.0,
                quality_score=float(s),
            )
            for s in scores
        ]

    def test_max_score_normalized_to_one(self):
        ms = self._make_metrics([0.1, 0.5, 0.9])
        normed = normalize_metrics(ms)
        assert abs(max(m.quality_score for m in normed) - 1.0) < 1e-7

    def test_all_scores_in_unit_interval(self):
        ms = self._make_metrics([0.2, 0.4, 0.6, 0.8])
        normed = normalize_metrics(ms)
        for m in normed:
            assert 0.0 <= m.quality_score <= 1.0 + 1e-7

    def test_constant_scores_all_one(self):
        ms = self._make_metrics([0.5, 0.5, 0.5])
        normed = normalize_metrics(ms)
        for m in normed:
            assert abs(m.quality_score - 1.0) < 1e-7

    def test_empty_returns_empty(self):
        assert normalize_metrics([]) == []


class TestBestOf:
    """best_of invariants."""

    def _make_metrics(self, scores):
        return [
            PlacementMetrics(
                n_placed=1, n_total=1,
                density=0.5, coverage=0.5,
                pairwise_overlap=0.0,
                quality_score=float(s),
            )
            for s in scores
        ]

    def test_returns_index_of_max(self):
        ms = self._make_metrics([0.1, 0.9, 0.3])
        assert best_of(ms) == 1

    def test_single_element(self):
        ms = self._make_metrics([0.7])
        assert best_of(ms) == 0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            best_of([])


class TestBatchQualityScores:
    """batch_quality_scores invariants."""

    def _make_metrics(self, scores):
        return [
            PlacementMetrics(
                n_placed=1, n_total=1,
                density=s, coverage=s,
                pairwise_overlap=0.0,
                quality_score=s,
            )
            for s in scores
        ]

    def test_length_preserved(self):
        ms = self._make_metrics([0.3, 0.6, 0.9])
        out = batch_quality_scores(ms)
        assert len(out) == 3

    def test_all_in_unit_interval(self):
        ms = self._make_metrics([0.0, 0.5, 1.0])
        out = batch_quality_scores(ms)
        for q in out:
            assert 0.0 <= q <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Tests — image_stats
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeEntropy:
    """compute_entropy invariants."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_entropy_in_0_8_bits(self, seed):
        img = _rng_img(32, 32, seed=seed)
        e = compute_entropy(img)
        assert 0.0 <= e <= 8.0

    def test_constant_image_has_zero_entropy(self):
        img = np.full((16, 16), 128, dtype=np.uint8)
        e = compute_entropy(img)
        assert abs(e) < 1e-9

    def test_color_image_works(self):
        img = _rng_img(32, 32, 3)
        e = compute_entropy(img)
        assert 0.0 <= e <= 8.0

    def test_uniform_random_has_high_entropy(self):
        img = _rng_img(256, 256, 1, seed=7)
        e = compute_entropy(img)
        assert e > 6.0   # near-uniform should be > 6 bits


class TestComputeSharpness:
    """compute_sharpness invariants."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_non_negative(self, seed):
        img = _rng_img(32, 32, seed=seed)
        s = compute_sharpness(img)
        assert s >= 0.0

    def test_constant_image_is_zero(self):
        img = np.full((16, 16), 200, dtype=np.uint8)
        s = compute_sharpness(img)
        assert abs(s) < 1e-9

    def test_high_contrast_image_sharper(self):
        # checkerboard is sharper than smooth gradient
        checker = np.zeros((32, 32), dtype=np.uint8)
        checker[::2, ::2] = 255
        checker[1::2, 1::2] = 255
        smooth = np.tile(np.linspace(0, 255, 32), (32, 1)).astype(np.uint8)
        assert compute_sharpness(checker) > compute_sharpness(smooth)


class TestComputeHistogramStats:
    """compute_histogram_stats invariants."""

    def test_returns_required_keys(self):
        img = _rng_img(32, 32)
        stats = compute_histogram_stats(img)
        for key in ("mean", "std", "skewness", "kurtosis"):
            assert key in stats

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_mean_in_pixel_range(self, seed):
        img = _rng_img(32, 32, seed=seed)
        stats = compute_histogram_stats(img)
        assert 0.0 <= stats["mean"] <= 255.0

    def test_std_non_negative(self):
        img = _rng_img(32, 32)
        stats = compute_histogram_stats(img)
        assert stats["std"] >= 0.0

    def test_constant_image_zero_std_zero_skew(self):
        img = np.full((16, 16), 100, dtype=np.uint8)
        stats = compute_histogram_stats(img)
        assert abs(stats["std"]) < 1e-9
        assert abs(stats["skewness"]) < 1e-9


class TestComputeImageStats:
    """compute_image_stats invariants."""

    @pytest.mark.parametrize("h,w", [(16, 16), (32, 32), (64, 64)])
    def test_n_pixels_correct(self, h, w):
        img = _rng_img(h, w)
        stats = compute_image_stats(img)
        assert stats.n_pixels == h * w

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_histogram_normalized(self, seed):
        img = _rng_img(32, 32, seed=seed)
        stats = compute_image_stats(img)
        assert abs(stats.histogram.sum() - 1.0) < 1e-5

    def test_histogram_non_negative(self):
        img = _rng_img(32, 32)
        stats = compute_image_stats(img)
        assert float(stats.histogram.min()) >= 0.0

    def test_mean_in_range(self):
        img = _rng_img(32, 32)
        stats = compute_image_stats(img)
        assert 0.0 <= stats.mean <= 255.0

    def test_std_non_negative(self):
        img = _rng_img(32, 32)
        stats = compute_image_stats(img)
        assert stats.std >= 0.0

    def test_entropy_non_negative(self):
        img = _rng_img(32, 32)
        stats = compute_image_stats(img)
        assert stats.entropy >= 0.0

    def test_sharpness_non_negative(self):
        img = _rng_img(32, 32)
        stats = compute_image_stats(img)
        assert stats.sharpness >= 0.0

    def test_percentiles_ordered(self):
        img = _rng_img(64, 64)
        stats = compute_image_stats(img)
        percs = [stats.percentiles[p] for p in sorted(stats.percentiles)]
        assert all(a <= b for a, b in zip(percs, percs[1:]))

    def test_contrast_equals_std(self):
        img = _rng_img(32, 32)
        stats = compute_image_stats(img)
        assert abs(stats.contrast - stats.std) < 1e-9

    def test_color_image_works(self):
        img = _rng_img(32, 32, 3)
        stats = compute_image_stats(img)
        assert stats.n_pixels == 32 * 32


class TestComputeGradientStats:
    """compute_gradient_stats invariants."""

    def test_required_keys_present(self):
        img = _rng_img(32, 32)
        gs = compute_gradient_stats(img)
        for key in ("grad_mean", "grad_std", "grad_max", "grad_energy"):
            assert key in gs

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_all_non_negative(self, seed):
        img = _rng_img(32, 32, seed=seed)
        gs = compute_gradient_stats(img)
        for v in gs.values():
            assert v >= 0.0

    def test_constant_image_all_zeros(self):
        img = np.full((16, 16), 50, dtype=np.uint8)
        gs = compute_gradient_stats(img)
        for v in gs.values():
            assert abs(v) < 1e-9


class TestCompareImages:
    """compare_images invariants."""

    def test_required_keys(self):
        img1 = _rng_img(32, 32)
        img2 = _rng_img(32, 32, seed=1)
        result = compare_images(img1, img2)
        for key in ("mean_diff", "std_ratio", "entropy_diff",
                    "sharpness_ratio", "hist_corr", "hist_bhatt"):
            assert key in result

    def test_same_image_hist_corr_is_one(self):
        img = _rng_img(32, 32)
        result = compare_images(img, img)
        assert abs(result["hist_corr"] - 1.0) < 1e-5

    def test_same_image_bhatt_is_zero(self):
        img = _rng_img(32, 32)
        result = compare_images(img, img)
        assert abs(result["hist_bhatt"]) < 1e-5

    def test_same_image_mean_diff_zero(self):
        img = _rng_img(32, 32)
        result = compare_images(img, img)
        assert abs(result["mean_diff"]) < 1e-7

    def test_same_image_std_ratio_one(self):
        img = _rng_img(64, 64)
        result = compare_images(img, img)
        assert abs(result["std_ratio"] - 1.0) < 1e-5


class TestBatchStats:
    """batch_stats invariants."""

    def test_length_preserved(self):
        imgs = [_rng_img(16, 16, seed=i) for i in range(5)]
        stats = batch_stats(imgs)
        assert len(stats) == 5

    def test_empty_list_returns_empty(self):
        assert batch_stats([]) == []

    def test_each_has_correct_n_pixels(self):
        imgs = [_rng_img(h, h, seed=i) for i, h in enumerate([16, 24, 32])]
        stats = batch_stats(imgs)
        assert stats[0].n_pixels == 16 * 16
        assert stats[1].n_pixels == 24 * 24
        assert stats[2].n_pixels == 32 * 32
