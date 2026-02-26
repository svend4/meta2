"""Integration tests for utils batch 3a.

Covers:
  - puzzle_reconstruction.utils.filter_pipeline_utils
  - puzzle_reconstruction.utils.fragment_filter_utils
  - puzzle_reconstruction.utils.fragment_stats
  - puzzle_reconstruction.utils.freq_metric_utils
  - puzzle_reconstruction.utils.frequency_utils
"""
import math
import pytest
import numpy as np

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from puzzle_reconstruction.utils.filter_pipeline_utils import (
    FilterStepConfig,
    FilterStepResult,
    FilterPipelineSummary,
    make_filter_step,
    steps_from_log,
    summarise_pipeline,
    filter_effective_steps,
    filter_by_removal_rate,
    most_aggressive_step,
    least_aggressive_step,
    pipeline_stats,
    compare_pipelines,
    batch_summarise_pipelines,
)
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
)
from puzzle_reconstruction.utils.fragment_stats import (
    FragmentMetrics,
    CollectionStats,
    compute_fragment_metrics,
    compute_collection_stats,
    area_histogram,
    compare_collections,
    outlier_indices,
)
from puzzle_reconstruction.utils.freq_metric_utils import (
    BandEnergyRecord,
    SpectrumComparisonRecord,
    FreqBatchSummary,
    MetricSnapshot,
    MetricRunSummary,
    MovingAverageResult,
    GreedyStepRecord,
    AssemblyRunRecord,
    make_band_energy_record,
    make_metric_snapshot,
    make_greedy_step,
)
from puzzle_reconstruction.utils.frequency_utils import (
    FrequencyConfig,
    compute_fft_magnitude,
    radial_power_spectrum,
    frequency_band_energy,
    high_frequency_ratio,
    low_pass_filter,
    high_pass_filter,
    compare_frequency_spectra,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_mask(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=np.uint8) * 255


def _rect_mask(h: int, w: int, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[r0:r1, c0:c1] = 255
    return m


def _rand_img(h: int = 64, w: int = 64) -> np.ndarray:
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _make_step(name: str, n_in: int, n_out: int) -> FilterStepResult:
    return make_filter_step(name, n_in, n_out)


# ===========================================================================
# filter_pipeline_utils (12 tests)
# ===========================================================================

class TestFilterPipelineUtils:

    def test_filter_step_config_defaults(self):
        cfg = FilterStepConfig()
        assert cfg.name == "threshold"
        assert cfg.threshold == 0.5
        assert cfg.top_k == 0
        assert cfg.deduplicate is False

    def test_filter_step_config_invalid_threshold(self):
        with pytest.raises(ValueError):
            FilterStepConfig(threshold=1.5)

    def test_filter_step_config_invalid_top_k(self):
        with pytest.raises(ValueError):
            FilterStepConfig(top_k=-1)

    def test_make_filter_step_removal(self):
        s = _make_step("dedupe", 100, 80)
        assert s.n_removed == 20
        assert s.step_name == "dedupe"

    def test_removal_rate_zero_input(self):
        s = _make_step("empty", 0, 0)
        assert s.removal_rate == 0.0

    def test_removal_rate_nonzero(self):
        s = _make_step("thresh", 200, 150)
        assert math.isclose(s.removal_rate, 50 / 200)

    def test_steps_from_log(self):
        log = [
            {"step_name": "a", "n_input": 100, "n_output": 80},
            {"step_name": "b", "n_input": 80, "n_output": 60},
        ]
        steps = steps_from_log(log)
        assert len(steps) == 2
        assert steps[0].step_name == "a"
        assert steps[1].n_output == 60

    def test_summarise_pipeline(self):
        steps = [_make_step("s1", 100, 70), _make_step("s2", 70, 50)]
        s = summarise_pipeline(steps)
        assert s.n_initial == 100
        assert s.n_final == 50
        assert s.total_removed == 50
        assert math.isclose(s.overall_removal_rate, 0.5)

    def test_summarise_pipeline_empty(self):
        s = summarise_pipeline([])
        assert s.n_initial == 0
        assert s.n_final == 0

    def test_filter_effective_steps(self):
        steps = [_make_step("a", 100, 100), _make_step("b", 100, 60)]
        eff = filter_effective_steps(steps)
        assert len(eff) == 1
        assert eff[0].step_name == "b"

    def test_most_and_least_aggressive(self):
        steps = [_make_step("a", 100, 90), _make_step("b", 90, 10)]
        assert most_aggressive_step(steps).step_name == "b"
        assert least_aggressive_step(steps).step_name == "a"

    def test_pipeline_stats_keys(self):
        steps = [_make_step("x", 200, 100), _make_step("y", 100, 80)]
        stats = pipeline_stats(steps)
        assert "n_steps" in stats
        assert stats["n_steps"] == 2
        assert stats["total_removed"] == 120

    def test_compare_pipelines(self):
        sa = summarise_pipeline([_make_step("a", 100, 50)])
        sb = summarise_pipeline([_make_step("b", 100, 70)])
        diff = compare_pipelines(sa, sb)
        assert diff["n_final_delta"] == -20

    def test_batch_summarise_pipelines(self):
        logs = [
            [{"step_name": "s", "n_input": 50, "n_output": 40}],
            [{"step_name": "t", "n_input": 80, "n_output": 60}],
        ]
        summaries = batch_summarise_pipelines(logs)
        assert len(summaries) == 2
        assert summaries[0].n_initial == 50
        assert summaries[1].n_initial == 80


# ===========================================================================
# fragment_filter_utils (11 tests)
# ===========================================================================

class TestFragmentFilterUtils:

    def test_compute_fragment_area_solid(self):
        mask = _solid_mask(10, 10)
        assert compute_fragment_area(mask) == 100.0

    def test_compute_fragment_area_empty(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        assert compute_fragment_area(mask) == 0.0

    def test_compute_aspect_ratio_square(self):
        mask = _solid_mask(20, 20)
        assert math.isclose(compute_aspect_ratio(mask), 1.0)

    def test_compute_aspect_ratio_rect(self):
        mask = _rect_mask(40, 20, 0, 40, 0, 10)
        ar = compute_aspect_ratio(mask)
        assert 0 < ar <= 1.0

    def test_compute_fill_ratio_solid(self):
        mask = _solid_mask(8, 8)
        assert math.isclose(compute_fill_ratio(mask), 1.0)

    def test_compute_fill_ratio_sparse(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0, 0] = 255
        mask[9, 9] = 255
        fr = compute_fill_ratio(mask)
        assert 0.0 < fr < 1.0

    def test_evaluate_fragment_passes(self):
        mask = _solid_mask(20, 20)
        cfg = FragmentFilterConfig(min_area=10, max_area=1000)
        q = evaluate_fragment(0, mask, cfg)
        assert q.passed is True
        assert q.reject_reason is None

    def test_evaluate_fragment_area_too_small(self):
        mask = _rect_mask(10, 10, 4, 6, 4, 6)
        cfg = FragmentFilterConfig(min_area=500)
        q = evaluate_fragment(1, mask, cfg)
        assert q.passed is False
        assert q.reject_reason == "area_too_small"

    def test_deduplicate_fragments(self):
        img = rng.integers(0, 256, (16, 16), dtype=np.uint8)
        pairs = [(0, img), (1, img), (2, img.copy())]
        deduped = deduplicate_fragments(pairs)
        # at least one kept, duplicates removed
        assert len(deduped) >= 1

    def test_filter_fragments_all_pass(self):
        mask = _solid_mask(30, 30)
        img = _rand_img(30, 30)
        frags = [(i, img, mask) for i in range(5)]
        cfg = FragmentFilterConfig(min_area=1, deduplicate=False)
        kept, quals = filter_fragments(frags, cfg)
        assert len(kept) == 5
        assert all(q.passed for q in quals)

    def test_sort_by_area(self):
        masks = [_rect_mask(20, 20, 0, i + 2, 0, i + 2) for i in range(4)]
        img = _rand_img(20, 20)
        frags = [(i, img, masks[i]) for i in range(4)]
        sorted_frags = sort_by_area(frags, descending=True)
        areas = [compute_fragment_area(t[2]) for t in sorted_frags]
        assert areas == sorted(areas, reverse=True)

    def test_top_k_fragments(self):
        mask = _solid_mask(10, 10)
        img = _rand_img(10, 10)
        frags = [(i, img, mask) for i in range(6)]
        top = top_k_fragments(frags, k=3)
        assert len(top) == 3


# ===========================================================================
# fragment_stats (11 tests)
# ===========================================================================

class TestFragmentStats:

    def _make_metrics(self, n: int = 5) -> list:
        masks = [_solid_mask(10 + i, 10 + i) for i in range(n)]
        return [compute_fragment_metrics(i, masks[i]) for i in range(n)]

    def test_fragment_metrics_valid(self):
        mask = _solid_mask(20, 20)
        m = compute_fragment_metrics(0, mask, n_edges=4)
        assert m.area == 400.0
        assert m.aspect == 1.0
        assert math.isclose(m.density, 1.0)
        assert m.n_edges == 4
        assert m.perimeter >= 0

    def test_fragment_metrics_invalid_id(self):
        with pytest.raises(ValueError):
            FragmentMetrics(fragment_id=-1, area=10, aspect=1.0,
                            density=0.5, n_edges=0, perimeter=0)

    def test_fragment_metrics_invalid_area(self):
        with pytest.raises(ValueError):
            FragmentMetrics(fragment_id=0, area=-1, aspect=1.0,
                            density=0.5, n_edges=0, perimeter=0)

    def test_fragment_metrics_invalid_aspect(self):
        with pytest.raises(ValueError):
            FragmentMetrics(fragment_id=0, area=10, aspect=0.0,
                            density=0.5, n_edges=0, perimeter=0)

    def test_compute_fragment_metrics_3d_raises(self):
        bad = np.ones((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_fragment_metrics(0, bad)

    def test_compute_collection_stats(self):
        mets = self._make_metrics(6)
        cs = compute_collection_stats(mets)
        assert cs.n_fragments == 6
        assert cs.total_area > 0
        assert cs.min_area <= cs.mean_area <= cs.max_area

    def test_compute_collection_stats_empty_raises(self):
        with pytest.raises(ValueError):
            compute_collection_stats([])

    def test_area_histogram_normalized(self):
        mets = self._make_metrics(8)
        counts, edges = area_histogram(mets, n_bins=4, normalize=True)
        assert abs(counts.sum() - 1.0) < 1e-9
        assert len(edges) == 5

    def test_area_histogram_unnormalized(self):
        mets = self._make_metrics(6)
        counts, edges = area_histogram(mets, n_bins=3, normalize=False)
        assert counts.sum() == 6

    def test_compare_collections(self):
        mets_a = self._make_metrics(4)
        mets_b = self._make_metrics(4)
        sa = compute_collection_stats(mets_a)
        sb = compute_collection_stats(mets_b)
        diff = compare_collections(sa, sb)
        assert "delta_total_area" in diff
        assert math.isclose(diff["delta_total_area"], sa.total_area - sb.total_area)

    def test_outlier_indices_detects_outlier(self):
        masks = [_solid_mask(10, 10) for _ in range(9)]
        masks.append(_solid_mask(200, 200))  # large outlier
        mets = [compute_fragment_metrics(i, masks[i]) for i in range(10)]
        idxs = outlier_indices(mets, z_threshold=2.0, by="area")
        assert 9 in idxs

    def test_collection_stats_to_dict(self):
        mets = self._make_metrics(3)
        cs = compute_collection_stats(mets)
        d = cs.to_dict()
        assert "n_fragments" in d
        assert d["n_fragments"] == 3


# ===========================================================================
# freq_metric_utils (11 tests)
# ===========================================================================

class TestFreqMetricUtils:

    def test_band_energy_record_basic(self):
        rec = make_band_energy_record(1, [0.1, 0.5, 0.4])
        assert rec.n_bands == 3
        assert rec.dominant_band == 1
        assert math.isclose(rec.total_energy, 1.0)

    def test_band_energy_normalized(self):
        rec = make_band_energy_record(0, [2.0, 2.0, 4.0])
        norms = rec.normalized_energies
        assert math.isclose(sum(norms), 1.0)
        assert math.isclose(norms[2], 0.5)

    def test_band_energy_zero_total(self):
        rec = make_band_energy_record(0, [0.0, 0.0])
        assert rec.normalized_energies == [0.0, 0.0]

    def test_spectrum_comparison_record_valid(self):
        r = SpectrumComparisonRecord(0, 1, similarity=0.8)
        assert r.is_match is True

    def test_spectrum_comparison_record_no_match(self):
        r = SpectrumComparisonRecord(0, 1, similarity=0.3)
        assert r.is_match is False

    def test_spectrum_comparison_invalid_similarity(self):
        with pytest.raises(ValueError):
            SpectrumComparisonRecord(0, 1, similarity=1.5)

    def test_freq_batch_summary_is_valid(self):
        s = FreqBatchSummary(n_fragments=10, mean_entropy=1.2,
                             mean_centroid=0.4, n_bands=16)
        assert s.is_valid is True

    def test_freq_batch_summary_invalid(self):
        s = FreqBatchSummary(n_fragments=0, mean_entropy=0.0,
                             mean_centroid=0.0, n_bands=0)
        assert s.is_valid is False

    def test_metric_snapshot_properties(self):
        snap = make_metric_snapshot(5, {"loss": 0.3, "acc": 0.9}, label="train")
        assert snap.n_metrics == 2
        assert snap.get("loss") == 0.3
        assert snap.get("missing", 99.0) == 99.0
        assert "loss" in snap.metric_names

    def test_metric_snapshot_negative_step_raises(self):
        with pytest.raises(ValueError):
            MetricSnapshot(step=-1, values={})

    def test_greedy_step_record(self):
        gs = make_greedy_step(0, fragment_id=3, anchor_id=1, score=0.75)
        assert gs.fragment_id == 3
        assert gs.score == 0.75

    def test_assembly_run_record_placement_rate(self):
        run = AssemblyRunRecord(n_fragments=10)
        run.steps.append(make_greedy_step(0, 1, 0, 0.9))
        run.steps.append(make_greedy_step(1, 2, 1, 0.8))
        assert run.n_placed == 2
        assert math.isclose(run.placement_rate, 0.2)


# ===========================================================================
# frequency_utils (11 tests)
# ===========================================================================

class TestFrequencyUtils:

    def test_frequency_config_defaults(self):
        cfg = FrequencyConfig()
        assert cfg.n_bands == 32
        assert cfg.log_scale is True

    def test_frequency_config_invalid_n_bands(self):
        with pytest.raises(ValueError):
            FrequencyConfig(n_bands=1)

    def test_compute_fft_magnitude_shape(self):
        img = _rand_img(32, 32)
        mag = compute_fft_magnitude(img)
        assert mag.shape == (32, 32)
        assert mag.dtype == np.float32

    def test_compute_fft_magnitude_normalized(self):
        img = _rand_img(32, 32)
        cfg = FrequencyConfig(normalize=True)
        mag = compute_fft_magnitude(img, cfg)
        assert mag.max() <= 1.0 + 1e-6

    def test_compute_fft_magnitude_invalid_ndim(self):
        bad = np.ones((10, 10, 3, 2), dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_fft_magnitude(bad)

    def test_radial_power_spectrum_length(self):
        img = _rand_img(48, 48)
        cfg = FrequencyConfig(n_bands=16)
        spec = radial_power_spectrum(img, cfg)
        assert spec.shape == (16,)
        assert spec.dtype == np.float32

    def test_frequency_band_energy_positive(self):
        img = _rand_img(32, 32)
        energy = frequency_band_energy(img, 0.0, 0.5)
        assert energy >= 0.0

    def test_frequency_band_energy_invalid_range(self):
        img = _rand_img(32, 32)
        with pytest.raises(ValueError):
            frequency_band_energy(img, 0.5, 0.3)

    def test_high_frequency_ratio_in_range(self):
        img = _rand_img(32, 32)
        ratio = high_frequency_ratio(img, threshold_frac=0.5)
        assert 0.0 <= ratio <= 1.0

    def test_low_pass_filter_output_shape(self):
        img = _rand_img(32, 32)
        out = low_pass_filter(img, cutoff_frac=0.3)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_high_pass_filter_output_shape(self):
        img = _rand_img(32, 32)
        out = high_pass_filter(img, cutoff_frac=0.3)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_compare_frequency_spectra_identical(self):
        img = _rand_img(32, 32)
        sim = compare_frequency_spectra(img, img)
        assert math.isclose(sim, 1.0, abs_tol=1e-5)

    def test_compare_frequency_spectra_range(self):
        img1 = _rand_img(32, 32)
        img2 = _rand_img(32, 32)
        sim = compare_frequency_spectra(img1, img2)
        assert 0.0 <= sim <= 1.0
