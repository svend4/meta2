"""Integration tests for utils batch 3a.

Covers:
  filter_pipeline_utils, fragment_filter_utils, fragment_stats,
  freq_metric_utils, frequency_utils
"""
import math
import pytest
import numpy as np

rng = np.random.default_rng(42)

from puzzle_reconstruction.utils.filter_pipeline_utils import (
    FilterStepConfig, FilterStepResult, make_filter_step, steps_from_log,
    summarise_pipeline, filter_effective_steps, most_aggressive_step,
    least_aggressive_step, pipeline_stats, compare_pipelines,
    batch_summarise_pipelines,
)
from puzzle_reconstruction.utils.fragment_filter_utils import (
    FragmentFilterConfig, compute_fragment_area, compute_aspect_ratio,
    compute_fill_ratio, evaluate_fragment, deduplicate_fragments,
    filter_fragments, sort_by_area, top_k_fragments,
)
from puzzle_reconstruction.utils.fragment_stats import (
    FragmentMetrics, compute_fragment_metrics, compute_collection_stats,
    area_histogram, compare_collections, outlier_indices,
)
from puzzle_reconstruction.utils.freq_metric_utils import (
    BandEnergyRecord, SpectrumComparisonRecord, FreqBatchSummary,
    MetricSnapshot, AssemblyRunRecord,
    make_band_energy_record, make_metric_snapshot, make_greedy_step,
)
from puzzle_reconstruction.utils.frequency_utils import (
    FrequencyConfig, compute_fft_magnitude, radial_power_spectrum,
    frequency_band_energy, high_frequency_ratio,
    low_pass_filter, high_pass_filter, compare_frequency_spectra,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _solid(h, w):
    return np.ones((h, w), dtype=np.uint8) * 255

def _rect(h, w, r0, r1, c0, c1):
    m = np.zeros((h, w), dtype=np.uint8)
    m[r0:r1, c0:c1] = 255
    return m

def _img(h=32, w=32):
    return rng.integers(0, 256, (h, w), dtype=np.uint8)

def _step(name, n_in, n_out):
    return make_filter_step(name, n_in, n_out)

def _metrics(n=5):
    return [compute_fragment_metrics(i, _solid(10 + i, 10 + i)) for i in range(n)]


# ===========================================================================
# filter_pipeline_utils  (12 tests)
# ===========================================================================
class TestFilterPipelineUtils:
    def test_config_defaults(self):
        cfg = FilterStepConfig()
        assert cfg.threshold == 0.5 and cfg.top_k == 0

    def test_config_invalid_threshold(self):
        with pytest.raises(ValueError): FilterStepConfig(threshold=2.0)

    def test_config_invalid_top_k(self):
        with pytest.raises(ValueError): FilterStepConfig(top_k=-1)

    def test_make_step_removal(self):
        s = _step("a", 100, 80)
        assert s.n_removed == 20

    def test_removal_rate_zero_input(self):
        assert _step("e", 0, 0).removal_rate == 0.0

    def test_removal_rate(self):
        s = _step("t", 200, 150)
        assert math.isclose(s.removal_rate, 0.25)

    def test_steps_from_log(self):
        log = [{"step_name": "a", "n_input": 100, "n_output": 80},
               {"step_name": "b", "n_input": 80, "n_output": 60}]
        steps = steps_from_log(log)
        assert len(steps) == 2 and steps[1].n_output == 60

    def test_summarise_pipeline(self):
        s = summarise_pipeline([_step("s1", 100, 70), _step("s2", 70, 50)])
        assert s.n_initial == 100 and s.n_final == 50
        assert math.isclose(s.overall_removal_rate, 0.5)

    def test_summarise_empty(self):
        s = summarise_pipeline([])
        assert s.n_initial == 0 and s.n_final == 0

    def test_filter_effective_steps(self):
        eff = filter_effective_steps([_step("a", 100, 100), _step("b", 100, 60)])
        assert len(eff) == 1 and eff[0].step_name == "b"

    def test_most_least_aggressive(self):
        steps = [_step("a", 100, 90), _step("b", 90, 10)]
        assert most_aggressive_step(steps).step_name == "b"
        assert least_aggressive_step(steps).step_name == "a"

    def test_pipeline_stats(self):
        stats = pipeline_stats([_step("x", 200, 100), _step("y", 100, 80)])
        assert stats["n_steps"] == 2 and stats["total_removed"] == 120

    def test_compare_pipelines(self):
        diff = compare_pipelines(
            summarise_pipeline([_step("a", 100, 50)]),
            summarise_pipeline([_step("b", 100, 70)]),
        )
        assert diff["n_final_delta"] == -20

    def test_batch_summarise(self):
        logs = [[{"step_name": "s", "n_input": 50, "n_output": 40}],
                [{"step_name": "t", "n_input": 80, "n_output": 60}]]
        sums = batch_summarise_pipelines(logs)
        assert len(sums) == 2 and sums[1].n_initial == 80


# ===========================================================================
# fragment_filter_utils  (11 tests)
# ===========================================================================
class TestFragmentFilterUtils:
    def test_area_solid(self):
        assert compute_fragment_area(_solid(10, 10)) == 100.0

    def test_area_empty(self):
        assert compute_fragment_area(np.zeros((20, 20), dtype=np.uint8)) == 0.0

    def test_aspect_square(self):
        assert math.isclose(compute_aspect_ratio(_solid(20, 20)), 1.0)

    def test_aspect_rect_le1(self):
        ar = compute_aspect_ratio(_rect(40, 20, 0, 40, 0, 10))
        assert 0 < ar <= 1.0

    def test_fill_solid(self):
        assert math.isclose(compute_fill_ratio(_solid(8, 8)), 1.0)

    def test_fill_sparse(self):
        m = np.zeros((10, 10), dtype=np.uint8)
        m[0, 0] = m[9, 9] = 255
        assert 0.0 < compute_fill_ratio(m) < 1.0

    def test_evaluate_passes(self):
        q = evaluate_fragment(0, _solid(20, 20), FragmentFilterConfig(min_area=10))
        assert q.passed and q.reject_reason is None

    def test_evaluate_area_too_small(self):
        q = evaluate_fragment(1, _rect(10, 10, 4, 6, 4, 6), FragmentFilterConfig(min_area=500))
        assert not q.passed and q.reject_reason == "area_too_small"

    def test_deduplicate(self):
        img = rng.integers(0, 256, (16, 16), dtype=np.uint8)
        deduped = deduplicate_fragments([(0, img), (1, img), (2, img)])
        assert 1 <= len(deduped) < 3

    def test_filter_all_pass(self):
        mask = _solid(30, 30)
        frags = [(i, _img(30, 30), mask) for i in range(5)]
        kept, quals = filter_fragments(frags, FragmentFilterConfig(min_area=1, deduplicate=False))
        assert len(kept) == 5 and all(q.passed for q in quals)

    def test_sort_by_area(self):
        masks = [_rect(20, 20, 0, i+2, 0, i+2) for i in range(4)]
        frags = [(i, _img(20, 20), masks[i]) for i in range(4)]
        sorted_f = sort_by_area(frags, descending=True)
        areas = [compute_fragment_area(t[2]) for t in sorted_f]
        assert areas == sorted(areas, reverse=True)

    def test_top_k(self):
        frags = [(i, _img(), _solid(10, 10)) for i in range(6)]
        assert len(top_k_fragments(frags, k=3)) == 3


# ===========================================================================
# fragment_stats  (11 tests)
# ===========================================================================
class TestFragmentStats:
    def test_metrics_valid(self):
        m = compute_fragment_metrics(0, _solid(20, 20), n_edges=4)
        assert m.area == 400.0 and m.n_edges == 4 and math.isclose(m.density, 1.0)

    def test_metrics_invalid_id(self):
        with pytest.raises(ValueError):
            FragmentMetrics(fragment_id=-1, area=10, aspect=1.0, density=0.5, n_edges=0, perimeter=0)

    def test_metrics_invalid_area(self):
        with pytest.raises(ValueError):
            FragmentMetrics(fragment_id=0, area=-1, aspect=1.0, density=0.5, n_edges=0, perimeter=0)

    def test_metrics_3d_raises(self):
        with pytest.raises(ValueError):
            compute_fragment_metrics(0, np.ones((10, 10, 3), dtype=np.uint8))

    def test_collection_stats(self):
        cs = compute_collection_stats(_metrics(6))
        assert cs.n_fragments == 6 and cs.min_area <= cs.mean_area <= cs.max_area

    def test_collection_stats_empty_raises(self):
        with pytest.raises(ValueError): compute_collection_stats([])

    def test_histogram_normalized(self):
        counts, edges = area_histogram(_metrics(8), n_bins=4, normalize=True)
        assert abs(counts.sum() - 1.0) < 1e-9 and len(edges) == 5

    def test_histogram_unnormalized(self):
        counts, _ = area_histogram(_metrics(6), n_bins=3, normalize=False)
        assert counts.sum() == 6

    def test_compare_collections(self):
        sa = compute_collection_stats(_metrics(4))
        sb = compute_collection_stats(_metrics(4))
        diff = compare_collections(sa, sb)
        assert math.isclose(diff["delta_total_area"], sa.total_area - sb.total_area)

    def test_outlier_detected(self):
        masks = [_solid(10, 10)] * 9 + [_solid(200, 200)]
        mets = [compute_fragment_metrics(i, masks[i]) for i in range(10)]
        assert 9 in outlier_indices(mets, z_threshold=2.0, by="area")

    def test_collection_to_dict(self):
        d = compute_collection_stats(_metrics(3)).to_dict()
        assert "n_fragments" in d and d["n_fragments"] == 3


# ===========================================================================
# freq_metric_utils  (11 tests)
# ===========================================================================
class TestFreqMetricUtils:
    def test_band_energy_basic(self):
        r = make_band_energy_record(1, [0.1, 0.5, 0.4])
        assert r.n_bands == 3 and r.dominant_band == 1
        assert math.isclose(r.total_energy, 1.0)

    def test_band_energy_normalized(self):
        r = make_band_energy_record(0, [2.0, 2.0, 4.0])
        norms = r.normalized_energies
        assert math.isclose(sum(norms), 1.0) and math.isclose(norms[2], 0.5)

    def test_band_energy_zero_total(self):
        assert make_band_energy_record(0, [0.0, 0.0]).normalized_energies == [0.0, 0.0]

    def test_spectrum_comparison_match(self):
        assert SpectrumComparisonRecord(0, 1, similarity=0.8).is_match is True

    def test_spectrum_comparison_no_match(self):
        assert SpectrumComparisonRecord(0, 1, similarity=0.3).is_match is False

    def test_spectrum_comparison_invalid(self):
        with pytest.raises(ValueError): SpectrumComparisonRecord(0, 1, similarity=1.5)

    def test_freq_batch_valid(self):
        assert FreqBatchSummary(10, 1.2, 0.4, 16).is_valid is True

    def test_freq_batch_invalid(self):
        assert FreqBatchSummary(0, 0.0, 0.0, 0).is_valid is False

    def test_metric_snapshot(self):
        snap = make_metric_snapshot(5, {"loss": 0.3, "acc": 0.9})
        assert snap.n_metrics == 2 and snap.get("loss") == 0.3
        assert snap.get("missing", 99.0) == 99.0

    def test_metric_snapshot_negative_step(self):
        with pytest.raises(ValueError): MetricSnapshot(step=-1, values={})

    def test_assembly_run_placement_rate(self):
        run = AssemblyRunRecord(n_fragments=10)
        run.steps += [make_greedy_step(i, i+1, i, 0.9) for i in range(2)]
        assert run.n_placed == 2 and math.isclose(run.placement_rate, 0.2)


# ===========================================================================
# frequency_utils  (11 tests)
# ===========================================================================
class TestFrequencyUtils:
    def test_config_defaults(self):
        cfg = FrequencyConfig()
        assert cfg.n_bands == 32 and cfg.log_scale is True

    def test_config_invalid_n_bands(self):
        with pytest.raises(ValueError): FrequencyConfig(n_bands=1)

    def test_fft_magnitude_shape(self):
        mag = compute_fft_magnitude(_img())
        assert mag.shape == (32, 32) and mag.dtype == np.float32

    def test_fft_magnitude_normalized(self):
        mag = compute_fft_magnitude(_img(), FrequencyConfig(normalize=True))
        assert mag.max() <= 1.0 + 1e-6

    def test_fft_magnitude_invalid_ndim(self):
        with pytest.raises(ValueError):
            compute_fft_magnitude(np.ones((4, 4, 3, 2), dtype=np.uint8))

    def test_radial_power_spectrum_length(self):
        spec = radial_power_spectrum(_img(48, 48), FrequencyConfig(n_bands=16))
        assert spec.shape == (16,) and spec.dtype == np.float32

    def test_band_energy_positive(self):
        assert frequency_band_energy(_img(), 0.0, 0.5) >= 0.0

    def test_band_energy_invalid_range(self):
        with pytest.raises(ValueError): frequency_band_energy(_img(), 0.5, 0.3)

    def test_high_freq_ratio_in_range(self):
        ratio = high_frequency_ratio(_img(), threshold_frac=0.5)
        assert 0.0 <= ratio <= 1.0

    def test_low_pass_filter_shape_dtype(self):
        out = low_pass_filter(_img(), cutoff_frac=0.3)
        assert out.shape == (32, 32) and out.dtype == np.uint8

    def test_high_pass_filter_shape_dtype(self):
        out = high_pass_filter(_img(), cutoff_frac=0.3)
        assert out.shape == (32, 32) and out.dtype == np.uint8

    def test_compare_spectra_identical(self):
        img = _img()
        assert math.isclose(compare_frequency_spectra(img, img), 1.0, abs_tol=1e-5)

    def test_compare_spectra_range(self):
        sim = compare_frequency_spectra(_img(), _img())
        assert 0.0 <= sim <= 1.0
