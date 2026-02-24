"""Extra tests for puzzle_reconstruction/utils/freq_metric_utils.py."""
from __future__ import annotations

import pytest

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


# ─── BandEnergyRecord ─────────────────────────────────────────────────────────

class TestBandEnergyRecordExtra:
    def test_n_bands_set(self):
        r = BandEnergyRecord(fragment_id=0, band_energies=[1.0, 2.0, 3.0])
        assert r.n_bands == 3

    def test_dominant_band(self):
        r = BandEnergyRecord(fragment_id=0, band_energies=[1.0, 5.0, 2.0])
        assert r.dominant_band == 1

    def test_total_energy(self):
        r = BandEnergyRecord(fragment_id=0, band_energies=[1.0, 2.0, 3.0])
        assert r.total_energy == pytest.approx(6.0)

    def test_normalized_energies_sum_one(self):
        r = BandEnergyRecord(fragment_id=0, band_energies=[1.0, 3.0])
        ne = r.normalized_energies
        assert sum(ne) == pytest.approx(1.0)

    def test_normalized_zero_energy(self):
        r = BandEnergyRecord(fragment_id=0, band_energies=[0.0, 0.0])
        ne = r.normalized_energies
        assert all(v == pytest.approx(0.0) for v in ne)

    def test_empty_bands_dominant(self):
        r = BandEnergyRecord(fragment_id=0, band_energies=[])
        assert r.dominant_band == 0


# ─── SpectrumComparisonRecord ─────────────────────────────────────────────────

class TestSpectrumComparisonRecordExtra:
    def test_stores_similarity(self):
        r = SpectrumComparisonRecord(fragment_id_a=0, fragment_id_b=1, similarity=0.7)
        assert r.similarity == pytest.approx(0.7)

    def test_similarity_out_of_range_raises(self):
        with pytest.raises(ValueError):
            SpectrumComparisonRecord(fragment_id_a=0, fragment_id_b=1, similarity=1.5)

    def test_similarity_negative_raises(self):
        with pytest.raises(ValueError):
            SpectrumComparisonRecord(fragment_id_a=0, fragment_id_b=1, similarity=-0.1)

    def test_is_match_true(self):
        r = SpectrumComparisonRecord(fragment_id_a=0, fragment_id_b=1, similarity=0.6)
        assert r.is_match is True

    def test_is_match_false(self):
        r = SpectrumComparisonRecord(fragment_id_a=0, fragment_id_b=1, similarity=0.4)
        assert r.is_match is False


# ─── FreqBatchSummary ─────────────────────────────────────────────────────────

class TestFreqBatchSummaryExtra:
    def test_is_valid_true(self):
        s = FreqBatchSummary(n_fragments=5, mean_entropy=1.0, mean_centroid=0.5, n_bands=4)
        assert s.is_valid is True

    def test_is_valid_false_zero_fragments(self):
        s = FreqBatchSummary(n_fragments=0, mean_entropy=1.0, mean_centroid=0.5, n_bands=4)
        assert s.is_valid is False

    def test_is_valid_false_zero_bands(self):
        s = FreqBatchSummary(n_fragments=5, mean_entropy=1.0, mean_centroid=0.5, n_bands=0)
        assert s.is_valid is False


# ─── MetricSnapshot ───────────────────────────────────────────────────────────

class TestMetricSnapshotExtra:
    def test_stores_step(self):
        s = MetricSnapshot(step=5, values={"loss": 0.3})
        assert s.step == 5

    def test_negative_step_raises(self):
        with pytest.raises(ValueError):
            MetricSnapshot(step=-1, values={})

    def test_metric_names(self):
        s = MetricSnapshot(step=0, values={"a": 1.0, "b": 2.0})
        assert set(s.metric_names) == {"a", "b"}

    def test_n_metrics(self):
        s = MetricSnapshot(step=0, values={"x": 1.0, "y": 2.0, "z": 3.0})
        assert s.n_metrics == 3

    def test_get_existing(self):
        s = MetricSnapshot(step=0, values={"loss": 0.5})
        assert s.get("loss") == pytest.approx(0.5)

    def test_get_missing_default(self):
        s = MetricSnapshot(step=0, values={})
        assert s.get("missing", 99.0) == pytest.approx(99.0)


# ─── MetricRunSummary ─────────────────────────────────────────────────────────

class TestMetricRunSummaryExtra:
    def test_stores_namespace(self):
        s = MetricRunSummary(namespace="train", total_steps=10)
        assert s.namespace == "train"

    def test_best_returns_value(self):
        s = MetricRunSummary(namespace="n", total_steps=5,
                              best_values={"acc": 0.95})
        assert s.best("acc") == pytest.approx(0.95)

    def test_best_missing_is_none(self):
        s = MetricRunSummary(namespace="n", total_steps=5)
        assert s.best("nonexistent") is None

    def test_final_returns_value(self):
        s = MetricRunSummary(namespace="n", total_steps=5,
                              final_values={"loss": 0.1})
        assert s.final("loss") == pytest.approx(0.1)

    def test_tracked_metrics(self):
        s = MetricRunSummary(namespace="n", total_steps=5,
                              final_values={"a": 1.0, "b": 2.0})
        assert set(s.tracked_metrics) == {"a", "b"}


# ─── MovingAverageResult ──────────────────────────────────────────────────────

class TestMovingAverageResultExtra:
    def test_stores_metric_name(self):
        r = MovingAverageResult(metric_name="loss", window=3, smoothed=[1.0, 2.0])
        assert r.metric_name == "loss"

    def test_window_lt_one_raises(self):
        with pytest.raises(ValueError):
            MovingAverageResult(metric_name="x", window=0, smoothed=[])

    def test_length_property(self):
        r = MovingAverageResult(metric_name="x", window=2, smoothed=[1.0, 2.0, 3.0])
        assert r.length == 3

    def test_at_index(self):
        r = MovingAverageResult(metric_name="x", window=2, smoothed=[5.0, 6.0])
        assert r.at(0) == pytest.approx(5.0)


# ─── GreedyStepRecord ─────────────────────────────────────────────────────────

class TestGreedyStepRecordExtra:
    def test_stores_step(self):
        r = GreedyStepRecord(step=2, fragment_id=5, anchor_id=3, score=0.9)
        assert r.step == 2

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            GreedyStepRecord(step=0, fragment_id=0, anchor_id=1, score=-0.1)

    def test_stores_position(self):
        r = GreedyStepRecord(step=0, fragment_id=0, anchor_id=1, score=0.5,
                              position=(10.0, 20.0))
        assert r.position == (pytest.approx(10.0), pytest.approx(20.0))


# ─── AssemblyRunRecord ────────────────────────────────────────────────────────

class TestAssemblyRunRecordExtra:
    def test_n_placed_empty(self):
        r = AssemblyRunRecord(n_fragments=5)
        assert r.n_placed == 0

    def test_n_placed_with_steps(self):
        steps = [GreedyStepRecord(step=i, fragment_id=i, anchor_id=0, score=0.5)
                 for i in range(3)]
        r = AssemblyRunRecord(n_fragments=5, steps=steps)
        assert r.n_placed == 3

    def test_placement_rate(self):
        steps = [GreedyStepRecord(step=i, fragment_id=i, anchor_id=0, score=0.5)
                 for i in range(4)]
        r = AssemblyRunRecord(n_fragments=8, steps=steps)
        assert r.placement_rate == pytest.approx(0.5)

    def test_placement_rate_zero_fragments(self):
        r = AssemblyRunRecord(n_fragments=0)
        assert r.placement_rate == pytest.approx(0.0)


# ─── make_* helpers ───────────────────────────────────────────────────────────

class TestMakeHelpersExtra:
    def test_make_band_energy_record(self):
        r = make_band_energy_record(3, [1.0, 2.0])
        assert isinstance(r, BandEnergyRecord) and r.fragment_id == 3

    def test_make_metric_snapshot(self):
        s = make_metric_snapshot(5, {"loss": 0.1})
        assert isinstance(s, MetricSnapshot) and s.step == 5

    def test_make_greedy_step(self):
        g = make_greedy_step(0, 1, 2, 0.7)
        assert isinstance(g, GreedyStepRecord) and g.score == pytest.approx(0.7)
