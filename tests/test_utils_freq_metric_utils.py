"""Tests for puzzle_reconstruction.utils.freq_metric_utils."""
import pytest
import numpy as np

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


np.random.seed(42)


# ── BandEnergyRecord ──────────────────────────────────────────────────────────

def test_band_energy_record_n_bands():
    rec = BandEnergyRecord(fragment_id=0, band_energies=[1.0, 2.0, 3.0])
    assert rec.n_bands == 3


def test_band_energy_record_dominant_band():
    rec = BandEnergyRecord(fragment_id=1, band_energies=[0.5, 3.0, 1.0])
    assert rec.dominant_band == 1


def test_band_energy_record_total_energy():
    rec = BandEnergyRecord(fragment_id=2, band_energies=[1.0, 2.0, 3.0])
    assert rec.total_energy == pytest.approx(6.0)


def test_band_energy_record_normalized_energies():
    rec = BandEnergyRecord(fragment_id=3, band_energies=[2.0, 2.0, 6.0])
    norms = rec.normalized_energies
    assert len(norms) == 3
    assert sum(norms) == pytest.approx(1.0)
    assert norms[2] == pytest.approx(0.6)


def test_band_energy_record_zero_total():
    rec = BandEnergyRecord(fragment_id=4, band_energies=[0.0, 0.0])
    assert rec.normalized_energies == [0.0, 0.0]


def test_make_band_energy_record():
    rec = make_band_energy_record(5, [1.0, 2.0])
    assert isinstance(rec, BandEnergyRecord)
    assert rec.fragment_id == 5
    assert rec.band_energies == [1.0, 2.0]


# ── SpectrumComparisonRecord ──────────────────────────────────────────────────

def test_spectrum_comparison_is_match_true():
    rec = SpectrumComparisonRecord(fragment_id_a=0, fragment_id_b=1, similarity=0.7)
    assert rec.is_match is True


def test_spectrum_comparison_is_match_false():
    rec = SpectrumComparisonRecord(fragment_id_a=0, fragment_id_b=1, similarity=0.3)
    assert rec.is_match is False


def test_spectrum_comparison_invalid_similarity():
    with pytest.raises(ValueError):
        SpectrumComparisonRecord(fragment_id_a=0, fragment_id_b=1, similarity=1.5)


def test_spectrum_comparison_boundary_similarity():
    rec = SpectrumComparisonRecord(fragment_id_a=0, fragment_id_b=1, similarity=0.5)
    assert rec.is_match is True


# ── FreqBatchSummary ──────────────────────────────────────────────────────────

def test_freq_batch_summary_is_valid_true():
    s = FreqBatchSummary(n_fragments=5, mean_entropy=3.5, mean_centroid=0.4, n_bands=8)
    assert s.is_valid is True


def test_freq_batch_summary_is_valid_false_fragments():
    s = FreqBatchSummary(n_fragments=0, mean_entropy=3.5, mean_centroid=0.4, n_bands=8)
    assert s.is_valid is False


def test_freq_batch_summary_is_valid_false_bands():
    s = FreqBatchSummary(n_fragments=5, mean_entropy=3.5, mean_centroid=0.4, n_bands=0)
    assert s.is_valid is False


# ── MetricSnapshot ────────────────────────────────────────────────────────────

def test_metric_snapshot_metric_names():
    snap = MetricSnapshot(step=0, values={"loss": 0.5, "acc": 0.9})
    assert set(snap.metric_names) == {"loss", "acc"}


def test_metric_snapshot_n_metrics():
    snap = MetricSnapshot(step=1, values={"loss": 0.5})
    assert snap.n_metrics == 1


def test_metric_snapshot_get_existing():
    snap = MetricSnapshot(step=2, values={"loss": 0.5})
    assert snap.get("loss") == pytest.approx(0.5)


def test_metric_snapshot_get_default():
    snap = MetricSnapshot(step=3, values={})
    assert snap.get("missing", 99.0) == pytest.approx(99.0)


def test_metric_snapshot_invalid_step():
    with pytest.raises(ValueError):
        MetricSnapshot(step=-1, values={})


def test_make_metric_snapshot():
    snap = make_metric_snapshot(10, {"acc": 0.8}, label="val")
    assert snap.step == 10
    assert snap.label == "val"
    assert snap.values["acc"] == pytest.approx(0.8)


# ── MetricRunSummary ──────────────────────────────────────────────────────────

def test_metric_run_summary_best():
    summary = MetricRunSummary(
        namespace="train",
        total_steps=100,
        best_values={"loss": 0.1},
        final_values={"loss": 0.2},
    )
    assert summary.best("loss") == pytest.approx(0.1)
    assert summary.best("missing") is None


def test_metric_run_summary_final():
    summary = MetricRunSummary(
        namespace="val",
        total_steps=50,
        final_values={"acc": 0.95},
    )
    assert summary.final("acc") == pytest.approx(0.95)


def test_metric_run_summary_tracked_metrics():
    summary = MetricRunSummary(
        namespace="test",
        total_steps=20,
        final_values={"a": 1.0, "b": 2.0},
    )
    assert set(summary.tracked_metrics) == {"a", "b"}


# ── MovingAverageResult ───────────────────────────────────────────────────────

def test_moving_average_result_length():
    mar = MovingAverageResult(metric_name="loss", window=3, smoothed=[0.5, 0.4, 0.3])
    assert mar.length == 3


def test_moving_average_result_at():
    mar = MovingAverageResult(metric_name="loss", window=3, smoothed=[0.5, 0.4, 0.3])
    assert mar.at(1) == pytest.approx(0.4)


def test_moving_average_result_invalid_window():
    with pytest.raises(ValueError):
        MovingAverageResult(metric_name="loss", window=0, smoothed=[])


# ── GreedyStepRecord / AssemblyRunRecord ──────────────────────────────────────

def test_greedy_step_record_basic():
    step = make_greedy_step(0, fragment_id=1, anchor_id=2, score=0.9)
    assert step.step == 0
    assert step.fragment_id == 1
    assert step.score == pytest.approx(0.9)


def test_greedy_step_record_invalid_score():
    with pytest.raises(ValueError):
        GreedyStepRecord(step=0, fragment_id=0, anchor_id=1, score=-0.1)


def test_assembly_run_record_n_placed():
    steps = [make_greedy_step(i, i, 0, float(i)) for i in range(5)]
    run = AssemblyRunRecord(n_fragments=10, steps=steps)
    assert run.n_placed == 5


def test_assembly_run_record_placement_rate():
    steps = [make_greedy_step(i, i, 0, 1.0) for i in range(4)]
    run = AssemblyRunRecord(n_fragments=8, steps=steps)
    assert run.placement_rate == pytest.approx(0.5)


def test_assembly_run_record_zero_fragments():
    run = AssemblyRunRecord(n_fragments=0)
    assert run.placement_rate == pytest.approx(0.0)
