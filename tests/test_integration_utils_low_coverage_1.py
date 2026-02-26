"""Integration tests for 13 under-tested utility modules in puzzle_reconstruction.

Covers:
    - alignment_utils
    - annealing_schedule
    - annealing_score_utils
    - assembly_config_utils
    - assembly_score_utils
    - blend_utils
    - candidate_rank_utils
    - canvas_build_utils
    - color_edge_export_utils
    - color_hist_utils
    - config_utils
    - consensus_score_utils
    - contour_profile_utils
"""
from __future__ import annotations

import math
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── Module imports ────────────────────────────────────────────────────────────

from puzzle_reconstruction.utils.alignment_utils import (
    AlignmentConfig,
    AlignmentResult,
    normalize_for_alignment,
    find_best_rotation,
    find_best_translation,
    compute_alignment_error,
    align_curves_procrustes,
    align_curves_icp,
    alignment_score,
    batch_align_curves,
)
from puzzle_reconstruction.utils.annealing_schedule import (
    ScheduleConfig,
    TemperatureRecord,
    linear_schedule,
    geometric_schedule,
    exponential_schedule,
    cosine_schedule,
    stepped_schedule,
    get_temperature,
    estimate_steps,
    batch_temperatures,
)
from puzzle_reconstruction.utils.annealing_score_utils import (
    AnnealingScoreConfig,
    AnnealingScoreEntry,
    AnnealingSummary,
    make_annealing_entry,
    entries_from_log,
    summarise_annealing,
    filter_accepted,
    filter_rejected,
    filter_by_min_score,
    filter_by_temperature_range,
    top_k_entries,
    annealing_score_stats,
    best_entry,
    compare_summaries,
    batch_summarise,
)
from puzzle_reconstruction.utils.assembly_config_utils import (
    AssemblyStateRecord,
    AssemblyStateHistory,
    ConfigChangeRecord,
    ConfigChangeLog,
    CandidateFilterRecord,
    FilterPipelineSummary,
    summarize_assembly_history,
    build_filter_pipeline_summary,
    build_config_change_log,
)
from puzzle_reconstruction.utils.assembly_score_utils import (
    AssemblyScoreConfig,
    AssemblyScoreEntry,
    AssemblySummary,
    make_assembly_entry,
    entries_from_assemblies,
    summarise_assemblies,
    filter_good_assemblies,
    filter_poor_assemblies,
    filter_by_method,
    filter_by_score_range,
    filter_by_min_fragments,
    top_k_assembly_entries,
    best_assembly_entry,
    assembly_score_stats,
    compare_assembly_summaries,
    batch_summarise_assemblies,
)
from puzzle_reconstruction.utils.blend_utils import (
    BlendConfig,
    alpha_blend,
    weighted_blend,
    feather_mask,
    paste_with_mask,
    horizontal_blend,
    vertical_blend,
    batch_blend,
)
from puzzle_reconstruction.utils.candidate_rank_utils import (
    CandidateRankConfig,
    CandidateRankEntry,
    CandidateRankSummary,
    make_candidate_entry,
    entries_from_pairs,
    summarise_rankings,
    filter_selected,
    filter_rejected_candidates,
    filter_by_score_range as crank_filter_by_score_range,
    filter_by_rank,
    top_k_candidate_entries,
    candidate_rank_stats,
    compare_rankings,
    batch_summarise_rankings,
)
from puzzle_reconstruction.utils.canvas_build_utils import (
    CanvasBuildConfig,
    PlacementEntry,
    CanvasBuildSummary,
    make_placement_entry,
    entries_from_placements,
    summarise_canvas_build,
    filter_by_area,
    filter_by_coverage_contribution,
    top_k_by_coverage,
    canvas_build_stats,
    compare_canvas_summaries,
    batch_summarise_canvas_builds,
)
from puzzle_reconstruction.utils.color_edge_export_utils import (
    ColorMatchAnalysisConfig,
    ColorMatchAnalysisEntry,
    ColorMatchAnalysisSummary,
    make_color_match_analysis_entry,
    summarise_color_match_analysis,
    filter_strong_color_matches,
    filter_weak_color_matches,
    filter_color_by_method,
    top_k_color_match_entries,
    best_color_match_entry,
    color_match_analysis_stats,
    compare_color_match_summaries,
    batch_summarise_color_match_analysis,
    EdgeDetectionAnalysisConfig,
    EdgeDetectionAnalysisEntry,
    EdgeDetectionAnalysisSummary,
    make_edge_detection_entry,
    summarise_edge_detection_entries,
    filter_edge_by_min_density,
    filter_edge_by_method,
    top_k_edge_density_entries,
    best_edge_density_entry,
    edge_detection_stats,
    compare_edge_detection_summaries,
    batch_summarise_edge_detection_entries,
)
from puzzle_reconstruction.utils.color_hist_utils import (
    ColorHistConfig,
    ColorHistEntry,
    ColorHistSummary,
    make_color_hist_entry,
    entries_from_comparisons,
    summarise_color_hist,
    filter_good_hist_entries,
    filter_poor_hist_entries,
    filter_by_intersection_range,
    filter_by_chi2_range,
    filter_by_space,
    top_k_hist_entries,
    best_hist_entry,
    color_hist_stats,
    compare_hist_summaries,
    batch_summarise_color_hist,
)
from puzzle_reconstruction.utils.config_utils import (
    validate_section,
    validate_range,
    merge_dicts,
    flatten_dict,
    unflatten_dict,
    overrides_from_env,
    ConfigProfile,
    PROFILES,
    apply_profile,
    list_profiles,
    load_json_config,
    save_json_config,
    diff_configs,
)
from puzzle_reconstruction.utils.consensus_score_utils import (
    ConsensusScoreConfig,
    ConsensusScoreEntry,
    ConsensusSummary,
    make_consensus_entry,
    entries_from_votes,
    summarise_consensus,
    filter_consensus_pairs,
    filter_non_consensus,
    filter_by_vote_fraction,
    top_k_consensus_entries,
    consensus_score_stats,
    agreement_score,
    compare_consensus,
    batch_summarise_consensus,
)
from puzzle_reconstruction.utils.contour_profile_utils import (
    ProfileConfig,
    ProfileMatchResult,
    sample_profile_along_contour,
    contour_curvature,
    smooth_profile,
    normalize_profile,
    profile_l2_distance,
    profile_cosine_similarity,
    best_cyclic_offset,
    align_profiles,
    match_profiles,
    batch_match_profiles,
    top_k_profile_matches,
)


RNG = np.random.default_rng(42)


# =============================================================================
# alignment_utils
# =============================================================================

class TestAlignmentConfig:
    def test_default_construction(self):
        cfg = AlignmentConfig()
        assert cfg.n_samples == 64
        assert cfg.max_icp_iter == 50
        assert cfg.icp_tol == 1e-6
        assert cfg.allow_reflection is False

    def test_custom_construction(self):
        cfg = AlignmentConfig(n_samples=32, max_icp_iter=10, icp_tol=1e-4)
        assert cfg.n_samples == 32

    def test_invalid_n_samples(self):
        with pytest.raises(ValueError):
            AlignmentConfig(n_samples=1)

    def test_invalid_max_icp_iter(self):
        with pytest.raises(ValueError):
            AlignmentConfig(max_icp_iter=0)

    def test_invalid_icp_tol(self):
        with pytest.raises(ValueError):
            AlignmentConfig(icp_tol=0.0)


class TestAlignmentFunctions:
    def _circle(self, n=20):
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return np.stack([np.cos(t), np.sin(t)], axis=1)

    def test_normalize_for_alignment_zero_mean(self):
        pts = RNG.random((30, 2)) * 100
        norm, centroid, scale = normalize_for_alignment(pts)
        assert np.allclose(norm.mean(axis=0), 0.0, atol=1e-10)

    def test_normalize_for_alignment_unit_rms(self):
        pts = RNG.random((30, 2)) * 100
        norm, centroid, scale = normalize_for_alignment(pts)
        rms = float(np.sqrt((norm ** 2).sum() / len(norm)))
        assert abs(rms - 1.0) < 1e-6

    def test_normalize_for_alignment_constant(self):
        pts = np.ones((10, 2)) * 5.0
        norm, centroid, scale = normalize_for_alignment(pts)
        assert scale == 1.0

    def test_find_best_rotation_identity(self):
        src = self._circle(20)
        angle, R = find_best_rotation(src, src)
        assert abs(angle) < 1e-6

    def test_find_best_translation(self):
        src = RNG.random((20, 2))
        tgt = src + np.array([3.0, -2.0])
        t = find_best_translation(src, tgt)
        assert np.allclose(t, [3.0, -2.0], atol=1e-10)

    def test_compute_alignment_error_zero(self):
        pts = RNG.random((15, 2))
        assert compute_alignment_error(pts, pts) == 0.0

    def test_compute_alignment_error_different_shapes(self):
        a = RNG.random((10, 2))
        b = RNG.random((5, 2))
        assert compute_alignment_error(a, b) == float("inf")

    def test_align_curves_procrustes_returns_result(self):
        src = self._circle(30)
        tgt = src + 0.01
        result = align_curves_procrustes(src, tgt)
        assert isinstance(result, AlignmentResult)
        assert result.converged is True
        assert result.error >= 0.0

    def test_align_curves_procrustes_custom_cfg(self):
        src = self._circle(30)
        tgt = self._circle(30)
        cfg = AlignmentConfig(n_samples=16)
        result = align_curves_procrustes(src, tgt, cfg)
        assert result.aligned.shape[1] == 2

    def test_align_curves_icp_returns_result(self):
        src = self._circle(20)
        tgt = src.copy()
        result = align_curves_icp(src, tgt)
        assert isinstance(result, AlignmentResult)
        assert result.scale == 1.0

    def test_alignment_score_range(self):
        src = self._circle(20)
        result = align_curves_procrustes(src, src)
        score = alignment_score(result, sigma=1.0)
        assert 0.0 <= score <= 1.0

    def test_alignment_score_invalid_sigma(self):
        src = self._circle(20)
        result = align_curves_procrustes(src, src)
        with pytest.raises(ValueError):
            alignment_score(result, sigma=0.0)

    def test_batch_align_curves_procrustes(self):
        sources = [self._circle(20) for _ in range(3)]
        targets = [s + 0.1 for s in sources]
        results = batch_align_curves(sources, targets, method="procrustes")
        assert len(results) == 3

    def test_batch_align_curves_icp(self):
        sources = [self._circle(20) for _ in range(2)]
        targets = [s.copy() for s in sources]
        results = batch_align_curves(sources, targets, method="icp")
        assert len(results) == 2

    def test_batch_align_curves_invalid_method(self):
        with pytest.raises(ValueError):
            batch_align_curves([self._circle(10)], [self._circle(10)], method="bad")

    def test_batch_align_curves_length_mismatch(self):
        with pytest.raises(ValueError):
            batch_align_curves([self._circle(10)], [], method="procrustes")

    def test_alignment_result_to_dict(self):
        src = self._circle(20)
        result = align_curves_procrustes(src, src)
        d = result.to_dict()
        assert "rotation" in d
        assert "translation" in d
        assert "scale" in d
        assert "error" in d


# =============================================================================
# annealing_schedule
# =============================================================================

class TestScheduleConfig:
    def test_default_values(self):
        cfg = ScheduleConfig()
        assert cfg.t_start == 1.0
        assert cfg.t_end == 1e-3
        assert cfg.n_steps == 1000
        assert cfg.kind == "geometric"

    def test_invalid_t_start(self):
        with pytest.raises(ValueError):
            ScheduleConfig(t_start=0.0)

    def test_invalid_t_end(self):
        with pytest.raises(ValueError):
            ScheduleConfig(t_end=0.0)

    def test_t_end_ge_t_start(self):
        with pytest.raises(ValueError):
            ScheduleConfig(t_start=1.0, t_end=2.0)

    def test_invalid_n_steps(self):
        with pytest.raises(ValueError):
            ScheduleConfig(n_steps=0)

    def test_invalid_kind(self):
        with pytest.raises(ValueError):
            ScheduleConfig(kind="quadratic")

    def test_cooling_rate(self):
        cfg = ScheduleConfig(t_start=1.0, t_end=0.001, n_steps=100)
        alpha = cfg.cooling_rate
        assert 0.0 < alpha < 1.0


class TestScheduleFunctions:
    def _cfg(self, kind="geometric", n=50):
        return ScheduleConfig(t_start=10.0, t_end=0.01, n_steps=n, kind=kind)

    def test_linear_schedule_length(self):
        records = linear_schedule(self._cfg("linear"))
        assert len(records) == 50

    def test_linear_schedule_monotone(self):
        records = linear_schedule(self._cfg("linear"))
        temps = [r.temperature for r in records]
        assert all(temps[i] >= temps[i+1] for i in range(len(temps)-1))

    def test_linear_schedule_bounds(self):
        records = linear_schedule(self._cfg("linear"))
        assert records[0].temperature == pytest.approx(10.0)
        assert records[-1].temperature == pytest.approx(0.01)

    def test_geometric_schedule_length(self):
        records = geometric_schedule(self._cfg("geometric"))
        assert len(records) == 50

    def test_geometric_schedule_monotone(self):
        records = geometric_schedule(self._cfg("geometric"))
        temps = [r.temperature for r in records]
        assert all(temps[i] >= temps[i+1] for i in range(len(temps)-1))

    def test_exponential_schedule_length(self):
        records = exponential_schedule(self._cfg("exponential"))
        assert len(records) == 50

    def test_cosine_schedule_length(self):
        records = cosine_schedule(self._cfg("cosine"))
        assert len(records) == 50

    def test_cosine_schedule_starts_high(self):
        records = cosine_schedule(self._cfg("cosine"))
        assert records[0].temperature >= records[-1].temperature

    def test_stepped_schedule_length(self):
        cfg = ScheduleConfig(t_start=10.0, t_end=0.01, n_steps=20, kind="stepped", step_size=5)
        records = stepped_schedule(cfg)
        assert len(records) == 20

    def test_get_temperature_step_0(self):
        cfg = self._cfg("geometric")
        t0 = get_temperature(0, cfg)
        assert t0 == pytest.approx(10.0)

    def test_get_temperature_invalid_step(self):
        cfg = self._cfg()
        with pytest.raises(ValueError):
            get_temperature(50, cfg)

    def test_get_temperature_negative_step(self):
        cfg = self._cfg()
        with pytest.raises(ValueError):
            get_temperature(-1, cfg)

    def test_estimate_steps_basic(self):
        n = estimate_steps(t_start=100.0, t_target=0.01, alpha=0.95)
        assert n >= 1

    def test_estimate_steps_invalid_alpha(self):
        with pytest.raises(ValueError):
            estimate_steps(100.0, 0.01, alpha=1.0)

    def test_estimate_steps_invalid_t_start(self):
        with pytest.raises(ValueError):
            estimate_steps(0.0, 0.01, alpha=0.9)

    def test_batch_temperatures_basic(self):
        cfg = self._cfg()
        steps = np.array([0, 10, 25, 49])
        temps = batch_temperatures(steps, cfg)
        assert temps.shape == (4,)
        assert all(t > 0 for t in temps)

    def test_batch_temperatures_empty(self):
        cfg = self._cfg()
        temps = batch_temperatures(np.array([], dtype=int), cfg)
        assert temps.shape == (0,)

    def test_batch_temperatures_out_of_bounds(self):
        cfg = self._cfg()
        with pytest.raises(ValueError):
            batch_temperatures(np.array([0, 100]), cfg)

    def test_temperature_record_invalid_step(self):
        with pytest.raises(ValueError):
            TemperatureRecord(step=-1, temperature=1.0, progress=0.0)

    def test_temperature_record_invalid_temp(self):
        with pytest.raises(ValueError):
            TemperatureRecord(step=0, temperature=0.0, progress=0.0)


# =============================================================================
# annealing_score_utils
# =============================================================================

class TestAnnealingScoreUtils:
    def _entries(self, n=10):
        entries = []
        for i in range(n):
            entries.append(make_annealing_entry(
                iteration=i,
                temperature=1.0 - i * 0.1,
                current_score=float(i) / n,
                best_score=float(i) / n,
                accepted=(i % 2 == 0),
            ))
        return entries

    def test_config_defaults(self):
        cfg = AnnealingScoreConfig()
        assert cfg.convergence_window == 10

    def test_config_invalid_window(self):
        with pytest.raises(ValueError):
            AnnealingScoreConfig(convergence_window=0)

    def test_config_invalid_threshold(self):
        with pytest.raises(ValueError):
            AnnealingScoreConfig(improvement_threshold=-0.1)

    def test_make_annealing_entry(self):
        e = make_annealing_entry(0, 1.0, 0.5, 0.5, True)
        assert e.iteration == 0
        assert e.accepted is True

    def test_entry_invalid_iteration(self):
        with pytest.raises(ValueError):
            AnnealingScoreEntry(iteration=-1, temperature=1.0, current_score=0.5, best_score=0.5, accepted=True)

    def test_entries_from_log(self):
        log = [
            {"iteration": i, "temperature": 1.0, "current_score": 0.8, "best_score": 0.8, "accepted": True}
            for i in range(5)
        ]
        entries = entries_from_log(log)
        assert len(entries) == 5

    def test_summarise_annealing_basic(self):
        entries = self._entries(10)
        summary = summarise_annealing(entries)
        assert summary.n_iterations == 10
        assert summary.n_accepted == 5

    def test_summarise_annealing_empty(self):
        summary = summarise_annealing([])
        assert summary.n_iterations == 0
        assert summary.converged is False

    def test_filter_accepted(self):
        entries = self._entries(10)
        accepted = filter_accepted(entries)
        assert all(e.accepted for e in accepted)

    def test_filter_rejected(self):
        entries = self._entries(10)
        rejected = filter_rejected(entries)
        assert all(not e.accepted for e in rejected)

    def test_filter_by_min_score(self):
        entries = self._entries(10)
        filtered = filter_by_min_score(entries, min_score=0.5)
        assert all(e.current_score >= 0.5 for e in filtered)

    def test_filter_by_temperature_range(self):
        entries = self._entries(10)
        filtered = filter_by_temperature_range(entries, 0.3, 0.7)
        assert all(0.3 <= e.temperature <= 0.7 for e in filtered)

    def test_top_k_entries(self):
        entries = self._entries(10)
        top = top_k_entries(entries, 3)
        assert len(top) == 3
        assert top[0].current_score >= top[1].current_score

    def test_annealing_score_stats_empty(self):
        stats = annealing_score_stats([])
        assert stats["count"] == 0

    def test_annealing_score_stats_nonempty(self):
        entries = self._entries(10)
        stats = annealing_score_stats(entries)
        assert stats["count"] == 10
        assert stats["min"] <= stats["max"]

    def test_best_entry_none(self):
        assert best_entry([]) is None

    def test_best_entry_nonempty(self):
        entries = self._entries(10)
        best = best_entry(entries)
        assert best is not None
        assert best.current_score == max(e.current_score for e in entries)

    def test_compare_summaries(self):
        entries_a = self._entries(10)
        entries_b = self._entries(5)
        sa = summarise_annealing(entries_a)
        sb = summarise_annealing(entries_b)
        cmp = compare_summaries(sa, sb)
        assert "best_score_delta" in cmp

    def test_batch_summarise(self):
        logs = [
            [{"iteration": i, "temperature": 1.0, "current_score": 0.5, "best_score": 0.5, "accepted": True} for i in range(5)]
            for _ in range(3)
        ]
        summaries = batch_summarise(logs)
        assert len(summaries) == 3


# =============================================================================
# assembly_config_utils
# =============================================================================

class TestAssemblyConfigUtils:
    def test_state_record_valid(self):
        r = AssemblyStateRecord(step=0, n_placed=3, n_fragments=10, coverage=0.3)
        assert r.is_complete is False

    def test_state_record_complete(self):
        r = AssemblyStateRecord(step=0, n_placed=10, n_fragments=10, coverage=1.0)
        assert r.is_complete is True

    def test_state_record_invalid_step(self):
        with pytest.raises(ValueError):
            AssemblyStateRecord(step=-1, n_placed=0, n_fragments=5, coverage=0.0)

    def test_state_record_invalid_coverage(self):
        with pytest.raises(ValueError):
            AssemblyStateRecord(step=0, n_placed=0, n_fragments=5, coverage=1.5)

    def test_state_record_invalid_n_fragments(self):
        with pytest.raises(ValueError):
            AssemblyStateRecord(step=0, n_placed=0, n_fragments=0, coverage=0.0)

    def test_history_append_and_n_steps(self):
        history = AssemblyStateHistory()
        for i in range(5):
            history.append(AssemblyStateRecord(step=i, n_placed=i, n_fragments=10, coverage=i/10))
        assert history.n_steps == 5

    def test_history_last_coverage(self):
        history = AssemblyStateHistory()
        history.append(AssemblyStateRecord(step=0, n_placed=3, n_fragments=10, coverage=0.3))
        history.append(AssemblyStateRecord(step=1, n_placed=7, n_fragments=10, coverage=0.7))
        assert history.last_coverage == pytest.approx(0.7)

    def test_history_is_monotone(self):
        history = AssemblyStateHistory()
        for i in range(5):
            history.append(AssemblyStateRecord(step=i, n_placed=i, n_fragments=5, coverage=i/5))
        assert history.is_monotone is True

    def test_history_not_monotone(self):
        history = AssemblyStateHistory()
        history.append(AssemblyStateRecord(step=0, n_placed=3, n_fragments=10, coverage=0.3))
        history.append(AssemblyStateRecord(step=1, n_placed=1, n_fragments=10, coverage=0.1))
        assert history.is_monotone is False

    def test_config_change_record_changed(self):
        r = ConfigChangeRecord(key="method", old_value="sa", new_value="greedy")
        assert r.changed is True

    def test_config_change_record_unchanged(self):
        r = ConfigChangeRecord(key="method", old_value="sa", new_value="sa")
        assert r.changed is False

    def test_config_change_record_invalid_key(self):
        with pytest.raises(ValueError):
            ConfigChangeRecord(key="", old_value="a", new_value="b")

    def test_config_change_log_n_changes(self):
        log = ConfigChangeLog()
        log.append(ConfigChangeRecord("a", 1, 2))
        log.append(ConfigChangeRecord("b", 3, 3))
        assert log.n_changes == 1

    def test_config_change_log_changed_keys(self):
        log = ConfigChangeLog()
        log.append(ConfigChangeRecord("method", "sa", "greedy"))
        log.append(ConfigChangeRecord("lr", 0.1, 0.2))
        keys = log.changed_keys
        assert "method" in keys and "lr" in keys

    def test_candidate_filter_record(self):
        r = CandidateFilterRecord("threshold", n_input=100, n_kept=60, n_removed=40, threshold=0.5)
        assert r.keep_ratio == pytest.approx(0.6)

    def test_filter_pipeline_summary(self):
        summary = FilterPipelineSummary()
        summary.add_stage(CandidateFilterRecord("f1", 100, 80, 20))
        summary.add_stage(CandidateFilterRecord("f2", 80, 60, 20))
        assert summary.n_stages == 2
        assert summary.total_removed == 40
        assert summary.final_n_kept == 60

    def test_summarize_assembly_history_empty(self):
        d = summarize_assembly_history(AssemblyStateHistory())
        assert d["n_steps"] == 0

    def test_summarize_assembly_history_nonempty(self):
        history = AssemblyStateHistory()
        for i in range(3):
            history.append(AssemblyStateRecord(step=i, n_placed=i+1, n_fragments=3, coverage=(i+1)/3))
        d = summarize_assembly_history(history)
        assert d["is_complete"] is True

    def test_build_filter_pipeline_summary(self):
        records = [CandidateFilterRecord("f", 10, 8, 2)]
        summary = build_filter_pipeline_summary(records)
        assert summary.n_stages == 1

    def test_build_config_change_log(self):
        diffs = [{"lr": (0.1, 0.01)}, {"method": ("sa", "greedy")}]
        log = build_config_change_log(diffs)
        assert log.n_changes == 2


# =============================================================================
# assembly_score_utils
# =============================================================================

class TestAssemblyScoreUtils:
    def _entries(self, n=5):
        return [
            make_assembly_entry(run_id=i, method="sa", n_fragments=10,
                                total_score=float(i) / n)
            for i in range(n)
        ]

    def test_config_defaults(self):
        cfg = AssemblyScoreConfig()
        assert cfg.min_score == 0.0
        assert cfg.max_entries == 1000

    def test_config_invalid_min_score(self):
        with pytest.raises(ValueError):
            AssemblyScoreConfig(min_score=-1.0)

    def test_config_invalid_max_entries(self):
        with pytest.raises(ValueError):
            AssemblyScoreConfig(max_entries=0)

    def test_entry_is_good(self):
        e = make_assembly_entry(0, "sa", 10, total_score=0.7)
        assert e.is_good is True

    def test_entry_is_poor(self):
        e = make_assembly_entry(0, "sa", 10, total_score=0.3)
        assert e.is_good is False

    def test_entry_score_per_fragment(self):
        e = make_assembly_entry(0, "sa", 4, total_score=2.0)
        assert e.score_per_fragment == pytest.approx(0.5)

    def test_entry_score_per_fragment_zero_fragments(self):
        e = make_assembly_entry(0, "sa", 0, total_score=0.0)
        assert e.score_per_fragment == 0.0

    def test_summarise_assemblies_empty(self):
        s = summarise_assemblies([])
        assert s.n_total == 0

    def test_summarise_assemblies_nonempty(self):
        entries = self._entries(5)
        s = summarise_assemblies(entries)
        assert s.n_total == 5
        assert s.n_good + s.n_poor == 5

    def test_filter_good_assemblies(self):
        entries = self._entries(5)
        good = filter_good_assemblies(entries)
        assert all(e.is_good for e in good)

    def test_filter_poor_assemblies(self):
        entries = self._entries(5)
        poor = filter_poor_assemblies(entries)
        assert all(not e.is_good for e in poor)

    def test_filter_by_method(self):
        entries = self._entries(5)
        entries[0] = make_assembly_entry(0, "greedy", 10, 0.8)
        filtered = filter_by_method(entries, "sa")
        assert all(e.method == "sa" for e in filtered)

    def test_filter_by_score_range(self):
        entries = self._entries(5)
        filtered = filter_by_score_range(entries, 0.3, 0.7)
        assert all(0.3 <= e.total_score <= 0.7 for e in filtered)

    def test_filter_by_min_fragments(self):
        entries = self._entries(3)
        entries[0] = make_assembly_entry(0, "sa", 2, 0.5)
        filtered = filter_by_min_fragments(entries, 5)
        assert all(e.n_fragments >= 5 for e in filtered)

    def test_top_k_assembly_entries(self):
        entries = self._entries(5)
        top = top_k_assembly_entries(entries, 2)
        assert len(top) == 2
        assert top[0].total_score >= top[1].total_score

    def test_top_k_zero(self):
        entries = self._entries(5)
        assert top_k_assembly_entries(entries, 0) == []

    def test_best_assembly_entry_none(self):
        assert best_assembly_entry([]) is None

    def test_best_assembly_entry(self):
        entries = self._entries(5)
        best = best_assembly_entry(entries)
        assert best.total_score == max(e.total_score for e in entries)

    def test_assembly_score_stats_empty(self):
        stats = assembly_score_stats([])
        assert stats["n"] == 0

    def test_assembly_score_stats_nonempty(self):
        entries = self._entries(5)
        stats = assembly_score_stats(entries)
        assert stats["n"] == 5

    def test_compare_assembly_summaries(self):
        s1 = summarise_assemblies(self._entries(5))
        s2 = summarise_assemblies(self._entries(3))
        cmp = compare_assembly_summaries(s1, s2)
        assert "delta_mean_score" in cmp

    def test_batch_summarise_assemblies(self):
        groups = [self._entries(3), self._entries(4)]
        summaries = batch_summarise_assemblies(groups)
        assert len(summaries) == 2

    def test_entries_from_assemblies(self):
        class FakeAsm:
            total_score = 0.8
            placements = {0: None, 1: None}
        entries = entries_from_assemblies([FakeAsm(), FakeAsm()], method="sa")
        assert len(entries) == 2
        assert all(e.rank > 0 for e in entries)


# =============================================================================
# blend_utils
# =============================================================================

class TestBlendUtils:
    def test_blend_config_defaults(self):
        cfg = BlendConfig()
        assert cfg.feather_px == 8
        assert cfg.gamma == 1.0
        assert cfg.clip_output is True

    def test_blend_config_invalid_feather(self):
        with pytest.raises(ValueError):
            BlendConfig(feather_px=-1)

    def test_blend_config_invalid_gamma(self):
        with pytest.raises(ValueError):
            BlendConfig(gamma=0.0)

    def test_alpha_blend_midpoint(self):
        src = np.full((4, 4), 200, dtype=np.uint8)
        dst = np.full((4, 4), 100, dtype=np.uint8)
        result = alpha_blend(src, dst, alpha=0.5)
        assert result.dtype == np.uint8
        assert np.all(result == 150)

    def test_alpha_blend_alpha_one(self):
        src = np.full((4, 4), 255, dtype=np.uint8)
        dst = np.full((4, 4), 0, dtype=np.uint8)
        result = alpha_blend(src, dst, alpha=1.0)
        assert np.all(result == 255)

    def test_alpha_blend_invalid_alpha(self):
        img = np.zeros((4, 4), dtype=np.uint8)
        with pytest.raises(ValueError):
            alpha_blend(img, img, alpha=1.5)

    def test_alpha_blend_shape_mismatch(self):
        a = np.zeros((4, 4), dtype=np.uint8)
        b = np.zeros((5, 5), dtype=np.uint8)
        with pytest.raises(ValueError):
            alpha_blend(a, b, alpha=0.5)

    def test_weighted_blend_equal(self):
        imgs = [np.full((4, 4), v, dtype=np.uint8) for v in [100, 200]]
        result = weighted_blend(imgs)
        assert np.all(result == 150)

    def test_weighted_blend_empty(self):
        with pytest.raises(ValueError):
            weighted_blend([])

    def test_weighted_blend_shape_mismatch(self):
        a = np.zeros((4, 4), dtype=np.uint8)
        b = np.zeros((5, 5), dtype=np.uint8)
        with pytest.raises(ValueError):
            weighted_blend([a, b])

    def test_feather_mask_shape(self):
        mask = feather_mask(20, 30, feather_px=4)
        assert mask.shape == (20, 30)
        assert mask.dtype == np.float32

    def test_feather_mask_values_in_range(self):
        mask = feather_mask(20, 20, feather_px=4)
        assert np.all(mask >= 0) and np.all(mask <= 1)

    def test_feather_mask_center_is_one(self):
        mask = feather_mask(20, 20, feather_px=2)
        assert mask[10, 10] == pytest.approx(1.0)

    def test_feather_mask_invalid_h(self):
        with pytest.raises(ValueError):
            feather_mask(0, 10)

    def test_paste_with_mask_basic(self):
        canvas = np.zeros((10, 10), dtype=np.uint8)
        patch = np.full((4, 4), 200, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.float32)
        result = paste_with_mask(canvas, patch, mask, 2, 2)
        assert result[3, 3] == 200

    def test_horizontal_blend_output_shape(self):
        left = np.full((10, 20), 100, dtype=np.uint8)
        right = np.full((10, 15), 200, dtype=np.uint8)
        result = horizontal_blend(left, right, overlap=5)
        assert result.shape == (10, 30)

    def test_horizontal_blend_no_overlap(self):
        left = np.zeros((5, 10), dtype=np.uint8)
        right = np.zeros((5, 8), dtype=np.uint8)
        result = horizontal_blend(left, right, overlap=0)
        assert result.shape == (5, 18)

    def test_horizontal_blend_height_mismatch(self):
        left = np.zeros((5, 10), dtype=np.uint8)
        right = np.zeros((6, 10), dtype=np.uint8)
        with pytest.raises(ValueError):
            horizontal_blend(left, right, overlap=0)

    def test_vertical_blend_output_shape(self):
        top = np.full((10, 20), 100, dtype=np.uint8)
        bottom = np.full((8, 20), 200, dtype=np.uint8)
        result = vertical_blend(top, bottom, overlap=3)
        assert result.shape == (15, 20)

    def test_vertical_blend_width_mismatch(self):
        top = np.zeros((5, 10), dtype=np.uint8)
        bottom = np.zeros((5, 11), dtype=np.uint8)
        with pytest.raises(ValueError):
            vertical_blend(top, bottom, overlap=0)

    def test_batch_blend(self):
        imgs = [(np.full((4, 4), 100, dtype=np.uint8), np.full((4, 4), 200, dtype=np.uint8))]
        results = batch_blend(imgs, alpha=0.5)
        assert len(results) == 1
        assert np.all(results[0] == 150)

    def test_alpha_blend_color(self):
        src = np.full((4, 4, 3), 200, dtype=np.uint8)
        dst = np.full((4, 4, 3), 100, dtype=np.uint8)
        result = alpha_blend(src, dst, alpha=0.5)
        assert result.shape == (4, 4, 3)


# =============================================================================
# candidate_rank_utils
# =============================================================================

class TestCandidateRankUtils:
    def _entries(self, n=5):
        pairs = [{"idx1": i, "idx2": i+1, "score": float(i) / n} for i in range(n)]
        return entries_from_pairs(pairs)

    def test_config_defaults(self):
        cfg = CandidateRankConfig()
        assert cfg.min_score == 0.5
        assert cfg.deduplicate is True

    def test_config_invalid_min_score(self):
        with pytest.raises(ValueError):
            CandidateRankConfig(min_score=1.5)

    def test_config_invalid_max_pairs(self):
        with pytest.raises(ValueError):
            CandidateRankConfig(max_pairs=-1)

    def test_make_candidate_entry_selected(self):
        cfg = CandidateRankConfig(min_score=0.3)
        e = make_candidate_entry(0, 1, 0.8, 0, cfg)
        assert e.is_selected is True

    def test_make_candidate_entry_not_selected(self):
        cfg = CandidateRankConfig(min_score=0.9)
        e = make_candidate_entry(0, 1, 0.5, 0, cfg)
        assert e.is_selected is False

    def test_entries_from_pairs_sorted_by_score(self):
        pairs = [{"idx1": 0, "idx2": 1, "score": 0.3}, {"idx1": 1, "idx2": 2, "score": 0.9}]
        entries = entries_from_pairs(pairs)
        assert entries[0].score >= entries[1].score

    def test_summarise_rankings_empty(self):
        s = summarise_rankings([])
        assert s.n_total == 0

    def test_summarise_rankings_nonempty(self):
        entries = self._entries(5)
        s = summarise_rankings(entries)
        assert s.n_total == 5

    def test_filter_selected(self):
        entries = self._entries(5)
        sel = filter_selected(entries)
        assert all(e.is_selected for e in sel)

    def test_filter_rejected_candidates(self):
        entries = self._entries(5)
        rej = filter_rejected_candidates(entries)
        assert all(not e.is_selected for e in rej)

    def test_filter_by_score_range(self):
        entries = self._entries(5)
        filtered = crank_filter_by_score_range(entries, 0.2, 0.7)
        assert all(0.2 <= e.score <= 0.7 for e in filtered)

    def test_filter_by_rank(self):
        entries = self._entries(5)
        filtered = filter_by_rank(entries, max_rank=2)
        assert all(e.rank <= 2 for e in filtered)

    def test_top_k_candidate_entries(self):
        entries = self._entries(5)
        top = top_k_candidate_entries(entries, 2)
        assert len(top) == 2

    def test_candidate_rank_stats_empty(self):
        stats = candidate_rank_stats([])
        assert stats["count"] == 0

    def test_candidate_rank_stats_nonempty(self):
        entries = self._entries(5)
        stats = candidate_rank_stats(entries)
        assert stats["count"] == 5

    def test_compare_rankings(self):
        s1 = summarise_rankings(self._entries(5))
        s2 = summarise_rankings(self._entries(3))
        cmp = compare_rankings(s1, s2)
        assert "n_total_delta" in cmp

    def test_batch_summarise_rankings(self):
        batches = [
            [{"idx1": 0, "idx2": 1, "score": 0.8}],
            [{"idx1": 2, "idx2": 3, "score": 0.3}],
        ]
        summaries = batch_summarise_rankings(batches)
        assert len(summaries) == 2

    def test_entry_repr(self):
        e = make_candidate_entry(0, 1, 0.7, 0)
        r = repr(e)
        assert "CandidateRankEntry" in r


# =============================================================================
# canvas_build_utils
# =============================================================================

class TestCanvasBuildUtils:
    def _entries(self, n=3):
        return [make_placement_entry(i, x=i*10, y=0, w=10, h=10, coverage_contribution=0.1*i) for i in range(n)]

    def test_config_defaults(self):
        cfg = CanvasBuildConfig()
        assert cfg.min_coverage == 0.0
        assert cfg.max_fragments == 1000

    def test_config_invalid_coverage(self):
        with pytest.raises(ValueError):
            CanvasBuildConfig(min_coverage=1.5)

    def test_config_invalid_max_fragments(self):
        with pytest.raises(ValueError):
            CanvasBuildConfig(max_fragments=0)

    def test_config_invalid_blend_mode(self):
        with pytest.raises(ValueError):
            CanvasBuildConfig(blend_mode="unknown")

    def test_placement_entry_area(self):
        e = make_placement_entry(0, 5, 5, 10, 20)
        assert e.area == 200

    def test_placement_entry_x2_y2(self):
        e = make_placement_entry(0, 3, 4, 10, 5)
        assert e.x2 == 13
        assert e.y2 == 9

    def test_placement_entry_invalid_id(self):
        with pytest.raises(ValueError):
            PlacementEntry(fragment_id=-1, x=0, y=0, w=10, h=10)

    def test_placement_entry_invalid_dimensions(self):
        with pytest.raises(ValueError):
            PlacementEntry(fragment_id=0, x=0, y=0, w=0, h=10)

    def test_entries_from_placements(self):
        placements = [(0, 0, 0, 10, 10), (1, 10, 0, 10, 10)]
        entries = entries_from_placements(placements)
        assert len(entries) == 2
        assert entries[0].fragment_id == 0

    def test_summarise_canvas_build(self):
        entries = self._entries(3)
        s = summarise_canvas_build(entries, canvas_w=100, canvas_h=100, coverage=0.3)
        assert s.n_placed == 3
        assert s.total_area == sum(e.area for e in entries)

    def test_filter_by_area(self):
        entries = self._entries(3)
        filtered = filter_by_area(entries, min_area=100, max_area=100)
        assert all(e.area == 100 for e in filtered)

    def test_filter_by_coverage_contribution(self):
        entries = self._entries(3)
        filtered = filter_by_coverage_contribution(entries, min_contrib=0.1)
        assert all(e.coverage_contribution >= 0.1 for e in filtered)

    def test_top_k_by_coverage(self):
        entries = self._entries(4)
        top = top_k_by_coverage(entries, k=2)
        assert len(top) == 2
        assert top[0].coverage_contribution >= top[1].coverage_contribution

    def test_top_k_by_coverage_invalid_k(self):
        entries = self._entries(3)
        with pytest.raises(ValueError):
            top_k_by_coverage(entries, k=0)

    def test_canvas_build_stats_empty(self):
        stats = canvas_build_stats([])
        assert stats["n"] == 0

    def test_canvas_build_stats_nonempty(self):
        entries = self._entries(3)
        stats = canvas_build_stats(entries)
        assert stats["n"] == 3
        assert stats["total_area"] == 300

    def test_compare_canvas_summaries(self):
        e1 = self._entries(3)
        e2 = self._entries(2)
        s1 = summarise_canvas_build(e1, 100, 100, 0.3)
        s2 = summarise_canvas_build(e2, 100, 100, 0.2)
        cmp = compare_canvas_summaries(s1, s2)
        assert cmp["n_placed_delta"] == 1

    def test_batch_summarise_canvas_builds(self):
        specs = [
            (self._entries(2), 100, 100, 0.2),
            (self._entries(3), 200, 200, 0.3),
        ]
        summaries = batch_summarise_canvas_builds(specs)
        assert len(summaries) == 2

    def test_summary_repr(self):
        entries = self._entries(2)
        s = summarise_canvas_build(entries, 100, 100, 0.2)
        r = repr(s)
        assert "CanvasBuildSummary" in r


# =============================================================================
# color_edge_export_utils
# =============================================================================

class TestColorEdgeExportUtils:
    def _color_entries(self, n=5):
        return [
            make_color_match_analysis_entry(i, i+1, score=float(i)/n,
                                            hist_score=0.5, moment_score=0.5,
                                            profile_score=0.5)
            for i in range(n)
        ]

    def _edge_entries(self, n=5):
        return [
            make_edge_detection_entry(i, density=float(i)/n, n_contours=i+1)
            for i in range(n)
        ]

    def test_color_match_config_defaults(self):
        cfg = ColorMatchAnalysisConfig()
        assert cfg.min_score == 0.0
        assert cfg.colorspace == "hsv"

    def test_make_color_match_entry(self):
        e = make_color_match_analysis_entry(0, 1, 0.8, 0.7, 0.9, 0.8)
        assert e.score == pytest.approx(0.8)

    def test_summarise_color_match_empty(self):
        s = summarise_color_match_analysis([])
        assert s.n_entries == 0

    def test_summarise_color_match_nonempty(self):
        entries = self._color_entries(5)
        s = summarise_color_match_analysis(entries)
        assert s.n_entries == 5

    def test_filter_strong_color_matches(self):
        entries = self._color_entries(5)
        strong = filter_strong_color_matches(entries, threshold=0.5)
        assert all(e.score >= 0.5 for e in strong)

    def test_filter_weak_color_matches(self):
        entries = self._color_entries(5)
        weak = filter_weak_color_matches(entries, threshold=0.5)
        assert all(e.score < 0.5 for e in weak)

    def test_filter_color_by_method(self):
        entries = self._color_entries(3)
        entries[0] = make_color_match_analysis_entry(0, 1, 0.8, 0.7, 0.9, 0.8, method="bgr")
        filtered = filter_color_by_method(entries, "hsv")
        assert all(e.method == "hsv" for e in filtered)

    def test_top_k_color_match_entries(self):
        entries = self._color_entries(5)
        top = top_k_color_match_entries(entries, k=2)
        assert len(top) == 2

    def test_best_color_match_entry_none(self):
        assert best_color_match_entry([]) is None

    def test_best_color_match_entry(self):
        entries = self._color_entries(5)
        best = best_color_match_entry(entries)
        assert best.score == max(e.score for e in entries)

    def test_color_match_stats_empty(self):
        stats = color_match_analysis_stats([])
        assert stats["count"] == 0

    def test_color_match_stats_nonempty(self):
        entries = self._color_entries(3)
        stats = color_match_analysis_stats(entries)
        assert stats["count"] == 3.0

    def test_compare_color_match_summaries(self):
        s1 = summarise_color_match_analysis(self._color_entries(5))
        s2 = summarise_color_match_analysis(self._color_entries(3))
        cmp = compare_color_match_summaries(s1, s2)
        assert "mean_score_delta" in cmp

    def test_batch_summarise_color_match(self):
        groups = [self._color_entries(3), self._color_entries(4)]
        summaries = batch_summarise_color_match_analysis(groups)
        assert len(summaries) == 2

    def test_edge_detection_config_defaults(self):
        cfg = EdgeDetectionAnalysisConfig()
        assert cfg.method == "canny"

    def test_make_edge_detection_entry(self):
        e = make_edge_detection_entry(0, density=0.5, n_contours=3)
        assert e.fragment_id == 0
        assert e.density == 0.5

    def test_summarise_edge_detection_empty(self):
        s = summarise_edge_detection_entries([])
        assert s.n_entries == 0

    def test_summarise_edge_detection_nonempty(self):
        entries = self._edge_entries(5)
        s = summarise_edge_detection_entries(entries)
        assert s.n_entries == 5
        assert isinstance(s.methods, list)

    def test_filter_edge_by_min_density(self):
        entries = self._edge_entries(5)
        filtered = filter_edge_by_min_density(entries, 0.4)
        assert all(e.density >= 0.4 for e in filtered)

    def test_filter_edge_by_method(self):
        entries = self._edge_entries(3)
        entries[0] = make_edge_detection_entry(0, 0.5, 2, method="sobel")
        filtered = filter_edge_by_method(entries, "canny")
        assert all(e.method == "canny" for e in filtered)

    def test_top_k_edge_density_entries(self):
        entries = self._edge_entries(5)
        top = top_k_edge_density_entries(entries, k=2)
        assert len(top) == 2

    def test_best_edge_density_entry_none(self):
        assert best_edge_density_entry([]) is None

    def test_edge_detection_stats_empty(self):
        stats = edge_detection_stats([])
        assert stats["count"] == 0

    def test_compare_edge_detection_summaries(self):
        s1 = summarise_edge_detection_entries(self._edge_entries(5))
        s2 = summarise_edge_detection_entries(self._edge_entries(3))
        cmp = compare_edge_detection_summaries(s1, s2)
        assert "mean_density_delta" in cmp

    def test_batch_summarise_edge(self):
        groups = [self._edge_entries(3), self._edge_entries(4)]
        summaries = batch_summarise_edge_detection_entries(groups)
        assert len(summaries) == 2


# =============================================================================
# color_hist_utils
# =============================================================================

class TestColorHistUtils:
    def _entries(self, n=5):
        return [
            make_color_hist_entry(i, i+1, intersection=0.5+i*0.05, chi2=0.5+i*0.05)
            for i in range(n)
        ]

    def test_config_defaults(self):
        cfg = ColorHistConfig()
        assert cfg.space == "hsv"
        assert cfg.good_threshold == 0.7

    def test_config_invalid_min_score(self):
        with pytest.raises(ValueError):
            ColorHistConfig(min_score=-0.1)

    def test_config_max_lt_min(self):
        with pytest.raises(ValueError):
            ColorHistConfig(min_score=0.8, max_score=0.3)

    def test_config_invalid_good_threshold(self):
        with pytest.raises(ValueError):
            ColorHistConfig(good_threshold=1.5)

    def test_entry_score(self):
        e = make_color_hist_entry(0, 1, intersection=0.8, chi2=0.6)
        assert e.score == pytest.approx(0.7)

    def test_make_color_hist_entry(self):
        e = make_color_hist_entry(2, 3, 0.7, 0.5, space="rgb", n_bins=64)
        assert e.frag_i == 2
        assert e.space == "rgb"
        assert e.n_bins == 64

    def test_entries_from_comparisons(self):
        pairs = [(0, 1), (2, 3)]
        entries = entries_from_comparisons(pairs, [0.8, 0.6], [0.7, 0.5])
        assert len(entries) == 2

    def test_entries_from_comparisons_length_mismatch(self):
        with pytest.raises(ValueError):
            entries_from_comparisons([(0, 1)], [0.8, 0.6], [0.7])

    def test_summarise_empty(self):
        s = summarise_color_hist([])
        assert s.n_entries == 0

    def test_summarise_nonempty(self):
        entries = self._entries(5)
        s = summarise_color_hist(entries)
        assert s.n_entries == 5
        assert s.min_score <= s.max_score

    def test_filter_good(self):
        entries = self._entries(5)
        good = filter_good_hist_entries(entries, threshold=0.6)
        assert all(e.score >= 0.6 for e in good)

    def test_filter_poor(self):
        entries = self._entries(5)
        poor = filter_poor_hist_entries(entries, threshold=0.6)
        assert all(e.score < 0.6 for e in poor)

    def test_filter_by_intersection_range(self):
        entries = self._entries(5)
        filtered = filter_by_intersection_range(entries, 0.5, 0.6)
        assert all(0.5 <= e.intersection <= 0.6 for e in filtered)

    def test_filter_by_chi2_range(self):
        entries = self._entries(5)
        filtered = filter_by_chi2_range(entries, 0.5, 0.6)
        assert all(0.5 <= e.chi2 <= 0.6 for e in filtered)

    def test_filter_by_space(self):
        entries = self._entries(3)
        entries[0] = make_color_hist_entry(0, 1, 0.7, 0.5, space="rgb")
        filtered = filter_by_space(entries, "hsv")
        assert all(e.space == "hsv" for e in filtered)

    def test_top_k_hist_entries(self):
        entries = self._entries(5)
        top = top_k_hist_entries(entries, k=2)
        assert len(top) == 2
        assert top[0].score >= top[1].score

    def test_best_hist_entry_none(self):
        assert best_hist_entry([]) is None

    def test_best_hist_entry(self):
        entries = self._entries(5)
        best = best_hist_entry(entries)
        assert best.score == max(e.score for e in entries)

    def test_color_hist_stats_empty(self):
        stats = color_hist_stats([])
        assert stats["count"] == 0

    def test_color_hist_stats_nonempty(self):
        entries = self._entries(5)
        stats = color_hist_stats(entries)
        assert stats["count"] == 5

    def test_compare_hist_summaries(self):
        s1 = summarise_color_hist(self._entries(5))
        s2 = summarise_color_hist(self._entries(3))
        cmp = compare_hist_summaries(s1, s2)
        assert "d_mean_score" in cmp

    def test_batch_summarise_color_hist(self):
        groups = [self._entries(3), self._entries(4)]
        summaries = batch_summarise_color_hist(groups)
        assert len(summaries) == 2


# =============================================================================
# config_utils
# =============================================================================

class TestConfigUtils:
    def test_validate_section_valid(self):
        errors = validate_section({"lr": 0.01, "method": "sa"}, {"lr": float, "method": str})
        assert errors == []

    def test_validate_section_missing_field(self):
        errors = validate_section({"a": 1}, {"a": int, "b": str})
        assert any("Missing" in e for e in errors)

    def test_validate_section_wrong_type(self):
        errors = validate_section({"lr": "not_a_float"}, {"lr": float})
        assert len(errors) == 1

    def test_validate_section_dict(self):
        errors = validate_section({"x": 10}, {"x": int})
        assert errors == []

    def test_validate_range_valid(self):
        assert validate_range(0.5, 0.0, 1.0) is None

    def test_validate_range_below(self):
        msg = validate_range(-0.1, 0.0, 1.0, name="alpha")
        assert msg is not None and "alpha" in msg

    def test_validate_range_above(self):
        msg = validate_range(1.5, 0.0, 1.0)
        assert msg is not None

    def test_merge_dicts_simple(self):
        result = merge_dicts({"a": 1, "b": 2}, {"b": 3, "c": 4})
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_dicts_nested(self):
        base = {"x": {"a": 1, "b": 2}}
        override = {"x": {"b": 99}}
        result = merge_dicts(base, override)
        assert result["x"]["b"] == 99
        assert result["x"]["a"] == 1

    def test_merge_dicts_no_mutation(self):
        base = {"a": 1}
        merge_dicts(base, {"a": 2})
        assert base["a"] == 1

    def test_flatten_dict(self):
        d = {"a": {"b": 1, "c": 2}, "d": 3}
        flat = flatten_dict(d)
        assert flat == {"a.b": 1, "a.c": 2, "d": 3}

    def test_unflatten_dict(self):
        flat = {"a.b": 1, "a.c": 2, "d": 3}
        result = unflatten_dict(flat)
        assert result == {"a": {"b": 1, "c": 2}, "d": 3}

    def test_flatten_unflatten_roundtrip(self):
        original = {"x": {"y": {"z": 42}}, "a": 1}
        assert unflatten_dict(flatten_dict(original)) == original

    def test_overrides_from_env(self):
        os.environ["PUZZLE_TEST_KEY"] = "123"
        overrides = overrides_from_env(prefix="PUZZLE_")
        del os.environ["PUZZLE_TEST_KEY"]
        assert "test_key" in overrides or any("test" in k for k in overrides)

    def test_config_profile_apply_to(self):
        p = ConfigProfile(name="test", description="t", overrides={"a": 1})
        result = p.apply_to({"a": 0, "b": 2})
        assert result["a"] == 1
        assert result["b"] == 2

    def test_apply_profile_fast(self):
        cfg = {"fractal": {"ifs_transforms": 8}, "assembly": {"method": "sa"}}
        result = apply_profile(cfg, "fast")
        assert result["assembly"]["method"] == "greedy"

    def test_apply_profile_accurate(self):
        cfg = {"fractal": {}, "assembly": {}}
        result = apply_profile(cfg, "accurate")
        assert result["assembly"]["method"] == "beam"

    def test_apply_profile_debug(self):
        cfg = {"fractal": {}, "synthesis": {}, "assembly": {}}
        result = apply_profile(cfg, "debug")
        assert result["assembly"]["method"] == "greedy"

    def test_apply_profile_unknown(self):
        with pytest.raises(ValueError):
            apply_profile({}, "nonexistent")

    def test_list_profiles(self):
        profiles = list_profiles()
        names = [p[0] for p in profiles]
        assert "fast" in names
        assert "accurate" in names

    def test_load_save_json_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            data = {"a": 1, "b": {"c": 2}}
            save_json_config(data, path)
            loaded = load_json_config(path)
            assert loaded == data

    def test_load_json_config_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_json_config(Path("/nonexistent/path/cfg.json"))

    def test_diff_configs_no_diff(self):
        cfg = {"a": 1, "b": 2}
        diffs = diff_configs(cfg, cfg)
        assert diffs == {}

    def test_diff_configs_with_diff(self):
        diffs = diff_configs({"a": 1}, {"a": 2})
        assert "a" in diffs
        assert diffs["a"] == (1, 2)


# =============================================================================
# consensus_score_utils
# =============================================================================

class TestConsensusScoreUtils:
    def _vote_dict(self):
        return {
            frozenset([0, 1]): 3,
            frozenset([1, 2]): 2,
            frozenset([0, 2]): 1,
        }

    def test_config_defaults(self):
        cfg = ConsensusScoreConfig()
        assert cfg.min_vote_fraction == 0.5
        assert cfg.min_pairs == 1

    def test_config_invalid_fraction(self):
        with pytest.raises(ValueError):
            ConsensusScoreConfig(min_vote_fraction=1.5)

    def test_config_invalid_min_pairs(self):
        with pytest.raises(ValueError):
            ConsensusScoreConfig(min_pairs=0)

    def test_make_consensus_entry_above_threshold(self):
        e = make_consensus_entry((0, 1), vote_count=3, n_methods=4, threshold=0.5)
        assert e.is_consensus is True

    def test_make_consensus_entry_below_threshold(self):
        e = make_consensus_entry((0, 1), vote_count=1, n_methods=4, threshold=0.5)
        assert e.is_consensus is False

    def test_make_consensus_entry_zero_methods(self):
        e = make_consensus_entry((0, 1), vote_count=0, n_methods=0)
        assert e.is_consensus is False

    def test_entry_vote_fraction(self):
        e = ConsensusScoreEntry(pair=(0, 1), vote_count=3, n_methods=4, is_consensus=True)
        assert e.vote_fraction == pytest.approx(0.75)

    def test_entries_from_votes(self):
        entries = entries_from_votes(self._vote_dict(), n_methods=4)
        assert len(entries) == 3

    def test_summarise_consensus_empty(self):
        s = summarise_consensus([])
        assert s.n_pairs == 0

    def test_summarise_consensus_nonempty(self):
        entries = entries_from_votes(self._vote_dict(), n_methods=4)
        s = summarise_consensus(entries)
        assert s.n_pairs == 3
        assert 0.0 <= s.agreement_score <= 1.0

    def test_filter_consensus_pairs(self):
        entries = entries_from_votes(self._vote_dict(), n_methods=4)
        cons = filter_consensus_pairs(entries)
        assert all(e.is_consensus for e in cons)

    def test_filter_non_consensus(self):
        entries = entries_from_votes(self._vote_dict(), n_methods=4)
        non_cons = filter_non_consensus(entries)
        assert all(not e.is_consensus for e in non_cons)

    def test_filter_by_vote_fraction(self):
        entries = entries_from_votes(self._vote_dict(), n_methods=4)
        filtered = filter_by_vote_fraction(entries, 0.5)
        assert all(e.vote_fraction >= 0.5 for e in filtered)

    def test_top_k_consensus_entries(self):
        entries = entries_from_votes(self._vote_dict(), n_methods=4)
        top = top_k_consensus_entries(entries, 2)
        assert len(top) == 2

    def test_consensus_score_stats_empty(self):
        stats = consensus_score_stats([])
        assert stats["count"] == 0

    def test_consensus_score_stats_nonempty(self):
        entries = entries_from_votes(self._vote_dict(), n_methods=4)
        stats = consensus_score_stats(entries)
        assert stats["count"] == 3

    def test_agreement_score_empty(self):
        assert agreement_score([]) == 0.0

    def test_agreement_score_all_consensus(self):
        entries = [make_consensus_entry((i, i+1), 4, 4, 0.5) for i in range(5)]
        assert agreement_score(entries) == 1.0

    def test_compare_consensus(self):
        e1 = entries_from_votes(self._vote_dict(), n_methods=4)
        e2 = entries_from_votes({frozenset([0, 1]): 1}, n_methods=4)
        s1 = summarise_consensus(e1)
        s2 = summarise_consensus(e2)
        cmp = compare_consensus(s1, s2)
        assert "n_pairs_delta" in cmp

    def test_batch_summarise_consensus(self):
        vote_dicts = [self._vote_dict(), {frozenset([0, 1]): 2}]
        n_methods = [4, 4]
        summaries = batch_summarise_consensus(vote_dicts, n_methods)
        assert len(summaries) == 2

    def test_entry_repr(self):
        e = make_consensus_entry((0, 1), 3, 4)
        r = repr(e)
        assert "ConsensusScoreEntry" in r


# =============================================================================
# contour_profile_utils
# =============================================================================

class TestContourProfileUtils:
    def _circle_contour(self, n=32):
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return np.stack([np.cos(t), np.sin(t)], axis=1)

    def test_profile_config_defaults(self):
        cfg = ProfileConfig()
        assert cfg.n_samples == 64
        assert cfg.normalize is True

    def test_sample_profile_along_contour_shape(self):
        pts = self._circle_contour(20)
        sampled = sample_profile_along_contour(pts, n_samples=32)
        assert sampled.shape == (32, 2)

    def test_sample_profile_empty(self):
        with pytest.raises(ValueError):
            sample_profile_along_contour(np.array([]).reshape(0, 2))

    def test_sample_profile_invalid_n(self):
        pts = self._circle_contour(10)
        with pytest.raises(ValueError):
            sample_profile_along_contour(pts, n_samples=0)

    def test_sample_profile_single_point(self):
        pts = np.array([[1.0, 2.0]])
        sampled = sample_profile_along_contour(pts, n_samples=5)
        assert sampled.shape == (5, 2)

    def test_contour_curvature_shape(self):
        pts = self._circle_contour(20)
        curvature = contour_curvature(pts)
        assert curvature.shape == (20,)

    def test_contour_curvature_short(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        curv = contour_curvature(pts)
        assert np.all(curv == 0.0)

    def test_smooth_profile_window_1(self):
        v = np.array([1.0, 2.0, 3.0, 4.0])
        result = smooth_profile(v, window=1)
        assert np.allclose(result, v)

    def test_smooth_profile_basic(self):
        v = np.ones(10) * 5.0
        result = smooth_profile(v, window=3)
        assert result.shape == v.shape

    def test_smooth_profile_invalid_window(self):
        with pytest.raises(ValueError):
            smooth_profile(np.ones(5), window=0)

    def test_normalize_profile_range(self):
        v = np.array([1.0, 3.0, 7.0])
        norm = normalize_profile(v)
        assert norm.min() == pytest.approx(0.0)
        assert norm.max() == pytest.approx(1.0)

    def test_normalize_profile_constant(self):
        v = np.ones(5) * 3.0
        result = normalize_profile(v)
        assert np.all(result == 1.0)

    def test_profile_l2_distance_zero(self):
        v = np.array([1.0, 2.0, 3.0])
        assert profile_l2_distance(v, v) == pytest.approx(0.0)

    def test_profile_l2_distance_nonzero(self):
        a = np.zeros(5)
        b = np.ones(5)
        d = profile_l2_distance(a, b)
        assert d == pytest.approx(math.sqrt(5))

    def test_profile_l2_distance_shape_mismatch(self):
        with pytest.raises(ValueError):
            profile_l2_distance(np.ones(3), np.ones(4))

    def test_profile_cosine_similarity_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        sim = profile_cosine_similarity(v, v)
        assert sim == pytest.approx(1.0)

    def test_profile_cosine_similarity_zero_vector(self):
        v = np.zeros(4)
        w = np.ones(4)
        assert profile_cosine_similarity(v, w) == 0.0

    def test_best_cyclic_offset_zero_shift(self):
        v = np.array([1.0, 2.0, 3.0, 4.0])
        offset, dist = best_cyclic_offset(v, v)
        assert offset == 0
        assert dist == pytest.approx(0.0)

    def test_best_cyclic_offset_empty(self):
        with pytest.raises(ValueError):
            best_cyclic_offset(np.array([]), np.array([]))

    def test_align_profiles_output(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.roll(a, 2)
        a_al, b_al, offset = align_profiles(a, b)
        assert a_al.shape == a.shape
        assert b_al.shape == b.shape

    def test_match_profiles_returns_result(self):
        a = RNG.random(32)
        b = RNG.random(32)
        result = match_profiles(a, b, cyclic=False, normalize=True)
        assert isinstance(result, ProfileMatchResult)
        assert 0.0 <= result.score <= 1.0

    def test_match_profiles_cyclic(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.roll(a, 1)
        result = match_profiles(a, b, cyclic=True)
        assert result.method == "cyclic"
        assert result.score > 0.9

    def test_match_profiles_with_cfg(self):
        a = RNG.random(32)
        b = RNG.random(32)
        cfg = ProfileConfig(normalize=False)
        result = match_profiles(a, b, cfg=cfg)
        assert isinstance(result, ProfileMatchResult)

    def test_batch_match_profiles(self):
        ref = RNG.random(16)
        candidates = [RNG.random(16) for _ in range(4)]
        results = batch_match_profiles(ref, candidates)
        assert len(results) == 4

    def test_top_k_profile_matches(self):
        ref = RNG.random(16)
        candidates = [RNG.random(16) for _ in range(5)]
        results = batch_match_profiles(ref, candidates)
        top = top_k_profile_matches(results, k=2)
        assert len(top) == 2
        assert top[0].score >= top[1].score

    def test_profile_match_result_repr(self):
        r = ProfileMatchResult(score=0.8, offset=2, distance=0.3, method="l2")
        rep = repr(r)
        assert "ProfileMatchResult" in rep
