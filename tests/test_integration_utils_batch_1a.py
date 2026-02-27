"""Integration tests for utils batch 1a.

Covers:
- puzzle_reconstruction.utils.alignment_utils
- puzzle_reconstruction.utils.annealing_schedule
- puzzle_reconstruction.utils.annealing_score_utils
- puzzle_reconstruction.utils.assembly_config_utils
- puzzle_reconstruction.utils.assembly_score_utils
"""
import math
import pytest
import numpy as np

rng = np.random.default_rng(42)

from puzzle_reconstruction.utils.alignment_utils import (
    AlignmentConfig, AlignmentResult,
    normalize_for_alignment, find_best_rotation,
    compute_alignment_error, align_curves_procrustes,
)
from puzzle_reconstruction.utils.annealing_schedule import (
    ScheduleConfig, TemperatureRecord,
    linear_schedule, geometric_schedule,
    exponential_schedule, cosine_schedule, stepped_schedule,
)
from puzzle_reconstruction.utils.annealing_score_utils import (
    AnnealingScoreConfig, AnnealingScoreEntry,
    make_annealing_entry, entries_from_log,
    summarise_annealing, filter_accepted, filter_rejected,
)
from puzzle_reconstruction.utils.assembly_config_utils import (
    AssemblyStateRecord, AssemblyStateHistory,
    ConfigChangeRecord, ConfigChangeLog,
    CandidateFilterRecord,
    summarize_assembly_history,
    build_filter_pipeline_summary, build_config_change_log,
)
from puzzle_reconstruction.utils.assembly_score_utils import (
    AssemblyScoreConfig, AssemblyScoreEntry,
    make_assembly_entry, entries_from_assemblies, summarise_assemblies,
    filter_good_assemblies, filter_poor_assemblies,
    filter_by_method, filter_by_score_range, filter_by_min_fragments,
)


# ===========================================================================
# alignment_utils  (11 tests)
# ===========================================================================

def test_alignment_config_defaults():
    cfg = AlignmentConfig()
    assert cfg.n_samples == 64 and cfg.max_icp_iter == 50

def test_alignment_config_invalid_n_samples():
    with pytest.raises(ValueError):
        AlignmentConfig(n_samples=1)

def test_alignment_config_invalid_icp_tol():
    with pytest.raises(ValueError):
        AlignmentConfig(icp_tol=0.0)

def test_normalize_zero_mean():
    pts = rng.random((20, 2)) * 100
    normed, _, _ = normalize_for_alignment(pts)
    assert np.allclose(normed.mean(axis=0), 0.0, atol=1e-10)

def test_normalize_constant_points_scale_one():
    _, _, scale = normalize_for_alignment(np.ones((5, 2)) * 7.0)
    assert scale == 1.0

def test_find_best_rotation_identity():
    pts = rng.random((8, 2)); pts -= pts.mean(axis=0)
    angle, R = find_best_rotation(pts, pts)
    assert abs(angle) < 1e-10 and np.allclose(R, np.eye(2), atol=1e-10)

def test_find_best_rotation_known_angle():
    pts = rng.random((12, 2)); pts -= pts.mean(axis=0)
    theta = math.pi / 4
    c, s = math.cos(theta), math.sin(theta)
    rotated = pts @ np.array([[c, -s], [s, c]]).T
    angle, _ = find_best_rotation(pts, rotated)
    assert abs(angle - theta) < 1e-8

def test_compute_alignment_error_identical():
    pts = rng.random((10, 2))
    assert compute_alignment_error(pts, pts) == pytest.approx(0.0)

def test_compute_alignment_error_mismatched_shapes():
    assert compute_alignment_error(rng.random((5, 2)), rng.random((6, 2))) == float("inf")

def test_compute_alignment_error_known():
    err = compute_alignment_error(np.zeros((4, 2)), np.ones((4, 2)))
    assert err == pytest.approx(2.0)

def test_align_curves_procrustes_self_low_error():
    pts = rng.random((20, 2))
    result = align_curves_procrustes(pts, pts)
    assert isinstance(result, AlignmentResult) and result.error < 1e-10


# ===========================================================================
# annealing_schedule  (11 tests)
# ===========================================================================

def test_schedule_config_defaults():
    cfg = ScheduleConfig()
    assert cfg.t_start == 1.0 and cfg.kind == "geometric"

def test_schedule_config_invalid_t_end():
    with pytest.raises(ValueError):
        ScheduleConfig(t_start=1.0, t_end=2.0)

def test_schedule_config_invalid_kind():
    with pytest.raises(ValueError):
        ScheduleConfig(kind="unknown")

def test_cooling_rate_between_0_1():
    cfg = ScheduleConfig(t_start=10.0, t_end=0.01, n_steps=100)
    assert 0.0 < cfg.cooling_rate < 1.0

def test_linear_schedule_length_and_bounds():
    cfg = ScheduleConfig(t_start=5.0, t_end=0.1, n_steps=20, kind="linear")
    recs = linear_schedule(cfg)
    assert len(recs) == 20
    assert recs[0].temperature == pytest.approx(5.0)
    assert recs[-1].temperature == pytest.approx(0.1)

def test_linear_schedule_monotone():
    cfg = ScheduleConfig(t_start=10.0, t_end=0.01, n_steps=30, kind="linear")
    temps = [r.temperature for r in linear_schedule(cfg)]
    assert all(temps[i] >= temps[i + 1] for i in range(len(temps) - 1))

def test_geometric_schedule_monotone():
    cfg = ScheduleConfig(t_start=10.0, t_end=0.01, n_steps=50, kind="geometric")
    temps = [r.temperature for r in geometric_schedule(cfg)]
    assert all(temps[i] >= temps[i + 1] for i in range(len(temps) - 1))

def test_exponential_schedule_length_and_start():
    cfg = ScheduleConfig(t_start=8.0, t_end=0.01, n_steps=40, kind="exponential")
    recs = exponential_schedule(cfg)
    assert len(recs) == 40 and recs[0].temperature == pytest.approx(8.0, rel=1e-6)

def test_cosine_schedule_endpoints():
    cfg = ScheduleConfig(t_start=10.0, t_end=0.1, n_steps=50, kind="cosine")
    recs = cosine_schedule(cfg)
    assert recs[0].temperature == pytest.approx(10.0, rel=1e-6)
    assert recs[-1].temperature == pytest.approx(0.1, rel=1e-6)

def test_stepped_schedule_length():
    cfg = ScheduleConfig(t_start=5.0, t_end=0.05, n_steps=20, kind="stepped", step_size=5)
    assert len(stepped_schedule(cfg)) == 20

def test_stepped_schedule_all_positive():
    cfg = ScheduleConfig(t_start=5.0, t_end=0.05, n_steps=20, kind="stepped", step_size=4)
    assert all(r.temperature > 0 for r in stepped_schedule(cfg))


# ===========================================================================
# annealing_score_utils  (11 tests)
# ===========================================================================

def test_annealing_score_config_invalid_window():
    with pytest.raises(ValueError):
        AnnealingScoreConfig(convergence_window=0)

def test_annealing_score_config_invalid_threshold():
    with pytest.raises(ValueError):
        AnnealingScoreConfig(improvement_threshold=-0.1)

def test_make_annealing_entry_fields():
    e = make_annealing_entry(0, 1.0, 0.5, 0.6, True)
    assert e.iteration == 0 and e.accepted is True and e.meta == {}

def test_make_annealing_entry_meta():
    e = make_annealing_entry(2, 0.3, 0.7, 0.8, True, meta={"x": 42})
    assert e.meta["x"] == 42

def test_entries_from_log_length():
    log = [{"iteration": i, "temperature": 1.0, "current_score": 0.5,
            "best_score": 0.5, "accepted": True} for i in range(5)]
    assert len(entries_from_log(log)) == 5

def test_entries_from_log_defaults_for_missing_keys():
    entries = entries_from_log([{}])
    assert entries[0].iteration == 0 and entries[0].accepted is False

def test_summarise_annealing_empty():
    s = summarise_annealing([])
    assert s.n_iterations == 0 and not s.converged

def test_summarise_annealing_n_accepted():
    entries = [make_annealing_entry(i, 1.0, 0.5, 0.5, i % 2 == 0) for i in range(10)]
    s = summarise_annealing(entries)
    assert s.n_accepted == 5 and 0.0 <= s.acceptance_rate <= 1.0

def test_summarise_annealing_converged_flat():
    entries = [make_annealing_entry(i, 0.1, 0.5, 0.5, True) for i in range(15)]
    cfg = AnnealingScoreConfig(convergence_window=10, improvement_threshold=1e-4)
    assert summarise_annealing(entries, cfg).converged is True

def test_filter_accepted():
    entries = [make_annealing_entry(i, 0.5, 0.3, 0.4, i % 2 == 0) for i in range(6)]
    accepted = filter_accepted(entries)
    assert all(e.accepted for e in accepted) and len(accepted) == 3

def test_filter_rejected():
    entries = [make_annealing_entry(i, 0.5, 0.3, 0.4, i % 3 != 0) for i in range(6)]
    rejected = filter_rejected(entries)
    assert all(not e.accepted for e in rejected)


# ===========================================================================
# assembly_config_utils  (11 tests)
# ===========================================================================

def test_assembly_state_record_is_complete():
    r = AssemblyStateRecord(step=1, n_placed=5, n_fragments=5, coverage=1.0)
    assert r.is_complete is True

def test_assembly_state_record_not_complete():
    assert not AssemblyStateRecord(0, 3, 5, 0.6).is_complete

def test_assembly_state_record_invalid_coverage():
    with pytest.raises(ValueError):
        AssemblyStateRecord(0, 0, 1, 1.5)

def test_assembly_state_history_append_and_last_coverage():
    h = AssemblyStateHistory()
    for i in range(3):
        h.append(AssemblyStateRecord(i, i, 5, i / 5.0))
    assert h.n_steps == 3 and h.last_coverage == pytest.approx(2 / 5.0)

def test_assembly_state_history_is_monotone_true():
    h = AssemblyStateHistory()
    for i in range(4):
        h.append(AssemblyStateRecord(i, i, 4, i / 4.0))
    assert h.is_monotone is True

def test_assembly_state_history_is_monotone_false():
    h = AssemblyStateHistory()
    h.append(AssemblyStateRecord(0, 2, 4, 0.5))
    h.append(AssemblyStateRecord(1, 1, 4, 0.25))
    assert h.is_monotone is False

def test_assembly_state_history_empty_last_coverage():
    assert AssemblyStateHistory().last_coverage == 0.0

def test_config_change_log_n_changes():
    log = ConfigChangeLog()
    log.append(ConfigChangeRecord("alpha", 0.1, 0.2, step=0))
    log.append(ConfigChangeRecord("beta", 5, 5, step=1))
    assert log.n_changes == 1 and set(log.changed_keys) == {"alpha"}

def test_build_filter_pipeline_summary():
    records = [CandidateFilterRecord("f1", 100, 80, 20),
               CandidateFilterRecord("f2", 80, 60, 20)]
    s = build_filter_pipeline_summary(records)
    assert s.total_removed == 40 and s.n_stages == 2 and s.final_n_kept == 60

def test_build_config_change_log():
    diffs = [{"lr": (0.01, 0.001), "temp": (1.0, 0.9)}, {"lr": (0.001, 0.001)}]
    assert build_config_change_log(diffs).n_changes == 2

def test_summarize_assembly_history():
    h = AssemblyStateHistory()
    h.append(AssemblyStateRecord(0, 1, 4, 0.25))
    h.append(AssemblyStateRecord(1, 4, 4, 1.0))
    s = summarize_assembly_history(h)
    assert s["n_steps"] == 2 and s["final_coverage"] == pytest.approx(1.0) and s["is_complete"]


# ===========================================================================
# assembly_score_utils  (11 tests)
# ===========================================================================

def test_assembly_score_config_invalid_min_score():
    with pytest.raises(ValueError):
        AssemblyScoreConfig(min_score=-1.0)

def test_assembly_score_config_invalid_max_entries():
    with pytest.raises(ValueError):
        AssemblyScoreConfig(max_entries=0)

def test_assembly_score_entry_is_good():
    assert make_assembly_entry(0, "sa", 10, 0.8).is_good is True

def test_assembly_score_entry_not_good():
    assert make_assembly_entry(1, "sa", 10, 0.3).is_good is False

def test_assembly_score_entry_score_per_fragment():
    assert make_assembly_entry(0, "ga", 4, 2.0).score_per_fragment == pytest.approx(0.5)

def test_assembly_score_entry_zero_fragments():
    e = AssemblyScoreEntry(run_id=0, method="x", n_fragments=0, total_score=0.0)
    assert e.score_per_fragment == 0.0

def test_summarise_assemblies_empty():
    s = summarise_assemblies([])
    assert s.n_total == 0 and s.mean_score == 0.0

def test_summarise_assemblies_counts():
    entries = [make_assembly_entry(i, "sa", 10, v) for i, v in enumerate([0.8, 0.3, 0.9])]
    s = summarise_assemblies(entries)
    assert s.n_total == 3 and s.n_good == 2 and s.n_poor == 1

def test_filter_good_poor():
    entries = [make_assembly_entry(i, "sa", 10, v) for i, v in enumerate([0.8, 0.3, 0.6])]
    assert len(filter_good_assemblies(entries)) == 2
    assert len(filter_poor_assemblies(entries)) == 1

def test_filter_by_method_and_score_range():
    entries = [make_assembly_entry(0, "sa", 10, 0.8), make_assembly_entry(1, "ga", 5, 0.3),
               make_assembly_entry(2, "sa", 8, 0.6)]
    assert len(filter_by_method(entries, "sa")) == 2
    assert all(0.5 <= e.total_score <= 0.85 for e in filter_by_score_range(entries, 0.5, 0.85))

def test_entries_from_assemblies_ranks():
    class FakeAsm:
        def __init__(self, score, n):
            self.total_score = score
            self.placements = {i: None for i in range(n)}
    assemblies = [FakeAsm(0.5, 3), FakeAsm(0.9, 5), FakeAsm(0.2, 2)]
    entries = entries_from_assemblies(assemblies, method="test")
    ranks = {e.total_score: e.rank for e in entries}
    assert ranks[0.9] == 1 and ranks[0.5] == 2 and ranks[0.2] == 3
