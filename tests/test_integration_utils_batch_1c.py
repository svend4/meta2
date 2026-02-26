"""Integration tests for utils batch 1c.

Modules covered:
  - puzzle_reconstruction.utils.assembly_records
  - puzzle_reconstruction.utils.classification_freq_records
  - puzzle_reconstruction.utils.config_utils
  - puzzle_reconstruction.utils.consensus_score_utils
  - puzzle_reconstruction.utils.contour_profile_utils
"""
import os
import pytest
import numpy as np

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# assembly_records
# ---------------------------------------------------------------------------
from puzzle_reconstruction.utils.assembly_records import (
    CollisionRecord,
    CostMatrixRecord,
    FragmentScoreRecord,
    AssemblyScoreRecord,
    OverlapRecord,
    ResolveRecord,
    make_collision_record,
    make_cost_matrix_record,
    make_overlap_record,
)


def test_collision_record_basic():
    rec = make_collision_record(3, 7, overlap_w=10, overlap_h=5, overlap_area=50)
    assert rec.id1 == 3
    assert rec.id2 == 7
    assert rec.overlap_area == 50


def test_collision_record_pair_key_ordered():
    rec = make_collision_record(9, 2)
    assert rec.pair_key == (2, 9)


def test_collision_record_zero_overlap():
    rec = make_collision_record(0, 1)
    assert rec.overlap_area == 0
    assert rec.overlap_w == 0


def test_cost_matrix_record():
    rec = make_cost_matrix_record(16, "sa", min_cost=0.1, max_cost=0.9, mean_cost=0.45, n_forbidden=2)
    assert rec.n_fragments == 16
    assert rec.method == "sa"
    assert rec.n_forbidden == 2
    assert rec.min_cost < rec.max_cost


def test_fragment_score_record():
    rec = FragmentScoreRecord(fragment_idx=5, local_score=0.8, n_neighbors=3, is_reliable=True)
    assert rec.is_reliable
    assert rec.local_score == pytest.approx(0.8)


def test_assembly_score_record_with_fragments():
    f0 = FragmentScoreRecord(0, 0.9, 4, True)
    f1 = FragmentScoreRecord(1, 0.5, 2, False)
    rec = AssemblyScoreRecord(
        global_score=0.75, coverage=0.9, mean_local=0.7,
        n_placed=2, n_reliable=1, fragment_scores={0: f0, 1: f1},
    )
    assert rec.n_placed == 2
    assert rec.fragment_scores[0].is_reliable
    assert not rec.fragment_scores[1].is_reliable


def test_overlap_record_has_overlap():
    rec = make_overlap_record(1, 2, area=25.0, dx=3.0, dy=-2.0)
    assert rec.has_overlap
    assert rec.pair_key == (1, 2)


def test_overlap_record_no_overlap():
    rec = make_overlap_record(4, 6, area=0.0)
    assert not rec.has_overlap


def test_overlap_record_pair_key_ordering():
    rec = make_overlap_record(10, 3, area=1.0)
    assert rec.pair_key == (3, 10)


def test_resolve_record_fields():
    rec = ResolveRecord(n_iter=5, resolved=True, final_n_overlaps=0, n_fragments=8)
    assert rec.resolved
    assert rec.final_n_overlaps == 0


def test_cost_matrix_record_default_forbidden():
    rec = make_cost_matrix_record(4, "greedy")
    assert rec.n_forbidden == 0


# ---------------------------------------------------------------------------
# classification_freq_records
# ---------------------------------------------------------------------------
from puzzle_reconstruction.utils.classification_freq_records import (
    FragmentClassifyRecord,
    FragmentMapRecord,
    FragmentValidationRecord,
    FreqDescriptorRecord,
    make_fragment_classify_record,
    make_freq_descriptor_record,
)


def test_fragment_classify_corner():
    rec = make_fragment_classify_record(0, "corner", 0.95, False, 0, [0, 1])
    assert rec.is_corner
    assert not rec.is_edge
    assert not rec.is_inner
    assert rec.n_straight_sides == 2


def test_fragment_classify_edge():
    rec = make_fragment_classify_record(1, "edge", 0.85, True, 2, [0])
    assert rec.is_edge
    assert rec.has_text
    assert rec.text_lines == 2


def test_fragment_classify_inner():
    rec = make_fragment_classify_record(2, "inner", 0.7, False, 0, [])
    assert rec.is_inner
    assert rec.n_straight_sides == 0


def test_fragment_map_record_coverage_ratio():
    rec = FragmentMapRecord(n_fragments=10, n_zones=20, n_assigned=15, canvas_w=640, canvas_h=480)
    assert rec.coverage_ratio == pytest.approx(0.75)


def test_fragment_map_record_zero_zones():
    rec = FragmentMapRecord(n_fragments=5, n_zones=0, n_assigned=0, canvas_w=100, canvas_h=100)
    assert rec.coverage_ratio == 0.0


def test_fragment_map_record_assignment_ratio():
    rec = FragmentMapRecord(n_fragments=8, n_zones=10, n_assigned=6, canvas_w=320, canvas_h=240)
    assert rec.assignment_ratio == pytest.approx(0.75)


def test_fragment_validation_record_aspect_ratio():
    rec = FragmentValidationRecord(1, True, 0, 0, 0, width=100.0, height=50.0)
    assert rec.aspect_ratio == pytest.approx(0.5)


def test_fragment_validation_record_zero_height():
    rec = FragmentValidationRecord(2, False, 1, 1, 0, width=50.0, height=0.0)
    assert rec.aspect_ratio == 0.0


def test_freq_descriptor_high_frequency():
    rec = make_freq_descriptor_record(3, 8, centroid=0.6, entropy=2.1, dominant_band=5, high_freq_ratio=0.7)
    assert rec.is_high_frequency
    assert not rec.is_smooth


def test_freq_descriptor_smooth():
    rec = make_freq_descriptor_record(4, 8, centroid=0.2, entropy=1.5, dominant_band=1, high_freq_ratio=0.3)
    assert not rec.is_high_frequency
    assert rec.is_smooth


def test_freq_descriptor_fields():
    rec = make_freq_descriptor_record(7, 16, 0.4, 3.0, 4, 0.5)
    assert rec.fragment_id == 7
    assert rec.n_bands == 16
    assert rec.dominant_band == 4


# ---------------------------------------------------------------------------
# config_utils
# ---------------------------------------------------------------------------
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
)


def test_validate_section_valid():
    d = {"alpha": 0.5, "n_iter": 10}
    errors = validate_section(d, {"alpha": float, "n_iter": int})
    assert errors == []


def test_validate_section_missing_field():
    d = {"alpha": 0.5}
    errors = validate_section(d, {"alpha": float, "n_iter": int})
    assert any("n_iter" in e for e in errors)


def test_validate_section_wrong_type():
    d = {"alpha": "bad"}
    errors = validate_section(d, {"alpha": float})
    assert errors


def test_validate_range_inside():
    assert validate_range(0.5, 0.0, 1.0) is None


def test_validate_range_outside():
    msg = validate_range(1.5, 0.0, 1.0, name="threshold")
    assert msg is not None
    assert "threshold" in msg


def test_merge_dicts_deep():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 99, "z": 0}}
    result = merge_dicts(base, override)
    assert result["a"]["x"] == 1
    assert result["a"]["y"] == 99
    assert result["a"]["z"] == 0
    assert result["b"] == 3


def test_flatten_unflatten_roundtrip():
    nested = {"a": {"b": {"c": 42}, "d": 7}, "e": "hello"}
    flat = flatten_dict(nested)
    assert flat["a.b.c"] == 42
    assert flat["a.d"] == 7
    restored = unflatten_dict(flat)
    assert restored == nested


def test_overrides_from_env(monkeypatch):
    monkeypatch.setenv("PUZZLE_ASSEMBLY__METHOD", "beam")
    monkeypatch.setenv("PUZZLE_FRACTAL__N_SCALES", "8")
    result = overrides_from_env("PUZZLE_")
    assert result["assembly"]["method"] == "beam"
    assert result["fractal"]["n_scales"] == 8


def test_profiles_exist():
    for name in ("fast", "accurate", "debug"):
        assert name in PROFILES


def test_apply_profile_fast():
    cfg = {"assembly": {"method": "sa"}, "fractal": {"n_scales": 8}}
    result = apply_profile(cfg, "fast")
    assert result["assembly"]["method"] == "greedy"


def test_apply_profile_unknown_raises():
    with pytest.raises(KeyError):
        apply_profile({}, "nonexistent_profile")


def test_config_profile_apply_to():
    profile = ConfigProfile("test", "desc", overrides={"key": 42})
    result = profile.apply_to({"key": 0, "other": 1})
    assert result["key"] == 42
    assert result["other"] == 1


# ---------------------------------------------------------------------------
# consensus_score_utils
# ---------------------------------------------------------------------------
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
)


def test_consensus_score_config_defaults():
    cfg = ConsensusScoreConfig()
    assert cfg.min_vote_fraction == pytest.approx(0.5)
    assert cfg.min_pairs == 1


def test_consensus_score_config_invalid_fraction():
    with pytest.raises(ValueError):
        ConsensusScoreConfig(min_vote_fraction=1.5)


def test_consensus_score_config_invalid_pairs():
    with pytest.raises(ValueError):
        ConsensusScoreConfig(min_pairs=0)


def test_make_consensus_entry_is_consensus():
    entry = make_consensus_entry((0, 1), vote_count=4, n_methods=5, threshold=0.5)
    assert entry.is_consensus
    assert entry.vote_fraction == pytest.approx(0.8)


def test_make_consensus_entry_not_consensus():
    entry = make_consensus_entry((2, 3), vote_count=1, n_methods=5, threshold=0.5)
    assert not entry.is_consensus


def test_entries_from_votes():
    votes = {frozenset([0, 1]): 3, frozenset([1, 2]): 1}
    entries = entries_from_votes(votes, n_methods=4, threshold=0.5)
    assert len(entries) == 2
    consensus_entries = [e for e in entries if e.is_consensus]
    assert len(consensus_entries) == 1


def test_summarise_consensus_empty():
    summary = summarise_consensus([])
    assert summary.n_pairs == 0
    assert summary.agreement_score == pytest.approx(0.0)


def test_summarise_consensus_full():
    e1 = make_consensus_entry((0, 1), 4, 4, 0.5)
    e2 = make_consensus_entry((1, 2), 2, 4, 0.5)
    e3 = make_consensus_entry((0, 2), 1, 4, 0.5)
    summary = summarise_consensus([e1, e2, e3])
    assert summary.n_pairs == 3
    assert summary.n_consensus == 2
    assert 0.0 <= summary.agreement_score <= 1.0


def test_filter_consensus_pairs():
    e1 = make_consensus_entry((0, 1), 3, 4, 0.5)
    e2 = make_consensus_entry((1, 2), 1, 4, 0.5)
    filtered = filter_consensus_pairs([e1, e2])
    assert all(e.is_consensus for e in filtered)


def test_filter_non_consensus():
    e1 = make_consensus_entry((0, 1), 3, 4, 0.5)
    e2 = make_consensus_entry((1, 2), 1, 4, 0.5)
    non_cons = filter_non_consensus([e1, e2])
    assert all(not e.is_consensus for e in non_cons)


def test_top_k_consensus_entries():
    entries = [make_consensus_entry((i, i+1), i, 5, 0.5) for i in range(5)]
    top2 = top_k_consensus_entries(entries, k=2)
    assert len(top2) == 2
    assert top2[0].vote_fraction >= top2[1].vote_fraction


def test_consensus_score_stats():
    entries = [make_consensus_entry((i, i+1), i+1, 5, 0.5) for i in range(4)]
    stats = consensus_score_stats(entries)
    assert stats["count"] == 4
    assert 0.0 <= stats["mean_fraction"] <= 1.0
    assert stats["n_consensus"] + stats["n_non_consensus"] == 4


# ---------------------------------------------------------------------------
# contour_profile_utils
# ---------------------------------------------------------------------------
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
)


def test_sample_profile_output_shape():
    contour = rng.random((20, 2)) * 100
    sampled = sample_profile_along_contour(contour, n_samples=32)
    assert sampled.shape == (32, 2)


def test_sample_profile_single_point():
    contour = np.array([[5.0, 10.0]])
    sampled = sample_profile_along_contour(contour, n_samples=8)
    assert sampled.shape == (8, 2)
    assert np.allclose(sampled, [[5.0, 10.0]])


def test_sample_profile_empty_raises():
    with pytest.raises(ValueError):
        sample_profile_along_contour(np.empty((0, 2)), n_samples=10)


def test_contour_curvature_shape():
    contour = rng.random((15, 2)) * 50
    curv = contour_curvature(contour)
    assert curv.shape == (15,)


def test_contour_curvature_short():
    assert contour_curvature(np.array([[0.0, 0.0], [1.0, 1.0]])).shape == (2,)


def test_smooth_profile_returns_same_length():
    v = rng.random(50)
    smoothed = smooth_profile(v, window=5)
    assert smoothed.shape == v.shape


def test_smooth_profile_window_1_identity():
    v = rng.random(20)
    assert np.allclose(smooth_profile(v, window=1), v)


def test_smooth_profile_invalid_window():
    with pytest.raises(ValueError):
        smooth_profile(np.array([1.0, 2.0]), window=0)


def test_normalize_profile_range():
    v = rng.random(30) * 10 + 5
    normed = normalize_profile(v)
    assert normed.min() >= 0.0 - 1e-9
    assert normed.max() <= 1.0 + 1e-9


def test_normalize_profile_constant():
    v = np.ones(10) * 3.0
    normed = normalize_profile(v)
    assert np.allclose(normed, 1.0)


def test_profile_l2_distance_zero():
    v = rng.random(16)
    assert profile_l2_distance(v, v) == pytest.approx(0.0)


def test_profile_l2_distance_known():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert profile_l2_distance(a, b) == pytest.approx(5.0)


def test_profile_cosine_similarity_identical():
    v = rng.random(10) + 0.1
    assert profile_cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_best_cyclic_offset_no_shift():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    offset, dist = best_cyclic_offset(a, a)
    assert offset == 0
    assert dist == pytest.approx(0.0)
