"""Tests for puzzle_reconstruction.utils.rotation_score_utils"""
import numpy as np
import pytest
from puzzle_reconstruction.utils.rotation_score_utils import (
    RotationScoreConfig,
    RotationScoreEntry,
    make_entry,
    filter_by_confidence,
    filter_by_method,
    filter_by_angle_range,
    rank_by_confidence,
    best_entry,
    aggregate_angles,
    rotation_score_stats,
    angle_agreement,
    batch_make_entries,
    top_k_entries,
    group_by_method,
)

np.random.seed(42)


# ─── RotationScoreConfig ──────────────────────────────────────────────────────

def test_config_defaults():
    cfg = RotationScoreConfig()
    assert cfg.min_confidence == 0.0
    assert cfg.angle_tolerance_deg == 5.0
    assert cfg.preferred_method == ""


def test_config_invalid_confidence():
    with pytest.raises(ValueError):
        RotationScoreConfig(min_confidence=1.5)


def test_config_negative_confidence():
    with pytest.raises(ValueError):
        RotationScoreConfig(min_confidence=-0.1)


def test_config_invalid_tolerance():
    with pytest.raises(ValueError):
        RotationScoreConfig(angle_tolerance_deg=-1.0)


# ─── RotationScoreEntry / make_entry ─────────────────────────────────────────

def test_make_entry_basic():
    e = make_entry(0, 45.0, 0.8, "pca")
    assert e.image_id == 0
    assert e.angle_deg == 45.0
    assert e.confidence == 0.8
    assert e.method == "pca"


def test_make_entry_with_meta():
    e = make_entry(1, 90.0, 0.9, "moments", meta={"channel": "gray"})
    assert e.meta == {"channel": "gray"}


def test_entry_invalid_image_id():
    with pytest.raises(ValueError):
        make_entry(-1, 0.0, 0.5, "pca")


def test_entry_invalid_confidence_high():
    with pytest.raises(ValueError):
        make_entry(0, 0.0, 1.5, "pca")


def test_entry_invalid_confidence_low():
    with pytest.raises(ValueError):
        make_entry(0, 0.0, -0.1, "pca")


def test_entry_repr():
    e = make_entry(0, 45.0, 0.8, "pca")
    r = repr(e)
    assert "45.00" in r
    assert "pca" in r


# ─── filter_by_confidence ─────────────────────────────────────────────────────

def test_filter_by_confidence_basic():
    entries = [make_entry(i, 0.0, c, "pca") for i, c in enumerate([0.9, 0.3, 0.7])]
    filtered = filter_by_confidence(entries, 0.7)
    assert len(filtered) == 2
    assert all(e.confidence >= 0.7 for e in filtered)


def test_filter_by_confidence_zero_threshold():
    entries = [make_entry(0, 0.0, 0.0, "pca")]
    filtered = filter_by_confidence(entries, 0.0)
    assert len(filtered) == 1


def test_filter_by_confidence_all_removed():
    entries = [make_entry(0, 0.0, 0.5, "pca")]
    filtered = filter_by_confidence(entries, 0.9)
    assert len(filtered) == 0


# ─── filter_by_method ─────────────────────────────────────────────────────────

def test_filter_by_method():
    entries = [
        make_entry(0, 0.0, 0.5, "pca"),
        make_entry(1, 0.0, 0.7, "moments"),
        make_entry(2, 0.0, 0.6, "pca"),
    ]
    filtered = filter_by_method(entries, "pca")
    assert len(filtered) == 2


# ─── filter_by_angle_range ────────────────────────────────────────────────────

def test_filter_by_angle_range():
    entries = [make_entry(i, float(a), 0.5, "pca") for i, a in enumerate([0, 30, 60, 90])]
    filtered = filter_by_angle_range(entries, 20.0, 70.0)
    assert len(filtered) == 2


def test_filter_by_angle_range_invalid():
    entries = [make_entry(0, 0.0, 0.5, "pca")]
    with pytest.raises(ValueError):
        filter_by_angle_range(entries, 90.0, 10.0)


def test_filter_by_angle_range_inclusive():
    entries = [make_entry(0, 45.0, 0.5, "pca")]
    filtered = filter_by_angle_range(entries, 45.0, 45.0)
    assert len(filtered) == 1


# ─── rank_by_confidence ───────────────────────────────────────────────────────

def test_rank_by_confidence_descending():
    entries = [make_entry(i, 0.0, c, "pca") for i, c in enumerate([0.3, 0.9, 0.6])]
    ranked = rank_by_confidence(entries)
    assert ranked[0].confidence == 0.9
    assert ranked[-1].confidence == 0.3


def test_rank_by_confidence_ascending():
    entries = [make_entry(i, 0.0, c, "pca") for i, c in enumerate([0.3, 0.9, 0.6])]
    ranked = rank_by_confidence(entries, reverse=False)
    assert ranked[0].confidence == 0.3


def test_rank_by_confidence_preserves_length():
    entries = [make_entry(i, 0.0, 0.5, "pca") for i in range(5)]
    assert len(rank_by_confidence(entries)) == 5


# ─── best_entry ───────────────────────────────────────────────────────────────

def test_best_entry_basic():
    entries = [make_entry(i, 0.0, c, "pca") for i, c in enumerate([0.3, 0.9, 0.6])]
    b = best_entry(entries)
    assert b.confidence == 0.9


def test_best_entry_empty():
    assert best_entry([]) is None


def test_best_entry_with_filter():
    entries = [make_entry(i, 0.0, c, "pca") for i, c in enumerate([0.3, 0.9, 0.6])]
    cfg = RotationScoreConfig(min_confidence=0.8)
    b = best_entry(entries, cfg)
    assert b.confidence == 0.9


def test_best_entry_preferred_method():
    entries = [
        make_entry(0, 0.0, 0.9, "pca"),
        make_entry(1, 0.0, 0.7, "moments"),
    ]
    cfg = RotationScoreConfig(preferred_method="moments")
    b = best_entry(entries, cfg)
    assert b.method == "moments"


def test_best_entry_all_filtered():
    entries = [make_entry(0, 0.0, 0.3, "pca")]
    cfg = RotationScoreConfig(min_confidence=0.9)
    assert best_entry(entries, cfg) is None


# ─── aggregate_angles ────────────────────────────────────────────────────────

def test_aggregate_angles_empty():
    assert aggregate_angles([]) == 0.0


def test_aggregate_angles_uniform():
    entries = [make_entry(i, 45.0, 0.5, "pca") for i in range(3)]
    result = aggregate_angles(entries)
    assert abs(result - 45.0) < 1e-9


def test_aggregate_angles_weighted():
    entries = [
        make_entry(0, 0.0, 0.0, "pca"),  # zero weight
        make_entry(1, 90.0, 1.0, "pca"),  # full weight
    ]
    result = aggregate_angles(entries)
    assert abs(result - 90.0) < 1e-9


def test_aggregate_angles_custom_weights():
    entries = [make_entry(0, 0.0, 0.5, "pca"), make_entry(1, 100.0, 0.5, "pca")]
    result = aggregate_angles(entries, weights=[1.0, 0.0])
    assert abs(result - 0.0) < 1e-9


# ─── rotation_score_stats ─────────────────────────────────────────────────────

def test_rotation_score_stats_empty():
    d = rotation_score_stats([])
    assert d["n"] == 0
    assert d["methods"] == []


def test_rotation_score_stats_basic():
    entries = [make_entry(i, float(a), c, m) for i, (a, c, m) in
               enumerate([(0.0, 0.8, "pca"), (90.0, 0.6, "moments")])]
    d = rotation_score_stats(entries)
    assert d["n"] == 2
    assert abs(d["mean_angle"] - 45.0) < 1e-9
    assert d["std_angle"] >= 0
    assert abs(d["mean_confidence"] - 0.7) < 1e-9
    assert "pca" in d["methods"] or "moments" in d["methods"]


# ─── angle_agreement ──────────────────────────────────────────────────────────

def test_angle_agreement_all_same():
    entries = [make_entry(i, 45.0, 0.5, "pca") for i in range(3)]
    assert angle_agreement(entries) == 1.0


def test_angle_agreement_single():
    entries = [make_entry(0, 45.0, 0.5, "pca")]
    assert angle_agreement(entries) == 1.0


def test_angle_agreement_no_agreement():
    entries = [make_entry(i, float(a), 0.5, "pca") for i, a in enumerate([0.0, 90.0])]
    result = angle_agreement(entries, tolerance_deg=5.0)
    assert result == 0.0


def test_angle_agreement_partial():
    entries = [make_entry(i, float(a), 0.5, "pca") for i, a in enumerate([0.0, 2.0, 90.0])]
    result = angle_agreement(entries, tolerance_deg=5.0)
    # Only pair (0, 2) agrees, pairs (0, 90) and (2, 90) don't
    assert 0.0 < result < 1.0


# ─── batch_make_entries ───────────────────────────────────────────────────────

def test_batch_make_entries_basic():
    ids = [0, 1, 2]
    angles = [0.0, 45.0, 90.0]
    confs = [0.5, 0.7, 0.9]
    methods = ["pca", "pca", "moments"]
    entries = batch_make_entries(ids, angles, confs, methods)
    assert len(entries) == 3
    assert entries[1].angle_deg == 45.0


def test_batch_make_entries_length_mismatch():
    with pytest.raises(ValueError):
        batch_make_entries([0, 1], [0.0], [0.5, 0.5], ["pca", "pca"])


# ─── top_k_entries ────────────────────────────────────────────────────────────

def test_top_k_entries():
    entries = [make_entry(i, 0.0, c, "pca") for i, c in enumerate([0.3, 0.9, 0.6])]
    top = top_k_entries(entries, 2)
    assert len(top) == 2
    assert top[0].confidence >= top[1].confidence


def test_top_k_entries_with_filter():
    entries = [make_entry(i, 0.0, c, "pca") for i, c in enumerate([0.3, 0.9, 0.6])]
    cfg = RotationScoreConfig(min_confidence=0.5)
    top = top_k_entries(entries, 2, cfg)
    assert all(e.confidence >= 0.5 for e in top)


# ─── group_by_method ──────────────────────────────────────────────────────────

def test_group_by_method():
    entries = [
        make_entry(0, 0.0, 0.5, "pca"),
        make_entry(1, 0.0, 0.7, "moments"),
        make_entry(2, 0.0, 0.6, "pca"),
    ]
    groups = group_by_method(entries)
    assert set(groups.keys()) == {"pca", "moments"}
    assert len(groups["pca"]) == 2
    assert len(groups["moments"]) == 1


def test_group_by_method_empty():
    groups = group_by_method([])
    assert groups == {}
