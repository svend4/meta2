"""Extra tests for puzzle_reconstruction/utils/rotation_score_utils.py."""
from __future__ import annotations

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _e(image_id=0, angle=45.0, confidence=0.8,
       method="pca") -> RotationScoreEntry:
    return RotationScoreEntry(image_id=image_id, angle_deg=angle,
                               confidence=confidence, method=method)


# ─── RotationScoreConfig ─────────────────────────────────────────────────────

class TestRotationScoreConfigExtra:
    def test_defaults(self):
        cfg = RotationScoreConfig()
        assert cfg.min_confidence == pytest.approx(0.0)
        assert cfg.angle_tolerance_deg == pytest.approx(5.0)

    def test_invalid_min_confidence_raises(self):
        with pytest.raises(ValueError):
            RotationScoreConfig(min_confidence=1.5)

    def test_negative_tolerance_raises(self):
        with pytest.raises(ValueError):
            RotationScoreConfig(angle_tolerance_deg=-1.0)


# ─── RotationScoreEntry ──────────────────────────────────────────────────────

class TestRotationScoreEntryExtra:
    def test_negative_image_id_raises(self):
        with pytest.raises(ValueError):
            _e(image_id=-1)

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError):
            _e(confidence=1.5)

    def test_repr_contains_angle(self):
        assert "45.00" in repr(_e())


# ─── make_entry ───────────────────────────────────────────────────────────────

class TestMakeEntryExtra:
    def test_returns_entry(self):
        e = make_entry(0, 45.0, 0.8, "pca")
        assert isinstance(e, RotationScoreEntry)

    def test_meta_stored(self):
        e = make_entry(0, 0.0, 0.5, "moments", meta={"k": 3})
        assert e.meta["k"] == 3

    def test_empty_meta_default(self):
        assert make_entry(0, 0.0, 0.5, "pca").meta == {}


# ─── filters ─────────────────────────────────────────────────────────────────

class TestFiltersExtra:
    def test_filter_by_confidence(self):
        entries = [_e(confidence=0.3), _e(confidence=0.9)]
        assert len(filter_by_confidence(entries, 0.5)) == 1

    def test_filter_by_method(self):
        entries = [_e(method="pca"), _e(method="moments")]
        assert len(filter_by_method(entries, "pca")) == 1

    def test_filter_by_angle_range(self):
        entries = [_e(angle=-10.0), _e(angle=30.0), _e(angle=100.0)]
        result = filter_by_angle_range(entries, 0.0, 50.0)
        assert len(result) == 1

    def test_filter_angle_range_invalid_raises(self):
        with pytest.raises(ValueError):
            filter_by_angle_range([], min_angle=90.0, max_angle=0.0)


# ─── rank_by_confidence ───────────────────────────────────────────────────────

class TestRankByConfidenceExtra:
    def test_descending(self):
        entries = [_e(confidence=0.3), _e(confidence=0.9)]
        ranked = rank_by_confidence(entries)
        assert ranked[0].confidence == pytest.approx(0.9)

    def test_ascending(self):
        entries = [_e(confidence=0.3), _e(confidence=0.9)]
        ranked = rank_by_confidence(entries, reverse=False)
        assert ranked[0].confidence == pytest.approx(0.3)


# ─── best_entry ───────────────────────────────────────────────────────────────

class TestBestEntryExtra:
    def test_returns_best(self):
        entries = [_e(confidence=0.4), _e(confidence=0.95)]
        assert best_entry(entries).confidence == pytest.approx(0.95)

    def test_empty_returns_none(self):
        assert best_entry([]) is None

    def test_with_preferred_method(self):
        entries = [_e(confidence=0.9, method="pca"),
                   _e(confidence=0.8, method="moments")]
        cfg = RotationScoreConfig(preferred_method="moments")
        b = best_entry(entries, cfg)
        assert b.method == "moments"


# ─── aggregate_angles ─────────────────────────────────────────────────────────

class TestAggregateAnglesExtra:
    def test_empty_returns_zero(self):
        assert aggregate_angles([]) == pytest.approx(0.0)

    def test_uniform_confidence(self):
        entries = [_e(angle=30.0, confidence=1.0),
                   _e(angle=90.0, confidence=1.0)]
        assert aggregate_angles(entries) == pytest.approx(60.0)

    def test_weighted(self):
        entries = [_e(angle=0.0, confidence=1.0),
                   _e(angle=90.0, confidence=0.0)]
        assert aggregate_angles(entries) == pytest.approx(0.0)


# ─── rotation_score_stats ────────────────────────────────────────────────────

class TestRotationScoreStatsExtra:
    def test_empty(self):
        s = rotation_score_stats([])
        assert s["n"] == 0

    def test_count_and_methods(self):
        entries = [_e(method="pca"), _e(method="moments")]
        s = rotation_score_stats(entries)
        assert s["n"] == 2
        assert len(s["methods"]) == 2


# ─── angle_agreement ─────────────────────────────────────────────────────────

class TestAngleAgreementExtra:
    def test_single_entry(self):
        assert angle_agreement([_e()]) == pytest.approx(1.0)

    def test_identical_angles(self):
        entries = [_e(angle=45.0), _e(angle=45.0)]
        assert angle_agreement(entries) == pytest.approx(1.0)

    def test_distant_angles(self):
        entries = [_e(angle=0.0), _e(angle=90.0)]
        assert angle_agreement(entries, tolerance_deg=5.0) == pytest.approx(0.0)


# ─── batch_make_entries ───────────────────────────────────────────────────────

class TestBatchMakeEntriesExtra:
    def test_returns_list(self):
        result = batch_make_entries([0, 1], [45.0, 90.0], [0.8, 0.7],
                                    ["pca", "moments"])
        assert len(result) == 2

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            batch_make_entries([0], [45.0, 90.0], [0.8], ["pca"])


# ─── top_k_entries ────────────────────────────────────────────────────────────

class TestTopKEntriesExtra:
    def test_returns_k(self):
        entries = [_e(confidence=0.3), _e(confidence=0.9), _e(confidence=0.5)]
        assert len(top_k_entries(entries, 2)) == 2

    def test_best_first(self):
        entries = [_e(confidence=0.3), _e(confidence=0.9)]
        assert top_k_entries(entries, 1)[0].confidence == pytest.approx(0.9)


# ─── group_by_method ──────────────────────────────────────────────────────────

class TestGroupByMethodExtra:
    def test_groups_correctly(self):
        entries = [_e(method="pca"), _e(method="moments"), _e(method="pca")]
        groups = group_by_method(entries)
        assert len(groups["pca"]) == 2 and len(groups["moments"]) == 1

    def test_empty(self):
        assert group_by_method([]) == {}
