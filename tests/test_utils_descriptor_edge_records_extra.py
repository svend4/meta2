"""Extra tests for puzzle_reconstruction/utils/descriptor_edge_records.py"""
import pytest
from puzzle_reconstruction.utils.descriptor_edge_records import (
    DescriptorCombineRecord,
    EdgeProfileRecord,
    ProfileMatchRecord,
    EdgeValidRecord,
)


# ─── DescriptorCombineRecord ──────────────────────────────────────────────────

class TestDescriptorCombineRecordExtra:
    def test_basic_fields(self):
        rec = DescriptorCombineRecord(
            fragment_id=1, used_names=["hog", "lbp"],
            original_dim=128, combined_dim=64
        )
        assert rec.fragment_id == 1
        assert rec.used_names == ["hog", "lbp"]
        assert rec.original_dim == 128
        assert rec.combined_dim == 64

    def test_defaults(self):
        rec = DescriptorCombineRecord(
            fragment_id=0, used_names=[], original_dim=64, combined_dim=64
        )
        assert rec.normalized is True
        assert rec.l2_final is True

    def test_is_reduced_true(self):
        rec = DescriptorCombineRecord(
            fragment_id=2, used_names=["sift"],
            original_dim=128, combined_dim=64
        )
        assert rec.is_reduced is True

    def test_is_reduced_false_equal(self):
        rec = DescriptorCombineRecord(
            fragment_id=3, used_names=["hog"],
            original_dim=64, combined_dim=64
        )
        assert rec.is_reduced is False

    def test_is_reduced_false_larger(self):
        rec = DescriptorCombineRecord(
            fragment_id=4, used_names=["a", "b"],
            original_dim=64, combined_dim=128
        )
        assert rec.is_reduced is False

    def test_compression_ratio_normal(self):
        rec = DescriptorCombineRecord(
            fragment_id=5, used_names=["x"],
            original_dim=100, combined_dim=25
        )
        assert abs(rec.compression_ratio - 0.25) < 1e-9

    def test_compression_ratio_zero_original(self):
        rec = DescriptorCombineRecord(
            fragment_id=6, used_names=[],
            original_dim=0, combined_dim=0
        )
        assert rec.compression_ratio == 0.0

    def test_override_defaults(self):
        rec = DescriptorCombineRecord(
            fragment_id=7, used_names=["lbp"],
            original_dim=32, combined_dim=16,
            normalized=False, l2_final=False
        )
        assert rec.normalized is False
        assert rec.l2_final is False


# ─── EdgeProfileRecord ────────────────────────────────────────────────────────

class TestEdgeProfileRecordExtra:
    def test_basic_fields(self):
        rec = EdgeProfileRecord(
            fragment_id=10, side=1, method="mean",
            n_samples=32, signal_mean=0.5, signal_std=0.1
        )
        assert rec.fragment_id == 10
        assert rec.side == 1
        assert rec.method == "mean"
        assert rec.n_samples == 32

    def test_is_uniform_true(self):
        rec = EdgeProfileRecord(
            fragment_id=0, side=0, method="linear",
            n_samples=16, signal_mean=1.0, signal_std=1e-8
        )
        assert rec.is_uniform is True

    def test_is_uniform_false(self):
        rec = EdgeProfileRecord(
            fragment_id=1, side=2, method="gaussian",
            n_samples=64, signal_mean=0.3, signal_std=0.5
        )
        assert rec.is_uniform is False

    def test_is_uniform_boundary(self):
        # Exactly at 1e-6 should NOT be uniform (strict <)
        rec = EdgeProfileRecord(
            fragment_id=2, side=0, method="raw",
            n_samples=8, signal_mean=0.0, signal_std=1e-6
        )
        assert rec.is_uniform is False

    def test_is_uniform_zero_std(self):
        rec = EdgeProfileRecord(
            fragment_id=3, side=3, method="raw",
            n_samples=4, signal_mean=0.0, signal_std=0.0
        )
        assert rec.is_uniform is True

    def test_signal_mean_stored(self):
        rec = EdgeProfileRecord(
            fragment_id=5, side=0, method="m",
            n_samples=10, signal_mean=3.14, signal_std=0.01
        )
        assert abs(rec.signal_mean - 3.14) < 1e-9

    def test_method_stored(self):
        rec = EdgeProfileRecord(
            fragment_id=6, side=1, method="custom_method",
            n_samples=20, signal_mean=0.0, signal_std=0.0
        )
        assert rec.method == "custom_method"


# ─── ProfileMatchRecord ───────────────────────────────────────────────────────

class TestProfileMatchRecordExtra:
    def test_basic_fields(self):
        rec = ProfileMatchRecord(
            idx1=0, idx2=1, side1=0, side2=1,
            score=0.8, correlation=0.7, dtw_score=0.5
        )
        assert rec.idx1 == 0
        assert rec.idx2 == 1
        assert rec.n_samples == 64  # default

    def test_pair_key_ordered(self):
        rec = ProfileMatchRecord(
            idx1=5, idx2=2, side1=0, side2=1,
            score=0.7, correlation=0.6, dtw_score=0.4
        )
        assert rec.pair_key == (2, 5)

    def test_pair_key_already_ordered(self):
        rec = ProfileMatchRecord(
            idx1=1, idx2=9, side1=0, side2=0,
            score=0.9, correlation=0.8, dtw_score=0.3
        )
        assert rec.pair_key == (1, 9)

    def test_is_good_match_above_threshold(self):
        rec = ProfileMatchRecord(
            idx1=0, idx2=1, side1=0, side2=1,
            score=0.75, correlation=0.5, dtw_score=0.2
        )
        assert rec.is_good_match is True

    def test_is_good_match_at_threshold(self):
        rec = ProfileMatchRecord(
            idx1=0, idx2=1, side1=0, side2=1,
            score=0.6, correlation=0.5, dtw_score=0.2
        )
        assert rec.is_good_match is True

    def test_is_good_match_below_threshold(self):
        rec = ProfileMatchRecord(
            idx1=0, idx2=2, side1=1, side2=2,
            score=0.59, correlation=0.3, dtw_score=0.8
        )
        assert rec.is_good_match is False

    def test_custom_n_samples(self):
        rec = ProfileMatchRecord(
            idx1=0, idx2=1, side1=0, side2=1,
            score=0.5, correlation=0.4, dtw_score=0.6,
            n_samples=128
        )
        assert rec.n_samples == 128

    def test_pair_key_same_indices(self):
        rec = ProfileMatchRecord(
            idx1=3, idx2=3, side1=0, side2=1,
            score=1.0, correlation=1.0, dtw_score=0.0
        )
        assert rec.pair_key == (3, 3)


# ─── EdgeValidRecord ──────────────────────────────────────────────────────────

class TestEdgeValidRecordExtra:
    def test_basic_fields(self):
        rec = EdgeValidRecord(
            idx1=0, idx2=1, valid=True, n_passed=8, n_failed=2
        )
        assert rec.idx1 == 0
        assert rec.idx2 == 1
        assert rec.valid is True
        assert rec.n_passed == 8
        assert rec.n_failed == 2

    def test_defaults(self):
        rec = EdgeValidRecord(
            idx1=0, idx2=1, valid=False, n_passed=0, n_failed=5
        )
        assert rec.intensity_value == 0.0
        assert rec.gap_value == 0.0
        assert rec.normal_value == 0.0

    def test_custom_values(self):
        rec = EdgeValidRecord(
            idx1=2, idx2=3, valid=True, n_passed=4, n_failed=1,
            intensity_value=0.5, gap_value=0.3, normal_value=0.7
        )
        assert abs(rec.intensity_value - 0.5) < 1e-9
        assert abs(rec.gap_value - 0.3) < 1e-9
        assert abs(rec.normal_value - 0.7) < 1e-9

    def test_valid_false(self):
        rec = EdgeValidRecord(
            idx1=5, idx2=6, valid=False, n_passed=0, n_failed=10
        )
        assert rec.valid is False

    def test_pass_rate_basic(self):
        rec = EdgeValidRecord(
            idx1=0, idx2=1, valid=True, n_passed=7, n_failed=3
        )
        assert abs(rec.pass_rate - 0.7) < 1e-9

    def test_pass_rate_all_passed(self):
        rec = EdgeValidRecord(
            idx1=0, idx2=1, valid=True, n_passed=10, n_failed=0
        )
        assert rec.pass_rate == 1.0

    def test_pass_rate_zero_total(self):
        rec = EdgeValidRecord(
            idx1=0, idx2=1, valid=False, n_passed=0, n_failed=0
        )
        assert rec.pass_rate == 0.0
