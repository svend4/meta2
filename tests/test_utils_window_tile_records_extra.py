"""Extra tests for puzzle_reconstruction/utils/window_tile_records.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.window_tile_records import (
    WindowOpRecord,
    WindowFunctionRecord,
    TileOpRecord,
    TileFilterRecord,
    OverlapSummaryRecord,
    ScoreSummaryRecord,
    make_window_op_record,
    make_tile_op_record,
)


# ─── WindowOpRecord ─────────────────────────────────────────────────────────

class TestWindowOpRecordExtra:
    def test_valid_operations(self):
        for op in ("mean", "std", "max", "min", "split", "merge"):
            r = WindowOpRecord(operation=op, signal_length=100,
                               window_size=10, step=5, n_windows=19)
            assert r.operation == op

    def test_invalid_operation_raises(self):
        with pytest.raises(ValueError):
            WindowOpRecord(operation="invalid", signal_length=100,
                           window_size=10, step=5, n_windows=19)

    def test_zero_signal_length_raises(self):
        with pytest.raises(ValueError):
            WindowOpRecord(operation="mean", signal_length=0,
                           window_size=10, step=5, n_windows=0)

    def test_zero_window_size_raises(self):
        with pytest.raises(ValueError):
            WindowOpRecord(operation="mean", signal_length=100,
                           window_size=0, step=5, n_windows=0)

    def test_zero_step_raises(self):
        with pytest.raises(ValueError):
            WindowOpRecord(operation="mean", signal_length=100,
                           window_size=10, step=0, n_windows=0)

    def test_negative_n_windows_raises(self):
        with pytest.raises(ValueError):
            WindowOpRecord(operation="mean", signal_length=100,
                           window_size=10, step=5, n_windows=-1)

    def test_coverage_full(self):
        r = WindowOpRecord(operation="mean", signal_length=100,
                           window_size=50, step=50, n_windows=2)
        assert r.coverage == pytest.approx(1.0)

    def test_coverage_partial(self):
        r = WindowOpRecord(operation="mean", signal_length=100,
                           window_size=10, step=10, n_windows=5)
        assert r.coverage == pytest.approx(0.5)

    def test_has_overlap_true(self):
        r = WindowOpRecord(operation="mean", signal_length=100,
                           window_size=10, step=5, n_windows=19)
        assert r.has_overlap is True

    def test_has_overlap_false(self):
        r = WindowOpRecord(operation="mean", signal_length=100,
                           window_size=10, step=10, n_windows=10)
        assert r.has_overlap is False

    def test_label_default_empty(self):
        r = WindowOpRecord(operation="mean", signal_length=100,
                           window_size=10, step=5, n_windows=19)
        assert r.label == ""


# ─── WindowFunctionRecord ────────────────────────────────────────────────────

class TestWindowFunctionRecordExtra:
    def test_valid_functions(self):
        for fn in ("rect", "hann", "hamming", "bartlett", "blackman"):
            r = WindowFunctionRecord(func_name=fn, input_length=256)
            assert r.func_name == fn

    def test_invalid_func_raises(self):
        with pytest.raises(ValueError):
            WindowFunctionRecord(func_name="kaiser", input_length=256)

    def test_zero_input_length_raises(self):
        with pytest.raises(ValueError):
            WindowFunctionRecord(func_name="hann", input_length=0)

    def test_attenuation_ratio_no_change(self):
        r = WindowFunctionRecord(func_name="rect", input_length=100,
                                 sum_before=10.0, sum_after=10.0)
        assert r.attenuation_ratio == pytest.approx(1.0)

    def test_attenuation_ratio_half(self):
        r = WindowFunctionRecord(func_name="hann", input_length=100,
                                 sum_before=10.0, sum_after=5.0)
        assert r.attenuation_ratio == pytest.approx(0.5)

    def test_attenuation_ratio_zero_before(self):
        r = WindowFunctionRecord(func_name="hann", input_length=100,
                                 sum_before=0.0, sum_after=5.0)
        assert r.attenuation_ratio == pytest.approx(1.0)


# ─── TileOpRecord ───────────────────────────────────────────────────────────

class TestTileOpRecordExtra:
    def test_valid_operations(self):
        for op in ("tile", "reassemble", "filter"):
            r = TileOpRecord(operation=op, image_shape=(100, 200),
                             tile_h=10, tile_w=20, n_tiles=100)
            assert r.operation == op

    def test_invalid_operation_raises(self):
        with pytest.raises(ValueError):
            TileOpRecord(operation="invalid", image_shape=(100, 200),
                         tile_h=10, tile_w=20, n_tiles=0)

    def test_zero_tile_h_raises(self):
        with pytest.raises(ValueError):
            TileOpRecord(operation="tile", image_shape=(100, 200),
                         tile_h=0, tile_w=20, n_tiles=0)

    def test_zero_tile_w_raises(self):
        with pytest.raises(ValueError):
            TileOpRecord(operation="tile", image_shape=(100, 200),
                         tile_h=10, tile_w=0, n_tiles=0)

    def test_negative_n_tiles_raises(self):
        with pytest.raises(ValueError):
            TileOpRecord(operation="tile", image_shape=(100, 200),
                         tile_h=10, tile_w=20, n_tiles=-1)

    def test_tile_area(self):
        r = TileOpRecord(operation="tile", image_shape=(100, 200),
                         tile_h=10, tile_w=20, n_tiles=100)
        assert r.tile_area == 200

    def test_image_area(self):
        r = TileOpRecord(operation="tile", image_shape=(100, 200),
                         tile_h=10, tile_w=20, n_tiles=100)
        assert r.image_area == 20000

    def test_coverage_ratio_full(self):
        r = TileOpRecord(operation="tile", image_shape=(100, 100),
                         tile_h=10, tile_w=10, n_tiles=100)
        assert r.coverage_ratio == pytest.approx(1.0)

    def test_coverage_ratio_capped(self):
        r = TileOpRecord(operation="tile", image_shape=(10, 10),
                         tile_h=10, tile_w=10, n_tiles=200)
        assert r.coverage_ratio == pytest.approx(1.0)

    def test_coverage_ratio_zero_area(self):
        r = TileOpRecord(operation="tile", image_shape=(0, 100),
                         tile_h=10, tile_w=10, n_tiles=5)
        assert r.coverage_ratio == pytest.approx(0.0)


# ─── TileFilterRecord ───────────────────────────────────────────────────────

class TestTileFilterRecordExtra:
    def test_n_removed(self):
        r = TileFilterRecord(n_input=10, n_kept=7, min_foreground=0.5)
        assert r.n_removed == 3

    def test_retention_rate_full(self):
        r = TileFilterRecord(n_input=10, n_kept=10, min_foreground=0.0)
        assert r.retention_rate == pytest.approx(1.0)

    def test_retention_rate_zero_input(self):
        r = TileFilterRecord(n_input=0, n_kept=0, min_foreground=0.5)
        assert r.retention_rate == pytest.approx(1.0)

    def test_negative_n_input_raises(self):
        with pytest.raises(ValueError):
            TileFilterRecord(n_input=-1, n_kept=0, min_foreground=0.5)

    def test_n_kept_exceeds_input_raises(self):
        with pytest.raises(ValueError):
            TileFilterRecord(n_input=5, n_kept=10, min_foreground=0.5)

    def test_min_foreground_out_of_range_raises(self):
        with pytest.raises(ValueError):
            TileFilterRecord(n_input=10, n_kept=5, min_foreground=1.5)

    def test_negative_min_foreground_raises(self):
        with pytest.raises(ValueError):
            TileFilterRecord(n_input=10, n_kept=5, min_foreground=-0.1)


# ─── OverlapSummaryRecord ───────────────────────────────────────────────────

class TestOverlapSummaryRecordExtra:
    def test_is_valid_no_overlap(self):
        r = OverlapSummaryRecord(n_fragments=5, n_pairs_checked=10,
                                  n_overlapping_pairs=0)
        assert r.is_valid is True

    def test_is_valid_with_overlap(self):
        r = OverlapSummaryRecord(n_fragments=5, n_pairs_checked=10,
                                  n_overlapping_pairs=3)
        assert r.is_valid is False

    def test_overlap_rate(self):
        r = OverlapSummaryRecord(n_fragments=5, n_pairs_checked=10,
                                  n_overlapping_pairs=2)
        assert r.overlap_rate == pytest.approx(0.2)

    def test_overlap_rate_zero_checked(self):
        r = OverlapSummaryRecord(n_fragments=0, n_pairs_checked=0,
                                  n_overlapping_pairs=0)
        assert r.overlap_rate == pytest.approx(0.0)

    def test_negative_fragments_raises(self):
        with pytest.raises(ValueError):
            OverlapSummaryRecord(n_fragments=-1, n_pairs_checked=0,
                                  n_overlapping_pairs=0)

    def test_negative_overlapping_raises(self):
        with pytest.raises(ValueError):
            OverlapSummaryRecord(n_fragments=5, n_pairs_checked=10,
                                  n_overlapping_pairs=-1)

    def test_invalid_max_iou_raises(self):
        with pytest.raises(ValueError):
            OverlapSummaryRecord(n_fragments=5, n_pairs_checked=10,
                                  n_overlapping_pairs=0, max_iou=1.5)

    def test_negative_overlap_area_raises(self):
        with pytest.raises(ValueError):
            OverlapSummaryRecord(n_fragments=5, n_pairs_checked=10,
                                  n_overlapping_pairs=0,
                                  total_overlap_area=-1.0)


# ─── ScoreSummaryRecord ─────────────────────────────────────────────────────

class TestScoreSummaryRecordExtra:
    def test_status_pass(self):
        r = ScoreSummaryRecord(n_metrics=3, total_score=0.8,
                                passed=True, pass_threshold=0.5)
        assert r.status == "pass"

    def test_status_fail(self):
        r = ScoreSummaryRecord(n_metrics=3, total_score=0.3,
                                passed=False, pass_threshold=0.5)
        assert r.status == "fail"

    def test_margin_positive(self):
        r = ScoreSummaryRecord(n_metrics=3, total_score=0.8,
                                passed=True, pass_threshold=0.5)
        assert r.margin == pytest.approx(0.3)

    def test_margin_negative(self):
        r = ScoreSummaryRecord(n_metrics=3, total_score=0.3,
                                passed=False, pass_threshold=0.5)
        assert r.margin == pytest.approx(-0.2)

    def test_worst_metric_none(self):
        r = ScoreSummaryRecord(n_metrics=3, total_score=0.8,
                                passed=True, pass_threshold=0.5)
        assert r.worst_metric is None

    def test_worst_metric_stored(self):
        r = ScoreSummaryRecord(n_metrics=3, total_score=0.3,
                                passed=False, pass_threshold=0.5,
                                worst_metric="color")
        assert r.worst_metric == "color"

    def test_invalid_total_score_raises(self):
        with pytest.raises(ValueError):
            ScoreSummaryRecord(n_metrics=3, total_score=1.5,
                                passed=True, pass_threshold=0.5)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            ScoreSummaryRecord(n_metrics=3, total_score=0.5,
                                passed=True, pass_threshold=-0.1)


# ─── make_window_op_record ──────────────────────────────────────────────────

class TestMakeWindowOpRecordExtra:
    def test_returns_record(self):
        r = make_window_op_record("mean", 100, 10, 5, 19)
        assert isinstance(r, WindowOpRecord)

    def test_fields_assigned(self):
        r = make_window_op_record("std", 200, 20, 10, 19, label="test")
        assert r.operation == "std"
        assert r.signal_length == 200
        assert r.label == "test"


# ─── make_tile_op_record ────────────────────────────────────────────────────

class TestMakeTileOpRecordExtra:
    def test_returns_record(self):
        r = make_tile_op_record((100, 200), 10, 20, 100)
        assert isinstance(r, TileOpRecord)

    def test_default_operation(self):
        r = make_tile_op_record((100, 200), 10, 20, 100)
        assert r.operation == "tile"

    def test_custom_operation(self):
        r = make_tile_op_record((100, 200), 10, 20, 100,
                                 operation="reassemble")
        assert r.operation == "reassemble"

    def test_label_stored(self):
        r = make_tile_op_record((100, 200), 10, 20, 100, label="batch1")
        assert r.label == "batch1"
