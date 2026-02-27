"""Tests for puzzle_reconstruction.utils.window_tile_records."""
import pytest
import numpy as np

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

np.random.seed(88)


# ── WindowOpRecord ────────────────────────────────────────────────────────────

def test_window_op_record_basic():
    r = make_window_op_record("mean", 100, 10, 5, 19)
    assert r.operation == "mean"
    assert r.signal_length == 100
    assert r.window_size == 10
    assert r.step == 5
    assert r.n_windows == 19


def test_window_op_record_invalid_operation():
    with pytest.raises(ValueError):
        WindowOpRecord(operation="unknown", signal_length=10, window_size=5, step=1, n_windows=6)


def test_window_op_record_invalid_signal_length():
    with pytest.raises(ValueError):
        WindowOpRecord(operation="mean", signal_length=0, window_size=5, step=1, n_windows=0)


def test_window_op_record_invalid_window_size():
    with pytest.raises(ValueError):
        WindowOpRecord(operation="std", signal_length=10, window_size=0, step=1, n_windows=10)


def test_window_op_record_invalid_step():
    with pytest.raises(ValueError):
        WindowOpRecord(operation="max", signal_length=10, window_size=3, step=0, n_windows=8)


def test_window_op_record_coverage_full():
    r = make_window_op_record("max", 100, 10, 10, 10)
    assert r.coverage == pytest.approx(1.0)


def test_window_op_record_coverage_partial():
    r = make_window_op_record("min", 100, 10, 10, 5)
    assert r.coverage == pytest.approx(0.5)


def test_window_op_record_has_overlap_true():
    r = make_window_op_record("mean", 100, 10, 5, 19)
    assert r.has_overlap is True


def test_window_op_record_has_overlap_false():
    r = make_window_op_record("std", 100, 10, 10, 10)
    assert r.has_overlap is False


def test_window_op_record_label():
    r = make_window_op_record("split", 50, 5, 5, 10, label="test")
    assert r.label == "test"


def test_window_op_record_all_valid_ops():
    for op in ["mean", "std", "max", "min", "split", "merge"]:
        r = WindowOpRecord(operation=op, signal_length=10, window_size=3, step=1, n_windows=8)
        assert r.operation == op


# ── WindowFunctionRecord ──────────────────────────────────────────────────────

def test_window_function_record_basic():
    r = WindowFunctionRecord(func_name="hann", input_length=64)
    assert r.func_name == "hann"
    assert r.input_length == 64


def test_window_function_record_invalid_func():
    with pytest.raises(ValueError):
        WindowFunctionRecord(func_name="unknown_window", input_length=64)


def test_window_function_record_invalid_length():
    with pytest.raises(ValueError):
        WindowFunctionRecord(func_name="rect", input_length=0)


def test_window_function_record_attenuation_ratio():
    r = WindowFunctionRecord(func_name="hamming", input_length=64,
                              sum_before=100.0, sum_after=54.0)
    assert r.attenuation_ratio == pytest.approx(0.54)


def test_window_function_record_attenuation_ratio_zero_before():
    r = WindowFunctionRecord(func_name="rect", input_length=64,
                              sum_before=0.0, sum_after=10.0)
    assert r.attenuation_ratio == pytest.approx(1.0)


def test_window_function_record_all_valid_funcs():
    for fn in ["rect", "hann", "hamming", "bartlett", "blackman"]:
        r = WindowFunctionRecord(func_name=fn, input_length=32)
        assert r.func_name == fn


# ── TileOpRecord ──────────────────────────────────────────────────────────────

def test_tile_op_record_basic():
    r = make_tile_op_record((100, 100), 10, 10, 100)
    assert r.image_shape == (100, 100)
    assert r.tile_h == 10
    assert r.tile_w == 10
    assert r.n_tiles == 100


def test_tile_op_record_tile_area():
    r = make_tile_op_record((200, 300), 20, 30, 10)
    assert r.tile_area == 600


def test_tile_op_record_image_area():
    r = make_tile_op_record((200, 300), 10, 10, 5)
    assert r.image_area == 60000


def test_tile_op_record_coverage_ratio_full():
    r = make_tile_op_record((100, 100), 10, 10, 100)
    assert r.coverage_ratio == pytest.approx(1.0)


def test_tile_op_record_coverage_ratio_partial():
    r = make_tile_op_record((100, 100), 10, 10, 50)
    assert r.coverage_ratio == pytest.approx(0.5)


def test_tile_op_record_invalid_operation():
    with pytest.raises(ValueError):
        TileOpRecord(operation="invalid", image_shape=(100, 100), tile_h=10, tile_w=10, n_tiles=10)


def test_tile_op_record_invalid_tile_h():
    with pytest.raises(ValueError):
        TileOpRecord(operation="tile", image_shape=(100, 100), tile_h=0, tile_w=10, n_tiles=10)


# ── TileFilterRecord ──────────────────────────────────────────────────────────

def test_tile_filter_record_n_removed():
    r = TileFilterRecord(n_input=100, n_kept=70, min_foreground=0.1)
    assert r.n_removed == 30


def test_tile_filter_record_retention_rate():
    r = TileFilterRecord(n_input=100, n_kept=80, min_foreground=0.2)
    assert r.retention_rate == pytest.approx(0.8)


def test_tile_filter_record_empty_input():
    r = TileFilterRecord(n_input=0, n_kept=0, min_foreground=0.1)
    assert r.retention_rate == pytest.approx(1.0)


def test_tile_filter_record_n_kept_exceeds_raises():
    with pytest.raises(ValueError):
        TileFilterRecord(n_input=10, n_kept=15, min_foreground=0.1)


def test_tile_filter_record_invalid_min_foreground():
    with pytest.raises(ValueError):
        TileFilterRecord(n_input=10, n_kept=5, min_foreground=1.5)


def test_tile_filter_record_invalid_n_input():
    with pytest.raises(ValueError):
        TileFilterRecord(n_input=-1, n_kept=0, min_foreground=0.1)


# ── OverlapSummaryRecord ──────────────────────────────────────────────────────

def test_overlap_summary_is_valid_true():
    r = OverlapSummaryRecord(n_fragments=5, n_pairs_checked=10, n_overlapping_pairs=0)
    assert r.is_valid is True


def test_overlap_summary_is_valid_false():
    r = OverlapSummaryRecord(n_fragments=5, n_pairs_checked=10, n_overlapping_pairs=2)
    assert r.is_valid is False


def test_overlap_summary_overlap_rate():
    r = OverlapSummaryRecord(n_fragments=5, n_pairs_checked=10, n_overlapping_pairs=3)
    assert r.overlap_rate == pytest.approx(0.3)


def test_overlap_summary_invalid_max_iou():
    with pytest.raises(ValueError):
        OverlapSummaryRecord(n_fragments=1, n_pairs_checked=1, n_overlapping_pairs=0, max_iou=1.5)


# ── ScoreSummaryRecord ────────────────────────────────────────────────────────

def test_score_summary_status_pass():
    r = ScoreSummaryRecord(n_metrics=3, total_score=0.8, passed=True, pass_threshold=0.7)
    assert r.status == "pass"
    assert r.margin == pytest.approx(0.1)


def test_score_summary_status_fail():
    r = ScoreSummaryRecord(n_metrics=3, total_score=0.5, passed=False, pass_threshold=0.7)
    assert r.status == "fail"
    assert r.margin == pytest.approx(-0.2)


def test_score_summary_invalid_total_score():
    with pytest.raises(ValueError):
        ScoreSummaryRecord(n_metrics=1, total_score=1.5, passed=True, pass_threshold=0.5)


def test_score_summary_invalid_pass_threshold():
    with pytest.raises(ValueError):
        ScoreSummaryRecord(n_metrics=1, total_score=0.5, passed=False, pass_threshold=-0.1)
