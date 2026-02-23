"""Extra tests for puzzle_reconstruction.io.result_exporter."""
from __future__ import annotations

import json

import numpy as np
import pytest

from puzzle_reconstruction.io.result_exporter import (
    AssemblyResult,
    ExportConfig,
    batch_export,
    export_result,
    from_json,
    render_annotated_image,
    summary_table,
    to_csv,
    to_json,
    to_text_report,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _result(n=3, scores=None):
    return AssemblyResult(
        fragment_ids=list(range(n)),
        positions=[(i * 20, 0) for i in range(n)],
        sizes=[(15, 15)] * n,
        canvas_w=200,
        canvas_h=100,
        scores=scores,
    )


# ─── TestExportConfigExtra ────────────────────────────────────────────────────

class TestExportConfigExtra:
    def test_indent_zero_ok(self):
        cfg = ExportConfig(indent=0)
        assert cfg.indent == 0

    def test_indent_4_ok(self):
        cfg = ExportConfig(indent=4)
        assert cfg.indent == 4

    def test_draw_ids_false(self):
        cfg = ExportConfig(draw_ids=False)
        assert cfg.draw_ids is False

    def test_draw_bboxes_false(self):
        cfg = ExportConfig(draw_bboxes=False)
        assert cfg.draw_bboxes is False

    def test_font_scale_large_ok(self):
        cfg = ExportConfig(font_scale=2.0)
        assert cfg.font_scale == pytest.approx(2.0)

    def test_all_formats_accepted(self):
        for fmt in ("json", "csv", "image", "text", "summary"):
            assert ExportConfig(fmt=fmt).fmt == fmt


# ─── TestAssemblyResultExtra ──────────────────────────────────────────────────

class TestAssemblyResultExtra:
    def test_fragment_ids_stored(self):
        r = _result(4)
        assert r.fragment_ids == [0, 1, 2, 3]

    def test_positions_stored(self):
        r = _result(2)
        assert r.positions == [(0, 0), (20, 0)]

    def test_sizes_stored(self):
        r = _result(2)
        assert r.sizes == [(15, 15), (15, 15)]

    def test_canvas_w_stored(self):
        r = _result()
        assert r.canvas_w == 200

    def test_canvas_h_stored(self):
        r = _result()
        assert r.canvas_h == 100

    def test_empty_result_ok(self):
        r = AssemblyResult(
            fragment_ids=[], positions=[], sizes=[],
            canvas_w=100, canvas_h=100,
        )
        assert len(r) == 0

    def test_scores_list_stored(self):
        r = _result(3, scores=[0.5, 0.6, 0.7])
        assert r.scores == pytest.approx([0.5, 0.6, 0.7])

    def test_metadata_default_empty(self):
        r = _result()
        assert isinstance(r.metadata, dict)


# ─── TestToJsonExtra ──────────────────────────────────────────────────────────

class TestToJsonExtra:
    def test_positions_in_json(self):
        data = json.loads(to_json(_result(2)))
        assert "positions" in data

    def test_sizes_in_json(self):
        data = json.loads(to_json(_result(2)))
        assert "sizes" in data

    def test_indent_0_compact(self):
        s = to_json(_result(), indent=0)
        assert isinstance(s, str)
        json.loads(s)

    def test_indent_4_valid(self):
        s = to_json(_result(), indent=4)
        data = json.loads(s)
        assert "fragment_ids" in data

    def test_scores_none_serialized(self):
        data = json.loads(to_json(_result(2, scores=None)))
        # scores=None may be serialized as null or []
        assert data.get("scores") is None or data.get("scores") == []

    def test_empty_result_valid_json(self):
        r = AssemblyResult(
            fragment_ids=[], positions=[], sizes=[],
            canvas_w=100, canvas_h=100,
        )
        data = json.loads(to_json(r))
        assert data["fragment_ids"] == []


# ─── TestFromJsonExtra ────────────────────────────────────────────────────────

class TestFromJsonExtra:
    def test_scores_round_trip(self):
        r = _result(3, scores=[0.5, 0.6, 0.7])
        r2 = from_json(to_json(r))
        assert r2.scores == pytest.approx([0.5, 0.6, 0.7])

    def test_canvas_dimensions_round_trip(self):
        r = _result()
        r2 = from_json(to_json(r))
        assert r2.canvas_w == 200
        assert r2.canvas_h == 100

    def test_sizes_round_trip(self):
        r = _result(2)
        r2 = from_json(to_json(r))
        assert r2.sizes == [(15, 15), (15, 15)]

    def test_empty_json_object_raises(self):
        with pytest.raises((ValueError, KeyError)):
            from_json("{}")

    def test_len_preserved(self):
        r = _result(5)
        r2 = from_json(to_json(r))
        assert len(r2) == 5


# ─── TestToCsvExtra ───────────────────────────────────────────────────────────

class TestToCsvExtra:
    def test_single_fragment(self):
        r = _result(1)
        csv_str = to_csv(r)
        lines = [l for l in csv_str.strip().splitlines() if l]
        assert len(lines) == 2  # header + 1 row

    def test_five_fragments(self):
        csv_str = to_csv(_result(5))
        lines = [l for l in csv_str.strip().splitlines() if l]
        assert len(lines) == 6  # header + 5 rows

    def test_header_contains_x_y(self):
        csv_str = to_csv(_result())
        first_line = csv_str.strip().splitlines()[0]
        assert "x" in first_line or "position" in first_line.lower()

    def test_returns_string(self):
        assert isinstance(to_csv(_result()), str)


# ─── TestToTextReportExtra ────────────────────────────────────────────────────

class TestToTextReportExtra:
    def test_contains_n_fragments(self):
        txt = to_text_report(_result(5))
        assert "5" in txt

    def test_metadata_key_present(self):
        r = _result()
        r.metadata["version"] = "1.0"
        txt = to_text_report(r)
        assert "version" in txt

    def test_returns_nonempty_string(self):
        txt = to_text_report(_result())
        assert len(txt) > 0

    def test_scores_mentioned_when_present(self):
        r = _result(2, scores=[0.8, 0.9])
        txt = to_text_report(r)
        # some reference to scores
        assert "score" in txt.lower() or "0.8" in txt or "0.9" in txt


# ─── TestRenderAnnotatedImageExtra ────────────────────────────────────────────

class TestRenderAnnotatedImageExtra:
    def test_channel_count_3(self):
        img = render_annotated_image(_result())
        assert img.shape[2] == 3

    def test_non_zero_pixels(self):
        img = render_annotated_image(_result())
        # should have some drawn content
        assert img.sum() > 0

    def test_custom_font_scale(self):
        img = render_annotated_image(_result(), font_scale=1.0)
        assert img.shape == (100, 200, 3)

    def test_canvas_dimensions(self):
        r = _result()
        r2 = AssemblyResult(
            fragment_ids=[0], positions=[(0, 0)], sizes=[(10, 10)],
            canvas_w=50, canvas_h=30,
        )
        img = render_annotated_image(r2)
        assert img.shape[:2] == (30, 50)

    def test_dtype_uint8(self):
        assert render_annotated_image(_result()).dtype == np.uint8


# ─── TestSummaryTableExtra ────────────────────────────────────────────────────

class TestSummaryTableExtra:
    def test_single_result(self):
        tbl = summary_table([_result(3)])
        assert tbl["n_fragments"] == [3]

    def test_three_results(self):
        tbl = summary_table([_result(2), _result(3), _result(4)])
        assert tbl["n_fragments"] == [2, 3, 4]

    def test_canvas_w_stored(self):
        tbl = summary_table([_result()])
        assert tbl["canvas_w"] == [200]

    def test_canvas_h_stored(self):
        tbl = summary_table([_result()])
        assert tbl["canvas_h"] == [100]

    def test_score_none_returns_none(self):
        tbl = summary_table([_result(2, scores=None)])
        assert tbl["mean_score"][0] is None
        assert tbl["min_score"][0] is None
        assert tbl["max_score"][0] is None

    def test_score_stats_correct(self):
        r = _result(3, scores=[0.5, 0.7, 0.9])
        tbl = summary_table([r])
        assert tbl["mean_score"][0] == pytest.approx(0.7, abs=1e-6)
        assert tbl["min_score"][0] == pytest.approx(0.5)
        assert tbl["max_score"][0] == pytest.approx(0.9)


# ─── TestExportResultExtra ────────────────────────────────────────────────────

class TestExportResultExtra:
    def test_json_valid(self):
        cfg = ExportConfig(fmt="json")
        content = export_result(_result(), cfg)
        json.loads(content)

    def test_csv_contains_fragment_id(self):
        cfg = ExportConfig(fmt="csv")
        content = export_result(_result(), cfg)
        assert "fragment_id" in content

    def test_text_nonempty(self):
        cfg = ExportConfig(fmt="text")
        content = export_result(_result(), cfg)
        assert len(content) > 0

    def test_summary_nonempty(self):
        cfg = ExportConfig(fmt="summary")
        content = export_result(_result(), cfg)
        assert len(content) > 0


# ─── TestBatchExportExtra ─────────────────────────────────────────────────────

class TestBatchExportExtra:
    def test_json_all_valid(self):
        cfg = ExportConfig(fmt="json")
        out = batch_export([_result(2), _result(3)], cfg)
        for s in out:
            json.loads(s)

    def test_csv_all_strings(self):
        cfg = ExportConfig(fmt="csv")
        out = batch_export([_result(), _result()], cfg)
        for s in out:
            assert isinstance(s, str)

    def test_single_result_batch(self):
        cfg = ExportConfig(fmt="json")
        out = batch_export([_result()], cfg)
        assert len(out) == 1

    def test_image_returns_nones(self):
        cfg = ExportConfig(fmt="image")
        out = batch_export([_result(), _result()], cfg)
        assert all(x is None for x in out)

    def test_text_all_nonempty(self):
        cfg = ExportConfig(fmt="text")
        out = batch_export([_result()], cfg)
        assert all(len(s) > 0 for s in out)
