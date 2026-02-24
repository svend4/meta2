"""Extra tests for puzzle_reconstruction/io/result_exporter.py."""
from __future__ import annotations

import json

import numpy as np
import pytest

from puzzle_reconstruction.io.result_exporter import (
    ExportConfig,
    AssemblyResult,
    to_json,
    from_json,
    to_csv,
    to_text_report,
    render_annotated_image,
    summary_table,
    export_result,
    batch_export,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _result(n=2, scores=None):
    return AssemblyResult(
        fragment_ids=list(range(n)),
        positions=[(i * 10, i * 20) for i in range(n)],
        sizes=[(30, 40) for _ in range(n)],
        canvas_w=200,
        canvas_h=200,
        scores=scores,
    )


# ─── ExportConfig ───────────────────────────────────────────────────────────

class TestExportConfigExtra:
    def test_defaults(self):
        cfg = ExportConfig()
        assert cfg.fmt == "json"
        assert cfg.indent == 2
        assert cfg.draw_ids is True
        assert cfg.font_scale == pytest.approx(0.5)

    def test_valid_formats(self):
        for fmt in ("json", "csv", "image", "text", "summary"):
            ExportConfig(fmt=fmt)

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            ExportConfig(fmt="xml")

    def test_negative_indent_raises(self):
        with pytest.raises(ValueError):
            ExportConfig(indent=-1)

    def test_zero_font_scale_raises(self):
        with pytest.raises(ValueError):
            ExportConfig(font_scale=0.0)

    def test_negative_font_scale_raises(self):
        with pytest.raises(ValueError):
            ExportConfig(font_scale=-1.0)


# ─── AssemblyResult ─────────────────────────────────────────────────────────

class TestAssemblyResultExtra:
    def test_fields_stored(self):
        r = _result(2)
        assert len(r) == 2
        assert r.canvas_w == 200

    def test_positions_mismatch_raises(self):
        with pytest.raises(ValueError):
            AssemblyResult(
                fragment_ids=[0, 1],
                positions=[(0, 0)],
                sizes=[(10, 10), (10, 10)],
                canvas_w=100, canvas_h=100,
            )

    def test_sizes_mismatch_raises(self):
        with pytest.raises(ValueError):
            AssemblyResult(
                fragment_ids=[0, 1],
                positions=[(0, 0), (10, 10)],
                sizes=[(10, 10)],
                canvas_w=100, canvas_h=100,
            )

    def test_scores_mismatch_raises(self):
        with pytest.raises(ValueError):
            AssemblyResult(
                fragment_ids=[0, 1],
                positions=[(0, 0), (10, 10)],
                sizes=[(10, 10), (10, 10)],
                canvas_w=100, canvas_h=100,
                scores=[0.5],
            )

    def test_zero_canvas_w_raises(self):
        with pytest.raises(ValueError):
            _result(0).__class__(
                fragment_ids=[], positions=[], sizes=[],
                canvas_w=0, canvas_h=100,
            )

    def test_zero_canvas_h_raises(self):
        with pytest.raises(ValueError):
            AssemblyResult(
                fragment_ids=[], positions=[], sizes=[],
                canvas_w=100, canvas_h=0,
            )

    def test_len(self):
        r = _result(3)
        assert len(r) == 3


# ─── to_json / from_json ────────────────────────────────────────────────────

class TestJsonRoundtripExtra:
    def test_roundtrip(self):
        r = _result(2, scores=[0.5, 0.9])
        s = to_json(r)
        r2 = from_json(s)
        assert len(r2) == 2
        assert r2.canvas_w == 200

    def test_json_valid(self):
        r = _result(1)
        s = to_json(r)
        data = json.loads(s)
        assert "fragment_ids" in data

    def test_negative_indent_raises(self):
        with pytest.raises(ValueError):
            to_json(_result(1), indent=-1)

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            from_json("{invalid")

    def test_metadata_preserved(self):
        r = _result(1)
        r.metadata = {"key": "val"}
        s = to_json(r)
        r2 = from_json(s)
        assert r2.metadata["key"] == "val"

    def test_no_scores_roundtrip(self):
        r = _result(2, scores=None)
        s = to_json(r)
        r2 = from_json(s)
        assert r2.scores is None


# ─── to_csv ─────────────────────────────────────────────────────────────────

class TestToCsvExtra:
    def test_header(self):
        r = _result(1)
        s = to_csv(r)
        assert "fragment_id" in s.splitlines()[0]

    def test_row_count(self):
        r = _result(3)
        s = to_csv(r)
        lines = [l for l in s.strip().splitlines() if l]
        assert len(lines) == 4  # header + 3 rows

    def test_with_scores(self):
        r = _result(2, scores=[0.5, 0.9])
        s = to_csv(r)
        assert "score" in s.splitlines()[0]

    def test_without_scores(self):
        r = _result(2)
        s = to_csv(r)
        assert "score" not in s.splitlines()[0]


# ─── to_text_report ─────────────────────────────────────────────────────────

class TestToTextReportExtra:
    def test_contains_header(self):
        r = _result(1)
        s = to_text_report(r)
        assert "ОТЧЁТ" in s or "=" in s

    def test_fragment_count(self):
        r = _result(3)
        s = to_text_report(r)
        assert "3" in s

    def test_with_metadata(self):
        r = _result(1)
        r.metadata = {"project": "test"}
        s = to_text_report(r)
        assert "project" in s


# ─── render_annotated_image ─────────────────────────────────────────────────

class TestRenderAnnotatedImageExtra:
    def test_default_canvas(self):
        r = _result(2)
        img = render_annotated_image(r)
        assert img.shape == (200, 200, 3)

    def test_custom_canvas(self):
        r = _result(1)
        canvas = np.full((200, 200, 3), 128, dtype=np.uint8)
        img = render_annotated_image(r, canvas=canvas)
        assert img.shape == (200, 200, 3)

    def test_no_ids_no_bboxes(self):
        r = _result(2)
        img = render_annotated_image(r, draw_ids=False, draw_bboxes=False)
        assert img.shape == (200, 200, 3)

    def test_zero_font_scale_raises(self):
        with pytest.raises(ValueError):
            render_annotated_image(_result(1), font_scale=0.0)

    def test_2d_canvas_raises(self):
        r = _result(1)
        canvas = np.full((200, 200), 128, dtype=np.uint8)
        with pytest.raises(ValueError):
            render_annotated_image(r, canvas=canvas)


# ─── summary_table ──────────────────────────────────────────────────────────

class TestSummaryTableExtra:
    def test_empty(self):
        tbl = summary_table([])
        assert len(tbl["n_fragments"]) == 0

    def test_single(self):
        r = _result(2, scores=[0.5, 0.9])
        tbl = summary_table([r])
        assert tbl["n_fragments"] == [2]
        assert tbl["mean_score"][0] == pytest.approx(0.7)

    def test_no_scores_none(self):
        r = _result(2)
        tbl = summary_table([r])
        assert tbl["mean_score"][0] is None

    def test_multiple(self):
        r1 = _result(1, scores=[0.5])
        r2 = _result(3, scores=[0.1, 0.2, 0.3])
        tbl = summary_table([r1, r2])
        assert len(tbl["n_fragments"]) == 2


# ─── export_result ──────────────────────────────────────────────────────────

class TestExportResultExtra:
    def test_json_format(self):
        r = _result(1)
        cfg = ExportConfig(fmt="json")
        s = export_result(r, cfg)
        assert s is not None
        json.loads(s)

    def test_csv_format(self):
        r = _result(1)
        cfg = ExportConfig(fmt="csv")
        s = export_result(r, cfg)
        assert "fragment_id" in s

    def test_text_format(self):
        r = _result(1)
        cfg = ExportConfig(fmt="text")
        s = export_result(r, cfg)
        assert s is not None and len(s) > 0

    def test_image_returns_none(self):
        r = _result(1)
        cfg = ExportConfig(fmt="image")
        s = export_result(r, cfg)
        assert s is None


# ─── batch_export ────────────────────────────────────────────────────────────

class TestBatchExportExtra:
    def test_length(self):
        results = [_result(1), _result(2)]
        cfg = ExportConfig(fmt="json")
        out = batch_export(results, cfg)
        assert len(out) == 2

    def test_empty(self):
        cfg = ExportConfig(fmt="json")
        out = batch_export([], cfg)
        assert out == []
