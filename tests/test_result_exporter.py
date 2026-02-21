"""Тесты для puzzle_reconstruction.io.result_exporter."""
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


# ─── TestExportConfig ─────────────────────────────────────────────────────────

class TestExportConfig:
    def test_default_values(self):
        cfg = ExportConfig()
        assert cfg.fmt == "json"
        assert cfg.indent == 2
        assert cfg.font_scale == pytest.approx(0.5)
        assert cfg.draw_ids is True
        assert cfg.draw_bboxes is True

    def test_valid_formats(self):
        for fmt in ("json", "csv", "image", "text", "summary"):
            cfg = ExportConfig(fmt=fmt)
            assert cfg.fmt == fmt

    def test_unknown_format_raises(self):
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


# ─── TestAssemblyResult ───────────────────────────────────────────────────────

class TestAssemblyResult:
    def test_basic_creation(self):
        r = _result(3)
        assert len(r) == 3

    def test_len(self):
        assert len(_result(5)) == 5

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
                positions=[(0, 0), (10, 0)],
                sizes=[(10, 10)],
                canvas_w=100, canvas_h=100,
            )

    def test_scores_mismatch_raises(self):
        with pytest.raises(ValueError):
            AssemblyResult(
                fragment_ids=[0, 1],
                positions=[(0, 0), (10, 0)],
                sizes=[(10, 10), (10, 10)],
                canvas_w=100, canvas_h=100,
                scores=[0.5],
            )

    def test_canvas_w_zero_raises(self):
        with pytest.raises(ValueError):
            AssemblyResult(
                fragment_ids=[], positions=[], sizes=[],
                canvas_w=0, canvas_h=100,
            )

    def test_canvas_h_zero_raises(self):
        with pytest.raises(ValueError):
            AssemblyResult(
                fragment_ids=[], positions=[], sizes=[],
                canvas_w=100, canvas_h=0,
            )

    def test_scores_none_valid(self):
        r = _result(2, scores=None)
        assert r.scores is None


# ─── TestToJson ───────────────────────────────────────────────────────────────

class TestToJson:
    def test_returns_string(self):
        assert isinstance(to_json(_result()), str)

    def test_valid_json(self):
        data = json.loads(to_json(_result()))
        assert "fragment_ids" in data

    def test_fragment_ids_present(self):
        r = _result(3)
        data = json.loads(to_json(r))
        assert data["fragment_ids"] == [0, 1, 2]

    def test_canvas_dimensions_present(self):
        r = _result()
        data = json.loads(to_json(r))
        assert data["canvas_w"] == 200
        assert data["canvas_h"] == 100

    def test_negative_indent_raises(self):
        with pytest.raises(ValueError):
            to_json(_result(), indent=-1)

    def test_scores_included(self):
        r = _result(2, scores=[0.8, 0.9])
        data = json.loads(to_json(r))
        assert data["scores"] == pytest.approx([0.8, 0.9])


# ─── TestFromJson ─────────────────────────────────────────────────────────────

class TestFromJson:
    def test_round_trip(self):
        r = _result(4)
        r2 = from_json(to_json(r))
        assert r2.fragment_ids == r.fragment_ids
        assert r2.canvas_w == r.canvas_w

    def test_positions_restored(self):
        r = _result(3)
        r2 = from_json(to_json(r))
        assert r2.positions == [(0, 0), (20, 0), (40, 0)]

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            from_json("not json {{{")

    def test_returns_assembly_result(self):
        r2 = from_json(to_json(_result()))
        assert isinstance(r2, AssemblyResult)


# ─── TestToCsv ────────────────────────────────────────────────────────────────

class TestToCsv:
    def test_returns_string(self):
        assert isinstance(to_csv(_result()), str)

    def test_has_header(self):
        csv_str = to_csv(_result())
        assert "fragment_id" in csv_str

    def test_row_count(self):
        csv_str = to_csv(_result(4))
        lines = [l for l in csv_str.strip().splitlines() if l]
        assert len(lines) == 5  # 1 заголовок + 4 строки данных

    def test_score_column_present_when_given(self):
        r = _result(2, scores=[0.7, 0.8])
        csv_str = to_csv(r)
        assert "score" in csv_str

    def test_no_score_column_when_none(self):
        csv_str = to_csv(_result(2, scores=None))
        assert "score" not in csv_str


# ─── TestToTextReport ─────────────────────────────────────────────────────────

class TestToTextReport:
    def test_returns_string(self):
        assert isinstance(to_text_report(_result()), str)

    def test_contains_canvas_info(self):
        txt = to_text_report(_result())
        assert "200" in txt
        assert "100" in txt

    def test_contains_fragment_count(self):
        txt = to_text_report(_result(4))
        assert "4" in txt

    def test_contains_fragment_ids(self):
        txt = to_text_report(_result(3))
        for fid in ["0", "1", "2"]:
            assert fid in txt

    def test_metadata_included(self):
        r = _result()
        r.metadata["author"] = "test"
        txt = to_text_report(r)
        assert "author" in txt


# ─── TestRenderAnnotatedImage ─────────────────────────────────────────────────

class TestRenderAnnotatedImage:
    def test_returns_ndarray(self):
        img = render_annotated_image(_result())
        assert isinstance(img, np.ndarray)

    def test_shape_matches_canvas(self):
        r = _result()
        img = render_annotated_image(r)
        assert img.shape == (r.canvas_h, r.canvas_w, 3)

    def test_dtype_uint8(self):
        img = render_annotated_image(_result())
        assert img.dtype == np.uint8

    def test_font_scale_zero_raises(self):
        with pytest.raises(ValueError):
            render_annotated_image(_result(), font_scale=0.0)

    def test_custom_canvas(self):
        r = _result()
        canvas = np.zeros((r.canvas_h, r.canvas_w, 3), dtype=np.uint8)
        img = render_annotated_image(r, canvas=canvas)
        assert img.shape == canvas.shape

    def test_non_3d_canvas_raises(self):
        r = _result()
        bad = np.zeros((r.canvas_h, r.canvas_w), dtype=np.uint8)
        with pytest.raises(ValueError):
            render_annotated_image(r, canvas=bad)


# ─── TestSummaryTable ─────────────────────────────────────────────────────────

class TestSummaryTable:
    def test_returns_dict(self):
        tbl = summary_table([_result()])
        assert isinstance(tbl, dict)

    def test_correct_keys(self):
        tbl = summary_table([_result()])
        for key in ("n_fragments", "canvas_w", "canvas_h",
                    "mean_score", "min_score", "max_score"):
            assert key in tbl

    def test_length_matches_results(self):
        tbl = summary_table([_result(3), _result(4)])
        assert len(tbl["n_fragments"]) == 2

    def test_n_fragments_correct(self):
        tbl = summary_table([_result(3), _result(5)])
        assert tbl["n_fragments"] == [3, 5]

    def test_score_stats_with_scores(self):
        r = _result(3, scores=[0.6, 0.8, 1.0])
        tbl = summary_table([r])
        assert tbl["mean_score"][0] == pytest.approx(0.8, abs=1e-6)
        assert tbl["min_score"][0] == pytest.approx(0.6, abs=1e-6)
        assert tbl["max_score"][0] == pytest.approx(1.0, abs=1e-6)

    def test_score_stats_none_when_no_scores(self):
        tbl = summary_table([_result(2, scores=None)])
        assert tbl["mean_score"][0] is None

    def test_empty_list(self):
        tbl = summary_table([])
        assert tbl["n_fragments"] == []


# ─── TestExportResult ─────────────────────────────────────────────────────────

class TestExportResult:
    def test_json_returns_string(self):
        cfg = ExportConfig(fmt="json")
        content = export_result(_result(), cfg)
        assert isinstance(content, str)
        json.loads(content)  # валидный JSON

    def test_csv_returns_string(self):
        cfg = ExportConfig(fmt="csv")
        content = export_result(_result(), cfg)
        assert isinstance(content, str)
        assert "fragment_id" in content

    def test_text_returns_string(self):
        cfg = ExportConfig(fmt="text")
        content = export_result(_result(), cfg)
        assert isinstance(content, str)

    def test_summary_returns_string(self):
        cfg = ExportConfig(fmt="summary")
        content = export_result(_result(), cfg)
        assert isinstance(content, str)

    def test_image_returns_none(self):
        cfg = ExportConfig(fmt="image")
        result = export_result(_result(), cfg)
        assert result is None


# ─── TestBatchExport ──────────────────────────────────────────────────────────

class TestBatchExport:
    def test_returns_list(self):
        cfg = ExportConfig(fmt="json")
        out = batch_export([_result(), _result(2)], cfg)
        assert isinstance(out, list)

    def test_correct_length(self):
        cfg = ExportConfig(fmt="csv")
        out = batch_export([_result(3), _result(4), _result(2)], cfg)
        assert len(out) == 3

    def test_empty_list(self):
        cfg = ExportConfig(fmt="json")
        assert batch_export([], cfg) == []

    def test_each_element_valid_json(self):
        cfg = ExportConfig(fmt="json")
        out = batch_export([_result(2), _result(3)], cfg)
        for s in out:
            json.loads(s)

    def test_image_fmt_returns_nones(self):
        cfg = ExportConfig(fmt="image")
        out = batch_export([_result(), _result()], cfg)
        assert all(x is None for x in out)
