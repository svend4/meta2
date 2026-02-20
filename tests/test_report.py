"""
Юнит-тесты для puzzle_reconstruction/verification/report.py.

Тесты покрывают:
    - build_report()          — создание Report из Assembly
    - Report.to_dict()        — JSON-сериализация
    - Report.save_json()      — файл создаётся и читаемый
    - Report.to_markdown()    — Markdown строка
    - Report.save_markdown()  — файл создаётся
    - Report.to_html()        — HTML строка (валидная структура)
    - Report.save_html()      — файл создаётся
    - _img_tag()              — base64 embedded image
    - FragmentInfo / ReportData — dataclass поля
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from puzzle_reconstruction.verification.report import (
    build_report,
    Report,
    ReportData,
    FragmentInfo,
)
from puzzle_reconstruction.models import (
    Assembly, Fragment,
    TangramSignature, FractalSignature,
    ShapeClass, EdgeSide, EdgeSignature,
)


# ─── Фикстуры ────────────────────────────────────────────────────────────

def _make_fragment(fid: int, h: int = 60, w: int = 50) -> Fragment:
    img  = np.full((h, w, 3), 200, dtype=np.uint8)
    mask = np.ones((h, w), dtype=np.uint8)
    cont = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=float)
    frag = Fragment(fragment_id=fid, image=img, mask=mask, contour=cont)
    frag.tangram = TangramSignature(
        polygon=cont / np.array([w, h]),
        shape_class=ShapeClass.RECTANGLE,
        centroid=np.array([0.5, 0.5]),
        angle=0.0, scale=1.0, area=0.5,
    )
    frag.fractal = FractalSignature(
        fd_box=1.3, fd_divider=1.35,
        ifs_coeffs=np.zeros(4),
        css_image=[], chain_code="", curve=np.zeros((8, 2)),
    )
    frag.edges = [EdgeSignature(
        edge_id=fid * 4, side=EdgeSide.TOP,
        virtual_curve=np.zeros((16, 2)),
        fd=1.3, css_vec=np.zeros(16),
        ifs_coeffs=np.zeros(4), length=50.0,
    )]
    return frag


def _make_assembly(n: int = 3, score: float = 0.75) -> Assembly:
    frags = [_make_fragment(i) for i in range(n)]
    placements = {i: (np.array([i * 100.0, 0.0]), 0.0) for i in range(n)}
    return Assembly(
        fragments=frags,
        placements=placements,
        compat_matrix=np.zeros((n * 4, n * 4)),
        total_score=score,
        ocr_score=0.6,
    )


# ─── build_report ────────────────────────────────────────────────────────

class TestBuildReport:

    def test_returns_report_object(self):
        asm    = _make_assembly(3)
        report = build_report(asm)
        assert isinstance(report, Report)

    def test_data_n_placed(self):
        asm    = _make_assembly(4)
        report = build_report(asm)
        assert report.data.n_placed == 4

    def test_data_n_input_from_fragments(self):
        asm    = _make_assembly(3)
        report = build_report(asm)
        assert report.data.n_input == 3

    def test_data_score(self):
        asm    = _make_assembly(3, score=0.88)
        report = build_report(asm)
        assert math.isclose(report.data.assembly_score, 0.88, rel_tol=1e-6)

    def test_data_ocr_score(self):
        asm         = _make_assembly(2)
        asm.ocr_score = 0.55
        report = build_report(asm)
        assert math.isclose(report.data.ocr_score, 0.55, rel_tol=1e-6)

    def test_fragments_info_count(self):
        asm    = _make_assembly(5)
        report = build_report(asm)
        assert len(report.data.fragments) == 5

    def test_fragment_info_fields(self):
        asm    = _make_assembly(2)
        report = build_report(asm)
        fi = report.data.fragments[0]
        assert "fragment_id"  in fi
        assert "shape_class"  in fi
        assert "fd_box"       in fi
        assert "placed"       in fi

    def test_fragment_placed_flag(self):
        asm    = _make_assembly(3)
        report = build_report(asm)
        for fi in report.data.fragments:
            assert fi["placed"] is True

    def test_notes_stored(self):
        asm    = _make_assembly(2)
        report = build_report(asm, notes="Тестовый прогон")
        assert "Тестовый" in report.data.notes

    def test_metrics_optional(self):
        """Без metrics — поля None."""
        asm    = _make_assembly(2)
        report = build_report(asm)
        assert report.data.neighbor_accuracy is None
        assert report.data.direct_comparison is None
        assert report.data.perfect is None

    def test_metrics_filled_when_provided(self):
        from puzzle_reconstruction.verification.metrics import ReconstructionMetrics
        asm = _make_assembly(3)
        m = ReconstructionMetrics(
            neighbor_accuracy=0.9, direct_comparison=0.8,
            perfect=False, position_rmse=12.5,
            angular_error_deg=5.0, n_fragments=3,
            n_correct_pairs=3, n_total_pairs=4, edge_match_rate=0.85,
        )
        report = build_report(asm, metrics=m)
        assert math.isclose(report.data.neighbor_accuracy, 0.9, rel_tol=1e-6)
        assert math.isclose(report.data.direct_comparison, 0.8, rel_tol=1e-6)
        assert math.isclose(report.data.position_rmse, 12.5, rel_tol=1e-6)

    def test_canvas_stored(self):
        asm    = _make_assembly(2)
        canvas = np.full((100, 200, 3), 128, dtype=np.uint8)
        report = build_report(asm, canvas=canvas)
        assert report.canvas is canvas

    def test_empty_assembly(self):
        asm = Assembly(fragments=[], placements={}, compat_matrix=np.array([]))
        report = build_report(asm)
        assert report.data.n_placed == 0
        assert report.data.n_input  == 0


# ─── to_dict / save_json ─────────────────────────────────────────────────

class TestReportJson:

    def test_to_dict_is_dict(self):
        asm    = _make_assembly(2)
        report = build_report(asm)
        d      = report.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_timestamp(self):
        asm    = _make_assembly(2)
        report = build_report(asm)
        assert "timestamp" in report.to_dict()

    def test_to_dict_json_serializable(self):
        asm    = _make_assembly(3)
        report = build_report(asm)
        text   = json.dumps(report.to_dict())
        assert len(text) > 0

    def test_save_json_creates_file(self, tmp_path):
        asm    = _make_assembly(2)
        report = build_report(asm)
        path   = tmp_path / "report.json"
        report.save_json(path)
        assert path.exists()

    def test_save_json_valid_json(self, tmp_path):
        asm    = _make_assembly(2)
        report = build_report(asm)
        path   = tmp_path / "report.json"
        report.save_json(path)
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        assert "assembly_score" in d


# ─── to_markdown / save_markdown ─────────────────────────────────────────

class TestReportMarkdown:

    def test_to_markdown_returns_string(self):
        asm    = _make_assembly(3)
        report = build_report(asm)
        md     = report.to_markdown()
        assert isinstance(md, str)
        assert len(md) > 0

    def test_to_markdown_has_header(self):
        asm    = _make_assembly(3)
        report = build_report(asm)
        md     = report.to_markdown()
        assert "# " in md

    def test_to_markdown_has_table(self):
        asm    = _make_assembly(3)
        report = build_report(asm)
        md     = report.to_markdown()
        assert "|" in md   # Markdown таблица

    def test_to_markdown_has_n_placed(self):
        asm    = _make_assembly(4)
        report = build_report(asm)
        md     = report.to_markdown()
        assert "4" in md   # n_placed

    def test_to_markdown_contains_metrics_when_provided(self):
        from puzzle_reconstruction.verification.metrics import ReconstructionMetrics
        asm = _make_assembly(3)
        m   = ReconstructionMetrics(
            neighbor_accuracy=0.75, direct_comparison=0.6,
            perfect=True, position_rmse=8.0,
            angular_error_deg=3.0, n_fragments=3,
            n_correct_pairs=3, n_total_pairs=3, edge_match_rate=0.9,
        )
        report = build_report(asm, metrics=m)
        md = report.to_markdown()
        assert "Neighbor Accuracy" in md or "75" in md

    def test_save_markdown_creates_file(self, tmp_path):
        asm    = _make_assembly(2)
        report = build_report(asm)
        path   = tmp_path / "report.md"
        report.save_markdown(path)
        assert path.exists()
        assert path.read_text(encoding="utf-8").startswith("#")


# ─── to_html / save_html ─────────────────────────────────────────────────

class TestReportHtml:

    def test_to_html_returns_string(self):
        asm    = _make_assembly(3)
        report = build_report(asm)
        html   = report.to_html()
        assert isinstance(html, str)
        assert len(html) > 0

    def test_to_html_has_doctype(self):
        asm    = _make_assembly(2)
        report = build_report(asm)
        html   = report.to_html()
        assert "<!DOCTYPE" in html or "<!doctype" in html.lower()

    def test_to_html_has_table(self):
        asm    = _make_assembly(3)
        report = build_report(asm)
        html   = report.to_html()
        assert "<table" in html.lower()

    def test_to_html_has_score(self):
        asm    = _make_assembly(2, score=0.91)
        report = build_report(asm)
        html   = report.to_html()
        assert "91" in html   # 0.91 → "91%"

    def test_to_html_embeds_canvas(self):
        asm    = _make_assembly(2)
        canvas = np.full((80, 120, 3), 200, dtype=np.uint8)
        report = build_report(asm, canvas=canvas)
        html   = report.to_html()
        assert "base64" in html   # Изображение встроено

    def test_to_html_no_canvas_no_img(self):
        asm    = _make_assembly(2)
        report = build_report(asm)   # Без canvas
        html   = report.to_html()
        # base64 не должно быть без изображений
        assert "base64" not in html

    def test_to_html_has_notes(self):
        asm    = _make_assembly(2)
        report = build_report(asm, notes="Важная заметка")
        html   = report.to_html()
        assert "Важная заметка" in html

    def test_save_html_creates_file(self, tmp_path):
        asm    = _make_assembly(2)
        report = build_report(asm)
        path   = tmp_path / "report.html"
        report.save_html(path)
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "<html" in content.lower()

    def test_save_html_utf8_encoding(self, tmp_path):
        asm    = _make_assembly(2)
        report = build_report(asm, notes="Кириллица 文字")
        path   = tmp_path / "report.html"
        report.save_html(path)
        content = path.read_text(encoding="utf-8")
        assert "Кириллица" in content
