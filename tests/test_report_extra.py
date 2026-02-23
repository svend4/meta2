"""Extra tests for puzzle_reconstruction.verification.report."""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _frag(fid, h=60, w=50):
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    mask = np.ones((h, w), dtype=np.uint8)
    cont = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
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


def _asm(n=3, score=0.75):
    frags = [_frag(i) for i in range(n)]
    placements = {i: (np.array([i * 100.0, 0.0]), 0.0) for i in range(n)}
    return Assembly(
        fragments=frags, placements=placements,
        compat_matrix=np.zeros((n * 4, n * 4)),
        total_score=score, ocr_score=0.6,
    )


# ─── TestBuildReportExtra ─────────────────────────────────────────────────────

class TestBuildReportExtra:
    def test_returns_report(self):
        assert isinstance(build_report(_asm()), Report)

    def test_n_placed(self):
        assert build_report(_asm(5)).data.n_placed == 5

    def test_n_input(self):
        assert build_report(_asm(4)).data.n_input == 4

    def test_assembly_score(self):
        r = build_report(_asm(3, score=0.91))
        assert math.isclose(r.data.assembly_score, 0.91, rel_tol=1e-6)

    def test_ocr_score(self):
        a = _asm(2)
        a.ocr_score = 0.42
        r = build_report(a)
        assert math.isclose(r.data.ocr_score, 0.42, rel_tol=1e-6)

    def test_fragments_count(self):
        assert len(build_report(_asm(6)).data.fragments) == 6

    def test_fragment_has_id(self):
        fi = build_report(_asm(2)).data.fragments[0]
        assert "fragment_id" in fi

    def test_fragment_has_placed(self):
        for fi in build_report(_asm(3)).data.fragments:
            assert fi["placed"] is True

    def test_notes_stored(self):
        r = build_report(_asm(2), notes="Заметка")
        assert "Заметка" in r.data.notes

    def test_metrics_none_by_default(self):
        r = build_report(_asm(2))
        assert r.data.neighbor_accuracy is None
        assert r.data.direct_comparison is None

    def test_metrics_provided(self):
        from puzzle_reconstruction.verification.metrics import ReconstructionMetrics
        m = ReconstructionMetrics(
            neighbor_accuracy=0.85, direct_comparison=0.7,
            perfect=True, position_rmse=5.0,
            angular_error_deg=2.0, n_fragments=3,
            n_correct_pairs=3, n_total_pairs=3, edge_match_rate=0.9,
        )
        r = build_report(_asm(3), metrics=m)
        assert math.isclose(r.data.neighbor_accuracy, 0.85, rel_tol=1e-6)

    def test_canvas_stored(self):
        canvas = np.zeros((100, 200, 3), dtype=np.uint8)
        r = build_report(_asm(2), canvas=canvas)
        assert r.canvas is canvas

    def test_empty_assembly(self):
        a = Assembly(fragments=[], placements={}, compat_matrix=np.array([]))
        r = build_report(a)
        assert r.data.n_placed == 0
        assert r.data.n_input == 0


# ─── TestReportJsonExtra ──────────────────────────────────────────────────────

class TestReportJsonExtra:
    def test_to_dict_is_dict(self):
        assert isinstance(build_report(_asm()).to_dict(), dict)

    def test_to_dict_has_timestamp(self):
        assert "timestamp" in build_report(_asm()).to_dict()

    def test_to_dict_json_serializable(self):
        text = json.dumps(build_report(_asm(3)).to_dict())
        assert len(text) > 0

    def test_save_json_creates_file(self, tmp_path):
        p = tmp_path / "r.json"
        build_report(_asm(2)).save_json(p)
        assert p.exists()

    def test_save_json_valid_json(self, tmp_path):
        p = tmp_path / "r.json"
        build_report(_asm(2)).save_json(p)
        d = json.loads(p.read_text(encoding="utf-8"))
        assert "assembly_score" in d

    def test_to_dict_assembly_score_present(self):
        d = build_report(_asm(2, score=0.5)).to_dict()
        assert "assembly_score" in d

    def test_to_dict_n_placed_present(self):
        d = build_report(_asm(4)).to_dict()
        assert d.get("n_placed") == 4 or "n_placed" in str(d)


# ─── TestReportMarkdownExtra ─────────────────────────────────────────────────

class TestReportMarkdownExtra:
    def test_returns_string(self):
        md = build_report(_asm(3)).to_markdown()
        assert isinstance(md, str) and len(md) > 0

    def test_has_header(self):
        assert "# " in build_report(_asm(3)).to_markdown()

    def test_has_table(self):
        assert "|" in build_report(_asm(3)).to_markdown()

    def test_contains_n_placed(self):
        assert "4" in build_report(_asm(4)).to_markdown()

    def test_save_creates_file(self, tmp_path):
        p = tmp_path / "r.md"
        build_report(_asm(2)).save_markdown(p)
        assert p.exists()

    def test_save_starts_with_header(self, tmp_path):
        p = tmp_path / "r.md"
        build_report(_asm(2)).save_markdown(p)
        assert p.read_text(encoding="utf-8").startswith("#")

    def test_metrics_in_markdown(self):
        from puzzle_reconstruction.verification.metrics import ReconstructionMetrics
        m = ReconstructionMetrics(
            neighbor_accuracy=0.8, direct_comparison=0.6,
            perfect=True, position_rmse=7.0,
            angular_error_deg=3.0, n_fragments=3,
            n_correct_pairs=3, n_total_pairs=3, edge_match_rate=0.9,
        )
        md = build_report(_asm(3), metrics=m).to_markdown()
        assert "80" in md or "Neighbor" in md


# ─── TestReportHtmlExtra ──────────────────────────────────────────────────────

class TestReportHtmlExtra:
    def test_returns_string(self):
        html = build_report(_asm(3)).to_html()
        assert isinstance(html, str) and len(html) > 0

    def test_has_doctype(self):
        html = build_report(_asm(2)).to_html()
        assert "<!DOCTYPE" in html or "<!doctype" in html.lower()

    def test_has_table(self):
        assert "<table" in build_report(_asm(3)).to_html().lower()

    def test_has_score(self):
        assert "91" in build_report(_asm(2, score=0.91)).to_html()

    def test_embeds_canvas(self):
        canvas = np.full((80, 120, 3), 200, dtype=np.uint8)
        html = build_report(_asm(2), canvas=canvas).to_html()
        assert "base64" in html

    def test_no_canvas_no_base64(self):
        html = build_report(_asm(2)).to_html()
        assert "base64" not in html

    def test_notes_in_html(self):
        html = build_report(_asm(2), notes="Тест заметки").to_html()
        assert "Тест заметки" in html

    def test_save_creates_file(self, tmp_path):
        p = tmp_path / "r.html"
        build_report(_asm(2)).save_html(p)
        assert p.exists()
        assert "<html" in p.read_text(encoding="utf-8").lower()

    def test_save_utf8(self, tmp_path):
        p = tmp_path / "r.html"
        build_report(_asm(2), notes="Кириллица").save_html(p)
        assert "Кириллица" in p.read_text(encoding="utf-8")
