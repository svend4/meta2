"""Extra tests for puzzle_reconstruction/verification/report.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from puzzle_reconstruction.verification.report import (
    FragmentInfo,
    ReportData,
    Report,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _report_data(**overrides):
    defaults = dict(
        timestamp="2025-01-01 00:00:00",
        method="greedy",
        n_input=3,
        n_placed=3,
        assembly_score=0.85,
        ocr_score=0.75,
    )
    defaults.update(overrides)
    return ReportData(**defaults)


def _report(**overrides):
    return Report(_report_data(**overrides))


# ─── FragmentInfo ────────────────────────────────────────────────────────────

class TestFragmentInfoExtra:
    def test_creation(self):
        fi = FragmentInfo(
            fragment_id=0, shape_class="rectangle",
            fd_box=1.2, fd_divider=1.1,
            n_edges=4, placed=True,
            position=[10, 20], angle_deg=0.0,
        )
        assert fi.fragment_id == 0
        assert fi.placed is True
        assert fi.position == [10, 20]

    def test_not_placed(self):
        fi = FragmentInfo(
            fragment_id=1, shape_class="triangle",
            fd_box=0.0, fd_divider=0.0,
            n_edges=3, placed=False,
            position=None, angle_deg=0.0,
        )
        assert fi.placed is False
        assert fi.position is None


# ─── ReportData ──────────────────────────────────────────────────────────────

class TestReportDataExtra:
    def test_defaults(self):
        d = _report_data()
        assert d.neighbor_accuracy is None
        assert d.direct_comparison is None
        assert d.runtime_sec is None
        assert d.fragments == []
        assert d.config == {}
        assert d.notes == ""

    def test_with_metrics(self):
        d = _report_data(
            neighbor_accuracy=0.9,
            direct_comparison=0.8,
            position_rmse=5.0,
            angular_error_deg=2.0,
            perfect=False,
        )
        assert d.neighbor_accuracy == 0.9
        assert d.perfect is False

    def test_with_runtime(self):
        d = _report_data(runtime_sec=10.5)
        assert d.runtime_sec == 10.5


# ─── Report ──────────────────────────────────────────────────────────────────

class TestReportExtra:
    def test_to_dict(self):
        r = _report()
        d = r.to_dict()
        assert d["method"] == "greedy"
        assert d["n_input"] == 3
        assert d["assembly_score"] == 0.85

    def test_save_json(self, tmp_path):
        r = _report()
        path = tmp_path / "report.json"
        r.save_json(path)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["method"] == "greedy"

    def test_to_markdown(self):
        r = _report()
        md = r.to_markdown()
        assert "greedy" in md
        assert "85.0%" in md

    def test_to_markdown_with_metrics(self):
        r = _report(neighbor_accuracy=0.9, direct_comparison=0.8,
                     position_rmse=5.0, angular_error_deg=2.0, perfect=False)
        md = r.to_markdown()
        assert "Neighbor Accuracy" in md
        assert "90.0%" in md

    def test_to_markdown_with_runtime(self):
        r = _report(runtime_sec=12.3)
        md = r.to_markdown()
        assert "12.30" in md

    def test_to_markdown_with_notes(self):
        r = _report(notes="Test note")
        md = r.to_markdown()
        assert "Test note" in md

    def test_save_markdown(self, tmp_path):
        r = _report()
        path = tmp_path / "report.md"
        r.save_markdown(path)
        assert path.exists()
        assert "greedy" in path.read_text(encoding="utf-8")

    def test_to_html(self):
        r = _report()
        html = r.to_html()
        assert "<html" in html
        assert "greedy" in html

    def test_to_html_with_metrics(self):
        r = _report(neighbor_accuracy=0.9, direct_comparison=0.8,
                     position_rmse=5.0, angular_error_deg=2.0, perfect=True)
        html = r.to_html()
        assert "Neighbor Accuracy" in html

    def test_save_html(self, tmp_path):
        r = _report()
        path = tmp_path / "report.html"
        r.save_html(path)
        assert path.exists()

    def test_with_canvas(self):
        canvas = np.full((100, 100, 3), 128, dtype=np.uint8)
        r = Report(_report_data(), canvas=canvas)
        html = r.to_html()
        assert "base64" in html

    def test_with_none_images(self):
        r = Report(_report_data(), canvas=None, heatmap=None, mosaic=None)
        html = r.to_html()
        assert "base64" not in html

    def test_img_tag_none(self):
        tag = Report._img_tag(None, "caption")
        assert tag == ""

    def test_img_tag_valid(self):
        img = np.full((50, 50, 3), 200, dtype=np.uint8)
        tag = Report._img_tag(img, "test")
        assert "base64" in tag
        assert "test" in tag
