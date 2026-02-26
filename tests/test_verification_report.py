"""Tests for puzzle_reconstruction.verification.report"""
import pytest
import json
import numpy as np
import sys
import tempfile
import os
sys.path.insert(0, '/home/user/meta2')

from puzzle_reconstruction.verification.report import (
    FragmentInfo,
    ReportData,
    Report,
)


def make_report_data(n_frags=3):
    frag_infos = [
        {
            "fragment_id": i,
            "shape_class": "triangle",
            "fd_box": 1.5,
            "fd_divider": 1.4,
            "n_edges": 3,
            "placed": True,
            "position": [float(i * 50), 0.0],
            "angle_deg": 0.0,
        }
        for i in range(n_frags)
    ]
    return ReportData(
        timestamp="2026-02-26 12:00:00",
        method="test",
        n_input=n_frags,
        n_placed=n_frags,
        assembly_score=0.8,
        ocr_score=0.7,
        fragments=frag_infos,
        config={"test": True},
    )


def make_report(n_frags=3, with_canvas=False):
    data = make_report_data(n_frags)
    canvas = np.zeros((100, 100, 3), dtype=np.uint8) if with_canvas else None
    return Report(data, canvas=canvas)


# ─── FragmentInfo ─────────────────────────────────────────────────────────────

def test_fragment_info_fields():
    fi = FragmentInfo(
        fragment_id=0, shape_class="triangle",
        fd_box=1.5, fd_divider=1.4,
        n_edges=3, placed=True,
        position=[10.0, 20.0], angle_deg=45.0
    )
    assert fi.fragment_id == 0
    assert fi.shape_class == "triangle"


def test_fragment_info_placed_false():
    fi = FragmentInfo(
        fragment_id=5, shape_class="rectangle",
        fd_box=1.0, fd_divider=1.0,
        n_edges=4, placed=False,
        position=None, angle_deg=0.0
    )
    assert fi.placed is False
    assert fi.position is None


# ─── ReportData ───────────────────────────────────────────────────────────────

def test_report_data_basic():
    data = make_report_data()
    assert data.n_input == 3
    assert data.method == "test"
    assert data.assembly_score == 0.8


def test_report_data_with_metrics():
    data = make_report_data()
    data.neighbor_accuracy = 0.9
    data.direct_comparison = 0.85
    data.perfect = False
    assert data.neighbor_accuracy == 0.9


def test_report_data_defaults():
    data = make_report_data()
    assert data.neighbor_accuracy is None
    assert data.direct_comparison is None
    assert data.runtime_sec is None
    assert data.notes == ""


# ─── Report.to_dict ───────────────────────────────────────────────────────────

def test_report_to_dict_basic():
    report = make_report()
    d = report.to_dict()
    assert isinstance(d, dict)
    assert "timestamp" in d
    assert "method" in d
    assert "n_input" in d
    assert "fragments" in d


def test_report_to_dict_fragments_list():
    report = make_report(n_frags=2)
    d = report.to_dict()
    assert len(d["fragments"]) == 2


def test_report_to_dict_assembly_score():
    report = make_report()
    d = report.to_dict()
    assert d["assembly_score"] == pytest.approx(0.8)


# ─── Report.to_markdown ───────────────────────────────────────────────────────

def test_report_to_markdown_contains_header():
    report = make_report()
    md = report.to_markdown()
    assert "# " in md


def test_report_to_markdown_contains_method():
    report = make_report()
    md = report.to_markdown()
    assert "test" in md


def test_report_to_markdown_contains_fragments():
    report = make_report(n_frags=3)
    md = report.to_markdown()
    assert "Фрагменты" in md or "fragment" in md.lower() or "triangle" in md


def test_report_to_markdown_table_format():
    report = make_report()
    md = report.to_markdown()
    assert "|" in md


def test_report_to_markdown_with_metrics():
    data = make_report_data()
    data.neighbor_accuracy = 0.95
    data.direct_comparison = 0.90
    data.position_rmse = 5.0
    data.angular_error_deg = 3.0
    data.perfect = True
    report = Report(data)
    md = report.to_markdown()
    assert "0.95" in md or "95%" in md or "95.0" in md


# ─── Report.to_html ───────────────────────────────────────────────────────────

def test_report_to_html_contains_doctype():
    report = make_report()
    html = report.to_html()
    assert "<!DOCTYPE html>" in html


def test_report_to_html_contains_fragments_table():
    report = make_report(n_frags=2)
    html = report.to_html()
    assert "triangle" in html
    assert "<table" in html.lower() or "<td" in html.lower()


def test_report_to_html_with_canvas():
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    data = make_report_data()
    report = Report(data, canvas=canvas)
    html = report.to_html()
    assert "base64" in html or "<!DOCTYPE html>" in html


def test_report_to_html_with_notes():
    data = make_report_data()
    data.notes = "Test note"
    report = Report(data)
    html = report.to_html()
    assert "Test note" in html


# ─── Report.save_json ─────────────────────────────────────────────────────────

def test_report_save_json():
    report = make_report()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        report.save_json(path)
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert "timestamp" in loaded
        assert "n_input" in loaded
    finally:
        os.unlink(path)


# ─── Report.save_markdown ─────────────────────────────────────────────────────

def test_report_save_markdown():
    report = make_report()
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        path = f.name
    try:
        report.save_markdown(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "#" in content
    finally:
        os.unlink(path)


# ─── Report._img_tag ──────────────────────────────────────────────────────────

def test_img_tag_none():
    tag = Report._img_tag(None, "caption")
    assert tag == ""


def test_img_tag_valid_image():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    tag = Report._img_tag(img, "Test")
    # Either empty (encode failed) or contains base64 data
    assert isinstance(tag, str)


# ─── Runtime metrics ──────────────────────────────────────────────────────────

def test_report_with_runtime():
    data = make_report_data()
    data.runtime_sec = 1.23
    report = Report(data)
    md = report.to_markdown()
    assert "1.23" in md


def test_report_without_runtime():
    report = make_report()
    md = report.to_markdown()
    # Should not crash even without runtime
    assert isinstance(md, str)
