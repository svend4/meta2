"""
Интеграционные тесты: экспорт и загрузка результатов.

Проверяет roundtrip: Assembly → AssemblyResult → JSON/CSV/text → parse back.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.tear_generator import tear_document, generate_test_document
from puzzle_reconstruction.config import Config
from puzzle_reconstruction.models import Assembly, Fragment
from puzzle_reconstruction.preprocessing.segmentation import segment_fragment
from puzzle_reconstruction.preprocessing.contour import extract_contour
from puzzle_reconstruction.algorithms.tangram.inscriber import fit_tangram
from puzzle_reconstruction.algorithms.synthesis import (
    compute_fractal_signature, build_edge_signatures,
)
from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
from puzzle_reconstruction.assembly.greedy import greedy_assembly
from puzzle_reconstruction.io.result_exporter import (
    AssemblyResult, ExportConfig,
    to_json, from_json, to_csv, to_text_report,
    export_result, batch_export,
)
from puzzle_reconstruction.io.image_loader import (
    LoadConfig, LoadedImage,
    load_image, load_from_array, load_from_directory,
    list_image_files, batch_load,
)
from puzzle_reconstruction.pipeline import Pipeline, PipelineResult

pytestmark = pytest.mark.integration


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_assembly_result(n: int = 3) -> AssemblyResult:
    return AssemblyResult(
        fragment_ids=list(range(n)),
        positions=[(i * 100, 0) for i in range(n)],
        sizes=[(80, 100) for _ in range(n)],
        canvas_w=n * 100 + 20,
        canvas_h=120,
        metadata={"total_score": 0.75, "method": "greedy"},
    )


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def doc():
    return generate_test_document(width=250, height=320, seed=77)


@pytest.fixture(scope="module")
def images_4(doc):
    return tear_document(doc, n_pieces=4, noise_level=0.3, seed=33)


@pytest.fixture(scope="module")
def pipeline_result(images_4):
    cfg = Config.default()
    cfg.assembly.method = "greedy"
    p = Pipeline(cfg=cfg, n_workers=1)
    return p.run(images_4)


@pytest.fixture(scope="module")
def assembly_result_3():
    return _build_assembly_result(3)


# ─── TestAssemblyResult ───────────────────────────────────────────────────────

class TestAssemblyResult:
    def test_creates_with_required_fields(self):
        ar = _build_assembly_result(2)
        assert ar.fragment_ids == [0, 1]
        assert len(ar.positions) == 2
        assert len(ar.sizes) == 2

    def test_canvas_dimensions_positive(self):
        ar = _build_assembly_result(2)
        assert ar.canvas_w >= 1
        assert ar.canvas_h >= 1

    def test_metadata_stored(self):
        ar = _build_assembly_result(3)
        assert ar.metadata["total_score"] == pytest.approx(0.75)


# ─── TestExportConfig ─────────────────────────────────────────────────────────

class TestExportConfig:
    def test_default_format_json(self):
        c = ExportConfig()
        assert c.fmt == "json"

    def test_valid_formats_accepted(self):
        for fmt in ("json", "csv", "text", "summary"):
            c = ExportConfig(fmt=fmt)
            assert c.fmt == fmt

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            ExportConfig(fmt="xml")

    def test_negative_indent_raises(self):
        with pytest.raises(ValueError):
            ExportConfig(indent=-1)

    def test_zero_font_scale_raises(self):
        with pytest.raises(ValueError):
            ExportConfig(font_scale=0.0)


# ─── TestToJson ───────────────────────────────────────────────────────────────

class TestToJson:
    def test_returns_string(self, assembly_result_3):
        j = to_json(assembly_result_3)
        assert isinstance(j, str)

    def test_is_valid_json(self, assembly_result_3):
        j = to_json(assembly_result_3)
        parsed = json.loads(j)
        assert isinstance(parsed, dict)

    def test_contains_fragment_ids(self, assembly_result_3):
        j = to_json(assembly_result_3)
        d = json.loads(j)
        assert "fragment_ids" in d

    def test_contains_positions(self, assembly_result_3):
        j = to_json(assembly_result_3)
        d = json.loads(j)
        assert "positions" in d

    def test_contains_canvas_dimensions(self, assembly_result_3):
        j = to_json(assembly_result_3)
        d = json.loads(j)
        assert "canvas_w" in d and "canvas_h" in d

    def test_metadata_preserved(self, assembly_result_3):
        j = to_json(assembly_result_3)
        d = json.loads(j)
        assert d.get("metadata", {}).get("total_score") == pytest.approx(0.75)


# ─── TestJsonRoundtrip ────────────────────────────────────────────────────────

class TestJsonRoundtrip:
    def test_roundtrip_fragment_ids(self, assembly_result_3):
        j  = to_json(assembly_result_3)
        ar = from_json(j)
        assert ar.fragment_ids == assembly_result_3.fragment_ids

    def test_roundtrip_positions(self, assembly_result_3):
        j  = to_json(assembly_result_3)
        ar = from_json(j)
        assert len(ar.positions) == len(assembly_result_3.positions)

    def test_roundtrip_canvas(self, assembly_result_3):
        j  = to_json(assembly_result_3)
        ar = from_json(j)
        assert ar.canvas_w == assembly_result_3.canvas_w
        assert ar.canvas_h == assembly_result_3.canvas_h

    def test_roundtrip_metadata(self, assembly_result_3):
        j  = to_json(assembly_result_3)
        ar = from_json(j)
        assert ar.metadata.get("total_score") == pytest.approx(0.75)


# ─── TestToCsv ────────────────────────────────────────────────────────────────

class TestToCsv:
    def test_returns_string(self, assembly_result_3):
        c = to_csv(assembly_result_3)
        assert isinstance(c, str)

    def test_has_header_row(self, assembly_result_3):
        c = to_csv(assembly_result_3)
        lines = c.strip().split("\n")
        assert len(lines) >= 2  # header + at least one data row

    def test_parseable_csv(self, assembly_result_3):
        c = to_csv(assembly_result_3)
        reader = csv.reader(c.splitlines())
        rows = list(reader)
        assert len(rows) >= 2

    def test_row_count_matches_fragments(self, assembly_result_3):
        c = to_csv(assembly_result_3)
        reader = csv.reader(c.splitlines())
        rows = list(reader)
        # header + N fragment rows
        assert len(rows) == len(assembly_result_3.fragment_ids) + 1


# ─── TestToTextReport ─────────────────────────────────────────────────────────

class TestToTextReport:
    def test_returns_string(self, assembly_result_3):
        t = to_text_report(assembly_result_3)
        assert isinstance(t, str)

    def test_contains_fragment_count(self, assembly_result_3):
        t = to_text_report(assembly_result_3)
        assert "3" in t or "fragment" in t.lower() or "фрагмент" in t.lower()


# ─── TestExportResult ─────────────────────────────────────────────────────────

class TestExportResult:
    def test_json_format_no_path(self, assembly_result_3):
        result = export_result(assembly_result_3, ExportConfig(fmt="json"))
        assert isinstance(result, str)
        json.loads(result)  # valid JSON

    def test_csv_format_no_path(self, assembly_result_3):
        result = export_result(assembly_result_3, ExportConfig(fmt="csv"))
        assert isinstance(result, str)

    def test_text_format_no_path(self, assembly_result_3):
        result = export_result(assembly_result_3, ExportConfig(fmt="text"))
        assert isinstance(result, str)

    def test_summary_format_no_path(self, assembly_result_3):
        result = export_result(assembly_result_3, ExportConfig(fmt="summary"))
        assert isinstance(result, str)

    def test_json_to_file(self, assembly_result_3):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        export_result(assembly_result_3, ExportConfig(fmt="json", output_path=path))
        content = Path(path).read_text(encoding="utf-8")
        assert len(content) > 0
        json.loads(content)


# ─── TestPipelineResultExport ─────────────────────────────────────────────────

class TestPipelineResultExport:
    def test_pipeline_export_json(self, pipeline_result):
        exported = pipeline_result.export(fmt="json")
        assert exported is not None
        assert isinstance(exported, str)

    def test_pipeline_export_csv(self, pipeline_result):
        exported = pipeline_result.export(fmt="csv")
        assert isinstance(exported, str)

    def test_pipeline_export_summary(self, pipeline_result):
        exported = pipeline_result.export(fmt="summary")
        assert isinstance(exported, str)

    def test_pipeline_export_to_tmpfile(self, pipeline_result):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        pipeline_result.export(fmt="json", output_path=path)
        content = Path(path).read_text()
        assert len(content) > 10


# ─── TestImageLoader ──────────────────────────────────────────────────────────

class TestImageLoader:
    def test_load_from_array_bgr(self):
        img = np.zeros((50, 60, 3), dtype=np.uint8)
        loaded = load_from_array(img, cfg=LoadConfig(color_mode="bgr"))
        assert isinstance(loaded, LoadedImage)
        assert loaded.data.shape == (50, 60, 3)

    def test_load_from_array_gray(self):
        img = np.zeros((50, 60, 3), dtype=np.uint8)
        loaded = load_from_array(img, cfg=LoadConfig(color_mode="gray"))
        assert loaded.data.ndim == 2

    def test_load_from_array_with_resize(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        loaded = load_from_array(img, cfg=LoadConfig(color_mode="bgr", target_size=(40, 30)))
        assert loaded.data.shape[1] == 40
        assert loaded.data.shape[0] == 30

    def test_load_from_directory_empty_tmpdir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            images = load_from_directory(tmpdir)
            assert isinstance(images, list)
            assert len(images) == 0

    def test_load_from_directory_with_images(self, images_4):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, img in enumerate(images_4):
                cv2.imwrite(str(Path(tmpdir) / f"frag_{i:03d}.png"), img)
            loaded = load_from_directory(tmpdir)
            assert len(loaded) == len(images_4)
            for li in loaded:
                assert isinstance(li, LoadedImage)
                assert li.data.ndim == 3

    def test_list_image_files_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = list_image_files(tmpdir)
            assert files == []

    def test_list_image_files_finds_png(self, images_4):
        with tempfile.TemporaryDirectory() as tmpdir:
            cv2.imwrite(str(Path(tmpdir) / "test.png"), images_4[0])
            files = list_image_files(tmpdir)
            assert len(files) == 1

    def test_batch_load_returns_list(self, images_4):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i, img in enumerate(images_4[:2]):
                p = str(Path(tmpdir) / f"f{i}.png")
                cv2.imwrite(p, img)
                paths.append(p)
            loaded = batch_load(paths)
            assert len(loaded) == 2
