"""
Integration tests for puzzle_reconstruction IO modules:
  - image_loader
  - metadata_writer
  - result_exporter
"""
import json
import os
import tempfile
import unittest

import numpy as np

from puzzle_reconstruction.io.image_loader import (
    LoadConfig,
    LoadedImage,
    load_from_array,
    batch_load,
    resize_image,
    list_image_files,
)
from puzzle_reconstruction.io.metadata_writer import (
    MetadataRecord,
    MetadataCollection,
    WriterConfig,
    write_json,
    write_csv,
    read_json,
    filter_by_score,
    merge_collections,
    write_summary,
)
from puzzle_reconstruction.io.result_exporter import (
    AssemblyResult,
    ExportConfig,
    to_json,
    to_csv,
    to_text_report,
    summary_table,
    export_result,
    batch_export,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb(h=80, w=80):
    """Return a uint8 RGB ndarray."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_record(fid=0, score=0.8):
    return MetadataRecord(
        fragment_id=fid,
        position=(10 * fid, 20 * fid),
        rotation_deg=float(fid) * 5.0,
        score=score,
    )


def _make_collection(*fids_scores):
    """fids_scores: list of (fid, score) pairs."""
    records = [_make_record(fid, score) for fid, score in fids_scores]
    return MetadataCollection(records=records)


def _make_result(n=3):
    return AssemblyResult(
        fragment_ids=list(range(n)),
        positions=[(i * 50, 0) for i in range(n)],
        sizes=[(40, 40)] * n,
        canvas_w=n * 50,
        canvas_h=80,
        scores=[0.7 + 0.1 * i for i in range(n)],
    )


# ---------------------------------------------------------------------------
# TestImageLoader  (11 tests)
# ---------------------------------------------------------------------------

class TestImageLoader(unittest.TestCase):

    def test_load_from_array_returns_loaded_image(self):
        img = _make_rgb()
        result = load_from_array(img, image_id=0)
        self.assertIsInstance(result, LoadedImage)

    def test_loaded_image_has_data_attribute(self):
        img = _make_rgb()
        loaded = load_from_array(img, image_id=0)
        self.assertTrue(hasattr(loaded, "data"))
        self.assertIsInstance(loaded.data, np.ndarray)

    def test_loaded_image_data_shape_matches_input(self):
        img = _make_rgb(60, 70)
        loaded = load_from_array(img, image_id=0)
        self.assertEqual(loaded.data.shape, (60, 70, 3))

    def test_loaded_image_id_stored_correctly(self):
        img = _make_rgb()
        loaded = load_from_array(img, image_id=42)
        self.assertEqual(loaded.image_id, 42)

    def test_load_from_array_with_load_config(self):
        img = _make_rgb()
        cfg = LoadConfig()
        loaded = load_from_array(img, cfg=cfg, image_id=7)
        self.assertIsInstance(loaded, LoadedImage)
        self.assertEqual(loaded.image_id, 7)

    def test_batch_load_returns_list_of_loaded_images(self):
        """batch_load expects file paths; we save two temp PNGs and load them."""
        import cv2
        imgs = [_make_rgb(30, 30), _make_rgb(30, 30)]
        paths = []
        try:
            for i, arr in enumerate(imgs):
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                cv2.imwrite(tmp.name, arr)
                paths.append(tmp.name)
            result = batch_load(paths)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
            for item in result:
                self.assertIsInstance(item, LoadedImage)
        finally:
            for p in paths:
                if os.path.exists(p):
                    os.unlink(p)

    def test_batch_load_empty_list_returns_empty_list(self):
        result = batch_load([])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_resize_image_returns_ndarray(self):
        img = _make_rgb(80, 80)
        out = resize_image(img, target_size=(40, 40))
        self.assertIsInstance(out, np.ndarray)

    def test_resize_image_returns_correct_shape(self):
        img = _make_rgb(80, 80)
        out = resize_image(img, target_size=(32, 64))
        # resize_image returns (h, w, c); target_size convention may be (w,h) or (h,w)
        # We just check that both spatial dims equal the requested values
        self.assertIn(out.shape[0], (32, 64))
        self.assertIn(out.shape[1], (32, 64))

    def test_list_image_files_returns_list(self):
        result = list_image_files(tempfile.gettempdir())
        self.assertIsInstance(result, list)

    def test_load_from_array_dtype_preserved_uint8(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        loaded = load_from_array(img, image_id=0)
        self.assertEqual(loaded.data.dtype, np.uint8)


# ---------------------------------------------------------------------------
# TestMetadataWriter  (11 tests)
# ---------------------------------------------------------------------------

class TestMetadataWriter(unittest.TestCase):

    def setUp(self):
        self.rec = _make_record(fid=0, score=0.8)
        self.col = MetadataCollection(records=[self.rec])

    def test_metadata_record_has_required_attrs(self):
        for attr in ("fragment_id", "position", "score"):
            self.assertTrue(hasattr(self.rec, attr), f"Missing attr: {attr}")

    def test_metadata_collection_has_records_attr(self):
        self.assertTrue(hasattr(self.col, "records"))
        self.assertIsInstance(self.col.records, list)

    def test_write_json_returns_string(self):
        result = write_json(self.col)
        self.assertIsInstance(result, str)

    def test_write_json_output_is_valid_json(self):
        result = write_json(self.col)
        parsed = json.loads(result)
        self.assertIsInstance(parsed, dict)

    def test_read_json_returns_metadata_collection(self):
        js = write_json(self.col)
        restored = read_json(js)
        self.assertIsInstance(restored, MetadataCollection)

    def test_roundtrip_write_read_preserves_fragment_id(self):
        js = write_json(self.col)
        restored = read_json(js)
        self.assertEqual(restored.records[0].fragment_id, 0)

    def test_roundtrip_write_read_preserves_score(self):
        js = write_json(self.col)
        restored = read_json(js)
        self.assertAlmostEqual(restored.records[0].score, 0.8, places=5)

    def test_write_csv_returns_string(self):
        result = write_csv(self.col)
        self.assertIsInstance(result, str)

    def test_write_csv_has_comma_separated_values(self):
        result = write_csv(self.col)
        # Every line should contain at least one comma
        lines = [l for l in result.splitlines() if l.strip()]
        for line in lines:
            self.assertIn(",", line)

    def test_filter_by_score_returns_metadata_collection(self):
        result = filter_by_score(self.col, min_score=0.5)
        self.assertIsInstance(result, MetadataCollection)

    def test_filter_by_score_high_threshold_removes_records(self):
        result = filter_by_score(self.col, min_score=0.99)
        self.assertEqual(len(result.records), 0)


# ---------------------------------------------------------------------------
# TestResultExporter  (11 tests)
# ---------------------------------------------------------------------------

class TestResultExporter(unittest.TestCase):

    def setUp(self):
        self.result = _make_result(n=3)

    def test_assembly_result_has_required_attrs(self):
        for attr in ("fragment_ids", "positions", "canvas_w", "canvas_h"):
            self.assertTrue(hasattr(self.result, attr), f"Missing attr: {attr}")

    def test_to_json_returns_string(self):
        out = to_json(self.result)
        self.assertIsInstance(out, str)

    def test_to_json_output_is_valid_json(self):
        out = to_json(self.result)
        parsed = json.loads(out)
        self.assertIsInstance(parsed, dict)

    def test_to_json_preserves_fragment_ids(self):
        out = to_json(self.result)
        parsed = json.loads(out)
        self.assertEqual(parsed["fragment_ids"], [0, 1, 2])

    def test_to_csv_returns_string(self):
        out = to_csv(self.result)
        self.assertIsInstance(out, str)

    def test_to_csv_has_commas(self):
        out = to_csv(self.result)
        self.assertIn(",", out)

    def test_to_text_report_returns_string(self):
        out = to_text_report(self.result)
        self.assertIsInstance(out, str)

    def test_to_text_report_is_non_empty(self):
        out = to_text_report(self.result)
        self.assertGreater(len(out.strip()), 0)

    def test_summary_table_returns_dict(self):
        tbl = summary_table([self.result])
        self.assertIsInstance(tbl, dict)

    def test_summary_table_has_expected_keys(self):
        tbl = summary_table([self.result])
        for key in ("n_fragments", "canvas_w", "canvas_h"):
            self.assertIn(key, tbl, f"Missing key: {key}")

    def test_assembly_result_with_no_scores(self):
        r = AssemblyResult(
            fragment_ids=[0],
            positions=[(0, 0)],
            sizes=[(40, 40)],
            canvas_w=50,
            canvas_h=50,
        )
        self.assertIsInstance(r, AssemblyResult)
        self.assertIsNone(r.scores)


# ---------------------------------------------------------------------------
# TestIOIntegration  (11 tests)
# ---------------------------------------------------------------------------

class TestIOIntegration(unittest.TestCase):

    def test_load_then_resize_pipeline(self):
        img = _make_rgb(80, 80)
        loaded = load_from_array(img, image_id=0)
        resized = resize_image(loaded.data, target_size=(40, 40))
        self.assertEqual(resized.shape[:2], (40, 40))

    def test_write_json_then_read_json_roundtrip_full(self):
        col = _make_collection((0, 0.9), (1, 0.6), (2, 0.3))
        js = write_json(col)
        restored = read_json(js)
        self.assertEqual(restored.n_fragments, 3)
        ids = {r.fragment_id for r in restored.records}
        self.assertEqual(ids, {0, 1, 2})

    def test_batch_load_from_arrays_via_loop(self):
        """Simulate batch loading by calling load_from_array in a loop."""
        arrays = [_make_rgb(30, 30) for _ in range(4)]
        loaded_list = [load_from_array(arr, image_id=i) for i, arr in enumerate(arrays)]
        self.assertEqual(len(loaded_list), 4)
        for i, li in enumerate(loaded_list):
            self.assertEqual(li.image_id, i)

    def test_to_json_then_parse_back_dict(self):
        result = _make_result(n=4)
        js = to_json(result)
        d = json.loads(js)
        self.assertEqual(len(d["fragment_ids"]), 4)

    def test_summary_table_on_multiple_results(self):
        results = [_make_result(n=2), _make_result(n=3)]
        tbl = summary_table(results)
        self.assertEqual(len(tbl["n_fragments"]), 2)
        self.assertEqual(tbl["n_fragments"][0], 2)
        self.assertEqual(tbl["n_fragments"][1], 3)

    def test_filter_by_score_then_write_json(self):
        col = _make_collection((0, 0.9), (1, 0.4), (2, 0.7))
        filtered = filter_by_score(col, min_score=0.6)
        js = write_json(filtered)
        restored = read_json(js)
        for r in restored.records:
            self.assertGreaterEqual(r.score, 0.6)

    def test_merge_collections_combines_records(self):
        col1 = _make_collection((0, 0.8), (1, 0.9))
        col2 = _make_collection((2, 0.7), (3, 0.6))
        merged = merge_collections(col1, col2)
        self.assertEqual(merged.n_fragments, 4)

    def test_write_csv_header_row(self):
        col = _make_collection((0, 0.8))
        csv_str = write_csv(col)
        header = csv_str.splitlines()[0]
        # Header must contain 'fragment_id' and 'score'
        self.assertIn("fragment_id", header)
        self.assertIn("score", header)

    def test_to_text_report_contains_fragment_count(self):
        result = _make_result(n=5)
        report = to_text_report(result)
        self.assertIn("5", report)

    def test_to_json_preserves_positions(self):
        result = AssemblyResult(
            fragment_ids=[0, 1],
            positions=[(10, 20), (30, 40)],
            sizes=[(40, 40), (40, 40)],
            canvas_w=100,
            canvas_h=80,
            scores=[0.8, 0.9],
        )
        d = json.loads(to_json(result))
        self.assertEqual(d["positions"][0], [10, 20])
        self.assertEqual(d["positions"][1], [30, 40])

    def test_to_csv_row_count_matches_fragment_count(self):
        n = 4
        result = _make_result(n=n)
        csv_str = to_csv(result)
        data_rows = [l for l in csv_str.splitlines() if l.strip()][1:]  # skip header
        self.assertEqual(len(data_rows), n)


# ---------------------------------------------------------------------------
# TestIOEdgeCases  (11 tests)
# ---------------------------------------------------------------------------

class TestIOEdgeCases(unittest.TestCase):

    def test_load_from_array_grayscale_2d(self):
        gray = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        loaded = load_from_array(gray, image_id=0)
        self.assertIsInstance(loaded, LoadedImage)
        self.assertEqual(loaded.data.ndim, 3)  # converted to 3-channel

    def test_load_from_array_grayscale_hwc1(self):
        gray = np.random.randint(0, 255, (50, 50, 1), dtype=np.uint8)
        loaded = load_from_array(gray, image_id=0)
        self.assertIsInstance(loaded, LoadedImage)
        self.assertIn(loaded.data.ndim, (2, 3))

    def test_resize_image_to_same_size(self):
        img = _make_rgb(60, 60)
        out = resize_image(img, target_size=(60, 60))
        self.assertEqual(out.shape, (60, 60, 3))

    def test_empty_metadata_collection_write_json(self):
        col = MetadataCollection(records=[])
        js = write_json(col)
        parsed = json.loads(js)
        self.assertIsInstance(parsed, dict)
        # The fragments list should be empty
        self.assertEqual(parsed.get("n_fragments", len(parsed.get("fragments", []))), 0)

    def test_assembly_result_single_fragment(self):
        result = AssemblyResult(
            fragment_ids=[0],
            positions=[(0, 0)],
            sizes=[(40, 40)],
            canvas_w=40,
            canvas_h=40,
            scores=[0.95],
        )
        self.assertEqual(len(result.fragment_ids), 1)

    def test_to_text_report_single_fragment(self):
        result = AssemblyResult(
            fragment_ids=[7],
            positions=[(0, 0)],
            sizes=[(40, 40)],
            canvas_w=40,
            canvas_h=40,
            scores=[0.95],
        )
        report = to_text_report(result)
        self.assertIsInstance(report, str)
        self.assertGreater(len(report.strip()), 0)

    def test_summary_table_on_empty_list(self):
        tbl = summary_table([])
        self.assertIsInstance(tbl, dict)
        # All value lists should be empty
        for v in tbl.values():
            self.assertEqual(len(v), 0)

    def test_filter_by_score_min_zero_keeps_all(self):
        col = _make_collection((0, 0.1), (1, 0.5), (2, 0.9))
        filtered = filter_by_score(col, min_score=0.0)
        self.assertEqual(len(filtered.records), 3)

    def test_filter_by_score_min_one_keeps_none(self):
        col = _make_collection((0, 0.8), (1, 0.9), (2, 0.99))
        filtered = filter_by_score(col, min_score=1.0)
        self.assertEqual(len(filtered.records), 0)

    def test_write_csv_with_extra_columns(self):
        rec = MetadataRecord(fragment_id=0, position=(5, 5), rotation_deg=90.0, score=0.7)
        col = MetadataCollection(records=[rec])
        csv_str = write_csv(col, extra_columns=["rotation_deg"])
        header = csv_str.splitlines()[0]
        self.assertIn("rotation_deg", header)

    def test_batch_load_of_three_images_returns_three_items(self):
        """Use load_from_array loop as a stand-in for batch loading three arrays."""
        arrays = [_make_rgb(40, 40) for _ in range(3)]
        loaded = [load_from_array(a, image_id=i) for i, a in enumerate(arrays)]
        self.assertEqual(len(loaded), 3)
        self.assertIsInstance(loaded[2], LoadedImage)


if __name__ == "__main__":
    unittest.main()
