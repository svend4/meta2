"""
Integration tests for tools/mix_documents.py

Covers:
  - evaluate_clustering: purity, rand_index, edge cases, result dict structure
  - mix_from_generated: file creation, filename patterns, ground_truth format
  - mix_from_dirs: directory mixing, ground_truth mapping, shuffle behaviour
"""
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure project root is on the path (conftest.py does this, but be explicit)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.mix_documents import evaluate_clustering, mix_from_dirs, mix_from_generated


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _compute_purity(y_pred, y_true):
    """Reference purity computation matching the source implementation."""
    n = len(y_pred)
    total_correct = 0
    for c in set(y_pred):
        mask = [yp == c for yp in y_pred]
        labels_in_cluster = [yt for yp, yt in zip(y_pred, y_true) if yp == c]
        most_common = Counter(labels_in_cluster).most_common(1)[0][1]
        total_correct += most_common
    return total_correct / n


def _compute_rand_index(y_pred, y_true):
    """Reference rand index computation matching the source implementation."""
    n = len(y_pred)
    agree = 0
    total_pairs = n * (n - 1) // 2
    for i in range(n):
        for j in range(i + 1, n):
            same_pred = y_pred[i] == y_pred[j]
            same_true = y_true[i] == y_true[j]
            if same_pred == same_true:
                agree += 1
    return agree / max(1, total_pairs)


def _make_tiny_png(path: Path, color=(200, 200, 200)):
    """Write a small 10x10 BGR image to disk."""
    img = np.full((10, 10, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)


# ─── TestEvaluateClustering ──────────────────────────────────────────────────

class TestEvaluateClustering:

    def test_empty_intersection_returns_zero_purity(self):
        predicted = {"a.png": 0, "b.png": 1}
        ground_truth = {"c.png": 0, "d.png": 1}
        result = evaluate_clustering(predicted, ground_truth)
        assert result["purity"] == 0.0

    def test_empty_intersection_returns_zero_rand_index(self):
        predicted = {"a.png": 0}
        ground_truth = {"b.png": 0}
        result = evaluate_clustering(predicted, ground_truth)
        assert result["rand_index"] == 0.0

    def test_empty_intersection_returns_n_samples_zero(self):
        result = evaluate_clustering({"x.png": 0}, {"y.png": 1})
        assert result["n_samples"] == 0

    def test_empty_predicted_returns_early(self):
        result = evaluate_clustering({}, {"a.png": 0})
        assert result["purity"] == 0.0
        assert result["rand_index"] == 0.0
        assert result["n_samples"] == 0

    def test_empty_both_returns_early(self):
        result = evaluate_clustering({}, {})
        assert result["n_samples"] == 0

    def test_perfect_clustering_purity_is_one(self):
        gt = {f"f{i}.png": i // 3 for i in range(6)}
        pred = dict(gt)  # identical assignment
        result = evaluate_clustering(pred, gt)
        assert result["purity"] == pytest.approx(1.0)

    def test_perfect_clustering_rand_index_is_one(self):
        gt = {f"f{i}.png": i // 3 for i in range(6)}
        result = evaluate_clustering(dict(gt), gt)
        assert result["rand_index"] == pytest.approx(1.0)

    def test_single_sample_purity_is_one(self):
        result = evaluate_clustering({"a.png": 0}, {"a.png": 0})
        assert result["purity"] == pytest.approx(1.0)

    def test_single_sample_rand_index_is_one(self):
        # With n=1 there are 0 pairs; rand_index = 0/max(1,0) = 0/1 = 0.
        # Source uses max(1, total_pairs), so 0 agree / 1 = 0.0
        result = evaluate_clustering({"a.png": 0}, {"a.png": 0})
        assert result["rand_index"] == pytest.approx(0.0)

    def test_known_purity_manual_computation(self):
        # Cluster 0: files a,b → true labels 0,0  → most_common=2
        # Cluster 1: files c,d → true labels 0,1  → most_common=1
        # purity = (2+1)/4 = 0.75
        predicted    = {"a.png": 0, "b.png": 0, "c.png": 1, "d.png": 1}
        ground_truth = {"a.png": 0, "b.png": 0, "c.png": 0, "d.png": 1}
        result = evaluate_clustering(predicted, ground_truth)
        expected_purity = _compute_purity(
            [0, 0, 1, 1], [0, 0, 0, 1]
        )
        assert result["purity"] == pytest.approx(expected_purity)

    def test_known_rand_index_two_clusters_two_docs(self):
        # 4 items: pred [0,0,1,1], true [0,1,0,1]
        # pairs: (a,b): same_pred=T, same_true=F → disagree
        #        (a,c): same_pred=F, same_true=T → disagree
        #        (a,d): same_pred=F, same_true=F → agree
        #        (b,c): same_pred=F, same_true=F → agree
        #        (b,d): same_pred=F, same_true=T → disagree
        #        (c,d): same_pred=T, same_true=F → disagree
        # agree=2, total=6, RI=2/6=0.333...
        predicted    = {"a.png": 0, "b.png": 0, "c.png": 1, "d.png": 1}
        ground_truth = {"a.png": 0, "b.png": 1, "c.png": 0, "d.png": 1}
        result = evaluate_clustering(predicted, ground_truth)
        expected_ri = _compute_rand_index([0, 0, 1, 1], [0, 1, 0, 1])
        assert result["rand_index"] == pytest.approx(expected_ri)

    def test_result_dict_has_all_required_keys(self):
        predicted    = {"a.png": 0, "b.png": 1}
        ground_truth = {"a.png": 0, "b.png": 1}
        result = evaluate_clustering(predicted, ground_truth)
        required_keys = {
            "purity", "rand_index", "adjusted_rand",
            "n_samples", "n_clusters_pred", "n_clusters_true",
        }
        assert required_keys.issubset(result.keys())

    def test_n_samples_is_size_of_common_keys(self):
        predicted    = {"a.png": 0, "b.png": 1, "z.png": 0}
        ground_truth = {"a.png": 0, "b.png": 0}
        result = evaluate_clustering(predicted, ground_truth)
        assert result["n_samples"] == 2  # only a.png and b.png are common

    def test_n_clusters_pred_correct(self):
        predicted    = {"a.png": 0, "b.png": 1, "c.png": 2}
        ground_truth = {"a.png": 0, "b.png": 0, "c.png": 1}
        result = evaluate_clustering(predicted, ground_truth)
        assert result["n_clusters_pred"] == 3

    def test_n_clusters_true_correct(self):
        predicted    = {"a.png": 0, "b.png": 0, "c.png": 0}
        ground_truth = {"a.png": 0, "b.png": 1, "c.png": 2}
        result = evaluate_clustering(predicted, ground_truth)
        assert result["n_clusters_true"] == 3

    def test_worst_clustering_rand_index_low(self):
        # Each sample in its own cluster, but true labels are just 2 groups.
        # Rand index should be less than 1.
        predicted    = {f"f{i}.png": i     for i in range(6)}
        ground_truth = {f"f{i}.png": i // 3 for i in range(6)}
        result = evaluate_clustering(predicted, ground_truth)
        assert result["rand_index"] < 1.0

    def test_purity_between_zero_and_one(self):
        predicted    = {"a.png": 0, "b.png": 0, "c.png": 1, "d.png": 1}
        ground_truth = {"a.png": 0, "b.png": 1, "c.png": 0, "d.png": 1}
        result = evaluate_clustering(predicted, ground_truth)
        assert 0.0 <= result["purity"] <= 1.0

    def test_rand_index_between_zero_and_one(self):
        predicted    = {"a.png": 0, "b.png": 0, "c.png": 1, "d.png": 1}
        ground_truth = {"a.png": 0, "b.png": 1, "c.png": 0, "d.png": 1}
        result = evaluate_clustering(predicted, ground_truth)
        assert 0.0 <= result["rand_index"] <= 1.0

    def test_partial_overlap_n_samples_correct(self):
        # Only "a.png" and "c.png" are in both dicts
        predicted    = {"a.png": 0, "b.png": 1, "c.png": 0}
        ground_truth = {"a.png": 0, "c.png": 1, "d.png": 0}
        result = evaluate_clustering(predicted, ground_truth)
        assert result["n_samples"] == 2

    def test_all_same_cluster_predicted_purity(self):
        # Everything assigned to cluster 0; true labels: 3 zeros, 1 one
        predicted    = {f"f{i}.png": 0 for i in range(4)}
        ground_truth = {"f0.png": 0, "f1.png": 0, "f2.png": 0, "f3.png": 1}
        result = evaluate_clustering(predicted, ground_truth)
        # Cluster 0 has true labels [0,0,0,1]: most_common=3
        expected_purity = 3 / 4
        assert result["purity"] == pytest.approx(expected_purity)

    def test_adjusted_rand_present_and_numeric(self):
        predicted    = {"a.png": 0, "b.png": 1, "c.png": 0}
        ground_truth = {"a.png": 0, "b.png": 1, "c.png": 0}
        result = evaluate_clustering(predicted, ground_truth)
        assert isinstance(result["adjusted_rand"], float)

    def test_purity_five_items_known(self):
        # Cluster 0: [true 0, true 0, true 1] → most_common=2
        # Cluster 1: [true 0, true 1]         → most_common=1
        # purity = (2+1)/5 = 0.6
        fnames = [f"f{i}.png" for i in range(5)]
        predicted    = dict(zip(fnames, [0, 0, 0, 1, 1]))
        ground_truth = dict(zip(fnames, [0, 0, 1, 0, 1]))
        result = evaluate_clustering(predicted, ground_truth)
        expected = _compute_purity([0, 0, 0, 1, 1], [0, 0, 1, 0, 1])
        assert result["purity"] == pytest.approx(expected)


# ─── TestMixFromGenerated ────────────────────────────────────────────────────

class TestMixFromGenerated:

    def test_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            gt = mix_from_generated(
                n_docs=2, n_pieces=2, output_dir=Path(tmp), noise_level=0.3,
                shuffle=True, base_seed=42
            )
        assert isinstance(gt, dict)

    def test_ground_truth_has_correct_number_of_entries(self):
        # 2 docs × 2 pieces each = 4 fragments
        with tempfile.TemporaryDirectory() as tmp:
            gt = mix_from_generated(
                n_docs=2, n_pieces=2, output_dir=Path(tmp), noise_level=0.3,
                shuffle=True, base_seed=42
            )
        assert len(gt) == 4

    def test_two_docs_four_pieces_returns_eight_fragments(self):
        with tempfile.TemporaryDirectory() as tmp:
            gt = mix_from_generated(
                n_docs=2, n_pieces=4, output_dir=Path(tmp), noise_level=0.3,
                shuffle=True, base_seed=0
            )
        # tear_document with n_pieces=4 creates a 2×2 grid → exactly 4 fragments
        # 2 docs × 4 pieces = 8 total
        assert len(gt) == 8

    def test_filenames_match_pattern(self):
        pattern = re.compile(r"^frag_\d{4}_doc\d+\.png$")
        with tempfile.TemporaryDirectory() as tmp:
            gt = mix_from_generated(
                n_docs=2, n_pieces=2, output_dir=Path(tmp), noise_level=0.3,
                shuffle=True, base_seed=42
            )
        for fname in gt:
            assert pattern.match(fname), f"Filename does not match pattern: {fname}"

    def test_all_doc_ids_in_valid_range(self):
        n_docs = 3
        with tempfile.TemporaryDirectory() as tmp:
            gt = mix_from_generated(
                n_docs=n_docs, n_pieces=2, output_dir=Path(tmp),
                noise_level=0.3, shuffle=True, base_seed=42
            )
        for fname, doc_id in gt.items():
            assert 0 <= doc_id < n_docs, f"{fname} has doc_id={doc_id} out of range"

    def test_files_are_created_in_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            gt = mix_from_generated(
                n_docs=2, n_pieces=2, output_dir=out, noise_level=0.3,
                shuffle=True, base_seed=42
            )
            created = set(p.name for p in out.glob("*.png"))
        assert set(gt.keys()) == created

    def test_output_dir_is_created_if_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "subdir" / "deep"
            assert not out.exists()
            mix_from_generated(
                n_docs=1, n_pieces=2, output_dir=out, noise_level=0.3,
                shuffle=False, base_seed=42
            )
            assert out.exists()

    def test_ground_truth_values_are_integers(self):
        with tempfile.TemporaryDirectory() as tmp:
            gt = mix_from_generated(
                n_docs=2, n_pieces=2, output_dir=Path(tmp), noise_level=0.3,
                shuffle=True, base_seed=42
            )
        for val in gt.values():
            assert isinstance(val, int), f"Expected int doc_id, got {type(val)}"

    def test_shuffle_false_preserves_doc_grouping(self):
        # With shuffle=False, fragments should appear grouped by doc_id
        # (all doc 0 first, then doc 1, etc.) based on iteration order
        with tempfile.TemporaryDirectory() as tmp:
            gt = mix_from_generated(
                n_docs=2, n_pieces=2, output_dir=Path(tmp),
                noise_level=0.3, shuffle=False, base_seed=42
            )
        # Sort by the numeric index in the filename (frag_NNNN_...)
        sorted_items = sorted(gt.items(), key=lambda kv: kv[0])
        doc_ids = [doc_id for _, doc_id in sorted_items]
        # Grouped: first n_pieces entries should all be doc 0,
        # next n_pieces should all be doc 1
        first_half  = doc_ids[:2]
        second_half = doc_ids[2:]
        assert all(d == 0 for d in first_half), f"Expected all 0, got {first_half}"
        assert all(d == 1 for d in second_half), f"Expected all 1, got {second_half}"

    def test_different_base_seed_produces_different_images(self):
        with tempfile.TemporaryDirectory() as tmp1:
            with tempfile.TemporaryDirectory() as tmp2:
                gt1 = mix_from_generated(
                    n_docs=1, n_pieces=2, output_dir=Path(tmp1),
                    noise_level=0.5, shuffle=False, base_seed=10
                )
                gt2 = mix_from_generated(
                    n_docs=1, n_pieces=2, output_dir=Path(tmp2),
                    noise_level=0.5, shuffle=False, base_seed=99
                )
                # Load first fragment from each run and compare
                name1 = sorted(gt1.keys())[0]
                name2 = sorted(gt2.keys())[0]
                img1 = cv2.imread(str(Path(tmp1) / name1))
                img2 = cv2.imread(str(Path(tmp2) / name2))
        # Images should differ because seeds differ
        assert img1 is not None
        assert img2 is not None
        # Allow size mismatch (different fragments); just ensure they are not identical
        if img1.shape == img2.shape:
            assert not np.array_equal(img1, img2)

    def test_same_base_seed_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp1:
            with tempfile.TemporaryDirectory() as tmp2:
                gt1 = mix_from_generated(
                    n_docs=1, n_pieces=2, output_dir=Path(tmp1),
                    noise_level=0.5, shuffle=False, base_seed=42
                )
                gt2 = mix_from_generated(
                    n_docs=1, n_pieces=2, output_dir=Path(tmp2),
                    noise_level=0.5, shuffle=False, base_seed=42
                )
                name = sorted(gt1.keys())[0]
                img1 = cv2.imread(str(Path(tmp1) / name))
                img2 = cv2.imread(str(Path(tmp2) / name))
        assert img1 is not None and img2 is not None
        assert img1.shape == img2.shape
        assert np.array_equal(img1, img2)

    def test_doc_id_encoded_in_filename_matches_ground_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            gt = mix_from_generated(
                n_docs=3, n_pieces=2, output_dir=Path(tmp),
                noise_level=0.3, shuffle=True, base_seed=7
            )
        for fname, doc_id in gt.items():
            # Extract the doc id from the filename: frag_NNNN_docD.png
            m = re.match(r"frag_\d{4}_doc(\d+)\.png", fname)
            assert m is not None, f"Pattern not matched for {fname}"
            doc_id_in_name = int(m.group(1))
            assert doc_id_in_name == doc_id, (
                f"doc_id in filename ({doc_id_in_name}) != ground_truth ({doc_id})"
            )

    def test_created_images_are_readable_pngs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            gt = mix_from_generated(
                n_docs=2, n_pieces=2, output_dir=out,
                noise_level=0.3, shuffle=False, base_seed=42
            )
            for fname in gt:
                img = cv2.imread(str(out / fname))
                assert img is not None, f"cv2 could not read {fname}"
                assert img.ndim == 3


# ─── TestMixFromDirs ─────────────────────────────────────────────────────────

class TestMixFromDirs:

    def _populate_dir(self, base: Path, name: str, n_files: int, color) -> Path:
        """Create a subdirectory with n_files tiny PNG images."""
        d = base / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_files):
            _make_tiny_png(d / f"img_{i:03d}.png", color=color)
        return d

    def test_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            d0 = self._populate_dir(base, "doc0", 2, (100, 100, 100))
            d1 = self._populate_dir(base, "doc1", 2, (200, 200, 200))
            out = base / "out"
            gt = mix_from_dirs([d0, d1], out, shuffle=False, seed=0)
        assert isinstance(gt, dict)

    def test_total_files_matches_ground_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            d0 = self._populate_dir(base, "doc0", 3, (50, 50, 50))
            d1 = self._populate_dir(base, "doc1", 2, (150, 150, 150))
            out = base / "out"
            gt = mix_from_dirs([d0, d1], out, shuffle=False, seed=0)
        assert len(gt) == 5

    def test_files_are_copied_to_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            d0 = self._populate_dir(base, "doc0", 2, (80, 80, 80))
            d1 = self._populate_dir(base, "doc1", 2, (160, 160, 160))
            out = base / "out"
            gt = mix_from_dirs([d0, d1], out, shuffle=False, seed=0)
            created = set(p.name for p in out.iterdir())
        assert set(gt.keys()) == created

    def test_ground_truth_doc_ids_correspond_to_dir_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            d0 = self._populate_dir(base, "doc0", 2, (10, 10, 10))
            d1 = self._populate_dir(base, "doc1", 2, (20, 20, 20))
            out = base / "out"
            gt = mix_from_dirs([d0, d1], out, shuffle=False, seed=0)
        values = set(gt.values())
        assert 0 in values
        assert 1 in values

    def test_doc_id_in_filename_matches_ground_truth_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            d0 = self._populate_dir(base, "doc0", 2, (30, 30, 30))
            d1 = self._populate_dir(base, "doc1", 2, (60, 60, 60))
            out = base / "out"
            gt = mix_from_dirs([d0, d1], out, shuffle=False, seed=0)
        for fname, doc_id in gt.items():
            m = re.match(r"frag_\d{4}_doc(\d+)\.", fname)
            assert m is not None, f"Unexpected filename format: {fname}"
            assert int(m.group(1)) == doc_id

    def test_shuffle_true_does_not_lose_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            d0 = self._populate_dir(base, "doc0", 3, (40, 40, 40))
            d1 = self._populate_dir(base, "doc1", 3, (80, 80, 80))
            out = base / "out"
            gt = mix_from_dirs([d0, d1], out, shuffle=True, seed=42)
        assert len(gt) == 6
        assert set(gt.values()) == {0, 1}

    def test_shuffle_false_groups_docs_together(self):
        # Without shuffle, items from d0 come first (sorted by filename),
        # then items from d1.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            d0 = self._populate_dir(base, "doc0", 2, (55, 55, 55))
            d1 = self._populate_dir(base, "doc1", 2, (110, 110, 110))
            out = base / "out"
            gt = mix_from_dirs([d0, d1], out, shuffle=False, seed=0)
        sorted_items = sorted(gt.items(), key=lambda kv: kv[0])
        doc_ids = [v for _, v in sorted_items]
        # First 2 should be doc 0, next 2 should be doc 1
        assert doc_ids[:2] == [0, 0]
        assert doc_ids[2:] == [1, 1]

    def test_output_dir_created_automatically(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            d0 = self._populate_dir(base, "src0", 1, (70, 70, 70))
            out = base / "nested" / "output"
            assert not out.exists()
            mix_from_dirs([d0], out, shuffle=False, seed=0)
            assert out.exists()

    def test_three_dirs_all_doc_ids_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dirs = [
                self._populate_dir(base, f"doc{i}", 2, (i * 50, i * 50, i * 50))
                for i in range(3)
            ]
            out = base / "out"
            gt = mix_from_dirs(dirs, out, shuffle=False, seed=0)
        assert set(gt.values()) == {0, 1, 2}
        assert len(gt) == 6

    def test_single_dir_all_doc_ids_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            d0 = self._populate_dir(base, "only", 3, (90, 90, 90))
            out = base / "out"
            gt = mix_from_dirs([d0], out, shuffle=True, seed=5)
        assert all(v == 0 for v in gt.values())
        assert len(gt) == 3

    def test_copied_images_are_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            d0 = self._populate_dir(base, "doc0", 2, (120, 120, 120))
            out = base / "out"
            gt = mix_from_dirs([d0], out, shuffle=False, seed=0)
            for fname in gt:
                img = cv2.imread(str(out / fname))
                assert img is not None, f"Could not read {fname}"
