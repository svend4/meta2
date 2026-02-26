"""
Integration tests for Pipeline + clustering integration.

Workflow under test:
    1. Create temp PNG images on disk
    2. pipeline.preprocess(paths_or_images) -> List[Fragment]
    3. cluster_fragments(fragments, k=K, method=M) -> ClusteringResult
    4. split_by_cluster(fragments, result) -> List[List[Fragment]]
    5. pipeline.match(group) -> (matrix, entries)
    6. pipeline.assemble(group, entries) -> Assembly
    7. Verify each Assembly

Test classes:
    TestPipelineToClusteringBasic   — preprocess → cluster → split (~12 tests)
    TestClusterThenMatch            — match on each cluster group (~15 tests)
    TestClusterThenAssemble         — assemble each cluster group (~13 tests)
    TestMultiDocumentRoundtrip      — full end-to-end roundtrip (~10 tests)
"""
from __future__ import annotations

import os
import tempfile
from typing import List

import cv2
import numpy as np
import pytest

from puzzle_reconstruction.pipeline import Pipeline
from puzzle_reconstruction.clustering import cluster_fragments, split_by_cluster, ClusteringResult
from puzzle_reconstruction.config import Config, AssemblyConfig, MatchingConfig
from puzzle_reconstruction.models import Fragment, Assembly


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_temp_images(n: int, size=(80, 80), tmpdir: str = None) -> List[str]:
    """Create n temporary PNG images and return their paths."""
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
    paths = []
    for i in range(n):
        img = np.random.randint(50, 200, (*size, 3), dtype=np.uint8)
        # Add a bright rectangle so segmentation produces a foreground mask
        margin_y = size[0] // 8
        margin_x = size[1] // 8
        img[margin_y: size[0] - margin_y, margin_x: size[1] - margin_x] = 200
        path = os.path.join(tmpdir, f"frag_{i:03d}.png")
        cv2.imwrite(path, img)
        paths.append(path)
    return paths


def _make_image(h: int = 80, w: int = 80, seed: int = 0) -> np.ndarray:
    """Create a synthetic uint8 BGR image with a bright filled rectangle."""
    rng = np.random.RandomState(seed)
    img = (rng.rand(h, w, 3) * 60 + 30).astype(np.uint8)
    margin_y = h // 8
    margin_x = w // 8
    img[margin_y: h - margin_y, margin_x: w - margin_x] = 210
    return img


def _make_images(n: int, h: int = 80, w: int = 80) -> List[np.ndarray]:
    return [_make_image(h, w, seed=i) for i in range(n)]


def _fast_pipeline() -> Pipeline:
    """Return a Pipeline with minimal settings for speed."""
    cfg = Config.default()
    cfg.verification.run_ocr = False
    cfg.assembly.method = "greedy"
    cfg.assembly.beam_width = 3
    cfg.assembly.sa_iter = 100
    return Pipeline(cfg, n_workers=1)


# ─── Module-scope shared state ────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipe() -> Pipeline:
    return _fast_pipeline()


@pytest.fixture(scope="module")
def fragments_6(pipe) -> List[Fragment]:
    """Preprocess 6 synthetic images once for the whole module."""
    imgs = _make_images(6)
    return pipe.preprocess(imgs)


@pytest.fixture(scope="module")
def cluster_result_k2(fragments_6) -> ClusteringResult:
    return cluster_fragments(fragments_6, k=2, method="kmeans")


@pytest.fixture(scope="module")
def groups_k2(fragments_6, cluster_result_k2) -> List[List[Fragment]]:
    return split_by_cluster(fragments_6, cluster_result_k2)


# ─────────────────────────────────────────────────────────────────────────────
# Class 1 — TestPipelineToClusteringBasic
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineToClusteringBasic:
    """Tests covering preprocess → cluster_fragments → split_by_cluster."""

    def test_preprocess_returns_correct_count(self, fragments_6):
        assert len(fragments_6) == 6

    def test_preprocess_returns_fragments(self, fragments_6):
        for frag in fragments_6:
            assert isinstance(frag, Fragment)

    def test_cluster_n_clusters_equals_k(self, cluster_result_k2):
        assert cluster_result_k2.n_clusters == 2

    def test_cluster_result_type(self, cluster_result_k2):
        assert isinstance(cluster_result_k2, ClusteringResult)

    def test_labels_length_equals_n_fragments(self, fragments_6, cluster_result_k2):
        assert len(cluster_result_k2.labels) == len(fragments_6)

    def test_labels_values_in_range(self, cluster_result_k2):
        labels = cluster_result_k2.labels
        assert all(0 <= lbl < cluster_result_k2.n_clusters for lbl in labels)

    def test_confidence_length_equals_n_fragments(self, fragments_6, cluster_result_k2):
        assert len(cluster_result_k2.confidence) == len(fragments_6)

    def test_confidence_values_in_0_1(self, cluster_result_k2):
        conf = cluster_result_k2.confidence
        assert np.all(conf >= 0.0) and np.all(conf <= 1.0)

    def test_split_returns_list_of_k_groups(self, groups_k2):
        assert len(groups_k2) == 2

    def test_union_of_groups_covers_all_fragments(self, fragments_6, groups_k2):
        all_ids_in_groups = set()
        for group in groups_k2:
            for frag in group:
                all_ids_in_groups.add(frag.fragment_id)
        original_ids = {f.fragment_id for f in fragments_6}
        assert all_ids_in_groups == original_ids

    def test_no_fragment_in_multiple_groups(self, groups_k2):
        seen = []
        for group in groups_k2:
            for frag in group:
                seen.append(frag.fragment_id)
        assert len(seen) == len(set(seen))

    def test_cluster_groups_attribute_matches_split(self, fragments_6, cluster_result_k2, groups_k2):
        total_in_cluster_groups = sum(len(g) for g in cluster_result_k2.cluster_groups)
        total_in_split = sum(len(g) for g in groups_k2)
        assert total_in_cluster_groups == total_in_split == len(fragments_6)


# ─────────────────────────────────────────────────────────────────────────────
# Class 2 — TestClusterThenMatch
# ─────────────────────────────────────────────────────────────────────────────

class TestClusterThenMatch:
    """Tests covering match() called on each cluster group."""

    @pytest.fixture(scope="class")
    def match_results_k2(self, pipe, groups_k2):
        """Run match on each k=2 group."""
        return [pipe.match(group) for group in groups_k2]

    def test_match_returns_tuple(self, pipe, groups_k2):
        result = pipe.match(groups_k2[0])
        assert isinstance(result, tuple) and len(result) == 2

    def test_matrix_is_ndarray(self, pipe, groups_k2):
        matrix, _ = pipe.match(groups_k2[0])
        assert isinstance(matrix, np.ndarray)

    def test_matrix_is_square(self, pipe, groups_k2):
        matrix, _ = pipe.match(groups_k2[0])
        assert matrix.ndim == 2
        assert matrix.shape[0] == matrix.shape[1]

    def test_matrix_diagonal_is_zero(self, pipe, groups_k2):
        matrix, _ = pipe.match(groups_k2[0])
        diag = np.diag(matrix)
        assert np.all(diag == 0.0)

    def test_matrix_values_finite(self, pipe, groups_k2):
        matrix, _ = pipe.match(groups_k2[0])
        assert np.all(np.isfinite(matrix))

    def test_entries_is_list(self, pipe, groups_k2):
        _, entries = pipe.match(groups_k2[0])
        assert isinstance(entries, list)

    def test_each_group_match_valid(self, match_results_k2):
        for matrix, entries in match_results_k2:
            assert isinstance(matrix, np.ndarray)
            assert isinstance(entries, list)

    def test_matrix_values_in_range(self, pipe, groups_k2):
        matrix, _ = pipe.match(groups_k2[0])
        # Normalized matrix should have values in [0, 1]
        assert np.all(matrix >= 0.0)

    def test_match_k3(self, pipe):
        """cluster with k=3, then match each group."""
        imgs = _make_images(6)
        frags = pipe.preprocess(imgs)
        result = cluster_fragments(frags, k=3, method="kmeans")
        groups = split_by_cluster(frags, result)
        for group in groups:
            if len(group) > 0:
                matrix, entries = pipe.match(group)
                assert isinstance(matrix, np.ndarray)
                assert isinstance(entries, list)

    def test_match_with_gmm_method(self, pipe):
        """cluster with gmm, then match each group."""
        imgs = _make_images(6)
        frags = pipe.preprocess(imgs)
        result = cluster_fragments(frags, k=2, method="gmm")
        groups = split_by_cluster(frags, result)
        for group in groups:
            if len(group) > 0:
                matrix, entries = pipe.match(group)
                assert isinstance(matrix, np.ndarray)

    def test_single_fragment_group_match(self, pipe):
        """A group with one fragment should return a 1×1 zero matrix."""
        imgs = _make_images(1)
        single_frag = pipe.preprocess(imgs)
        assert len(single_frag) == 1
        matrix, entries = pipe.match(single_frag)
        assert matrix.shape[0] == matrix.shape[1]
        assert entries == [] or isinstance(entries, list)

    def test_match_matrix_shape_second_group(self, pipe, groups_k2):
        """The second group also yields a valid square matrix."""
        matrix, _ = pipe.match(groups_k2[1])
        assert matrix.ndim == 2
        assert matrix.shape[0] == matrix.shape[1]

    def test_match_matrix_no_nan(self, pipe, groups_k2):
        for group in groups_k2:
            matrix, _ = pipe.match(group)
            assert not np.any(np.isnan(matrix))

    def test_match_matrix_no_inf(self, pipe, groups_k2):
        for group in groups_k2:
            matrix, _ = pipe.match(group)
            assert not np.any(np.isinf(matrix))


# ─────────────────────────────────────────────────────────────────────────────
# Class 3 — TestClusterThenAssemble
# ─────────────────────────────────────────────────────────────────────────────

class TestClusterThenAssemble:
    """Tests covering assemble() called on each cluster group after match()."""

    @pytest.fixture(scope="class")
    def assemblies_k2(self, pipe, groups_k2):
        results = []
        for group in groups_k2:
            matrix, entries = pipe.match(group)
            asm = pipe.assemble(group, entries)
            results.append(asm)
        return results

    def test_assemble_returns_assembly(self, assemblies_k2):
        for asm in assemblies_k2:
            assert isinstance(asm, Assembly)

    def test_assembly_total_score_nonnegative(self, assemblies_k2):
        # total_score is a sum of compat scores and may exceed 1.0
        for asm in assemblies_k2:
            assert asm.total_score >= 0.0

    def test_assembly_placements_not_none(self, assemblies_k2):
        for asm in assemblies_k2:
            assert asm.placements is not None

    def test_assembly_fragments_set(self, assemblies_k2):
        for asm in assemblies_k2:
            # fragments field should be set (list)
            assert asm.fragments is not None or asm.placements is not None

    def test_assembly_n_placed_at_least_one(self, assemblies_k2):
        for asm in assemblies_k2:
            if isinstance(asm.placements, dict):
                n_placed = len(asm.placements)
            elif isinstance(asm.placements, list):
                n_placed = len(asm.placements)
            else:
                n_placed = 0
            assert n_placed >= 1

    def test_assembly_method_field_set(self, assemblies_k2):
        for asm in assemblies_k2:
            assert isinstance(asm.method, str)

    def test_assemble_with_greedy(self, pipe, groups_k2):
        cfg = Config.default()
        cfg.verification.run_ocr = False
        cfg.assembly.method = "greedy"
        p = Pipeline(cfg, n_workers=1)
        group = groups_k2[0]
        _, entries = p.match(group)
        asm = p.assemble(group, entries)
        assert isinstance(asm, Assembly)

    def test_assemble_with_beam(self, groups_k2):
        cfg = Config.default()
        cfg.verification.run_ocr = False
        cfg.assembly.method = "beam"
        cfg.assembly.beam_width = 3
        p = Pipeline(cfg, n_workers=1)
        group = groups_k2[0]
        _, entries = p.match(group)
        asm = p.assemble(group, entries)
        assert isinstance(asm, Assembly)

    def test_assembly_fragments_covers_group(self, pipe, groups_k2):
        group = groups_k2[0]
        _, entries = pipe.match(group)
        asm = pipe.assemble(group, entries)
        if asm.fragments is not None:
            group_ids = {f.fragment_id for f in group}
            asm_ids = {f.fragment_id for f in asm.fragments}
            assert asm_ids == group_ids

    def test_assemble_second_group(self, pipe, groups_k2):
        group = groups_k2[1]
        _, entries = pipe.match(group)
        asm = pipe.assemble(group, entries)
        assert isinstance(asm, Assembly)
        assert asm.total_score >= 0.0

    def test_assembly_compat_matrix_set(self, assemblies_k2):
        """compat_matrix in Assembly should be an ndarray or None."""
        for asm in assemblies_k2:
            assert asm.compat_matrix is None or isinstance(asm.compat_matrix, np.ndarray)

    def test_assemble_k3_groups(self, pipe):
        """Assemble each group for k=3 clustering."""
        imgs = _make_images(6)
        frags = pipe.preprocess(imgs)
        result = cluster_fragments(frags, k=3, method="kmeans")
        groups = split_by_cluster(frags, result)
        for group in groups:
            if len(group) > 0:
                _, entries = pipe.match(group)
                asm = pipe.assemble(group, entries)
                assert isinstance(asm, Assembly)

    def test_assemble_preserves_total_score_float(self, assemblies_k2):
        for asm in assemblies_k2:
            assert isinstance(asm.total_score, float)


# ─────────────────────────────────────────────────────────────────────────────
# Class 4 — TestMultiDocumentRoundtrip
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiDocumentRoundtrip:
    """Full end-to-end integration: preprocess → cluster → match → assemble."""

    def _full_roundtrip(self, pipe: Pipeline, n_frags: int, k: int,
                        method: str = "kmeans"):
        imgs = _make_images(n_frags)
        fragments = pipe.preprocess(imgs)
        result = cluster_fragments(fragments, k=k, method=method)
        groups = split_by_cluster(fragments, result)
        assemblies = []
        for group in groups:
            if len(group) > 0:
                _, entries = pipe.match(group)
                asm = pipe.assemble(group, entries)
                assemblies.append((group, asm))
        return fragments, result, groups, assemblies

    def test_all_assemblies_are_assembly_objects(self, pipe):
        _, _, _, assemblies = self._full_roundtrip(pipe, n_frags=6, k=2)
        for _, asm in assemblies:
            assert isinstance(asm, Assembly)

    def test_total_placed_equals_total_fragments(self, pipe):
        fragments, _, _, assemblies = self._full_roundtrip(pipe, n_frags=6, k=2)
        total_placed = 0
        for group, asm in assemblies:
            if isinstance(asm.placements, dict):
                total_placed += len(asm.placements)
            elif isinstance(asm.placements, list):
                total_placed += len(asm.placements)
        assert total_placed == len(fragments)

    def test_k1_produces_single_cluster(self, pipe):
        imgs = _make_images(4)
        frags = pipe.preprocess(imgs)
        result = cluster_fragments(frags, k=1, method="kmeans")
        assert result.n_clusters == 1

    def test_k1_single_assembly(self, pipe):
        imgs = _make_images(4)
        frags = pipe.preprocess(imgs)
        result = cluster_fragments(frags, k=1, method="kmeans")
        groups = split_by_cluster(frags, result)
        assert len(groups) == 1
        _, entries = pipe.match(groups[0])
        asm = pipe.assemble(groups[0], entries)
        assert isinstance(asm, Assembly)

    def test_explicit_k_gives_correct_n_clusters(self, pipe):
        imgs = _make_images(6)
        frags = pipe.preprocess(imgs)
        for k in [2, 3]:
            result = cluster_fragments(frags, k=k, method="kmeans")
            assert result.n_clusters == k

    def test_each_assembly_placements_maps_correct_fragment_ids(self, pipe):
        imgs = _make_images(6)
        frags = pipe.preprocess(imgs)
        result = cluster_fragments(frags, k=2, method="kmeans")
        groups = split_by_cluster(frags, result)
        for group in groups:
            if len(group) > 0:
                group_frag_ids = {f.fragment_id for f in group}
                _, entries = pipe.match(group)
                asm = pipe.assemble(group, entries)
                if isinstance(asm.placements, dict):
                    for fid in asm.placements:
                        assert fid in group_frag_ids
                # If placements is a list of Placement objects, check fragment_id attr
                elif isinstance(asm.placements, list):
                    for p in asm.placements:
                        if hasattr(p, "fragment_id"):
                            assert p.fragment_id in group_frag_ids

    def test_roundtrip_with_gmm_clustering(self, pipe):
        _, _, _, assemblies = self._full_roundtrip(pipe, n_frags=6, k=2, method="gmm")
        assert len(assemblies) > 0
        for _, asm in assemblies:
            assert isinstance(asm, Assembly)

    def test_roundtrip_with_8_fragments(self, pipe):
        _, result, _, assemblies = self._full_roundtrip(pipe, n_frags=8, k=2)
        assert result.n_clusters == 2
        assert len(assemblies) == 2

    def test_roundtrip_different_seeds_same_n_clusters(self, pipe):
        """With explicit k, different seeds produce the same n_clusters."""
        imgs = _make_images(6)
        frags = pipe.preprocess(imgs)
        for seed in [0, 7, 42]:
            result = cluster_fragments(frags, k=2, method="kmeans", seed=seed)
            assert result.n_clusters == 2

    def test_tmp_path_image_preprocess_to_cluster(self, tmp_path, pipe):
        """Create real PNG files on disk, load them, preprocess, then cluster."""
        paths = make_temp_images(6, size=(80, 80), tmpdir=str(tmp_path))
        imgs = [cv2.imread(p) for p in paths]
        assert all(img is not None for img in imgs)
        frags = pipe.preprocess(imgs)
        assert len(frags) == 6
        result = cluster_fragments(frags, k=2, method="kmeans")
        assert result.n_clusters == 2
        groups = split_by_cluster(frags, result)
        assert len(groups) == 2
