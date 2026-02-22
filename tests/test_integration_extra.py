"""
Additional integration tests supplementing test_integration.py.

These tests exercise additional code paths: alternate segmentation methods,
contour properties, fractal signature bounds, compat matrix properties,
and assembly edge cases.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.tear_generator import tear_document, generate_test_document
from puzzle_reconstruction.models import Fragment
from puzzle_reconstruction.preprocessing.segmentation import segment_fragment
from puzzle_reconstruction.preprocessing.contour import extract_contour
from puzzle_reconstruction.algorithms.tangram.inscriber import fit_tangram
from puzzle_reconstruction.algorithms.synthesis import (
    compute_fractal_signature, build_edge_signatures
)
from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
from puzzle_reconstruction.assembly.greedy import greedy_assembly
from puzzle_reconstruction.assembly.annealing import simulated_annealing
from puzzle_reconstruction.assembly.beam_search import beam_search
from puzzle_reconstruction.verification.metrics import evaluate_reconstruction
from puzzle_reconstruction.config import Config


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def doc_300x400():
    return generate_test_document(width=300, height=400, seed=77)


@pytest.fixture(scope="module")
def torn_2(doc_300x400):
    return tear_document(doc_300x400, n_pieces=2, noise_level=0.2, seed=5)


@pytest.fixture(scope="module")
def torn_6(doc_300x400):
    return tear_document(doc_300x400, n_pieces=6, noise_level=0.5, seed=20)


@pytest.fixture(scope="module")
def frags_2(torn_2):
    return _build_fragments(torn_2)


@pytest.fixture(scope="module")
def frags_6(torn_6):
    return _build_fragments(torn_6)


def _build_fragments(images):
    fragments = []
    for idx, img in enumerate(images):
        try:
            mask = segment_fragment(img, method="otsu")
            contour = extract_contour(mask)
            tangram = fit_tangram(contour)
            fractal = compute_fractal_signature(contour)
            frag = Fragment(fragment_id=idx, image=img, mask=mask, contour=contour)
            frag.tangram = tangram
            frag.fractal = fractal
            frag.edges = build_edge_signatures(frag, alpha=0.5, n_sides=4)
            fragments.append(frag)
        except Exception:
            pass
    return fragments


# ─── TestDocumentGenerationExtra ─────────────────────────────────────────────

class TestDocumentGenerationExtra:
    def test_doc_300x400_shape(self, doc_300x400):
        assert doc_300x400.shape == (400, 300, 3)

    def test_doc_dtype_uint8(self, doc_300x400):
        assert doc_300x400.dtype == np.uint8

    def test_two_fragments_count(self, torn_2):
        assert len(torn_2) == 2

    def test_six_fragments_count(self, torn_6):
        assert len(torn_6) >= 4

    def test_fragment_images_are_3channel(self, torn_6):
        for img in torn_6:
            assert img.ndim == 3
            assert img.shape[2] == 3

    def test_fragment_images_uint8(self, torn_2):
        for img in torn_2:
            assert img.dtype == np.uint8

    def test_no_fragment_all_zeros(self, torn_2):
        for img in torn_2:
            assert np.any(img > 0)


# ─── TestPreprocessingExtra ───────────────────────────────────────────────────

class TestPreprocessingExtra:
    def test_segment_adaptive_mask(self, torn_2):
        img = torn_2[0]
        mask = segment_fragment(img, method="adaptive")
        assert mask.shape == img.shape[:2]
        assert np.any(mask > 0)

    def test_contour_min_4_points(self, torn_2):
        img = torn_2[0]
        mask = segment_fragment(img)
        contour = extract_contour(mask)
        assert len(contour) >= 4

    def test_contour_2d_shape(self, torn_2):
        img = torn_2[0]
        mask = segment_fragment(img)
        contour = extract_contour(mask)
        assert contour.ndim == 2
        assert contour.shape[1] == 2

    def test_contour_values_within_image(self, torn_2):
        img = torn_2[0]
        h, w = img.shape[:2]
        mask = segment_fragment(img)
        contour = extract_contour(mask)
        assert np.all(contour[:, 0] >= 0) and np.all(contour[:, 0] <= w)
        assert np.all(contour[:, 1] >= 0) and np.all(contour[:, 1] <= h)

    def test_mask_binary(self, torn_2):
        img = torn_2[0]
        mask = segment_fragment(img)
        unique = np.unique(mask)
        assert set(unique).issubset({0, 255})


# ─── TestAlgorithmsExtra ─────────────────────────────────────────────────────

class TestAlgorithmsExtra:
    def test_tangram_polygon_at_least_3(self, torn_2):
        img = torn_2[0]
        mask = segment_fragment(img)
        contour = extract_contour(mask)
        sig = fit_tangram(contour)
        assert len(sig.polygon) >= 3

    def test_fractal_fd_box_in_range(self, torn_2):
        img = torn_2[0]
        mask = segment_fragment(img)
        contour = extract_contour(mask)
        fd = compute_fractal_signature(contour)
        assert 1.0 <= fd.fd_box <= 2.0

    def test_fractal_ifs_coeffs_bounded(self, torn_2):
        img = torn_2[0]
        mask = segment_fragment(img)
        contour = extract_contour(mask)
        fd = compute_fractal_signature(contour)
        assert np.all(np.abs(fd.ifs_coeffs) < 1.0)

    def test_edges_fd_in_range(self, frags_2):
        for frag in frags_2:
            for edge in frag.edges:
                assert 1.0 <= edge.fd <= 2.0

    def test_edges_virtual_curve_2d(self, frags_2):
        for frag in frags_2:
            for edge in frag.edges:
                assert edge.virtual_curve.ndim == 2
                assert edge.virtual_curve.shape[1] == 2


# ─── TestMatchingExtra ────────────────────────────────────────────────────────

class TestMatchingExtra:
    def test_compat_matrix_float(self, frags_2):
        matrix, _ = build_compat_matrix(frags_2, threshold=0.0)
        assert np.issubdtype(matrix.dtype, np.floating)

    def test_compat_matrix_diagonal_zero(self, frags_2):
        matrix, _ = build_compat_matrix(frags_2, threshold=0.0)
        np.testing.assert_array_equal(np.diag(matrix), 0.0)

    def test_compat_scores_in_01(self, frags_2):
        _, entries = build_compat_matrix(frags_2, threshold=0.0)
        for e in entries:
            assert 0.0 <= e.score <= 1.0

    def test_entries_have_edge_refs(self, frags_2):
        _, entries = build_compat_matrix(frags_2, threshold=0.0)
        for e in entries:
            assert e.edge_i is not None
            assert e.edge_j is not None

    def test_threshold_filters_entries(self, frags_2):
        _, entries_low = build_compat_matrix(frags_2, threshold=0.0)
        _, entries_high = build_compat_matrix(frags_2, threshold=0.8)
        assert len(entries_high) <= len(entries_low)


# ─── TestAssemblyExtra ────────────────────────────────────────────────────────

class TestAssemblyExtra:
    def test_greedy_2_fragments_places_both(self, frags_2):
        _, entries = build_compat_matrix(frags_2)
        asm = greedy_assembly(frags_2, entries)
        assert len(asm.placements) == len(frags_2)

    def test_beam_width_1_places_all(self, frags_2):
        _, entries = build_compat_matrix(frags_2)
        asm = beam_search(frags_2, entries, beam_width=1)
        assert len(asm.placements) == len(frags_2)

    def test_sa_n_iter_100_terminates(self, frags_2):
        _, entries = build_compat_matrix(frags_2)
        asm0 = greedy_assembly(frags_2, entries)
        asm1 = simulated_annealing(asm0, entries, T_max=50, max_iter=100, seed=0)
        assert len(asm1.placements) == len(frags_2)

    def test_greedy_placements_are_dicts(self, frags_2):
        _, entries = build_compat_matrix(frags_2)
        asm = greedy_assembly(frags_2, entries)
        assert isinstance(asm.placements, dict)

    def test_placements_positions_2d_array(self, frags_2):
        _, entries = build_compat_matrix(frags_2)
        asm = greedy_assembly(frags_2, entries)
        for pos, _ in asm.placements.values():
            assert pos.shape == (2,)


# ─── TestMetricsExtra ─────────────────────────────────────────────────────────

class TestMetricsExtra:
    def test_perfect_reconstruction_dc_geq_09(self, frags_2):
        gt = {f.fragment_id: (np.array([float(f.fragment_id) * 100, 0.0]), 0.0)
              for f in frags_2}
        m = evaluate_reconstruction(gt, gt)
        assert m.direct_comparison >= 0.9

    def test_perfect_reconstruction_position_rmse_0(self, frags_2):
        gt = {f.fragment_id: (np.array([float(f.fragment_id) * 100, 0.0]), 0.0)
              for f in frags_2}
        m = evaluate_reconstruction(gt, gt)
        assert m.position_rmse < 1.0

    def test_metrics_neighbor_accuracy_range(self, frags_2):
        gt = {f.fragment_id: (np.array([float(f.fragment_id) * 100, 0.0]), 0.0)
              for f in frags_2}
        m = evaluate_reconstruction(gt, gt)
        assert 0.0 <= m.neighbor_accuracy <= 1.0

    def test_angular_error_perfect_is_zero(self, frags_2):
        gt = {f.fragment_id: (np.array([0.0, 0.0]), 0.0) for f in frags_2}
        m = evaluate_reconstruction(gt, gt)
        assert m.angular_error_deg == pytest.approx(0.0, abs=1e-6)


# ─── TestConfigExtra ──────────────────────────────────────────────────────────

class TestConfigExtra:
    def test_default_config_assembly_method_valid(self):
        cfg = Config.default()
        assert cfg.assembly.method in ("greedy", "sa", "beam", "gamma")

    def test_default_matching_threshold_nonneg(self):
        cfg = Config.default()
        assert cfg.matching.threshold >= 0.0

    def test_config_alpha_override(self):
        cfg = Config.default()
        cfg.apply_overrides(alpha=0.6)
        assert cfg.synthesis.alpha == pytest.approx(0.6)

    def test_config_serialization_roundtrip(self, tmp_path):
        cfg = Config.default()
        cfg.synthesis.alpha = 0.33
        p = tmp_path / "cfg.json"
        cfg.to_json(p)
        loaded = Config.from_file(p)
        assert loaded.synthesis.alpha == pytest.approx(0.33)

    def test_config_sa_iter_override(self):
        cfg = Config.default()
        cfg.apply_overrides(sa_iter=500)
        assert cfg.assembly.sa_iter == 500
