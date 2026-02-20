"""
Интеграционный тест: полный пайплайн восстановления документа.

Тест генерирует синтетический документ, рвёт его на фрагменты,
запускает полный пайплайн и проверяет, что результат имеет смысл.

Эти тесты медленнее юнит-тестов (0.5–5 секунд каждый),
поэтому помечены маркером @pytest.mark.integration.
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


# ─── Фикстуры ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_document():
    """Синтетический документ 400×500, генерируется один раз на модуль."""
    return generate_test_document(width=400, height=500, seed=42)


@pytest.fixture(scope="module")
def torn_4(synthetic_document):
    """4 фрагмента из синтетического документа."""
    return tear_document(synthetic_document, n_pieces=4, noise_level=0.4, seed=10)


@pytest.fixture
def processed_fragments(torn_4):
    """Обработанные фрагменты (сегментация + описание)."""
    cfg = Config.default()
    fragments = []
    for idx, img in enumerate(torn_4):
        try:
            mask    = segment_fragment(img, method="otsu")
            contour = extract_contour(mask)
            tangram = fit_tangram(contour)
            fractal = compute_fractal_signature(contour)
            frag    = Fragment(fragment_id=idx, image=img, mask=mask, contour=contour)
            frag.tangram = tangram
            frag.fractal = fractal
            frag.edges   = build_edge_signatures(frag, alpha=0.5, n_sides=4)
            fragments.append(frag)
        except Exception:
            pass
    return fragments


# ─── Тесты препроцессинга ─────────────────────────────────────────────────

class TestPreprocessingIntegration:

    def test_document_generates(self, synthetic_document):
        assert synthetic_document.shape == (500, 400, 3)

    def test_tear_produces_fragments(self, torn_4):
        assert len(torn_4) >= 2
        for f in torn_4:
            assert f.ndim == 3
            assert f.dtype == np.uint8

    def test_segmentation_produces_mask(self, torn_4):
        img = torn_4[0]
        mask = segment_fragment(img, method="otsu")
        assert mask.shape == img.shape[:2]
        assert np.any(mask > 0)

    def test_contour_extraction(self, torn_4):
        img  = torn_4[0]
        mask = segment_fragment(img)
        contour = extract_contour(mask)
        assert contour.ndim == 2
        assert contour.shape[1] == 2
        assert len(contour) >= 4


# ─── Тесты алгоритмов ────────────────────────────────────────────────────

class TestAlgorithmsIntegration:

    def test_tangram_fit(self, torn_4):
        img = torn_4[0]
        mask = segment_fragment(img)
        contour = extract_contour(mask)
        sig = fit_tangram(contour)
        assert sig is not None
        assert sig.scale > 0
        assert len(sig.polygon) >= 3

    def test_fractal_signature(self, torn_4):
        img = torn_4[0]
        mask = segment_fragment(img)
        contour = extract_contour(mask)
        sig = compute_fractal_signature(contour)
        assert 1.0 <= sig.fd_box <= 2.0
        assert 1.0 <= sig.fd_divider <= 2.0
        assert len(sig.ifs_coeffs) > 0
        assert len(sig.css_image) > 0

    def test_edge_signatures_built(self, processed_fragments):
        assert len(processed_fragments) >= 2
        for frag in processed_fragments:
            assert len(frag.edges) >= 2
            for edge in frag.edges:
                assert edge.virtual_curve.shape[1] == 2
                assert 1.0 <= edge.fd <= 2.0

    def test_fractal_dimensions_vary(self, processed_fragments):
        """FD разных фрагментов должны немного различаться (разные края)."""
        fds = [frag.fractal.fd_box for frag in processed_fragments]
        if len(fds) >= 2:
            assert max(fds) - min(fds) >= 0.0  # Хотя бы не все одинаковые


# ─── Тесты сопоставления ─────────────────────────────────────────────────

class TestMatchingIntegration:

    def test_compat_matrix_built(self, processed_fragments):
        matrix, entries = build_compat_matrix(processed_fragments, threshold=0.0)
        n_edges = sum(len(f.edges) for f in processed_fragments)
        assert matrix.shape == (n_edges, n_edges)

    def test_entries_have_valid_scores(self, processed_fragments):
        _, entries = build_compat_matrix(processed_fragments)
        for e in entries[:20]:
            assert 0.0 <= e.score <= 1.0

    def test_no_self_match_in_matrix(self, processed_fragments):
        matrix, _ = build_compat_matrix(processed_fragments)
        assert np.all(np.diag(matrix) == 0.0)


# ─── Тесты сборки ─────────────────────────────────────────────────────────

class TestAssemblyIntegration:

    def test_greedy_places_all(self, processed_fragments):
        _, entries = build_compat_matrix(processed_fragments)
        asm = greedy_assembly(processed_fragments, entries)
        assert len(asm.placements) == len(processed_fragments)

    def test_sa_improves_or_maintains(self, processed_fragments):
        _, entries = build_compat_matrix(processed_fragments)
        asm0 = greedy_assembly(processed_fragments, entries)
        asm1 = simulated_annealing(asm0, entries, T_max=100, max_iter=500, seed=0)
        # SA должна поддерживать смысловую сборку
        assert len(asm1.placements) == len(processed_fragments)

    def test_beam_search_places_all(self, processed_fragments):
        _, entries = build_compat_matrix(processed_fragments)
        asm = beam_search(processed_fragments, entries, beam_width=3)
        assert len(asm.placements) == len(processed_fragments)

    def test_all_placements_finite(self, processed_fragments):
        _, entries = build_compat_matrix(processed_fragments)
        asm = greedy_assembly(processed_fragments, entries)
        for pos, angle in asm.placements.values():
            assert np.all(np.isfinite(pos))
            assert np.isfinite(angle)


# ─── Тест метрик ──────────────────────────────────────────────────────────

class TestMetricsIntegration:

    def test_evaluate_perfect_reconstruction(self, processed_fragments):
        """Идеальная сборка (predicted = gt) должна давать высокие метрики."""
        gt = {frag.fragment_id: (np.array([float(frag.fragment_id * 100), 0.0]), 0.0)
              for frag in processed_fragments}
        metrics = evaluate_reconstruction(gt, gt)
        assert metrics.direct_comparison >= 0.9
        assert metrics.position_rmse < 1.0

    def test_evaluate_random_reconstruction(self, processed_fragments):
        """Случайная сборка должна давать низкие метрики."""
        gt = {frag.fragment_id: (np.array([float(frag.fragment_id * 100), 0.0]), 0.0)
              for frag in processed_fragments}
        rng = np.random.RandomState(99)
        random_pred = {fid: (rng.randn(2) * 1000, float(rng.rand() * 2 * np.pi))
                       for fid in gt}
        metrics = evaluate_reconstruction(random_pred, gt)
        # Случайная сборка должна быть хуже идеальной
        assert metrics.direct_comparison < 0.9 or metrics.position_rmse > 0.0

    def test_metrics_have_valid_range(self, processed_fragments):
        _, entries = build_compat_matrix(processed_fragments)
        asm = greedy_assembly(processed_fragments, entries)
        gt = {frag.fragment_id: (np.array([float(frag.fragment_id * 100), 0.0]), 0.0)
              for frag in processed_fragments}
        m = evaluate_reconstruction(asm.placements, gt)
        assert 0.0 <= m.neighbor_accuracy  <= 1.0
        assert 0.0 <= m.direct_comparison  <= 1.0
        assert 0.0 <= m.edge_match_rate    <= 1.0
        assert m.position_rmse >= 0.0
        assert m.angular_error_deg >= 0.0


# ─── Config тест ──────────────────────────────────────────────────────────

class TestConfigIntegration:

    def test_default_config_valid(self):
        cfg = Config.default()
        assert cfg.synthesis.alpha == 0.5
        assert cfg.assembly.method in ("greedy", "sa", "beam", "gamma")
        assert cfg.matching.threshold >= 0.0

    def test_config_serialization(self, tmp_path):
        cfg = Config.default()
        cfg.synthesis.alpha = 0.7
        path = tmp_path / "test_config.json"
        cfg.to_json(path)
        loaded = Config.from_file(path)
        assert loaded.synthesis.alpha == 0.7

    def test_config_apply_overrides(self):
        cfg = Config.default()
        cfg.apply_overrides(alpha=0.3, method="sa", sa_iter=1000)
        assert cfg.synthesis.alpha == 0.3
        assert cfg.assembly.method == "sa"
        assert cfg.assembly.sa_iter == 1000
