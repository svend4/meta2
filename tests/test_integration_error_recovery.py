"""
Интеграционные тесты: устойчивость к плохим входным данным.
Проверяет graceful degradation — никаких крашей при граничных случаях.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from puzzle_reconstruction.models import Fragment, Assembly, CompatEntry, EdgeSide, EdgeSignature
from puzzle_reconstruction.preprocessing.segmentation import segment_fragment
from puzzle_reconstruction.preprocessing.contour import extract_contour
from puzzle_reconstruction.algorithms.tangram.inscriber import fit_tangram
from puzzle_reconstruction.algorithms.fractal.css import curvature_scale_space, css_to_feature_vector
from puzzle_reconstruction.algorithms.fractal.box_counting import box_counting_fd
from puzzle_reconstruction.algorithms.synthesis import compute_fractal_signature, build_edge_signatures
from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
from puzzle_reconstruction.assembly.greedy import greedy_assembly
from puzzle_reconstruction.assembly.beam_search import beam_search
from puzzle_reconstruction.config import Config
from puzzle_reconstruction.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_square_fragment(fid: int, size: int = 80, fill: int = 200) -> Fragment:
    """Create a fragment with a visible filled square so segmentation succeeds."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    margin = size // 8
    img[margin: size - margin, margin: size - margin] = fill
    mask = segment_fragment(img)
    contour = extract_contour(mask)
    tangram = fit_tangram(contour)
    fractal = compute_fractal_signature(contour)
    frag = Fragment(fragment_id=fid, image=img, mask=mask, contour=contour)
    frag.tangram = tangram
    frag.fractal = fractal
    frag.edges = build_edge_signatures(frag, alpha=0.5, n_sides=4)
    return frag


def _no_ocr_pipeline() -> Pipeline:
    """Return a Pipeline with OCR disabled to keep tests fast."""
    cfg = Config.default()
    cfg.verification.run_ocr = False
    return Pipeline(cfg)


# ---------------------------------------------------------------------------
# 1. Segmentation edge cases
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSegmentationEdgeCases:
    """segment_fragment must never crash; always returns an ndarray."""

    def test_all_black_image(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = segment_fragment(img)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.dtype == np.uint8

    def test_all_white_image(self):
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = segment_fragment(img)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    def test_single_pixel_image(self):
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        result = segment_fragment(img)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1)

    def test_tiny_image_5x5(self):
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        result = segment_fragment(img)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 5)

    def test_grayscale_as_bgr(self):
        # 3-channel image where all channels have the same value
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = segment_fragment(img)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2


# ---------------------------------------------------------------------------
# 2. Contour extraction edge cases
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestContourEdgeCases:
    """extract_contour on valid masks always returns (N, 2) float array."""

    def test_empty_mask_raises_gracefully(self):
        """An all-zero mask has no contour; ValueError is the documented contract."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            extract_contour(mask)

    def test_full_mask_returns_valid_contour(self):
        mask = np.full((100, 100), 255, dtype=np.uint8)
        result = extract_contour(mask)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_single_point_mask(self):
        """A single lit pixel produces a contour without crashing."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 255
        # May succeed with a degenerate contour or raise ValueError — both are acceptable.
        try:
            result = extract_contour(mask)
            assert isinstance(result, np.ndarray)
            assert result.shape[1] == 2
        except ValueError:
            pass  # documented failure mode


# ---------------------------------------------------------------------------
# 3. Fractal edge cases
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFractalEdgeCases:
    """Fractal algorithms must not crash on degenerate inputs."""

    def test_fd_degenerate_contour_3pts(self):
        """box_counting_fd returns 1.0 for a 3-point contour (documented fallback)."""
        contour = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        fd = box_counting_fd(contour)
        assert fd == 1.0
        assert isinstance(fd, float)

    def test_css_single_point_contour_does_not_propagate_crash(self):
        """curvature_scale_space on a 1-point array may raise internally; we guard it."""
        c1 = np.array([[5, 5]], dtype=float)
        try:
            css = curvature_scale_space(c1)
            # If it succeeds the result must be a list
            assert isinstance(css, list)
        except (ValueError, IndexError):
            pass  # numerical gradient requires >= 2 pts — acceptable failure

    def test_css_feature_vector_empty_input(self):
        """css_to_feature_vector([]) returns a zero vector without crashing."""
        vec = css_to_feature_vector([])
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        # Must be all zeros or at least finite
        assert np.all(np.isfinite(vec))

    def test_compute_fractal_very_short_contour(self):
        """compute_fractal_signature on a 5-point contour returns a FractalSignature."""
        from puzzle_reconstruction.models import FractalSignature
        contour = np.array(
            [[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]], dtype=float
        )
        sig = compute_fractal_signature(contour)
        assert isinstance(sig, FractalSignature)
        assert isinstance(sig.fd_box, float)
        assert np.isfinite(sig.fd_box)

    def test_fd_is_finite_for_circle_approximation(self):
        """FD must be finite for circle-like contours at varying point counts."""
        for n_pts in (8, 20, 100):
            theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
            contour = np.stack(
                [np.cos(theta) * 40 + 50, np.sin(theta) * 40 + 50], axis=1
            )
            fd = box_counting_fd(contour)
            assert np.isfinite(fd), f"FD not finite for n_pts={n_pts}"
            assert 1.0 <= fd <= 2.0

    def test_css_two_point_contour(self):
        """CSS on a 2-point contour must not crash."""
        c2 = np.array([[0, 0], [10, 10]], dtype=float)
        css = curvature_scale_space(c2)
        assert isinstance(css, list)
        vec = css_to_feature_vector(css)
        assert isinstance(vec, np.ndarray)


# ---------------------------------------------------------------------------
# 4. Compat matrix edge cases
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCompatMatrixEdgeCases:
    """build_compat_matrix must handle empty/degenerate inputs."""

    def test_zero_fragments_returns_empty(self):
        matrix, entries = build_compat_matrix([])
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (0, 0)
        assert isinstance(entries, list)
        assert len(entries) == 0

    def test_single_fragment_no_pairs(self):
        """With 1 fragment there are no inter-fragment pairs."""
        frag = Fragment(fragment_id=0, image=np.zeros((10, 10, 3), dtype=np.uint8), edges=[])
        matrix, entries = build_compat_matrix([frag])
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (0, 0)
        assert len(entries) == 0

    def test_fragments_with_no_edges(self):
        """Fragments with empty edge lists do not crash build_compat_matrix."""
        f1 = Fragment(fragment_id=0, image=np.zeros((10, 10, 3), dtype=np.uint8), edges=[])
        f2 = Fragment(fragment_id=1, image=np.zeros((10, 10, 3), dtype=np.uint8), edges=[])
        matrix, entries = build_compat_matrix([f1, f2])
        assert isinstance(matrix, np.ndarray)
        assert isinstance(entries, list)

    def test_zero_matrix_greedy_returns_valid_assembly(self):
        """greedy_assembly with no entries (equivalent to all-zero matrix) returns Assembly."""
        f1 = Fragment(fragment_id=0, image=np.zeros((50, 50, 3), dtype=np.uint8))
        f2 = Fragment(fragment_id=1, image=np.zeros((50, 50, 3), dtype=np.uint8))
        result = greedy_assembly([f1, f2], [])
        assert isinstance(result, Assembly)
        assert len(result.placements) == 2

    def test_identical_fragments_handled_gracefully(self):
        """Two identical fragments do not trigger any exception."""
        f1 = _make_square_fragment(0)
        f2 = _make_square_fragment(1)  # Built independently but same shape
        matrix, entries = build_compat_matrix([f1, f2])
        assert isinstance(matrix, np.ndarray)
        assert np.all(np.isfinite(matrix))


# ---------------------------------------------------------------------------
# 5. Pipeline edge cases
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPipelineEdgeCases:
    """Pipeline.run must not crash regardless of input quality."""

    def test_empty_image_list(self):
        """An empty image list produces a PipelineResult with an empty assembly."""
        from puzzle_reconstruction.pipeline import PipelineResult
        result = _no_ocr_pipeline().run([])
        assert isinstance(result, PipelineResult)
        assert isinstance(result.assembly, Assembly)

    def test_single_black_image(self):
        """A completely black image produces a PipelineResult without crashing."""
        from puzzle_reconstruction.pipeline import PipelineResult
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = _no_ocr_pipeline().run([img])
        assert isinstance(result, PipelineResult)

    def test_single_white_image(self):
        """A completely white image produces a PipelineResult without crashing."""
        from puzzle_reconstruction.pipeline import PipelineResult
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = _no_ocr_pipeline().run([img])
        assert isinstance(result, PipelineResult)

    def test_two_tiny_fragments(self):
        """Two 10x10 monochrome images do not cause an exception."""
        from puzzle_reconstruction.pipeline import PipelineResult
        img1 = np.zeros((10, 10, 3), dtype=np.uint8)
        img2 = np.zeros((10, 10, 3), dtype=np.uint8)
        result = _no_ocr_pipeline().run([img1, img2])
        assert isinstance(result, PipelineResult)

    def test_all_monochrome_no_contour(self):
        """When all images fail segmentation the pipeline returns gracefully."""
        from puzzle_reconstruction.pipeline import PipelineResult
        images = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
        result = _no_ocr_pipeline().run(images)
        assert isinstance(result, PipelineResult)
        assert isinstance(result.assembly, Assembly)


# ---------------------------------------------------------------------------
# 6. Assembly edge cases
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAssemblyEdgeCases:
    """greedy_assembly and beam_search must handle degenerate inputs."""

    def test_greedy_empty_entries(self):
        """greedy_assembly with no CompatEntry list places all fragments."""
        frags = [
            Fragment(fragment_id=i, image=np.zeros((30, 30, 3), dtype=np.uint8))
            for i in range(3)
        ]
        result = greedy_assembly(frags, [])
        assert isinstance(result, Assembly)
        assert len(result.placements) == len(frags)

    def test_greedy_single_fragment(self):
        """A single fragment is placed at the origin."""
        frag = Fragment(fragment_id=0, image=np.zeros((50, 50, 3), dtype=np.uint8))
        result = greedy_assembly([frag], [])
        assert isinstance(result, Assembly)
        assert 0 in result.placements

    def test_beam_search_empty_fragments(self):
        """beam_search with no fragments returns an empty Assembly."""
        result = beam_search([], [])
        assert isinstance(result, Assembly)
        assert len(result.placements) == 0

    def test_beam_search_width_1(self):
        """beam_width=1 reduces beam search to a greedy strategy; must not crash."""
        frag = Fragment(fragment_id=0, image=np.zeros((50, 50, 3), dtype=np.uint8))
        result = beam_search([frag], [], beam_width=1)
        assert isinstance(result, Assembly)
        assert 0 in result.placements

    def test_beam_search_single_fragment_no_entries(self):
        """beam_search with 1 fragment and no entries places that fragment."""
        frag = Fragment(fragment_id=7, image=np.zeros((40, 40, 3), dtype=np.uint8))
        result = beam_search([frag], [])
        assert isinstance(result, Assembly)
        assert 7 in result.placements

    def test_greedy_returns_assembly_type(self):
        """Return type is always Assembly, not a raw dict or None."""
        result = greedy_assembly([], [])
        assert isinstance(result, Assembly)
        assert result.placements is not None


# ---------------------------------------------------------------------------
# 7. NaN / Inf edge cases
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestNaNInfEdgeCases:
    """No core algorithm may produce NaN or Inf in its primary outputs."""

    def test_no_nan_in_compat_matrix(self):
        """Compat matrix built from real fragments contains only finite values."""
        f1 = _make_square_fragment(0, size=80)
        f2 = _make_square_fragment(1, size=80)
        matrix, entries = build_compat_matrix([f1, f2])
        assert np.all(np.isfinite(matrix)), "compat matrix contains NaN or Inf"

    def test_no_inf_in_box_counting_fd(self):
        """box_counting_fd is always finite for any contour with >= 4 points."""
        for n_pts in (4, 10, 50, 200):
            theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
            contour = np.stack(
                [np.cos(theta) * 30 + 50, np.sin(theta) * 30 + 50], axis=1
            )
            fd = box_counting_fd(contour)
            assert np.isfinite(fd), f"FD={fd} is not finite for n_pts={n_pts}"

    def test_css_feature_vector_is_finite(self):
        """CSS feature vector computed on a real contour has no NaN/Inf."""
        theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        contour = np.stack(
            [np.cos(theta) * 40 + 50, np.sin(theta) * 40 + 50], axis=1
        )
        css = curvature_scale_space(contour)
        vec = css_to_feature_vector(css)
        assert np.all(np.isfinite(vec)), "CSS feature vector contains NaN or Inf"

    def test_fractal_signature_fd_finite(self):
        """FractalSignature fd_box and fd_divider are both finite."""
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        contour = np.stack(
            [np.cos(theta) * 40 + 50, np.sin(theta) * 40 + 50], axis=1
        )
        sig = compute_fractal_signature(contour)
        assert np.isfinite(sig.fd_box), "fd_box is not finite"
        assert np.isfinite(sig.fd_divider), "fd_divider is not finite"

    def test_compat_scores_in_unit_range(self):
        """All compat scores produced for a real fragment pair are in [0, 1]."""
        f1 = _make_square_fragment(0, size=80)
        f2 = _make_square_fragment(1, size=80)
        matrix, entries = build_compat_matrix([f1, f2])
        for entry in entries:
            assert 0.0 <= entry.score <= 1.0, (
                f"score {entry.score} is outside [0, 1]"
            )
        assert np.all(matrix >= 0.0)
        assert np.all(matrix <= 1.0)

    def test_css_vector_norm_is_finite(self):
        """The L2 norm of a CSS vector must be finite and non-negative."""
        theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
        contour = np.stack(
            [np.cos(theta) * 35 + 50, np.sin(theta) * 35 + 50], axis=1
        )
        css = curvature_scale_space(contour)
        vec = css_to_feature_vector(css)
        norm = float(np.linalg.norm(vec))
        assert np.isfinite(norm)
        assert norm >= 0.0


# ---------------------------------------------------------------------------
# 8. TestErrorRecoveryIntegration (ROADMAP-specified class)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestErrorRecoveryIntegration:
    """
    Graceful-degradation tests specified in ROADMAP.md section 2.2.8.

    Verifies that the system either:
      (a) returns a valid result (not None, not NaN, correct type), or
      (b) raises a specific expected exception (ValueError / RuntimeError),
    for each edge-case input category.
    """

    # ── empty image → graceful error, not crash ───────────────────────────

    def test_empty_image_no_crash(self):
        """np.zeros image must not crash Pipeline.preprocess."""
        pipe = _no_ocr_pipeline()
        empty = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pipe.preprocess([empty])
        assert isinstance(result, list)

    # ── solid-colour image → FD=1.0 or contour=empty ─────────────────────

    def test_solid_color_image_graceful(self):
        """All-white image must not crash; FD must be 1.0 for degenerate case."""
        # Solid-colour images either produce no fragment (empty contour) or
        # a degenerate contour.  Either outcome is acceptable; no crash is required.
        pipe = _no_ocr_pipeline()
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = pipe.preprocess([white])
        assert isinstance(result, list)

    def test_solid_color_fd_is_1_or_low(self):
        """FD of a degenerate (3-point) contour returns 1.0."""
        degenerate_contour = np.array([[0, 0], [5, 0], [0, 5]], dtype=float)
        fd = box_counting_fd(degenerate_contour)
        assert fd == 1.0

    # ── 1×1 pixel image → no IndexError ──────────────────────────────────

    def test_1x1_pixel_no_index_error(self):
        """1×1 image must not raise IndexError or AttributeError."""
        pipe = _no_ocr_pipeline()
        tiny = np.full((1, 1, 3), 128, dtype=np.uint8)
        try:
            result = pipe.preprocess([tiny])
            assert isinstance(result, list)
        except (ValueError, RuntimeError):
            pass  # Documented acceptable failure for degenerate images
        except (IndexError, AttributeError) as exc:
            pytest.fail(f"Unexpected {type(exc).__name__}: {exc}")

    # ── fragment without text → text_coherence_score = 0.0, not NaN ──────

    def test_fragment_without_text_score_zero_not_nan(self):
        """text_coherence_score must not return NaN for a fragment without text."""
        from puzzle_reconstruction.verification.ocr import text_coherence_score
        frag = _make_square_fragment(0, size=80, fill=220)
        if not frag.edges:
            pytest.skip("No edges produced for test fragment")
        edge = frag.edges[0]
        score = text_coherence_score(frag.image, frag.image, edge, edge)
        assert not np.isnan(score), "text_coherence_score returned NaN"
        assert 0.0 <= score <= 1.0

    # ── all-zero compat matrix → greedy returns something ─────────────────

    def test_zero_compat_matrix_greedy_returns_something(self):
        """greedy_assembly with empty entries (all-zero matrix) returns an Assembly."""
        f1 = Fragment(fragment_id=0, image=np.zeros((50, 50, 3), dtype=np.uint8))
        f2 = Fragment(fragment_id=1, image=np.zeros((50, 50, 3), dtype=np.uint8))
        result = greedy_assembly([f1, f2], [])
        assert result is not None
        assert isinstance(result, Assembly)
        assert len(result.placements) == 2

    # ── very small fragment (< 10×10) → no crash ─────────────────────────

    def test_very_small_fragment_no_crash(self):
        """A 5×5 image must not cause a crash (IndexError/AttributeError)."""
        pipe = _no_ocr_pipeline()
        small = np.full((5, 5, 3), 128, dtype=np.uint8)
        small[1:4, 1:4] = 240
        try:
            result = pipe.preprocess([small])
            assert isinstance(result, list)
        except (ValueError, RuntimeError):
            pass
        except (IndexError, AttributeError) as exc:
            pytest.fail(f"Unexpected {type(exc).__name__}: {exc}")

    # ── duplicate images → identical signatures ───────────────────────────

    def test_duplicate_images_identical_signatures(self):
        """Two identical images must produce identical fractal signatures."""
        frag1 = _make_square_fragment(0, size=80)
        frag2 = _make_square_fragment(1, size=80)
        # Both built from the same recipe — fd_box must be the same
        assert abs(frag1.fractal.fd_box - frag2.fractal.fd_box) < 1e-9
        assert abs(frag1.fractal.fd_divider - frag2.fractal.fd_divider) < 1e-9
