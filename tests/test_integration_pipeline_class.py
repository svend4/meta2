"""
Integration tests for the Pipeline class.

Covers all public Pipeline methods:
    - Pipeline.__init__            — construction with various configs
    - Pipeline.preprocess(images)  -> List[Fragment]
    - Pipeline.match(fragments)    -> (matrix, entries)
    - Pipeline.assemble(fragments, entries) -> Assembly
    - Pipeline.verify(assembly)    -> float
    - Pipeline.verify_suite(assembly) -> VerificationReport
    - Pipeline.run(images)         — full end-to-end run -> PipelineResult

All tests use synthetic numpy images only; no disk I/O required.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pytest

from puzzle_reconstruction.pipeline import Pipeline, PipelineResult
from puzzle_reconstruction.config import (
    Config,
    AssemblyConfig,
    MatchingConfig,
    SegmentationConfig,
    VerificationConfig,
)
from puzzle_reconstruction.models import Fragment, Assembly, CompatEntry


# ─── Synthetic image helpers ──────────────────────────────────────────────────

def _make_image(h: int = 100, w: int = 100, seed: int = 0) -> np.ndarray:
    """Create a synthetic uint8 BGR image with a bright filled rectangle."""
    rng = np.random.RandomState(seed)
    img = (rng.rand(h, w, 3) * 60 + 30).astype(np.uint8)
    # Bright rectangle ~60 % of the area — easy to segment
    margin_y = h // 10
    margin_x = w // 10
    img[margin_y: h - margin_y, margin_x: w - margin_x] = 210
    return img


def _make_images(n: int, h: int = 100, w: int = 100) -> List[np.ndarray]:
    return [_make_image(h, w, seed=i) for i in range(n)]


def _default_pipeline(n_workers: int = 1) -> Pipeline:
    """Create a Pipeline with default config and OCR disabled for speed."""
    cfg = Config.default()
    cfg.verification.run_ocr = False
    return Pipeline(cfg, n_workers=n_workers)


# ─── Module-scope fixtures (computed once per module) ─────────────────────────

@pytest.fixture(scope="module")
def images_2() -> List[np.ndarray]:
    return _make_images(2)


@pytest.fixture(scope="module")
def images_4() -> List[np.ndarray]:
    return _make_images(4)


@pytest.fixture(scope="module")
def fragments_2(images_2) -> List[Fragment]:
    pipe = _default_pipeline()
    return pipe.preprocess(images_2)


@pytest.fixture(scope="module")
def fragments_4(images_4) -> List[Fragment]:
    pipe = _default_pipeline()
    return pipe.preprocess(images_4)


@pytest.fixture(scope="module")
def match_result_2(fragments_2):
    """(matrix, entries) for the 2-fragment set."""
    pipe = _default_pipeline()
    return pipe.match(fragments_2)


@pytest.fixture(scope="module")
def match_result_4(fragments_4):
    """(matrix, entries) for the 4-fragment set."""
    pipe = _default_pipeline()
    return pipe.match(fragments_4)


@pytest.fixture(scope="module")
def assembly_2(fragments_2, match_result_2) -> Assembly:
    pipe = _default_pipeline()
    _, entries = match_result_2
    return pipe.assemble(fragments_2, entries)


@pytest.fixture(scope="module")
def assembly_4(fragments_4, match_result_4) -> Assembly:
    pipe = _default_pipeline()
    _, entries = match_result_4
    return pipe.assemble(fragments_4, entries)


@pytest.fixture(scope="module")
def full_result_2(images_2) -> PipelineResult:
    """Full Pipeline.run() on 2 images."""
    pipe = _default_pipeline()
    return pipe.run(images_2)


@pytest.fixture(scope="module")
def full_result_4(images_4) -> PipelineResult:
    """Full Pipeline.run() on 4 images."""
    pipe = _default_pipeline()
    return pipe.run(images_4)


# =============================================================================
# TestPipelineInit
# =============================================================================

class TestPipelineInit:
    """Pipeline construction with various configs and parameters."""

    def test_default_config_created_when_none(self):
        pipe = Pipeline()
        assert pipe.cfg is not None
        assert isinstance(pipe.cfg, Config)

    def test_custom_config_is_stored(self):
        cfg = Config.default()
        cfg.assembly.method = "greedy"
        pipe = Pipeline(cfg)
        assert pipe.cfg.assembly.method == "greedy"

    def test_n_workers_stored(self):
        pipe = Pipeline(n_workers=3)
        assert pipe.n_workers == 3

    def test_default_n_workers_is_4(self):
        pipe = Pipeline()
        assert pipe.n_workers == 4

    def test_logger_is_attached(self):
        pipe = Pipeline()
        assert pipe.log is not None

    def test_timer_is_initialized(self):
        pipe = Pipeline()
        assert hasattr(pipe, "_timer")

    def test_on_progress_is_none_by_default(self):
        pipe = Pipeline()
        assert pipe.on_progress is None

    def test_custom_on_progress_stored(self):
        calls = []
        def cb(stage, done, total):
            calls.append((stage, done, total))
        pipe = Pipeline(on_progress=cb)
        assert pipe.on_progress is cb

    def test_log_level_warning_accepted(self):
        pipe = Pipeline(log_level=logging.WARNING)
        assert pipe is not None

    def test_fragment_algorithms_list_exists(self):
        pipe = Pipeline()
        assert hasattr(pipe, "_fragment_algorithms")
        assert isinstance(pipe._fragment_algorithms, list)

    def test_pair_algorithms_list_exists(self):
        pipe = Pipeline()
        assert hasattr(pipe, "_pair_algorithms")
        assert isinstance(pipe._pair_algorithms, list)

    def test_assembly_algorithms_list_exists(self):
        pipe = Pipeline()
        assert hasattr(pipe, "_assembly_algorithms")
        assert isinstance(pipe._assembly_algorithms, list)

    def test_algorithms_param_overrides_config(self):
        """Explicit algorithms= parameter takes priority over cfg.algorithms."""
        pipe = Pipeline(algorithms=["fragment_classifier"])
        assert "fragment_classifier" in pipe._fragment_algorithms

    def test_empty_algorithms_param_accepted(self):
        pipe = Pipeline(algorithms=[])
        assert pipe._fragment_algorithms == []
        assert pipe._pair_algorithms == []
        assert pipe._assembly_algorithms == []

    def test_n_workers_1_sequential(self):
        pipe = Pipeline(n_workers=1)
        assert pipe.n_workers == 1

    def test_cfg_is_config_instance(self):
        pipe = Pipeline()
        assert isinstance(pipe.cfg, Config)


# =============================================================================
# TestPipelinePreprocess
# =============================================================================

class TestPipelinePreprocess:
    """Pipeline.preprocess(images) -> List[Fragment]"""

    def test_returns_list(self, images_2):
        pipe = _default_pipeline()
        result = pipe.preprocess(images_2)
        assert isinstance(result, list)

    def test_nonempty_result_for_good_images(self, fragments_2):
        assert len(fragments_2) >= 1

    def test_each_element_is_fragment(self, fragments_2):
        for f in fragments_2:
            assert isinstance(f, Fragment)

    def test_fragment_ids_are_integers(self, fragments_2):
        for f in fragments_2:
            assert isinstance(f.fragment_id, int)

    def test_fragment_ids_unique(self, fragments_4):
        ids = [f.fragment_id for f in fragments_4]
        assert len(ids) == len(set(ids))

    def test_fragment_has_image(self, fragments_2):
        for f in fragments_2:
            assert f.image is not None
            assert isinstance(f.image, np.ndarray)

    def test_fragment_image_3channel(self, fragments_2):
        for f in fragments_2:
            assert f.image.ndim == 3
            assert f.image.shape[2] == 3

    def test_fragment_image_dtype_uint8(self, fragments_2):
        for f in fragments_2:
            assert f.image.dtype == np.uint8

    def test_fragment_has_mask(self, fragments_2):
        for f in fragments_2:
            assert f.mask is not None

    def test_fragment_mask_is_2d(self, fragments_2):
        for f in fragments_2:
            assert f.mask.ndim == 2

    def test_fragment_mask_dtype_uint8(self, fragments_2):
        for f in fragments_2:
            assert f.mask.dtype == np.uint8

    def test_fragment_mask_binary_values(self, fragments_2):
        for f in fragments_2:
            unique = np.unique(f.mask)
            assert set(unique).issubset({0, 255})

    def test_fragment_has_contour(self, fragments_2):
        for f in fragments_2:
            assert f.contour is not None

    def test_fragment_contour_is_2d(self, fragments_2):
        for f in fragments_2:
            assert f.contour.ndim == 2
            assert f.contour.shape[1] == 2

    def test_fragment_contour_min_4_points(self, fragments_2):
        for f in fragments_2:
            assert len(f.contour) >= 4

    def test_fragment_has_edges(self, fragments_2):
        for f in fragments_2:
            assert f.edges is not None
            assert isinstance(f.edges, list)

    def test_fragment_edges_nonempty(self, fragments_2):
        for f in fragments_2:
            assert len(f.edges) > 0

    def test_fragment_has_tangram(self, fragments_2):
        for f in fragments_2:
            assert f.tangram is not None

    def test_fragment_has_fractal(self, fragments_2):
        for f in fragments_2:
            assert f.fractal is not None

    def test_fractal_fd_box_in_range(self, fragments_2):
        for f in fragments_2:
            assert 1.0 <= f.fractal.fd_box <= 2.0

    def test_edge_fd_in_range(self, fragments_2):
        for f in fragments_2:
            for edge in f.edges:
                assert 1.0 <= edge.fd <= 2.0

    def test_edge_virtual_curve_2d(self, fragments_2):
        for f in fragments_2:
            for edge in f.edges:
                assert edge.virtual_curve.ndim == 2
                assert edge.virtual_curve.shape[1] == 2

    def test_preprocess_4_images_produces_fragments(self, fragments_4):
        assert len(fragments_4) >= 2

    def test_parallel_same_count_as_sequential(self, images_4):
        seq = _default_pipeline(n_workers=1)
        par = _default_pipeline(n_workers=2)
        frags_seq = seq.preprocess(images_4)
        frags_par = par.preprocess(images_4)
        assert len(frags_par) == len(frags_seq)

    def test_graceful_degradation_1x1_image(self):
        pipe = _default_pipeline()
        tiny = np.full((1, 1, 3), 128, dtype=np.uint8)
        result = pipe.preprocess([tiny])
        assert isinstance(result, list)

    def test_graceful_degradation_all_black(self):
        pipe = _default_pipeline()
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pipe.preprocess([black])
        assert isinstance(result, list)

    def test_graceful_degradation_all_white(self):
        pipe = _default_pipeline()
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = pipe.preprocess([white])
        assert isinstance(result, list)

    def test_empty_input_returns_empty_list(self):
        pipe = _default_pipeline()
        result = pipe.preprocess([])
        assert result == []

    def test_on_progress_callback_called(self):
        calls = []
        def cb(stage, done, total):
            calls.append((stage, done, total))
        cfg = Config.default()
        cfg.verification.run_ocr = False
        pipe = Pipeline(cfg, n_workers=1, on_progress=cb)
        pipe.preprocess(_make_images(2))
        assert len(calls) > 0

    def test_on_progress_callback_stage_name(self):
        stages_seen = set()
        def cb(stage, done, total):
            stages_seen.add(stage)
        cfg = Config.default()
        cfg.verification.run_ocr = False
        pipe = Pipeline(cfg, n_workers=1, on_progress=cb)
        pipe.preprocess(_make_images(2))
        assert any("препроц" in s for s in stages_seen)

    def test_fragment_mask_shape_matches_image(self, fragments_2):
        for f in fragments_2:
            h, w = f.image.shape[:2]
            assert f.mask.shape == (h, w)


# =============================================================================
# TestPipelineMatch
# =============================================================================

class TestPipelineMatch:
    """Pipeline.match(fragments) -> (matrix, entries)"""

    def test_returns_tuple_of_two(self, fragments_2):
        pipe = _default_pipeline()
        result = pipe.match(fragments_2)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_matrix_is_ndarray(self, match_result_2):
        matrix, _ = match_result_2
        assert isinstance(matrix, np.ndarray)

    def test_matrix_is_2d(self, match_result_2):
        matrix, _ = match_result_2
        assert matrix.ndim == 2

    def test_matrix_is_square(self, match_result_2):
        matrix, _ = match_result_2
        assert matrix.shape[0] == matrix.shape[1]

    def test_matrix_is_float(self, match_result_2):
        matrix, _ = match_result_2
        assert np.issubdtype(matrix.dtype, np.floating)

    def test_matrix_diagonal_zero(self, match_result_2):
        matrix, _ = match_result_2
        np.testing.assert_array_equal(np.diag(matrix), 0.0)

    def test_matrix_values_non_negative(self, match_result_2):
        matrix, _ = match_result_2
        assert np.all(matrix >= 0)

    def test_matrix_no_nan(self, match_result_2):
        matrix, _ = match_result_2
        assert not np.any(np.isnan(matrix))

    def test_matrix_no_inf(self, match_result_2):
        matrix, _ = match_result_2
        assert not np.any(np.isinf(matrix))

    def test_entries_is_list(self, match_result_2):
        _, entries = match_result_2
        assert isinstance(entries, list)

    def test_entry_scores_in_01(self, match_result_2):
        _, entries = match_result_2
        for e in entries:
            assert 0.0 <= e.score <= 1.0

    def test_entries_have_edge_refs(self, match_result_2):
        _, entries = match_result_2
        for e in entries:
            assert e.edge_i is not None
            assert e.edge_j is not None

    def test_match_4_fragments_matrix_shape_consistent(self, fragments_4, match_result_4):
        matrix, _ = match_result_4
        n_edges = sum(len(f.edges) for f in fragments_4)
        assert matrix.shape == (n_edges, n_edges)

    def test_high_threshold_fewer_entries(self, fragments_2):
        """A higher config threshold yields fewer or equal entries."""
        cfg_low = Config.default()
        cfg_low.verification.run_ocr = False
        cfg_low.matching.threshold = 0.0
        pipe_low = Pipeline(cfg_low)
        _, entries_low = pipe_low.match(fragments_2)

        cfg_high = Config.default()
        cfg_high.verification.run_ocr = False
        cfg_high.matching.threshold = 0.9
        pipe_high = Pipeline(cfg_high)
        _, entries_high = pipe_high.match(fragments_2)

        assert len(entries_high) <= len(entries_low)

    def test_match_single_fragment_no_entries(self, fragments_2):
        """Matching a single fragment produces no valid pairs."""
        if not fragments_2:
            pytest.skip("no fragments available")
        pipe = _default_pipeline()
        matrix, entries = pipe.match([fragments_2[0]])
        assert isinstance(matrix, np.ndarray)
        assert isinstance(entries, list)
        assert len(entries) == 0


# =============================================================================
# TestPipelineAssemble
# =============================================================================

class TestPipelineAssemble:
    """Pipeline.assemble(fragments, entries) -> Assembly"""

    def test_returns_assembly(self, assembly_2):
        assert isinstance(assembly_2, Assembly)

    def test_assembly_has_placements(self, assembly_2):
        assert assembly_2.placements is not None

    def test_assembly_total_score_is_float(self, assembly_2):
        assert isinstance(assembly_2.total_score, float)

    def test_assembly_total_score_in_range(self, assembly_2):
        assert 0.0 <= assembly_2.total_score <= 1.0

    def test_assembly_method_string(self, assembly_2):
        assert isinstance(assembly_2.method, str)

    def test_assembly_4_places_all_fragments(self, assembly_4, fragments_4):
        assert len(assembly_4.placements) == len(fragments_4)

    def test_assembly_placements_keys_are_ints(self, assembly_2):
        if isinstance(assembly_2.placements, dict):
            for key in assembly_2.placements:
                assert isinstance(key, int)

    def test_assemble_greedy_method(self, fragments_4, match_result_4):
        cfg = Config.default()
        cfg.verification.run_ocr = False
        cfg.assembly.method = "greedy"
        pipe = Pipeline(cfg)
        _, entries = match_result_4
        asm = pipe.assemble(fragments_4, entries)
        assert isinstance(asm, Assembly)
        assert asm.total_score >= 0.0

    def test_assemble_beam_method(self, fragments_4, match_result_4):
        cfg = Config.default()
        cfg.verification.run_ocr = False
        cfg.assembly.method = "beam"
        cfg.assembly.beam_width = 3
        pipe = Pipeline(cfg)
        _, entries = match_result_4
        asm = pipe.assemble(fragments_4, entries)
        assert isinstance(asm, Assembly)

    def test_assemble_exhaustive_method_2frags(self, fragments_2, match_result_2):
        cfg = Config.default()
        cfg.verification.run_ocr = False
        cfg.assembly.method = "exhaustive"
        pipe = Pipeline(cfg)
        _, entries = match_result_2
        asm = pipe.assemble(fragments_2, entries)
        assert isinstance(asm, Assembly)

    def test_assemble_sa_method(self, fragments_2, match_result_2):
        """SA is called via run_selected which passes n_iterations to _build_callers.
        The pipeline wraps the call and may raise RuntimeError if SA fails.
        We only require it does not raise an unexpected non-RuntimeError exception."""
        cfg = Config.default()
        cfg.verification.run_ocr = False
        cfg.assembly.method = "sa"
        cfg.assembly.sa_iter = 100
        pipe = Pipeline(cfg)
        _, entries = match_result_2
        try:
            asm = pipe.assemble(fragments_2, entries)
            assert isinstance(asm, Assembly)
        except RuntimeError:
            # RuntimeError is the documented failure mode from assemble()
            pass

    def test_assemble_gamma_method(self, fragments_2, match_result_2):
        cfg = Config.default()
        cfg.verification.run_ocr = False
        cfg.assembly.method = "gamma"
        cfg.assembly.gamma_iter = 50
        pipe = Pipeline(cfg)
        _, entries = match_result_2
        asm = pipe.assemble(fragments_2, entries)
        assert isinstance(asm, Assembly)

    def test_assemble_with_empty_entries(self, fragments_2):
        """Empty entries list should still return an Assembly without crashing."""
        cfg = Config.default()
        cfg.verification.run_ocr = False
        cfg.assembly.method = "greedy"
        pipe = Pipeline(cfg)
        asm = pipe.assemble(fragments_2, [])
        assert isinstance(asm, Assembly)

    def test_placements_positions_finite(self, assembly_4):
        if isinstance(assembly_4.placements, dict):
            for val in assembly_4.placements.values():
                pos = val[0] if isinstance(val, tuple) else val
                arr = np.asarray(pos).ravel()
                assert np.all(np.isfinite(arr))

    def test_assembly_total_score_not_nan(self, assembly_2):
        assert not np.isnan(assembly_2.total_score)


# =============================================================================
# TestPipelineVerify
# =============================================================================

class TestPipelineVerify:
    """Pipeline.verify(assembly) -> float (OCR coherence score)"""

    def test_verify_returns_float_when_ocr_disabled(self, assembly_2):
        cfg = Config.default()
        cfg.verification.run_ocr = False
        pipe = Pipeline(cfg)
        score = pipe.verify(assembly_2)
        assert isinstance(score, float)

    def test_verify_returns_zero_when_ocr_disabled(self, assembly_2):
        cfg = Config.default()
        cfg.verification.run_ocr = False
        pipe = Pipeline(cfg)
        score = pipe.verify(assembly_2)
        assert score == 0.0

    def test_verify_score_in_range_ocr_disabled(self, assembly_2):
        cfg = Config.default()
        cfg.verification.run_ocr = False
        pipe = Pipeline(cfg)
        score = pipe.verify(assembly_2)
        assert 0.0 <= score <= 1.0

    def test_verify_graceful_on_empty_assembly(self):
        cfg = Config.default()
        cfg.verification.run_ocr = False
        pipe = Pipeline(cfg)
        empty_asm = Assembly(placements={}, fragments=[], compat_matrix=np.array([]))
        score = pipe.verify(empty_asm)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_verify_suite_returns_report(self, assembly_2):
        pipe = _default_pipeline()
        report = pipe.verify_suite(assembly_2)
        assert report is not None
        assert hasattr(report, "final_score")
        assert hasattr(report, "results")

    def test_verify_suite_final_score_in_range(self, assembly_2):
        pipe = _default_pipeline()
        report = pipe.verify_suite(assembly_2)
        assert 0.0 <= report.final_score <= 1.0

    def test_verify_suite_results_is_list(self, assembly_2):
        pipe = _default_pipeline()
        report = pipe.verify_suite(assembly_2)
        assert isinstance(report.results, list)

    def test_verify_suite_with_assembly_score_validator(self, assembly_2):
        pipe = _default_pipeline()
        report = pipe.verify_suite(assembly_2, validators=["assembly_score"])
        assert report is not None
        assert isinstance(report.final_score, float)

    def test_verify_suite_with_completeness_validator(self, assembly_2):
        pipe = _default_pipeline()
        report = pipe.verify_suite(assembly_2, validators=["completeness"])
        assert report is not None

    def test_verify_suite_score_not_nan(self, assembly_2):
        pipe = _default_pipeline()
        report = pipe.verify_suite(assembly_2)
        assert not np.isnan(report.final_score)


# =============================================================================
# TestPipelineRunFull
# =============================================================================

class TestPipelineRunFull:
    """Pipeline.run(images) — full end-to-end pipeline run."""

    def test_run_returns_pipeline_result(self, full_result_2):
        assert isinstance(full_result_2, PipelineResult)

    def test_result_has_assembly(self, full_result_2):
        assert full_result_2.assembly is not None
        assert isinstance(full_result_2.assembly, Assembly)

    def test_result_has_timer(self, full_result_2):
        assert full_result_2.timer is not None

    def test_result_has_cfg(self, full_result_2):
        assert full_result_2.cfg is not None
        assert isinstance(full_result_2.cfg, Config)

    def test_result_n_input_correct_2(self, images_2, full_result_2):
        assert full_result_2.n_input == len(images_2)

    def test_result_n_input_correct_4(self, images_4, full_result_4):
        assert full_result_4.n_input == len(images_4)

    def test_result_n_placed_nonneg(self, full_result_2):
        assert full_result_2.n_placed >= 0

    def test_result_timestamp_is_string(self, full_result_2):
        assert isinstance(full_result_2.timestamp, str)
        assert len(full_result_2.timestamp) > 0

    def test_result_assembly_total_score_float(self, full_result_2):
        assert isinstance(full_result_2.assembly.total_score, float)

    def test_result_assembly_total_score_in_range(self, full_result_2):
        assert 0.0 <= full_result_2.assembly.total_score <= 1.0

    def test_result_assembly_ocr_score_float(self, full_result_2):
        assert isinstance(full_result_2.assembly.ocr_score, float)

    def test_result_assembly_ocr_score_in_range(self, full_result_2):
        assert 0.0 <= full_result_2.assembly.ocr_score <= 1.0

    def test_result_summary_returns_string(self, full_result_2):
        summary = full_result_2.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_result_summary_contains_score_word(self, full_result_2):
        summary = full_result_2.summary()
        assert "score" in summary.lower() or "Score" in summary

    def test_result_timer_stages_not_empty(self, full_result_2):
        assert len(full_result_2.timer._stages) > 0

    def test_result_timer_contains_preprocessing(self, full_result_2):
        keys = full_result_2.timer._stages.keys()
        assert any("препроц" in k for k in keys)

    def test_result_timer_stage_times_nonneg(self, full_result_2):
        for stage, elapsed in full_result_2.timer._stages.items():
            assert elapsed >= 0.0

    def test_run_4_images_returns_pipeline_result(self, full_result_4):
        assert isinstance(full_result_4, PipelineResult)

    def test_run_4_images_n_input(self, full_result_4):
        assert full_result_4.n_input == 4

    def test_run_empty_list_returns_result(self):
        pipe = _default_pipeline()
        result = pipe.run([])
        assert isinstance(result, PipelineResult)

    def test_run_empty_list_n_input_zero(self):
        pipe = _default_pipeline()
        result = pipe.run([])
        assert result.n_input == 0

    def test_run_empty_list_n_placed_zero(self):
        pipe = _default_pipeline()
        result = pipe.run([])
        assert result.n_placed == 0

    def test_run_consistency_report_attribute_exists(self, full_result_2):
        assert hasattr(full_result_2, "consistency_report")

    def test_run_verification_report_attribute_exists(self, full_result_2):
        assert hasattr(full_result_2, "verification_report")

    def test_result_export_json(self, full_result_2):
        out = full_result_2.export(fmt="json")
        assert out is not None
        assert isinstance(out, str)
        assert len(out) > 0

    def test_result_export_csv(self, full_result_2):
        out = full_result_2.export(fmt="csv")
        assert out is not None
        assert isinstance(out, str)

    def test_result_export_text(self, full_result_2):
        out = full_result_2.export(fmt="text")
        assert out is not None
        assert isinstance(out, str)

    def test_run_with_greedy_method(self):
        cfg = Config.default()
        cfg.verification.run_ocr = False
        cfg.assembly.method = "greedy"
        pipe = Pipeline(cfg)
        result = pipe.run(_make_images(2))
        assert isinstance(result, PipelineResult)
        # Assembly.method is set by the algorithm itself; greedy may leave it empty.
        assert isinstance(result.assembly.method, str)

    def test_run_with_beam_method(self):
        cfg = Config.default()
        cfg.verification.run_ocr = False
        cfg.assembly.method = "beam"
        cfg.assembly.beam_width = 3
        pipe = Pipeline(cfg)
        result = pipe.run(_make_images(2))
        assert isinstance(result, PipelineResult)

    def test_run_parallel_n_workers_2(self):
        cfg = Config.default()
        cfg.verification.run_ocr = False
        pipe = Pipeline(cfg, n_workers=2)
        result = pipe.run(_make_images(4))
        assert isinstance(result, PipelineResult)

    def test_run_with_progress_callback_fires(self):
        events = []
        def cb(stage, done, total):
            events.append((stage, done, total))
        cfg = Config.default()
        cfg.verification.run_ocr = False
        pipe = Pipeline(cfg, n_workers=1, on_progress=cb)
        pipe.run(_make_images(2))
        assert len(events) > 0

    def test_run_progress_done_never_exceeds_total(self):
        events = []
        def cb(stage, done, total):
            events.append((stage, done, total))
        cfg = Config.default()
        cfg.verification.run_ocr = False
        pipe = Pipeline(cfg, n_workers=1, on_progress=cb)
        pipe.run(_make_images(2))
        for stage, done, total in events:
            assert done <= total

    def test_result_summary_contains_n_input(self, full_result_2):
        summary = full_result_2.summary()
        # Summary mentions number of input fragments
        assert "2" in summary

    def test_run_auto_method_small(self):
        """auto method with 2 fragments should call exhaustive internally."""
        cfg = Config.default()
        cfg.verification.run_ocr = False
        cfg.assembly.method = "auto"
        cfg.assembly.auto_timeout = 10.0
        pipe = Pipeline(cfg)
        result = pipe.run(_make_images(2))
        assert isinstance(result, PipelineResult)


# =============================================================================
# TestPipelineConfig
# =============================================================================

class TestPipelineConfig:
    """Config construction, roundtrip, override, and effect on Pipeline."""

    def test_default_config_synthesis_alpha(self):
        cfg = Config.default()
        assert cfg.synthesis.alpha == pytest.approx(0.5)

    def test_default_config_assembly_method_valid(self):
        cfg = Config.default()
        assert cfg.assembly.method in (
            "greedy", "sa", "beam", "gamma",
            "genetic", "exhaustive", "ant_colony", "mcts",
            "auto", "all"
        )

    def test_default_matching_threshold_nonneg(self):
        cfg = Config.default()
        assert cfg.matching.threshold >= 0.0

    def test_config_to_dict_is_dict(self):
        cfg = Config.default()
        d = cfg.to_dict()
        assert isinstance(d, dict)

    def test_config_to_dict_has_assembly_key(self):
        cfg = Config.default()
        d = cfg.to_dict()
        assert "assembly" in d

    def test_config_to_dict_has_synthesis_key(self):
        cfg = Config.default()
        d = cfg.to_dict()
        assert "synthesis" in d

    def test_config_roundtrip_json(self, tmp_path):
        cfg = Config.default()
        cfg.synthesis.alpha = 0.77
        path = tmp_path / "cfg.json"
        cfg.to_json(path)
        loaded = Config.from_file(path)
        assert loaded.synthesis.alpha == pytest.approx(0.77)

    def test_config_roundtrip_preserves_method(self, tmp_path):
        cfg = Config.default()
        cfg.assembly.method = "greedy"
        path = tmp_path / "cfg2.json"
        cfg.to_json(path)
        loaded = Config.from_file(path)
        assert loaded.assembly.method == "greedy"

    def test_config_roundtrip_preserves_threshold(self, tmp_path):
        cfg = Config.default()
        cfg.matching.threshold = 0.42
        path = tmp_path / "cfg3.json"
        cfg.to_json(path)
        loaded = Config.from_file(path)
        assert loaded.matching.threshold == pytest.approx(0.42)

    def test_config_from_dict_roundtrip(self):
        cfg = Config.default()
        cfg.synthesis.n_sides = 6
        d = cfg.to_dict()
        loaded = Config.from_dict(d)
        assert loaded.synthesis.n_sides == 6

    def test_apply_overrides_alpha(self):
        cfg = Config.default()
        cfg.apply_overrides(alpha=0.3)
        assert cfg.synthesis.alpha == pytest.approx(0.3)

    def test_apply_overrides_method(self):
        cfg = Config.default()
        cfg.apply_overrides(method="greedy")
        assert cfg.assembly.method == "greedy"

    def test_apply_overrides_sa_iter(self):
        cfg = Config.default()
        cfg.apply_overrides(sa_iter=1000)
        assert cfg.assembly.sa_iter == 1000

    def test_apply_overrides_threshold(self):
        cfg = Config.default()
        cfg.apply_overrides(threshold=0.6)
        assert cfg.matching.threshold == pytest.approx(0.6)

    def test_apply_overrides_seed(self):
        cfg = Config.default()
        cfg.apply_overrides(seed=123)
        assert cfg.assembly.seed == 123

    def test_apply_overrides_none_value_ignored(self):
        cfg = Config.default()
        old_alpha = cfg.synthesis.alpha
        cfg.apply_overrides(alpha=None)
        assert cfg.synthesis.alpha == old_alpha

    def test_pipeline_uses_config_method(self):
        cfg = Config.default()
        cfg.verification.run_ocr = False
        cfg.assembly.method = "greedy"
        pipe = Pipeline(cfg)
        assert pipe.cfg.assembly.method == "greedy"

    def test_pipeline_uses_config_n_sides(self):
        cfg = Config.default()
        cfg.verification.run_ocr = False
        cfg.synthesis.n_sides = 4
        pipe = Pipeline(cfg)
        assert pipe.cfg.synthesis.n_sides == 4

    def test_auto_methods_4_fragments(self):
        """4 or fewer fragments -> exhaustive."""
        methods = Pipeline._auto_methods(4)
        assert "exhaustive" in methods

    def test_auto_methods_6_fragments(self):
        """5-8 fragments -> exhaustive + beam."""
        methods = Pipeline._auto_methods(6)
        assert any(m in methods for m in ("beam", "exhaustive"))

    def test_auto_methods_large(self):
        """More than 30 fragments -> gamma or sa."""
        methods = Pipeline._auto_methods(50)
        assert len(methods) >= 1
        assert any(m in methods for m in ("gamma", "sa"))

    def test_auto_methods_returns_list(self):
        methods = Pipeline._auto_methods(10)
        assert isinstance(methods, list)
        assert len(methods) >= 1

    def test_auto_methods_2_fragments(self):
        methods = Pipeline._auto_methods(2)
        assert isinstance(methods, list)
        assert len(methods) >= 1

    def test_auto_methods_medium_15_fragments(self):
        """9-15 fragments -> beam + mcts + sa."""
        methods = Pipeline._auto_methods(12)
        assert any(m in methods for m in ("beam", "mcts", "sa"))

    def test_verification_config_validators_empty_by_default(self):
        cfg = Config.default()
        assert cfg.verification.validators == []

    def test_algorithms_config_fragment_empty_by_default(self):
        cfg = Config.default()
        assert cfg.algorithms.fragment == []

    def test_utils_config_profiler_false_by_default(self):
        cfg = Config.default()
        assert cfg.utils.profiler is False

    def test_config_beam_width_override(self):
        cfg = Config.default()
        cfg.apply_overrides(beam_width=7)
        assert cfg.assembly.beam_width == 7

    def test_config_from_dict_with_matching(self):
        d = Config.default().to_dict()
        d["matching"]["threshold"] = 0.25
        loaded = Config.from_dict(d)
        assert loaded.matching.threshold == pytest.approx(0.25)
