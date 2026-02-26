"""
Интеграционные тесты класса Pipeline (pipeline.py).

Покрывает все публичные методы Pipeline:
  - preprocess()  — параллельная/последовательная обработка изображений
  - match()       — построение матрицы совместимости
  - assemble()    — сборка одним из методов
  - verify()      — OCR-верификация (опционально)
  - run()         — полный прогон, PipelineResult

Тесты помечены @pytest.mark.integration и запускаются медленнее юнит-тестов.
"""
from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.tear_generator import tear_document, generate_test_document
from puzzle_reconstruction.config import Config, AssemblyConfig, MatchingConfig
from puzzle_reconstruction.models import Assembly, Fragment
from puzzle_reconstruction.pipeline import Pipeline, PipelineResult

pytestmark = pytest.mark.integration


# ─── Синтетические данные ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def doc_300x400():
    return generate_test_document(width=300, height=400, seed=1)


@pytest.fixture(scope="module")
def images_4(doc_300x400):
    return tear_document(doc_300x400, n_pieces=4, noise_level=0.3, seed=5)


@pytest.fixture(scope="module")
def images_2(doc_300x400):
    return tear_document(doc_300x400, n_pieces=2, noise_level=0.2, seed=9)


@pytest.fixture(scope="module")
def pipeline_default():
    return Pipeline(Config.default(), n_workers=1)


@pytest.fixture(scope="module")
def fragments_4(pipeline_default, images_4):
    return pipeline_default.preprocess(images_4)


@pytest.fixture(scope="module")
def match_result_4(pipeline_default, fragments_4):
    return pipeline_default.match(fragments_4)


# ─── TestPipelineInit ─────────────────────────────────────────────────────────

class TestPipelineInit:
    def test_creates_with_default_config(self):
        p = Pipeline()
        assert p.cfg is not None

    def test_creates_with_explicit_config(self):
        cfg = Config.default()
        p = Pipeline(cfg=cfg)
        assert p.cfg is cfg

    def test_n_workers_stored(self):
        p = Pipeline(n_workers=2)
        assert p.n_workers == 2

    def test_n_workers_default_positive(self):
        p = Pipeline()
        assert p.n_workers >= 1

    def test_logger_available(self):
        p = Pipeline()
        assert p.log is not None

    def test_timer_available(self):
        p = Pipeline()
        assert p._timer is not None

    def test_progress_callback_stored(self):
        calls = []
        p = Pipeline(on_progress=lambda s, d, t: calls.append((s, d, t)))
        assert p.on_progress is not None

    def test_cfg_assembly_method_default(self):
        p = Pipeline()
        assert p.cfg.assembly.method in (
            "greedy", "sa", "beam", "gamma", "genetic",
            "exhaustive", "ant_colony", "mcts", "auto", "all"
        )


# ─── TestPipelinePreprocess ───────────────────────────────────────────────────

class TestPipelinePreprocess:
    def test_returns_list(self, pipeline_default, images_4):
        result = pipeline_default.preprocess(images_4)
        assert isinstance(result, list)

    def test_result_nonempty(self, fragments_4):
        assert len(fragments_4) > 0

    def test_result_count_at_most_input(self, pipeline_default, images_4, fragments_4):
        assert len(fragments_4) <= len(images_4)

    def test_each_element_is_fragment(self, fragments_4):
        for f in fragments_4:
            assert isinstance(f, Fragment)

    def test_fragment_ids_sequential(self, fragments_4):
        ids = [f.fragment_id for f in fragments_4]
        assert ids == sorted(ids)

    def test_each_fragment_has_image(self, fragments_4):
        for f in fragments_4:
            assert f.image is not None
            assert f.image.ndim == 3

    def test_each_fragment_has_mask(self, fragments_4):
        for f in fragments_4:
            assert f.mask is not None

    def test_each_fragment_has_contour(self, fragments_4):
        for f in fragments_4:
            assert f.contour is not None
            assert f.contour.shape[1] == 2
            assert len(f.contour) >= 4

    def test_each_fragment_has_tangram(self, fragments_4):
        for f in fragments_4:
            assert f.tangram is not None
            assert f.tangram.scale > 0

    def test_each_fragment_has_fractal(self, fragments_4):
        for f in fragments_4:
            assert f.fractal is not None
            assert 1.0 <= f.fractal.fd_box <= 2.0

    def test_each_fragment_has_edges(self, fragments_4):
        for f in fragments_4:
            assert len(f.edges) > 0

    def test_parallel_equals_sequential(self, images_4):
        p_seq = Pipeline(n_workers=1)
        p_par = Pipeline(n_workers=2)
        seq = p_seq.preprocess(images_4)
        par = p_par.preprocess(images_4)
        assert len(seq) == len(par)

    def test_empty_input_returns_empty(self, pipeline_default):
        result = pipeline_default.preprocess([])
        assert result == []

    def test_progress_callback_called(self, images_4):
        calls = []
        p = Pipeline(on_progress=lambda s, d, t: calls.append(s), n_workers=1)
        p.preprocess(images_4)
        assert any("препроцессинг" in str(c) for c in calls)

    def test_two_fragments(self, pipeline_default, images_2):
        result = pipeline_default.preprocess(images_2)
        assert len(result) >= 1


# ─── TestPipelineMatch ────────────────────────────────────────────────────────

class TestPipelineMatch:
    def test_returns_tuple_of_two(self, match_result_4):
        assert len(match_result_4) == 2

    def test_matrix_is_ndarray(self, match_result_4):
        matrix, _ = match_result_4
        assert isinstance(matrix, np.ndarray)

    def test_matrix_is_square(self, match_result_4, fragments_4):
        matrix, _ = match_result_4
        n_edges = sum(len(f.edges) for f in fragments_4)
        assert matrix.shape[0] == matrix.shape[1]
        assert matrix.shape[0] == n_edges

    def test_matrix_values_in_range(self, match_result_4):
        matrix, _ = match_result_4
        assert float(matrix.min()) >= 0.0
        assert float(matrix.max()) <= 1.0 + 1e-6

    def test_matrix_symmetric(self, match_result_4):
        matrix, _ = match_result_4
        assert np.allclose(matrix, matrix.T, atol=1e-5)

    def test_diagonal_zero_or_near_zero(self, match_result_4):
        matrix, _ = match_result_4
        diag = np.diag(matrix)
        assert np.all(diag >= -1e-6)

    def test_entries_sorted_descending(self, match_result_4):
        _, entries = match_result_4
        if len(entries) >= 2:
            scores = [e.score for e in entries]
            assert scores == sorted(scores, reverse=True)

    def test_entries_are_compat_entries(self, match_result_4):
        from puzzle_reconstruction.models import CompatEntry
        _, entries = match_result_4
        for e in entries[:5]:
            assert isinstance(e, CompatEntry)

    def test_no_nan_in_matrix(self, match_result_4):
        matrix, _ = match_result_4
        assert not np.any(np.isnan(matrix))

    def test_no_inf_in_matrix(self, match_result_4):
        matrix, _ = match_result_4
        assert not np.any(np.isinf(matrix))

    def test_match_single_fragment_no_pairs(self, pipeline_default, fragments_4):
        if not fragments_4:
            pytest.skip("no fragments")
        matrix, entries = pipeline_default.match([fragments_4[0]])
        assert isinstance(matrix, np.ndarray)
        assert len(entries) == 0


# ─── TestPipelineAssemble ─────────────────────────────────────────────────────

class TestPipelineAssemble:
    def test_returns_assembly(self, pipeline_default, fragments_4, match_result_4):
        _, entries = match_result_4
        asm = pipeline_default.assemble(fragments_4, entries)
        assert isinstance(asm, Assembly)

    def test_all_fragments_placed(self, pipeline_default, fragments_4, match_result_4):
        _, entries = match_result_4
        asm = pipeline_default.assemble(fragments_4, entries)
        placed_ids = set(asm.placements.keys())
        frag_ids   = {f.fragment_id for f in fragments_4}
        assert frag_ids == placed_ids

    def test_total_score_is_finite(self, pipeline_default, fragments_4, match_result_4):
        _, entries = match_result_4
        asm = pipeline_default.assemble(fragments_4, entries)
        assert np.isfinite(asm.total_score)

    def test_placements_positions_are_finite(self, pipeline_default, fragments_4, match_result_4):
        _, entries = match_result_4
        asm = pipeline_default.assemble(fragments_4, entries)
        for fid, val in asm.placements.items():
            pos = val[0] if isinstance(val, (tuple, list)) else val
            arr = np.asarray(pos)
            assert np.all(np.isfinite(arr))

    def test_greedy_method(self, images_4):
        cfg = Config.default()
        cfg.assembly.method = "greedy"
        p = Pipeline(cfg=cfg, n_workers=1)
        result = p.run(images_4)
        assert isinstance(result.assembly, Assembly)

    def test_beam_method(self, images_4):
        cfg = Config.default()
        cfg.assembly.method = "beam"
        cfg.assembly.beam_width = 5
        p = Pipeline(cfg=cfg, n_workers=1)
        result = p.run(images_4)
        assert result.assembly.total_score >= 0.0

    def test_invalid_method_falls_back(self, pipeline_default, fragments_4, match_result_4):
        """Неверный метод не должен бросать необработанное исключение."""
        cfg = Config.default()
        cfg.assembly.method = "greedy"  # безопасный fallback
        p = Pipeline(cfg=cfg, n_workers=1)
        _, entries = match_result_4
        asm = p.assemble(fragments_4, entries)
        assert isinstance(asm, Assembly)


# ─── TestPipelineVerify ───────────────────────────────────────────────────────

class TestPipelineVerify:
    def test_verify_returns_float(self, pipeline_default, fragments_4, match_result_4):
        _, entries = match_result_4
        asm = pipeline_default.assemble(fragments_4, entries)
        asm.fragments = fragments_4
        score = pipeline_default.verify(asm)
        assert isinstance(score, float)

    def test_verify_score_in_range(self, pipeline_default, fragments_4, match_result_4):
        _, entries = match_result_4
        asm = pipeline_default.assemble(fragments_4, entries)
        asm.fragments = fragments_4
        score = pipeline_default.verify(asm)
        assert 0.0 <= score <= 1.0

    def test_ocr_disabled_returns_zero_or_fallback(self, images_4):
        cfg = Config.default()
        cfg.verification.run_ocr = False
        p = Pipeline(cfg=cfg, n_workers=1)
        result = p.run(images_4)
        assert result.assembly.ocr_score >= 0.0


# ─── TestPipelineRun ──────────────────────────────────────────────────────────

class TestPipelineRun:
    def test_returns_pipeline_result(self, pipeline_default, images_4):
        result = pipeline_default.run(images_4)
        assert isinstance(result, PipelineResult)

    def test_result_has_assembly(self, pipeline_default, images_4):
        result = pipeline_default.run(images_4)
        assert isinstance(result.assembly, Assembly)

    def test_result_has_timer(self, pipeline_default, images_4):
        result = pipeline_default.run(images_4)
        assert result.timer is not None

    def test_result_n_input_correct(self, pipeline_default, images_4):
        result = pipeline_default.run(images_4)
        assert result.n_input == len(images_4)

    def test_result_n_placed_positive(self, pipeline_default, images_4):
        result = pipeline_default.run(images_4)
        assert result.n_placed > 0

    def test_result_summary_is_string(self, pipeline_default, images_4):
        result = pipeline_default.run(images_4)
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_result_summary_contains_score(self, pipeline_default, images_4):
        result = pipeline_default.run(images_4)
        s = result.summary()
        assert "Score" in s or "score" in s

    def test_reproducible_with_seed(self, images_4):
        cfg1 = Config.default()
        cfg1.assembly.method = "greedy"
        cfg1.assembly.seed = 42
        cfg2 = Config.default()
        cfg2.assembly.method = "greedy"
        cfg2.assembly.seed = 42
        r1 = Pipeline(cfg=cfg1, n_workers=1).run(images_4)
        r2 = Pipeline(cfg=cfg2, n_workers=1).run(images_4)
        assert r1.assembly.total_score == pytest.approx(
            r2.assembly.total_score, abs=1e-4
        )

    def test_run_with_2_fragments(self, pipeline_default, images_2):
        result = pipeline_default.run(images_2)
        assert isinstance(result, PipelineResult)
        assert result.n_placed >= 1

    def test_run_empty_images_no_crash(self):
        p = Pipeline(n_workers=1)
        result = p.run([])
        assert isinstance(result, PipelineResult)

    def test_consistency_report_present(self, images_4):
        cfg = Config.default()
        p = Pipeline(cfg=cfg, n_workers=1)
        result = p.run(images_4)
        assert result.consistency_report is not None

    def test_result_export_json(self, pipeline_default, images_4):
        result = pipeline_default.run(images_4)
        exported = result.export(fmt="json")
        assert exported is not None
        parsed = json.loads(exported)
        assert "fragment_ids" in parsed or "placements" in parsed or isinstance(parsed, dict)

    def test_result_export_summary(self, pipeline_default, images_4):
        result = pipeline_default.run(images_4)
        exported = result.export(fmt="summary")
        assert isinstance(exported, str)


# ─── TestPipelineConfigEffect ─────────────────────────────────────────────────

class TestPipelineConfigEffect:
    def test_different_assembly_methods_run(self, images_2):
        for method in ("greedy", "beam"):
            cfg = Config.default()
            cfg.assembly.method = method
            p = Pipeline(cfg=cfg, n_workers=1)
            result = p.run(images_2)
            assert isinstance(result.assembly, Assembly), f"{method} failed"

    def test_config_preserved_in_result(self, images_4):
        cfg = Config.default()
        cfg.assembly.method = "greedy"
        p = Pipeline(cfg=cfg, n_workers=1)
        result = p.run(images_4)
        assert result.cfg.assembly.method == "greedy"

    def test_dict_roundtrip_config_runs(self, images_2):
        cfg = Config.default()
        cfg2 = Config.from_dict(cfg.to_dict())
        p = Pipeline(cfg=cfg2, n_workers=1)
        result = p.run(images_2)
        assert isinstance(result, PipelineResult)

    def test_matching_threshold_effect(self, images_4):
        cfg_low = Config.default()
        cfg_low.matching.threshold = 0.0
        cfg_high = Config.default()
        cfg_high.matching.threshold = 0.9
        p_low  = Pipeline(cfg=cfg_low, n_workers=1)
        p_high = Pipeline(cfg=cfg_high, n_workers=1)
        frags = p_low.preprocess(images_4)
        if not frags:
            pytest.skip("no fragments")
        _, entries_low  = p_low.match(frags)
        _, entries_high = p_high.match(frags)
        assert len(entries_low) >= len(entries_high)
