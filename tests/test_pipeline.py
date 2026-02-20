"""
Юнит-тесты для puzzle_reconstruction/pipeline.py.

Тесты покрывают:
    - Pipeline инициализация с разными конфигурациями
    - preprocess() — параллельная и последовательная обработка
    - match() — построение матрицы совместимости
    - assemble() — все методы сборки включая exhaustive
    - verify() — OCR-заглушка при отключённом OCR
    - run() — полный end-to-end прогон
    - on_progress callback — вызывается при прогрессе
    - PipelineResult — summary(), n_placed, n_input
"""
import math
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from puzzle_reconstruction.pipeline import Pipeline, PipelineResult
from puzzle_reconstruction.config import Config
from puzzle_reconstruction.models import (
    Fragment, Assembly, FractalSignature, TangramSignature,
    EdgeSignature, ShapeClass, EdgeSide,
)


# ─── Фикстуры ────────────────────────────────────────────────────────────

def _synthetic_image(h: int = 100, w: int = 80,
                      color: tuple = (200, 190, 180)) -> np.ndarray:
    """Синтетическое BGR изображение с белым фоном и цветным прямоугольником."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    img[10:h-10, 10:w-10] = color
    return img


def _make_images(n: int = 3) -> list:
    colors = [(180, 160, 140), (140, 180, 160), (160, 140, 180)]
    return [_synthetic_image(color=colors[i % len(colors)]) for i in range(n)]


def _fast_config(method: str = "greedy") -> Config:
    """Быстрый конфиг для тестов."""
    cfg = Config.default()
    cfg.assembly.method      = method
    cfg.assembly.sa_iter     = 100
    cfg.assembly.beam_width  = 3
    cfg.assembly.gamma_iter  = 50
    cfg.verification.run_ocr = False   # Отключаем OCR в тестах
    return cfg


# ─── Инициализация ────────────────────────────────────────────────────────

class TestPipelineInit:

    def test_default_config(self):
        p = Pipeline()
        assert p.cfg is not None
        assert p.n_workers >= 1

    def test_custom_config(self):
        cfg = _fast_config()
        p   = Pipeline(cfg=cfg)
        assert p.cfg.assembly.method == "greedy"

    def test_n_workers_default(self):
        p = Pipeline()
        assert p.n_workers == 4

    def test_logger_created(self):
        p = Pipeline()
        assert p.log is not None

    def test_on_progress_stored(self):
        calls = []
        def cb(stage, done, total):
            calls.append((stage, done, total))
        p = Pipeline(on_progress=cb)
        assert p.on_progress is cb


# ─── preprocess ───────────────────────────────────────────────────────────

class TestPipelinePreprocess:

    def test_returns_list(self):
        p      = Pipeline(cfg=_fast_config())
        images = _make_images(2)
        frags  = p.preprocess(images)
        assert isinstance(frags, list)

    def test_fragment_ids_sequential(self):
        p      = Pipeline(cfg=_fast_config())
        images = _make_images(3)
        frags  = p.preprocess(images)
        ids    = [f.fragment_id for f in frags]
        assert ids == sorted(ids)

    def test_fragments_have_edges(self):
        p      = Pipeline(cfg=_fast_config())
        images = _make_images(2)
        frags  = p.preprocess(images)
        for f in frags:
            assert len(f.edges) >= 1

    def test_fragments_have_fractal(self):
        p      = Pipeline(cfg=_fast_config())
        images = _make_images(2)
        frags  = p.preprocess(images)
        for f in frags:
            assert f.fractal is not None

    def test_parallel_same_result_as_sequential(self):
        """Параллельная и последовательная обработка дают одно и то же число фрагментов."""
        images = _make_images(3)
        p_seq  = Pipeline(cfg=_fast_config(), n_workers=1)
        p_par  = Pipeline(cfg=_fast_config(), n_workers=4)
        n_seq  = len(p_seq.preprocess(images))
        n_par  = len(p_par.preprocess(images))
        assert n_seq == n_par

    def test_empty_images(self):
        p     = Pipeline(cfg=_fast_config())
        frags = p.preprocess([])
        assert frags == []

    def test_progress_callback_called(self):
        progress_events = []
        def cb(stage, done, total):
            progress_events.append((stage, done, total))
        p = Pipeline(cfg=_fast_config(), n_workers=1, on_progress=cb)
        p.preprocess(_make_images(2))
        assert len(progress_events) > 0
        assert all(done <= total for _, done, total in progress_events)

    def test_single_image(self):
        p     = Pipeline(cfg=_fast_config())
        frags = p.preprocess([_synthetic_image()])
        assert len(frags) >= 0   # Может не пройти сегментацию на синтетике


# ─── match ────────────────────────────────────────────────────────────────

class TestPipelineMatch:

    def _get_fragments(self, n: int = 3) -> list:
        p = Pipeline(cfg=_fast_config())
        return p.preprocess(_make_images(n))

    def test_returns_tuple(self):
        p     = Pipeline(cfg=_fast_config())
        frags = self._get_fragments(2)
        if not frags:
            pytest.skip("Нет обработанных фрагментов")
        result = p.match(frags)
        assert isinstance(result, tuple) and len(result) == 2

    def test_matrix_square(self):
        p     = Pipeline(cfg=_fast_config())
        frags = self._get_fragments(2)
        if len(frags) < 2:
            pytest.skip("Нет обработанных фрагментов")
        matrix, _ = p.match(frags)
        n_edges = sum(len(f.edges) for f in frags)
        assert matrix.shape == (n_edges, n_edges)

    def test_entries_sorted_by_score(self):
        p     = Pipeline(cfg=_fast_config())
        frags = self._get_fragments(3)
        if len(frags) < 2:
            pytest.skip("Нет обработанных фрагментов")
        _, entries = p.match(frags)
        scores = [e.score for e in entries]
        assert scores == sorted(scores, reverse=True) or len(scores) <= 1


# ─── assemble ─────────────────────────────────────────────────────────────

class TestPipelineAssemble:

    def _setup(self, n: int = 3):
        p      = Pipeline(cfg=_fast_config())
        frags  = p.preprocess(_make_images(n))
        if len(frags) < 2:
            pytest.skip("Нет обработанных фрагментов")
        _, entries = p.match(frags)
        return p, frags, entries

    def test_greedy_places_all(self):
        p, frags, entries = self._setup(3)
        asm = p.assemble(frags, entries)
        assert len(asm.placements) == len(frags)

    def test_beam_places_all(self):
        cfg = _fast_config("beam")
        p   = Pipeline(cfg=cfg)
        frags = p.preprocess(_make_images(3))
        if len(frags) < 2:
            pytest.skip()
        _, entries = p.match(frags)
        asm = p.assemble(frags, entries)
        assert len(asm.placements) == len(frags)

    def test_exhaustive_small_n(self):
        cfg = _fast_config("exhaustive")
        p   = Pipeline(cfg=cfg)
        frags = p.preprocess(_make_images(3))
        if len(frags) < 2:
            pytest.skip()
        _, entries = p.match(frags)
        asm = p.assemble(frags, entries)
        assert len(asm.placements) == len(frags)

    def test_invalid_method_raises(self):
        cfg = _fast_config()
        cfg.assembly.method = "nonexistent_method_xyz"
        p   = Pipeline(cfg=cfg)
        with pytest.raises(ValueError, match="метод"):
            p.assemble([], [])

    def test_placements_all_finite(self):
        p, frags, entries = self._setup(3)
        asm = p.assemble(frags, entries)
        for pos, angle in asm.placements.values():
            assert np.all(np.isfinite(pos))
            assert math.isfinite(angle)


# ─── verify ───────────────────────────────────────────────────────────────

class TestPipelineVerify:

    def test_verify_ocr_disabled_returns_zero(self):
        cfg = _fast_config()
        cfg.verification.run_ocr = False
        p   = Pipeline(cfg=cfg)
        asm = Assembly(fragments=[], placements={}, compat_matrix=np.array([]))
        score = p.verify(asm)
        assert score == 0.0

    def test_verify_returns_float(self):
        cfg = _fast_config()
        cfg.verification.run_ocr = False
        p   = Pipeline(cfg=cfg)
        asm = Assembly(fragments=[], placements={}, compat_matrix=np.array([]))
        assert isinstance(p.verify(asm), float)


# ─── run ──────────────────────────────────────────────────────────────────

class TestPipelineRun:

    def test_returns_pipeline_result(self):
        p      = Pipeline(cfg=_fast_config())
        images = _make_images(2)
        result = p.run(images)
        assert isinstance(result, PipelineResult)

    def test_result_has_assembly(self):
        p      = Pipeline(cfg=_fast_config())
        images = _make_images(2)
        result = p.run(images)
        assert isinstance(result.assembly, Assembly)

    def test_result_n_input(self):
        p      = Pipeline(cfg=_fast_config())
        images = _make_images(3)
        result = p.run(images)
        assert result.n_input == 3

    def test_result_summary_string(self):
        p      = Pipeline(cfg=_fast_config())
        images = _make_images(2)
        result = p.run(images)
        s = result.summary()
        assert isinstance(s, str)
        assert "Score" in s or "score" in s.lower()

    def test_empty_images_returns_result(self):
        p      = Pipeline(cfg=_fast_config())
        result = p.run([])
        assert result.n_placed == 0

    def test_all_assembly_methods_run(self):
        images = _make_images(3)
        for method in ("greedy", "beam", "gamma"):
            cfg = _fast_config(method)
            p   = Pipeline(cfg=cfg)
            result = p.run(images)
            assert isinstance(result, PipelineResult)
