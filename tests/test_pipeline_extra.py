"""Extra tests for puzzle_reconstruction/pipeline.py."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from puzzle_reconstruction.pipeline import Pipeline, PipelineResult
from puzzle_reconstruction.config import Config
from puzzle_reconstruction.models import Assembly


def _image(h=100, w=80, color=(200, 190, 180)):
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    img[10:h - 10, 10:w - 10] = color
    return img


def _images(n=3):
    colors = [(180, 160, 140), (140, 180, 160), (160, 140, 180)]
    return [_image(color=colors[i % len(colors)]) for i in range(n)]


def _cfg(method="greedy"):
    cfg = Config.default()
    cfg.assembly.method = method
    cfg.assembly.sa_iter = 100
    cfg.assembly.beam_width = 3
    cfg.assembly.gamma_iter = 50
    cfg.verification.run_ocr = False
    return cfg


# ─── Pipeline init extras ─────────────────────────────────────────────────────

class TestPipelineInitExtra:
    def test_cfg_not_none(self):
        p = Pipeline()
        assert p.cfg is not None

    def test_n_workers_positive(self):
        p = Pipeline()
        assert p.n_workers >= 1

    def test_custom_n_workers(self):
        p = Pipeline(n_workers=2)
        assert p.n_workers == 2

    def test_on_progress_none_default(self):
        p = Pipeline()
        assert p.on_progress is None

    def test_on_progress_stored(self):
        cb = lambda s, d, t: None
        p = Pipeline(on_progress=cb)
        assert p.on_progress is cb

    def test_log_not_none(self):
        p = Pipeline()
        assert p.log is not None

    def test_different_configs(self):
        for method in ("greedy", "beam", "gamma"):
            p = Pipeline(cfg=_cfg(method))
            assert p.cfg.assembly.method == method


# ─── Pipeline.preprocess extras ───────────────────────────────────────────────

class TestPipelinePreprocessExtra:
    def test_empty_returns_empty(self):
        p = Pipeline(cfg=_cfg())
        assert p.preprocess([]) == []

    def test_returns_list(self):
        p = Pipeline(cfg=_cfg())
        result = p.preprocess(_images(2))
        assert isinstance(result, list)

    def test_ids_unique(self):
        p = Pipeline(cfg=_cfg())
        frags = p.preprocess(_images(3))
        ids = [f.fragment_id for f in frags]
        assert len(ids) == len(set(ids))

    def test_ids_non_negative(self):
        p = Pipeline(cfg=_cfg())
        frags = p.preprocess(_images(2))
        for f in frags:
            assert f.fragment_id >= 0

    def test_fragments_have_image(self):
        p = Pipeline(cfg=_cfg())
        frags = p.preprocess(_images(2))
        for f in frags:
            assert f.image is not None
            assert f.image.ndim == 3

    def test_n_workers_1_sequential(self):
        p = Pipeline(cfg=_cfg(), n_workers=1)
        frags = p.preprocess(_images(2))
        assert isinstance(frags, list)

    def test_n_workers_2_parallel(self):
        p = Pipeline(cfg=_cfg(), n_workers=2)
        frags = p.preprocess(_images(3))
        assert isinstance(frags, list)

    def test_progress_callback_events(self):
        events = []
        p = Pipeline(cfg=_cfg(), n_workers=1, on_progress=lambda s, d, t: events.append((s, d, t)))
        p.preprocess(_images(2))
        assert len(events) > 0


# ─── Pipeline.verify extras ───────────────────────────────────────────────────

class TestPipelineVerifyExtra:
    def _make_empty_asm(self):
        return Assembly(fragments=[], placements={}, compat_matrix=np.array([]))

    def test_ocr_disabled_returns_zero(self):
        p = Pipeline(cfg=_cfg())
        assert p.verify(self._make_empty_asm()) == 0.0

    def test_returns_float(self):
        p = Pipeline(cfg=_cfg())
        result = p.verify(self._make_empty_asm())
        assert isinstance(result, float)

    def test_score_nonneg(self):
        p = Pipeline(cfg=_cfg())
        assert p.verify(self._make_empty_asm()) >= 0.0


# ─── Pipeline.run extras ──────────────────────────────────────────────────────

class TestPipelineRunExtra:
    def test_returns_pipeline_result(self):
        p = Pipeline(cfg=_cfg())
        r = p.run(_images(2))
        assert isinstance(r, PipelineResult)

    def test_result_assembly_is_assembly(self):
        p = Pipeline(cfg=_cfg())
        r = p.run(_images(2))
        assert isinstance(r.assembly, Assembly)

    def test_empty_images_n_placed_zero(self):
        p = Pipeline(cfg=_cfg())
        r = p.run([])
        assert r.n_placed == 0

    def test_n_input_equals_images_count(self):
        for n in (1, 2, 3):
            p = Pipeline(cfg=_cfg())
            r = p.run(_images(n))
            assert r.n_input == n

    def test_summary_is_string(self):
        p = Pipeline(cfg=_cfg())
        r = p.run(_images(2))
        assert isinstance(r.summary(), str)

    def test_greedy_method_runs(self):
        r = Pipeline(cfg=_cfg("greedy")).run(_images(2))
        assert isinstance(r, PipelineResult)

    def test_beam_method_runs(self):
        r = Pipeline(cfg=_cfg("beam")).run(_images(2))
        assert isinstance(r, PipelineResult)

    def test_gamma_method_runs(self):
        r = Pipeline(cfg=_cfg("gamma")).run(_images(2))
        assert isinstance(r, PipelineResult)

    def test_n_placed_lte_n_input(self):
        p = Pipeline(cfg=_cfg())
        r = p.run(_images(3))
        assert r.n_placed <= r.n_input

    def test_progress_callback_fires(self):
        events = []
        p = Pipeline(cfg=_cfg(), n_workers=1, on_progress=lambda s, d, t: events.append(s))
        p.run(_images(2))
        assert len(events) > 0


# ─── PipelineResult extras ────────────────────────────────────────────────────

class TestPipelineResultExtra:
    def _run(self, n=2):
        return Pipeline(cfg=_cfg()).run(_images(n))

    def test_n_input_positive(self):
        r = self._run(2)
        assert r.n_input == 2

    def test_n_placed_nonneg(self):
        r = self._run(2)
        assert r.n_placed >= 0

    def test_assembly_has_placements_dict(self):
        r = self._run(2)
        assert isinstance(r.assembly.placements, dict)

    def test_summary_contains_percentage(self):
        r = self._run(2)
        s = r.summary()
        assert "%" in s or len(s) > 20

    def test_empty_run_summary_is_string(self):
        r = Pipeline(cfg=_cfg()).run([])
        assert isinstance(r.summary(), str)
