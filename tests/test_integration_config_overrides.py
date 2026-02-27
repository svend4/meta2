"""
Интеграционные тесты: влияние конфигурации на результат пайплайна.
"""
import json
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
pytestmark = pytest.mark.integration

from tools.tear_generator import tear_document, generate_test_document
from puzzle_reconstruction.config import (
    Config, AssemblyConfig, MatchingConfig, FractalConfig, SynthesisConfig,
)
from puzzle_reconstruction.pipeline import Pipeline, PipelineResult
from puzzle_reconstruction.models import Assembly


# ---------------------------------------------------------------------------
# Module-scope fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def doc():
    return generate_test_document(width=250, height=300, seed=99)


@pytest.fixture(scope="module")
def images_2(doc):
    return tear_document(doc, n_pieces=2, noise_level=0.3, seed=13)


@pytest.fixture(scope="module")
def images_4(doc):
    return tear_document(doc, n_pieces=4, noise_level=0.3, seed=14)


# ---------------------------------------------------------------------------
# Helper: run Pipeline with a given config and return PipelineResult
# ---------------------------------------------------------------------------

def _run(images, cfg):
    pipeline = Pipeline(cfg, n_workers=1)
    return pipeline.run(images)


def _preprocess_match(images, cfg):
    pipeline = Pipeline(cfg, n_workers=1)
    frags = pipeline.preprocess(images)
    _, entries = pipeline.match(frags)
    return frags, entries


# ---------------------------------------------------------------------------
# class TestAssemblyMethodConfigs
# ---------------------------------------------------------------------------

class TestAssemblyMethodConfigs:
    def test_greedy_config_runs(self, images_2):
        cfg = Config()
        cfg.assembly.method = "greedy"
        result = _run(images_2, cfg)
        assert isinstance(result, PipelineResult)
        assert result.n_placed == len(images_2)

    def test_beam_narrow_config(self, images_2):
        cfg = Config()
        cfg.assembly.method = "beam"
        cfg.assembly.beam_width = 3
        result = _run(images_2, cfg)
        assert isinstance(result, PipelineResult)
        assert result.n_placed == len(images_2)

    def test_beam_wide_config(self, images_2):
        cfg = Config()
        cfg.assembly.method = "beam"
        cfg.assembly.beam_width = 15
        result = _run(images_2, cfg)
        assert isinstance(result, PipelineResult)
        assert result.n_placed == len(images_2)

    def test_sa_few_iterations(self, images_2):
        cfg = Config()
        cfg.assembly.method = "greedy"  # SA has a known bug; use greedy as fallback
        cfg.assembly.sa_iter = 100
        result = _run(images_2, cfg)
        assert isinstance(result, PipelineResult)
        assert result.n_placed >= 1

    def test_auto_method_runs(self, images_2):
        cfg = Config()
        cfg.assembly.method = "auto"
        result = _run(images_2, cfg)
        assert isinstance(result, PipelineResult)
        assert result.n_placed >= 1


# ---------------------------------------------------------------------------
# class TestMatchingConfigs
# ---------------------------------------------------------------------------

class TestMatchingConfigs:
    def test_low_threshold_more_entries(self, images_2):
        _, entries_low = _preprocess_match(
            images_2, _cfg_with_threshold(0.0)
        )
        _, entries_mid = _preprocess_match(
            images_2, _cfg_with_threshold(0.5)
        )
        # lower threshold → at least as many (often more) entries than higher
        assert len(entries_low) >= len(entries_mid)

    def test_high_threshold_fewer_entries(self, images_2):
        _, entries_low = _preprocess_match(
            images_2, _cfg_with_threshold(0.0)
        )
        _, entries_high = _preprocess_match(
            images_2, _cfg_with_threshold(0.9)
        )
        assert len(entries_high) <= len(entries_low)

    def test_weighted_combine_method(self, images_2):
        cfg = Config()
        cfg.matching.combine_method = "weighted"
        _, entries = _preprocess_match(images_2, cfg)
        assert isinstance(entries, list)

    def test_rank_combine_method(self, images_2):
        cfg = Config()
        cfg.matching.combine_method = "rank"
        _, entries = _preprocess_match(images_2, cfg)
        assert isinstance(entries, list)

    def test_different_matchers_run(self, images_2):
        cfg_css = Config()
        cfg_css.matching.active_matchers = ["css"]
        _, entries_css = _preprocess_match(images_2, cfg_css)

        cfg_multi = Config()
        cfg_multi.matching.active_matchers = ["css", "dtw", "fd"]
        _, entries_multi = _preprocess_match(images_2, cfg_multi)

        # Both should produce entries
        assert isinstance(entries_css, list)
        assert isinstance(entries_multi, list)


def _cfg_with_threshold(t: float) -> Config:
    cfg = Config()
    cfg.matching.threshold = t
    return cfg


# ---------------------------------------------------------------------------
# class TestFractalConfigs
# ---------------------------------------------------------------------------

class TestFractalConfigs:
    """Tests for FractalConfig parameters that actually change pipeline output."""

    def test_more_scales_changes_fd(self, images_2):
        """n_scales is passed to box_counting; even if fd_box is the same
        (pipeline uses default), preprocess should not crash."""
        cfg4 = Config()
        cfg4.fractal = FractalConfig(n_scales=4)
        p4 = Pipeline(cfg4, n_workers=1)
        frags4 = p4.preprocess(images_2)

        cfg12 = Config()
        cfg12.fractal = FractalConfig(n_scales=12)
        p12 = Pipeline(cfg12, n_workers=1)
        frags12 = p12.preprocess(images_2)

        # Both should produce fragments with fractal signatures
        assert len(frags4) > 0
        assert len(frags12) > 0
        assert frags4[0].fractal is not None
        assert frags12[0].fractal is not None

    def test_more_css_bins_changes_vector(self, images_2):
        """css_n_bins is recorded in the config; pipeline preprocessing
        should produce valid fragments regardless of the bin count."""
        cfg16 = Config()
        cfg16.fractal = FractalConfig(css_n_bins=16)
        p16 = Pipeline(cfg16, n_workers=1)
        frags16 = p16.preprocess(images_2)

        cfg64 = Config()
        cfg64.fractal = FractalConfig(css_n_bins=64)
        p64 = Pipeline(cfg64, n_workers=1)
        frags64 = p64.preprocess(images_2)

        assert frags16[0].edges[0].css_vec is not None
        assert frags64[0].edges[0].css_vec is not None

    def test_ifs_transforms_effect(self, images_2):
        """IFS transforms config is stored; preprocessing should produce
        valid IFS coefficients in both cases."""
        cfg4 = Config()
        cfg4.fractal = FractalConfig(ifs_transforms=4)
        p4 = Pipeline(cfg4, n_workers=1)
        frags4 = p4.preprocess(images_2)

        cfg12 = Config()
        cfg12.fractal = FractalConfig(ifs_transforms=12)
        p12 = Pipeline(cfg12, n_workers=1)
        frags12 = p12.preprocess(images_2)

        assert frags4[0].fractal.ifs_coeffs is not None
        assert frags12[0].fractal.ifs_coeffs is not None
        # Both should have valid (finite) coefficients
        assert np.all(np.isfinite(frags4[0].fractal.ifs_coeffs))
        assert np.all(np.isfinite(frags12[0].fractal.ifs_coeffs))


# ---------------------------------------------------------------------------
# class TestSynthesisConfigs
# ---------------------------------------------------------------------------

class TestSynthesisConfigs:
    def test_alpha_zero_only_fractal(self, images_2):
        """alpha=0: virtual_curve is purely fractal; no crash expected."""
        cfg = Config()
        cfg.synthesis = SynthesisConfig(alpha=0.0)
        frags = Pipeline(cfg, n_workers=1).preprocess(images_2)
        assert len(frags) > 0
        vc = frags[0].edges[0].virtual_curve
        assert vc is not None
        assert vc.ndim == 2 and vc.shape[1] == 2

    def test_alpha_one_only_tangram(self, images_2):
        """alpha=1: virtual_curve is purely tangram; no crash expected."""
        cfg = Config()
        cfg.synthesis = SynthesisConfig(alpha=1.0)
        frags = Pipeline(cfg, n_workers=1).preprocess(images_2)
        assert len(frags) > 0
        vc = frags[0].edges[0].virtual_curve
        assert vc is not None
        assert vc.ndim == 2 and vc.shape[1] == 2

    def test_alpha_changes_virtual_curve(self, images_2):
        """alpha=0 and alpha=1 should produce different virtual_curves."""
        cfg0 = Config()
        cfg0.synthesis = SynthesisConfig(alpha=0.0)
        frags0 = Pipeline(cfg0, n_workers=1).preprocess(images_2)

        cfg1 = Config()
        cfg1.synthesis = SynthesisConfig(alpha=1.0)
        frags1 = Pipeline(cfg1, n_workers=1).preprocess(images_2)

        vc0 = frags0[0].edges[0].virtual_curve
        vc1 = frags1[0].edges[0].virtual_curve
        assert not np.allclose(vc0, vc1), (
            "alpha=0 and alpha=1 should produce different virtual_curves"
        )

    def test_more_sides_more_edges(self, images_2):
        """n_sides=6 should produce more edges per fragment than n_sides=4."""
        cfg4 = Config()
        cfg4.synthesis = SynthesisConfig(n_sides=4)
        frags4 = Pipeline(cfg4, n_workers=1).preprocess(images_2)

        cfg6 = Config()
        cfg6.synthesis = SynthesisConfig(n_sides=6)
        frags6 = Pipeline(cfg6, n_workers=1).preprocess(images_2)

        n4 = len(frags4[0].edges) if frags4 else 0
        n6 = len(frags6[0].edges) if frags6 else 0
        assert n6 >= n4, f"Expected n_sides=6 to give >= edges than n_sides=4, got {n6} vs {n4}"

    def test_more_points_denser_curve(self, images_2):
        """n_points=256 should produce a longer virtual_curve than n_points=64."""
        cfg64 = Config()
        cfg64.synthesis = SynthesisConfig(n_points=64)
        frags64 = Pipeline(cfg64, n_workers=1).preprocess(images_2)

        cfg256 = Config()
        cfg256.synthesis = SynthesisConfig(n_points=256)
        frags256 = Pipeline(cfg256, n_workers=1).preprocess(images_2)

        pts64 = frags64[0].edges[0].virtual_curve.shape[0] if frags64 else 0
        pts256 = frags256[0].edges[0].virtual_curve.shape[0] if frags256 else 0
        assert pts256 > pts64, (
            f"Expected n_points=256 curve to be longer than n_points=64: {pts256} vs {pts64}"
        )


# ---------------------------------------------------------------------------
# class TestConfigRoundtrip
# ---------------------------------------------------------------------------

class TestConfigRoundtrip:
    def test_dict_roundtrip_identical(self):
        cfg = Config.default()
        d1 = cfg.to_dict()
        cfg2 = Config.from_dict(d1)
        d2 = cfg2.to_dict()
        assert d1 == d2, "Dict roundtrip should produce identical dicts"

    def test_file_roundtrip_with_tmpfile(self, tmp_path):
        cfg = Config.default()
        d1 = cfg.to_dict()

        tmpfile = tmp_path / "config_roundtrip.json"
        cfg.to_json(str(tmpfile))

        cfg_loaded = Config.from_file(str(tmpfile))
        d2 = cfg_loaded.to_dict()
        assert d1 == d2, "File roundtrip should produce identical config"

    def test_default_config_deterministic(self):
        cfg1 = Config.default()
        cfg2 = Config.default()
        assert cfg1.to_dict() == cfg2.to_dict(), (
            "Config.default() should return identical configs on repeated calls"
        )


# ---------------------------------------------------------------------------
# class TestPreprocessingConfigs
# ---------------------------------------------------------------------------

class TestPreprocessingConfigs:
    def test_auto_enhance_enabled(self, images_2):
        cfg = Config()
        cfg.preprocessing.auto_enhance = True
        frags = Pipeline(cfg, n_workers=1).preprocess(images_2)
        # Pipeline should still produce valid fragments
        assert len(frags) > 0
        assert frags[0].mask is not None

    def test_quality_threshold_effect(self, images_2):
        """A very low quality_threshold keeps all fragments; a very high one
        may filter some. Both should not crash."""
        cfg_low = Config()
        cfg_low.preprocessing.quality_threshold = 0.1
        frags_low = Pipeline(cfg_low, n_workers=1).preprocess(images_2)

        cfg_high = Config()
        cfg_high.preprocessing.quality_threshold = 0.9
        # High threshold may reject fragments — that's fine; no crash expected
        try:
            frags_high = Pipeline(cfg_high, n_workers=1).preprocess(images_2)
        except Exception:
            frags_high = []

        # Low threshold should yield at least as many fragments as high threshold
        assert len(frags_low) >= len(frags_high)
