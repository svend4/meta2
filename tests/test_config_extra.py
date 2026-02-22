"""Additional tests for puzzle_reconstruction/config.py."""
import json
import math
import pytest
from pathlib import Path

from puzzle_reconstruction.config import (
    Config,
    SegmentationConfig,
    SynthesisConfig,
    FractalConfig,
    MatchingConfig,
    AssemblyConfig,
    VerificationConfig,
)


# ─── TestDefaultValuesExtra ───────────────────────────────────────────────────

class TestDefaultValuesExtra:
    def test_fractal_n_scales_gte_2(self):
        assert Config.default().fractal.n_scales >= 2

    def test_fractal_ifs_transforms_gte_1(self):
        assert Config.default().fractal.ifs_transforms >= 1

    def test_fractal_css_n_bins_gte_8(self):
        assert Config.default().fractal.css_n_bins >= 8

    def test_fractal_css_n_sigmas_gte_1(self):
        assert Config.default().fractal.css_n_sigmas >= 1

    def test_matching_dtw_window_positive(self):
        assert Config.default().matching.dtw_window > 0

    def test_matching_threshold_in_01(self):
        t = Config.default().matching.threshold
        assert 0.0 <= t <= 1.0

    def test_verification_ocr_lang_nonempty(self):
        lang = Config.default().verification.ocr_lang
        assert isinstance(lang, str) and len(lang) > 0

    def test_verification_run_ocr_bool(self):
        assert isinstance(Config.default().verification.run_ocr, bool)

    def test_verification_export_pdf_bool(self):
        assert isinstance(Config.default().verification.export_pdf, bool)

    def test_assembly_seed_is_int_or_none(self):
        seed = Config.default().assembly.seed
        assert seed is None or isinstance(seed, int)

    def test_assembly_sa_cooling_lt1(self):
        assert Config.default().assembly.sa_cooling < 1.0

    def test_assembly_beam_width_positive(self):
        assert Config.default().assembly.beam_width >= 1

    def test_synthesis_n_points_positive(self):
        assert Config.default().synthesis.n_points > 0


# ─── TestDictRoundtripExtra ───────────────────────────────────────────────────

class TestDictRoundtripExtra:
    def test_fractal_roundtrip(self):
        cfg = Config.default()
        cfg.fractal.n_scales = 16
        cfg.fractal.css_n_bins = 64
        restored = Config.from_dict(cfg.to_dict())
        assert restored.fractal.n_scales == 16
        assert restored.fractal.css_n_bins == 64

    def test_matching_roundtrip(self):
        cfg = Config.default()
        cfg.matching.threshold = 0.77
        cfg.matching.dtw_window = 35
        restored = Config.from_dict(cfg.to_dict())
        assert math.isclose(restored.matching.threshold, 0.77, rel_tol=1e-6)
        assert restored.matching.dtw_window == 35

    def test_verification_roundtrip(self):
        cfg = Config.default()
        cfg.verification.run_ocr = False
        cfg.verification.export_pdf = True
        cfg.verification.ocr_lang = "deu"
        restored = Config.from_dict(cfg.to_dict())
        assert restored.verification.run_ocr is False
        assert restored.verification.export_pdf is True
        assert restored.verification.ocr_lang == "deu"

    def test_to_dict_has_fractal_key(self):
        d = Config.default().to_dict()
        assert "fractal" in d

    def test_to_dict_has_matching_key(self):
        d = Config.default().to_dict()
        assert "matching" in d

    def test_to_dict_has_verification_key(self):
        d = Config.default().to_dict()
        assert "verification" in d

    def test_from_dict_partial_fractal(self):
        cfg = Config.from_dict({"fractal": {"n_scales": 4}})
        assert cfg.fractal.n_scales == 4
        assert cfg.synthesis.alpha == Config.default().synthesis.alpha

    def test_multiple_modifications_roundtrip(self):
        cfg = Config.default()
        cfg.synthesis.alpha = 0.42
        cfg.fractal.ifs_transforms = 4
        cfg.matching.threshold = 0.15
        cfg.assembly.method = "beam"
        cfg.assembly.beam_width = 5
        restored = Config.from_dict(cfg.to_dict())
        assert math.isclose(restored.synthesis.alpha, 0.42, rel_tol=1e-6)
        assert restored.fractal.ifs_transforms == 4
        assert math.isclose(restored.matching.threshold, 0.15, rel_tol=1e-6)
        assert restored.assembly.method == "beam"
        assert restored.assembly.beam_width == 5


# ─── TestJsonIoExtra ──────────────────────────────────────────────────────────

class TestJsonIoExtra:
    def test_fractal_section_survives_json(self, tmp_path):
        cfg = Config.default()
        cfg.fractal.n_scales = 20
        p = tmp_path / "cfg.json"
        cfg.to_json(p)
        loaded = Config.from_file(p)
        assert loaded.fractal.n_scales == 20

    def test_matching_section_survives_json(self, tmp_path):
        cfg = Config.default()
        cfg.matching.dtw_window = 50
        p = tmp_path / "m.json"
        cfg.to_json(p)
        loaded = Config.from_file(p)
        assert loaded.matching.dtw_window == 50

    def test_verification_section_survives_json(self, tmp_path):
        cfg = Config.default()
        cfg.verification.export_pdf = True
        p = tmp_path / "v.json"
        cfg.to_json(p)
        loaded = Config.from_file(p)
        assert loaded.verification.export_pdf is True

    def test_json_is_indented(self, tmp_path):
        p = tmp_path / "pretty.json"
        Config.default().to_json(p)
        text = p.read_text()
        assert "\n" in text  # has newlines → indented

    def test_all_sections_in_json(self, tmp_path):
        p = tmp_path / "all.json"
        Config.default().to_json(p)
        d = json.loads(p.read_text())
        for key in ("segmentation", "synthesis", "fractal",
                    "matching", "assembly", "verification"):
            assert key in d

    def test_from_file_preserves_synthesis_n_points(self, tmp_path):
        cfg = Config.default()
        cfg.synthesis.n_points = 256
        p = tmp_path / "np.json"
        cfg.to_json(p)
        assert Config.from_file(p).synthesis.n_points == 256


# ─── TestApplyOverridesExtra ──────────────────────────────────────────────────

class TestApplyOverridesExtra:
    def test_override_gamma_iter_unknown_silently_ignored(self):
        """gamma_iter not in override mapping → value unchanged, no crash."""
        cfg = Config.default()
        original = cfg.assembly.gamma_iter
        cfg.apply_overrides(gamma_iter=9999)
        assert cfg.assembly.gamma_iter == original

    def test_override_seed(self):
        cfg = Config.default()
        cfg.apply_overrides(seed=99)
        assert cfg.assembly.seed == 99

    def test_override_seg_method(self):
        cfg = Config.default()
        cfg.apply_overrides(seg_method="adaptive")
        assert cfg.segmentation.method == "adaptive"

    def test_override_chained(self):
        cfg = Config.default()
        cfg.apply_overrides(alpha=0.2).apply_overrides(method="beam")
        assert math.isclose(cfg.synthesis.alpha, 0.2)
        assert cfg.assembly.method == "beam"

    def test_override_sa_T_max_unknown_silently_ignored(self):
        """sa_T_max not in override mapping → value unchanged, no crash."""
        cfg = Config.default()
        original = cfg.assembly.sa_T_max
        cfg.apply_overrides(sa_T_max=500.0)
        assert cfg.assembly.sa_T_max == pytest.approx(original)

    def test_override_unknown_ignored(self):
        """Unknown keys should not crash."""
        cfg = Config.default()
        cfg.apply_overrides(no_such_field=123)

    def test_all_overrides_in_one_call(self):
        cfg = Config.default()
        cfg.apply_overrides(
            alpha=0.1, method="sa", sa_iter=500,
            beam_width=3, threshold=0.5, n_sides=8
        )
        assert math.isclose(cfg.synthesis.alpha, 0.1)
        assert cfg.assembly.method == "sa"
        assert cfg.assembly.sa_iter == 500
        assert cfg.assembly.beam_width == 3
        assert math.isclose(cfg.matching.threshold, 0.5)
        assert cfg.synthesis.n_sides == 8


# ─── TestSectionsExtra ────────────────────────────────────────────────────────

class TestSectionsExtra:
    def test_fractal_config_standalone(self):
        fc = FractalConfig(n_scales=4, ifs_transforms=2, css_n_sigmas=3, css_n_bins=16)
        assert fc.n_scales == 4
        assert fc.ifs_transforms == 2

    def test_matching_config_standalone(self):
        mc = MatchingConfig(threshold=0.6, dtw_window=10)
        assert math.isclose(mc.threshold, 0.6)
        assert mc.dtw_window == 10

    def test_verification_config_standalone(self):
        vc = VerificationConfig(run_ocr=False, ocr_lang="fra", export_pdf=True)
        assert vc.run_ocr is False
        assert vc.ocr_lang == "fra"
        assert vc.export_pdf is True

    def test_synthesis_config_n_points(self):
        sc = SynthesisConfig(alpha=0.3, n_sides=6, n_points=64)
        assert sc.n_points == 64

    def test_assembly_config_sa_params(self):
        ac = AssemblyConfig(method="sa", sa_iter=2000,
                            sa_T_max=200.0, sa_T_min=0.01, sa_cooling=0.95)
        assert ac.sa_iter == 2000
        assert ac.sa_T_max == pytest.approx(200.0)
        assert ac.sa_cooling == pytest.approx(0.95)

    def test_segmentation_config_method(self):
        sc = SegmentationConfig(method="grabcut", morph_kernel=7)
        assert sc.method == "grabcut"
        assert sc.morph_kernel == 7
