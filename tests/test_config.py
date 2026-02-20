"""
Юнит-тесты для puzzle_reconstruction/config.py.

Тесты покрывают:
    - Дефолтные значения всех секций
    - from_dict() / to_dict() — сериализация
    - from_file() — JSON файл
    - to_json() — запись в JSON
    - apply_overrides() — CLI-переопределения
    - Граничные случаи: несуществующий файл, пустой dict
"""
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


# ─── Дефолтные значения ───────────────────────────────────────────────────

class TestDefaultValues:

    def test_segmentation_defaults(self):
        cfg = Config.default()
        assert cfg.segmentation.method in ("otsu", "adaptive", "grabcut")
        assert cfg.segmentation.morph_kernel >= 1

    def test_synthesis_defaults(self):
        cfg = Config.default()
        assert 0.0 <= cfg.synthesis.alpha <= 1.0
        assert cfg.synthesis.n_sides >= 3
        assert cfg.synthesis.n_points >= 8

    def test_fractal_defaults(self):
        cfg = Config.default()
        assert cfg.fractal.n_scales >= 2
        assert cfg.fractal.ifs_transforms >= 1
        assert cfg.fractal.css_n_sigmas >= 1
        assert cfg.fractal.css_n_bins >= 8

    def test_matching_defaults(self):
        cfg = Config.default()
        assert 0.0 <= cfg.matching.threshold <= 1.0
        assert cfg.matching.dtw_window >= 1

    def test_assembly_defaults(self):
        cfg = Config.default()
        assert cfg.assembly.method in ("greedy", "sa", "beam", "gamma", "exhaustive")
        assert cfg.assembly.beam_width >= 1
        assert cfg.assembly.sa_iter >= 1
        assert cfg.assembly.sa_T_max > cfg.assembly.sa_T_min > 0
        assert 0.0 < cfg.assembly.sa_cooling < 1.0
        assert cfg.assembly.gamma_iter >= 1

    def test_verification_defaults(self):
        cfg = Config.default()
        assert isinstance(cfg.verification.run_ocr, bool)
        assert isinstance(cfg.verification.ocr_lang, str)
        assert isinstance(cfg.verification.export_pdf, bool)

    def test_all_sections_present(self):
        cfg = Config.default()
        assert hasattr(cfg, "segmentation")
        assert hasattr(cfg, "synthesis")
        assert hasattr(cfg, "fractal")
        assert hasattr(cfg, "matching")
        assert hasattr(cfg, "assembly")
        assert hasattr(cfg, "verification")


# ─── to_dict / from_dict ─────────────────────────────────────────────────

class TestDictRoundtrip:

    def test_to_dict_returns_dict(self):
        cfg = Config.default()
        d   = cfg.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_all_sections(self):
        d = Config.default().to_dict()
        assert "segmentation" in d
        assert "synthesis"    in d
        assert "assembly"     in d

    def test_from_dict_default_empty(self):
        cfg = Config.from_dict({})
        # Все поля должны получить значения по умолчанию
        assert cfg.synthesis.alpha == Config.default().synthesis.alpha

    def test_roundtrip_preserves_values(self):
        cfg = Config.default()
        cfg.synthesis.alpha   = 0.333
        cfg.assembly.method   = "sa"
        cfg.assembly.sa_iter  = 9999
        cfg.fractal.n_scales  = 12

        restored = Config.from_dict(cfg.to_dict())
        assert math.isclose(restored.synthesis.alpha, 0.333, rel_tol=1e-6)
        assert restored.assembly.method  == "sa"
        assert restored.assembly.sa_iter == 9999
        assert restored.fractal.n_scales == 12

    def test_from_dict_partial(self):
        """Неполный dict — остальные поля = дефолт."""
        cfg = Config.from_dict({"assembly": {"method": "gamma"}})
        assert cfg.assembly.method   == "gamma"
        assert cfg.synthesis.alpha   == Config.default().synthesis.alpha


# ─── to_json / from_file ──────────────────────────────────────────────────

class TestJsonIo:

    def test_to_json_creates_file(self, tmp_path):
        path = tmp_path / "cfg.json"
        Config.default().to_json(path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_to_json_valid_json(self, tmp_path):
        path = tmp_path / "cfg.json"
        Config.default().to_json(path)
        with open(path) as f:
            d = json.load(f)
        assert isinstance(d, dict)
        assert "assembly" in d

    def test_from_file_json(self, tmp_path):
        path = tmp_path / "cfg.json"
        cfg  = Config.default()
        cfg.synthesis.n_sides = 6
        cfg.to_json(path)

        loaded = Config.from_file(path)
        assert loaded.synthesis.n_sides == 6

    def test_from_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Config.from_file(tmp_path / "nonexistent.json")

    def test_from_file_roundtrip_full(self, tmp_path):
        """Полный цикл: default → изменить → JSON → загрузить → проверить."""
        cfg = Config.default()
        cfg.assembly.method     = "beam"
        cfg.assembly.beam_width = 15
        cfg.matching.threshold  = 0.42
        cfg.fractal.css_n_bins  = 64

        path = tmp_path / "full.json"
        cfg.to_json(path)
        loaded = Config.from_file(path)

        assert loaded.assembly.method     == "beam"
        assert loaded.assembly.beam_width == 15
        assert math.isclose(loaded.matching.threshold, 0.42, rel_tol=1e-6)
        assert loaded.fractal.css_n_bins  == 64

    def test_from_file_empty_json(self, tmp_path):
        """Пустой JSON {} → все поля = дефолт."""
        path = tmp_path / "empty.json"
        path.write_text("{}")
        loaded = Config.from_file(path)
        assert loaded.synthesis.alpha == Config.default().synthesis.alpha


# ─── apply_overrides ──────────────────────────────────────────────────────

class TestApplyOverrides:

    def test_override_alpha(self):
        cfg = Config.default()
        cfg.apply_overrides(alpha=0.7)
        assert math.isclose(cfg.synthesis.alpha, 0.7)

    def test_override_method(self):
        cfg = Config.default()
        cfg.apply_overrides(method="sa")
        assert cfg.assembly.method == "sa"

    def test_override_sa_iter(self):
        cfg = Config.default()
        cfg.apply_overrides(sa_iter=12345)
        assert cfg.assembly.sa_iter == 12345

    def test_override_beam_width(self):
        cfg = Config.default()
        cfg.apply_overrides(beam_width=20)
        assert cfg.assembly.beam_width == 20

    def test_override_threshold(self):
        cfg = Config.default()
        cfg.apply_overrides(threshold=0.25)
        assert math.isclose(cfg.matching.threshold, 0.25)

    def test_override_n_sides(self):
        cfg = Config.default()
        cfg.apply_overrides(n_sides=6)
        assert cfg.synthesis.n_sides == 6

    def test_override_none_ignored(self):
        """None-значения не должны перезаписывать текущее значение."""
        cfg = Config.default()
        original_alpha = cfg.synthesis.alpha
        cfg.apply_overrides(alpha=None)
        assert cfg.synthesis.alpha == original_alpha

    def test_override_returns_self(self):
        """apply_overrides возвращает self для цепочки вызовов."""
        cfg    = Config.default()
        result = cfg.apply_overrides(alpha=0.5)
        assert result is cfg

    def test_multiple_overrides(self):
        cfg = Config.default()
        cfg.apply_overrides(alpha=0.3, method="gamma", beam_width=7)
        assert math.isclose(cfg.synthesis.alpha, 0.3)
        assert cfg.assembly.method     == "gamma"
        assert cfg.assembly.beam_width == 7


# ─── Вложенные dataclass-секции ──────────────────────────────────────────

class TestSections:

    def test_segmentation_config_standalone(self):
        s = SegmentationConfig(method="adaptive", morph_kernel=5)
        assert s.method == "adaptive"
        assert s.morph_kernel == 5

    def test_assembly_config_standalone(self):
        a = AssemblyConfig(method="gamma", gamma_iter=5000)
        assert a.method == "gamma"
        assert a.gamma_iter == 5000

    def test_synthesis_config_standalone(self):
        s = SynthesisConfig(alpha=0.8, n_sides=5, n_points=256)
        assert math.isclose(s.alpha, 0.8)
        assert s.n_sides   == 5
        assert s.n_points  == 256
