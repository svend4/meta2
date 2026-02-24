"""
Централизованная конфигурация системы восстановления пазлов.

Поддерживает загрузку из YAML/JSON файлов и override через аргументы CLI.

Пример конфига (config.yaml):
    segmentation:
      method: otsu
    synthesis:
      alpha: 0.5
      n_sides: 4
    assembly:
      method: beam
      beam_width: 10
      sa_iter: 5000
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Literal, Optional


@dataclass
class SegmentationConfig:
    method:       Literal["otsu", "adaptive", "grabcut"] = "otsu"
    morph_kernel: int   = 3


@dataclass
class SynthesisConfig:
    alpha:   float = 0.5     # Вес танграма vs фрактала
    n_sides: int   = 4       # Ожидаемое число краёв на фрагмент
    n_points: int  = 128     # Точек дискретизации в кривой края


@dataclass
class FractalConfig:
    n_scales:      int   = 8     # Число масштабов (Box-counting, Divider)
    ifs_transforms: int  = 8     # Число IFS-преобразований
    css_n_sigmas:  int   = 7     # Число масштабов для CSS
    css_n_bins:    int   = 32    # Число бинов в CSS-гистограмме


@dataclass
class MatchingConfig:
    threshold:       float = 0.3    # Минимальный score для включения в матрицу
    dtw_window:      int   = 20     # Ширина окна Сакое-Чибы
    active_matchers: List[str] = field(
        default_factory=lambda: ["css", "dtw", "fd", "text"]
    )  # Список активных матчеров
    matcher_weights: Dict[str, float] = field(
        default_factory=lambda: {"css": 0.35, "dtw": 0.30, "fd": 0.20, "text": 0.15}
    )  # Веса матчеров (сумма = 1.0)
    combine_method:  Literal["weighted", "rank", "min", "max"] = "weighted"


@dataclass
class AssemblyConfig:
    method:      Literal[
        "greedy", "sa", "beam", "gamma",
        "genetic", "exhaustive", "ant_colony", "mcts",
        "auto", "all"
    ] = "beam"
    beam_width:      int   = 10
    sa_iter:         int   = 5000
    sa_T_max:        float = 1000.0
    sa_T_min:        float = 0.1
    sa_cooling:      float = 0.995
    gamma_iter:      int   = 3000     # Итераций для гамма-оптимизатора
    seed:            int   = 42
    # Параметры алгоритмов, ранее не подключённых к CLI:
    genetic_pop:     int   = 50       # Размер популяции (genetic)
    genetic_gen:     int   = 100      # Число поколений (genetic)
    aco_ants:        int   = 20       # Число агентов-муравьёв (ant_colony)
    aco_iter:        int   = 100      # Итерации ACO (ant_colony)
    mcts_sim:        int   = 200      # Симуляции MCTS (mcts)
    exhaustive_max_n: int  = 9        # Макс. N для exact search (exhaustive)
    auto_timeout:    float = 60.0     # Таймаут на метод в auto/all режиме


@dataclass
class VerificationConfig:
    run_ocr:     bool  = True
    ocr_lang:    str   = "rus+eng"
    export_pdf:  bool  = False


@dataclass
class Config:
    """Корневой конфиг — объединяет все секции."""
    segmentation:  SegmentationConfig  = field(default_factory=SegmentationConfig)
    synthesis:     SynthesisConfig     = field(default_factory=SynthesisConfig)
    fractal:       FractalConfig       = field(default_factory=FractalConfig)
    matching:      MatchingConfig      = field(default_factory=MatchingConfig)
    assembly:      AssemblyConfig      = field(default_factory=AssemblyConfig)
    verification:  VerificationConfig  = field(default_factory=VerificationConfig)

    # ── Сериализация ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(
            segmentation = SegmentationConfig(**d.get("segmentation", {})),
            synthesis    = SynthesisConfig(**d.get("synthesis", {})),
            fractal      = FractalConfig(**d.get("fractal", {})),
            matching     = MatchingConfig(**d.get("matching", {})),
            assembly     = AssemblyConfig(**d.get("assembly", {})),
            verification = VerificationConfig(**d.get("verification", {})),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """Загружает конфиг из JSON или YAML файла."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл конфига не найден: {path}")

        text = path.read_text(encoding="utf-8")

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                d = yaml.safe_load(text)
            except ImportError:
                raise ImportError("pip install pyyaml для поддержки YAML")
        else:
            d = json.loads(text)

        return cls.from_dict(d or {})

    @classmethod
    def default(cls) -> "Config":
        return cls()

    def apply_overrides(self, **kwargs) -> "Config":
        """
        Применяет переопределения из аргументов командной строки.
        kwargs: плоский словарь {секция__поле: значение} или известные поля.

        Пример:
            cfg.apply_overrides(alpha=0.7, method="sa", sa_iter=3000)
        """
        mapping = {
            "alpha":           ("synthesis",    "alpha"),
            "n_sides":         ("synthesis",    "n_sides"),
            "seg_method":      ("segmentation", "method"),
            "threshold":       ("matching",     "threshold"),
            "method":          ("assembly",     "method"),
            "beam_width":      ("assembly",     "beam_width"),
            "sa_iter":         ("assembly",     "sa_iter"),
            "seed":            ("assembly",     "seed"),
            "genetic_pop":     ("assembly",     "genetic_pop"),
            "genetic_gen":     ("assembly",     "genetic_gen"),
            "aco_ants":        ("assembly",     "aco_ants"),
            "aco_iter":        ("assembly",     "aco_iter"),
            "mcts_sim":        ("assembly",     "mcts_sim"),
            "auto_timeout":    ("assembly",     "auto_timeout"),
        }
        for key, value in kwargs.items():
            if value is None:
                continue
            if key in mapping:
                section_name, field_name = mapping[key]
                section = getattr(self, section_name)
                setattr(section, field_name, value)
        return self
