"""
Профилировщик производительности пайплайна восстановления документов.

Инструмент измеряет время выполнения каждого этапа на синтетических данных
и выводит сводную таблицу. При --cprofile выполняет полное функциональное
профилирование с помощью cProfile + pstats.

Использование:
    python tools/profile.py --fragments 8 --iters 3
    python tools/profile.py --fragments 16 --cprofile --cprofile-out profile.stats
    python tools/profile.py --fragments 8 --json profile_results.json
    python tools/profile.py --help

Этапы, замеряемые профилировщиком:
    segmentation   — выделение маски каждого фрагмента
    denoise        — шумоподавление (auto_denoise)
    color_norm     — нормализация цвета
    descriptor     — вычисление FD + CSS + IFS + Tangram
    synthesis      — синтез EdgeSignature из двух описаний
    compat_matrix  — построение матрицы совместимости N_edges × N_edges
    assembly_*     — сборка каждым из методов (greedy/sa/beam/gamma/genetic)
    total          — суммарное время пайплайна
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import math
import os
import pstats
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Dict, Generator, List, Optional

import numpy as np

# Добавляем корень проекта в sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puzzle_reconstruction.models import (
    Fragment, Assembly, CompatEntry, EdgeSignature, EdgeSide,
    FractalSignature, TangramSignature, ShapeClass,
)
from puzzle_reconstruction.preprocessing.denoise import auto_denoise
from puzzle_reconstruction.preprocessing.color_norm import normalize_color
from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
from puzzle_reconstruction.assembly.greedy import greedy_assembly
from puzzle_reconstruction.assembly.annealing import simulated_annealing
from puzzle_reconstruction.assembly.beam_search import beam_search
from puzzle_reconstruction.assembly.gamma_optimizer import gamma_optimizer
from puzzle_reconstruction.assembly.genetic import genetic_assembly


# ─── Таймер ───────────────────────────────────────────────────────────────────

@dataclass
class StageTiming:
    stage:   str
    elapsed: float   # секунды
    count:   int     # число обработанных единиц (фрагментов, рёбер и т.п.)

    @property
    def per_unit(self) -> float:
        return self.elapsed / self.count if self.count > 0 else 0.0


@dataclass
class ProfileResult:
    n_fragments:  int
    n_edges:      int
    timings:      List[StageTiming] = field(default_factory=list)
    errors:       List[str]         = field(default_factory=list)

    def total(self) -> float:
        return sum(t.elapsed for t in self.timings)

    def to_dict(self) -> dict:
        return {
            "n_fragments":  self.n_fragments,
            "n_edges":      self.n_edges,
            "timings":      [asdict(t) for t in self.timings],
            "total_sec":    self.total(),
            "errors":       self.errors,
        }

    def table(self) -> str:
        """Форматированная таблица результатов."""
        lines = [
            "╔══════════════════════════════╦══════════╦══════════╦════════════╗",
            "║ Этап                         ║   Сек    ║  Единиц  ║ Сек/ед     ║",
            "╠══════════════════════════════╬══════════╬══════════╬════════════╣",
        ]
        for t in self.timings:
            lines.append(
                f"║ {t.stage:<28} ║ {t.elapsed:>8.3f} ║ {t.count:>8} "
                f"║ {t.per_unit:>10.4f} ║"
            )
        lines += [
            "╠══════════════════════════════╬══════════╬══════════╬════════════╣",
            f"║ {'ИТОГО':<28} ║ {self.total():>8.3f} ║ {'':<8} ║ {'':<10} ║",
            "╚══════════════════════════════╩══════════╩══════════╩════════════╝",
        ]
        return "\n".join(lines)


# ─── Синтетические фрагменты ──────────────────────────────────────────────────

def _make_synthetic_fragments(n: int, size: int = 128) -> List[Fragment]:
    """Создаёт N синтетических фрагментов с заполненными edge-дескрипторами."""
    rng = np.random.RandomState(42)
    fragments = []

    for i in range(n):
        # Изображение с «текстом»: белый фон + серый прямоугольник
        img = np.ones((size, size, 3), dtype=np.uint8) * 230
        img[20:size-20, 20:size-20] = (200, 200, 200)

        # Простая прямоугольная маска
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[5:size-5, 5:size-5] = 255

        # Контур прямоугольника (40 точек)
        h, w = size - 10, size - 10
        pts = []
        pts += [(5 + x, 5)     for x in np.linspace(0, w, 10)]
        pts += [(5 + w, 5 + y) for y in np.linspace(0, h, 10)]
        pts += [(5 + w - x, 5 + h) for x in np.linspace(0, w, 10)]
        pts += [(5, 5 + h - y) for y in np.linspace(0, h, 10)]
        contour = np.array(pts, dtype=np.float64)

        frag = Fragment(
            fragment_id=i,
            image=img,
            mask=mask,
            contour=contour,
            tangram=TangramSignature(
                polygon=np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float64),
                shape_class=ShapeClass.RECTANGLE,
                centroid=np.array([0.5, 0.5]),
                angle=0.0,
                scale=1.0,
                area=1.0,
            ),
            fractal=FractalSignature(
                fd_box=1.2 + rng.rand() * 0.2,
                fd_divider=1.1 + rng.rand() * 0.2,
                ifs_coeffs=rng.rand(6),
                css_image=[],
                chain_code="00001111222233334444",
                curve=contour[:20],
            ),
        )

        # Создаём рёбра (4 стороны)
        n_pts = 20
        sides = [EdgeSide.TOP, EdgeSide.RIGHT, EdgeSide.BOTTOM, EdgeSide.LEFT]
        for j, side in enumerate(sides):
            vc = rng.rand(n_pts, 2) * size
            frag.edges.append(EdgeSignature(
                edge_id=i * 4 + j,
                side=side,
                virtual_curve=vc,
                fd=1.2 + rng.rand() * 0.2,
                css_vec=rng.rand(8),
                ifs_coeffs=rng.rand(6),
                length=float(size),
            ))

        fragments.append(frag)

    return fragments


# ─── Профилирование ───────────────────────────────────────────────────────────

@contextmanager
def _timed(label: str,
           results: ProfileResult,
           count: int = 1) -> Generator[None, None, None]:
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    results.timings.append(StageTiming(stage=label, elapsed=elapsed, count=count))


def run_profile(n_fragments: int = 8,
                n_iters: int = 1,
                verbose: bool = True) -> ProfileResult:
    """
    Запускает полный пайплайн и измеряет каждый этап.

    Args:
        n_fragments: Число синтетических фрагментов.
        n_iters:     Число повторений для усреднения (среднее не считается,
                      суммируются).
        verbose:     Печатать прогресс в stdout.

    Returns:
        ProfileResult с итогами.
    """
    result = ProfileResult(n_fragments=n_fragments, n_edges=n_fragments * 4)

    for iteration in range(n_iters):
        if verbose and n_iters > 1:
            print(f"  [итерация {iteration + 1}/{n_iters}]")

        fragments = _make_synthetic_fragments(n_fragments)

        # ── Шумоподавление ───────────────────────────────────────────────
        with _timed("denoise", result, count=n_fragments):
            for frag in fragments:
                frag.image = auto_denoise(frag.image)

        # ── Нормализация цвета ───────────────────────────────────────────
        with _timed("color_norm", result, count=n_fragments):
            for frag in fragments:
                frag.image = normalize_color(frag.image)

        # ── Матрица совместимости ────────────────────────────────────────
        all_edges = [e for f in fragments for e in f.edges]
        with _timed("compat_matrix", result, count=len(all_edges)):
            try:
                entries = build_compat_matrix(fragments)
            except Exception as exc:
                result.errors.append(f"compat_matrix: {exc}")
                # Синтетические entries для продолжения
                rng = np.random.RandomState(0)
                entries = []
                for fi in fragments:
                    for fj in fragments:
                        if fi.fragment_id >= fj.fragment_id:
                            continue
                        ei = fi.edges[0]
                        ej = fj.edges[0]
                        entries.append(CompatEntry(
                            edge_i=ei, edge_j=ej,
                            score=float(rng.rand()),
                            dtw_dist=0.0, css_sim=0.0,
                            fd_diff=0.0, text_score=0.0,
                        ))

        # ── Greedy ───────────────────────────────────────────────────────
        with _timed("assembly_greedy", result, count=n_fragments):
            try:
                greedy_assembly(fragments, entries)
            except Exception as exc:
                result.errors.append(f"greedy: {exc}")

        # ── Simulated Annealing ──────────────────────────────────────────
        with _timed("assembly_sa", result, count=n_fragments):
            try:
                simulated_annealing(fragments, entries,
                                     n_iter=500, seed=42)
            except Exception as exc:
                result.errors.append(f"sa: {exc}")

        # ── Beam Search ──────────────────────────────────────────────────
        with _timed("assembly_beam", result, count=n_fragments):
            try:
                beam_search(fragments, entries, beam_width=5)
            except Exception as exc:
                result.errors.append(f"beam: {exc}")

        # ── Gamma Optimizer ──────────────────────────────────────────────
        with _timed("assembly_gamma", result, count=n_fragments):
            try:
                gamma_optimizer(fragments, entries, n_iter=300, seed=42)
            except Exception as exc:
                result.errors.append(f"gamma: {exc}")

        # ── Genetic Algorithm ────────────────────────────────────────────
        with _timed("assembly_genetic", result, count=n_fragments):
            try:
                genetic_assembly(fragments, entries,
                                   population_size=20, n_generations=50,
                                   seed=42)
            except Exception as exc:
                result.errors.append(f"genetic: {exc}")

    return result


def run_cprofile(n_fragments: int,
                  stats_path: Optional[str] = None,
                  top_n: int = 20) -> None:
    """Запускает пайплайн под cProfile и печатает топ функций."""
    pr = cProfile.Profile()
    pr.enable()

    fragments = _make_synthetic_fragments(n_fragments)
    rng = np.random.RandomState(0)
    entries: List[CompatEntry] = []
    for fi in fragments:
        for fj in fragments:
            if fi.fragment_id >= fj.fragment_id:
                continue
            entries.append(CompatEntry(
                edge_i=fi.edges[0], edge_j=fj.edges[0],
                score=float(rng.rand()),
                dtw_dist=0.0, css_sim=0.0,
                fd_diff=0.0, text_score=0.0,
            ))

    greedy_assembly(fragments, entries)
    simulated_annealing(fragments, entries, n_iter=200, seed=0)
    genetic_assembly(fragments, entries, population_size=15, n_generations=30)

    pr.disable()

    if stats_path:
        pr.dump_stats(stats_path)
        print(f"Статистика cProfile сохранена: {stats_path}")

    # Вывод топ функций
    buf  = io.StringIO()
    ps   = pstats.Stats(pr, stream=buf).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(top_n)
    print(buf.getvalue())


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Профилировщик производительности пайплайна",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--fragments", type=int, default=8,
                   help="Число синтетических фрагментов (default: 8)")
    p.add_argument("--iters", type=int, default=1,
                   help="Число повторений замеров (default: 1)")
    p.add_argument("--json", metavar="FILE",
                   help="Сохранить результаты в JSON-файл")
    p.add_argument("--cprofile", action="store_true",
                   help="Дополнительно запустить cProfile")
    p.add_argument("--cprofile-out", metavar="FILE", default=None,
                   help="Путь для сохранения .stats файла cProfile")
    p.add_argument("--top", type=int, default=20,
                   help="Число функций в отчёте cProfile (default: 20)")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="Минимальный вывод")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    verbose = not args.quiet

    if verbose:
        print(f"\nПрофилирование пайплайна: {args.fragments} фрагментов, "
              f"{args.iters} повторений\n")

    result = run_profile(
        n_fragments=args.fragments,
        n_iters=args.iters,
        verbose=verbose,
    )

    if verbose:
        print()
        print(result.table())

        if result.errors:
            print("\nОшибки при профилировании:")
            for err in result.errors:
                print(f"  ✗ {err}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\nРезультаты сохранены: {args.json}")

    if args.cprofile:
        print("\n=== cProfile ===")
        run_cprofile(
            n_fragments=args.fragments,
            stats_path=args.cprofile_out,
            top_n=args.top,
        )


if __name__ == "__main__":
    main()
