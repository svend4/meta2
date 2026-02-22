#!/usr/bin/env python3
"""
Автоматический бенчмарк системы восстановления пазлов.

Генерирует синтетические документы, рвёт их на фрагменты,
запускает каждый метод сборки и сравнивает с ground-truth.

Использование:
    python tools/benchmark.py
    python tools/benchmark.py --pieces 4 6 9 --methods greedy sa beam gamma
    python tools/benchmark.py --trials 5 --output results.json

Выходные данные:
    - Таблица сравнения методов в консоли
    - JSON-файл с полными результатами (если --output задан)
"""
import argparse
import json
import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cv2
except ImportError:
    print("Нужен opencv-python: pip install opencv-python")
    sys.exit(1)

from tools.tear_generator import tear_document, generate_test_document
from puzzle_reconstruction.models import Fragment
from puzzle_reconstruction.preprocessing.segmentation import segment_fragment
from puzzle_reconstruction.preprocessing.contour import extract_contour
from puzzle_reconstruction.algorithms.tangram.inscriber import fit_tangram
from puzzle_reconstruction.algorithms.synthesis import (
    compute_fractal_signature, build_edge_signatures
)
from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
from puzzle_reconstruction.assembly.greedy import greedy_assembly
from puzzle_reconstruction.assembly.annealing import simulated_annealing
from puzzle_reconstruction.assembly.beam_search import beam_search
from puzzle_reconstruction.assembly.gamma_optimizer import gamma_optimizer
from puzzle_reconstruction.verification.metrics import (
    evaluate_reconstruction, compare_methods, BenchmarkResult
)
from puzzle_reconstruction.config import Config
from puzzle_reconstruction.utils.logger import get_logger, PipelineTimer

log = get_logger("benchmark")


# ─── Методы сборки ────────────────────────────────────────────────────────

ASSEMBLY_METHODS = {
    "greedy": lambda frags, entries, cfg: greedy_assembly(frags, entries),
    "sa":     lambda frags, entries, cfg: simulated_annealing(
                  greedy_assembly(frags, entries), entries,
                  T_max=cfg.assembly.sa_T_max,
                  max_iter=cfg.assembly.sa_iter,
                  seed=cfg.assembly.seed),
    "beam":   lambda frags, entries, cfg: beam_search(
                  frags, entries,
                  beam_width=cfg.assembly.beam_width),
    "gamma":  lambda frags, entries, cfg: gamma_optimizer(
                  frags, entries,
                  n_iter=cfg.assembly.gamma_iter,
                  seed=cfg.assembly.seed),
}


# ─── Один прогон бенчмарка ────────────────────────────────────────────────

def run_trial(n_pieces: int,
              methods: List[str],
              cfg: Config,
              noise: float = 0.5,
              doc_seed: int = 0,
              tear_seed: int = 42) -> Dict[str, BenchmarkResult]:
    """
    Один прогон бенчмарка: генерирует документ, рвёт, собирает каждым методом.

    Returns:
        {method_name: BenchmarkResult}
    """
    log.info(f"  Генерация документа ({n_pieces} фрагментов, noise={noise})...")
    document = generate_test_document(800, 1000, seed=doc_seed)

    fragments_raw, gt_placements = _tear_and_track(document, n_pieces, noise, tear_seed)
    if not fragments_raw:
        log.warning("  Нет фрагментов — пропускаю прогон")
        return {}

    log.info(f"  Обработка {len(fragments_raw)} фрагментов...")
    fragments = _process_fragments(fragments_raw, cfg)
    if not fragments:
        log.warning("  Не удалось обработать фрагменты")
        return {}

    log.info("  Построение матрицы совместимости...")
    _, entries = build_compat_matrix(fragments, threshold=cfg.matching.threshold)

    results = {}
    for method in methods:
        if method not in ASSEMBLY_METHODS:
            log.warning(f"  Неизвестный метод: {method}")
            continue

        log.info(f"  Метод: {method}...")
        t0 = time.perf_counter()
        try:
            assembly = ASSEMBLY_METHODS[method](fragments, entries, cfg)
        except Exception as e:
            log.error(f"  Ошибка в методе {method}: {e}")
            continue
        elapsed = time.perf_counter() - t0

        metrics = evaluate_reconstruction(assembly.placements, gt_placements)
        results[method] = BenchmarkResult(
            method=method,
            metrics=metrics,
            runtime_sec=elapsed,
        )
        log.info(f"    NA={metrics.neighbor_accuracy:.1%}  "
                  f"DC={metrics.direct_comparison:.1%}  "
                  f"t={elapsed:.2f}с")

    return results


# ─── Полный бенчмарк ──────────────────────────────────────────────────────

def run_benchmark(n_pieces_list: List[int],
                  methods: List[str],
                  n_trials: int = 3,
                  noise: float = 0.5,
                  output_path: str = None) -> None:

    cfg = Config.default()
    cfg.assembly.beam_width = 8
    cfg.assembly.sa_iter    = 2000
    cfg.assembly.gamma_iter = 1500

    all_results: Dict[str, Dict[int, List[BenchmarkResult]]] = {
        m: {n: [] for n in n_pieces_list} for m in methods
    }

    timer = PipelineTimer()

    log.info("=" * 60)
    log.info("БЕНЧМАРК СИСТЕМЫ ВОССТАНОВЛЕНИЯ ПАЗЛОВ")
    log.info(f"Методы: {', '.join(methods)}")
    log.info(f"Размеры: {n_pieces_list}  |  Прогонов: {n_trials}")
    log.info("=" * 60)

    for n_pieces in n_pieces_list:
        log.info(f"\n[{n_pieces} фрагментов]")

        for trial in range(n_trials):
            log.info(f"  Прогон {trial + 1}/{n_trials}")
            with timer.measure(f"{n_pieces}пц прогон{trial}"):
                trial_results = run_trial(
                    n_pieces=n_pieces,
                    methods=methods,
                    cfg=cfg,
                    noise=noise,
                    doc_seed=trial,
                    tear_seed=trial + 100,
                )
            for method, result in trial_results.items():
                all_results[method][n_pieces].append(result)

    # ── Вывод сводных результатов ──────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("СВОДНЫЕ РЕЗУЛЬТАТЫ")
    log.info("=" * 60)

    for n_pieces in n_pieces_list:
        log.info(f"\n── {n_pieces} фрагментов ──")
        summary_results = []
        for method in methods:
            trial_list = all_results[method][n_pieces]
            if not trial_list:
                continue
            avg_na  = np.mean([r.metrics.neighbor_accuracy for r in trial_list])
            avg_dc  = np.mean([r.metrics.direct_comparison for r in trial_list])
            avg_pos = np.mean([r.metrics.position_rmse for r in trial_list])
            avg_ang = np.mean([r.metrics.angular_error_deg for r in trial_list])
            avg_t   = np.mean([r.runtime_sec for r in trial_list])
            n_perf  = sum(1 for r in trial_list if r.metrics.perfect)

            # Берём «представителя» для compare_methods
            from puzzle_reconstruction.verification.metrics import ReconstructionMetrics
            avg_metrics = ReconstructionMetrics(
                neighbor_accuracy=avg_na,
                direct_comparison=avg_dc,
                perfect=(n_perf == len(trial_list)),
                position_rmse=avg_pos,
                angular_error_deg=avg_ang,
                n_fragments=n_pieces,
                n_correct_pairs=0,
                n_total_pairs=0,
                edge_match_rate=0.0,
            )
            summary_results.append(BenchmarkResult(
                method=method,
                metrics=avg_metrics,
                runtime_sec=avg_t,
            ))

        print(compare_methods(summary_results))

    log.info("\n" + timer.report())

    # ── Сохранение JSON ───────────────────────────────────────────────────
    if output_path:
        _save_json(all_results, output_path)
        log.info(f"\nРезультаты сохранены: {output_path}")


def _save_json(all_results: dict, path: str) -> None:
    """Сериализует результаты бенчмарка в JSON."""
    serializable = {}
    for method, pieces_map in all_results.items():
        serializable[method] = {}
        for n_pieces, trials in pieces_map.items():
            serializable[method][str(n_pieces)] = [
                {
                    "na":          r.metrics.neighbor_accuracy,
                    "dc":          r.metrics.direct_comparison,
                    "rmse":        r.metrics.position_rmse,
                    "angle_err":   r.metrics.angular_error_deg,
                    "perfect":     r.metrics.perfect,
                    "runtime_sec": r.runtime_sec,
                }
                for r in trials
            ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


# ─── Генерация и отслеживание GT ──────────────────────────────────────────

def _tear_and_track(document: np.ndarray,
                     n_pieces: int,
                     noise: float,
                     seed: int):
    """
    Рвёт документ и возвращает фрагменты + ground-truth позиции.

    Ground truth: центр каждого фрагмента в исходном документе.
    """
    from tools.tear_generator import _grid_shape, _divide_with_jitter, _torn_mask
    rng_state = np.random.RandomState(seed)
    h, w = document.shape[:2]
    cols, rows = _grid_shape(n_pieces)
    col_bounds = _divide_with_jitter(w, cols, rng_state, 0.15)
    row_bounds = _divide_with_jitter(h, rows, rng_state, 0.15)

    fragments_raw = []
    gt_placements = {}

    fid = 0
    for row in range(rows):
        for col in range(cols):
            x0, x1 = col_bounds[col], col_bounds[col + 1]
            y0, y1 = row_bounds[row], row_bounds[row + 1]

            mask = _torn_mask(h, w, x0, x1, y0, y1,
                               noise_level=noise, rng=rng_state)
            ys, xs = np.where(mask > 0)
            if len(xs) < 10:
                continue

            pad = 10
            x_lo, x_hi = max(0, xs.min() - pad), min(w, xs.max() + pad)
            y_lo, y_hi = max(0, ys.min() - pad), min(h, ys.max() + pad)

            fragment = np.full_like(document, 255)
            fragment[mask > 0] = document[mask > 0]
            fragment = fragment[y_lo:y_hi, x_lo:x_hi]

            fragments_raw.append(fragment)
            # GT: центр фрагмента в пространстве оригинала
            gt_placements[fid] = (
                np.array([(x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0]),
                0.0
            )
            fid += 1

    return fragments_raw, gt_placements


def _process_fragments(images: list, cfg: Config) -> List[Fragment]:
    """Сегментирует и описывает фрагменты."""
    fragments = []
    for idx, img in enumerate(images):
        try:
            mask    = segment_fragment(img, method=cfg.segmentation.method)
            contour = extract_contour(mask)
            tangram = fit_tangram(contour)
            fractal = compute_fractal_signature(contour)

            frag = Fragment(fragment_id=idx, image=img, mask=mask, contour=contour)
            frag.tangram = tangram
            frag.fractal = fractal
            frag.edges   = build_edge_signatures(frag,
                                                  alpha=cfg.synthesis.alpha,
                                                  n_sides=cfg.synthesis.n_sides)
            fragments.append(frag)
        except Exception:
            pass  # Пропускаем фрагменты с ошибками
    return fragments


# ─── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Бенчмарк системы восстановления пазлов",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pieces",  "-n", type=int, nargs="+",
                         default=[4, 6, 9],
                         help="Число фрагментов для тестирования")
    parser.add_argument("--methods", "-m", nargs="+",
                         default=["greedy", "sa", "beam"],
                         choices=["greedy", "sa", "beam", "gamma"],
                         help="Методы сборки для сравнения")
    parser.add_argument("--trials",  "-t", type=int, default=3,
                         help="Число прогонов для усреднения")
    parser.add_argument("--noise",   type=float, default=0.5,
                         help="Интенсивность рваного края [0..1]")
    parser.add_argument("--output",  "-o", default=None,
                         help="Сохранить результаты в JSON-файл")
    args = parser.parse_args()

    run_benchmark(
        n_pieces_list=args.pieces,
        methods=args.methods,
        n_trials=args.trials,
        noise=args.noise,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
