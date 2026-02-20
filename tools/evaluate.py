#!/usr/bin/env python3
"""
Инструмент оценки качества системы восстановления документов.

Запускает полный цикл:
    1. Генерация синтетических документов (known ground truth)
    2. Разрыв на фрагменты
    3. Прогон каждого метода сборки
    4. Измерение NA, DC, RMSE и других метрик
    5. Генерация HTML/JSON/Markdown отчёта

Использование:
    python tools/evaluate.py
    python tools/evaluate.py --pieces 4 6 --methods beam gamma --trials 5
    python tools/evaluate.py --output reports/ --html --markdown

Выходные файлы:
    reports/evaluation.json     — полные результаты в JSON
    reports/evaluation.html     — красивый HTML-отчёт с таблицами
    reports/evaluation.md       — Markdown для GitHub
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cv2
except ImportError:
    print("Нужен opencv-python: pip install opencv-python")
    sys.exit(1)

from tools.tear_generator import tear_document, generate_test_document
from puzzle_reconstruction.config import Config
from puzzle_reconstruction.pipeline import Pipeline
from puzzle_reconstruction.verification.metrics import (
    evaluate_reconstruction, compare_methods, BenchmarkResult,
    ReconstructionMetrics,
)
from puzzle_reconstruction.verification.report import build_report
from puzzle_reconstruction.export import render_canvas, render_heatmap, render_mosaic
from puzzle_reconstruction.utils.logger import get_logger, PipelineTimer

log = get_logger("evaluate")


# ─── Параметры по умолчанию ──────────────────────────────────────────────

DEFAULT_METHODS = ["greedy", "beam", "gamma"]
DEFAULT_PIECES  = [4, 6]
DEFAULT_TRIALS  = 3


# ─── Генерация GT и фрагментов ───────────────────────────────────────────

def generate_test_case(n_pieces: int,
                        noise: float,
                        doc_seed: int,
                        tear_seed: int):
    """
    Генерирует документ, рвёт, возвращает (images, gt_placements).
    """
    doc   = generate_test_document(800, 1000, seed=doc_seed)
    h, w  = doc.shape[:2]

    from tools.tear_generator import _grid_shape, _divide_with_jitter, _torn_mask
    rng_s = np.random.RandomState(tear_seed)
    cols, rows = _grid_shape(n_pieces)
    col_bounds = _divide_with_jitter(w, cols, rng_s, 0.15)
    row_bounds = _divide_with_jitter(h, rows, rng_s, 0.15)

    images, gt = [], {}
    fid = 0
    for row in range(rows):
        for col in range(cols):
            x0, x1 = col_bounds[col], col_bounds[col + 1]
            y0, y1 = row_bounds[row], row_bounds[row + 1]
            mask = _torn_mask(h, w, x0, x1, y0, y1, noise_level=noise, rng=rng_s)
            ys, xs = np.where(mask > 0)
            if len(xs) < 10:
                continue
            pad = 10
            x_lo = max(0, xs.min() - pad);  x_hi = min(w, xs.max() + pad)
            y_lo = max(0, ys.min() - pad);  y_hi = min(h, ys.max() + pad)
            frag = np.full_like(doc, 255)
            frag[mask > 0] = doc[mask > 0]
            frag = frag[y_lo:y_hi, x_lo:x_hi]
            images.append(frag)
            gt[fid] = (np.array([(x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0]), 0.0)
            fid += 1

    return images, gt


# ─── Один прогон ─────────────────────────────────────────────────────────

def run_one(method: str, images: list, gt: dict, cfg: Config):
    """Запускает один метод, возвращает (BenchmarkResult, Assembly)."""
    cfg.assembly.method = method
    pipeline = Pipeline(cfg=cfg, n_workers=4)

    t0       = time.perf_counter()
    result   = pipeline.run(images)
    elapsed  = time.perf_counter() - t0

    assembly = result.assembly
    metrics  = evaluate_reconstruction(assembly.placements, gt)

    br = BenchmarkResult(
        method=method,
        metrics=metrics,
        runtime_sec=elapsed,
    )
    return br, assembly


# ─── Полная оценка ───────────────────────────────────────────────────────

def run_evaluation(methods: List[str],
                    n_pieces_list: List[int],
                    n_trials: int,
                    noise: float,
                    output_dir: Path,
                    save_html: bool,
                    save_md: bool) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.default()
    cfg.assembly.beam_width  = 8
    cfg.assembly.sa_iter     = 1500
    cfg.assembly.gamma_iter  = 1000
    cfg.verification.run_ocr = False

    all_results: Dict[str, Dict[int, List[BenchmarkResult]]] = {
        m: {n: [] for n in n_pieces_list} for m in methods
    }

    timer = PipelineTimer()

    log.info("=" * 60)
    log.info("ОЦЕНКА СИСТЕМЫ ВОССТАНОВЛЕНИЯ ДОКУМЕНТОВ")
    log.info(f"Методы: {', '.join(methods)}")
    log.info(f"Размеры: {n_pieces_list}  |  Прогонов: {n_trials}")
    log.info("=" * 60)

    # Хранение сборки последнего прогона для отчёта
    last_assembly = None
    last_metrics  = None

    for n_pieces in n_pieces_list:
        log.info(f"\n[{n_pieces} фрагментов]")

        for trial in range(n_trials):
            log.info(f"  Прогон {trial + 1}/{n_trials}...")

            with timer.measure(f"{n_pieces}шт_прогон{trial}"):
                images, gt = generate_test_case(
                    n_pieces=n_pieces,
                    noise=noise,
                    doc_seed=trial,
                    tear_seed=trial + 50,
                )

            for method in methods:
                log.info(f"    {method}: ", )
                with timer.measure(f"{method}_{n_pieces}_{trial}"):
                    try:
                        br, asm = run_one(method, images, gt, cfg)
                        all_results[method][n_pieces].append(br)
                        last_assembly = asm
                        last_metrics  = br.metrics
                        log.info(
                            f"      NA={br.metrics.neighbor_accuracy:.1%}  "
                            f"DC={br.metrics.direct_comparison:.1%}  "
                            f"t={br.runtime_sec:.1f}с"
                        )
                    except Exception as e:
                        log.error(f"      Ошибка: {e}")

    # ── Вывод сводной таблицы ─────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("РЕЗУЛЬТАТЫ")

    all_br_by_method: Dict[str, BenchmarkResult] = {}

    for n_pieces in n_pieces_list:
        log.info(f"\n── {n_pieces} фрагментов ──")
        summary_list = []
        for method in methods:
            trials = all_results[method][n_pieces]
            if not trials:
                continue
            avg_na  = float(np.mean([r.metrics.neighbor_accuracy for r in trials]))
            avg_dc  = float(np.mean([r.metrics.direct_comparison for r in trials]))
            avg_pos = float(np.mean([r.metrics.position_rmse for r in trials]))
            avg_ang = float(np.mean([r.metrics.angular_error_deg for r in trials]))
            avg_t   = float(np.mean([r.runtime_sec for r in trials]))
            n_perf  = sum(1 for r in trials if r.metrics.perfect)

            avg_m = ReconstructionMetrics(
                neighbor_accuracy=avg_na, direct_comparison=avg_dc,
                perfect=(n_perf == len(trials)), position_rmse=avg_pos,
                angular_error_deg=avg_ang, n_fragments=n_pieces,
                n_correct_pairs=0, n_total_pairs=0, edge_match_rate=0.0,
            )
            avg_br = BenchmarkResult(method=method, metrics=avg_m, runtime_sec=avg_t)
            summary_list.append(avg_br)
            all_br_by_method[method] = avg_br

        print(compare_methods(summary_list))

    log.info("\n" + timer.report())

    # ── Сохранение JSON ───────────────────────────────────────────────────
    json_path = output_dir / "evaluation.json"
    serializable = {}
    for method, pieces_map in all_results.items():
        serializable[method] = {}
        for n_pieces, trials in pieces_map.items():
            serializable[method][str(n_pieces)] = [
                {"na": r.metrics.neighbor_accuracy,
                 "dc": r.metrics.direct_comparison,
                 "rmse": r.metrics.position_rmse,
                 "angle": r.metrics.angular_error_deg,
                 "perfect": r.metrics.perfect,
                 "t": r.runtime_sec}
                for r in trials
            ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    log.info(f"\nJSON: {json_path}")

    # ── Генерация отчёта ──────────────────────────────────────────────────
    if last_assembly is not None and (save_html or save_md):
        canvas  = render_canvas(last_assembly)
        heatmap = render_heatmap(last_assembly, canvas.shape)
        mosaic  = render_mosaic(last_assembly)

        methods_summary = "\n".join(
            f"  {method}: NA={br.metrics.neighbor_accuracy:.1%}, "
            f"DC={br.metrics.direct_comparison:.1%}"
            for method, br in all_br_by_method.items()
        )
        notes = f"Методы сравнения:\n{methods_summary}"

        report = build_report(
            last_assembly,
            metrics=last_metrics,
            notes=notes,
            canvas=canvas,
            heatmap=heatmap,
            mosaic=mosaic,
        )

        if save_html:
            html_path = output_dir / "evaluation.html"
            report.save_html(html_path)
            log.info(f"HTML: {html_path}")

        if save_md:
            md_path = output_dir / "evaluation.md"
            report.save_markdown(md_path)
            log.info(f"Markdown: {md_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Оценка качества восстановления документов",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pieces",  "-n", type=int, nargs="+",
                         default=DEFAULT_PIECES,
                         help="Число фрагментов")
    parser.add_argument("--methods", "-m", nargs="+",
                         default=DEFAULT_METHODS,
                         choices=["greedy", "sa", "beam", "gamma", "exhaustive"],
                         help="Методы для сравнения")
    parser.add_argument("--trials",  "-t", type=int, default=DEFAULT_TRIALS,
                         help="Прогонов для усреднения")
    parser.add_argument("--noise",   type=float, default=0.5,
                         help="Интенсивность рваного края")
    parser.add_argument("--output",  "-o", default="reports",
                         help="Директория для выходных файлов")
    parser.add_argument("--html",    action="store_true",
                         help="Генерировать HTML-отчёт")
    parser.add_argument("--markdown", action="store_true",
                         help="Генерировать Markdown-отчёт")
    args = parser.parse_args()

    run_evaluation(
        methods      = args.methods,
        n_pieces_list = args.pieces,
        n_trials     = args.trials,
        noise        = args.noise,
        output_dir   = Path(args.output),
        save_html    = args.html,
        save_md      = args.markdown,
    )


if __name__ == "__main__":
    main()
