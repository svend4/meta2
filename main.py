#!/usr/bin/env python3
"""
Восстановление разорванного документа из отсканированных фрагментов.

Использование:
    python main.py --input scans/ --output result.png
    python main.py --input scans/ --output result.png --method beam --beam-width 10
    python main.py --input scans/ --output result.png --config config.json
    python main.py --input scans/ --output result.png --visualize

Методы сборки (--method):
    greedy  — жадный алгоритм (быстрый, < 1 сек)
    sa      — имитация отжига (лучше, ~10–30 сек)
    beam    — beam search (точнее, ~5–20 сек) [по умолчанию]
    gamma   — гамма-оптимизатор, статья 2026 (экспериментальный)

Алгоритм (6 этапов):
    1. Сегментация каждого фрагмента (Otsu / Adaptive / GrabCut)
    2. Описание краёв: Танграм (изнутри) + Фрактальная кромка (снаружи)
    3. Синтез EdgeSignature (виртуальная линия пересечения)
    4. Матрица совместимости всех краёв (CSS + DTW + FD + OCR)
    5. Сборка выбранным методом
    6. OCR-верификация и экспорт
"""
import argparse
import sys
import time
import numpy as np
from pathlib import Path

try:
    import cv2
except ImportError:
    print("Ошибка: opencv-python не установлен. Запустите: pip install opencv-python")
    sys.exit(1)

from puzzle_reconstruction.config import Config
from puzzle_reconstruction.models import Fragment
from puzzle_reconstruction.preprocessing.segmentation import segment_fragment
from puzzle_reconstruction.preprocessing.contour import extract_contour
from puzzle_reconstruction.preprocessing.orientation import (
    estimate_orientation, rotate_to_upright
)
from puzzle_reconstruction.algorithms.tangram.inscriber import fit_tangram
from puzzle_reconstruction.algorithms.synthesis import (
    compute_fractal_signature, build_edge_signatures
)
from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
from puzzle_reconstruction.assembly.greedy import greedy_assembly
from puzzle_reconstruction.assembly.annealing import simulated_annealing
from puzzle_reconstruction.assembly.beam_search import beam_search
from puzzle_reconstruction.assembly.gamma_optimizer import gamma_optimizer
from puzzle_reconstruction.verification.ocr import (
    verify_full_assembly, render_assembly_image
)
from puzzle_reconstruction.utils.logger import get_logger, stage, PipelineTimer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Восстановление разорванного документа",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Основные аргументы
    parser.add_argument("--input",  "-i", required=True,
                        help="Директория с отсканированными фрагментами")
    parser.add_argument("--output", "-o", default="result.png",
                        help="Путь для сохранения результата")

    # Конфигурация
    parser.add_argument("--config", "-c", default=None,
                        help="JSON/YAML файл конфигурации")

    # Метод сборки
    parser.add_argument("--method", "-M", default="beam",
                        choices=["greedy", "sa", "beam", "gamma"],
                        help="Алгоритм сборки")

    # Параметры (перекрывают конфиг)
    parser.add_argument("--alpha",      type=float, default=None,
                        help="Вес танграма в синтезе EdgeSignature (0..1)")
    parser.add_argument("--n-sides",    type=int,   default=None,
                        help="Ожидаемое число краёв на фрагмент")
    parser.add_argument("--seg-method", default=None,
                        choices=["otsu", "adaptive", "grabcut"],
                        help="Метод сегментации")
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Минимальная оценка совместимости")
    parser.add_argument("--beam-width", type=int,   default=None,
                        help="Ширина луча (только для --method beam)")
    parser.add_argument("--sa-iter",    type=int,   default=None,
                        help="Итераций отжига (только для --method sa)")

    # Режимы
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Показать результат в окне OpenCV")
    parser.add_argument("--interactive", "-I", action="store_true",
                        help="Открыть интерактивный редактор сборки")
    parser.add_argument("--verbose",   action="store_true",
                        help="Подробный вывод (DEBUG-уровень)")
    parser.add_argument("--log-file",  default=None,
                        help="Записывать лог в файл")
    return parser


def load_fragments(input_dir: Path, log) -> list[Fragment]:
    exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts)

    if not paths:
        log.error(f"Нет изображений в {input_dir}")
        sys.exit(1)

    log.info(f"Найдено {len(paths)} файлов")
    fragments = []
    for idx, path in enumerate(paths):
        img = cv2.imread(str(path))
        if img is None:
            log.warning(f"  Не удалось загрузить {path.name}")
            continue
        frag = Fragment(fragment_id=idx, image=img, mask=None, contour=None)
        fragments.append(frag)
        log.debug(f"  [{idx:3d}] {path.name}  ({img.shape[1]}×{img.shape[0]})")

    return fragments


def process_fragment(frag: Fragment, cfg: Config, log) -> Fragment:
    """Полная обработка одного фрагмента."""
    # Сегментация
    frag.mask = segment_fragment(frag.image, method=cfg.segmentation.method,
                                  morph_kernel=cfg.segmentation.morph_kernel)

    # Коррекция ориентации
    angle = estimate_orientation(frag.image, frag.mask)
    if abs(angle) > 0.05:
        frag.image = rotate_to_upright(frag.image, angle)
        frag.mask  = segment_fragment(frag.image, method=cfg.segmentation.method)

    # Контур
    frag.contour = extract_contour(frag.mask)

    # Танграм
    frag.tangram = fit_tangram(frag.contour)

    # Фрактал
    frag.fractal = compute_fractal_signature(frag.contour)

    # Синтез подписей краёв
    frag.edges = build_edge_signatures(
        frag,
        alpha=cfg.synthesis.alpha,
        n_sides=cfg.synthesis.n_sides,
        n_points=cfg.synthesis.n_points,
    )
    return frag


def assemble(fragments, entries, cfg: Config, log):
    """Запускает выбранный метод сборки."""
    method = cfg.assembly.method
    log.info(f"  Метод: {method}")

    if method == "greedy":
        return greedy_assembly(fragments, entries)

    elif method == "sa":
        init = greedy_assembly(fragments, entries)
        return simulated_annealing(
            init, entries,
            T_max=cfg.assembly.sa_T_max,
            T_min=cfg.assembly.sa_T_min,
            cooling=cfg.assembly.sa_cooling,
            max_iter=cfg.assembly.sa_iter,
            seed=cfg.assembly.seed,
        )

    elif method == "beam":
        return beam_search(fragments, entries,
                            beam_width=cfg.assembly.beam_width)

    elif method == "gamma":
        init = greedy_assembly(fragments, entries)
        return gamma_optimizer(fragments, entries,
                                n_iter=cfg.assembly.gamma_iter,
                                init_assembly=init,
                                seed=cfg.assembly.seed)
    else:
        log.error(f"Неизвестный метод: {method}")
        sys.exit(1)


def run(args: argparse.Namespace) -> None:
    import logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log = get_logger("puzzle", level=log_level, log_file=args.log_file)

    timer = PipelineTimer()

    # ── Конфигурация ──────────────────────────────────────────────────────
    if args.config:
        log.info(f"Загрузка конфига: {args.config}")
        cfg = Config.from_file(args.config)
    else:
        cfg = Config.default()

    # Применяем CLI-переопределения
    cfg.apply_overrides(
        alpha=args.alpha,
        n_sides=args.n_sides,
        seg_method=args.seg_method,
        threshold=args.threshold,
        method=args.method,
        beam_width=args.beam_width,
        sa_iter=args.sa_iter,
    )

    input_dir   = Path(args.input)
    output_path = Path(args.output)

    log.info("=" * 55)
    log.info("ВОССТАНОВЛЕНИЕ РАЗОРВАННОГО ДОКУМЕНТА")
    log.info(f"Метод: {cfg.assembly.method}  |  α={cfg.synthesis.alpha}")
    log.info("=" * 55)

    # ── Этап 1: Загрузка ──────────────────────────────────────────────────
    with stage("Загрузка фрагментов", log), timer.measure("загрузка"):
        fragments = load_fragments(input_dir, log)

    # ── Этап 2: Обработка ─────────────────────────────────────────────────
    with stage("Обработка фрагментов", log), timer.measure("обработка"):
        processed = []
        for i, frag in enumerate(fragments):
            try:
                frag = process_fragment(frag, cfg, log)
                fd = (frag.fractal.fd_box + frag.fractal.fd_divider) / 2
                log.debug(f"  #{frag.fragment_id:3d}  "
                           f"форма={frag.tangram.shape_class.value:<12}  "
                           f"FD={fd:.3f}  "
                           f"краёв={len(frag.edges)}")
                processed.append(frag)
            except Exception as e:
                log.warning(f"  Фрагмент #{i}: {e}")

    log.info(f"  Обработано: {len(processed)}/{len(fragments)}")

    if not processed:
        log.error("Ни один фрагмент не обработан. Выход.")
        sys.exit(1)

    # ── Этап 3: Матрица совместимости ─────────────────────────────────────
    with stage("Матрица совместимости", log), timer.measure("сопоставление"):
        compat_matrix, entries = build_compat_matrix(
            processed, threshold=cfg.matching.threshold
        )
        log.info(f"  Пар: {len(entries)}  (порог {cfg.matching.threshold})")
        if entries:
            log.info(f"  Лучшая пара: score={entries[0].score:.4f}")

    # ── Этап 4: Сборка ────────────────────────────────────────────────────
    with stage("Сборка", log), timer.measure("сборка"):
        assembly = assemble(processed, entries, cfg, log)
        assembly.compat_matrix = compat_matrix
        log.info(f"  Score: {assembly.total_score:.4f}")

    # ── Этап 5: Верификация ───────────────────────────────────────────────
    with stage("OCR-верификация", log), timer.measure("верификация"):
        if cfg.verification.run_ocr:
            assembly.ocr_score = verify_full_assembly(
                assembly, lang=cfg.verification.ocr_lang
            )
            log.info(f"  OCR coherence: {assembly.ocr_score:.1%}")
        else:
            log.info("  OCR отключён")

    # ── Этап 6: Экспорт ───────────────────────────────────────────────────
    with stage("Экспорт", log), timer.measure("экспорт"):
        canvas = render_assembly_image(assembly)
        if canvas is not None:
            cv2.imwrite(str(output_path), canvas)
            log.info(f"  Сохранено: {output_path}")
        else:
            log.warning("  Не удалось создать изображение результата")

    # ── Итог ──────────────────────────────────────────────────────────────
    log.info("\n" + timer.report())
    log.info(f"\nРезультат:")
    log.info(f"  Уверенность сборки: {assembly.total_score:.1%}")
    log.info(f"  Связность текста:   {assembly.ocr_score:.1%}")
    log.info(f"  Файл:               {output_path}")

    # ── Интерактивный режим ────────────────────────────────────────────────
    if args.interactive:
        from puzzle_reconstruction.ui.viewer import show
        log.info("\nОткрываю интерактивный редактор...")
        assembly = show(assembly, output_path=str(output_path))

    elif args.visualize and canvas is not None:
        _quick_preview(canvas, assembly)


def _quick_preview(canvas: np.ndarray, assembly) -> None:
    """Быстрый просмотр результата без редактирования."""
    h, w = canvas.shape[:2]
    scale = min(1.0, 1200 / w, 900 / h)
    preview = cv2.resize(canvas, (int(w * scale), int(h * scale)))
    cv2.imshow("Результат (Q — выход)", preview)
    while cv2.waitKey(100) & 0xFF not in (ord('q'), ord('Q'), 27):
        pass
    cv2.destroyAllWindows()


def main():
    parser = build_parser()
    args   = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
