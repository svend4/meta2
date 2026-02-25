#!/usr/bin/env python3
"""
Восстановление разорванного документа из отсканированных фрагментов.

Использование:
    python main.py --input scans/ --output result.png
    python main.py --input scans/ --output result.png --method beam --beam-width 10
    python main.py --input scans/ --output result.png --config config.json
    python main.py --input scans/ --output result.png --visualize
    python main.py --input scans/ --method all --research          # исследовательский режим
    python main.py --input-list dirs.txt --output results/         # пакетная обработка

Методы сборки (--method):
    greedy     — жадный алгоритм (быстрый, < 1 сек)
    sa         — имитация отжига (лучше, ~10–30 сек)
    beam       — beam search (точнее, ~5–20 сек) [по умолчанию]
    gamma      — гамма-оптимизатор, статья 2026 (экспериментальный)
    genetic    — генетический алгоритм (15–40 фрагментов)
    exhaustive — полный перебор (≤8 фрагментов, точный)
    ant_colony — муравьиная колония (20–60 фрагментов)
    mcts       — Monte Carlo Tree Search (6–25 фрагментов)
    auto       — автовыбор по числу фрагментов
    all        — все 8 методов, выбор лучшего

Алгоритм (6 этапов):
    1. Сегментация каждого фрагмента (Otsu / Adaptive / GrabCut)
    2. Описание краёв: Танграм (изнутри) + Фрактальная кромка (снаружи)
    3. Синтез EdgeSignature (виртуальная линия пересечения)
    4. Матрица совместимости всех краёв (CSS + DTW + FD + OCR)
    5. Сборка выбранным методом
    6. OCR-верификация и экспорт
"""
import argparse
import hashlib
import json
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
from puzzle_reconstruction.assembly.parallel import (
    run_all_methods,
    run_selected,
    pick_best,
    summary_table,
    ALL_METHODS,
)
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
    input_grp = parser.add_mutually_exclusive_group(required=True)
    input_grp.add_argument("--input",  "-i",
                           help="Директория с отсканированными фрагментами")
    input_grp.add_argument("--input-list",
                           help="Файл со списком директорий (по одной на строку) "
                                "для пакетной обработки (batch mode)")
    parser.add_argument("--output", "-o", default="result.png",
                        help="Путь для сохранения результата "
                             "(в batch mode — директория)")

    # Конфигурация
    parser.add_argument("--config", "-c", default=None,
                        help="JSON/YAML файл конфигурации")

    # Метод сборки
    _all_choices = ALL_METHODS + ["auto", "all"]
    parser.add_argument("--method", "-M", default="beam",
                        choices=_all_choices,
                        help=(
                            "Алгоритм сборки: "
                            "greedy/sa/beam/gamma/genetic/exhaustive/ant_colony/mcts — "
                            "одиночный метод; "
                            "auto — автовыбор по числу фрагментов; "
                            "all — все 8 методов с выбором лучшего"
                        ))

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
    parser.add_argument("--sa-iter",      type=int,   default=None,
                        help="Итераций отжига (только для --method sa)")
    parser.add_argument("--mcts-sim",     type=int,   default=None,
                        help="Симуляции MCTS (только для --method mcts)")
    parser.add_argument("--genetic-pop",  type=int,   default=None,
                        help="Размер популяции (только для --method genetic)")
    parser.add_argument("--genetic-gen",  type=int,   default=None,
                        help="Число поколений (только для --method genetic)")
    parser.add_argument("--aco-ants",     type=int,   default=None,
                        help="Число муравьёв (только для --method ant_colony)")
    parser.add_argument("--aco-iter",     type=int,   default=None,
                        help="Итерации ACO (только для --method ant_colony)")
    parser.add_argument("--auto-timeout", type=float, default=None,
                        help="Таймаут на метод в секундах (для --method auto/all)")

    # Режимы
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Показать результат в окне OpenCV")
    parser.add_argument("--interactive", "-I", action="store_true",
                        help="Открыть интерактивный редактор сборки")
    parser.add_argument("--verbose",   action="store_true",
                        help="Подробный вывод (DEBUG-уровень)")
    parser.add_argument("--log-file",  default=None,
                        help="Записывать лог в файл")

    # Исследовательский режим (Фаза 7)
    parser.add_argument("--research", action="store_true",
                        help="Исследовательский режим: сравнение методов, "
                             "консенсусная сборка, экспорт JSON (работает с "
                             "--method all или --method auto)")
    parser.add_argument("--no-consensus", action="store_true",
                        help="Отключить консенсусную сборку в research mode")
    parser.add_argument("--export-json", default=None,
                        help="Путь для экспорта comparison.json "
                             "(по умолчанию: comparison.json рядом с --output)")

    # Кэш дескрипторов (Фаза 6)
    parser.add_argument("--cache-dir", default=None,
                        help="Директория для кэша EdgeSignatures между запусками "
                             "(ускоряет повторные прогоны на том же наборе фрагментов)")

    # Верификация (расширенная)
    parser.add_argument("--validators", default=None,
                        help="Список валидаторов через запятую "
                             "(напр. boundary,metrics,placement) "
                             "или 'all' для запуска всех 21. "
                             "Перекрывает verification.validators из конфига.")
    parser.add_argument("--export-report", default=None, metavar="PATH",
                        help="Экспортировать отчёт верификации в файл. "
                             "Формат определяется расширением: "
                             ".json — структурированный JSON, "
                             ".md / .txt — Markdown, "
                             ".html — HTML-таблица.")
    parser.add_argument("--list-validators", action="store_true",
                        help="Показать список всех 21 доступных валидатора и выйти.")

    # Export — export.py (render_canvas, save_pdf, heatmap, mosaic)
    parser.add_argument("--pdf", default=None, metavar="PATH",
                        help="Сохранить результат в PDF (через export.save_pdf). "
                             "Если PATH пустой — сохраняет рядом с --output.")
    parser.add_argument("--heatmap", default=None, metavar="PATH",
                        help="Сохранить тепловую карту уверенности в PNG-файл.")
    parser.add_argument("--mosaic", default=None, metavar="PATH",
                        help="Сохранить мозаику всех фрагментов в PNG-файл.")

    # Clustering — clustering.py (multi-document scenario)
    parser.add_argument("--cluster", action="store_true", default=False,
                        help="Запустить кластеризацию фрагментов перед сборкой. "
                             "Полезно при наличии фрагментов из нескольких документов.")
    parser.add_argument("--n-docs", type=int, default=None, metavar="N",
                        help="Ожидаемое число документов для --cluster. "
                             "None = определить автоматически (до 8).")

    # Bridge #2 — PreprocessingChain
    parser.add_argument("--preprocessing-chain", default=None, metavar="FILTERS",
                        help="Цепочка фильтров предобработки через запятую. "
                             "Пример: quality_assessor,denoise,contrast,deskew. "
                             "Доступно 35 фильтров (--list-filters для списка).")
    parser.add_argument("--quality-threshold", type=float, default=None,
                        help="Порог качества фрагмента [0..1]. "
                             "Фрагменты ниже порога отбрасываются. "
                             "0.0 = без фильтрации (по умолчанию).")
    parser.add_argument("--auto-enhance", action="store_true",
                        help="Автоматически добавить denoise+contrast "
                             "если --preprocessing-chain не задан.")
    parser.add_argument("--list-filters", action="store_true",
                        help="Показать список всех доступных фильтров "
                             "предобработки и выйти.")

    # Bridge #5 — Algorithms
    parser.add_argument("--algorithms", default=None, metavar="NAMES",
                        help="Список алгоритмов Bridge #5 через запятую. "
                             "Автоматически определяет уровень применения: "
                             "fragment (на каждый фрагмент), "
                             "pair (после Otsu-фильтрации), "
                             "assembly (постобработка сборки). "
                             "Пример: fragment_classifier,seam_evaluator,path_planner. "
                             "(--list-algorithms для полного списка)")
    parser.add_argument("--list-algorithms", action="store_true",
                        help="Показать список всех 34 алгоритмов Bridge #5 и выйти.")
    # Bridge #6 — Utils
    parser.add_argument("--utils-profiler", action="store_true", default=False,
                        help="Bridge #6: активировать PipelineProfiler для шагов пайплайна.")
    parser.add_argument("--utils-event-log", action="store_true", default=False,
                        help="Bridge #6: включить журнал событий пайплайна (EventLog).")
    parser.add_argument("--utils-progress", action="store_true", default=False,
                        help="Bridge #6: включить ProgressTracker для отслеживания прогресса.")
    parser.add_argument("--utils-image-stats", action="store_true", default=False,
                        help="Bridge #6: вычислять image_stats для каждого фрагмента.")
    parser.add_argument("--utils-export-log", default="", metavar="PATH",
                        help="Bridge #6: путь для экспорта журнала событий (JSONL).")
    parser.add_argument("--list-utils", action="store_true",
                        help="Показать список утилит Bridge #6 (124 утилиты) и выйти.")
    # Bridge #7 — Tools
    parser.add_argument(
        "--tool",
        choices=["benchmark", "evaluate", "mix", "profile", "serve", "tear"],
        default=None, metavar="TOOL",
        help=("Bridge #7: запустить автономный инструмент. "
              "Инструменты: benchmark, evaluate, mix, profile, serve, tear. "
              "Дополнительные параметры: --n-fragments, --n-pieces, --methods, "
              "--n-trials, --port, --host."),
    )
    parser.add_argument("--list-tools", action="store_true",
                        help="Показать список инструментов Bridge #7 и выйти.")
    # Дополнительные параметры для инструментов
    parser.add_argument("--n-fragments", type=int, default=8,
                        help="Число фрагментов для --tool profile / benchmark.")
    parser.add_argument("--n-pieces", type=int, default=6,
                        help="Число частей для --tool tear / mix / evaluate.")
    parser.add_argument("--n-trials", type=int, default=3,
                        help="Число повторений для --tool benchmark / evaluate.")
    parser.add_argument("--methods", default="beam,sa", metavar="METHODS",
                        help="Методы сборки через запятую для --tool benchmark/evaluate.")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Хост для --tool serve.")
    parser.add_argument("--port", type=int, default=5000,
                        help="Порт для --tool serve.")
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
    # Предобработка (цепочка фильтров — Мост №2)
    if not cfg.preprocessing.is_empty() if hasattr(cfg.preprocessing, 'is_empty') else \
            (cfg.preprocessing.chain or cfg.preprocessing.auto_enhance):
        from puzzle_reconstruction.preprocessing.chain import PreprocessingChain
        chain = PreprocessingChain(
            filters=cfg.preprocessing.chain,
            quality_threshold=cfg.preprocessing.quality_threshold,
            auto_enhance=cfg.preprocessing.auto_enhance,
        )
        enhanced = chain.apply(frag.image)
        if enhanced is None:
            raise ValueError(
                f"фрагмент #{frag.fragment_id} отклонён по качеству"
                f" (порог={cfg.preprocessing.quality_threshold})"
            )
        frag.image = enhanced

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


def _auto_methods(n_fragments: int) -> list:
    """Выбирает методы сборки по числу фрагментов."""
    if n_fragments <= 4:
        return ["exhaustive"]
    elif n_fragments <= 8:
        return ["exhaustive", "beam"]
    elif n_fragments <= 15:
        return ["beam", "mcts", "sa"]
    elif n_fragments <= 30:
        return ["genetic", "gamma", "ant_colony"]
    else:
        return ["gamma", "sa"]


def assemble(fragments, entries, cfg: Config, log):
    """
    Запускает выбранный метод(ы) сборки через единый реестр parallel.py.

    Returns:
        (Assembly, List[MethodResult]) — лучшая сборка и список всех результатов.
    """
    method = cfg.assembly.method
    log.info(f"  Метод: {method}")

    kwargs = dict(
        beam_width=cfg.assembly.beam_width,
        n_iterations=cfg.assembly.sa_iter,
        n_simulations=cfg.assembly.mcts_sim,
        seed=cfg.assembly.seed,
    )

    if method == "all":
        results = run_all_methods(
            fragments, entries,
            methods=ALL_METHODS,
            timeout=cfg.assembly.auto_timeout,
            n_workers=min(4, len(ALL_METHODS)),
            **kwargs,
        )
        log.info("\n" + summary_table(results))
        best = pick_best(results)
        if best is None:
            log.error("Все методы завершились с ошибкой")
            sys.exit(1)
        return best, results

    if method == "auto":
        methods = _auto_methods(len(fragments))
        log.info(f"  Авто-выбор ({len(fragments)} фрагментов): {methods}")
        results = run_all_methods(
            fragments, entries,
            methods=methods,
            timeout=cfg.assembly.auto_timeout,
            **kwargs,
        )
        log.info("\n" + summary_table(results))
        best = pick_best(results)
        if best is None:
            log.error("Все автовыбранные методы завершились с ошибкой")
            sys.exit(1)
        return best, results

    # Одиночный метод
    try:
        results = run_selected(fragments, entries, methods=[method], **kwargs)
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)
    if not results or not results[0].success:
        err = results[0].error if results else "нет результата"
        log.error(f"Метод {method!r} завершился с ошибкой: {err}")
        sys.exit(1)
    return results[0].assembly, results


def _fragment_cache_key(img: np.ndarray) -> str:
    """Вычисляет хэш изображения для кэширования дескрипторов."""
    h = hashlib.md5(img.tobytes()).hexdigest()
    return f"frag_{img.shape[0]}x{img.shape[1]}_{h}"


def _try_load_cached_fragment(cache, key: str):
    """Загружает Fragment из кэша. Возвращает None если кэш пуст или недоступен."""
    if cache is None:
        return None
    try:
        return cache.get(key)
    except Exception:
        return None


def _try_save_fragment(cache, key: str, frag) -> None:
    """Сохраняет Fragment в кэш (игнорирует ошибки)."""
    if cache is None:
        return
    try:
        cache.put(key, frag)
    except Exception:
        pass


def _make_descriptor_cache(cache_dir: str | None):
    """Создаёт ResultCache для кэша дескрипторов или возвращает None."""
    if cache_dir is None:
        return None
    try:
        from puzzle_reconstruction.utils.result_cache import ResultCache, CachePolicy
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return ResultCache(CachePolicy(namespace="fragments", ttl=0.0))
    except Exception:
        return None


def _run_research_mode(assembly_results, entries, cfg, log,
                       fragments=None) -> dict:
    """
    Исследовательский режим (Фаза 7):
    - Строит consensus-сборку из успешных методов
    - Собирает метрики через MetricTracker
    - Возвращает словарь для экспорта в JSON

    Args:
        assembly_results: List[MethodResult] из run_all_methods
        entries:          Список CompatEntry (матрица совместимости)
        cfg:              Config
        log:              Logger
        fragments:        List[Fragment] (для консенсусной сборки)

    Returns:
        dict с полями methods, consensus_score, best_method
    """
    report = {"methods": [], "consensus_score": 0.0, "best_method": None}

    # ── MetricTracker: регистрируем score каждого метода ──────────────────
    try:
        from puzzle_reconstruction.utils.metric_tracker import make_tracker
        tracker = make_tracker()
        for i, r in enumerate(assembly_results):
            if r.success:
                tracker.record(r.name, r.assembly.total_score, step=i)
        report["metric_stats"] = {
            name: {"mean": float(st.mean), "max": float(st.maximum), "last": float(st.last)}
            for name, st in (
                (m, tracker.stats(m))
                for m in tracker.metric_names()
            )
            if st is not None
        }
    except Exception:
        pass

    # ── Таблица методов ───────────────────────────────────────────────────
    for r in assembly_results:
        entry = {
            "method":  r.name,
            "success": r.success,
            "score":   float(r.assembly.total_score) if r.success else 0.0,
            "time_s":  float(r.elapsed) if hasattr(r, "elapsed") else 0.0,
            "error":   r.error if not r.success else None,
        }
        report["methods"].append(entry)

    successful = [r for r in assembly_results if r.success]
    if successful:
        best = max(successful, key=lambda r: r.assembly.total_score)
        report["best_method"] = best.method
        report["best_score"]  = float(best.assembly.total_score)

    # ── Консенсусная сборка ───────────────────────────────────────────────
    if cfg.research.consensus and len(successful) >= 2:
        try:
            from puzzle_reconstruction.matching.consensus import build_consensus
            assemblies = [r.assembly for r in successful]
            threshold  = cfg.research.consensus_threshold
            consensus_result = build_consensus(
                assemblies, fragments or [], entries, threshold=threshold
            )
            if (consensus_result is not None
                    and consensus_result.assembly is not None):
                c_score = float(consensus_result.assembly.total_score)
                report["consensus_score"] = c_score
                log.info(f"  Консенсус ({len(assemblies)} методов, "
                         f"порог={threshold:.0%}): "
                         f"score={c_score:.1%}")
        except Exception as exc:
            log.debug(f"  Консенсус недоступен: {exc}")

    return report


def _export_comparison(report: dict, path: str | Path, log) -> None:
    """Сохраняет comparison.json."""
    try:
        path = Path(path)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        log.info(f"  Сравнение экспортировано: {path}")
    except Exception as exc:
        log.warning(f"  Не удалось сохранить comparison.json: {exc}")


def _export_verification_report(v_report, path: str | Path, log) -> None:
    """Экспортирует VerificationReport в JSON / Markdown / HTML.

    Формат определяется расширением файла:
        .json        — структурированный JSON
        .md / .txt   — Markdown (таблица + итог)
        .html        — HTML-страница с таблицей

    Делегирует форматирование методам VerificationReport:
        as_dict() / to_json() / to_markdown() / to_html()
    """
    path = Path(path)
    ext  = path.suffix.lower()

    try:
        if ext == ".json":
            path.write_text(v_report.to_json(), encoding="utf-8")

        elif ext in (".md", ".txt"):
            path.write_text(v_report.to_markdown(), encoding="utf-8")

        elif ext == ".html":
            path.write_text(v_report.to_html(), encoding="utf-8")

        else:
            # Неизвестное расширение — сохраняем как Markdown
            _export_verification_report(v_report, path.with_suffix(".md"), log)
            return

        log.info(f"  Отчёт верификации экспортирован: {path}")
    except Exception as exc:
        log.warning(f"  Не удалось экспортировать отчёт верификации: {exc}")


def _run_batch(input_list_file: str, args: argparse.Namespace, log) -> None:
    """
    Пакетная обработка (Фаза 6): читает список директорий из файла и
    запускает run() для каждой, используя BatchProcessor для отчётности.
    """
    try:
        from puzzle_reconstruction.utils.batch_processor import (
            process_items, ProcessConfig, filter_successful
        )
    except Exception as exc:
        log.error(f"batch_processor недоступен: {exc}")
        sys.exit(1)

    lines = Path(input_list_file).read_text(encoding="utf-8").splitlines()
    dirs  = [l.strip() for l in lines if l.strip() and not l.startswith("#")]

    if not dirs:
        log.error(f"Файл {input_list_file!r} не содержит директорий")
        sys.exit(1)

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    log.info(f"Пакетная обработка: {len(dirs)} директорий → {output_root}")

    def _process_one_dir(input_dir: str) -> dict:
        dir_name   = Path(input_dir).name
        out_file   = str(output_root / f"{dir_name}_result.png")
        batch_args = argparse.Namespace(**vars(args))
        batch_args.input       = input_dir
        batch_args.input_list  = None
        batch_args.output      = out_file
        batch_args.visualize   = False
        batch_args.interactive = False
        run(batch_args)
        return {"input": input_dir, "output": out_file}

    cfg_proc = ProcessConfig(batch_size=1, max_retries=1, stop_on_error=False)
    summary  = process_items(dirs, _process_one_dir, cfg_proc)
    ok       = filter_successful(summary)

    failed = [item for item in summary.items if not item.success]
    log.info(f"\nПакетная обработка завершена:")
    log.info(f"  Успешно: {len(ok)}/{summary.total}")
    if failed:
        log.warning(f"  Ошибки:")
        for item in failed:
            log.warning(f"    [{item.index}] {dirs[item.index]}: {item.error}")


def _bridge5_fragment(frag, alg_names: list, log) -> None:
    """Запускает fragment-level алгоритмы Bridge #5 для одного фрагмента."""
    if not alg_names:
        return
    try:
        from puzzle_reconstruction.algorithms.bridge import get_algorithm, get_category
    except Exception:
        return
    for name in alg_names:
        if get_category(name) != "fragment":
            continue
        fn = get_algorithm(name)
        if fn is None:
            continue
        try:
            if name in ("fragment_classifier", "line_detector",
                        "word_segmentation", "region_segmenter"):
                fn(frag.image)
            elif name == "fragment_quality":
                fn(frag.image, frag.mask)
            elif name in ("boundary_descriptor", "fourier_descriptor",
                          "shape_context", "contour_smoother"):
                if frag.contour is not None and len(frag.contour) >= 4:
                    fn(frag.contour)
            elif name in ("contour_tracker", "region_splitter"):
                if frag.mask is not None:
                    fn(frag.mask)
            else:
                fn(frag.image)
        except Exception as exc:
            log.debug(f"  Bridge #5 fragment/{name}: {exc}")


def _bridge5_pair(entries: list, fragments: list, alg_names: list, log) -> list:
    """Запускает pair-level алгоритмы Bridge #5 после Otsu-фильтрации."""
    if not alg_names or not entries:
        return entries
    try:
        from puzzle_reconstruction.algorithms.bridge import get_algorithm, get_category
    except Exception:
        return entries
    frag_by_id = {f.fragment_id: f for f in fragments}
    for name in alg_names:
        if get_category(name) != "pair":
            continue
        fn = get_algorithm(name)
        if fn is None:
            continue
        try:
            if name == "edge_filter":
                filtered = fn(entries)
                if filtered is not None:
                    entries = filtered
                    log.info(f"  Bridge #5 edge_filter: {len(entries)} пар")
            elif name == "score_aggregator":
                scores = [e.score for e in entries]
                if scores:
                    fn(scores)
            elif name in ("seam_evaluator", "edge_scorer"):
                top = sorted(entries, key=lambda e: e.score, reverse=True)[:50]
                for entry in top:
                    fi = frag_by_id.get(getattr(entry.edge_i, "fragment_id", -1))
                    fj = frag_by_id.get(getattr(entry.edge_j, "fragment_id", -1))
                    if fi and fj:
                        fn(fi.image, getattr(entry.edge_i, "side", None),
                           fj.image, getattr(entry.edge_j, "side", None))
            elif name in ("sift_matcher", "patch_matcher"):
                top = sorted(entries, key=lambda e: e.score, reverse=True)[:20]
                for entry in top:
                    fi = frag_by_id.get(getattr(entry.edge_i, "fragment_id", -1))
                    fj = frag_by_id.get(getattr(entry.edge_j, "fragment_id", -1))
                    if fi and fj:
                        fn(fi.image, fj.image)
        except Exception as exc:
            log.debug(f"  Bridge #5 pair/{name}: {exc}")
    return entries


def _bridge5_assembly(assembly, fragments: list, alg_names: list, log) -> None:
    """Запускает assembly-level алгоритмы Bridge #5 после сборки."""
    if not alg_names:
        return
    try:
        from puzzle_reconstruction.algorithms.bridge import get_algorithm, get_category
        import numpy as np
    except Exception:
        return
    for name in alg_names:
        if get_category(name) != "assembly":
            continue
        fn = get_algorithm(name)
        if fn is None:
            continue
        try:
            if name == "path_planner":
                if (assembly.compat_matrix is not None
                        and assembly.compat_matrix.size > 1):
                    n = assembly.compat_matrix.shape[0]
                    result = fn(assembly.compat_matrix, 0, n - 1)
                    log.debug(f"  Bridge #5 path_planner: cost="
                              f"{getattr(result, 'cost', result)}")
            elif name == "overlap_resolver":
                contours = {f.fragment_id: f.contour for f in fragments
                            if f.contour is not None}
                if contours:
                    fn(assembly, contours)
            elif name == "position_estimator":
                placements = assembly.placements
                if isinstance(placements, dict) and len(placements) >= 2:
                    ids = list(placements.keys())
                    offsets = {}
                    for i, a in enumerate(ids):
                        for b in ids[i + 1:]:
                            pa = np.asarray(placements[a]).ravel()[:2]
                            pb = np.asarray(placements[b]).ravel()[:2]
                            if len(pa) == 2 and len(pb) == 2:
                                offsets[(a, b)] = pb - pa
                    if offsets:
                        from puzzle_reconstruction.algorithms.position_estimator \
                            import build_offset_graph
                        graph = build_offset_graph(list(offsets.keys()),
                                                   list(offsets.values()))
                        fn(graph, ids[0])
            elif name in ("descriptor_aggregator", "descriptor_combiner"):
                vecs = []
                for frag in fragments:
                    for edge in (frag.edges or []):
                        v = getattr(edge, "css_vec", None)
                        if v is not None:
                            vecs.append(np.asarray(v, dtype=float).ravel())
                if vecs:
                    fn(vecs)
        except Exception as exc:
            log.debug(f"  Bridge #5 assembly/{name}: {exc}")


def run(args: argparse.Namespace) -> None:
    import logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log = get_logger("puzzle", level=log_level, log_file=args.log_file)

    # ── Пакетный режим (batch) ────────────────────────────────────────────
    if getattr(args, "input_list", None):
        _run_batch(args.input_list, args, log)
        return

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
        mcts_sim=args.mcts_sim,
        genetic_pop=args.genetic_pop,
        genetic_gen=args.genetic_gen,
        aco_ants=args.aco_ants,
        aco_iter=args.aco_iter,
        auto_timeout=args.auto_timeout,
    )

    # Research mode: --research включает cfg.research.enabled
    if getattr(args, "research", False):
        cfg.research.enabled = True
    if getattr(args, "no_consensus", False):
        cfg.research.consensus = False
    if getattr(args, "export_json", None):
        cfg.research.comparison_file = args.export_json
        cfg.research.export_comparison = True

    # Bridge #2 — PreprocessingChain (CLI переопределения)
    if getattr(args, "preprocessing_chain", None):
        cfg.preprocessing.chain = [
            f.strip() for f in args.preprocessing_chain.split(",") if f.strip()
        ]
    if getattr(args, "quality_threshold", None) is not None:
        cfg.preprocessing.quality_threshold = args.quality_threshold
    if getattr(args, "auto_enhance", False):
        cfg.preprocessing.auto_enhance = True

    # Bridge #5 — Algorithms (CLI → cfg.algorithms)
    _alg_names: list = []
    if getattr(args, "algorithms", None):
        _alg_names = [n.strip() for n in args.algorithms.split(",") if n.strip()]
        try:
            from puzzle_reconstruction.algorithms.bridge import get_category
            cfg.algorithms.fragment = [n for n in _alg_names
                                       if get_category(n) == "fragment"]
            cfg.algorithms.pair     = [n for n in _alg_names
                                       if get_category(n) == "pair"]
            cfg.algorithms.assembly = [n for n in _alg_names
                                       if get_category(n) == "assembly"]
            if cfg.algorithms.fragment or cfg.algorithms.pair or cfg.algorithms.assembly:
                log.info(
                    "Bridge #5: fragment=%d, pair=%d, assembly=%d",
                    len(cfg.algorithms.fragment),
                    len(cfg.algorithms.pair),
                    len(cfg.algorithms.assembly),
                )
        except Exception as exc:
            log.debug(f"Bridge #5 init error: {exc}")

    # Bridge #6 — Utils (CLI → cfg.utils)
    if getattr(args, "utils_profiler", False):
        cfg.utils.profiler = True
    if getattr(args, "utils_event_log", False):
        cfg.utils.event_log = True
    if getattr(args, "utils_progress", False):
        cfg.utils.progress = True
    if getattr(args, "utils_image_stats", False):
        cfg.utils.image_stats = True
    if getattr(args, "utils_export_log", ""):
        cfg.utils.export_log = args.utils_export_log

    input_dir   = Path(args.input)
    output_path = Path(args.output)

    log.info("=" * 55)
    log.info("ВОССТАНОВЛЕНИЕ РАЗОРВАННОГО ДОКУМЕНТА")
    log.info(f"Метод: {cfg.assembly.method}  |  α={cfg.synthesis.alpha}")
    if cfg.research.enabled:
        log.info("  [Research Mode: ON]")
    log.info("=" * 55)

    # Кэш дескрипторов (Фаза 6 — ResultCache)
    descriptor_cache = _make_descriptor_cache(getattr(args, "cache_dir", None))
    if descriptor_cache is not None:
        log.info(f"  Кэш дескрипторов: {args.cache_dir}")

    # ── Этап 1: Загрузка ──────────────────────────────────────────────────
    with stage("Загрузка фрагментов", log), timer.measure("загрузка"):
        fragments = load_fragments(input_dir, log)

    # ── Этап 2: Обработка ─────────────────────────────────────────────────
    with stage("Обработка фрагментов", log), timer.measure("обработка"):
        processed = []
        cache_hits = 0
        for i, frag in enumerate(fragments):
            try:
                # Попытка получить из кэша (Фаза 6)
                cache_key    = _fragment_cache_key(frag.image)
                cached_frag  = _try_load_cached_fragment(descriptor_cache, cache_key)
                if cached_frag is not None:
                    cached_frag.fragment_id = frag.fragment_id
                    frag = cached_frag
                    cache_hits += 1
                else:
                    frag = process_fragment(frag, cfg, log)
                    _try_save_fragment(descriptor_cache, cache_key, frag)

                # Bridge #5: fragment-level алгоритмы
                _bridge5_fragment(frag, cfg.algorithms.fragment, log)

                fd = (frag.fractal.fd_box + frag.fractal.fd_divider) / 2
                log.debug(f"  #{frag.fragment_id:3d}  "
                           f"форма={frag.tangram.shape_class.value:<12}  "
                           f"FD={fd:.3f}  "
                           f"краёв={len(frag.edges)}")
                processed.append(frag)
            except Exception as e:
                log.warning(f"  Фрагмент #{i}: {e}")

        if cache_hits:
            log.info(f"  Из кэша: {cache_hits}/{len(fragments)}")

    log.info(f"  Обработано: {len(processed)}/{len(fragments)}")

    if not processed:
        log.error("Ни один фрагмент не обработан. Выход.")
        sys.exit(1)

    # ── Кластеризация (clustering.py) ─────────────────────────────────────
    if getattr(args, "cluster", False):
        with stage("Кластеризация фрагментов", log), timer.measure("кластеризация"):
            try:
                from puzzle_reconstruction.clustering import (
                    cluster_fragments, split_by_cluster
                )
                n_docs = getattr(args, "n_docs", None)
                result = cluster_fragments(processed, k=n_docs)
                log.info(
                    f"  Кластеров: {result.n_clusters}  "
                    f"(метод={result.method}  "
                    f"silhouette={result.silhouette:.3f})"
                )
                clusters = split_by_cluster(processed, result)
                log.info(
                    f"  Размеры кластеров: "
                    + ", ".join(f"#{i}:{len(c)}" for i, c in clusters.items())
                )
                # Продолжаем с наибольшим кластером
                largest = max(clusters, key=lambda k: len(clusters[k]))
                if len(clusters) > 1:
                    log.info(
                        f"  Используем кластер #{largest} "
                        f"({len(clusters[largest])} фрагментов)"
                    )
                    processed = clusters[largest]
            except Exception as exc:
                log.warning(f"  Кластеризация недоступна: {exc}")

    # ── Этап 3: Матрица совместимости ─────────────────────────────────────
    with stage("Матрица совместимости", log), timer.measure("сопоставление"):
        compat_matrix, entries = build_compat_matrix(
            processed, threshold=cfg.matching.threshold
        )
        log.info(f"  Пар: {len(entries)}  (порог {cfg.matching.threshold})")
        if entries:
            log.info(f"  Лучшая пара: score={entries[0].score:.4f}")
        # Bridge #5: pair-level алгоритмы
        entries = _bridge5_pair(entries, processed, cfg.algorithms.pair, log)

    # ── Этап 4: Сборка ────────────────────────────────────────────────────
    with stage("Сборка", log), timer.measure("сборка"):
        assembly, _all_results = assemble(processed, entries, cfg, log)
        assembly.compat_matrix = compat_matrix
        log.info(f"  Score: {assembly.total_score:.4f}")
        # Bridge #5: assembly-level алгоритмы
        _bridge5_assembly(assembly, processed, cfg.algorithms.assembly, log)

    # ── Этап 5: Верификация ───────────────────────────────────────────────
    with stage("Верификация", log), timer.measure("верификация"):
        # OCR-верификация
        if cfg.verification.run_ocr:
            assembly.ocr_score = verify_full_assembly(
                assembly, lang=cfg.verification.ocr_lang
            )
            log.info(f"  OCR coherence: {assembly.ocr_score:.1%}")
        else:
            log.info("  OCR отключён")

        # Расширенная верификация (VerificationSuite — 21 валидатор)
        cli_validators = getattr(args, "validators", None)
        if cli_validators is not None:
            from puzzle_reconstruction.verification.suite import (
                VerificationSuite, all_validator_names
            )
            if cli_validators.strip().lower() == "all":
                validators_list = all_validator_names()
            else:
                validators_list = [v.strip() for v in cli_validators.split(",")
                                   if v.strip()]
            cfg.verification.validators = validators_list

        if cfg.verification.validators:
            from puzzle_reconstruction.verification.suite import VerificationSuite
            suite = VerificationSuite(validators=cfg.verification.validators)
            v_report = suite.run(assembly)
            log.info(v_report.summary())
            log.info(f"  Suite score: {v_report.final_score:.1%}")

            export_path = getattr(args, "export_report", None)
            if export_path:
                _export_verification_report(v_report, export_path, log)

    # ── Research Mode (Фаза 7) ────────────────────────────────────────────
    if cfg.research.enabled and cfg.assembly.method in ("all", "auto"):
        with stage("Research Mode", log), timer.measure("research"):
            research_report = _run_research_mode(
                _all_results, entries, cfg, log, fragments=processed
            )
            research_report["input"]  = str(input_dir)
            research_report["output"] = str(output_path)
            research_report["method"] = cfg.assembly.method

            if cfg.research.export_comparison:
                cmp_path = cfg.research.comparison_file or (
                    str(output_path.with_suffix("")) + "_comparison.json"
                )
                _export_comparison(research_report, cmp_path, log)

    # ── Этап 6: Экспорт ───────────────────────────────────────────────────
    with stage("Экспорт", log), timer.measure("экспорт"):
        # Основной рендер: сначала пробуем export.render_canvas (лучший
        # рендерер), при неудаче — fallback на render_assembly_image (OCR).
        canvas = None
        try:
            from puzzle_reconstruction.export import render_canvas as _render_canvas
            canvas = _render_canvas(assembly)
        except Exception:
            pass
        if canvas is None:
            canvas = render_assembly_image(assembly)

        if canvas is not None:
            cv2.imwrite(str(output_path), canvas)
            log.info(f"  Сохранено: {output_path}")
        else:
            log.warning("  Не удалось создать изображение результата")

        # PDF экспорт (опционально)
        pdf_path = getattr(args, "pdf", None)
        if pdf_path is not None:
            pdf_file = pdf_path if pdf_path else str(output_path.with_suffix(".pdf"))
            try:
                from puzzle_reconstruction.export import save_pdf
                save_pdf(assembly, canvas or np.zeros((100, 100, 3), dtype=np.uint8),
                         pdf_file)
                log.info(f"  PDF сохранён: {pdf_file}")
            except Exception as exc:
                log.debug(f"  export.save_pdf ошибка: {exc}")

        # Тепловая карта (опционально)
        heatmap_path = getattr(args, "heatmap", None)
        if heatmap_path and canvas is not None:
            try:
                from puzzle_reconstruction.export import render_heatmap
                from puzzle_reconstruction.export import save_png
                heatmap = render_heatmap(assembly, canvas.shape)
                save_png(heatmap, heatmap_path)
                log.info(f"  Тепловая карта: {heatmap_path}")
            except Exception as exc:
                log.debug(f"  export.render_heatmap ошибка: {exc}")

        # Мозаика фрагментов (опционально)
        mosaic_path = getattr(args, "mosaic", None)
        if mosaic_path:
            try:
                from puzzle_reconstruction.export import render_mosaic, save_png
                mosaic = render_mosaic(assembly)
                save_png(mosaic, mosaic_path)
                log.info(f"  Мозаика: {mosaic_path}")
            except Exception as exc:
                log.debug(f"  export.render_mosaic ошибка: {exc}")

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


def _run_tool_from_args(args) -> None:
    """
    Bridge #7: запускает автономный инструмент по args.tool.

    Поддерживаемые инструменты:
        profile    — run_profile(n_fragments, verbose=True)
        benchmark  — run_benchmark(n_pieces_list, methods, n_trials)
        evaluate   — run_evaluation(methods, n_pieces_list, n_trials, ...)
        mix        — mix_from_generated(n_docs, n_pieces, output_dir)
        tear       — tear_document(image, n_pieces, noise_level) → сохраняет в --output
        serve      — Flask REST API (host, port)
    """
    from tools.registry import get_tool
    log = get_logger("tools")

    name = args.tool
    info = get_tool(name)
    if info is None:
        print(f"Ошибка: инструмент {name!r} не найден.")
        sys.exit(1)

    fn = info.load()
    if fn is None:
        print(f"Ошибка: инструмент {name!r} недоступен "
              f"(отсутствует зависимость). "
              f"Проверьте вывод --list-tools.")
        sys.exit(1)

    methods_list = [m.strip() for m in getattr(args, "methods", "beam,sa").split(",")
                    if m.strip()]

    if name == "profile":
        n_frags = getattr(args, "n_fragments", 8)
        log.info(f"[tools] profile: n_fragments={n_frags}")
        result = fn(n_fragments=n_frags, verbose=True)
        print(f"\nПрофиль завершён. Итого этапов: {len(result.stages)}")

    elif name == "benchmark":
        n_pieces = getattr(args, "n_pieces", 6)
        n_trials = getattr(args, "n_trials", 3)
        out      = getattr(args, "output", None)
        log.info(f"[tools] benchmark: pieces={n_pieces} methods={methods_list}")
        fn(
            n_pieces_list=[n_pieces],
            methods=methods_list,
            n_trials=n_trials,
            output_path=out,
        )

    elif name == "evaluate":
        n_pieces = getattr(args, "n_pieces", 6)
        n_trials = getattr(args, "n_trials", 3)
        out_dir  = Path(getattr(args, "output", "evaluation_results"))
        log.info(f"[tools] evaluate: methods={methods_list} pieces={n_pieces}")
        fn(
            methods=methods_list,
            n_pieces_list=[n_pieces],
            n_trials=n_trials,
            noise=0.5,
            output_dir=out_dir,
            save_html=True,
            save_md=True,
        )
        log.info(f"[tools] evaluate: результаты в {out_dir}")

    elif name == "mix":
        n_pieces = getattr(args, "n_pieces", 6)
        out_dir  = Path(getattr(args, "output", "mixed_fragments"))
        log.info(f"[tools] mix: n_docs=2 n_pieces={n_pieces}")
        fn(
            n_docs=2,
            n_pieces=n_pieces,
            output_dir=out_dir,
        )
        log.info(f"[tools] mix: фрагменты сохранены в {out_dir}")

    elif name == "tear":
        input_path = getattr(args, "input", None) or getattr(args, "output", None)
        out_dir    = Path(getattr(args, "output", "torn_fragments"))
        n_pieces   = getattr(args, "n_pieces", 6)
        if input_path and Path(input_path).is_file():
            img = cv2.imread(str(input_path))
            if img is None:
                print(f"Ошибка: не удалось загрузить изображение {input_path}")
                sys.exit(1)
        else:
            # Генерируем тестовый документ
            from tools.tear_generator import generate_test_document
            img = generate_test_document()
            log.info("[tools] tear: синтетический документ сгенерирован")
        pieces = fn(image=img, n_pieces=n_pieces)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, piece in enumerate(pieces):
            path = out_dir / f"fragment_{i:03d}.png"
            cv2.imwrite(str(path), piece)
        log.info(f"[tools] tear: {len(pieces)} фрагментов → {out_dir}")

    elif name == "serve":
        host = getattr(args, "host", "0.0.0.0")
        port = getattr(args, "port", 5000)
        log.info(f"[tools] serve: http://{host}:{port}")
        fn(host=host, port=port, debug=False)

    else:
        print(f"Инструмент {name!r}: обработчик не реализован.")
        sys.exit(1)


def main():
    # --list-validators не требует --input, поэтому обрабатываем до parse_args
    if "--list-validators" in sys.argv:
        from puzzle_reconstruction.verification.suite import all_validator_names
        names = all_validator_names()
        print(f"Доступные валидаторы ({len(names)} штук):")
        for i, name in enumerate(names, 1):
            print(f"  {i:2d}. {name}")
        print()
        print("Использование:")
        print("  --validators all              # запустить все 21")
        print("  --validators boundary,metrics # запустить подмножество")
        return

    if "--list-filters" in sys.argv:
        from puzzle_reconstruction.preprocessing.chain import list_filters
        names = list_filters()
        print(f"Доступные фильтры предобработки ({len(names)} штук):")
        for i, name in enumerate(names, 1):
            print(f"  {i:2d}. {name}")
        print()
        print("Использование:")
        print("  --preprocessing-chain denoise,contrast,deskew")
        print("  --preprocessing-chain quality_assessor,perspective,warp_correct")
        return

    if "--list-algorithms" in sys.argv:
        from puzzle_reconstruction.algorithms.bridge import (
            list_algorithms, ALGORITHM_CATEGORIES
        )
        print("Доступные алгоритмы Bridge #5:")
        for cat, names in ALGORITHM_CATEGORIES.items():
            avail = list_algorithms(category=cat)
            print(f"\n  [{cat}] ({len(avail)} из {len(names)}):")
            for name in avail:
                print(f"    {name}")
        print()
        print("Использование:")
        print("  --algorithms fragment_classifier,seam_evaluator,path_planner")
        return

    if "--list-utils" in sys.argv:
        from puzzle_reconstruction.utils.bridge import (
            list_utils, UTIL_CATEGORIES
        )
        print("Доступные утилиты Bridge #6:")
        for cat, names in UTIL_CATEGORIES.items():
            avail = list_utils(category=cat)
            print(f"\n  [{cat}] ({len(avail)} из {len(names)}):")
            for name in avail:
                print(f"    {name}")
        print()
        print("Использование:")
        print("  --utils-event-log --utils-progress --utils-profiler")
        print("  --utils-image-stats --utils-export-log events.jsonl")
        return

    if "--list-tools" in sys.argv:
        from tools.registry import list_tools
        tools = list_tools()
        print(f"Доступные инструменты Bridge #7 ({len(tools)} штук):\n")
        for name, info in tools.items():
            fn    = info.load()
            avail = "доступен" if fn else "недоступен (зависимость отсутствует)"
            print(f"  {name:14s} [{avail}]")
            print(f"               {info.description}")
            if info.params:
                print(f"               Параметры: {', '.join(info.params)}")
            print()
        print("Использование:")
        print("  python main.py --tool profile --n-fragments 8")
        print("  python main.py --tool benchmark --methods beam,sa --n-pieces 4")
        print("  python main.py --tool tear --input doc.png --n-pieces 6 --output frags/")
        print("  python main.py --tool serve --port 5000")
        return

    # Bridge #7 — запуск инструмента до parse_args (--input не требуется)
    if "--tool" in sys.argv:
        _tool_parser = argparse.ArgumentParser(add_help=False)
        _tool_parser.add_argument("--tool",        default=None)
        _tool_parser.add_argument("--input", "-i", default=None)
        _tool_parser.add_argument("--output",      default="output")
        _tool_parser.add_argument("--n-fragments",  type=int, default=8)
        _tool_parser.add_argument("--n-pieces",     type=int, default=6)
        _tool_parser.add_argument("--n-trials",     type=int, default=3)
        _tool_parser.add_argument("--methods",      default="beam,sa")
        _tool_parser.add_argument("--host",         default="0.0.0.0")
        _tool_parser.add_argument("--port",         type=int, default=5000)
        _tool_args, _ = _tool_parser.parse_known_args()
        _run_tool_from_args(_tool_args)
        return

    parser = build_parser()
    args   = parser.parse_args()

    # Нормализуем имена атрибутов (argparse конвертирует дефисы в подчёркивания)
    if not hasattr(args, "input"):
        args.input = None
    if not hasattr(args, "input_list"):
        args.input_list = None
    run(args)


if __name__ == "__main__":
    main()
