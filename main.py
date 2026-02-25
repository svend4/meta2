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
        assembly, _all_results = assemble(processed, entries, cfg, log)
        assembly.compat_matrix = compat_matrix
        log.info(f"  Score: {assembly.total_score:.4f}")

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
