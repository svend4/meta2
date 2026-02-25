"""
Унифицированный пайплайн восстановления разорванного документа.

Класс Pipeline объединяет все 6 этапов обработки в один объект:
    1. Загрузка и нормализация цвета
    2. Сегментация (Otsu / Adaptive / GrabCut)
    3. Описание краёв (Танграм + Фрактал → EdgeSignature)
    4. Матрица совместимости (CSS + DTW + FD)
    5. Сборка (Greedy / SA / Beam / Gamma / Exhaustive)
    6. Верификация (OCR coherence score)

Преимущества перед скриптовым кодом в main.py:
    - Параллельная обработка фрагментов (concurrent.futures)
    - Подробные callback-хуки для мониторинга прогресса
    - Воспроизводимость: конфиг сохраняется вместе с результатом
    - Тестируемость: каждый этап можно запустить отдельно

Использование:
    pipeline = Pipeline(cfg)
    assembly = pipeline.run(images)

    # Или этап за этапом:
    fragments = pipeline.preprocess(images)
    matrix, entries = pipeline.match(fragments)
    assembly = pipeline.assemble(fragments, entries)
    score = pipeline.verify(assembly)
"""
from __future__ import annotations

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .config import Config
from .models import Fragment, Assembly
from .preprocessing.segmentation import segment_fragment
from .preprocessing.contour import extract_contour
from .preprocessing.orientation import estimate_orientation, rotate_to_upright
from .preprocessing.color_norm import normalize_color
from .algorithms.tangram.inscriber import fit_tangram
from .algorithms.synthesis import compute_fractal_signature, build_edge_signatures
from .matching.compat_matrix import build_compat_matrix
from .assembly.parallel import (
    run_all_methods,
    run_selected,
    pick_best,
    summary_table,
    ALL_METHODS,
)
from .verification.ocr import verify_full_assembly
from .utils.logger import get_logger, PipelineTimer
from .utils.event_bus import EventBus, make_event_bus
from .scoring.score_normalizer import normalize_score_matrix
from .scoring.threshold_selector import select_threshold, ThresholdConfig
from .scoring.consistency_checker import run_consistency_check, ConsistencyReport
from .io.image_loader import load_from_directory
from .io.result_exporter import export_result, ExportConfig, AssemblyResult
from .algorithms.bridge import (
    get_algorithm,
    list_algorithms,
    get_category,
    ALGORITHM_CATEGORIES,
)


class PipelineResult:
    """
    Результат полного прогона пайплайна.
    Содержит сборку, профиль времени и метаданные.
    """

    def __init__(self, assembly: Assembly, timer: PipelineTimer,
                 cfg: Config, n_input: int,
                 consistency_report: Optional[ConsistencyReport] = None,
                 verification_report=None):
        self.assembly              = assembly
        self.timer                 = timer
        self.cfg                   = cfg
        self.n_input               = n_input
        self.n_placed              = len(assembly.placements)
        self.timestamp             = time.strftime("%Y-%m-%d %H:%M:%S")
        self.consistency_report    = consistency_report
        # VerificationReport из VerificationSuite (если был запущен)
        self.verification_report   = verification_report

    def summary(self) -> str:
        lines = [
            f"=== Результат пайплайна ({self.timestamp}) ===",
            f"  Фрагментов на входе: {self.n_input}",
            f"  Размещено:           {self.n_placed}",
            f"  Score (уверенность сборки): {self.assembly.total_score:.1%}",
            f"  OCR-связность:       {self.assembly.ocr_score:.1%}",
            f"  Метод:               {self.cfg.assembly.method}",
        ]
        if self.consistency_report is not None:
            cr = self.consistency_report
            status = "OK" if cr.is_consistent else f"{cr.n_errors} ошибок"
            lines.append(f"  Согласованность:     {status}  "
                          f"({cr.n_warnings} предупреждений)")
        if self.verification_report is not None:
            vr = self.verification_report
            lines.append(f"  Верификация (suite): {vr.final_score:.1%}  "
                          f"({len(vr.results)} валидаторов)")
        lines += ["", self.timer.report()]
        return "\n".join(lines)

    def export(self, fmt: str = "json",
               output_path: Optional[str] = None) -> Optional[str]:
        """Экспортировать результат сборки через puzzle_reconstruction.io.

        Args:
            fmt:         Формат ('json', 'csv', 'text', 'summary', 'image').
            output_path: Путь к файлу (None → только в памяти).

        Returns:
            Строку (JSON/CSV/текст) или None для формата 'image'.
        """
        asm   = self.assembly
        frags = asm.fragments or []

        positions: List[Tuple[int, int]] = []
        sizes: List[Tuple[int, int]]     = []
        for frag in frags:
            pos = (0, 0)
            if frag.position is not None:
                arr = np.asarray(frag.position).ravel()
                if len(arr) >= 2:
                    pos = (int(arr[0]), int(arr[1]))
            positions.append(pos)
            h, w = frag.image.shape[:2] if frag.image is not None else (1, 1)
            sizes.append((w, h))

        canvas_w = max(
            (p[0] + s[0] for p, s in zip(positions, sizes)), default=1
        )
        canvas_h = max(
            (p[1] + s[1] for p, s in zip(positions, sizes)), default=1
        )

        ar = AssemblyResult(
            fragment_ids=[f.fragment_id for f in frags],
            positions=positions,
            sizes=sizes,
            canvas_w=max(canvas_w, 1),
            canvas_h=max(canvas_h, 1),
            metadata={
                "total_score": asm.total_score,
                "ocr_score":   asm.ocr_score,
                "method":      asm.method,
                "timestamp":   self.timestamp,
            },
        )
        return export_result(ar, ExportConfig(fmt=fmt, output_path=output_path))


# ─── Главный класс ────────────────────────────────────────────────────────

class Pipeline:
    """
    Полный пайплайн восстановления документа.

    Args:
        cfg:            Конфигурация всех этапов.
        n_workers:      Число потоков для параллельной обработки фрагментов.
                        1 = последовательно, -1 = os.cpu_count().
        on_progress:    Опциональный callback: on_progress(stage, done, total).
        log_level:      Уровень логирования (logging.DEBUG / INFO / WARNING).
        log_file:       Файл для записи лога (None = только консоль).
    """

    def __init__(self,
                 cfg:          Optional[Config] = None,
                 n_workers:    int = 4,
                 on_progress:  Optional[Callable[[str, int, int], None]] = None,
                 log_level:    int = logging.INFO,
                 log_file:     Optional[str] = None,
                 algorithms:   Optional[List[str]] = None):
        """
        Args:
            algorithms: Список алгоритмов из Bridge №5 для активации на уровне
                        фрагмента (fragment-level). None = отключено.
                        Пример: ["fragment_classifier", "fragment_quality",
                                 "fourier_descriptor", "shape_context"]
        """
        self.cfg         = cfg or Config.default()
        self.n_workers   = n_workers
        self.on_progress = on_progress
        self.log         = get_logger("pipeline", level=log_level, log_file=log_file)
        self._timer      = PipelineTimer()
        self._bus: Optional[EventBus] = None

        # Bridge №5: алгоритмы по уровням
        # Приоритет: явный параметр algorithms= > cfg.algorithms
        alg_cfg = self.cfg.algorithms
        if algorithms is not None:
            # Плоский список → сортируем по категории через get_category()
            self._fragment_algorithms: List[str] = [
                n for n in algorithms if get_category(n) == "fragment"
            ]
            self._pair_algorithms: List[str] = [
                n for n in algorithms if get_category(n) == "pair"
            ]
            self._assembly_algorithms: List[str] = [
                n for n in algorithms if get_category(n) == "assembly"
            ]
        else:
            # Читаем из cfg.algorithms (раздельно по уровням)
            self._fragment_algorithms = list(alg_cfg.fragment)
            self._pair_algorithms     = list(alg_cfg.pair)
            self._assembly_algorithms = list(alg_cfg.assembly)

        total = (len(self._fragment_algorithms)
                 + len(self._pair_algorithms)
                 + len(self._assembly_algorithms))
        if total:
            self.log.info(
                "Bridge №5 (algorithms): fragment=%d, pair=%d, assembly=%d",
                len(self._fragment_algorithms),
                len(self._pair_algorithms),
                len(self._assembly_algorithms),
            )

        try:
            self._bus = make_event_bus()
        except Exception:
            pass  # event_bus опционален

    # ── Полный прогон ────────────────────────────────────────────────────

    def run(self, images: Union[List[np.ndarray], str]) -> PipelineResult:
        """
        Запускает полный пайплайн от сырых изображений до верифицированной сборки.

        Args:
            images: Список BGR изображений фрагментов **или** путь к директории
                    с изображениями (будет загружена через puzzle_reconstruction.io).

        Returns:
            PipelineResult с готовой сборкой и отчётом согласованности.
        """
        if isinstance(images, str):
            loaded = load_from_directory(images)
            images = [li.data for li in loaded]

        n_input = len(images)
        self.log.info(f"Старт пайплайна: {n_input} фрагментов, "
                       f"метод={self.cfg.assembly.method}")

        with self._timer.measure("препроцессинг"):
            fragments = self.preprocess(images)

        if not fragments:
            self.log.error("Ни один фрагмент не прошёл препроцессинг")
            return PipelineResult(
                Assembly(fragments=[], placements={}, compat_matrix=np.array([])),
                self._timer, self.cfg, n_input,
            )

        with self._timer.measure("сопоставление"):
            matrix, entries = self.match(fragments)

        with self._timer.measure("сборка"):
            assembly = self.assemble(fragments, entries)
        assembly.compat_matrix = matrix
        if assembly.fragments is None:
            assembly.fragments = fragments

        with self._timer.measure("верификация"):
            assembly.ocr_score = self.verify(assembly)

        # VerificationSuite (21 валидатор) — если включены в конфиге
        verification_report = None
        if self.cfg.verification.validators:
            with self._timer.measure("verification_suite"):
                verification_report = self.verify_suite(assembly)

        with self._timer.measure("согласованность"):
            consistency_report = self._consistency_check(assembly, fragments)

        result = PipelineResult(
            assembly, self._timer, self.cfg, n_input,
            consistency_report=consistency_report,
            verification_report=verification_report,
        )
        self.log.info(f"Готово. Score={assembly.total_score:.1%}  "
                       f"OCR={assembly.ocr_score:.1%}")
        return result

    # ── Этап 1: Препроцессинг ────────────────────────────────────────────

    def preprocess(self, images: List[np.ndarray]) -> List[Fragment]:
        """
        Обрабатывает список изображений: нормализация, сегментация, описание.

        Параллельная версия: обрабатывает n_workers фрагментов одновременно
        (используя потоки; GIL освобождается внутри OpenCV/scipy для numpy).

        Args:
            images: BGR изображения фрагментов.

        Returns:
            Список Fragment со всеми заполненными полями.
        """
        n = len(images)
        self.log.info(f"  Препроцессинг {n} фрагментов  "
                       f"(workers={self.n_workers})")

        if self.n_workers == 1 or n <= 2:
            # Последовательно
            results = []
            for idx, img in enumerate(images):
                frag = self._process_one(idx, img)
                if frag is not None:
                    results.append(frag)
                self._progress("препроцессинг", idx + 1, n)
            return results

        # Параллельно через ThreadPoolExecutor
        results: dict[int, Optional[Fragment]] = {}
        with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            futures = {pool.submit(self._process_one, idx, img): idx
                       for idx, img in enumerate(images)}
            done = 0
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    self.log.warning(f"    Фрагмент #{idx}: {e}")
                    results[idx] = None
                done += 1
                self._progress("препроцессинг", done, n)

        # Возвращаем в исходном порядке
        return [results[i] for i in range(n) if results.get(i) is not None]

    def _process_one(self, idx: int, img: np.ndarray) -> Optional[Fragment]:
        """Полная обработка одного изображения → Fragment."""
        try:
            # 1. Нормализация цвета (CLAHE + white balance)
            if self.cfg.segmentation.method != "grabcut":  # GrabCut чувствителен к цвету
                img = normalize_color(img)

            # 2. Коррекция ориентации
            mask_raw = segment_fragment(img, method=self.cfg.segmentation.method)
            angle    = estimate_orientation(img, mask_raw)
            if abs(angle) > 0.05:
                img      = rotate_to_upright(img, angle)
                mask_raw = segment_fragment(img, method=self.cfg.segmentation.method)

            # 3. Контур
            contour = extract_contour(mask_raw)
            if len(contour) < 4:
                self.log.debug(f"    #{idx}: контур слишком короткий, пропуск")
                return None

            # 4. Создаём Fragment
            frag = Fragment(fragment_id=idx, image=img, mask=mask_raw, contour=contour)

            # 5. Танграм
            frag.tangram = fit_tangram(contour)

            # 6. Фрактал
            frag.fractal = compute_fractal_signature(contour)

            # 7. Синтез EdgeSignature
            frag.edges = build_edge_signatures(
                frag,
                alpha=self.cfg.synthesis.alpha,
                n_sides=self.cfg.synthesis.n_sides,
                n_points=self.cfg.synthesis.n_points,
            )
            self.log.debug(f"    #{idx}  краёв={len(frag.edges)}  "
                            f"FD={frag.fractal.fd_box:.3f}  "
                            f"форма={frag.tangram.shape_class.value}")

            # 8. Bridge №5: опциональные fragment-level алгоритмы
            self._run_fragment_algorithms(idx, frag)

            return frag

        except Exception as e:
            self.log.debug(f"    #{idx}: {type(e).__name__}: {e}")
            return None

    def _run_fragment_algorithms(self, idx: int, frag: "Fragment") -> None:
        """
        Запускает алгоритмы Bridge №5 (fragment-level) для одного фрагмента.

        Каждый алгоритм вызывается с img и/или contour фрагмента.
        Результаты логируются; фрагмент не изменяется (алгоритмы read-only).
        """
        if not self._fragment_algorithms:
            return
        for name in self._fragment_algorithms:
            fn = get_algorithm(name)
            if fn is None:
                continue
            try:
                if name in ("fragment_classifier", "line_detector",
                            "word_segmentation", "region_segmenter"):
                    result = fn(frag.image)
                elif name in ("fragment_quality",):
                    result = fn(frag.image, frag.mask)
                elif name in ("boundary_descriptor", "fourier_descriptor",
                              "shape_context", "contour_smoother"):
                    if frag.contour is not None and len(frag.contour) >= 4:
                        result = fn(frag.contour)
                    else:
                        continue
                elif name in ("contour_tracker", "region_splitter"):
                    if frag.mask is not None:
                        result = fn(frag.mask)
                    else:
                        continue
                elif name in ("color_palette", "color_space",
                              "texture_descriptor", "gradient_flow",
                              "edge_extractor", "rotation_estimator"):
                    result = fn(frag.image)
                else:
                    result = fn(frag.image)
                self.log.debug("    #%d  %s → %s", idx, name, type(result).__name__)
            except Exception as exc:
                self.log.debug("    #%d  %s failed: %s", idx, name, exc)

    def _run_pair_algorithms(
        self,
        entries: list,
        fragments: List["Fragment"],
    ) -> list:
        """
        Запускает pair-level алгоритмы Bridge №5 после Otsu-фильтрации.

        edge_filter     — дополнительная фильтрация/дедупликация пар
        score_aggregator — диагностика распределения оценок
        seam_evaluator  — оценка качества шва для топ-N пар (read-only)
        sift_matcher    — SIFT-верификация для топ-N пар (read-only)
        Все остальные: диагностический прогон по подмножеству пар.

        Returns:
            Список пар (возможно, отфильтрованный edge_filter).
        """
        if not self._pair_algorithms or not entries:
            return entries

        frag_by_id = {f.fragment_id: f for f in fragments}

        for name in self._pair_algorithms:
            fn = get_algorithm(name)
            if fn is None:
                continue
            try:
                if name == "edge_filter":
                    filtered = fn(entries)
                    if filtered is not None:
                        entries = filtered
                        self.log.info(
                            "  Bridge №5 edge_filter: %d пар", len(entries)
                        )

                elif name == "score_aggregator":
                    scores = [e.score for e in entries]
                    if scores:
                        fn(scores)
                        self.log.debug(
                            "  Bridge №5 score_aggregator: n=%d", len(scores)
                        )

                elif name in ("seam_evaluator", "edge_scorer"):
                    # Прогон по топ-50 парам для диагностики / логирования
                    top = sorted(entries, key=lambda e: e.score, reverse=True)[:50]
                    for entry in top:
                        fi = frag_by_id.get(getattr(entry.edge_i, "fragment_id", -1))
                        fj = frag_by_id.get(getattr(entry.edge_j, "fragment_id", -1))
                        if fi is None or fj is None:
                            continue
                        si = getattr(entry.edge_i, "side", None)
                        sj = getattr(entry.edge_j, "side", None)
                        fn(fi.image, si, fj.image, sj)

                elif name in ("sift_matcher", "patch_matcher"):
                    top = sorted(entries, key=lambda e: e.score, reverse=True)[:20]
                    for entry in top:
                        fi = frag_by_id.get(getattr(entry.edge_i, "fragment_id", -1))
                        fj = frag_by_id.get(getattr(entry.edge_j, "fragment_id", -1))
                        if fi is None or fj is None:
                            continue
                        fn(fi.image, fj.image)

                elif name in ("fragment_aligner", "patch_aligner"):
                    top = sorted(entries, key=lambda e: e.score, reverse=True)[:10]
                    for entry in top:
                        fi = frag_by_id.get(getattr(entry.edge_i, "fragment_id", -1))
                        fj = frag_by_id.get(getattr(entry.edge_j, "fragment_id", -1))
                        if fi is None or fj is None:
                            continue
                        fn(fi.image, fj.image)

                else:
                    self.log.debug("  Bridge №5 pair/%s: нет обработчика", name)

            except Exception as exc:
                self.log.debug("  Bridge №5 pair/%s failed: %s", name, exc)

        return entries

    # ── Этап 2: Сопоставление ────────────────────────────────────────────

    def match(self, fragments: List[Fragment]
              ) -> Tuple[np.ndarray, list]:
        """
        Строит матрицу совместимости краёв.

        Нормализует матрицу оценок через scoring.score_normalizer и
        вычисляет адаптивный порог методом Отсу через scoring.threshold_selector.

        Returns:
            (normalized_compat_matrix, filtered_entries)
        """
        n_edges = sum(len(f.edges) for f in fragments)
        self.log.info(f"  Матрица совместимости: {n_edges} краёв, "
                       f"порог={self.cfg.matching.threshold}")
        matrix, entries = build_compat_matrix(
            fragments, threshold=self.cfg.matching.threshold
        )
        self.log.info(f"  Найдено {len(entries)} пар выше порога")

        # Нормализация матрицы оценок
        norm_result = normalize_score_matrix(matrix)
        matrix = norm_result.data

        # Адаптивный порог на основе распределения оценок (метод Отсу)
        if entries:
            entry_scores = np.array([e.score for e in entries])
            thr_result = select_threshold(
                entry_scores, ThresholdConfig(method="otsu")
            )
            adaptive_thr = thr_result.threshold
            entries = [e for e in entries if e.score >= adaptive_thr]
            self.log.info(
                f"  Адаптивный порог Отсу: {adaptive_thr:.4f}  "
                f"→ {len(entries)} пар"
            )

        # Bridge №5: pair-level алгоритмы
        entries = self._run_pair_algorithms(entries, fragments)

        return matrix, entries

    # ── Этап 3: Сборка ───────────────────────────────────────────────────

    def assemble(self, fragments: List[Fragment], entries: list) -> Assembly:
        """
        Собирает фрагменты выбранным в конфиге методом через реестр parallel.py.
        Поддерживает все 8 алгоритмов + auto/all режимы.
        """
        method = self.cfg.assembly.method
        self.log.info(f"  Сборка: метод={method}")
        self._emit("assembly_start", {"method": method, "n_fragments": len(fragments)})

        cfg = self.cfg.assembly
        kwargs = dict(
            beam_width=cfg.beam_width,
            n_iterations=cfg.sa_iter,
            n_simulations=cfg.mcts_sim,
            seed=cfg.seed,
        )

        if method == "all":
            results = run_all_methods(
                fragments, entries,
                methods=ALL_METHODS,
                timeout=cfg.auto_timeout,
                n_workers=min(4, len(ALL_METHODS)),
                **kwargs,
            )
            self.log.info("\n" + summary_table(results))
            asm = pick_best(results)
            if asm is None:
                raise RuntimeError("Все методы завершились с ошибкой")

        elif method == "auto":
            methods = self._auto_methods(len(fragments))
            self.log.info(f"  Авто-выбор ({len(fragments)} фрагментов): {methods}")
            results = run_all_methods(
                fragments, entries,
                methods=methods,
                timeout=cfg.auto_timeout,
                **kwargs,
            )
            self.log.info("\n" + summary_table(results))
            asm = pick_best(results)
            if asm is None:
                raise RuntimeError("Все авто-выбранные методы завершились с ошибкой")

        else:
            results = run_selected(fragments, entries, methods=[method], **kwargs)
            if not results or not results[0].success:
                err = results[0].error if results else "нет результата"
                raise RuntimeError(f"Метод '{method}' завершился с ошибкой: {err}")
            asm = results[0].assembly

        self.log.info(f"  Score: {asm.total_score:.4f}")
        self._emit("assembly_done", {"score": asm.total_score, "method": method})

        # Bridge №5: assembly-level постобработка
        asm = self._run_assembly_algorithms(asm, fragments)

        return asm

    def _run_assembly_algorithms(
        self,
        asm: "Assembly",
        fragments: List["Fragment"],
    ) -> "Assembly":
        """
        Запускает assembly-level алгоритмы Bridge №5 после основной сборки.

        path_planner       — вычисляет кратчайший путь по матрице совместимости
                             (диагностика; логирует стоимость пути)
        position_estimator — уточняет позиции фрагментов по попарным смещениям
        overlap_resolver   — разрешает перекрытия между размещёнными фрагментами
        descriptor_aggregator / descriptor_combiner — агрегация дескрипторов
                             (диагностика; логирует статистику)

        Returns:
            Assembly (позиции могут быть обновлены overlap_resolver /
            position_estimator если они активированы).
        """
        if not self._assembly_algorithms:
            return asm

        for name in self._assembly_algorithms:
            fn = get_algorithm(name)
            if fn is None:
                continue
            try:
                if name == "path_planner":
                    # shortest_path(score_matrix, start, end) — диагностика
                    if asm.compat_matrix is not None and asm.compat_matrix.size > 1:
                        n = asm.compat_matrix.shape[0]
                        result = fn(asm.compat_matrix, 0, n - 1)
                        self.log.debug(
                            "  Bridge №5 path_planner: cost=%s",
                            getattr(result, "cost", result),
                        )

                elif name == "position_estimator":
                    # estimate_positions(offset_graph, root) — уточнение позиций
                    # Строим offset_graph из placements если он dict
                    placements = asm.placements
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
                            from .algorithms.position_estimator import build_offset_graph
                            graph = build_offset_graph(list(offsets.keys()),
                                                       list(offsets.values()))
                            refined = fn(graph, ids[0])
                            self.log.debug(
                                "  Bridge №5 position_estimator: %d позиций", len(refined)
                            )

                elif name == "overlap_resolver":
                    # resolve_all_conflicts(state, contours, ...) — правка перекрытий
                    contours = {
                        f.fragment_id: f.contour
                        for f in fragments
                        if f.contour is not None
                    }
                    if contours:
                        fn(asm, contours)
                        self.log.debug("  Bridge №5 overlap_resolver: завершён")

                elif name in ("descriptor_aggregator", "descriptor_combiner"):
                    # Агрегируем css_vec из EdgeSignature всех фрагментов
                    vecs = []
                    for frag in fragments:
                        for edge in (frag.edges or []):
                            v = getattr(edge, "css_vec", None)
                            if v is not None and hasattr(v, "__len__"):
                                vecs.append(np.asarray(v, dtype=float).ravel())
                    if vecs:
                        fn(vecs)
                        self.log.debug(
                            "  Bridge №5 %s: обработано %d дескрипторов", name, len(vecs)
                        )

                else:
                    self.log.debug("  Bridge №5 assembly/%s: нет обработчика", name)

            except Exception as exc:
                self.log.debug("  Bridge №5 assembly/%s failed: %s", name, exc)

        return asm

    @staticmethod
    def _auto_methods(n_fragments: int) -> list:
        """Выбирает методы по числу фрагментов."""
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

    # ── Этап 4: Верификация ──────────────────────────────────────────────

    def verify(self, assembly: Assembly) -> float:
        """
        Запускает OCR и возвращает оценку текстовой связности.
        Возвращает 0.0 если OCR отключён или недоступен.
        """
        if not self.cfg.verification.run_ocr:
            return 0.0
        try:
            score = verify_full_assembly(assembly, lang=self.cfg.verification.ocr_lang)
            self.log.info(f"  OCR-связность: {score:.1%}")
            return score
        except Exception as e:
            self.log.debug(f"  OCR ошибка: {e}")
            return 0.0

    def verify_suite(self, assembly: Assembly,
                     validators: Optional[List[str]] = None):
        """
        Запускает VerificationSuite и возвращает VerificationReport.

        Args:
            assembly:   Собранный документ.
            validators: Список имён валидаторов. None → берёт из
                        cfg.verification.validators; если тот тоже пуст —
                        запускает все 21 валидатора (run_all).

        Returns:
            VerificationReport с результатами всех запрошенных валидаторов.
        """
        try:
            from .verification.suite import VerificationSuite, all_validator_names

            names = validators or self.cfg.verification.validators or None
            if names:
                suite  = VerificationSuite(validators=names)
                report = suite.run(assembly)
            else:
                suite  = VerificationSuite()
                report = suite.run_all(assembly)

            self.log.info(
                f"  VerificationSuite: {len(report.results)} валидаторов  "
                f"score={report.final_score:.1%}"
            )
            return report
        except Exception as exc:
            self.log.debug(f"  VerificationSuite ошибка: {exc}")
            from .verification.suite import VerificationReport
            return VerificationReport(results=[], final_score=0.0)

    # ── Согласованность ──────────────────────────────────────────────────

    def _consistency_check(
        self,
        assembly: Assembly,
        fragments: List[Fragment],
    ) -> Optional[ConsistencyReport]:
        """Проверяет согласованность сборки через scoring.consistency_checker."""
        try:
            placed_frags = assembly.fragments or fragments
            frag_ids     = [f.fragment_id for f in placed_frags]
            expected_ids = [f.fragment_id for f in fragments]

            positions: List[Tuple[int, int]] = []
            sizes: List[Tuple[int, int]]     = []
            for frag in placed_frags:
                pos = (0, 0)
                if frag.position is not None:
                    arr = np.asarray(frag.position).ravel()
                    if len(arr) >= 2:
                        pos = (int(arr[0]), int(arr[1]))
                positions.append(pos)
                h, w = frag.image.shape[:2] if frag.image is not None else (1, 1)
                sizes.append((w, h))

            canvas_w = max(
                (p[0] + s[0] for p, s in zip(positions, sizes)), default=1
            )
            canvas_h = max(
                (p[1] + s[1] for p, s in zip(positions, sizes)), default=1
            )

            report = run_consistency_check(
                fragment_ids=frag_ids,
                expected_ids=expected_ids,
                positions=positions,
                sizes=sizes,
                canvas_w=max(canvas_w, 1),
                canvas_h=max(canvas_h, 1),
            )
            status = "OK" if report.is_consistent else f"{report.n_errors} ошибок"
            self.log.info(f"  Согласованность: {status}  "
                           f"({report.n_warnings} предупреждений)")
            return report
        except Exception as e:
            self.log.debug(f"  Consistency check ошибка: {e}")
            return None

    # ── Утилиты ──────────────────────────────────────────────────────────

    def _progress(self, stage: str, done: int, total: int) -> None:
        if self.on_progress is not None:
            try:
                self.on_progress(stage, done, total)
            except Exception:
                pass
        self._emit("progress", {"stage": stage, "done": done, "total": total})

    def _emit(self, event: str, data: dict) -> None:
        """Публикует событие в event_bus (если доступен)."""
        if self._bus is not None:
            try:
                from .utils.event_bus import log_event
                log_event(self._bus, event, data)
            except Exception:
                pass
