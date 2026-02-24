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
from typing import Callable, List, Optional, Tuple

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


class PipelineResult:
    """
    Результат полного прогона пайплайна.
    Содержит сборку, профиль времени и метаданные.
    """

    def __init__(self, assembly: Assembly, timer: PipelineTimer,
                 cfg: Config, n_input: int):
        self.assembly  = assembly
        self.timer     = timer
        self.cfg       = cfg
        self.n_input   = n_input
        self.n_placed  = len(assembly.placements)
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    def summary(self) -> str:
        lines = [
            f"=== Результат пайплайна ({self.timestamp}) ===",
            f"  Фрагментов на входе: {self.n_input}",
            f"  Размещено:           {self.n_placed}",
            f"  Score (уверенность сборки): {self.assembly.total_score:.1%}",
            f"  OCR-связность:       {self.assembly.ocr_score:.1%}",
            f"  Метод:               {self.cfg.assembly.method}",
            "",
            self.timer.report(),
        ]
        return "\n".join(lines)


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
                 log_file:     Optional[str] = None):
        self.cfg         = cfg or Config.default()
        self.n_workers   = n_workers
        self.on_progress = on_progress
        self.log         = get_logger("pipeline", level=log_level, log_file=log_file)
        self._timer      = PipelineTimer()
        self._bus: Optional[EventBus] = None
        try:
            self._bus = make_event_bus()
        except Exception:
            pass  # event_bus опционален

    # ── Полный прогон ────────────────────────────────────────────────────

    def run(self, images: List[np.ndarray]) -> PipelineResult:
        """
        Запускает полный пайплайн от сырых изображений до верифицированной сборки.

        Args:
            images: Список BGR изображений фрагментов.

        Returns:
            PipelineResult с готовой сборкой.
        """
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

        with self._timer.measure("верификация"):
            assembly.ocr_score = self.verify(assembly)

        result = PipelineResult(assembly, self._timer, self.cfg, n_input)
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
            return frag

        except Exception as e:
            self.log.debug(f"    #{idx}: {type(e).__name__}: {e}")
            return None

    # ── Этап 2: Сопоставление ────────────────────────────────────────────

    def match(self, fragments: List[Fragment]
              ) -> Tuple[np.ndarray, list]:
        """
        Строит матрицу совместимости краёв.

        Returns:
            (compat_matrix, sorted_entries)
        """
        n_edges = sum(len(f.edges) for f in fragments)
        self.log.info(f"  Матрица совместимости: {n_edges} краёв, "
                       f"порог={self.cfg.matching.threshold}")
        matrix, entries = build_compat_matrix(
            fragments, threshold=self.cfg.matching.threshold
        )
        self.log.info(f"  Найдено {len(entries)} пар выше порога")
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
