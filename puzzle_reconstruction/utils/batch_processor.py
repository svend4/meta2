"""Обобщённый пакетный процессор для пайплайна восстановления.

Модуль предоставляет структуры и функции для последовательной обработки
элементов пакетами с отслеживанием успехов, ошибок и повторных попыток.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


# ─── ProcessConfig ────────────────────────────────────────────────────────────

@dataclass
class ProcessConfig:
    """Параметры пакетного процессора.

    Атрибуты:
        batch_size:  Размер одного пакета (>= 1).
        max_retries: Максимальное число повторных попыток при ошибке (>= 0).
        stop_on_error: Остановить обработку при первой ошибке.
        verbose:     Вывод отладочной информации.
    """

    batch_size: int = 32
    max_retries: int = 0
    stop_on_error: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(
                f"batch_size должен быть >= 1, получено {self.batch_size}"
            )
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries должен быть >= 0, получено {self.max_retries}"
            )


# ─── ProcessItem ──────────────────────────────────────────────────────────────

@dataclass
class ProcessItem:
    """Результат обработки одного элемента.

    Атрибуты:
        index:    Индекс элемента во входном списке (>= 0).
        success:  True если обработка завершилась без ошибок.
        result:   Результат обработки (None при ошибке).
        error:    Сообщение об ошибке (None при успехе).
        retries:  Число совершённых повторных попыток (>= 0).
    """

    index: int
    success: bool
    result: Any = None
    error: Optional[str] = None
    retries: int = 0

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(
                f"index должен быть >= 0, получено {self.index}"
            )
        if self.retries < 0:
            raise ValueError(
                f"retries должен быть >= 0, получено {self.retries}"
            )


# ─── BatchSummary ─────────────────────────────────────────────────────────────

@dataclass
class BatchSummary:
    """Сводка по результатам пакетной обработки.

    Атрибуты:
        total:      Общее число элементов (>= 0).
        n_success:  Число успешно обработанных (>= 0).
        n_failed:   Число неудачных обработок (>= 0).
        n_retried:  Число элементов, потребовавших повтора (>= 0).
        items:      Список ProcessItem.
    """

    total: int
    n_success: int
    n_failed: int
    n_retried: int
    items: List[ProcessItem]

    def __post_init__(self) -> None:
        for name, val in (
            ("total", self.total),
            ("n_success", self.n_success),
            ("n_failed", self.n_failed),
            ("n_retried", self.n_retried),
        ):
            if val < 0:
                raise ValueError(f"{name} должен быть >= 0, получено {val}")

    @property
    def success_ratio(self) -> float:
        """Доля успешно обработанных (0 если total == 0)."""
        if self.total == 0:
            return 0.0
        return float(self.n_success) / float(self.total)

    @property
    def total_items(self) -> int:
        """Псевдоним для total (общее число элементов)."""
        return self.total

    @property
    def failed_indices(self) -> List[int]:
        """Индексы элементов, обработка которых завершилась с ошибкой."""
        return [item.index for item in self.items if not item.success]

    @property
    def failed_items(self) -> List[ProcessItem]:
        """Элементы, обработка которых завершилась с ошибкой."""
        return [item for item in self.items if not item.success]

    @property
    def successful_results(self) -> List[Any]:
        """Результаты успешно обработанных элементов."""
        return [item.result for item in self.items if item.success]


# ─── make_processor ───────────────────────────────────────────────────────────

def make_processor(
    batch_size: int = 32,
    max_retries: int = 0,
    stop_on_error: bool = False,
) -> ProcessConfig:
    """Создать конфигурацию процессора.

    Аргументы:
        batch_size:    Размер пакета.
        max_retries:   Максимум повторных попыток.
        stop_on_error: Остановить при первой ошибке.

    Возвращает:
        ProcessConfig.
    """
    return ProcessConfig(
        batch_size=batch_size,
        max_retries=max_retries,
        stop_on_error=stop_on_error,
    )


# ─── process_items ────────────────────────────────────────────────────────────

def process_items(
    items: List[Any],
    fn: Callable[[Any], Any],
    cfg: Optional[ProcessConfig] = None,
) -> BatchSummary:
    """Обработать список элементов функцией fn с возможностью повторов.

    Аргументы:
        items: Список входных элементов.
        fn:    Функция обработки (вызывается как fn(item)).
        cfg:   Параметры процессора (None → ProcessConfig()).

    Возвращает:
        BatchSummary с результатами.
    """
    if cfg is None:
        cfg = ProcessConfig()

    results: List[ProcessItem] = []
    n_success = 0
    n_failed = 0
    n_retried = 0

    for idx, item in enumerate(items):
        retries = 0
        last_error: Optional[str] = None
        success = False
        result = None

        for attempt in range(cfg.max_retries + 1):
            try:
                result = fn(item)
                success = True
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < cfg.max_retries:
                    retries += 1

        pi = ProcessItem(
            index=idx,
            success=success,
            result=result,
            error=last_error if not success else None,
            retries=retries,
        )
        results.append(pi)

        if success:
            n_success += 1
        else:
            n_failed += 1

        if retries > 0:
            n_retried += 1

        if not success and cfg.stop_on_error:
            break

    return BatchSummary(
        total=len(items),
        n_success=n_success,
        n_failed=n_failed,
        n_retried=n_retried,
        items=results,
    )


# ─── filter_successful ────────────────────────────────────────────────────────

def filter_successful(summary: BatchSummary) -> List[ProcessItem]:
    """Вернуть только успешно обработанные элементы.

    Аргументы:
        summary: BatchSummary.

    Возвращает:
        Список ProcessItem с success=True.
    """
    return [item for item in summary.items if item.success]


# ─── retry_failed_items ───────────────────────────────────────────────────────

def retry_failed_items(
    original_items: List[Any],
    summary: BatchSummary,
    fn: Callable[[Any], Any],
    cfg: Optional[ProcessConfig] = None,
) -> BatchSummary:
    """Повторно обработать элементы, завершившиеся с ошибкой.

    Аргументы:
        original_items: Исходный список элементов.
        summary:        BatchSummary с предыдущего прогона.
        fn:             Функция обработки.
        cfg:            Параметры процессора.

    Возвращает:
        Новый BatchSummary только для повторно обработанных элементов.
    """
    failed_indices = summary.failed_indices
    items_to_retry = [original_items[i] for i in failed_indices]
    return process_items(items_to_retry, fn, cfg)


# ─── split_batch ──────────────────────────────────────────────────────────────

def split_batch(
    items: List[Any],
    batch_size: int,
) -> List[List[Any]]:
    """Разбить список на пакеты заданного размера.

    Аргументы:
        items:      Список элементов.
        batch_size: Размер пакета (>= 1).

    Возвращает:
        Список списков-пакетов.

    Исключения:
        ValueError: Если batch_size < 1.
    """
    if batch_size < 1:
        raise ValueError(
            f"batch_size должен быть >= 1, получено {batch_size}"
        )
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


# ─── merge_batch_results ──────────────────────────────────────────────────────

def merge_batch_results(summaries: List[BatchSummary]) -> BatchSummary:
    """Объединить несколько BatchSummary в один.

    Аргументы:
        summaries: Список BatchSummary.

    Возвращает:
        Объединённый BatchSummary.

    Исключения:
        ValueError: Если summaries пуст.
    """
    if not summaries:
        raise ValueError("summaries не должен быть пустым")

    all_items: List[ProcessItem] = []
    offset = 0
    n_success = 0
    n_failed = 0
    n_retried = 0

    for s in summaries:
        for item in s.items:
            new_item = ProcessItem(
                index=item.index + offset,
                success=item.success,
                result=item.result,
                error=item.error,
                retries=item.retries,
            )
            all_items.append(new_item)
        n_success += s.n_success
        n_failed += s.n_failed
        n_retried += s.n_retried
        offset += s.total

    return BatchSummary(
        total=sum(s.total for s in summaries),
        n_success=n_success,
        n_failed=n_failed,
        n_retried=n_retried,
        items=all_items,
    )
