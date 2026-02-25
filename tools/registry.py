"""
Реестр инструментов (Bridge #7) — единая точка входа для 6 standalone-инструментов.

Каждый инструмент представлен дескриптором ToolInfo, который хранит:
  - имя, описание, список параметров
  - ссылку на основную функцию (lazy-импорт)

Использование:
    from tools.registry import get_tool, list_tools, run_tool

    # Список доступных инструментов:
    for name, info in list_tools().items():
        print(f"  {name}: {info.description}")

    # Запуск профилировщика:
    run_tool("profile", n_fragments=8, verbose=True)

    # Запуск генератора тестовых данных:
    import cv2
    img = cv2.imread("document.png")
    run_tool("tear", image=img, n_pieces=6, output_dir="./frags")

    # Запуск бенчмарка:
    run_tool("benchmark", methods=["beam", "sa"], n_pieces_list=[4, 8])

Инструменты:
    benchmark      — run_benchmark(n_pieces_list, methods, n_trials, noise, output_path)
    evaluate       — run_evaluation(methods, n_pieces_list, n_trials, noise, output_dir, ...)
    mix            — mix_from_generated(n_docs, n_pieces, output_dir, noise_level, shuffle)
    profile        — run_profile(n_fragments, n_iters, verbose) → ProfileResult
    serve          — запускает Flask REST API сервер
    tear           — tear_document(image, n_pieces, noise_level, seed) → List[np.ndarray]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─── Дескриптор инструмента ──────────────────────────────────────────────────

@dataclass
class ToolInfo:
    """Метаданные одного инструмента."""
    name:        str
    description: str
    params:      List[str]             = field(default_factory=list)
    _loader:     Optional[Callable]    = field(default=None, repr=False)
    _fn:         Optional[Callable]    = field(default=None, repr=False)

    def load(self) -> Optional[Callable]:
        """Возвращает основную callable инструмента (lazy-import)."""
        if self._fn is not None:
            return self._fn
        if self._loader is not None:
            try:
                self._fn = self._loader()
            except (Exception, SystemExit) as exc:
                logger.warning("tool %r: не удалось загрузить: %s", self.name, exc)
        return self._fn

    def run(self, **kwargs) -> Any:
        """Запускает инструмент с переданными параметрами."""
        fn = self.load()
        if fn is None:
            raise RuntimeError(f"Инструмент {self.name!r} недоступен")
        return fn(**kwargs)


# ─── Реестр ──────────────────────────────────────────────────────────────────

def build_tool_registry() -> Dict[str, ToolInfo]:
    """
    Строит словарь {tool_name: ToolInfo} для всех 6 инструментов.

    Все import-ы ленивые (выполняются при первом вызове .load()/.run()),
    поэтому недоступные зависимости (Flask, cProfile и т.д.) не блокируют
    инициализацию реестра.

    Returns:
        Словарь инструментов.
    """
    registry: Dict[str, ToolInfo] = {}

    # ── benchmark ──────────────────────────────────────────────────────────

    def _load_benchmark():
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from tools.benchmark import run_benchmark
        return run_benchmark

    registry["benchmark"] = ToolInfo(
        name        = "benchmark",
        description = ("Автоматический бенчмарк методов сборки. "
                       "Генерирует синтетические документы, рвёт их на фрагменты, "
                       "запускает каждый метод и сравнивает результаты."),
        params      = ["n_pieces_list", "methods", "n_trials", "noise",
                       "output_path"],
        _loader     = _load_benchmark,
    )

    # ── evaluate ───────────────────────────────────────────────────────────

    def _load_evaluate():
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from tools.evaluate import run_evaluation
        return run_evaluation

    registry["evaluate"] = ToolInfo(
        name        = "evaluate",
        description = ("Оценка качества системы на синтетических данных. "
                       "Генерирует known ground truth, запускает пайплайн "
                       "и вычисляет метрики точности."),
        params      = ["methods", "n_pieces_list", "n_trials", "noise",
                       "output_dir", "save_html", "save_md"],
        _loader     = _load_evaluate,
    )

    # ── mix ────────────────────────────────────────────────────────────────

    def _load_mix():
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from tools.mix_documents import mix_from_generated
        return mix_from_generated

    registry["mix"] = ToolInfo(
        name        = "mix",
        description = ("Генератор смешанных фрагментов из нескольких документов. "
                       "Создаёт тестовые данные для проверки кластеризации: "
                       "N₁ фрагментов документа A + N₂ фрагментов документа B."),
        params      = ["n_docs", "n_pieces", "output_dir", "noise_level", "shuffle"],
        _loader     = _load_mix,
    )

    # ── profile ────────────────────────────────────────────────────────────

    def _load_profile():
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from tools.profile import run_profile
        return run_profile

    registry["profile"] = ToolInfo(
        name        = "profile",
        description = ("Профилировщик производительности пайплайна. "
                       "Измеряет время каждого этапа на синтетических данных "
                       "и выводит сводную таблицу."),
        params      = ["n_fragments", "n_iters", "verbose"],
        _loader     = _load_profile,
    )

    # ── serve ──────────────────────────────────────────────────────────────

    def _load_serve():
        # Проверяем зависимости до import server (который вызывает sys.exit без них)
        try:
            import flask  # noqa: F401
        except ImportError:
            logger.warning("tool 'serve': Flask не установлен (pip install flask)")
            return None
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from tools.server import app
        def _run_server(host: str = "0.0.0.0", port: int = 5000,
                        debug: bool = False) -> None:
            app.run(host=host, port=port, debug=debug)
        return _run_server

    registry["serve"] = ToolInfo(
        name        = "serve",
        description = ("Запускает REST API сервер (Flask). "
                       "Предоставляет HTTP-эндпоинты для загрузки фрагментов "
                       "и получения результата сборки."),
        params      = ["host", "port", "debug"],
        _loader     = _load_serve,
    )

    # ── tear ───────────────────────────────────────────────────────────────

    def _load_tear():
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from tools.tear_generator import tear_document
        return tear_document

    registry["tear"] = ToolInfo(
        name        = "tear",
        description = ("Генератор синтетических тестовых данных. "
                       "«Рвёт» изображение документа на N фрагментов "
                       "с реалистичными рваными краями."),
        params      = ["image", "n_pieces", "noise_level", "seed"],
        _loader     = _load_tear,
    )

    return registry


# ─── Глобальный реестр ───────────────────────────────────────────────────────

TOOL_REGISTRY: Dict[str, ToolInfo] = {}


def _ensure_registry() -> None:
    global TOOL_REGISTRY
    if not TOOL_REGISTRY:
        TOOL_REGISTRY = build_tool_registry()


def list_tools() -> Dict[str, ToolInfo]:
    """
    Возвращает словарь всех зарегистрированных инструментов.

    Returns:
        Dict[name → ToolInfo].
    """
    _ensure_registry()
    return dict(TOOL_REGISTRY)


def get_tool(name: str) -> Optional[ToolInfo]:
    """
    Возвращает ToolInfo по имени инструмента.

    Args:
        name: Имя инструмента ('benchmark', 'evaluate', 'mix',
              'profile', 'serve', 'tear').

    Returns:
        ToolInfo или None если инструмент не найден.
    """
    _ensure_registry()
    info = TOOL_REGISTRY.get(name)
    if info is None:
        logger.warning("Инструмент %r не найден. Доступно: %s",
                       name, sorted(TOOL_REGISTRY))
    return info


def run_tool(name: str, **kwargs: Any) -> Any:
    """
    Запускает инструмент по имени с переданными kwargs.

    Args:
        name:    Имя инструмента.
        **kwargs: Параметры инструмента.

    Returns:
        Результат инструмента (зависит от конкретного инструмента).

    Raises:
        KeyError:    Инструмент не найден.
        RuntimeError: Инструмент недоступен (зависимость не установлена).
    """
    _ensure_registry()
    info = TOOL_REGISTRY.get(name)
    if info is None:
        available = sorted(TOOL_REGISTRY)
        raise KeyError(
            f"Инструмент {name!r} не найден. Доступно: {available}"
        )
    return info.run(**kwargs)
