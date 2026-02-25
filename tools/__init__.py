"""
Автономные инструменты для работы с системой восстановления документов.

Инструменты:
    benchmark      — автоматический бенчмарк методов сборки
    evaluate       — оценка качества на синтетических данных
    mix_documents  — генератор смешанных фрагментов
    profile        — профилировщик производительности пайплайна
    server         — REST API сервер (Flask)
    tear_generator — генератор тестовых рваных фрагментов

Использование через реестр:
    from tools.registry import get_tool, list_tools
    tool = get_tool("profile")
    tool.run(n_fragments=8, verbose=True)

Использование из CLI:
    python main.py --list-tools
    python main.py --tool profile --n-fragments 8
    python main.py --tool benchmark --methods beam,sa --pieces 4,8,16
    python main.py --tool tear --input doc.png --n-pieces 6 --output frags/
    python main.py --tool serve --port 5000
"""
from .registry import (
    build_tool_registry,
    list_tools,
    get_tool,
    run_tool,
    TOOL_REGISTRY,
)

__all__ = [
    "build_tool_registry",
    "list_tools",
    "get_tool",
    "run_tool",
    "TOOL_REGISTRY",
]
