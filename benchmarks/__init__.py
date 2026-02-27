"""
Benchmarks пакет для puzzle_reconstruction.

Запуск:
    # Только бенчмарки (пропустить обычные тесты)
    python -m pytest benchmarks/ -v -m benchmark

    # Всё, включая memory
    python -m pytest benchmarks/ -v -m "benchmark or memory"

    # Конкретный модуль
    python -m pytest benchmarks/bench_descriptors.py -v
"""
