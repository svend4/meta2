"""
pytest конфигурация: пути, общие фикстуры.
"""
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path, чтобы тесты находили пакеты
sys.path.insert(0, str(Path(__file__).parent.parent))
