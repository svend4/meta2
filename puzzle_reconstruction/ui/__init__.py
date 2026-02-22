"""
Интерактивный интерфейс для просмотра и редактирования сборки.

Модули:
    viewer — AssemblyViewer (OpenCV-окно) и функция show()
"""
from .viewer import AssemblyViewer, show

__all__ = ["AssemblyViewer", "show"]
