"""Модули ввода-вывода результатов сборки пазла.

Доступные модули:
    result_exporter — экспорт результатов сборки (ExportConfig, AssemblyResult,
                      to_json, from_json, to_csv, to_text_report,
                      render_annotated_image, summary_table,
                      export_result, batch_export)
    image_loader    — загрузка изображений (LoadConfig, LoadedImage,
                      load_image, load_from_array, list_image_files,
                      batch_load, load_from_directory, resize_image)
"""
from .result_exporter import (
    ExportConfig,
    AssemblyResult,
    to_json,
    from_json,
    to_csv,
    to_text_report,
    render_annotated_image,
    summary_table,
    export_result,
    batch_export,
)
from .image_loader import (
    LoadConfig,
    LoadedImage,
    load_image,
    load_from_array,
    list_image_files,
    batch_load,
    load_from_directory,
    resize_image,
)

__all__ = [
    # Экспорт результатов
    "ExportConfig",
    "AssemblyResult",
    "to_json",
    "from_json",
    "to_csv",
    "to_text_report",
    "render_annotated_image",
    "summary_table",
    "export_result",
    "batch_export",
    # Загрузка изображений
    "LoadConfig",
    "LoadedImage",
    "load_image",
    "load_from_array",
    "list_image_files",
    "batch_load",
    "load_from_directory",
    "resize_image",
]
