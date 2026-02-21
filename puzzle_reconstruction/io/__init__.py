"""Модули ввода-вывода результатов сборки пазла.

Доступные модули:
    result_exporter — экспорт результатов сборки (ExportConfig, AssemblyResult,
                      to_json, from_json, to_csv, to_text_report,
                      render_annotated_image, summary_table,
                      export_result, batch_export)
    image_loader    — загрузка изображений (LoadConfig, LoadedImage,
                      load_image, load_from_array, list_image_files,
                      batch_load, load_from_directory, resize_image)
    metadata_writer — запись метаданных реконструкции (WriterConfig,
                      MetadataRecord, MetadataCollection, write_json,
                      read_json, write_csv, write_summary,
                      filter_by_score, merge_collections)
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
from .metadata_writer import (
    WriterConfig,
    MetadataRecord,
    MetadataCollection,
    write_json,
    read_json,
    write_csv,
    write_summary,
    filter_by_score,
    merge_collections,
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
    # Запись метаданных реконструкции
    "WriterConfig",
    "MetadataRecord",
    "MetadataCollection",
    "write_json",
    "read_json",
    "write_csv",
    "write_summary",
    "filter_by_score",
    "merge_collections",
]
