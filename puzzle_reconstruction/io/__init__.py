"""Модули ввода-вывода результатов сборки пазла.

Доступные модули:
    result_exporter — экспорт результатов сборки (ExportConfig, AssemblyResult,
                      to_json, from_json, to_csv, to_text_report,
                      render_annotated_image, summary_table,
                      export_result, batch_export)
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

__all__ = [
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
]
