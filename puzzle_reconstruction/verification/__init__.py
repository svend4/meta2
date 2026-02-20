"""
Верификация и оценка качества восстановленного документа.

Модули:
    ocr      — OCR-верификация текстовой связности (pytesseract)
    metrics  — Количественные метрики качества сборки (NA, DC, RMSE, ...)
"""
from .ocr import verify_full_assembly, render_assembly_image
from .metrics import (
    evaluate_reconstruction,
    compare_methods,
    ReconstructionMetrics,
    BenchmarkResult,
)

__all__ = [
    "verify_full_assembly",
    "render_assembly_image",
    "evaluate_reconstruction",
    "compare_methods",
    "ReconstructionMetrics",
    "BenchmarkResult",
]
