"""
Верификация и оценка качества восстановленного документа.

Модули:
    ocr            — OCR-верификация текстовой связности (pytesseract)
    metrics        — Количественные метрики качества сборки (NA, DC, RMSE, ...)
    report         — Генератор отчётов (JSON / Markdown / HTML)
    text_coherence — N-граммная языковая модель, оценка стыков по биграммам
"""
from .ocr import verify_full_assembly, render_assembly_image
from .metrics import (
    evaluate_reconstruction,
    compare_methods,
    ReconstructionMetrics,
    BenchmarkResult,
)
from .report import build_report, Report
from .text_coherence import (
    NGramModel,
    TextCoherenceScorer,
    score_assembly_coherence,
    seam_bigram_score,
    word_boundary_score,
    build_ngram_model,
)

__all__ = [
    "verify_full_assembly",
    "render_assembly_image",
    "evaluate_reconstruction",
    "compare_methods",
    "ReconstructionMetrics",
    "BenchmarkResult",
    "build_report",
    "Report",
    "NGramModel",
    "TextCoherenceScorer",
    "score_assembly_coherence",
    "seam_bigram_score",
    "word_boundary_score",
    "build_ngram_model",
]
