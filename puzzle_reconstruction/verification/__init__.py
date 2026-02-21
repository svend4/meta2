"""
Верификация и оценка качества восстановленного документа.

Модули:
    ocr            — OCR-верификация текстовой связности (pytesseract)
    metrics        — Количественные метрики качества сборки (NA, DC, RMSE, ...)
    report         — Генератор отчётов (JSON / Markdown / HTML)
    text_coherence  — N-граммная языковая модель, оценка стыков по биграммам
    layout_verifier — верификация пространственного расположения (overlap, gap, alignment)
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
from .layout_verifier import (
    ConstraintType,
    LayoutConstraint,
    FragmentBox,
    LayoutVerificationResult,
    build_layout_boxes,
    check_overlaps,
    check_gaps,
    check_column_alignment,
    check_row_alignment,
    check_out_of_bounds,
    check_duplicate_placements,
    verify_layout,
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
    "ConstraintType",
    "LayoutConstraint",
    "FragmentBox",
    "LayoutVerificationResult",
    "build_layout_boxes",
    "check_overlaps",
    "check_gaps",
    "check_column_alignment",
    "check_row_alignment",
    "check_out_of_bounds",
    "check_duplicate_placements",
    "verify_layout",
]
