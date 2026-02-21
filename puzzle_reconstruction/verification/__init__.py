"""
Верификация и оценка качества восстановленного документа.

Модули:
    ocr                 — OCR-верификация текстовой связности (pytesseract)
    metrics             — Количественные метрики качества сборки (NA, DC, RMSE, ...)
    report              — Генератор отчётов (JSON / Markdown / HTML)
    text_coherence      — N-граммная языковая модель, оценка стыков по биграммам
    layout_verifier     — верификация пространственного расположения (overlap, gap, alignment)
    confidence_scorer   — итоговая оценка уверенности в качестве сборки (A–F)
    consistency_checker — проверка согласованности межстрочного интервала, высоты
                          символов, угла текста и полей между фрагментами
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
from .confidence_scorer import (
    ScoreComponent,
    AssemblyConfidence,
    grade_from_score,
    score_edge_compat,
    score_layout,
    score_coverage,
    score_uniqueness,
    score_assembly_score,
    compute_confidence,
)
from .consistency_checker import (
    ConsistencyType,
    ConsistencyViolation,
    ConsistencyResult,
    estimate_line_spacing,
    estimate_char_height,
    estimate_text_angle,
    check_line_spacing,
    check_char_height,
    check_text_angle,
    check_margin_alignment,
    check_consistency,
    batch_check_consistency,
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
    "ScoreComponent",
    "AssemblyConfidence",
    "grade_from_score",
    "score_edge_compat",
    "score_layout",
    "score_coverage",
    "score_uniqueness",
    "score_assembly_score",
    "compute_confidence",
    # Проверка согласованности
    "ConsistencyType",
    "ConsistencyViolation",
    "ConsistencyResult",
    "estimate_line_spacing",
    "estimate_char_height",
    "estimate_text_angle",
    "check_line_spacing",
    "check_char_height",
    "check_text_angle",
    "check_margin_alignment",
    "check_consistency",
    "batch_check_consistency",
]
