"""
Набор валидаторов качества итоговой сборки документа.

Позволяет подключать любые из 20 модулей verification через конфигурацию.
Каждый валидатор получает Assembly и возвращает score [0..1].

Использование:
    from puzzle_reconstruction.verification.suite import VerificationSuite

    suite = VerificationSuite(validators=["assembly_score", "layout",
                                           "completeness", "seam"])
    report = suite.run(assembly)
    print(report.summary())

Доступные валидаторы:
    assembly_score   — геометрия + покрытие + швы + уникальность
    layout           — корректность 2D-компоновки (перекрытия, зазоры)
    completeness     — полнота: все фрагменты размещены?
    seam             — качество швов между фрагментами
    overlap          — пересечения фрагментов
    text_coherence   — связность текста между фрагментами
    confidence       — уверенность в каждом размещении
    consistency      — глобальная согласованность сборки
    edge_quality     — качество совместимости краёв
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─── Отчёт о верификации ──────────────────────────────────────────────────────

@dataclass
class ValidatorResult:
    """Результат одного валидатора."""
    name:    str
    score:   float   # [0..1], выше — лучше
    details: str = ""
    error:   Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class VerificationReport:
    """Полный отчёт VerificationSuite."""
    results:      List[ValidatorResult] = field(default_factory=list)
    final_score:  float = 0.0

    def summary(self) -> str:
        lines = ["=== Верификация сборки ==="]
        for r in self.results:
            status = f"{r.score:.3f}" if r.success else f"ERROR: {r.error}"
            lines.append(f"  {r.name:<18} {status}")
            if r.details:
                lines.append(f"    {r.details}")
        lines.append(f"  {'ИТОГО':<18} {self.final_score:.3f}")
        return "\n".join(lines)


# ─── Вспомогательная функция ──────────────────────────────────────────────────

def _safe_run(name: str, fn: Callable, assembly) -> ValidatorResult:
    """Запускает валидатор с защитой от исключений."""
    try:
        score, details = fn(assembly)
        score = float(max(0.0, min(1.0, score)))
        return ValidatorResult(name=name, score=score, details=details)
    except Exception as exc:
        logger.debug("validator %r failed: %s", name, exc)
        return ValidatorResult(name=name, score=0.0, error=str(exc))


# ─── Реестр валидаторов ───────────────────────────────────────────────────────

def _build_validator_registry() -> Dict[str, Callable]:
    """Строит словарь {name: fn(assembly) -> (score, details)}."""
    registry: Dict[str, Callable] = {}

    # ── assembly_score ────────────────────────────────────────────────────────
    try:
        from .assembly_scorer import compute_assembly_score

        def _assembly_score(asm):
            placements = asm.placements or []
            n_placed = len(placements)
            n_total = len(asm.fragments) if asm.fragments else n_placed
            report = compute_assembly_score(
                n_placed=n_placed,
                n_total=n_total,
            )
            return report.total_score, report.grade if hasattr(report, "grade") else ""

        registry["assembly_score"] = _assembly_score
    except Exception:
        pass

    # ── layout ────────────────────────────────────────────────────────────────
    try:
        from .layout_checker import check_layout

        def _layout(asm):
            placements = asm.placements or []
            frag_ids = []
            positions = {}
            for p in placements:
                fid = getattr(p, "fragment_id", None)
                pos = getattr(p, "position", None)
                if fid is not None and pos is not None:
                    frag_ids.append(fid)
                    x, y = float(pos[0]), float(pos[1])
                    # Ширина/высота: берём из фрагментов если доступны
                    w, h = 100.0, 100.0
                    if asm.fragments:
                        frag = next((f for f in asm.fragments
                                     if f.fragment_id == fid), None)
                        if frag is not None and frag.image is not None:
                            h, w = float(frag.image.shape[0]), float(frag.image.shape[1])
                    positions[fid] = (x, y, w, h)

            if len(frag_ids) < 2:
                return 1.0, "недостаточно фрагментов для проверки компоновки"

            result = check_layout(frag_ids, positions)
            score = getattr(result, "score", 1.0)
            violations = len(getattr(result, "violations", []))
            return score, f"{violations} нарушений"

        registry["layout"] = _layout
    except Exception:
        pass

    # ── completeness ──────────────────────────────────────────────────────────
    try:
        from .completeness_checker import completeness_score

        def _completeness(asm):
            placements = asm.placements or []
            n_placed = len(placements)
            n_total = len(asm.fragments) if asm.fragments else n_placed
            placed_ids = {getattr(p, "fragment_id", i)
                          for i, p in enumerate(placements)}
            result = completeness_score(
                placed_fragment_ids=list(placed_ids),
                total_fragment_ids=list(range(n_total)),
            )
            score = getattr(result, "score", float(n_placed) / max(n_total, 1))
            return score, f"{n_placed}/{n_total} фрагментов"

        registry["completeness"] = _completeness
    except Exception:
        # fallback: вычислить самостоятельно
        def _completeness_fallback(asm):
            n_placed = len(asm.placements or [])
            n_total = len(asm.fragments) if asm.fragments else n_placed
            score = float(n_placed) / max(n_total, 1)
            return score, f"{n_placed}/{n_total} фрагментов"

        registry["completeness"] = _completeness_fallback

    # ── seam ──────────────────────────────────────────────────────────────────
    try:
        from .seam_analyzer import batch_analyze_seams, score_seam_quality

        def _seam(asm):
            placements = asm.placements or []
            if not placements or not asm.fragments:
                return 1.0, "нет данных для анализа швов"

            # Собираем пары соседних фрагментов из compat_matrix
            frags_by_id = {f.fragment_id: f for f in (asm.fragments or [])}
            seam_inputs = []

            # Берём топ-N пар из placements
            frag_list = []
            for p in placements[:10]:
                fid = getattr(p, "fragment_id", None)
                if fid is not None and fid in frags_by_id:
                    frag_list.append(frags_by_id[fid])

            for i in range(len(frag_list) - 1):
                img_a = frag_list[i].image
                img_b = frag_list[i + 1].image
                if img_a is not None and img_b is not None:
                    seam_inputs.append((img_a, img_b, 3))  # side=3 (bottom-top)

            if not seam_inputs:
                return 1.0, "нет изображений для анализа швов"

            analyses = batch_analyze_seams(seam_inputs)
            scores = [score_seam_quality(a) for a in analyses]
            avg = float(sum(scores) / len(scores)) if scores else 1.0
            return avg, f"{len(scores)} швов проверено"

        registry["seam"] = _seam
    except Exception:
        pass

    # ── overlap ───────────────────────────────────────────────────────────────
    try:
        from .overlap_checker import check_all_overlaps
        import numpy as np

        def _overlap(asm):
            placements = asm.placements or []
            if not placements or not asm.fragments:
                return 1.0, "нет данных для проверки перекрытий"

            frags_by_id = {f.fragment_id: f for f in (asm.fragments or [])}
            polygons = []
            for p in placements:
                fid = getattr(p, "fragment_id", None)
                pos = getattr(p, "position", None)
                if fid is None or pos is None:
                    continue
                frag = frags_by_id.get(fid)
                if frag is None or frag.contour is None:
                    continue
                x, y = float(pos[0]), float(pos[1])
                shifted = frag.contour.astype(float) + np.array([x, y])
                polygons.append(shifted.astype(np.float32))

            if len(polygons) < 2:
                return 1.0, "недостаточно контуров для проверки"

            results = check_all_overlaps(polygons)
            n_overlapping = sum(1 for r in results
                                if getattr(r, "has_overlap", False))
            score = 1.0 - float(n_overlapping) / max(len(results), 1)
            return score, f"{n_overlapping}/{len(results)} пар с перекрытием"

        registry["overlap"] = _overlap
    except Exception:
        pass

    # ── text_coherence ────────────────────────────────────────────────────────
    try:
        from .text_coherence import check_text_coherence

        def _text_coherence(asm):
            if not asm.fragments:
                return 1.0, "нет фрагментов"
            result = check_text_coherence(asm)
            score = getattr(result, "score", getattr(result, "coherence_score", 0.5))
            return float(score), ""

        registry["text_coherence"] = _text_coherence
    except Exception:
        pass

    # ── confidence ────────────────────────────────────────────────────────────
    try:
        from .confidence_scorer import score_assembly_confidence

        def _confidence(asm):
            result = score_assembly_confidence(asm)
            score = getattr(result, "mean_confidence",
                           getattr(result, "score", asm.total_score))
            return float(score), ""

        registry["confidence"] = _confidence
    except Exception:
        # fallback: total_score как прокси
        def _confidence_fallback(asm):
            return float(asm.total_score), "из total_score"

        registry["confidence"] = _confidence_fallback

    # ── consistency ───────────────────────────────────────────────────────────
    try:
        from .consistency_checker import check_consistency

        def _consistency(asm):
            result = check_consistency(asm)
            score = getattr(result, "score",
                           1.0 - getattr(result, "violation_rate", 0.0))
            return float(score), ""

        registry["consistency"] = _consistency
    except Exception:
        pass

    # ── edge_quality ──────────────────────────────────────────────────────────
    try:
        from .edge_validator import validate_edges

        def _edge_quality(asm):
            result = validate_edges(asm)
            score = getattr(result, "score",
                           getattr(result, "valid_ratio", 1.0))
            return float(score), ""

        registry["edge_quality"] = _edge_quality
    except Exception:
        pass

    return registry


# Глобальный реестр
_VALIDATOR_REGISTRY: Dict[str, Callable] = {}


def _ensure_registry() -> None:
    global _VALIDATOR_REGISTRY
    if not _VALIDATOR_REGISTRY:
        _VALIDATOR_REGISTRY = _build_validator_registry()


def list_validators() -> List[str]:
    """Список всех доступных валидаторов."""
    _ensure_registry()
    return sorted(_VALIDATOR_REGISTRY.keys())


# ─── VerificationSuite ────────────────────────────────────────────────────────

@dataclass
class VerificationSuite:
    """
    Набор валидаторов качества итоговой сборки.

    Запускает выбранные валидаторы, агрегирует оценки и формирует
    полный VerificationReport.

    Attributes:
        validators: Список имён валидаторов для запуска.
                    Пустой список → только OCR (поведение по умолчанию).

    Example:
        suite = VerificationSuite(
            validators=["assembly_score", "layout", "completeness", "seam"]
        )
        report = suite.run(assembly)
        print(report.summary())
        assembly.verification_score = report.final_score
    """
    validators: List[str] = field(default_factory=list)

    def run(self, assembly) -> VerificationReport:
        """
        Запускает все настроенные валидаторы на сборке.

        Args:
            assembly: Assembly объект после завершения сборки.

        Returns:
            VerificationReport с per-validator и итоговым score.
        """
        if not self.validators:
            return VerificationReport(results=[], final_score=assembly.total_score)

        _ensure_registry()

        results: List[ValidatorResult] = []
        for name in self.validators:
            fn = _VALIDATOR_REGISTRY.get(name)
            if fn is None:
                logger.debug("validator %r not available, skipping", name)
                results.append(ValidatorResult(
                    name=name, score=0.0,
                    error="валидатор недоступен",
                ))
                continue

            result = _safe_run(name, fn, assembly)
            results.append(result)
            logger.info("  [%s] %.3f%s",
                        name, result.score,
                        f"  {result.details}" if result.details else "")

        # Финальный score: среднее по успешным валидаторам
        successful = [r for r in results if r.success]
        if successful:
            final = sum(r.score for r in successful) / len(successful)
        else:
            final = assembly.total_score

        return VerificationReport(results=results, final_score=float(final))

    def is_empty(self) -> bool:
        """True если нет настроенных валидаторов."""
        return not self.validators
