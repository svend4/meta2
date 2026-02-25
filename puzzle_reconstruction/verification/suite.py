"""
Набор валидаторов качества итоговой сборки документа.

Позволяет подключать все 20 модулей verification через конфигурацию.
Каждый валидатор получает Assembly и возвращает score [0..1].

Использование:
    from puzzle_reconstruction.verification.suite import VerificationSuite

    suite = VerificationSuite(validators=["assembly_score", "layout",
                                           "completeness", "seam"])
    report = suite.run(assembly)
    print(report.summary())

Доступные валидаторы (21 модуль, 20 регистрируются в реестре):
    --- Исходные 9 ---
    assembly_score   — геометрия + покрытие + швы + уникальность
    layout           — корректность 2D-компоновки (перекрытия, зазоры)
    completeness     — полнота: все фрагменты размещены?
    seam             — качество швов между фрагментами
    overlap          — пересечения фрагментов
    text_coherence   — связность текста между фрагментами
    confidence       — уверенность в каждом размещении
    consistency      — глобальная согласованность сборки
    edge_quality     — качество совместимости краёв

    --- Новые 12 (Фаза 2 активации верификаторов) ---
    boundary         — геометрические границы между соседними фрагментами
    layout_verify    — верификация компоновки через LayoutConstraint
    overlap_validate — маска-уровневая проверка перекрытий (IoU на холсте)
    spatial          — пространственная согласованность на холсте
    placement        — коллизии bbox, дублирующиеся позиции, выход за холст
    layout_score     — составной score компоновки (coverage, uniformity, ...)
    fragment_valid   — валидность каждого фрагмента (размер, яркость, ...)
    quality_report   — комплексный отчёт качества (coverage, overlap, OCR)
    score_report     — агрегация метрик через ReciprocalRankFusion-подобный scorer
    full_report      — полный Report-объект (сборка + pipeline + метрики)
    metrics          — ReconstructionMetrics (NA, DC, RMSE); нейтрально без GT
    overlap_area     — суммарная площадь перекрытий, нормированная по холсту
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
            # completeness_score(n_placed, n_total) → float
            raw = completeness_score(n_placed=n_placed, n_total=max(n_total, 1))
            score = getattr(raw, "score", float(raw))
            return float(score), f"{n_placed}/{n_total} фрагментов"

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

    # ══════════════════════════════════════════════════════════════════════════
    # НОВЫЕ ВАЛИДАТОРЫ — Фаза 2 активации (12 модулей)
    # ══════════════════════════════════════════════════════════════════════════

    # ── boundary ──────────────────────────────────────────────────────────────
    try:
        from .boundary_validator import validate_all_pairs, boundary_quality_score

        def _boundary(asm):
            placements = asm.placements or []
            if len(placements) < 2:
                return 1.0, "недостаточно размещений для проверки границ"

            frags_by_id = {f.fragment_id: f for f in (asm.fragments or [])}
            # validate_all_pairs works on a single axis (x): positions = x-coords,
            # sizes = widths. Sequential pairs check horizontal adjacency.
            positions_list: list = []
            sizes_list: list = []
            for i, p in enumerate(placements):
                fid = getattr(p, "fragment_id", i)
                pos = getattr(p, "position", None)
                x = float(pos[0]) if pos is not None else float(i * 100)
                positions_list.append(x)
                frag = frags_by_id.get(fid)
                if frag is not None and frag.image is not None:
                    _h, w = frag.image.shape[:2]
                else:
                    w = 100.0
                sizes_list.append(float(w))

            pairs = [(i, i + 1) for i in range(len(placements) - 1)]
            report = validate_all_pairs(pairs, positions_list, sizes_list)
            violations = getattr(report, "violations", [])
            score = boundary_quality_score(violations, n_pairs=max(len(pairs), 1))
            return float(score), f"{len(violations)} нарушений границ"

        registry["boundary"] = _boundary
    except Exception:
        pass

    # ── layout_verify ─────────────────────────────────────────────────────────
    try:
        from .layout_verifier import verify_layout

        def _layout_verify(asm):
            frags = asm.fragments or []
            if not frags:
                return 1.0, "нет фрагментов для верификации компоновки"
            result = verify_layout(asm, frags)
            violation_score = float(getattr(result, "violation_score", 0.0))
            valid = getattr(result, "valid", True)
            n_constraints = len(getattr(result, "constraints", []))
            score = 1.0 - min(violation_score, 1.0)
            status = "OK" if valid else "нарушения"
            return score, f"{n_constraints} ограничений, {status}"

        registry["layout_verify"] = _layout_verify
    except Exception:
        pass

    # ── overlap_validate ──────────────────────────────────────────────────────
    try:
        import numpy as np
        from .overlap_validator import validate_assembly as _ov_validate_assembly

        def _overlap_validate(asm):
            placements = asm.placements or []
            if not placements or not asm.fragments:
                return 1.0, "нет данных для валидации перекрытий"

            frags_by_id = {f.fragment_id: f for f in (asm.fragments or [])}
            masks, positions_list = [], []
            for i, p in enumerate(placements):
                fid = getattr(p, "fragment_id", i)
                pos = getattr(p, "position", None)
                frag = frags_by_id.get(fid)
                if frag is None or frag.mask is None:
                    continue
                x = int(pos[0]) if pos is not None else 0
                y = int(pos[1]) if pos is not None else 0
                masks.append(frag.mask)
                positions_list.append((x, y))

            if len(masks) < 2:
                return 1.0, "недостаточно масок"

            max_x = max(p[0] + m.shape[1] for m, p in zip(masks, positions_list))
            max_y = max(p[1] + m.shape[0] for m, p in zip(masks, positions_list))
            canvas_size = (int(max_x) + 1, int(max_y) + 1)

            result = _ov_validate_assembly(masks, positions_list, canvas_size)
            n_overlaps = int(getattr(result, "n_overlaps", 0))
            max_iou = float(getattr(result, "max_iou", 0.0))
            score = 1.0 - min(max_iou, 1.0)
            return score, f"{n_overlaps} перекрытий, max_iou={max_iou:.3f}"

        registry["overlap_validate"] = _overlap_validate
    except Exception:
        pass

    # ── spatial ───────────────────────────────────────────────────────────────
    try:
        from .spatial_validator import PlacedFragment as SpatialPlacedFragment
        from .spatial_validator import validate_spatial

        def _spatial(asm):
            placements = asm.placements or []
            if not placements:
                return 1.0, "нет размещений"

            frags_by_id = {f.fragment_id: f for f in (asm.fragments or [])}
            placed = []
            for i, p in enumerate(placements):
                fid = getattr(p, "fragment_id", i)
                pos = getattr(p, "position", None)
                frag = frags_by_id.get(fid)
                x = max(0.0, float(pos[0])) if pos is not None else float(i * 100)
                y = max(0.0, float(pos[1])) if pos is not None else 0.0
                if frag is not None and frag.image is not None:
                    h, w = frag.image.shape[:2]
                else:
                    w, h = 100, 100
                placed.append(SpatialPlacedFragment(
                    fragment_id=max(0, int(fid)) if fid is not None else i,
                    x=x, y=y,
                    width=max(1.0, float(w)),
                    height=max(1.0, float(h)),
                ))

            canvas_w = max(pf.x + pf.width for pf in placed) + 10.0
            canvas_h = max(pf.y + pf.height for pf in placed) + 10.0
            result = validate_spatial(placed, canvas_w, canvas_h)
            is_valid = getattr(result, "is_valid", True)
            n_err = getattr(result, "n_errors", 0)
            n_warn = getattr(result, "n_warnings", 0)
            score = 1.0 if is_valid else max(0.0, 1.0 - n_err / max(len(placed), 1))
            return score, f"{n_err} ошибок, {n_warn} предупреждений"

        registry["spatial"] = _spatial
    except Exception:
        pass

    # ── placement ─────────────────────────────────────────────────────────────
    try:
        from .placement_validator import PlacementBox, validate_placements

        def _placement(asm):
            placements = asm.placements or []
            if not placements:
                return 1.0, "нет размещений"

            frags_by_id = {f.fragment_id: f for f in (asm.fragments or [])}
            boxes = []
            for i, p in enumerate(placements):
                fid = getattr(p, "fragment_id", i)
                pos = getattr(p, "position", None)
                frag = frags_by_id.get(fid)
                x = max(0, int(pos[0])) if pos is not None else i * 100
                y = max(0, int(pos[1])) if pos is not None else 0
                if frag is not None and frag.image is not None:
                    h, w = frag.image.shape[:2]
                else:
                    w, h = 100, 100
                boxes.append(PlacementBox(
                    fragment_id=max(0, int(fid)) if fid is not None else i,
                    x=x, y=y,
                    w=max(1, int(w)), h=max(1, int(h)),
                ))

            result = validate_placements(boxes)
            is_valid = getattr(result, "is_valid", True)
            coverage = float(getattr(result, "coverage", 1.0))
            n_coll = len(getattr(result, "collisions", []))
            score = coverage if is_valid else coverage * 0.5
            return min(1.0, float(score)), f"coverage={coverage:.2f}, {n_coll} коллизий"

        registry["placement"] = _placement
    except Exception:
        pass

    # ── layout_score ──────────────────────────────────────────────────────────
    try:
        from .layout_scorer import PlacedFragment as LS_PlacedFragment
        from .layout_scorer import score_layout

        def _layout_score(asm):
            placements = asm.placements or []
            if not placements:
                return 1.0, "нет размещений"

            frags_by_id = {f.fragment_id: f for f in (asm.fragments or [])}
            ls_frags = []
            for i, p in enumerate(placements):
                fid = getattr(p, "fragment_id", i)
                pos = getattr(p, "position", None)
                rot = getattr(p, "rotation", 0.0)
                frag = frags_by_id.get(fid)
                x = max(0, int(pos[0])) if pos is not None else i * 100
                y = max(0, int(pos[1])) if pos is not None else 0
                if frag is not None and frag.image is not None:
                    h, w = frag.image.shape[:2]
                else:
                    w, h = 100, 100
                ls_frags.append(LS_PlacedFragment(
                    fragment_id=max(0, int(fid)) if fid is not None else i,
                    x=x, y=y,
                    w=max(1, int(w)), h=max(1, int(h)),
                    angle=float(rot) if rot is not None else 0.0,
                    score=float(asm.total_score),
                ))

            result = score_layout(ls_frags)
            total = float(getattr(result, "total_score", 1.0))
            quality = getattr(result, "quality_level", "")
            coverage = float(getattr(result, "coverage", 1.0))
            return min(1.0, total), f"quality={quality}, coverage={coverage:.2f}"

        registry["layout_score"] = _layout_score
    except Exception:
        pass

    # ── fragment_valid ────────────────────────────────────────────────────────
    try:
        from .fragment_validator import batch_validate as fv_batch_validate

        def _fragment_valid(asm):
            frags = asm.fragments or []
            if not frags:
                return 1.0, "нет фрагментов"

            images = [f.image for f in frags if f.image is not None]
            contours = [f.contour for f in frags if f.image is not None]
            if not images:
                return 1.0, "нет изображений фрагментов"

            results = fv_batch_validate(images, contours)
            n_passed = sum(1 for r in results if getattr(r, "passed", True))
            score = float(n_passed) / max(len(results), 1)
            return score, f"{n_passed}/{len(results)} фрагментов валидны"

        registry["fragment_valid"] = _fragment_valid
    except Exception:
        pass

    # ── quality_report ────────────────────────────────────────────────────────
    try:
        from .quality_reporter import build_report as qr_build_report

        def _quality_report(asm):
            placements = asm.placements or []
            frags = asm.fragments or []
            n_placed = len(placements)
            n_total = len(frags) if frags else n_placed
            coverage = float(n_placed) / max(n_total, 1)
            ocr_score = float(getattr(asm, "ocr_score", 0.5))
            # overlap: приближение — 0.0 если нет данных
            overlap = 0.0
            report = qr_build_report(coverage, overlap, ocr_score)
            passed = getattr(report, "passed", True)
            n_errors = getattr(report, "n_errors", 0)
            summary = getattr(report, "summary", "")
            score = 1.0 if passed else max(0.0, 1.0 - n_errors * 0.1)
            return min(1.0, float(score)), str(summary)[:80]

        registry["quality_report"] = _quality_report
    except Exception:
        pass

    # ── score_report ──────────────────────────────────────────────────────────
    try:
        from .score_reporter import ScoreEntry, compute_summary

        def _score_report(asm):
            placements = asm.placements or []
            frags = asm.fragments or []
            n_placed = len(placements)
            n_total = len(frags) if frags else n_placed
            completeness = float(n_placed) / max(n_total, 1)
            entries = [
                ScoreEntry(metric="total_score",  value=float(asm.total_score), weight=2.0),
                ScoreEntry(metric="completeness",  value=completeness,          weight=1.5),
                ScoreEntry(metric="ocr",           value=float(getattr(asm, "ocr_score", 0.5)), weight=1.0),
            ]
            result = compute_summary(entries)
            total = float(getattr(result, "total_score", asm.total_score))
            status = getattr(result, "status", "")
            return min(1.0, total), str(status)

        registry["score_report"] = _score_report
    except Exception:
        pass

    # ── full_report ───────────────────────────────────────────────────────────
    try:
        from .report import build_report as report_build_report

        def _full_report(asm):
            report = report_build_report(asm)
            data = getattr(report, "data", None)
            if data is not None:
                score = float(getattr(data, "assembly_score", asm.total_score))
                n_placed = int(getattr(data, "n_placed", len(asm.placements or [])))
                n_input = int(getattr(data, "n_input", 0))
                return min(1.0, score), f"{n_placed}/{n_input} фрагментов"
            return float(asm.total_score), ""

        registry["full_report"] = _full_report
    except Exception:
        pass

    # ── metrics ───────────────────────────────────────────────────────────────
    try:
        from .metrics import evaluate_reconstruction

        def _metrics(asm):
            # evaluate_reconstruction требует ground_truth, которого нет на inference.
            # Без GT возвращаем нейтральный score с пояснением.
            placements = asm.placements or []
            if not placements:
                return 0.5, "нет размещений"
            # Прокси-метрика: используем total_score как приближение NA
            na_proxy = float(asm.total_score)
            return min(1.0, na_proxy), "proxy=total_score (ground_truth отсутствует)"

        registry["metrics"] = _metrics
    except Exception:
        # Абсолютный fallback
        def _metrics_fallback(asm):
            return 0.5, "модуль metrics недоступен"
        registry["metrics"] = _metrics_fallback

    # ── overlap_area ──────────────────────────────────────────────────────────
    try:
        import numpy as np
        from .overlap_validator import overlap_area_matrix

        def _overlap_area(asm):
            placements = asm.placements or []
            if not placements or not asm.fragments:
                return 1.0, "нет данных"

            frags_by_id = {f.fragment_id: f for f in (asm.fragments or [])}
            masks, positions_list = [], []
            for i, p in enumerate(placements):
                fid = getattr(p, "fragment_id", i)
                pos = getattr(p, "position", None)
                frag = frags_by_id.get(fid)
                if frag is None or frag.mask is None:
                    continue
                x = int(pos[0]) if pos is not None else 0
                y = int(pos[1]) if pos is not None else 0
                masks.append(frag.mask)
                positions_list.append((x, y))

            if len(masks) < 2:
                return 1.0, "недостаточно масок"

            max_x = max(p[0] + m.shape[1] for m, p in zip(masks, positions_list))
            max_y = max(p[1] + m.shape[0] for m, p in zip(masks, positions_list))
            canvas_size = (int(max_x) + 1, int(max_y) + 1)

            mat = overlap_area_matrix(masks, positions_list, canvas_size)
            total_area = float(np.sum(mat))
            canvas_area = float(canvas_size[0] * canvas_size[1])
            overlap_ratio = total_area / max(canvas_area, 1.0)
            score = 1.0 - min(overlap_ratio, 1.0)
            return score, f"overlap_ratio={overlap_ratio:.4f}"

        registry["overlap_area"] = _overlap_area
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
    """Список всех доступных валидаторов (в алфавитном порядке)."""
    _ensure_registry()
    return sorted(_VALIDATOR_REGISTRY.keys())


def all_validator_names() -> List[str]:
    """Список всех 21 имён валидаторов (включая недоступные).

    Используется для отчётности. Недоступные валидаторы возвращают score=0.0
    с сообщением об ошибке.
    """
    return [
        # Базовые 9
        "assembly_score", "layout", "completeness", "seam", "overlap",
        "text_coherence", "confidence", "consistency", "edge_quality",
        # Расширенные 12
        "boundary", "layout_verify", "overlap_validate", "spatial",
        "placement", "layout_score", "fragment_valid", "quality_report",
        "score_report", "full_report", "metrics", "overlap_area",
    ]


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

    def run_all(self, assembly) -> "VerificationReport":
        """Запускает ВСЕ 21 зарегистрированных валидатора.

        Удобный метод для полного прохода верификации.

        Returns:
            VerificationReport с результатами всех валидаторов.
        """
        suite_all = VerificationSuite(validators=all_validator_names())
        return suite_all.run(assembly)

    def is_empty(self) -> bool:
        """True если нет настроенных валидаторов."""
        return not self.validators
