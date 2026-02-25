"""
Конфигурируемая цепочка предобработки фрагментов документа.

Позволяет подключать любые из 38 модулей preprocessing через конфигурацию.
Каждый фильтр получает np.ndarray и возвращает np.ndarray (или None для
фильтрации по качеству).

Использование:
    from puzzle_reconstruction.preprocessing.chain import PreprocessingChain

    chain = PreprocessingChain(
        filters=["quality_assessor", "denoise", "contrast", "deskew"],
        quality_threshold=0.4,
    )
    result = chain.apply(image)
    if result is not None:
        processed_fragment = result

Доступные фильтры:
    quality_assessor    — оценка качества + фильтрация по порогу
    denoise             — автошумоподавление
    contrast            — автоулучшение контраста
    deskew              — коррекция наклона
    background_remove   — удаление фона
    binarize            — автобинаризация
    edge_enhance        — усиление краёв
    sharpen             — усиление резкости краёв
    color_normalize     — нормализация цвета
    illumination        — коррекция освещения
    morph               — морфологические операции
    crop                — автообрезка к содержательной области
    noise_analyze       — анализ шума (диагностика, без изменений)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─── Вспомогательная функция: извлечь изображение из результата ───────────────

def _extract_image(result) -> Optional[np.ndarray]:
    """Извлекает np.ndarray из результата функции предобработки.

    Поддерживает как прямой возврат ndarray, так и объекты результата
    с атрибутами .image или .result.
    """
    if result is None:
        return None
    if isinstance(result, np.ndarray):
        return result
    for attr in ("image", "result", "img", "output"):
        img = getattr(result, attr, None)
        if isinstance(img, np.ndarray):
            return img
    return None


# ─── Реестр фильтров ──────────────────────────────────────────────────────────

def _build_filter_registry() -> Dict[str, Callable[[np.ndarray], Optional[np.ndarray]]]:
    """Строит словарь {filter_name: callable(image) -> image | None}."""
    registry: Dict[str, Callable[[np.ndarray], Optional[np.ndarray]]] = {}

    # ── quality_assessor: возвращает None если качество ниже порога ──────────
    # (порог передаётся через замыкание при создании PreprocessingChain)
    # Регистрируется отдельно в PreprocessingChain.__post_init__

    # ── denoise ───────────────────────────────────────────────────────────────
    try:
        from .denoise import auto_denoise

        def _denoise(img: np.ndarray) -> np.ndarray:
            return auto_denoise(img)

        registry["denoise"] = _denoise
    except Exception:
        pass

    # ── contrast ──────────────────────────────────────────────────────────────
    try:
        from .contrast import auto_enhance

        def _contrast(img: np.ndarray) -> Optional[np.ndarray]:
            return _extract_image(auto_enhance(img))

        registry["contrast"] = _contrast
    except Exception:
        pass

    # ── deskew ────────────────────────────────────────────────────────────────
    try:
        from .deskewer import auto_deskew

        def _deskew(img: np.ndarray) -> Optional[np.ndarray]:
            return _extract_image(auto_deskew(img))

        registry["deskew"] = _deskew
    except Exception:
        pass

    # ── background_remove ─────────────────────────────────────────────────────
    try:
        from .background_remover import auto_remove_background

        def _bg_remove(img: np.ndarray) -> Optional[np.ndarray]:
            return _extract_image(auto_remove_background(img))

        registry["background_remove"] = _bg_remove
    except Exception:
        pass

    # ── binarize ──────────────────────────────────────────────────────────────
    try:
        from .binarizer import auto_binarize

        def _binarize(img: np.ndarray) -> Optional[np.ndarray]:
            return _extract_image(auto_binarize(img))

        registry["binarize"] = _binarize
    except Exception:
        pass

    # ── edge_enhance ──────────────────────────────────────────────────────────
    try:
        from .edge_enhancer import enhance_edges

        def _edge_enhance(img: np.ndarray) -> Optional[np.ndarray]:
            return _extract_image(enhance_edges(img))

        registry["edge_enhance"] = _edge_enhance
    except Exception:
        pass

    # ── sharpen ───────────────────────────────────────────────────────────────
    try:
        from .edge_sharpener import sharpen_edges

        def _sharpen(img: np.ndarray) -> Optional[np.ndarray]:
            return _extract_image(sharpen_edges(img))

        registry["sharpen"] = _sharpen
    except Exception:
        pass

    # ── color_normalize ───────────────────────────────────────────────────────
    try:
        from .color_normalizer import normalize_color

        def _color_norm(img: np.ndarray) -> Optional[np.ndarray]:
            return _extract_image(normalize_color(img))

        registry["color_normalize"] = _color_norm
    except Exception:
        try:
            from .color_norm import normalize

            def _color_norm2(img: np.ndarray) -> Optional[np.ndarray]:
                return _extract_image(normalize(img))

            registry["color_normalize"] = _color_norm2
        except Exception:
            pass

    # ── illumination ──────────────────────────────────────────────────────────
    try:
        from .illumination_corrector import correct_by_retinex

        def _illumination(img: np.ndarray) -> Optional[np.ndarray]:
            return _extract_image(correct_by_retinex(img))

        registry["illumination"] = _illumination
    except Exception:
        pass

    # ── morph ─────────────────────────────────────────────────────────────────
    try:
        from .morphology_ops import auto_morph

        def _morph(img: np.ndarray) -> Optional[np.ndarray]:
            return _extract_image(auto_morph(img))

        registry["morph"] = _morph
    except Exception:
        pass

    # ── crop ──────────────────────────────────────────────────────────────────
    try:
        from .fragment_cropper import auto_crop

        def _crop(img: np.ndarray) -> Optional[np.ndarray]:
            return _extract_image(auto_crop(img))

        registry["crop"] = _crop
    except Exception:
        pass

    # ── noise_analyze (диагностика без изменений) ─────────────────────────────
    try:
        from .noise_analyzer import analyze

        def _noise_analyze(img: np.ndarray) -> np.ndarray:
            analyze(img)  # только логирует результат
            return img

        registry["noise_analyze"] = _noise_analyze
    except Exception:
        pass

    return registry


# Глобальный реестр фильтров
FILTER_REGISTRY: Dict[str, Callable[[np.ndarray], Optional[np.ndarray]]] = {}


def _ensure_registry() -> None:
    global FILTER_REGISTRY
    if not FILTER_REGISTRY:
        FILTER_REGISTRY = _build_filter_registry()


def list_filters() -> List[str]:
    """Список всех доступных фильтров предобработки."""
    _ensure_registry()
    return sorted(["quality_assessor"] + list(FILTER_REGISTRY.keys()))


# ─── PreprocessingChain ───────────────────────────────────────────────────────

@dataclass
class PreprocessingChain:
    """
    Конфигурируемая цепочка предобработки изображений фрагментов.

    Применяет список фильтров последовательно. Если quality_assessor
    включён в цепочку и quality_threshold > 0, фрагменты с оценкой
    ниже порога отсеиваются (apply() возвращает None).

    Attributes:
        filters:           Список имён фильтров (в порядке применения).
        quality_threshold: Минимальный acceptable score (0.0 = без фильтрации).
        auto_enhance:      True → автоматически добавить denoise+contrast если
                           цепочка пустая.

    Example:
        chain = PreprocessingChain(
            filters=["quality_assessor", "denoise", "contrast", "deskew"],
            quality_threshold=0.4,
        )
        result = chain.apply(fragment_image)
        if result is None:
            print("фрагмент отклонён по качеству")
    """
    filters: List[str] = field(default_factory=list)
    quality_threshold: float = 0.0
    auto_enhance: bool = False

    def apply(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Применяет цепочку фильтров к изображению.

        Args:
            image: BGR изображение uint8 фрагмента.

        Returns:
            Обработанное изображение или None если фрагмент отклонён
            по качеству (quality_assessor + quality_threshold).
        """
        if image is None or image.size == 0:
            return None

        _ensure_registry()

        active = list(self.filters)
        if self.auto_enhance and not active:
            active = ["denoise", "contrast"]

        result = image
        for name in active:
            try:
                result = self._apply_one(name, result)
            except Exception as exc:
                logger.warning("preprocessing filter %r failed: %s", name, exc)
                # продолжаем с предыдущим результатом
            if result is None:
                return None  # quality_assessor отклонил фрагмент

        return result

    def _apply_one(self, name: str, image: np.ndarray) -> Optional[np.ndarray]:
        """Применяет один именованный фильтр."""
        if name == "quality_assessor":
            return self._quality_gate(image)

        fn = FILTER_REGISTRY.get(name)
        if fn is None:
            logger.debug("preprocessing filter %r not available, skipping", name)
            return image  # пропускаем недоступный фильтр, не останавливаем цепочку

        out = fn(image)
        if out is None:
            logger.debug("preprocessing filter %r returned None, keeping original", name)
            return image  # фильтр не смог обработать — оставляем оригинал
        return out

    def _quality_gate(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Оценивает качество и возвращает None если ниже порога."""
        if self.quality_threshold <= 0.0:
            return image  # фильтрация по качеству отключена

        try:
            from .quality_assessor import assess_quality
            report = assess_quality(image, min_score=self.quality_threshold)
            if not report.is_acceptable:
                logger.debug(
                    "fragment rejected: quality=%.3f < threshold=%.3f",
                    report.overall_score, self.quality_threshold,
                )
                return None
        except Exception as exc:
            logger.debug("quality_assessor failed: %s", exc)

        return image

    def is_empty(self) -> bool:
        """True если цепочка не содержит ни одного фильтра."""
        return not self.filters and not self.auto_enhance
