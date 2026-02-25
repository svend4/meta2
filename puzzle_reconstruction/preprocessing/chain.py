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
    illumination        — коррекция освещения (Retinex)
    morph               — морфологические операции
    crop                — автообрезка к содержательной области
    noise_analyze       — анализ шума (диагностика, без изменений)
    adaptive_threshold  — адаптивная бинаризация (Гауссова)
    scan_augment        — симуляция артефактов сканирования
    channel_equalize    — эквализация гистограммы по каналам
    contour_analyze     — анализ контура (диагностика, без изменений)
    contrast_enhance    — расширенное улучшение контраста (CLAHE/stretch/gamma)
    document_clean      — очистка документа (тени, засветки, блобы)
    edge_detect         — детектирование краёв (Canny/Sobel/Laplacian)
    freq_analyze        — анализ частотного спектра (диагностика)
    freq_low_pass       — низкочастотный фильтр (Гауссова размытие в FFT)
    freq_high_pass      — высокочастотный фильтр (усиление деталей в FFT)
    freq_band_pass      — полосовой частотный фильтр
    gradient_analyze    — анализ градиентного поля (диагностика)
    illumination_norm   — нормализация освещения (mean/std / CLAHE)
    image_enhance       — комплексное улучшение (резкость + шум + контраст)
    noise_filter        — медианная фильтрация шума
    noise_reduce        — автовыбор шумоподавителя по уровню шума
    smart_denoise       — умное шумоподавление (лёгкий/тяжёлый режим)
    patch_normalize     — нормализация патча (eqhist/stretch/standardize)
    patch_sample        — семплирование патчей (диагностика, без изменений)
    perspective         — коррекция перспективных искажений
    skew_correct        — коррекция наклона (Hough/projection/FFT)
    texture_analyze     — анализ текстурных признаков (диагностика)
    warp_correct        — коррекция аффинных искажений
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

    # ── adaptive_threshold ────────────────────────────────────────────────────
    try:
        from .adaptive_threshold import adaptive_gaussian

        def _adaptive_threshold(img: np.ndarray) -> np.ndarray:
            return adaptive_gaussian(img)

        registry["adaptive_threshold"] = _adaptive_threshold
    except Exception:
        pass

    # ── scan_augment (симуляция артефактов сканирования) ─────────────────────
    try:
        from .augment import simulate_scan_noise

        def _scan_augment(img: np.ndarray) -> np.ndarray:
            return simulate_scan_noise(img)

        registry["scan_augment"] = _scan_augment
    except Exception:
        pass

    # ── channel_equalize ──────────────────────────────────────────────────────
    try:
        from .channel_splitter import apply_per_channel, equalize_channel

        def _channel_equalize(img: np.ndarray) -> np.ndarray:
            return apply_per_channel(img, equalize_channel)

        registry["channel_equalize"] = _channel_equalize
    except Exception:
        pass

    # ── contour_analyze (диагностика: сегментация → контур → статистики) ─────
    try:
        from .contour_processor import process_contour

        def _contour_analyze(img: np.ndarray) -> np.ndarray:
            try:
                from .segmentation import segment_fragment
                from .contour import extract_contour
                mask = segment_fragment(img)
                pts = extract_contour(mask)
                if len(pts) >= 4:
                    process_contour(pts)
            except Exception:
                pass
            return img

        registry["contour_analyze"] = _contour_analyze
    except Exception:
        pass

    # ── contrast_enhance ──────────────────────────────────────────────────────
    try:
        from .contrast_enhancer import enhance_contrast as _ce_fn

        def _contrast_enhance(img: np.ndarray) -> Optional[np.ndarray]:
            return _extract_image(_ce_fn(img))

        registry["contrast_enhance"] = _contrast_enhance
    except Exception:
        pass

    # ── document_clean ────────────────────────────────────────────────────────
    try:
        from .document_cleaner import auto_clean

        def _document_clean(img: np.ndarray) -> Optional[np.ndarray]:
            return _extract_image(auto_clean(img))

        registry["document_clean"] = _document_clean
    except Exception:
        pass

    # ── edge_detect ───────────────────────────────────────────────────────────
    try:
        from .edge_detector import detect_edges

        def _edge_detect(img: np.ndarray) -> np.ndarray:
            res = detect_edges(img)
            for attr in ("edges", "edge_map", "image", "result", "img", "output"):
                val = getattr(res, attr, None)
                if isinstance(val, np.ndarray):
                    return val
            return img

        registry["edge_detect"] = _edge_detect
    except Exception:
        pass

    # ── freq_analyze (диагностика: вычисляет спектр, не меняет изображение) ──
    try:
        from .frequency_analyzer import extract_freq_descriptor

        def _freq_analyze(img: np.ndarray) -> np.ndarray:
            extract_freq_descriptor(img)
            return img

        registry["freq_analyze"] = _freq_analyze
    except Exception:
        pass

    # ── freq_low_pass ─────────────────────────────────────────────────────────
    try:
        from .frequency_filter import gaussian_low_pass

        def _freq_low_pass(img: np.ndarray) -> np.ndarray:
            return gaussian_low_pass(img)

        registry["freq_low_pass"] = _freq_low_pass
    except Exception:
        pass

    # ── freq_high_pass ────────────────────────────────────────────────────────
    try:
        from .frequency_filter import gaussian_high_pass

        def _freq_high_pass(img: np.ndarray) -> np.ndarray:
            return gaussian_high_pass(img)

        registry["freq_high_pass"] = _freq_high_pass
    except Exception:
        pass

    # ── freq_band_pass ────────────────────────────────────────────────────────
    try:
        from .frequency_filter import band_pass_filter

        def _freq_band_pass(img: np.ndarray) -> np.ndarray:
            return band_pass_filter(img)

        registry["freq_band_pass"] = _freq_band_pass
    except Exception:
        pass

    # ── gradient_analyze (диагностика: вычисляет градиентный профиль) ────────
    try:
        from .gradient_analyzer import extract_gradient_profile

        def _gradient_analyze(img: np.ndarray) -> np.ndarray:
            extract_gradient_profile(img)
            return img

        registry["gradient_analyze"] = _gradient_analyze
    except Exception:
        pass

    # ── illumination_norm ─────────────────────────────────────────────────────
    try:
        from .illumination_normalizer import normalize_illumination as _illum_fn

        def _illumination_norm(img: np.ndarray) -> Optional[np.ndarray]:
            return _extract_image(_illum_fn(img))

        registry["illumination_norm"] = _illumination_norm
    except Exception:
        pass

    # ── image_enhance ─────────────────────────────────────────────────────────
    try:
        from .image_enhancer import enhance_image

        def _image_enhance(img: np.ndarray) -> Optional[np.ndarray]:
            return _extract_image(enhance_image(img))

        registry["image_enhance"] = _image_enhance
    except Exception:
        pass

    # ── noise_filter ──────────────────────────────────────────────────────────
    try:
        from .noise_filter import median_filter as _nf_median

        def _noise_filter(img: np.ndarray) -> np.ndarray:
            return _nf_median(img)

        registry["noise_filter"] = _noise_filter
    except Exception:
        pass

    # ── noise_reduce ──────────────────────────────────────────────────────────
    try:
        from .noise_reducer import auto_reduce

        def _noise_reduce(img: np.ndarray) -> Optional[np.ndarray]:
            return _extract_image(auto_reduce(img))

        registry["noise_reduce"] = _noise_reduce
    except Exception:
        pass

    # ── smart_denoise ─────────────────────────────────────────────────────────
    try:
        from .noise_reduction import smart_denoise

        def _smart_denoise(img: np.ndarray) -> Optional[np.ndarray]:
            return _extract_image(smart_denoise(img))

        registry["smart_denoise"] = _smart_denoise
    except Exception:
        pass

    # ── patch_normalize ───────────────────────────────────────────────────────
    try:
        from .patch_normalizer import normalize_patch

        def _patch_normalize(img: np.ndarray) -> np.ndarray:
            return normalize_patch(img)

        registry["patch_normalize"] = _patch_normalize
    except Exception:
        pass

    # ── patch_sample (диагностика: семплирует патчи, не меняет изображение) ──
    try:
        from .patch_sampler import sample_patches

        def _patch_sample(img: np.ndarray) -> np.ndarray:
            sample_patches(img)
            return img

        registry["patch_sample"] = _patch_sample
    except Exception:
        pass

    # ── perspective ───────────────────────────────────────────────────────────
    try:
        from .perspective import auto_correct_perspective

        def _perspective(img: np.ndarray) -> Optional[np.ndarray]:
            return _extract_image(auto_correct_perspective(img))

        registry["perspective"] = _perspective
    except Exception:
        pass

    # ── skew_correct ──────────────────────────────────────────────────────────
    try:
        from .skew_correction import auto_correct_skew

        def _skew_correct(img: np.ndarray) -> Optional[np.ndarray]:
            return _extract_image(auto_correct_skew(img))

        registry["skew_correct"] = _skew_correct
    except Exception:
        pass

    # ── texture_analyze (диагностика: вычисляет текстурные признаки) ─────────
    try:
        from .texture_analyzer import extract_texture_features

        def _texture_analyze(img: np.ndarray) -> np.ndarray:
            extract_texture_features(img)
            return img

        registry["texture_analyze"] = _texture_analyze
    except Exception:
        pass

    # ── warp_correct ──────────────────────────────────────────────────────────
    try:
        from .warp_corrector import correct_warp

        def _warp_correct(img: np.ndarray) -> Optional[np.ndarray]:
            return _extract_image(correct_warp(img))

        registry["warp_correct"] = _warp_correct
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
