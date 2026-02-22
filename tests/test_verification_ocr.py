"""Расширенные тесты для puzzle_reconstruction/verification/ocr.py.

OCR-функции используют pytesseract (опциональная зависимость).
Большинство функций возвращают нейтральную оценку 0.5, если OCR недоступен.
Тесты охватывают все публичные функции в обоих режимах.
"""
import pytest
import numpy as np

from puzzle_reconstruction.verification.ocr import (
    text_coherence_score,
    verify_full_assembly,
    render_assembly_image,
    _score_text_quality,
    _extract_edge_strip,
    _rotate_image,
    _TESSERACT_AVAILABLE,
)
from puzzle_reconstruction.models import Assembly, Fragment, EdgeSignature, EdgeSide


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_bgr(h=64, w=64):
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_edge():
    return EdgeSignature(
        edge_id=0,
        side=EdgeSide.RIGHT,
        virtual_curve=np.zeros((8, 2)),
        fd=0.0,
        css_vec=np.zeros(16),
        ifs_coeffs=np.zeros(8),
        length=64.0,
    )


def make_fragment(fid=0, h=64, w=64):
    return Fragment(
        fragment_id=fid,
        image=make_bgr(h=h, w=w),
        mask=np.zeros((h, w), dtype=np.uint8),
        contour=np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64),
    )


def make_assembly(n=2):
    frags = [make_fragment(fid=i) for i in range(n)]
    placements = {i: (np.array([float(i * 80), 0.0]), 0.0) for i in range(n)}
    return Assembly(
        fragments=frags,
        placements=placements,
        compat_matrix=np.eye(n, dtype=np.float32),
        total_score=0.5,
    )


# ─── TestScoreTextQuality ─────────────────────────────────────────────────────

class TestScoreTextQuality:
    def test_empty_string_returns_zero(self):
        assert _score_text_quality("") == pytest.approx(0.0)

    def test_whitespace_only_returns_zero(self):
        assert _score_text_quality("   \n\t") == pytest.approx(0.0)

    def test_clean_text_high_score(self):
        result = _score_text_quality("Hello World this is clean text")
        assert result > 0.5

    def test_garbage_text_low_score(self):
        result = _score_text_quality("@#$%^&*(){}[]<>")
        assert result < 0.5

    def test_returns_float(self):
        result = _score_text_quality("hello")
        assert isinstance(result, float)

    def test_in_0_1(self):
        for text in ["", "hello world", "###@@@", "The quick brown fox"]:
            result = _score_text_quality(text)
            assert 0.0 <= result <= 1.0

    def test_long_words_bonus(self):
        short = _score_text_quality("aaa")
        long = _score_text_quality("beautiful magnificent extraordinary")
        assert long >= short

    def test_returns_float_type(self):
        assert type(_score_text_quality("test")) is float

    def test_all_alpha_high_score(self):
        result = _score_text_quality("abcdefghij")
        assert result > 0.5

    def test_mixed_clean_dirty(self):
        result = _score_text_quality("hello ### world")
        assert 0.0 <= result <= 1.0

    def test_single_char_alpha(self):
        result = _score_text_quality("a")
        assert 0.0 <= result <= 1.0

    def test_single_char_garbage(self):
        result = _score_text_quality("@")
        assert 0.0 <= result <= 1.0

    def test_newlines_treated_as_space(self):
        result = _score_text_quality("hello\nworld\n")
        assert result > 0.0

    def test_punctuation_allowed(self):
        result = _score_text_quality("Hello, world!")
        assert result > 0.7

    def test_clip_max_one(self):
        result = _score_text_quality("the beautiful magnificent wonderfully extraordinary")
        assert result <= 1.0

    def test_clip_min_zero(self):
        result = _score_text_quality("#@!$%^&*()")
        assert result >= 0.0


# ─── TestExtractEdgeStrip ─────────────────────────────────────────────────────

class TestExtractEdgeStrip:
    def test_right_strip_returns_ndarray(self):
        img = make_bgr(h=64, w=64)
        edge = make_edge()
        result = _extract_edge_strip(img, edge, width=20, side="right")
        assert isinstance(result, np.ndarray)

    def test_left_strip_returns_ndarray(self):
        img = make_bgr(h=64, w=64)
        edge = make_edge()
        result = _extract_edge_strip(img, edge, width=20, side="left")
        assert isinstance(result, np.ndarray)

    def test_right_strip_height_preserved(self):
        img = make_bgr(h=64, w=64)
        edge = make_edge()
        result = _extract_edge_strip(img, edge, width=20, side="right")
        assert result.shape[0] == 64

    def test_left_strip_height_preserved(self):
        img = make_bgr(h=64, w=64)
        edge = make_edge()
        result = _extract_edge_strip(img, edge, width=20, side="left")
        assert result.shape[0] == 64

    def test_right_strip_width_at_most_w(self):
        img = make_bgr(h=64, w=64)
        edge = make_edge()
        result = _extract_edge_strip(img, edge, width=20, side="right")
        assert result.shape[1] <= 20

    def test_left_strip_width_at_most_w(self):
        img = make_bgr(h=64, w=64)
        edge = make_edge()
        result = _extract_edge_strip(img, edge, width=20, side="left")
        assert result.shape[1] <= 20

    def test_width_exceeds_image_clamped(self):
        img = make_bgr(h=64, w=30)
        edge = make_edge()
        result = _extract_edge_strip(img, edge, width=50, side="right")
        assert result.shape[1] <= 30

    def test_full_width_strip(self):
        img = make_bgr(h=32, w=32)
        edge = make_edge()
        result = _extract_edge_strip(img, edge, width=32, side="left")
        assert result.shape == (32, 32, 3)

    def test_dtype_preserved(self):
        img = make_bgr()
        edge = make_edge()
        result = _extract_edge_strip(img, edge, width=10, side="right")
        assert result.dtype == np.uint8


# ─── TestRotateImage ──────────────────────────────────────────────────────────

class TestRotateImage:
    def test_returns_ndarray(self):
        img = make_bgr()
        result = _rotate_image(img, 0.0)
        assert isinstance(result, np.ndarray)

    def test_zero_angle_same_shape(self):
        img = make_bgr(h=64, w=80)
        result = _rotate_image(img, 0.0)
        assert result.shape == img.shape

    def test_nonzero_angle_same_shape(self):
        img = make_bgr(h=64, w=80)
        result = _rotate_image(img, np.pi / 4)
        assert result.shape == img.shape

    def test_dtype_preserved(self):
        img = make_bgr()
        result = _rotate_image(img, 0.1)
        assert result.dtype == np.uint8

    def test_pi_angle_same_shape(self):
        img = make_bgr(32, 32)
        result = _rotate_image(img, np.pi)
        assert result.shape == img.shape

    def test_negative_angle(self):
        img = make_bgr(64, 64)
        result = _rotate_image(img, -np.pi / 4)
        assert result.shape == img.shape


# ─── TestTextCoherenceScore ───────────────────────────────────────────────────

class TestTextCoherenceScore:
    def test_returns_float(self):
        img_a = make_bgr()
        img_b = make_bgr()
        edge_a = make_edge()
        edge_b = make_edge()
        result = text_coherence_score(img_a, img_b, edge_a, edge_b)
        assert isinstance(result, float)

    def test_in_0_1(self):
        img_a = make_bgr()
        img_b = make_bgr()
        edge_a = make_edge()
        edge_b = make_edge()
        result = text_coherence_score(img_a, img_b, edge_a, edge_b)
        assert 0.0 <= result <= 1.0

    def test_without_tesseract_returns_neutral(self):
        """Without pytesseract the function returns 0.5."""
        if _TESSERACT_AVAILABLE:
            pytest.skip("Tesseract is available on this system")
        img_a = make_bgr()
        img_b = make_bgr()
        edge_a = make_edge()
        edge_b = make_edge()
        result = text_coherence_score(img_a, img_b, edge_a, edge_b)
        assert result == pytest.approx(0.5)

    def test_custom_strip_width(self):
        img_a = make_bgr(h=64, w=128)
        img_b = make_bgr(h=64, w=128)
        edge_a = make_edge()
        edge_b = make_edge()
        result = text_coherence_score(img_a, img_b, edge_a, edge_b, strip_width=60)
        assert 0.0 <= result <= 1.0

    def test_different_image_sizes(self):
        img_a = make_bgr(h=32, w=32)
        img_b = make_bgr(h=64, w=64)
        edge_a = make_edge()
        edge_b = make_edge()
        result = text_coherence_score(img_a, img_b, edge_a, edge_b)
        assert isinstance(result, float)

    def test_narrow_strip_width(self):
        img_a = make_bgr(h=64, w=64)
        img_b = make_bgr(h=64, w=64)
        edge_a = make_edge()
        edge_b = make_edge()
        result = text_coherence_score(img_a, img_b, edge_a, edge_b, strip_width=5)
        assert 0.0 <= result <= 1.0


# ─── TestVerifyFullAssembly ───────────────────────────────────────────────────

class TestVerifyFullAssembly:
    def test_returns_float(self):
        asm = make_assembly(n=2)
        result = verify_full_assembly(asm)
        assert isinstance(result, float)

    def test_in_0_1(self):
        asm = make_assembly(n=2)
        result = verify_full_assembly(asm)
        assert 0.0 <= result <= 1.0

    def test_without_tesseract_returns_neutral(self):
        if _TESSERACT_AVAILABLE:
            pytest.skip("Tesseract is available on this system")
        asm = make_assembly(n=2)
        result = verify_full_assembly(asm)
        assert result == pytest.approx(0.5)

    def test_custom_lang_accepted(self):
        asm = make_assembly(n=2)
        result = verify_full_assembly(asm, lang="eng")
        assert 0.0 <= result <= 1.0

    def test_empty_placements_returns_neutral(self):
        asm = make_assembly(n=2)
        asm.placements = {}
        result = verify_full_assembly(asm)
        assert result == pytest.approx(0.5)

    def test_single_fragment(self):
        asm = make_assembly(n=1)
        result = verify_full_assembly(asm)
        assert 0.0 <= result <= 1.0


# ─── TestRenderAssemblyImage ──────────────────────────────────────────────────

class TestRenderAssemblyImage:
    def test_returns_ndarray_or_none(self):
        asm = make_assembly(n=2)
        result = render_assembly_image(asm)
        assert result is None or isinstance(result, np.ndarray)

    def test_empty_placements_returns_none(self):
        asm = make_assembly(n=2)
        asm.placements = {}
        result = render_assembly_image(asm)
        assert result is None

    def test_color_image_3_channels(self):
        asm = make_assembly(n=2)
        result = render_assembly_image(asm)
        if result is not None:
            assert result.ndim == 3
            assert result.shape[2] == 3

    def test_dtype_uint8(self):
        asm = make_assembly(n=2)
        result = render_assembly_image(asm)
        if result is not None:
            assert result.dtype == np.uint8

    def test_larger_assembly(self):
        asm = make_assembly(n=4)
        result = render_assembly_image(asm)
        if result is not None:
            assert result.shape[2] == 3

    def test_single_fragment_assembly(self):
        asm = make_assembly(n=1)
        result = render_assembly_image(asm)
        if result is not None:
            assert isinstance(result, np.ndarray)
