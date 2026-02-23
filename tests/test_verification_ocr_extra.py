"""Extra tests for puzzle_reconstruction.verification.ocr."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.verification.ocr import (
    _TESSERACT_AVAILABLE,
    _extract_edge_strip,
    _rotate_image,
    _score_text_quality,
    render_assembly_image,
    text_coherence_score,
    verify_full_assembly,
)
from puzzle_reconstruction.models import Assembly, Fragment, EdgeSignature, EdgeSide


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _bgr(h=64, w=64, val=0):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _rand_bgr(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _edge(side=EdgeSide.RIGHT):
    return EdgeSignature(
        edge_id=0,
        side=side,
        virtual_curve=np.zeros((8, 2)),
        fd=0.0,
        css_vec=np.zeros(16),
        ifs_coeffs=np.zeros(8),
        length=64.0,
    )


def _fragment(fid=0, h=64, w=64):
    return Fragment(
        fragment_id=fid,
        image=_bgr(h, w),
        mask=np.zeros((h, w), dtype=np.uint8),
        contour=np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64),
    )


def _assembly(n=2):
    frags = [_fragment(fid=i) for i in range(n)]
    placements = {i: (np.array([float(i * 80), 0.0]), 0.0) for i in range(n)}
    return Assembly(
        fragments=frags,
        placements=placements,
        compat_matrix=np.eye(n, dtype=np.float32),
        total_score=0.5,
    )


# ─── TestScoreTextQualityExtra ───────────────────────────────────────────────

class TestScoreTextQualityExtra:
    def test_long_clean_text_ge_short(self):
        short = _score_text_quality("hello")
        long = _score_text_quality("hello world this is clean text content")
        assert long >= short

    def test_numeric_text_in_range(self):
        result = _score_text_quality("1234567890")
        assert 0.0 <= result <= 1.0

    def test_mixed_alpha_numeric(self):
        result = _score_text_quality("Hello 123 World")
        assert 0.0 <= result <= 1.0

    def test_only_spaces_zero(self):
        assert _score_text_quality("     ") == pytest.approx(0.0)

    def test_tabs_and_newlines_zero(self):
        assert _score_text_quality("\t\t\n\n") == pytest.approx(0.0)

    def test_returns_float_type(self):
        assert type(_score_text_quality("abc")) is float

    def test_all_garbage_low(self):
        assert _score_text_quality("!@#$%^&*") < 0.5

    def test_good_sentence_above_half(self):
        assert _score_text_quality("The quick brown fox jumps over the lazy dog") > 0.5

    def test_min_zero(self):
        assert _score_text_quality("") >= 0.0

    def test_max_one(self):
        assert _score_text_quality("beautiful extraordinary magnificent wonderful") <= 1.0


# ─── TestExtractEdgeStripExtra ───────────────────────────────────────────────

class TestExtractEdgeStripExtra:
    def test_various_widths(self):
        img = _bgr(64, 64)
        e = _edge()
        for w in [5, 10, 20, 30]:
            result = _extract_edge_strip(img, e, width=w, side="right")
            assert result.shape[1] <= w

    def test_non_square_image_right(self):
        img = _bgr(32, 64)
        e = _edge()
        result = _extract_edge_strip(img, e, width=20, side="right")
        assert result.shape[0] == 32

    def test_non_square_image_left(self):
        img = _bgr(32, 64)
        e = _edge()
        result = _extract_edge_strip(img, e, width=20, side="left")
        assert result.shape[0] == 32

    def test_width_one(self):
        img = _bgr(64, 64)
        e = _edge()
        result = _extract_edge_strip(img, e, width=1, side="right")
        assert result.shape[1] <= 1

    def test_color_preserved(self):
        img = _bgr(64, 64, val=128)
        e = _edge()
        result = _extract_edge_strip(img, e, width=20, side="left")
        assert result.dtype == np.uint8

    def test_left_side_edge_type(self):
        e = _edge(side=EdgeSide.LEFT)
        result = _extract_edge_strip(_bgr(), e, width=15, side="left")
        assert isinstance(result, np.ndarray)


# ─── TestRotateImageExtra ────────────────────────────────────────────────────

class TestRotateImageExtra:
    def test_90_degrees(self):
        img = _bgr(64, 64)
        result = _rotate_image(img, np.pi / 2)
        assert result.shape == img.shape

    def test_180_degrees(self):
        img = _bgr(32, 48)
        result = _rotate_image(img, np.pi)
        assert result.shape == img.shape

    def test_270_degrees(self):
        img = _bgr(64, 64)
        result = _rotate_image(img, 3 * np.pi / 2)
        assert result.shape == img.shape

    def test_small_angle(self):
        img = _bgr(64, 80)
        result = _rotate_image(img, 0.01)
        assert result.shape == img.shape

    def test_dtype_uint8(self):
        result = _rotate_image(_bgr(), 1.0)
        assert result.dtype == np.uint8

    def test_non_square(self):
        img = _bgr(32, 64)
        result = _rotate_image(img, np.pi / 6)
        assert result.shape == img.shape


# ─── TestTextCoherenceScoreExtra ─────────────────────────────────────────────

class TestTextCoherenceScoreExtra:
    def test_returns_float(self):
        r = text_coherence_score(_bgr(), _bgr(), _edge(), _edge())
        assert isinstance(r, float)

    def test_in_range(self):
        r = text_coherence_score(_bgr(), _bgr(), _edge(), _edge())
        assert 0.0 <= r <= 1.0

    def test_no_tesseract_neutral(self):
        if _TESSERACT_AVAILABLE:
            pytest.skip("Tesseract available")
        r = text_coherence_score(_bgr(), _bgr(), _edge(), _edge())
        assert r == pytest.approx(0.5)

    def test_wide_strip(self):
        r = text_coherence_score(_bgr(64, 128), _bgr(64, 128),
                                 _edge(), _edge(), strip_width=50)
        assert 0.0 <= r <= 1.0

    def test_narrow_strip(self):
        r = text_coherence_score(_bgr(), _bgr(), _edge(), _edge(), strip_width=3)
        assert 0.0 <= r <= 1.0

    def test_small_images(self):
        r = text_coherence_score(_bgr(8, 8), _bgr(8, 8), _edge(), _edge())
        assert 0.0 <= r <= 1.0

    def test_random_images(self):
        r = text_coherence_score(_rand_bgr(64, 64, seed=0),
                                 _rand_bgr(64, 64, seed=1),
                                 _edge(), _edge())
        assert 0.0 <= r <= 1.0


# ─── TestVerifyFullAssemblyExtra ─────────────────────────────────────────────

class TestVerifyFullAssemblyExtra:
    def test_returns_float(self):
        assert isinstance(verify_full_assembly(_assembly()), float)

    def test_in_range(self):
        assert 0.0 <= verify_full_assembly(_assembly(3)) <= 1.0

    def test_no_tesseract_neutral(self):
        if _TESSERACT_AVAILABLE:
            pytest.skip("Tesseract available")
        assert verify_full_assembly(_assembly()) == pytest.approx(0.5)

    def test_empty_placements_neutral(self):
        asm = _assembly(2)
        asm.placements = {}
        assert verify_full_assembly(asm) == pytest.approx(0.5)

    def test_single_fragment(self):
        assert 0.0 <= verify_full_assembly(_assembly(1)) <= 1.0

    def test_four_fragments(self):
        assert 0.0 <= verify_full_assembly(_assembly(4)) <= 1.0

    def test_lang_parameter(self):
        r = verify_full_assembly(_assembly(), lang="eng")
        assert 0.0 <= r <= 1.0


# ─── TestRenderAssemblyImageExtra ────────────────────────────────────────────

class TestRenderAssemblyImageExtra:
    def test_result_type(self):
        result = render_assembly_image(_assembly())
        assert result is None or isinstance(result, np.ndarray)

    def test_empty_placements_none(self):
        asm = _assembly()
        asm.placements = {}
        assert render_assembly_image(asm) is None

    def test_color_3_channels(self):
        result = render_assembly_image(_assembly())
        if result is not None:
            assert result.shape[2] == 3

    def test_dtype_uint8(self):
        result = render_assembly_image(_assembly())
        if result is not None:
            assert result.dtype == np.uint8

    def test_five_fragments(self):
        result = render_assembly_image(_assembly(5))
        if result is not None:
            assert result.ndim == 3

    def test_larger_fragments(self):
        frags = [_fragment(fid=i, h=128, w=128) for i in range(2)]
        placements = {i: (np.array([float(i * 150), 0.0]), 0.0) for i in range(2)}
        asm = Assembly(fragments=frags, placements=placements,
                       compat_matrix=np.eye(2, dtype=np.float32), total_score=0.5)
        result = render_assembly_image(asm)
        if result is not None:
            assert result.dtype == np.uint8
