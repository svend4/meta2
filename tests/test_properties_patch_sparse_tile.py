"""
Property-based invariant tests for:
  - puzzle_reconstruction.utils.patch_utils
  - puzzle_reconstruction.utils.sparse_utils
  - puzzle_reconstruction.utils.tile_utils

Verifies mathematical invariants:

patch_utils:
    extract_patch:      shape = (patch_h, patch_w) or (patch_h, patch_w, C);
                        pixels outside image filled with pad_value
    normalize_patch:    minmax → values in [0, 1]; constant → zeros
    patch_ssd:          ≥ 0; ssd(a, a) = 0; symmetric
    patch_ncc:          ∈ [-1, 1]; ncc(a, a) = 1 for non-constant; symmetric
    patch_mse:          ≥ 0; mse(a, a) = 0; symmetric
    batch_compare:      output length = len(pairs)

sparse_utils:
    to_sparse_entries/from_sparse_entries:  roundtrip identity
    to_sparse_entries:  every |value| > threshold
    symmetrize_matrix:  result is symmetric; result[i,j] = max(M[i,j], M[j,i])
    diagonal_zeros:     diagonal all zeros; off-diagonal preserved
    normalize_matrix:   all values in [0, 1]; row max ≤ 1
    matrix_sparsity:    ∈ [0, 1]; zeros-only → 1.0
    threshold_matrix:   all values ≥ threshold or == fill
    top_k_per_row:      non-zero count per row ≤ k

tile_utils:
    tile_image:         all tiles shape = (tile_h, tile_w); list non-empty
    tile_overlap_ratio: ∈ [0, 1]; commutative; self-overlap = 1.0
    compute_tile_grid:  at least 1 entry; (y, x) within image bounds
    filter_tiles_by_content: result is subset; min_fg=0 → all tiles returned
    batch_tile_images:  output length = len(images)
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest

from puzzle_reconstruction.utils.patch_utils import (
    PatchConfig,
    extract_patch,
    extract_patches,
    normalize_patch,
    patch_ssd,
    patch_ncc,
    patch_mse,
    compare_patches,
    batch_compare,
)
from puzzle_reconstruction.utils.sparse_utils import (
    SparseEntry,
    to_sparse_entries,
    from_sparse_entries,
    sparse_top_k,
    threshold_matrix,
    symmetrize_matrix,
    normalize_matrix,
    diagonal_zeros,
    matrix_sparsity,
    top_k_per_row,
)
from puzzle_reconstruction.utils.tile_utils import (
    TileConfig,
    Tile,
    tile_image,
    reassemble_tiles,
    tile_overlap_ratio,
    filter_tiles_by_content,
    compute_tile_grid,
    batch_tile_images,
)

RNG = np.random.default_rng(2028)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rand_uint8(h: int = 32, w: int = 32, channels: int = 1) -> np.ndarray:
    if channels == 1:
        return RNG.integers(0, 256, (h, w), dtype=np.uint8)
    return RNG.integers(0, 256, (h, w, channels), dtype=np.uint8)


def _rand_float(h: int = 32, w: int = 32) -> np.ndarray:
    return RNG.random((h, w)).astype(np.float32)


def _rand_square_matrix(n: int, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    mat = RNG.uniform(low, high, (n, n))
    return mat.astype(np.float64)


def _rand_contour_like_points(n: int = 8) -> np.ndarray:
    """Random (n, 2) float64 array."""
    return RNG.uniform(0, 100, (n, 2))


# ═══════════════════════════════════════════════════════════════════════════════
# patch_utils — extract_patch
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractPatch:
    """extract_patch: output shape and content invariants."""

    @pytest.mark.parametrize("ph,pw", [(8, 8), (16, 32), (32, 16)])
    def test_shape_grayscale(self, ph: int, pw: int) -> None:
        img = _rand_uint8(64, 64, 1)
        cfg = PatchConfig(patch_h=ph, patch_w=pw)
        patch = extract_patch(img, 32, 32, cfg)
        assert patch.shape == (ph, pw)

    @pytest.mark.parametrize("ph,pw,c", [(8, 8, 3), (16, 16, 3), (12, 20, 3)])
    def test_shape_color(self, ph: int, pw: int, c: int) -> None:
        img = _rand_uint8(64, 64, c)
        cfg = PatchConfig(patch_h=ph, patch_w=pw)
        patch = extract_patch(img, 32, 32, cfg)
        assert patch.shape == (ph, pw, c)

    @pytest.mark.parametrize("cy,cx", [(0, 0), (63, 63), (0, 63)])
    def test_shape_at_corners(self, cy: int, cx: int) -> None:
        img = _rand_uint8(64, 64, 1)
        cfg = PatchConfig(patch_h=16, patch_w=16)
        patch = extract_patch(img, cy, cx, cfg)
        assert patch.shape == (16, 16)

    def test_padding_fills_outside(self) -> None:
        img = _rand_uint8(64, 64, 1)
        pad_val = 128
        cfg = PatchConfig(patch_h=32, patch_w=32, pad_value=pad_val)
        # Center at corner so top-left padding region exists
        patch = extract_patch(img, 0, 0, cfg)
        assert patch.shape == (32, 32)
        # Top-left quadrant (16x16) should be pad_value since center is (0,0)
        assert int(patch[0, 0]) == pad_val

    def test_full_image_no_padding(self) -> None:
        """Patch centered at image center with tile_h=img_h should not need padding."""
        img = _rand_uint8(8, 8, 1)
        cfg = PatchConfig(patch_h=8, patch_w=8)
        patch = extract_patch(img, 4, 4, cfg)
        assert patch.shape == (8, 8)

    def test_normalize_minmax_values_in_range(self) -> None:
        img = _rand_uint8(32, 32, 1)
        cfg = PatchConfig(patch_h=16, patch_w=16, normalize=True, norm_mode="minmax")
        patch = extract_patch(img, 16, 16, cfg)
        assert patch.dtype == np.float32
        assert float(patch.min()) >= 0.0 - 1e-6
        assert float(patch.max()) <= 1.0 + 1e-6

    def test_extract_patches_length(self) -> None:
        img = _rand_uint8(64, 64, 1)
        centers = [(16, 16), (32, 32), (48, 48), (10, 10)]
        patches = extract_patches(img, centers)
        assert len(patches) == len(centers)

    def test_extract_patches_each_shape(self) -> None:
        img = _rand_uint8(64, 64, 1)
        cfg = PatchConfig(patch_h=8, patch_w=8)
        centers = [(16, 16), (32, 32), (48, 48)]
        patches = extract_patches(img, centers, cfg)
        for p in patches:
            assert p.shape == (8, 8)


# ═══════════════════════════════════════════════════════════════════════════════
# patch_utils — normalize_patch
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalizePatch:
    """normalize_patch: range and dtype invariants."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_minmax_in_unit_range(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        patch = rng.integers(0, 256, (16, 16), dtype=np.uint8)
        result = normalize_patch(patch, mode="minmax")
        assert result.dtype == np.float32
        assert float(result.min()) >= -1e-6
        assert float(result.max()) <= 1.0 + 1e-6

    def test_constant_patch_minmax_gives_zeros(self) -> None:
        patch = np.full((8, 8), 128, dtype=np.uint8)
        result = normalize_patch(patch, mode="minmax")
        assert np.allclose(result, 0.0)

    def test_constant_patch_zscore_gives_zeros(self) -> None:
        patch = np.full((8, 8), 200.0, dtype=np.float32)
        result = normalize_patch(patch, mode="zscore")
        assert np.allclose(result, 0.0)

    @pytest.mark.parametrize("seed", [10, 11, 12])
    def test_zscore_float_output(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        patch = rng.random((12, 12)).astype(np.float32) * 200
        result = normalize_patch(patch, mode="zscore")
        assert result.dtype == np.float32


# ═══════════════════════════════════════════════════════════════════════════════
# patch_utils — patch_ssd, patch_ncc, patch_mse
# ═══════════════════════════════════════════════════════════════════════════════

class TestPatchMetrics:
    """patch_ssd, patch_ncc, patch_mse: invariants."""

    def _make_pair(self, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        a = rng.integers(0, 256, (16, 16), dtype=np.uint8)
        b = rng.integers(0, 256, (16, 16), dtype=np.uint8)
        return a, b

    # --- patch_ssd ---

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_ssd_nonnegative(self, seed: int) -> None:
        a, b = self._make_pair(seed)
        assert patch_ssd(a, b) >= 0.0

    @pytest.mark.parametrize("size", [8, 16, 32])
    def test_ssd_self_is_zero(self, size: int) -> None:
        a = _rand_uint8(size, size, 1)
        assert patch_ssd(a, a) == pytest.approx(0.0)

    @pytest.mark.parametrize("seed", [0, 3, 7])
    def test_ssd_symmetric(self, seed: int) -> None:
        a, b = self._make_pair(seed)
        assert patch_ssd(a, b) == pytest.approx(patch_ssd(b, a))

    # --- patch_ncc ---

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_ncc_in_range(self, seed: int) -> None:
        a, b = self._make_pair(seed)
        val = patch_ncc(a, b)
        assert -1.0 - 1e-9 <= val <= 1.0 + 1e-9

    @pytest.mark.parametrize("seed", [5, 10, 15])
    def test_ncc_self_is_one_for_nonconstant(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        a = rng.integers(1, 256, (16, 16), dtype=np.uint8)
        # Ensure not constant
        a[0, 0] = 0
        a[15, 15] = 255
        val = patch_ncc(a, a)
        assert val == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("seed", [0, 3, 6])
    def test_ncc_symmetric(self, seed: int) -> None:
        a, b = self._make_pair(seed)
        assert patch_ncc(a, b) == pytest.approx(patch_ncc(b, a), abs=1e-12)

    def test_ncc_constant_patch_returns_zero(self) -> None:
        a = np.full((8, 8), 100, dtype=np.uint8)
        b = _rand_uint8(8, 8, 1)
        assert patch_ncc(a, b) == pytest.approx(0.0)

    # --- patch_mse ---

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_mse_nonnegative(self, seed: int) -> None:
        a, b = self._make_pair(seed)
        assert patch_mse(a, b) >= 0.0

    @pytest.mark.parametrize("size", [8, 16, 32])
    def test_mse_self_is_zero(self, size: int) -> None:
        a = _rand_uint8(size, size, 1)
        assert patch_mse(a, a) == pytest.approx(0.0)

    @pytest.mark.parametrize("seed", [0, 4, 8])
    def test_mse_symmetric(self, seed: int) -> None:
        a, b = self._make_pair(seed)
        assert patch_mse(a, b) == pytest.approx(patch_mse(b, a))

    def test_mse_leq_ssd_divided_by_size(self) -> None:
        """MSE = SSD / n_pixels."""
        a = _rand_uint8(8, 8, 1)
        b = _rand_uint8(8, 8, 1)
        ssd = patch_ssd(a, b)
        mse = patch_mse(a, b)
        assert mse == pytest.approx(ssd / a.size, abs=1e-8)


# ═══════════════════════════════════════════════════════════════════════════════
# patch_utils — batch_compare, compare_patches
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchCompare:
    """batch_compare: length and value-range invariants."""

    @pytest.mark.parametrize("method", ["ncc", "ssd", "mse"])
    def test_length_preserved(self, method: str) -> None:
        pairs = [
            (_rand_uint8(8, 8, 1), _rand_uint8(8, 8, 1))
            for _ in range(5)
        ]
        results = batch_compare(pairs, method=method)
        assert len(results) == 5

    def test_ncc_values_in_range(self) -> None:
        pairs = [
            (_rand_uint8(8, 8, 1), _rand_uint8(8, 8, 1))
            for _ in range(6)
        ]
        results = batch_compare(pairs, method="ncc")
        for v in results:
            assert -1.0 - 1e-9 <= v <= 1.0 + 1e-9

    def test_ssd_nonnegative(self) -> None:
        pairs = [
            (_rand_uint8(8, 8, 1), _rand_uint8(8, 8, 1))
            for _ in range(6)
        ]
        results = batch_compare(pairs, method="ssd")
        for v in results:
            assert v >= 0.0

    def test_compare_patches_delegates(self) -> None:
        a = _rand_uint8(8, 8, 1)
        b = _rand_uint8(8, 8, 1)
        assert compare_patches(a, b, "ncc") == patch_ncc(a, b)
        assert compare_patches(a, b, "ssd") == patch_ssd(a, b)
        assert compare_patches(a, b, "mse") == patch_mse(a, b)


# ═══════════════════════════════════════════════════════════════════════════════
# sparse_utils — to_sparse_entries / from_sparse_entries
# ═══════════════════════════════════════════════════════════════════════════════

class TestSparseRoundtrip:
    """to_sparse_entries + from_sparse_entries: roundtrip and filter invariants."""

    @pytest.mark.parametrize("n", [4, 5, 6, 8, 10])
    def test_roundtrip_zero_threshold(self, n: int) -> None:
        mat = _rand_square_matrix(n)
        entries = to_sparse_entries(mat, threshold=0.0)
        recovered = from_sparse_entries(entries, n, n)
        # Values > 0 are preserved; near-zero may differ slightly
        np.testing.assert_allclose(recovered, mat, atol=1e-10)

    @pytest.mark.parametrize("thresh", [0.3, 0.5, 0.7])
    def test_entries_above_threshold(self, thresh: float) -> None:
        mat = _rand_square_matrix(6)
        entries = to_sparse_entries(mat, threshold=thresh)
        for e in entries:
            assert abs(e.value) > thresh

    @pytest.mark.parametrize("n", [3, 5, 7])
    def test_from_entries_shape(self, n: int) -> None:
        mat = _rand_square_matrix(n)
        entries = to_sparse_entries(mat)
        recovered = from_sparse_entries(entries, n, n)
        assert recovered.shape == (n, n)

    def test_empty_matrix_gives_no_entries(self) -> None:
        mat = np.zeros((5, 5))
        entries = to_sparse_entries(mat, threshold=0.0)
        assert len(entries) == 0

    def test_from_entries_fills_zeros(self) -> None:
        entries = [SparseEntry(row=0, col=1, value=0.5)]
        out = from_sparse_entries(entries, 3, 3)
        assert out[0, 1] == pytest.approx(0.5)
        assert out[0, 0] == pytest.approx(0.0)
        assert out[1, 0] == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# sparse_utils — symmetrize_matrix
# ═══════════════════════════════════════════════════════════════════════════════

class TestSymmetrizeMatrix:
    """symmetrize_matrix: symmetry and element-wise max."""

    @pytest.mark.parametrize("n", [3, 4, 5, 6, 7])
    def test_result_is_symmetric(self, n: int) -> None:
        mat = _rand_square_matrix(n)
        result = symmetrize_matrix(mat)
        np.testing.assert_allclose(result, result.T, atol=1e-15)

    @pytest.mark.parametrize("n", [3, 5, 7])
    def test_elementwise_is_max(self, n: int) -> None:
        mat = _rand_square_matrix(n)
        result = symmetrize_matrix(mat)
        for i in range(n):
            for j in range(n):
                expected = max(mat[i, j], mat[j, i])
                assert result[i, j] == pytest.approx(expected)

    def test_already_symmetric_unchanged(self) -> None:
        mat = _rand_square_matrix(5)
        sym = (mat + mat.T) / 2
        result = symmetrize_matrix(sym)
        np.testing.assert_allclose(result, sym, atol=1e-14)

    def test_result_gte_original(self) -> None:
        mat = _rand_square_matrix(6)
        result = symmetrize_matrix(mat)
        assert np.all(result >= mat - 1e-15)


# ═══════════════════════════════════════════════════════════════════════════════
# sparse_utils — diagonal_zeros
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiagonalZeros:
    """diagonal_zeros: diagonal = 0; off-diagonal preserved."""

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_diagonal_all_zero(self, n: int) -> None:
        mat = _rand_square_matrix(n, low=0.1, high=1.0)
        result = diagonal_zeros(mat)
        diag = np.diag(result)
        assert np.allclose(diag, 0.0)

    @pytest.mark.parametrize("n", [3, 5, 6])
    def test_off_diagonal_preserved(self, n: int) -> None:
        mat = _rand_square_matrix(n)
        result = diagonal_zeros(mat)
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert result[i, j] == pytest.approx(mat[i, j])

    def test_idempotent(self) -> None:
        mat = _rand_square_matrix(5)
        r1 = diagonal_zeros(mat)
        r2 = diagonal_zeros(r1)
        np.testing.assert_allclose(r1, r2)


# ═══════════════════════════════════════════════════════════════════════════════
# sparse_utils — normalize_matrix
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalizeMatrix:
    """normalize_matrix: values in [0, 1]; row max property."""

    @pytest.mark.parametrize("n,axis", [(4, 0), (4, 1), (5, 0), (5, 1), (6, 1)])
    def test_values_in_unit_range(self, n: int, axis: int) -> None:
        mat = _rand_square_matrix(n, low=0.0, high=10.0)
        result = normalize_matrix(mat, axis=axis)
        assert float(result.min()) >= -1e-9
        assert float(result.max()) <= 1.0 + 1e-9

    @pytest.mark.parametrize("n", [4, 5, 6])
    def test_row_max_leq_one(self, n: int) -> None:
        mat = _rand_square_matrix(n, low=0.0, high=5.0)
        result = normalize_matrix(mat, axis=1)
        for i in range(n):
            assert float(result[i].max()) <= 1.0 + 1e-9

    def test_all_zero_row_stays_zero(self) -> None:
        mat = np.array([[0.0, 0.0], [1.0, 2.0]])
        result = normalize_matrix(mat, axis=1)
        np.testing.assert_allclose(result[0], 0.0)

    def test_positive_values_preserved_relative(self) -> None:
        mat = np.array([[1.0, 2.0, 4.0]])
        result = normalize_matrix(mat, axis=1)
        # Max should be 1.0, others proportionally smaller
        assert float(result[0, 2]) == pytest.approx(1.0)
        assert float(result[0, 0]) < float(result[0, 1]) < float(result[0, 2])


# ═══════════════════════════════════════════════════════════════════════════════
# sparse_utils — matrix_sparsity
# ═══════════════════════════════════════════════════════════════════════════════

class TestMatrixSparsity:
    """matrix_sparsity: range and special cases."""

    @pytest.mark.parametrize("n", [3, 5, 7, 10])
    def test_in_unit_range(self, n: int) -> None:
        mat = _rand_square_matrix(n)
        s = matrix_sparsity(mat)
        assert 0.0 <= s <= 1.0

    def test_all_zeros_gives_one(self) -> None:
        mat = np.zeros((5, 5))
        assert matrix_sparsity(mat) == pytest.approx(1.0)

    def test_all_nonzero_gives_zero(self) -> None:
        mat = np.ones((5, 5))
        assert matrix_sparsity(mat) == pytest.approx(0.0)

    def test_half_zeros(self) -> None:
        mat = np.zeros((4, 4))
        mat[:2, :] = 1.0
        s = matrix_sparsity(mat)
        assert s == pytest.approx(0.5)

    @pytest.mark.parametrize("n", [4, 6])
    def test_sparsity_plus_density_is_one(self, n: int) -> None:
        mat = _rand_square_matrix(n)
        s = matrix_sparsity(mat)
        density = float(np.count_nonzero(mat)) / mat.size
        assert s + density == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# sparse_utils — threshold_matrix, top_k_per_row
# ═══════════════════════════════════════════════════════════════════════════════

class TestThresholdAndTopK:
    """threshold_matrix and top_k_per_row invariants."""

    @pytest.mark.parametrize("thresh,fill", [(0.5, 0.0), (0.3, -1.0), (0.7, 0.0)])
    def test_threshold_matrix_no_values_below(self, thresh: float, fill: float) -> None:
        mat = _rand_square_matrix(6)
        result = threshold_matrix(mat, threshold=thresh, fill=fill)
        # All values are either fill or >= thresh
        mask_fill = np.isclose(result, fill)
        mask_above = result >= thresh - 1e-12
        assert np.all(mask_fill | mask_above)

    def test_threshold_matrix_shape_preserved(self) -> None:
        mat = _rand_square_matrix(5)
        result = threshold_matrix(mat, 0.5)
        assert result.shape == mat.shape

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_top_k_per_row_count(self, k: int) -> None:
        mat = _rand_square_matrix(5)
        result = top_k_per_row(mat, k)
        for i in range(5):
            n_nonzero = int(np.count_nonzero(result[i]))
            assert n_nonzero <= k

    def test_top_k_shape_preserved(self) -> None:
        mat = _rand_square_matrix(6)
        result = top_k_per_row(mat, k=2)
        assert result.shape == mat.shape

    def test_sparse_top_k_count_per_row(self) -> None:
        mat = _rand_square_matrix(5)
        k = 2
        entries = sparse_top_k(mat, k=k)
        from collections import Counter
        row_counts = Counter(e.row for e in entries)
        for row, cnt in row_counts.items():
            assert cnt <= k


# ═══════════════════════════════════════════════════════════════════════════════
# tile_utils — tile_image
# ═══════════════════════════════════════════════════════════════════════════════

class TestTileImage:
    """tile_image: shape, count, non-empty invariants."""

    @pytest.mark.parametrize("th,tw", [(8, 8), (16, 16), (32, 32)])
    def test_all_tiles_correct_shape(self, th: int, tw: int) -> None:
        img = _rand_uint8(64, 64, 1)
        cfg = TileConfig(tile_h=th, tile_w=tw)
        tiles = tile_image(img, cfg)
        assert len(tiles) > 0
        for tile in tiles:
            assert tile.data.shape == (th, tw)

    @pytest.mark.parametrize("th,tw,c", [(8, 8, 3), (16, 16, 3)])
    def test_color_tiles_shape(self, th: int, tw: int, c: int) -> None:
        img = _rand_uint8(64, 64, c)
        cfg = TileConfig(tile_h=th, tile_w=tw)
        tiles = tile_image(img, cfg)
        for tile in tiles:
            assert tile.data.shape == (th, tw, c)

    def test_no_overlap_count(self) -> None:
        img = _rand_uint8(64, 64, 1)
        cfg = TileConfig(tile_h=8, tile_w=8)  # stride=tile size by default
        tiles = tile_image(img, cfg)
        # 64/8 = 8 rows × 8 cols = 64 tiles
        assert len(tiles) == 64

    def test_tile_dtype_preserved(self) -> None:
        img = _rand_uint8(32, 32, 1)
        cfg = TileConfig(tile_h=8, tile_w=8)
        tiles = tile_image(img, cfg)
        for tile in tiles:
            assert tile.data.dtype == np.uint8

    def test_row_col_monotone(self) -> None:
        img = _rand_uint8(32, 32, 1)
        cfg = TileConfig(tile_h=8, tile_w=8)
        tiles = tile_image(img, cfg)
        rows = [t.row for t in tiles]
        # Rows should be non-decreasing
        assert rows == sorted(rows)

    def test_source_dimensions_recorded(self) -> None:
        img = _rand_uint8(48, 64, 1)
        cfg = TileConfig(tile_h=16, tile_w=16)
        tiles = tile_image(img, cfg)
        for t in tiles:
            assert t.source_h == 48
            assert t.source_w == 64

    def test_batch_tile_images_length(self) -> None:
        images = [_rand_uint8(32, 32, 1) for _ in range(4)]
        cfg = TileConfig(tile_h=8, tile_w=8)
        result = batch_tile_images(images, cfg)
        assert len(result) == 4


# ═══════════════════════════════════════════════════════════════════════════════
# tile_utils — tile_overlap_ratio
# ═══════════════════════════════════════════════════════════════════════════════

class TestTileOverlapRatio:
    """tile_overlap_ratio: IoU invariants."""

    def _make_tile(self, y: int, x: int, h: int = 8, w: int = 8) -> Tile:
        data = _rand_uint8(h, w, 1)
        return Tile(data=data, row=0, col=0, y=y, x=x, source_h=64, source_w=64)

    @pytest.mark.parametrize("ya,xa,yb,xb", [
        (0, 0, 10, 10), (0, 0, 4, 4), (0, 0, 0, 20)
    ])
    def test_in_unit_range(self, ya: int, xa: int, yb: int, xb: int) -> None:
        a = self._make_tile(ya, xa)
        b = self._make_tile(yb, xb)
        ratio = tile_overlap_ratio(a, b)
        assert 0.0 <= ratio <= 1.0

    @pytest.mark.parametrize("y,x", [(0, 0), (8, 8), (16, 32)])
    def test_self_overlap_is_one(self, y: int, x: int) -> None:
        a = self._make_tile(y, x)
        ratio = tile_overlap_ratio(a, a)
        assert ratio == pytest.approx(1.0)

    @pytest.mark.parametrize("offset", [8, 16, 24])
    def test_no_overlap_gives_zero(self, offset: int) -> None:
        a = self._make_tile(0, 0)
        b = self._make_tile(0, offset)  # 8px wide, offset=8 → just touching
        ratio = tile_overlap_ratio(a, b)
        # Non-overlapping tiles should give 0
        if offset >= 8:
            assert ratio == pytest.approx(0.0)

    @pytest.mark.parametrize("ya,xa,yb,xb", [(0, 0, 4, 4), (0, 0, 6, 2)])
    def test_commutative(self, ya: int, xa: int, yb: int, xb: int) -> None:
        a = self._make_tile(ya, xa)
        b = self._make_tile(yb, xb)
        assert tile_overlap_ratio(a, b) == pytest.approx(tile_overlap_ratio(b, a))


# ═══════════════════════════════════════════════════════════════════════════════
# tile_utils — compute_tile_grid
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeTileGrid:
    """compute_tile_grid: at least 1 entry; coordinates within bounds."""

    @pytest.mark.parametrize("img_h,img_w", [(32, 32), (48, 64), (16, 16)])
    def test_nonempty(self, img_h: int, img_w: int) -> None:
        cfg = TileConfig(tile_h=8, tile_w=8)
        grid = compute_tile_grid(img_h, img_w, cfg)
        assert len(grid) >= 1

    @pytest.mark.parametrize("img_h,img_w", [(32, 32), (48, 64)])
    def test_y_within_image(self, img_h: int, img_w: int) -> None:
        cfg = TileConfig(tile_h=8, tile_w=8)
        grid = compute_tile_grid(img_h, img_w, cfg)
        for (y, x, row, col) in grid:
            assert 0 <= y < img_h

    @pytest.mark.parametrize("img_h,img_w", [(32, 32), (48, 64)])
    def test_x_within_image(self, img_h: int, img_w: int) -> None:
        cfg = TileConfig(tile_h=8, tile_w=8)
        grid = compute_tile_grid(img_h, img_w, cfg)
        for (y, x, row, col) in grid:
            assert 0 <= x < img_w

    def test_returns_tuples(self) -> None:
        cfg = TileConfig(tile_h=8, tile_w=8)
        grid = compute_tile_grid(32, 32, cfg)
        for item in grid:
            assert len(item) == 4


# ═══════════════════════════════════════════════════════════════════════════════
# tile_utils — filter_tiles_by_content
# ═══════════════════════════════════════════════════════════════════════════════

class TestFilterTilesByContent:
    """filter_tiles_by_content: subset and edge case invariants."""

    def _make_tiles(self, n: int) -> List[Tile]:
        tiles = []
        for i in range(n):
            data = _rand_uint8(8, 8, 1)
            tiles.append(Tile(data=data, row=0, col=i, y=0, x=i*8,
                              source_h=8, source_w=n*8))
        return tiles

    def test_result_is_subset(self) -> None:
        tiles = self._make_tiles(6)
        result = filter_tiles_by_content(tiles, min_foreground=0.3)
        tile_ids = {id(t) for t in tiles}
        for t in result:
            assert id(t) in tile_ids

    def test_min_fg_zero_returns_all(self) -> None:
        data_all_ones = np.ones((8, 8), dtype=np.uint8)
        tiles = [
            Tile(data=data_all_ones, row=0, col=i, y=0, x=i*8,
                 source_h=8, source_w=6*8)
            for i in range(6)
        ]
        result = filter_tiles_by_content(tiles, min_foreground=0.0)
        assert len(result) == len(tiles)

    def test_all_black_tiles_filtered_with_high_threshold(self) -> None:
        data_zeros = np.zeros((8, 8), dtype=np.uint8)
        tiles = [
            Tile(data=data_zeros, row=0, col=i, y=0, x=i*8,
                 source_h=8, source_w=6*8)
            for i in range(4)
        ]
        result = filter_tiles_by_content(tiles, min_foreground=0.5)
        assert len(result) == 0

    def test_result_length_leq_input(self) -> None:
        tiles = self._make_tiles(8)
        result = filter_tiles_by_content(tiles, min_foreground=0.5)
        assert len(result) <= len(tiles)
