"""Integration tests for utils batch 3b (gradient, image_cluster, image_pipeline,
image_transform, interpolation)."""
import math

import numpy as np
import pytest

rng = np.random.default_rng(42)

from puzzle_reconstruction.utils.gradient_utils import (
    GradientConfig,
    batch_compute_gradients,
    compute_edge_density,
    compute_gradient_direction,
    compute_gradient_magnitude,
    compute_laplacian,
    compute_sobel,
    suppress_non_maximum,
    threshold_gradient,
)
from puzzle_reconstruction.utils.image_cluster_utils import (
    ImageStatsAnalysisConfig,
    ImageStatsAnalysisEntry,
    batch_summarise_image_stats_entries,
    best_image_stats_entry,
    compare_image_stats_summaries,
    filter_by_max_entropy,
    filter_by_min_contrast,
    filter_by_min_sharpness,
    image_stats_score_stats,
    make_image_stats_entry,
    summarise_image_stats_entries,
    top_k_sharpest,
)
from puzzle_reconstruction.utils.image_pipeline_utils import (
    CanvasBuildRecord,
    CanvasBuildSummary,
    FrequencyMatchRecord,
    FrequencyMatchSummary,
    PatchMatchRecord,
    PatchMatchSummary,
    filter_frequency_matches,
    summarize_canvas_builds,
    summarize_frequency_matches,
    summarize_patch_matches,
    top_frequency_matches,
)
from puzzle_reconstruction.utils.image_transform_utils import (
    ImageTransformConfig,
    TransformResult,
    apply_affine,
    batch_pad,
    batch_resize_to_max,
    batch_rotate,
    crop_image,
    flip_horizontal,
    flip_vertical,
    pad_image,
    resize_image,
    resize_to_max_side,
    rotate_image,
    rotation_matrix_2x3,
)
from puzzle_reconstruction.utils.interpolation_utils import (
    InterpolationConfig,
    batch_resample,
    bilinear_interpolate,
    fill_missing,
    interpolate_scores,
    lerp,
    lerp_array,
    resample_1d,
    smooth_interpolate,
)

def _gray(h=32, w=32):
    return rng.integers(0, 256, (h, w), dtype=np.uint8)

def _color(h=32, w=32):
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

def _make_entry(fid, sharpness=1.0, entropy=5.0, contrast=20.0):
    return make_image_stats_entry(fid, sharpness, entropy, contrast, 128.0, 1024)

# ===========================================================================
# gradient_utils
# ===========================================================================

class TestGradientUtils:
    def test_gradient_config_defaults(self):
        cfg = GradientConfig()
        assert cfg.ksize == 3 and cfg.normalize is True and cfg.threshold == 32.0

    def test_gradient_config_invalid_ksize(self):
        with pytest.raises(ValueError):
            GradientConfig(ksize=2)

    def test_compute_gradient_magnitude_shape_and_dtype(self):
        mag = compute_gradient_magnitude(_gray())
        assert mag.dtype == np.float32 and mag.max() <= 1.0 + 1e-6

    def test_compute_gradient_magnitude_color(self):
        assert compute_gradient_magnitude(_color()).ndim == 2

    def test_compute_gradient_direction_range(self):
        d = compute_gradient_direction(_gray())
        assert d.min() >= -math.pi - 1e-5 and d.max() <= math.pi + 1e-5

    def test_compute_sobel_returns_triple(self):
        img = _gray()
        mag, dx, dy = compute_sobel(img)
        assert mag.shape == dx.shape == dy.shape == img.shape

    def test_compute_laplacian_shape_and_dtype(self):
        img = _gray()
        lap = compute_laplacian(img)
        assert lap.shape == img.shape
        assert lap.dtype == np.float32

    def test_threshold_gradient_binary(self):
        img = _gray()
        mag = compute_gradient_magnitude(img)
        mask = threshold_gradient(mag)
        assert mask.dtype == bool

    def test_suppress_non_maximum_shape(self):
        img = _gray()
        mag, _, _ = compute_sobel(img)
        d = compute_gradient_direction(img)
        out = suppress_non_maximum(mag, d)
        assert out.shape == mag.shape

    def test_compute_edge_density_range(self):
        img = _gray()
        density = compute_edge_density(img)
        assert 0.0 <= density <= 1.0

    def test_batch_compute_gradients_count(self):
        imgs = [_gray() for _ in range(3)]
        mags = batch_compute_gradients(imgs)
        assert len(mags) == 3 and all(m.ndim == 2 for m in mags)


# ===========================================================================
# image_cluster_utils
# ===========================================================================

class TestImageClusterUtils:
    def _entries(self):
        return [
            _make_entry(0, sharpness=10.0, entropy=4.0, contrast=30.0),
            _make_entry(1, sharpness=5.0,  entropy=6.0, contrast=15.0),
            _make_entry(2, sharpness=8.0,  entropy=7.5, contrast=25.0),
        ]

    def test_make_entry_fields(self):
        e = _make_entry(7, sharpness=3.5, entropy=2.0, contrast=12.0)
        assert e.fragment_id == 7 and e.sharpness == pytest.approx(3.5)

    def test_summarise_empty(self):
        s = summarise_image_stats_entries([])
        assert s.n_images == 0 and s.sharpest_id is None

    def test_summarise_nonempty(self):
        s = summarise_image_stats_entries(self._entries())
        assert s.n_images == 3 and s.sharpest_id == 0 and s.blurriest_id == 1

    def test_filter_by_min_sharpness(self):
        entries = self._entries()
        result = filter_by_min_sharpness(entries, 8.0)
        ids = {e.fragment_id for e in result}
        assert 0 in ids and 2 in ids and 1 not in ids

    def test_filter_by_max_entropy(self):
        entries = self._entries()
        result = filter_by_max_entropy(entries, 6.0)
        assert all(e.entropy <= 6.0 for e in result)

    def test_filter_by_min_contrast(self):
        entries = self._entries()
        result = filter_by_min_contrast(entries, 20.0)
        assert all(e.contrast >= 20.0 for e in result)

    def test_top_k_sharpest(self):
        entries = self._entries()
        top = top_k_sharpest(entries, 2)
        assert len(top) == 2
        assert top[0].sharpness >= top[1].sharpness

    def test_best_image_stats_entry(self):
        entries = self._entries()
        best = best_image_stats_entry(entries)
        assert best.sharpness == max(e.sharpness for e in entries)

    def test_image_stats_score_stats_keys(self):
        stats = image_stats_score_stats(self._entries())
        assert {"min", "max", "mean", "std", "count"} <= stats.keys()

    def test_compare_summaries_delta(self):
        entries = self._entries()
        a = summarise_image_stats_entries(entries[:2])
        b = summarise_image_stats_entries(entries)
        assert "delta_mean_sharpness" in compare_image_stats_summaries(a, b)

    def test_batch_summarise_length(self):
        entries = self._entries()
        summaries = batch_summarise_image_stats_entries([entries[:2], entries[2:]])
        assert len(summaries) == 2


# ===========================================================================
# image_pipeline_utils
# ===========================================================================

class TestImagePipelineUtils:
    def _freq_records(self):
        return [
            FrequencyMatchRecord(0, 1, 0.9),
            FrequencyMatchRecord(0, 2, 0.3),
            FrequencyMatchRecord(1, 2, 0.7),
        ]

    def test_freq_match_record_pair_and_similar(self):
        r = FrequencyMatchRecord(3, 5, 0.8)
        assert r.pair == (3, 5) and r.is_similar is True

    def test_freq_match_record_invalid_similarity(self):
        with pytest.raises(ValueError):
            FrequencyMatchRecord(0, 1, 1.5)

    def test_summarize_frequency_matches(self):
        s = summarize_frequency_matches(self._freq_records())
        assert s.total_pairs == 3 and s.similar_pairs == 2

    def test_summarize_frequency_matches_empty(self):
        assert summarize_frequency_matches([]).total_pairs == 0

    def test_filter_and_top_frequency_matches(self):
        records = self._freq_records()
        assert all(r.similarity >= 0.5 for r in filter_frequency_matches(records, 0.5))
        top = top_frequency_matches(records, 2)
        assert len(top) == 2 and top[0].similarity >= top[1].similarity

    def test_canvas_build_record_properties(self):
        rec = CanvasBuildRecord(n_fragments=5, coverage=0.8, canvas_w=100, canvas_h=100)
        assert rec.canvas_area == 10000 and rec.is_well_covered is True

    def test_summarize_canvas_builds(self):
        records = [CanvasBuildRecord(4, 0.9, 50, 50), CanvasBuildRecord(3, 0.5, 50, 50)]
        s = summarize_canvas_builds(records)
        assert s.n_canvases == 2 and s.total_fragments == 7

    def test_patch_match_displacement_and_summary(self):
        assert PatchMatchRecord(10, 20, 15, 25, 0.95).displacement == (5, 5)
        batch = [[PatchMatchRecord(0, 0, 1, 1, 0.8), PatchMatchRecord(0, 1, 1, 2, 0.7)],
                 [PatchMatchRecord(1, 0, 2, 1, 0.6)]]
        s = summarize_patch_matches(batch)
        assert s.n_pairs == 2 and s.n_total_matches == 3

    def test_freq_summary_similar_ratio(self):
        s = FrequencyMatchSummary(total_pairs=4, similar_pairs=2, mean_similarity=0.5,
                                  max_similarity=1.0, min_similarity=0.0)
        assert s.similar_ratio == pytest.approx(0.5)


# ===========================================================================
# image_transform_utils
# ===========================================================================

class TestImageTransformUtils:
    def test_rotate_preserves_shape(self):
        img = _gray()
        assert rotate_image(img, math.pi / 4).shape == img.shape

    def test_rotate_zero_is_identity(self):
        img = _gray()
        assert np.array_equal(rotate_image(img, 0.0), img)

    def test_flips(self):
        img = _gray()
        assert np.array_equal(flip_horizontal(img), img[:, ::-1])
        assert np.array_equal(flip_vertical(img), img[::-1, :])

    def test_pad_image_shape(self):
        assert pad_image(_gray(10, 10), top=2, bottom=3, left=1, right=4).shape == (15, 15)

    def test_crop_image_shape(self):
        assert crop_image(_gray(20, 20), 2, 3, 10, 15).shape == (8, 12)

    def test_resize_image_target_size(self):
        assert resize_image(_gray(40, 40), (20, 20)).shape == (20, 20)

    def test_resize_to_max_side(self):
        assert max(resize_to_max_side(_gray(100, 50), 50).shape[:2]) == 50

    def test_rotation_matrix_2x3_shape(self):
        assert rotation_matrix_2x3(math.pi / 6, 16.0, 16.0).shape == (2, 3)

    def test_batch_rotate_and_pad(self):
        imgs = [_gray(10, 10) for _ in range(3)]
        assert len(batch_rotate(imgs, math.pi / 8)) == 3
        assert all(r.shape == (20, 20) for r in batch_pad(imgs, 5))

    def test_transform_result_to_dict(self):
        tr = TransformResult(image=_gray(), angle_rad=0.5, scale=1.0, translation=(0.0, 0.0))
        assert "angle_deg" in tr.to_dict()


# ===========================================================================
# interpolation_utils
# ===========================================================================

class TestInterpolationUtils:
    def test_lerp_endpoints(self):
        assert lerp(0.0, 10.0, 0.0) == pytest.approx(0.0)
        assert lerp(0.0, 10.0, 1.0) == pytest.approx(10.0)

    def test_lerp_midpoint(self):
        assert lerp(0.0, 10.0, 0.5) == pytest.approx(5.0)

    def test_lerp_invalid_t(self):
        with pytest.raises(ValueError):
            lerp(0.0, 1.0, -0.1)

    def test_lerp_array_shape(self):
        a = np.array([0.0, 2.0, 4.0])
        b = np.array([1.0, 3.0, 5.0])
        out = lerp_array(a, b, 0.5)
        assert out.shape == a.shape
        np.testing.assert_allclose(out, [0.5, 2.5, 4.5])

    def test_lerp_array_mismatch_raises(self):
        with pytest.raises(ValueError):
            lerp_array(np.ones(3), np.ones(4), 0.5)

    def test_bilinear_interpolate_exact_corner(self):
        grid = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert bilinear_interpolate(grid, 0.0, 0.0) == pytest.approx(1.0)
        assert bilinear_interpolate(grid, 1.0, 1.0) == pytest.approx(4.0)

    def test_bilinear_interpolate_center(self):
        grid = np.array([[0.0, 0.0], [0.0, 4.0]])
        val = bilinear_interpolate(grid, 0.5, 0.5)
        assert val == pytest.approx(1.0)

    def test_resample_1d_upsample_downsample(self):
        arr = np.array([0.0, 1.0, 2.0, 3.0])
        assert len(resample_1d(arr, 8)) == 8
        assert len(resample_1d(rng.random(20), 5)) == 5

    def test_fill_missing_no_nans(self):
        arr = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(fill_missing(arr), arr)

    def test_fill_missing_interior_nan(self):
        arr = np.array([1.0, np.nan, 3.0])
        out = fill_missing(arr)
        assert not np.any(np.isnan(out))
        assert out[1] == pytest.approx(2.0)

    def test_interpolate_scores_symmetry(self):
        M = rng.random((4, 4))
        out = interpolate_scores(M, alpha=0.5)
        np.testing.assert_allclose(out, out.T, atol=1e-10)

    def test_smooth_interpolate_same_length(self):
        arr = rng.random(10)
        assert len(smooth_interpolate(arr, window=3)) == len(arr)

    def test_batch_resample_lengths(self):
        results = batch_resample([rng.random(n) for n in [5, 10, 15]], 8)
        assert all(len(r) == 8 for r in results)

    def test_interpolation_config_invalid_method(self):
        with pytest.raises(ValueError):
            InterpolationConfig(method="cubic")
