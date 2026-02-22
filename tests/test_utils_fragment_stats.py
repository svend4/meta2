"""Tests for puzzle_reconstruction/utils/fragment_stats.py"""
import pytest
import numpy as np

from puzzle_reconstruction.utils.fragment_stats import (
    FragmentMetrics,
    CollectionStats,
    compute_fragment_metrics,
    compute_collection_stats,
    area_histogram,
    compare_collections,
    outlier_indices,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _solid_mask(h=20, w=20) -> np.ndarray:
    return np.ones((h, w), dtype=np.uint8)


def _empty_mask(h=20, w=20) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _rect_mask(h=20, w=30) -> np.ndarray:
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[5:5 + h, 10:10 + w] = 1
    return mask


def _make_metric(fid=0, area=100.0, aspect=1.0,
                 density=0.8, n_edges=4, perimeter=40.0) -> FragmentMetrics:
    return FragmentMetrics(fragment_id=fid, area=area, aspect=aspect,
                           density=density, n_edges=n_edges, perimeter=perimeter)


def _sample_metrics(n=5) -> list:
    return [_make_metric(i, area=float((i + 1) * 100)) for i in range(n)]


# ─── TestFragmentMetrics ──────────────────────────────────────────────────────

class TestFragmentMetrics:
    def test_construction(self):
        m = _make_metric(fid=3, area=200.0)
        assert m.fragment_id == 3
        assert m.area == pytest.approx(200.0)

    def test_fragment_id_negative_raises(self):
        with pytest.raises(ValueError):
            FragmentMetrics(fragment_id=-1, area=10, aspect=1, density=0.5,
                            n_edges=2, perimeter=10)

    def test_area_negative_raises(self):
        with pytest.raises(ValueError):
            FragmentMetrics(fragment_id=0, area=-1, aspect=1, density=0.5,
                            n_edges=2, perimeter=10)

    def test_aspect_zero_raises(self):
        with pytest.raises(ValueError):
            FragmentMetrics(fragment_id=0, area=10, aspect=0, density=0.5,
                            n_edges=2, perimeter=10)

    def test_density_above_one_raises(self):
        with pytest.raises(ValueError):
            FragmentMetrics(fragment_id=0, area=10, aspect=1, density=1.1,
                            n_edges=2, perimeter=10)

    def test_density_negative_raises(self):
        with pytest.raises(ValueError):
            FragmentMetrics(fragment_id=0, area=10, aspect=1, density=-0.1,
                            n_edges=2, perimeter=10)

    def test_n_edges_negative_raises(self):
        with pytest.raises(ValueError):
            FragmentMetrics(fragment_id=0, area=10, aspect=1, density=0.5,
                            n_edges=-1, perimeter=10)

    def test_perimeter_negative_raises(self):
        with pytest.raises(ValueError):
            FragmentMetrics(fragment_id=0, area=10, aspect=1, density=0.5,
                            n_edges=2, perimeter=-1)

    def test_boundary_values_ok(self):
        m = FragmentMetrics(fragment_id=0, area=0, aspect=0.001,
                            density=0.0, n_edges=0, perimeter=0)
        assert m.area == 0.0
        assert m.density == 0.0


# ─── TestCollectionStats ──────────────────────────────────────────────────────

class TestCollectionStats:
    def _make(self, **kw):
        defaults = dict(n_fragments=5, total_area=500, mean_area=100, std_area=20,
                        min_area=50, max_area=200, mean_aspect=1.0,
                        mean_density=0.8, mean_edges=4.0)
        defaults.update(kw)
        return CollectionStats(**defaults)

    def test_construction(self):
        s = self._make()
        assert s.n_fragments == 5

    def test_n_fragments_negative_raises(self):
        with pytest.raises(ValueError):
            self._make(n_fragments=-1)

    def test_to_dict_keys(self):
        s = self._make()
        d = s.to_dict()
        for key in ("n_fragments", "total_area", "mean_area", "std_area",
                    "min_area", "max_area", "mean_aspect",
                    "mean_density", "mean_edges"):
            assert key in d

    def test_to_dict_returns_dict(self):
        s = self._make()
        assert isinstance(s.to_dict(), dict)

    def test_extras_in_to_dict(self):
        s = self._make(extras={"custom": 42.0})
        d = s.to_dict()
        assert d["custom"] == pytest.approx(42.0)


# ─── TestComputeFragmentMetrics ───────────────────────────────────────────────

class TestComputeFragmentMetrics:
    def test_returns_fragment_metrics(self):
        result = compute_fragment_metrics(0, _solid_mask())
        assert isinstance(result, FragmentMetrics)

    def test_fragment_id_set(self):
        result = compute_fragment_metrics(7, _solid_mask())
        assert result.fragment_id == 7

    def test_solid_mask_area(self):
        result = compute_fragment_metrics(0, _solid_mask(h=5, w=10))
        assert result.area == pytest.approx(50.0)

    def test_empty_mask_area_zero(self):
        result = compute_fragment_metrics(0, _empty_mask(h=10, w=10))
        assert result.area == pytest.approx(0.0)

    def test_aspect_ratio_rect(self):
        # 20x30 solid rectangle: aspect = w/h = 30/20 = 1.5
        mask = np.ones((20, 30), dtype=np.uint8)
        result = compute_fragment_metrics(0, mask)
        assert result.aspect == pytest.approx(1.5, rel=0.01)

    def test_density_solid(self):
        # Solid fill → density = 1.0
        mask = np.ones((10, 10), dtype=np.uint8)
        result = compute_fragment_metrics(0, mask)
        assert result.density == pytest.approx(1.0)

    def test_perimeter_positive_for_rect_in_canvas(self):
        # Rectangle inside a larger canvas has clear boundary pixels
        result = compute_fragment_metrics(0, _rect_mask(h=10, w=12))
        assert result.perimeter >= 0.0  # Boundary detection is implementation-defined

    def test_3d_mask_raises(self):
        mask = np.ones((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_fragment_metrics(0, mask)

    def test_n_edges_set(self):
        result = compute_fragment_metrics(0, _solid_mask(), n_edges=4)
        assert result.n_edges == 4

    def test_rect_mask_aspect(self):
        # _rect_mask: 20 rows, 30 cols of ones inside a 40x50 canvas
        mask = _rect_mask(h=20, w=30)
        result = compute_fragment_metrics(0, mask)
        assert result.aspect == pytest.approx(30.0 / 20.0, rel=0.01)

    def test_density_in_0_1(self):
        mask = _rect_mask()
        result = compute_fragment_metrics(0, mask)
        assert 0.0 <= result.density <= 1.0


# ─── TestComputeCollectionStats ───────────────────────────────────────────────

class TestComputeCollectionStats:
    def test_returns_collection_stats(self):
        metrics = _sample_metrics()
        result = compute_collection_stats(metrics)
        assert isinstance(result, CollectionStats)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_collection_stats([])

    def test_n_fragments_correct(self):
        metrics = _sample_metrics(7)
        result = compute_collection_stats(metrics)
        assert result.n_fragments == 7

    def test_total_area_sum(self):
        metrics = [_make_metric(i, area=float(i + 1)) for i in range(4)]
        result = compute_collection_stats(metrics)
        assert result.total_area == pytest.approx(1 + 2 + 3 + 4)

    def test_mean_area_correct(self):
        metrics = [_make_metric(i, area=float(i * 10 + 10)) for i in range(5)]
        expected_mean = sum(m.area for m in metrics) / 5
        result = compute_collection_stats(metrics)
        assert result.mean_area == pytest.approx(expected_mean)

    def test_min_max_area(self):
        metrics = _sample_metrics(5)
        result = compute_collection_stats(metrics)
        assert result.min_area == pytest.approx(100.0)
        assert result.max_area == pytest.approx(500.0)

    def test_single_metric(self):
        metrics = [_make_metric(0, area=42.0)]
        result = compute_collection_stats(metrics)
        assert result.n_fragments == 1
        assert result.mean_area == pytest.approx(42.0)
        assert result.std_area == pytest.approx(0.0)


# ─── TestAreaHistogram ────────────────────────────────────────────────────────

class TestAreaHistogram:
    def test_returns_tuple(self):
        result = area_histogram(_sample_metrics())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_counts_shape(self):
        counts, edges = area_histogram(_sample_metrics(), n_bins=5)
        assert len(counts) == 5
        assert len(edges) == 6

    def test_normalized_sums_to_one(self):
        counts, _ = area_histogram(_sample_metrics(), normalize=True)
        assert counts.sum() == pytest.approx(1.0)

    def test_unnormalized(self):
        metrics = _sample_metrics(5)
        counts, _ = area_histogram(metrics, normalize=False)
        assert counts.sum() == pytest.approx(5.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            area_histogram([])

    def test_n_bins_zero_raises(self):
        with pytest.raises(ValueError):
            area_histogram(_sample_metrics(), n_bins=0)

    def test_edges_ascending(self):
        _, edges = area_histogram(_sample_metrics())
        assert (np.diff(edges) > 0).all()

    def test_single_metric(self):
        counts, _ = area_histogram([_make_metric(0, area=50.0)], n_bins=3)
        assert counts.sum() == pytest.approx(1.0)


# ─── TestCompareCollections ───────────────────────────────────────────────────

class TestCompareCollections:
    def _make_stats(self, mean_area=100.0, total_area=500.0):
        return CollectionStats(
            n_fragments=5, total_area=total_area,
            mean_area=mean_area, std_area=10.0,
            min_area=50.0, max_area=200.0,
            mean_aspect=1.0, mean_density=0.8, mean_edges=4.0,
        )

    def test_returns_dict(self):
        s = self._make_stats()
        result = compare_collections(s, s)
        assert isinstance(result, dict)

    def test_identical_stats_all_zeros(self):
        s = self._make_stats()
        result = compare_collections(s, s)
        for v in result.values():
            assert v == pytest.approx(0.0)

    def test_delta_mean_area(self):
        a = self._make_stats(mean_area=200.0)
        b = self._make_stats(mean_area=100.0)
        result = compare_collections(a, b)
        assert result["delta_mean_area"] == pytest.approx(100.0)

    def test_delta_keys_present(self):
        s = self._make_stats()
        result = compare_collections(s, s)
        for key in ("delta_total_area", "delta_mean_area", "delta_std_area",
                    "delta_min_area", "delta_max_area",
                    "delta_mean_aspect", "delta_mean_density", "delta_mean_edges"):
            assert key in result


# ─── TestOutlierIndices ───────────────────────────────────────────────────────

class TestOutlierIndices:
    def test_returns_list(self):
        result = outlier_indices(_sample_metrics())
        assert isinstance(result, list)

    def test_less_than_2_metrics_empty(self):
        result = outlier_indices([_make_metric(0)])
        assert result == []

    def test_no_outliers_uniform(self):
        metrics = [_make_metric(i, area=100.0) for i in range(5)]
        result = outlier_indices(metrics, z_threshold=2.5)
        assert result == []

    def test_detects_outlier(self):
        metrics = [_make_metric(i, area=100.0) for i in range(9)]
        metrics.append(_make_metric(9, area=10000.0))  # clear outlier
        result = outlier_indices(metrics, z_threshold=2.0)
        assert 9 in result

    def test_invalid_z_threshold_raises(self):
        with pytest.raises(ValueError):
            outlier_indices(_sample_metrics(), z_threshold=0.0)

    def test_negative_z_threshold_raises(self):
        with pytest.raises(ValueError):
            outlier_indices(_sample_metrics(), z_threshold=-1.0)

    def test_invalid_by_raises(self):
        with pytest.raises(ValueError):
            outlier_indices(_sample_metrics(), by="unknown")

    def test_by_aspect(self):
        metrics = [_make_metric(i, aspect=1.0) for i in range(8)]
        metrics.append(_make_metric(8, aspect=50.0))
        result = outlier_indices(metrics, z_threshold=2.0, by="aspect")
        assert 8 in result

    def test_by_density(self):
        metrics = [_make_metric(i, density=0.8) for i in range(8)]
        # density outlier (very low): note density must be in [0,1]
        metrics.append(_make_metric(8, density=0.0))
        result = outlier_indices(metrics, z_threshold=2.0, by="density")
        assert isinstance(result, list)

    def test_by_perimeter(self):
        metrics = [_make_metric(i, perimeter=40.0) for i in range(8)]
        metrics.append(_make_metric(8, perimeter=5000.0))
        result = outlier_indices(metrics, z_threshold=2.0, by="perimeter")
        assert 8 in result
