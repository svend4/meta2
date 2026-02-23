"""Extra tests for puzzle_reconstruction.utils.fragment_stats."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.fragment_stats import (
    CollectionStats,
    FragmentMetrics,
    area_histogram,
    compare_collections,
    compute_collection_stats,
    compute_fragment_metrics,
    outlier_indices,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _solid(h=20, w=20):
    return np.ones((h, w), dtype=np.uint8)


def _empty(h=20, w=20):
    return np.zeros((h, w), dtype=np.uint8)


def _metric(fid=0, area=100.0, aspect=1.0,
            density=0.8, n_edges=4, perimeter=40.0):
    return FragmentMetrics(fragment_id=fid, area=area, aspect=aspect,
                           density=density, n_edges=n_edges, perimeter=perimeter)


def _metrics(n=5):
    return [_metric(i, area=float((i + 1) * 100)) for i in range(n)]


# ─── TestFragmentMetricsExtra ─────────────────────────────────────────────────

class TestFragmentMetricsExtra:
    def test_fragment_id_stored(self):
        assert _metric(fid=3).fragment_id == 3

    def test_area_stored(self):
        assert _metric(area=250.0).area == pytest.approx(250.0)

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            FragmentMetrics(fragment_id=-1, area=10, aspect=1, density=0.5,
                            n_edges=2, perimeter=10)

    def test_negative_area_raises(self):
        with pytest.raises(ValueError):
            FragmentMetrics(fragment_id=0, area=-1, aspect=1, density=0.5,
                            n_edges=2, perimeter=10)

    def test_zero_aspect_raises(self):
        with pytest.raises(ValueError):
            FragmentMetrics(fragment_id=0, area=10, aspect=0, density=0.5,
                            n_edges=2, perimeter=10)

    def test_density_above_1_raises(self):
        with pytest.raises(ValueError):
            FragmentMetrics(fragment_id=0, area=10, aspect=1, density=1.1,
                            n_edges=2, perimeter=10)

    def test_density_negative_raises(self):
        with pytest.raises(ValueError):
            FragmentMetrics(fragment_id=0, area=10, aspect=1, density=-0.1,
                            n_edges=2, perimeter=10)

    def test_negative_n_edges_raises(self):
        with pytest.raises(ValueError):
            FragmentMetrics(fragment_id=0, area=10, aspect=1, density=0.5,
                            n_edges=-1, perimeter=10)

    def test_negative_perimeter_raises(self):
        with pytest.raises(ValueError):
            FragmentMetrics(fragment_id=0, area=10, aspect=1, density=0.5,
                            n_edges=2, perimeter=-1)

    def test_boundary_zeros_ok(self):
        m = FragmentMetrics(fragment_id=0, area=0, aspect=0.001,
                            density=0.0, n_edges=0, perimeter=0)
        assert m.area == 0.0 and m.density == 0.0


# ─── TestCollectionStatsExtra ─────────────────────────────────────────────────

class TestCollectionStatsExtra:
    def _make(self, **kw):
        defaults = dict(n_fragments=5, total_area=500, mean_area=100,
                        std_area=20, min_area=50, max_area=200,
                        mean_aspect=1.0, mean_density=0.8, mean_edges=4.0)
        defaults.update(kw)
        return CollectionStats(**defaults)

    def test_n_fragments_stored(self):
        assert self._make(n_fragments=7).n_fragments == 7

    def test_negative_n_fragments_raises(self):
        with pytest.raises(ValueError):
            self._make(n_fragments=-1)

    def test_to_dict_returns_dict(self):
        assert isinstance(self._make().to_dict(), dict)

    def test_to_dict_required_keys(self):
        d = self._make().to_dict()
        for key in ("n_fragments", "total_area", "mean_area", "std_area",
                    "min_area", "max_area", "mean_aspect",
                    "mean_density", "mean_edges"):
            assert key in d

    def test_extras_in_to_dict(self):
        s = self._make(extras={"custom": 42.0})
        assert s.to_dict()["custom"] == pytest.approx(42.0)


# ─── TestComputeFragmentMetricsExtra ─────────────────────────────────────────

class TestComputeFragmentMetricsExtra:
    def test_returns_fragment_metrics(self):
        assert isinstance(compute_fragment_metrics(0, _solid()), FragmentMetrics)

    def test_fragment_id_set(self):
        assert compute_fragment_metrics(7, _solid()).fragment_id == 7

    def test_solid_area(self):
        assert compute_fragment_metrics(0, _solid(5, 10)).area == pytest.approx(50.0)

    def test_empty_area_zero(self):
        assert compute_fragment_metrics(0, _empty(10, 10)).area == pytest.approx(0.0)

    def test_aspect_ratio_rect(self):
        m = compute_fragment_metrics(0, np.ones((20, 30), dtype=np.uint8))
        assert m.aspect == pytest.approx(1.5, rel=0.01)

    def test_density_solid_mask(self):
        m = compute_fragment_metrics(0, np.ones((10, 10), dtype=np.uint8))
        assert m.density == pytest.approx(1.0)

    def test_density_in_0_1(self):
        mask = np.zeros((40, 50), dtype=np.uint8)
        mask[5:25, 10:40] = 1
        m = compute_fragment_metrics(0, mask)
        assert 0.0 <= m.density <= 1.0

    def test_3d_mask_raises(self):
        with pytest.raises(ValueError):
            compute_fragment_metrics(0, np.ones((10, 10, 3), dtype=np.uint8))

    def test_n_edges_param(self):
        m = compute_fragment_metrics(0, _solid(), n_edges=6)
        assert m.n_edges == 6


# ─── TestComputeCollectionStatsExtra ─────────────────────────────────────────

class TestComputeCollectionStatsExtra:
    def test_returns_collection_stats(self):
        assert isinstance(compute_collection_stats(_metrics()), CollectionStats)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_collection_stats([])

    def test_n_fragments(self):
        assert compute_collection_stats(_metrics(7)).n_fragments == 7

    def test_total_area(self):
        mets = [_metric(i, area=float(i + 1)) for i in range(4)]
        result = compute_collection_stats(mets)
        assert result.total_area == pytest.approx(10.0)

    def test_mean_area(self):
        mets = [_metric(i, area=float(i * 10 + 10)) for i in range(5)]
        expected = sum(m.area for m in mets) / 5
        assert compute_collection_stats(mets).mean_area == pytest.approx(expected)

    def test_min_max_area(self):
        result = compute_collection_stats(_metrics(5))
        assert result.min_area == pytest.approx(100.0)
        assert result.max_area == pytest.approx(500.0)

    def test_single_metric_std_zero(self):
        result = compute_collection_stats([_metric(0, area=42.0)])
        assert result.std_area == pytest.approx(0.0)


# ─── TestAreaHistogramExtra ───────────────────────────────────────────────────

class TestAreaHistogramExtra:
    def test_returns_tuple_len_2(self):
        result = area_histogram(_metrics())
        assert isinstance(result, tuple) and len(result) == 2

    def test_counts_length_equals_n_bins(self):
        counts, edges = area_histogram(_metrics(), n_bins=5)
        assert len(counts) == 5 and len(edges) == 6

    def test_normalized_sums_to_1(self):
        counts, _ = area_histogram(_metrics(), normalize=True)
        assert counts.sum() == pytest.approx(1.0)

    def test_unnormalized_sum_equals_count(self):
        counts, _ = area_histogram(_metrics(5), normalize=False)
        assert counts.sum() == pytest.approx(5.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            area_histogram([])

    def test_n_bins_zero_raises(self):
        with pytest.raises(ValueError):
            area_histogram(_metrics(), n_bins=0)

    def test_edges_ascending(self):
        _, edges = area_histogram(_metrics())
        assert (np.diff(edges) > 0).all()

    def test_single_metric_no_crash(self):
        counts, _ = area_histogram([_metric(0, area=50.0)], n_bins=3)
        assert counts.sum() == pytest.approx(1.0)


# ─── TestCompareCollectionsExtra ─────────────────────────────────────────────

class TestCompareCollectionsExtra:
    def _make_stats(self, mean_area=100.0, total_area=500.0):
        return CollectionStats(
            n_fragments=5, total_area=total_area,
            mean_area=mean_area, std_area=10.0,
            min_area=50.0, max_area=200.0,
            mean_aspect=1.0, mean_density=0.8, mean_edges=4.0,
        )

    def test_returns_dict(self):
        s = self._make_stats()
        assert isinstance(compare_collections(s, s), dict)

    def test_identical_all_zeros(self):
        s = self._make_stats()
        for v in compare_collections(s, s).values():
            assert v == pytest.approx(0.0)

    def test_delta_mean_area(self):
        a = self._make_stats(mean_area=200.0)
        b = self._make_stats(mean_area=100.0)
        assert compare_collections(a, b)["delta_mean_area"] == pytest.approx(100.0)

    def test_all_delta_keys_present(self):
        s = self._make_stats()
        result = compare_collections(s, s)
        for key in ("delta_total_area", "delta_mean_area", "delta_std_area",
                    "delta_min_area", "delta_max_area",
                    "delta_mean_aspect", "delta_mean_density", "delta_mean_edges"):
            assert key in result


# ─── TestOutlierIndicesExtra ──────────────────────────────────────────────────

class TestOutlierIndicesExtra:
    def test_returns_list(self):
        assert isinstance(outlier_indices(_metrics()), list)

    def test_single_metric_empty(self):
        assert outlier_indices([_metric(0)]) == []

    def test_uniform_no_outliers(self):
        mets = [_metric(i, area=100.0) for i in range(5)]
        assert outlier_indices(mets, z_threshold=2.5) == []

    def test_detects_outlier(self):
        mets = [_metric(i, area=100.0) for i in range(9)]
        mets.append(_metric(9, area=10000.0))
        assert 9 in outlier_indices(mets, z_threshold=2.0)

    def test_z_threshold_zero_raises(self):
        with pytest.raises(ValueError):
            outlier_indices(_metrics(), z_threshold=0.0)

    def test_negative_z_threshold_raises(self):
        with pytest.raises(ValueError):
            outlier_indices(_metrics(), z_threshold=-1.0)

    def test_unknown_by_raises(self):
        with pytest.raises(ValueError):
            outlier_indices(_metrics(), by="unknown")

    def test_by_aspect(self):
        mets = [_metric(i, aspect=1.0) for i in range(8)]
        mets.append(_metric(8, aspect=50.0))
        assert 8 in outlier_indices(mets, z_threshold=2.0, by="aspect")

    def test_by_perimeter(self):
        mets = [_metric(i, perimeter=40.0) for i in range(8)]
        mets.append(_metric(8, perimeter=5000.0))
        assert 8 in outlier_indices(mets, z_threshold=2.0, by="perimeter")
