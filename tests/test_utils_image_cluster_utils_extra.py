"""Extra tests for puzzle_reconstruction/utils/image_cluster_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.image_cluster_utils import (
    ImageStatsAnalysisConfig,
    ImageStatsAnalysisEntry,
    ImageStatsAnalysisSummary,
    make_image_stats_entry,
    summarise_image_stats_entries,
    filter_by_min_sharpness,
    filter_by_max_entropy,
    filter_by_min_contrast,
    top_k_sharpest,
    best_image_stats_entry,
    image_stats_score_stats,
    compare_image_stats_summaries,
    batch_summarise_image_stats_entries,
    ClusteringAnalysisConfig,
    ClusteringAnalysisEntry,
    ClusteringAnalysisSummary,
    make_clustering_entry,
    summarise_clustering_entries,
    filter_clustering_by_min_silhouette,
    filter_clustering_by_max_inertia,
    filter_clustering_by_algorithm,
    filter_clustering_by_n_clusters,
    top_k_clustering_entries,
    best_clustering_entry,
    clustering_score_stats,
    compare_clustering_summaries,
    batch_summarise_clustering_entries,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _ise(fid=0, sharp=500.0, entropy=5.0, contrast=50.0) -> ImageStatsAnalysisEntry:
    return make_image_stats_entry(fid, sharp, entropy, contrast, 128.0, 1000)


def _ises(n=4) -> list:
    return [_ise(fid=i, sharp=float(i+1)*100) for i in range(n)]


def _ce(rid=0, nc=3, inertia=100.0, sil=0.6, algo="kmeans") -> ClusteringAnalysisEntry:
    return make_clustering_entry(rid, nc, inertia, sil, algo, 50)


def _ces(n=4) -> list:
    return [_ce(rid=i, sil=float(i+1)/n) for i in range(n)]


# ─── ImageStatsAnalysisConfig ─────────────────────────────────────────────────

class TestImageStatsAnalysisConfigExtra:
    def test_default_min_sharpness(self):
        assert ImageStatsAnalysisConfig().min_sharpness == pytest.approx(0.0)

    def test_default_max_entropy(self):
        assert ImageStatsAnalysisConfig().max_entropy == pytest.approx(8.0)

    def test_custom_values(self):
        cfg = ImageStatsAnalysisConfig(min_sharpness=100.0, min_contrast=30.0)
        assert cfg.min_sharpness == pytest.approx(100.0)


# ─── ImageStatsAnalysisEntry ──────────────────────────────────────────────────

class TestImageStatsAnalysisEntryExtra:
    def test_stores_fragment_id(self):
        assert _ise(fid=5).fragment_id == 5

    def test_stores_sharpness(self):
        assert _ise(sharp=300.0).sharpness == pytest.approx(300.0)

    def test_stores_entropy(self):
        assert _ise(entropy=4.5).entropy == pytest.approx(4.5)


# ─── summarise_image_stats_entries ────────────────────────────────────────────

class TestSummariseImageStatsEntriesExtra:
    def test_returns_summary(self):
        assert isinstance(summarise_image_stats_entries(_ises()), ImageStatsAnalysisSummary)

    def test_n_images_correct(self):
        assert summarise_image_stats_entries(_ises(3)).n_images == 3

    def test_empty_returns_defaults(self):
        s = summarise_image_stats_entries([])
        assert s.n_images == 0 and s.sharpest_id is None

    def test_sharpest_id_correct(self):
        entries = [_ise(fid=0, sharp=100.0), _ise(fid=1, sharp=800.0)]
        s = summarise_image_stats_entries(entries)
        assert s.sharpest_id == 1


# ─── filter functions ─────────────────────────────────────────────────────────

class TestFilterImageStatsExtra:
    def test_filter_by_min_sharpness(self):
        entries = [_ise(sharp=100.0), _ise(fid=1, sharp=600.0)]
        result = filter_by_min_sharpness(entries, 400.0)
        assert all(e.sharpness >= 400.0 for e in result)

    def test_filter_by_max_entropy(self):
        entries = [_ise(entropy=4.0), _ise(fid=1, entropy=7.0)]
        result = filter_by_max_entropy(entries, 5.0)
        assert all(e.entropy <= 5.0 for e in result)

    def test_filter_by_min_contrast(self):
        entries = [_ise(contrast=20.0), _ise(fid=1, contrast=80.0)]
        result = filter_by_min_contrast(entries, 50.0)
        assert all(e.contrast >= 50.0 for e in result)


# ─── top_k / best ─────────────────────────────────────────────────────────────

class TestTopKBestImageStatsExtra:
    def test_top_k_sharpest(self):
        result = top_k_sharpest(_ises(5), 3)
        assert len(result) == 3

    def test_best_entry_max_sharpness(self):
        entries = [_ise(sharp=100.0), _ise(fid=1, sharp=700.0)]
        best = best_image_stats_entry(entries)
        assert best.sharpness == pytest.approx(700.0)

    def test_best_empty_is_none(self):
        assert best_image_stats_entry([]) is None


# ─── image_stats_score_stats ──────────────────────────────────────────────────

class TestImageStatsScoreStatsExtra:
    def test_returns_dict(self):
        assert isinstance(image_stats_score_stats(_ises()), dict)

    def test_keys_present(self):
        for k in ("min", "max", "mean", "std", "count"):
            assert k in image_stats_score_stats(_ises(3))

    def test_empty_returns_zero_count(self):
        assert image_stats_score_stats([])["count"] == 0


# ─── compare / batch ──────────────────────────────────────────────────────────

class TestCompareImageStatsExtra:
    def test_returns_dict(self):
        s = summarise_image_stats_entries(_ises(3))
        assert isinstance(compare_image_stats_summaries(s, s), dict)

    def test_identical_zero_delta(self):
        s = summarise_image_stats_entries(_ises(3))
        d = compare_image_stats_summaries(s, s)
        assert d["delta_mean_sharpness"] == pytest.approx(0.0)

    def test_batch_summarise(self):
        result = batch_summarise_image_stats_entries([_ises(2), _ises(3)])
        assert len(result) == 2


# ─── ClusteringAnalysisEntry ──────────────────────────────────────────────────

class TestClusteringAnalysisEntryExtra:
    def test_stores_run_id(self):
        assert _ce(rid=5).run_id == 5

    def test_stores_n_clusters(self):
        assert _ce(nc=5).n_clusters == 5

    def test_stores_silhouette(self):
        assert _ce(sil=0.75).silhouette == pytest.approx(0.75)


# ─── summarise_clustering_entries ────────────────────────────────────────────

class TestSummariseClusteringEntriesExtra:
    def test_returns_summary(self):
        assert isinstance(summarise_clustering_entries(_ces()), ClusteringAnalysisSummary)

    def test_n_runs_correct(self):
        assert summarise_clustering_entries(_ces(3)).n_runs == 3

    def test_empty_returns_defaults(self):
        s = summarise_clustering_entries([])
        assert s.n_runs == 0 and s.best_run_id is None


# ─── clustering filter / top_k / best ────────────────────────────────────────

class TestClusteringFiltersExtra:
    def test_filter_by_min_silhouette(self):
        entries = [_ce(sil=0.3), _ce(rid=1, sil=0.8)]
        result = filter_clustering_by_min_silhouette(entries, 0.5)
        assert all(e.silhouette >= 0.5 for e in result)

    def test_filter_by_max_inertia(self):
        entries = [_ce(inertia=50.0), _ce(rid=1, inertia=200.0)]
        result = filter_clustering_by_max_inertia(entries, 100.0)
        assert all(e.inertia <= 100.0 for e in result)

    def test_filter_by_algorithm(self):
        entries = [_ce(algo="kmeans"), _ce(rid=1, algo="dbscan")]
        result = filter_clustering_by_algorithm(entries, "kmeans")
        assert all(e.algorithm == "kmeans" for e in result)

    def test_filter_by_n_clusters(self):
        entries = [_ce(nc=3), _ce(rid=1, nc=5)]
        result = filter_clustering_by_n_clusters(entries, 3)
        assert all(e.n_clusters == 3 for e in result)

    def test_top_k(self):
        result = top_k_clustering_entries(_ces(5), 3)
        assert len(result) == 3

    def test_best_entry(self):
        entries = [_ce(sil=0.2), _ce(rid=1, sil=0.9)]
        best = best_clustering_entry(entries)
        assert best.silhouette == pytest.approx(0.9)

    def test_best_empty_is_none(self):
        assert best_clustering_entry([]) is None


# ─── clustering_score_stats / compare / batch ─────────────────────────────────

class TestClusteringStatsExtra:
    def test_returns_dict(self):
        assert isinstance(clustering_score_stats(_ces()), dict)

    def test_empty_returns_zero(self):
        assert clustering_score_stats([])["count"] == 0

    def test_compare_returns_dict(self):
        s = summarise_clustering_entries(_ces(3))
        assert isinstance(compare_clustering_summaries(s, s), dict)

    def test_compare_identical_zero(self):
        s = summarise_clustering_entries(_ces(3))
        d = compare_clustering_summaries(s, s)
        assert d["delta_mean_silhouette"] == pytest.approx(0.0)

    def test_batch_summarise(self):
        result = batch_summarise_clustering_entries([_ces(2), _ces(3)])
        assert len(result) == 2
