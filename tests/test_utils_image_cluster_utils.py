"""Tests for puzzle_reconstruction.utils.image_cluster_utils"""
import pytest
from puzzle_reconstruction.utils.image_cluster_utils import (
    ImageStatsAnalysisConfig,
    ImageStatsAnalysisEntry,
    ImageStatsAnalysisSummary,
    ClusteringAnalysisConfig,
    ClusteringAnalysisEntry,
    ClusteringAnalysisSummary,
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


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_stats_entries():
    return [
        make_image_stats_entry(0, sharpness=10.0, entropy=3.0, contrast=20.0, mean=128.0, n_pixels=1000),
        make_image_stats_entry(1, sharpness=5.0, entropy=4.5, contrast=15.0, mean=100.0, n_pixels=900),
        make_image_stats_entry(2, sharpness=8.0, entropy=2.0, contrast=25.0, mean=150.0, n_pixels=1100),
    ]


def make_cluster_entries():
    return [
        make_clustering_entry(0, n_clusters=3, inertia=100.0, silhouette=0.7, algorithm="kmeans", n_samples=50),
        make_clustering_entry(1, n_clusters=3, inertia=120.0, silhouette=0.5, algorithm="kmeans", n_samples=50),
        make_clustering_entry(2, n_clusters=4, inertia=90.0, silhouette=0.8, algorithm="dbscan", n_samples=50),
    ]


# ─── ImageStatsAnalysisConfig ─────────────────────────────────────────────────

def test_config_defaults():
    cfg = ImageStatsAnalysisConfig()
    assert cfg.min_sharpness == 0.0
    assert cfg.max_entropy == 8.0
    assert cfg.min_contrast == 0.0


# ─── make_image_stats_entry ───────────────────────────────────────────────────

def test_make_entry_types():
    e = make_image_stats_entry(5, 10.0, 3.0, 20.0, 128.0, 1000, extra="test")
    assert isinstance(e, ImageStatsAnalysisEntry)
    assert e.fragment_id == 5
    assert isinstance(e.sharpness, float)
    assert e.params == {"extra": "test"}


def test_make_entry_n_pixels_conversion():
    e = make_image_stats_entry(0, 5.0, 2.0, 3.0, 100.0, 500.5)
    assert isinstance(e.n_pixels, int)
    assert e.n_pixels == 500


# ─── summarise_image_stats_entries ────────────────────────────────────────────

def test_summarise_empty():
    s = summarise_image_stats_entries([])
    assert s.n_images == 0
    assert s.sharpest_id is None
    assert s.blurriest_id is None


def test_summarise_basic():
    entries = make_stats_entries()
    s = summarise_image_stats_entries(entries)
    assert s.n_images == 3
    assert s.sharpest_id == 0   # sharpness=10.0
    assert s.blurriest_id == 1  # sharpness=5.0
    assert abs(s.mean_sharpness - (10 + 5 + 8) / 3) < 1e-9


def test_summarise_mean_entropy():
    entries = make_stats_entries()
    s = summarise_image_stats_entries(entries)
    assert abs(s.mean_entropy - (3.0 + 4.5 + 2.0) / 3) < 1e-9


# ─── filter_by_min_sharpness ──────────────────────────────────────────────────

def test_filter_by_min_sharpness():
    entries = make_stats_entries()
    filtered = filter_by_min_sharpness(entries, 8.0)
    assert len(filtered) == 2
    assert all(e.sharpness >= 8.0 for e in filtered)


def test_filter_by_min_sharpness_all():
    entries = make_stats_entries()
    filtered = filter_by_min_sharpness(entries, 0.0)
    assert len(filtered) == 3


# ─── filter_by_max_entropy ────────────────────────────────────────────────────

def test_filter_by_max_entropy():
    entries = make_stats_entries()
    filtered = filter_by_max_entropy(entries, 3.5)
    assert len(filtered) == 2
    assert all(e.entropy <= 3.5 for e in filtered)


# ─── filter_by_min_contrast ───────────────────────────────────────────────────

def test_filter_by_min_contrast():
    entries = make_stats_entries()
    filtered = filter_by_min_contrast(entries, 20.0)
    assert len(filtered) == 2


# ─── top_k_sharpest ───────────────────────────────────────────────────────────

def test_top_k_sharpest():
    entries = make_stats_entries()
    top = top_k_sharpest(entries, 2)
    assert len(top) == 2
    assert top[0].sharpness >= top[1].sharpness


def test_top_k_sharpest_more_than_available():
    entries = make_stats_entries()
    top = top_k_sharpest(entries, 10)
    assert len(top) == 3


# ─── best_image_stats_entry ───────────────────────────────────────────────────

def test_best_image_stats_entry():
    entries = make_stats_entries()
    best = best_image_stats_entry(entries)
    assert best.fragment_id == 0  # sharpness=10.0


def test_best_image_stats_entry_empty():
    assert best_image_stats_entry([]) is None


# ─── image_stats_score_stats ──────────────────────────────────────────────────

def test_image_stats_score_stats_empty():
    d = image_stats_score_stats([])
    assert d["count"] == 0
    assert d["mean"] == 0.0


def test_image_stats_score_stats_basic():
    entries = make_stats_entries()
    d = image_stats_score_stats(entries)
    assert d["count"] == 3
    assert d["min"] == 5.0
    assert d["max"] == 10.0
    assert abs(d["mean"] - (10 + 5 + 8) / 3) < 1e-9
    assert d["std"] >= 0.0


# ─── compare_image_stats_summaries ────────────────────────────────────────────

def test_compare_image_stats_summaries():
    a = ImageStatsAnalysisSummary(n_images=3, mean_sharpness=5.0, mean_entropy=3.0,
                                   sharpest_id=0, blurriest_id=1)
    b = ImageStatsAnalysisSummary(n_images=4, mean_sharpness=7.0, mean_entropy=2.5,
                                   sharpest_id=2, blurriest_id=3)
    diff = compare_image_stats_summaries(a, b)
    assert abs(diff["delta_mean_sharpness"] - 2.0) < 1e-9
    assert diff["delta_n_images"] == 1


# ─── batch_summarise_image_stats_entries ──────────────────────────────────────

def test_batch_summarise():
    entries = make_stats_entries()
    result = batch_summarise_image_stats_entries([entries[:2], entries[2:]])
    assert len(result) == 2
    assert result[0].n_images == 2


# ─── ClusteringAnalysis functions ─────────────────────────────────────────────

def test_make_clustering_entry():
    e = make_clustering_entry(0, 3, 100.0, 0.7, "kmeans", 50, key="val")
    assert isinstance(e, ClusteringAnalysisEntry)
    assert e.params == {"key": "val"}


def test_summarise_clustering_empty():
    s = summarise_clustering_entries([])
    assert s.n_runs == 0
    assert s.best_run_id is None


def test_summarise_clustering_basic():
    entries = make_cluster_entries()
    s = summarise_clustering_entries(entries)
    assert s.n_runs == 3
    assert s.best_run_id == 2  # silhouette=0.8
    assert s.worst_run_id == 1  # silhouette=0.5


def test_filter_clustering_by_min_silhouette():
    entries = make_cluster_entries()
    filtered = filter_clustering_by_min_silhouette(entries, 0.6)
    assert len(filtered) == 2


def test_filter_clustering_by_max_inertia():
    entries = make_cluster_entries()
    filtered = filter_clustering_by_max_inertia(entries, 100.0)
    assert len(filtered) == 2


def test_filter_clustering_by_algorithm():
    entries = make_cluster_entries()
    filtered = filter_clustering_by_algorithm(entries, "kmeans")
    assert len(filtered) == 2


def test_filter_clustering_by_n_clusters():
    entries = make_cluster_entries()
    filtered = filter_clustering_by_n_clusters(entries, 3)
    assert len(filtered) == 2


def test_top_k_clustering_entries():
    entries = make_cluster_entries()
    top = top_k_clustering_entries(entries, 2)
    assert len(top) == 2
    assert top[0].silhouette >= top[1].silhouette


def test_best_clustering_entry():
    entries = make_cluster_entries()
    best = best_clustering_entry(entries)
    assert best.run_id == 2


def test_best_clustering_entry_empty():
    assert best_clustering_entry([]) is None


def test_clustering_score_stats():
    entries = make_cluster_entries()
    d = clustering_score_stats(entries)
    assert d["count"] == 3
    assert d["min"] == pytest.approx(0.5, abs=1e-9)
    assert d["max"] == pytest.approx(0.8, abs=1e-9)


def test_compare_clustering_summaries():
    a = ClusteringAnalysisSummary(n_runs=2, mean_inertia=100.0, mean_silhouette=0.6,
                                   best_run_id=0, worst_run_id=1)
    b = ClusteringAnalysisSummary(n_runs=3, mean_inertia=90.0, mean_silhouette=0.7,
                                   best_run_id=2, worst_run_id=0)
    diff = compare_clustering_summaries(a, b)
    assert abs(diff["delta_mean_inertia"] - (-10.0)) < 1e-9
    assert diff["same_best"] is False


def test_batch_summarise_clustering():
    entries = make_cluster_entries()
    result = batch_summarise_clustering_entries([entries[:2], entries[2:]])
    assert len(result) == 2
