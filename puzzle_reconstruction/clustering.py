"""
Кластеризация фрагментов из разных документов.

Сценарий: набор фрагментов принадлежит K ≥ 2 разным документам,
перемешанным произвольным образом (типичная судебно-криминалистическая задача).

Алгоритм:
1. Строим матрицу фичей: для каждого фрагмента — вектор из
   - Фрактальной размерности краёв (стабильная хар-ка типа бумаги)
   - Статистики CSS (описание формы)
   - Гистограммы яркости (тип материала/чернила)
   - Гистограммы края (текстура бумаги)

2. Оцениваем число документов K через критерий силуэта + BIC по GMM.

3. Кластеризуем фрагменты (k-means / GMM / spectral).

4. Для каждого кластера строим отдельную матрицу совместимости.

Источник: Paixão, T. et al. (2025) "Multi-document shredded fragment clustering
via fractal and texture features", Forensic Science International.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

from .models import Fragment


# ─── Результаты кластеризации ─────────────────────────────────────────────

@dataclass
class ClusteringResult:
    """Результат разделения фрагментов по документам."""
    labels:          np.ndarray   # (N,) — номер кластера для каждого фрагмента
    n_clusters:      int
    silhouette:      float        # Средний коэффициент силуэта [-1, 1]
    confidence:      np.ndarray   # (N,) — вероятность принадлежности кластеру
    cluster_groups:  List[List[int]]   # [[frag_id, ...], ...] по кластерам

    def summary(self) -> str:
        lines = [
            f"=== Кластеризация фрагментов ===",
            f"  Найдено документов: {self.n_clusters}",
            f"  Силуэт:             {self.silhouette:+.3f}",
        ]
        for k, group in enumerate(self.cluster_groups):
            avg_conf = float(np.mean([self.confidence[i]
                                       for i, l in enumerate(self.labels) if l == k]))
            lines.append(f"  Кластер {k}: {len(group)} фрагментов  "
                          f"(ср. уверенность {avg_conf:.1%})")
        return "\n".join(lines)


# ─── Главная функция ──────────────────────────────────────────────────────

def cluster_fragments(fragments: List[Fragment],
                       k: Optional[int] = None,
                       k_max: int = 8,
                       method: str = "gmm",
                       seed: int = 42) -> ClusteringResult:
    """
    Разбивает фрагменты на кластеры (разные документы).

    Args:
        fragments:  Список фрагментов (должны быть заполнены edges, fractal).
        k:          Число кластеров (если None — определяется автоматически).
        k_max:      Максимальное число кластеров при автопоиске.
        method:     Алгоритм: "gmm" | "kmeans" | "spectral".
        seed:       Random seed.

    Returns:
        ClusteringResult с метками и уверенностями.

    Raises:
        ImportError: Если не установлен scikit-learn.
        ValueError:  Если список фрагментов пуст.
    """
    if not _HAS_SKLEARN:
        raise ImportError("pip install scikit-learn для кластеризации")

    if not fragments:
        raise ValueError("Список фрагментов пуст")

    if len(fragments) == 1:
        return ClusteringResult(
            labels=np.array([0]),
            n_clusters=1,
            silhouette=0.0,
            confidence=np.array([1.0]),
            cluster_groups=[[fragments[0].fragment_id]],
        )

    # ── Строим матрицу фичей ──────────────────────────────────────────────
    features = _build_feature_matrix(fragments)

    # ── Нормализация ──────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X        = scaler.fit_transform(features)

    # ── Выбор K ───────────────────────────────────────────────────────────
    if k is None:
        k = _estimate_n_clusters(X, k_max=k_max, seed=seed)

    k = max(1, min(k, len(fragments)))

    # ── Кластеризация ─────────────────────────────────────────────────────
    labels, confidence = _run_clustering(X, k=k, method=method, seed=seed)

    # ── Силуэт ────────────────────────────────────────────────────────────
    sil = 0.0
    if len(set(labels)) > 1 and len(fragments) > 2:
        try:
            sil = float(silhouette_score(X, labels))
        except Exception:
            pass

    # ── Группировка по кластерам ──────────────────────────────────────────
    cluster_groups: List[List[int]] = [[] for _ in range(k)]
    for idx, lbl in enumerate(labels):
        cluster_groups[lbl].append(fragments[idx].fragment_id)

    return ClusteringResult(
        labels=labels,
        n_clusters=k,
        silhouette=sil,
        confidence=confidence,
        cluster_groups=cluster_groups,
    )


# ─── Матрица фичей ────────────────────────────────────────────────────────

def _build_feature_matrix(fragments: List[Fragment]) -> np.ndarray:
    """
    Строит матрицу признаков: каждая строка — дескриптор одного фрагмента.

    Признаки:
        [0:2]   Фрактальные размерности (box + divider)
        [2:10]  Статистики CSS (средние нулевые пересечения по масштабам)
        [10:18] Гистограмма яркости (8 бинов)
        [18:26] Гистограмма градиента края (8 бинов)
        [26:28] Форм-фактор: площадь, периметр нормализованные
    """
    rows = []
    for frag in fragments:
        vec = _fragment_features(frag)
        rows.append(vec)
    return np.array(rows, dtype=np.float64)


def _fragment_features(frag: Fragment) -> np.ndarray:
    """Строит вектор признаков для одного фрагмента."""
    parts = []

    # 1. Фрактальные размерности
    if frag.fractal is not None:
        parts.append([frag.fractal.fd_box, frag.fractal.fd_divider])
    else:
        parts.append([1.5, 1.5])

    # 2. CSS-дескриптор (средние нулевые пересечения по масштабам)
    css_vec = np.zeros(8)
    if frag.fractal is not None and frag.fractal.css_image:
        for i, (sigma, zeros) in enumerate(frag.fractal.css_image[:8]):
            css_vec[i] = len(zeros) / max(1.0, sigma)
    parts.append(css_vec)

    # 3. Гистограмма яркости изображения (8 бинов)
    brightness_hist = np.zeros(8)
    if frag.image is not None:
        gray = frag.image.mean(axis=2)
        if frag.mask is not None:
            gray_vals = gray[frag.mask > 0]
        else:
            gray_vals = gray.ravel()
        if len(gray_vals) > 0:
            hist, _ = np.histogram(gray_vals, bins=8, range=(0, 255), density=True)
            brightness_hist = hist
    parts.append(brightness_hist)

    # 4. Гистограмма градиентов по контуру (текстура края — характерна для типа бумаги)
    edge_hist = np.zeros(8)
    if frag.contour is not None and len(frag.contour) > 2:
        diffs  = np.diff(frag.contour, axis=0)
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])  # (-π, π)
        hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi), density=True)
        edge_hist = hist
    parts.append(edge_hist)

    # 5. Форм-фактор
    if frag.tangram is not None:
        shape_feats = [frag.tangram.area, frag.tangram.scale]
    else:
        shape_feats = [0.5, 1.0]
    parts.append(shape_feats)

    # Собираем в плоский вектор
    return np.concatenate([np.ravel(p) for p in parts])


# ─── Оценка числа кластеров ───────────────────────────────────────────────

def _estimate_n_clusters(X: np.ndarray,
                          k_max: int,
                          seed: int) -> int:
    """
    Оценивает оптимальное K через:
    1. BIC по GMM (критерий Байеса)
    2. Силуэт по k-means

    Возвращает K с наименьшим BIC (при достаточном силуэте).
    """
    n = len(X)
    k_max = min(k_max, n - 1)

    if k_max < 2:
        return 1

    best_bic  = np.inf
    best_sil  = -1.0
    best_k    = 1

    for k in range(2, k_max + 1):
        try:
            gm  = GaussianMixture(n_components=k, covariance_type="full",
                                   random_state=seed, n_init=3)
            gm.fit(X)
            bic = gm.bic(X)

            km = KMeans(n_clusters=k, random_state=seed, n_init=5)
            km_labels = km.fit_predict(X)
            if len(set(km_labels)) > 1:
                sil = float(silhouette_score(X, km_labels))
            else:
                sil = 0.0

            # Комбинированный критерий: минимум BIC при силуэте > 0.15
            if bic < best_bic and sil > 0.10:
                best_bic = bic
                best_sil = sil
                best_k   = k
        except Exception:
            continue

    return best_k


# ─── Запуск выбранного алгоритма ──────────────────────────────────────────

def _run_clustering(X: np.ndarray,
                     k: int,
                     method: str,
                     seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Запускает кластеризацию, возвращает (labels, confidence).

    confidence[i] — вероятность или дистанция (нормализованная) для i-го элемента.
    """
    n = len(X)

    if method == "gmm":
        gm = GaussianMixture(n_components=k, covariance_type="full",
                              random_state=seed, n_init=5)
        labels     = gm.fit_predict(X)
        proba      = gm.predict_proba(X)
        confidence = proba[np.arange(n), labels]

    elif method == "kmeans":
        km     = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=500)
        labels = km.fit_predict(X)
        # Уверенность: обратно пропорционально расстоянию до центроида
        dists  = np.min(km.transform(X), axis=1)
        max_d  = dists.max() if dists.max() > 0 else 1.0
        confidence = 1.0 - dists / max_d

    elif method == "spectral":
        sc     = SpectralClustering(n_clusters=k, affinity="rbf",
                                     random_state=seed, n_init=10)
        labels = sc.fit_predict(X)
        # Спектральная кластеризация не даёт вероятностей — заглушка
        confidence = np.ones(n, dtype=float) * 0.7

    else:
        raise ValueError(f"Неизвестный метод кластеризации: '{method}'")

    return labels.astype(int), confidence.astype(float)


# ─── Утилита: разбить сборку по кластерам ─────────────────────────────────

def split_by_cluster(fragments: List[Fragment],
                      result: ClusteringResult) -> List[List[Fragment]]:
    """
    Разбивает список фрагментов на подсписки по кластерам.

    Returns:
        Список из k списков фрагментов (один список = один документ).
    """
    frag_by_id = {f.fragment_id: f for f in fragments}
    return [
        [frag_by_id[fid] for fid in group if fid in frag_by_id]
        for group in result.cluster_groups
    ]
