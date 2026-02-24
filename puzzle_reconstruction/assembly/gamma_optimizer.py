"""
Гамма-распределение стохастический оптимизатор (метод 2026).

Источник: Journal of Forensic Sciences, 2026
"Stochastic optimization of shredded document reconstruction
 via Gamma distribution edge deviation modeling"

Ключевая идея:
    Отклонения краёв от идеального совпадения моделируются
    гамма-распределением: deviation ~ Gamma(k, θ)

    Правдоподобие стыка: L(i,j) = Π_t Gamma_pdf(|edge_i(t) - edge_j(1-t)|; k, θ)

    Задача: найти конфигурацию P, максимизирующую суммарное log-правдоподобие
    по всем смежным парам фрагментов.

Преимущества перед SA и GA:
    - Явная вероятностная модель погрешности разрыва
    - Коэффициенты k, θ оцениваются из данных (MLE)
    - Работает на смешанных фрагментах из нескольких документов
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.stats import gamma as gamma_dist
from scipy.optimize import minimize_scalar

from ..models import Fragment, Assembly, CompatEntry, EdgeSignature


class GammaEdgeModel:
    """
    Модель отклонений края на основе гамма-распределения.

    Параметры k (shape) и θ (scale) оцениваются методом максимального
    правдоподобия на выборке наблюдаемых отклонений.
    """

    def __init__(self, k: float = 2.0, theta: float = 0.5):
        self.k     = k       # Параметр формы (shape)
        self.theta = theta   # Параметр масштаба (scale)

    def fit(self, deviations: np.ndarray) -> "GammaEdgeModel":
        """
        Оценивает параметры гамма-распределения методом MLE.

        Args:
            deviations: Массив наблюдаемых отклонений краёв (≥ 0).
        """
        deviations = np.asarray(deviations, dtype=float)
        deviations = deviations[deviations > 1e-10]  # Убираем нули
        if len(deviations) < 5:
            return self  # Недостаточно данных — оставляем дефолты
        if np.std(deviations) < 1e-10:  # Constant values → scipy.fit fails
            self.k     = 1.0
            self.theta = float(deviations[0])
            return self

        # scipy.stats.gamma.fit возвращает (a=k, loc, scale=θ)
        k_fit, _, theta_fit = gamma_dist.fit(deviations, floc=0)
        self.k     = float(k_fit)
        self.theta = float(theta_fit)
        return self

    def log_likelihood(self, deviations: np.ndarray) -> float:
        """
        Суммарное лог-правдоподобие вектора отклонений.
        """
        deviations = np.asarray(deviations, dtype=float)
        deviations = np.clip(deviations, 1e-10, None)
        ll = gamma_dist.logpdf(deviations, a=self.k, scale=self.theta)
        return float(ll.sum())

    def pair_score(self, edge_a: np.ndarray, edge_b: np.ndarray) -> float:
        """
        Оценивает правдоподобие совпадения двух краёв
        (edge_b считается зеркальным к edge_a).

        Args:
            edge_a, edge_b: (N, 2) нормализованные параметрические кривые.

        Returns:
            score ∈ (-∞, 0] — лог-правдоподобие (ближе к 0 = лучше).
        """
        n = min(len(edge_a), len(edge_b))
        if n == 0:
            return -np.inf

        a = edge_a[:n]
        b = edge_b[::-1][:n]  # Зеркалируем

        # Расстояния точка-в-точку
        deviations = np.linalg.norm(a - b, axis=1)
        return self.log_likelihood(deviations)


def gamma_optimizer(fragments: List[Fragment],
                    entries: List[CompatEntry],
                    n_iter: int = 3000,
                    init_assembly: Optional[Assembly] = None,
                    seed: int = 42) -> Assembly:
    """
    Стохастическая оптимизация сборки с гамма-моделью отклонений.

    Алгоритм:
    1. Оцениваем параметры гамма-распределения из наблюдаемых данных (MLE).
    2. Инициализируем случайную (или переданную) конфигурацию.
    3. На каждом шаге:
       a. Случайный ход: перестановка, поворот или сдвиг.
       b. Считаем суммарное лог-правдоподобие всех стыков.
       c. Принимаем ход с вероятностью Метрополиса.
    4. Возвращаем лучшую найденную конфигурацию.

    Args:
        fragments:      Все фрагменты с заполненными edges.
        entries:        Отсортированный список CompatEntry.
        n_iter:         Число итераций.
        init_assembly:  Начальная конфигурация (если None — жадный старт).
        seed:           Random seed.

    Returns:
        Assembly с наилучшей конфигурацией.
    """
    rng = np.random.RandomState(seed)

    if not fragments:
        return Assembly(fragments=[], placements={}, compat_matrix=np.array([]))

    # ── Оценка параметров гамма-модели ──────────────────────────────────
    model = _fit_gamma_model(entries)

    # ── Начальная конфигурация ───────────────────────────────────────────
    if init_assembly is not None:
        placements = {fid: (np.asarray(pos).copy(), float(angle))
                      for fid, (pos, angle) in init_assembly.placements.items()}
    else:
        from .greedy import greedy_assembly
        init = greedy_assembly(fragments, entries)
        placements = {fid: (np.asarray(pos).copy(), float(angle))
                      for fid, (pos, angle) in init.placements.items()}

    # ── Индекс edge → fragment ───────────────────────────────────────────
    edge_to_frag: Dict[int, int] = {
        edge.edge_id: frag.fragment_id
        for frag in fragments
        for edge in frag.edges
    }

    frag_ids = [f.fragment_id for f in fragments]

    # ── Начальная оценка ─────────────────────────────────────────────────
    current_ll = _evaluate_ll(placements, entries, edge_to_frag, model)
    best_ll    = current_ll
    best_placements = {fid: (pos.copy(), angle) for fid, (pos, angle) in placements.items()}

    # Температура: адаптивная (начинаем с T пропорциональной разбросу score)
    T = max(0.5, abs(current_ll) / max(1, len(entries)))
    cooling = np.exp(np.log(0.01) / n_iter)  # T → 0.01·T за n_iter шагов

    for step in range(n_iter):
        # Случайный ход
        move = int(rng.randint(0, 3))
        new_placements = {fid: (pos.copy(), angle)
                          for fid, (pos, angle) in placements.items()}

        if move == 0 and len(frag_ids) >= 2:
            # Перестановка позиций двух фрагментов
            a, b = rng.choice(len(frag_ids), 2, replace=False)
            fid_a, fid_b = frag_ids[a], frag_ids[b]
            pos_a, ang_a = new_placements[fid_a]
            pos_b, ang_b = new_placements[fid_b]
            new_placements[fid_a] = (pos_b.copy(), ang_b)
            new_placements[fid_b] = (pos_a.copy(), ang_a)

        elif move == 1:
            # Поворот на 90°/180°/270°
            fid = frag_ids[int(rng.randint(0, len(frag_ids)))]
            pos, angle = new_placements[fid]
            delta = rng.choice([np.pi / 2, np.pi, 3 * np.pi / 2])
            new_placements[fid] = (pos, angle + delta)

        else:
            # Малый сдвиг (гауссовский)
            fid = frag_ids[int(rng.randint(0, len(frag_ids)))]
            pos, angle = new_placements[fid]
            sigma = max(5.0, 50.0 * T / (abs(current_ll) / max(1, len(entries)) + 1))
            new_pos = pos + rng.randn(2) * sigma
            new_placements[fid] = (new_pos, angle)

        # Вычисляем новое лог-правдоподобие
        new_ll = _evaluate_ll(new_placements, entries, edge_to_frag, model)
        delta_ll = new_ll - current_ll

        # Критерий Метрополиса: exp(ΔLL / T)
        accept_prob = np.exp(min(0.0, delta_ll / (T + 1e-10)))
        if rng.rand() < accept_prob:
            placements   = new_placements
            current_ll   = new_ll
            if new_ll > best_ll:
                best_ll = new_ll
                best_placements = {fid: (pos.copy(), angle)
                                   for fid, (pos, angle) in placements.items()}

        T *= cooling

    return Assembly(
        fragments=fragments,
        placements=best_placements,
        compat_matrix=np.array([]),
        total_score=float(best_ll),
    )


# ─── Вспомогательные функции ──────────────────────────────────────────────

def _fit_gamma_model(entries: List[CompatEntry]) -> GammaEdgeModel:
    """
    Оценивает параметры гамма-распределения из наблюдаемых DTW-расстояний.
    DTW-расстояния используются как прокси-мера отклонений краёв.
    """
    model = GammaEdgeModel()
    if not entries:
        return model
    deviations = np.array([e.dtw_dist for e in entries if e.dtw_dist > 0])
    if len(deviations) > 5:
        model.fit(deviations)
    return model


def _evaluate_ll(placements: Dict[int, Tuple[np.ndarray, float]],
                  entries: List[CompatEntry],
                  edge_to_frag: Dict[int, int],
                  model: GammaEdgeModel) -> float:
    """
    Суммирует лог-правдоподобие по всем смежным парам краёв.
    Близко расположенные фрагменты получают больший вес.
    """
    frag_pos: Dict[int, np.ndarray] = {
        fid: np.asarray(pos) for fid, (pos, _) in placements.items()
    }
    frag_angle: Dict[int, float] = {
        fid: angle for fid, (_, angle) in placements.items()
    }

    total_ll = 0.0
    n_counted = 0

    for entry in entries[:200]:  # Ограничиваем для производительности
        fid_i = edge_to_frag.get(entry.edge_i.edge_id)
        fid_j = edge_to_frag.get(entry.edge_j.edge_id)
        if fid_i is None or fid_j is None:
            continue
        pos_i = frag_pos.get(fid_i)
        pos_j = frag_pos.get(fid_j)
        if pos_i is None or pos_j is None:
            continue

        # Трансформируем кривые в мировые координаты
        a_i = float(frag_angle[fid_i])
        a_j = float(frag_angle[fid_j])
        curve_i = _rotate_curve(entry.edge_i.virtual_curve, a_i)
        curve_j = _rotate_curve(entry.edge_j.virtual_curve, a_j)

        # Лог-правдоподобие совпадения края
        ll = model.pair_score(curve_i, curve_j)
        if not np.isfinite(ll):
            continue

        # Взвешиваем: ближе = больше влияние
        dist = float(np.linalg.norm(pos_i - pos_j))
        weight = np.exp(-dist / 300.0)

        total_ll += ll * weight * entry.score
        n_counted += 1

    return total_ll / max(1, n_counted)


def _rotate_curve(curve: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return (R @ curve.T).T
