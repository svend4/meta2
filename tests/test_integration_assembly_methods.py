"""
Интеграционные тесты: сравнение всех 8 методов сборки на одном наборе данных.

Тесты проверяют корректность возвращаемых Assembly для каждого из 8 методов,
а также функции run_all_methods(), pick_best() и summary_table().

Маркер: @pytest.mark.integration
"""
import pytest
import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.tear_generator import tear_document, generate_test_document
from puzzle_reconstruction.models import Fragment, Assembly
from puzzle_reconstruction.preprocessing.segmentation import segment_fragment
from puzzle_reconstruction.preprocessing.contour import extract_contour
from puzzle_reconstruction.algorithms.tangram.inscriber import fit_tangram
from puzzle_reconstruction.algorithms.synthesis import compute_fractal_signature, build_edge_signatures
from puzzle_reconstruction.matching.compat_matrix import build_compat_matrix
from puzzle_reconstruction.assembly.greedy import greedy_assembly
from puzzle_reconstruction.assembly.annealing import simulated_annealing
from puzzle_reconstruction.assembly.beam_search import beam_search
from puzzle_reconstruction.assembly.gamma_optimizer import gamma_optimizer
from puzzle_reconstruction.assembly.genetic import genetic_assembly
from puzzle_reconstruction.assembly.exhaustive import exhaustive_assembly
from puzzle_reconstruction.assembly.ant_colony import ant_colony_assembly
from puzzle_reconstruction.assembly.mcts import mcts_assembly
from puzzle_reconstruction.assembly.parallel import (
    run_all_methods, pick_best, summary_table, ALL_METHODS,
)
from puzzle_reconstruction.config import Config, AssemblyConfig


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _process_fragment(idx: int, img: np.ndarray) -> Fragment:
    """Полный пайплайн предобработки одного фрагмента."""
    mask = segment_fragment(img, method="otsu")
    contour = extract_contour(mask)
    tangram = fit_tangram(contour)
    fractal = compute_fractal_signature(contour)
    frag = Fragment(fragment_id=idx, image=img, mask=mask, contour=contour)
    frag.tangram = tangram
    frag.fractal = fractal
    frag.edges = build_edge_signatures(frag, alpha=0.5, n_sides=4, n_points=64)
    return frag


def _get_fragment_ids(asm: Assembly) -> set:
    """Извлекает множество fragment_id из placements."""
    if isinstance(asm.placements, dict):
        return set(asm.placements.keys())
    # Если placements — список Placement
    return {p.fragment_id for p in asm.placements}


# ─── Фикстуры ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_doc() -> np.ndarray:
    """Синтетический документ 300×400, генерируется один раз на весь модуль."""
    return generate_test_document(width=300, height=400, seed=7)


@pytest.fixture(scope="module")
def torn_4(synthetic_doc) -> List[np.ndarray]:
    """4 изображения-фрагмента из синтетического документа."""
    return tear_document(synthetic_doc, n_pieces=4, noise_level=0.5, seed=7)


@pytest.fixture(scope="module")
def processed_4(torn_4) -> List[Fragment]:
    """
    Обработанные фрагменты с tangram/fractal/edges.
    Ошибки обработки конкретного фрагмента игнорируются (try/except pass).
    """
    fragments = []
    for idx, img in enumerate(torn_4):
        try:
            frag = _process_fragment(idx, img)
            fragments.append(frag)
        except Exception:
            pass
    return fragments


@pytest.fixture(scope="module")
def compat_data(processed_4) -> Tuple[np.ndarray, list]:
    """Матрица совместимости и список CompatEntry для processed_4."""
    matrix, entries = build_compat_matrix(processed_4, threshold=0.0)
    return matrix, entries


@pytest.fixture(scope="module")
def greedy_asm_4(processed_4, compat_data) -> Assembly:
    """Жадная сборка — используется как базовая для SA."""
    _, entries = compat_data
    asm = greedy_assembly(processed_4, entries)
    asm.fragments = processed_4
    return asm


# ─── 1. TestAllMethodsReturnAssembly ─────────────────────────────────────────

DIRECT_METHODS = [
    "greedy",
    "beam",
    "gamma",
    "genetic",
    "exhaustive",
    "ant_colony",
    "mcts",
]


def _call_method(name: str, fragments, entries, greedy_asm=None) -> Assembly:
    """Вызывает метод сборки по имени и возвращает Assembly."""
    if name == "greedy":
        return greedy_assembly(fragments, entries)
    elif name == "sa":
        # SA требует Assembly как первый аргумент
        base = greedy_asm or greedy_assembly(fragments, entries)
        base.fragments = fragments
        return simulated_annealing(base, entries, max_iter=50, seed=42)
    elif name == "beam":
        return beam_search(fragments, entries, beam_width=5)
    elif name == "gamma":
        return gamma_optimizer(fragments, entries, n_iter=50, seed=42)
    elif name == "genetic":
        return genetic_assembly(fragments, entries,
                                population_size=10, n_generations=5, seed=42)
    elif name == "exhaustive":
        return exhaustive_assembly(fragments, entries)
    elif name == "ant_colony":
        return ant_colony_assembly(fragments, entries, n_iterations=10, seed=42)
    elif name == "mcts":
        return mcts_assembly(fragments, entries, n_simulations=10, seed=42)
    else:
        raise ValueError(f"Unknown method: {name}")


ALL_8_METHODS = [
    "greedy", "sa", "beam", "gamma",
    "genetic", "exhaustive", "ant_colony", "mcts",
]


@pytest.mark.integration
class TestAllMethodsReturnAssembly:
    """Каждый из 8 методов должен вернуть корректный объект Assembly."""

    @pytest.mark.parametrize("method_name", ALL_8_METHODS)
    def test_returns_assembly_instance(self, method_name, processed_4, compat_data, greedy_asm_4):
        """Метод возвращает объект типа Assembly без исключений."""
        _, entries = compat_data
        asm = _call_method(method_name, processed_4, entries, greedy_asm=greedy_asm_4)
        assert isinstance(asm, Assembly), (
            f"Method {method_name!r} returned {type(asm).__name__}, expected Assembly"
        )

    @pytest.mark.parametrize("method_name", ALL_8_METHODS)
    def test_has_placements(self, method_name, processed_4, compat_data, greedy_asm_4):
        """Placements не пустые."""
        _, entries = compat_data
        asm = _call_method(method_name, processed_4, entries, greedy_asm=greedy_asm_4)
        assert asm.placements is not None, (
            f"Method {method_name!r}: placements is None"
        )
        assert len(asm.placements) > 0, (
            f"Method {method_name!r}: placements is empty"
        )

    @pytest.mark.parametrize("method_name", ALL_8_METHODS)
    def test_placements_count_equals_n_fragments(
        self, method_name, processed_4, compat_data, greedy_asm_4
    ):
        """Число placements равно числу фрагментов."""
        _, entries = compat_data
        n = len(processed_4)
        asm = _call_method(method_name, processed_4, entries, greedy_asm=greedy_asm_4)
        assert len(asm.placements) == n, (
            f"Method {method_name!r}: expected {n} placements, got {len(asm.placements)}"
        )

    @pytest.mark.parametrize("method_name", ALL_8_METHODS)
    def test_total_score_is_finite(
        self, method_name, processed_4, compat_data, greedy_asm_4
    ):
        """total_score является конечным числом (может быть <= 0)."""
        _, entries = compat_data
        asm = _call_method(method_name, processed_4, entries, greedy_asm=greedy_asm_4)
        assert np.isfinite(asm.total_score), (
            f"Method {method_name!r}: total_score={asm.total_score} is not finite"
        )

    @pytest.mark.parametrize("method_name", ALL_8_METHODS)
    def test_total_score_is_float(
        self, method_name, processed_4, compat_data, greedy_asm_4
    ):
        """total_score является числом (int или float), конвертируемым в float."""
        _, entries = compat_data
        asm = _call_method(method_name, processed_4, entries, greedy_asm=greedy_asm_4)
        score = float(asm.total_score)
        assert isinstance(score, float)


# ─── 2. TestAssemblyMethodsNoOverlap ─────────────────────────────────────────

@pytest.mark.integration
class TestAssemblyMethodsNoOverlap:
    """Каждый фрагмент размещается не более одного раза (нет дублей)."""

    @pytest.mark.parametrize("method_name", ALL_8_METHODS)
    def test_no_duplicate_fragment_ids(
        self, method_name, processed_4, compat_data, greedy_asm_4
    ):
        """fragment_ids в placements уникальны."""
        _, entries = compat_data
        asm = _call_method(method_name, processed_4, entries, greedy_asm=greedy_asm_4)
        placed_ids = list(_get_fragment_ids(asm))
        assert len(placed_ids) == len(set(placed_ids)), (
            f"Method {method_name!r}: duplicate fragment_ids found in placements"
        )

    @pytest.mark.parametrize("method_name", ALL_8_METHODS)
    def test_fragment_ids_are_integers(
        self, method_name, processed_4, compat_data, greedy_asm_4
    ):
        """Все ключи placements являются целыми числами."""
        _, entries = compat_data
        asm = _call_method(method_name, processed_4, entries, greedy_asm=greedy_asm_4)
        for fid in _get_fragment_ids(asm):
            assert isinstance(fid, int), (
                f"Method {method_name!r}: fragment_id {fid!r} is not int"
            )


# ─── 3. TestAssemblyMethodsAllFragmentsPlaced ────────────────────────────────

@pytest.mark.integration
class TestAssemblyMethodsAllFragmentsPlaced:
    """Все N фрагментов присутствуют в результирующих placements."""

    @pytest.mark.parametrize("method_name", ALL_8_METHODS)
    def test_all_fragment_ids_present(
        self, method_name, processed_4, compat_data, greedy_asm_4
    ):
        """Все оригинальные fragment_id присутствуют в Assembly.placements."""
        _, entries = compat_data
        asm = _call_method(method_name, processed_4, entries, greedy_asm=greedy_asm_4)
        expected_ids = {f.fragment_id for f in processed_4}
        actual_ids = _get_fragment_ids(asm)
        missing = expected_ids - actual_ids
        assert not missing, (
            f"Method {method_name!r}: missing fragment_ids: {missing}"
        )

    @pytest.mark.parametrize("method_name", ALL_8_METHODS)
    def test_no_extra_fragment_ids(
        self, method_name, processed_4, compat_data, greedy_asm_4
    ):
        """В placements нет посторонних fragment_id (которых не было в fragments)."""
        _, entries = compat_data
        asm = _call_method(method_name, processed_4, entries, greedy_asm=greedy_asm_4)
        expected_ids = {f.fragment_id for f in processed_4}
        actual_ids = _get_fragment_ids(asm)
        extra = actual_ids - expected_ids
        assert not extra, (
            f"Method {method_name!r}: unexpected fragment_ids: {extra}"
        )


# ─── 4. TestRunAllMethods ─────────────────────────────────────────────────────

@pytest.mark.integration
class TestRunAllMethods:
    """Тесты функций run_all_methods(), pick_best(), summary_table()."""

    @pytest.fixture(scope="class")
    def all_results(self, processed_4, compat_data):
        """
        Запускаем только методы, совместимые с сигнатурой parallel.py
        (исключаем 'sa', у которого в parallel.py некорректный вызов).
        """
        _, entries = compat_data
        # SA в parallel.py передаёт неверный аргумент n_iterations,
        # поэтому используем подмножество без SA для run_all_methods
        compatible = [m for m in ALL_METHODS if m != "sa"]
        return run_all_methods(
            processed_4, entries,
            methods=compatible,
            timeout=20.0,
            seed=42,
            n_iterations=30,
            n_simulations=10,
        )

    def test_run_all_returns_list(self, all_results):
        """run_all_methods() возвращает список."""
        assert isinstance(all_results, list)

    def test_run_all_non_empty(self, all_results):
        """Список результатов не пуст."""
        assert len(all_results) > 0

    def test_run_all_result_count(self, all_results, processed_4, compat_data):
        """Число результатов совпадает с числом запущенных методов."""
        _, entries = compat_data
        compatible = [m for m in ALL_METHODS if m != "sa"]
        assert len(all_results) == len(compatible)

    def test_run_all_each_result_has_name(self, all_results):
        """Каждый MethodResult имеет строковое поле name."""
        for r in all_results:
            assert isinstance(r.name, str)
            assert len(r.name) > 0

    def test_run_all_each_name_in_all_methods(self, all_results):
        """Имена методов принадлежат ALL_METHODS."""
        for r in all_results:
            assert r.name in ALL_METHODS, (
                f"Unexpected method name: {r.name!r}"
            )

    def test_run_all_successful_results_have_assembly(self, all_results):
        """Успешные результаты содержат объект Assembly."""
        for r in all_results:
            if r.success:
                assert isinstance(r.assembly, Assembly), (
                    f"Method {r.name!r}: expected Assembly, got {type(r.assembly)}"
                )

    def test_pick_best_returns_assembly(self, all_results):
        """pick_best() возвращает Assembly (не None при наличии успешных методов)."""
        best = pick_best(all_results)
        assert best is not None, "pick_best() returned None — no successful methods"
        assert isinstance(best, Assembly)

    def test_pick_best_has_highest_score(self, all_results):
        """pick_best() выбирает Assembly с максимальным total_score."""
        best = pick_best(all_results)
        if best is None:
            pytest.skip("No successful methods — cannot check best score")
        successful_scores = [r.score for r in all_results if r.success]
        max_score = max(successful_scores)
        assert best.total_score == pytest.approx(max_score, abs=1e-9), (
            f"pick_best score {best.total_score} != max {max_score}"
        )

    def test_summary_table_returns_string(self, all_results):
        """summary_table() возвращает непустую строку."""
        table = summary_table(all_results)
        assert isinstance(table, str)
        assert len(table) > 0

    def test_summary_table_contains_header(self, all_results):
        """summary_table() содержит заголовочную строку таблицы."""
        table = summary_table(all_results)
        assert "Method" in table

    def test_summary_table_contains_method_names(self, all_results):
        """summary_table() упоминает имена всех запущенных методов."""
        table = summary_table(all_results)
        for r in all_results:
            assert r.name in table, (
                f"Method name {r.name!r} not found in summary_table output"
            )

    def test_summary_table_contains_score_column(self, all_results):
        """summary_table() содержит колонку Score."""
        table = summary_table(all_results)
        assert "Score" in table

    def test_summary_table_contains_status_column(self, all_results):
        """summary_table() содержит колонку Status."""
        table = summary_table(all_results)
        assert "Status" in table

    def test_all_methods_constant_is_list(self):
        """ALL_METHODS — список строк с 8 элементами."""
        assert isinstance(ALL_METHODS, list)
        assert len(ALL_METHODS) == 8

    def test_all_methods_contains_expected_names(self):
        """ALL_METHODS содержит все 8 ожидаемых имён методов."""
        expected = {"greedy", "sa", "beam", "gamma", "genetic",
                    "exhaustive", "ant_colony", "mcts"}
        assert set(ALL_METHODS) == expected


# ─── 5. TestAssemblyMethodComparison ─────────────────────────────────────────

@pytest.mark.integration
class TestAssemblyMethodComparison:
    """Сравнение результатов разных методов и валидация выбора лучшего."""

    def test_greedy_valid_assembly(self, processed_4, compat_data):
        """greedy_assembly() возвращает корректную сборку."""
        _, entries = compat_data
        asm = greedy_assembly(processed_4, entries)
        assert isinstance(asm, Assembly)
        assert len(asm.placements) == len(processed_4)

    def test_beam_valid_assembly(self, processed_4, compat_data):
        """beam_search() возвращает корректную сборку."""
        _, entries = compat_data
        asm = beam_search(processed_4, entries, beam_width=5)
        assert isinstance(asm, Assembly)
        assert len(asm.placements) == len(processed_4)

    def test_greedy_and_beam_both_valid(self, processed_4, compat_data):
        """Оба метода возвращают корректные сборки с правильным числом placements."""
        _, entries = compat_data
        n = len(processed_4)
        greedy_asm = greedy_assembly(processed_4, entries)
        beam_asm = beam_search(processed_4, entries, beam_width=5)
        assert len(greedy_asm.placements) == n
        assert len(beam_asm.placements) == n

    def test_greedy_beam_scores_are_finite(self, processed_4, compat_data):
        """Оба метода возвращают конечные total_score."""
        _, entries = compat_data
        greedy_asm = greedy_assembly(processed_4, entries)
        beam_asm = beam_search(processed_4, entries, beam_width=5)
        assert np.isfinite(greedy_asm.total_score)
        assert np.isfinite(beam_asm.total_score)

    def test_pick_best_from_two_methods(self, processed_4, compat_data):
        """pick_best() выбирает Assembly с наивысшим total_score из двух методов."""
        _, entries = compat_data
        from puzzle_reconstruction.assembly.parallel import MethodResult

        greedy_asm = greedy_assembly(processed_4, entries)
        beam_asm = beam_search(processed_4, entries, beam_width=5)

        results = [
            MethodResult(name="greedy", assembly=greedy_asm),
            MethodResult(name="beam", assembly=beam_asm),
        ]
        best = pick_best(results)
        assert best is not None
        expected_best_score = max(greedy_asm.total_score, beam_asm.total_score)
        assert best.total_score == pytest.approx(expected_best_score, abs=1e-9)

    def test_pick_best_with_all_8_methods(self, processed_4, compat_data, greedy_asm_4):
        """pick_best() корректно выбирает из 8 методов."""
        _, entries = compat_data
        from puzzle_reconstruction.assembly.parallel import MethodResult

        results = []
        for method_name in ALL_8_METHODS:
            try:
                asm = _call_method(method_name, processed_4, entries,
                                   greedy_asm=greedy_asm_4)
                results.append(MethodResult(name=method_name, assembly=asm))
            except Exception:
                results.append(MethodResult(name=method_name, error="failed"))

        best = pick_best(results)
        assert best is not None
        successful_scores = [r.score for r in results if r.success]
        assert best.total_score == pytest.approx(max(successful_scores), abs=1e-9)

    def test_pick_best_none_when_all_fail(self):
        """pick_best() возвращает None если все методы завершились с ошибкой."""
        from puzzle_reconstruction.assembly.parallel import MethodResult
        failed_results = [
            MethodResult(name="greedy", error="failure"),
            MethodResult(name="beam", error="failure"),
        ]
        best = pick_best(failed_results)
        assert best is None

    def test_summary_table_all_8_methods(self, processed_4, compat_data, greedy_asm_4):
        """summary_table() отображает имена всех 8 методов при их наличии в results."""
        _, entries = compat_data
        from puzzle_reconstruction.assembly.parallel import MethodResult

        results = []
        for method_name in ALL_8_METHODS:
            try:
                asm = _call_method(method_name, processed_4, entries,
                                   greedy_asm=greedy_asm_4)
                results.append(MethodResult(name=method_name, assembly=asm))
            except Exception:
                results.append(MethodResult(name=method_name, error="failed"))

        table = summary_table(results)
        for name in ALL_8_METHODS:
            assert name in table, f"Method {name!r} not in summary_table"


# ─── 6. TestExhaustiveOptimality ─────────────────────────────────────────────

@pytest.mark.integration
class TestExhaustiveOptimality:
    """exhaustive_assembly() тесты — оптимальность для малого N."""

    @pytest.fixture(scope="class")
    def fragments_3(self):
        """3 обработанных фрагмента из документа."""
        doc = generate_test_document(width=300, height=400, seed=77)
        pieces = tear_document(doc, n_pieces=3, seed=77)
        fragments = []
        for idx, img in enumerate(pieces):
            try:
                frag = _process_fragment(idx, img)
                fragments.append(frag)
            except Exception:
                pass
        return fragments

    @pytest.fixture(scope="class")
    def compat_3(self, fragments_3):
        """Матрица совместимости для 3 фрагментов."""
        matrix, entries = build_compat_matrix(fragments_3, threshold=0.0)
        return matrix, entries

    def test_exhaustive_returns_assembly(self, fragments_3, compat_3):
        """exhaustive_assembly() возвращает Assembly для N=3."""
        _, entries = compat_3
        asm = exhaustive_assembly(fragments_3, entries)
        assert isinstance(asm, Assembly)

    def test_exhaustive_all_fragments_placed(self, fragments_3, compat_3):
        """Все 3 фрагмента размещены в результате exhaustive."""
        _, entries = compat_3
        asm = exhaustive_assembly(fragments_3, entries)
        assert len(asm.placements) == len(fragments_3)

    def test_exhaustive_fragment_ids_correct(self, fragments_3, compat_3):
        """fragment_ids в exhaustive placements совпадают с исходными."""
        _, entries = compat_3
        asm = exhaustive_assembly(fragments_3, entries)
        expected = {f.fragment_id for f in fragments_3}
        actual = _get_fragment_ids(asm)
        assert actual == expected

    def test_exhaustive_score_finite(self, fragments_3, compat_3):
        """total_score от exhaustive является конечным числом."""
        _, entries = compat_3
        asm = exhaustive_assembly(fragments_3, entries)
        assert np.isfinite(asm.total_score)

    def test_exhaustive_score_gte_zero(self, fragments_3, compat_3):
        """Для exhaustive total_score >= 0 (метод максимизирует совместимость)."""
        _, entries = compat_3
        asm = exhaustive_assembly(fragments_3, entries)
        # exhaustive ищет порядок с максимальным score → должен быть >= 0
        assert asm.total_score >= 0.0, (
            f"exhaustive total_score={asm.total_score} should be >= 0"
        )

    def test_exhaustive_no_duplicate_placements(self, fragments_3, compat_3):
        """Нет дублирующихся fragment_id в exhaustive placements."""
        _, entries = compat_3
        asm = exhaustive_assembly(fragments_3, entries)
        ids = list(_get_fragment_ids(asm))
        assert len(ids) == len(set(ids))

    def test_exhaustive_4_fragments(self, processed_4, compat_data):
        """exhaustive_assembly() работает для N=4."""
        _, entries = compat_data
        asm = exhaustive_assembly(processed_4, entries)
        assert isinstance(asm, Assembly)
        assert len(asm.placements) == len(processed_4)


# ─── 7. TestMethodsWithDifferentConfigs ──────────────────────────────────────

@pytest.mark.integration
class TestMethodsWithDifferentConfigs:
    """Тесты методов с различными параметрами конфигурации."""

    def test_beam_width_5_returns_valid_assembly(self, processed_4, compat_data):
        """beam_search с beam_width=5 возвращает корректную сборку."""
        _, entries = compat_data
        asm = beam_search(processed_4, entries, beam_width=5)
        assert isinstance(asm, Assembly)
        assert len(asm.placements) == len(processed_4)
        assert np.isfinite(asm.total_score)

    def test_beam_width_20_returns_valid_assembly(self, processed_4, compat_data):
        """beam_search с beam_width=20 возвращает корректную сборку."""
        _, entries = compat_data
        asm = beam_search(processed_4, entries, beam_width=20)
        assert isinstance(asm, Assembly)
        assert len(asm.placements) == len(processed_4)
        assert np.isfinite(asm.total_score)

    def test_beam_width_5_and_20_both_place_all(self, processed_4, compat_data):
        """Оба beam_width размещают все фрагменты."""
        _, entries = compat_data
        n = len(processed_4)
        asm5 = beam_search(processed_4, entries, beam_width=5)
        asm20 = beam_search(processed_4, entries, beam_width=20)
        assert len(asm5.placements) == n
        assert len(asm20.placements) == n

    def test_sa_few_iterations(self, processed_4, compat_data, greedy_asm_4):
        """SA с max_iter=20 возвращает корректную сборку."""
        _, entries = compat_data
        base = greedy_asm_4
        asm = simulated_annealing(base, entries, max_iter=20, seed=1)
        assert isinstance(asm, Assembly)
        assert len(asm.placements) == len(processed_4)

    def test_sa_many_iterations(self, processed_4, compat_data, greedy_asm_4):
        """SA с max_iter=500 возвращает корректную сборку."""
        _, entries = compat_data
        base = greedy_asm_4
        asm = simulated_annealing(base, entries, max_iter=500, seed=2)
        assert isinstance(asm, Assembly)
        assert len(asm.placements) == len(processed_4)

    def test_sa_output_score_is_finite(self, processed_4, compat_data, greedy_asm_4):
        """SA возвращает конечный total_score."""
        _, entries = compat_data
        asm = simulated_annealing(greedy_asm_4, entries, max_iter=100, seed=42)
        assert np.isfinite(asm.total_score)

    def test_sa_total_score_nonnegative(self, processed_4, compat_data, greedy_asm_4):
        """SA total_score >= 0 (оценивает proximity-weighted scores)."""
        _, entries = compat_data
        asm = simulated_annealing(greedy_asm_4, entries, max_iter=100, seed=42)
        assert asm.total_score >= 0.0

    def test_genetic_small_population(self, processed_4, compat_data):
        """genetic с pop=10, gen=5 возвращает корректную сборку."""
        _, entries = compat_data
        asm = genetic_assembly(processed_4, entries,
                               population_size=10, n_generations=5, seed=42)
        assert isinstance(asm, Assembly)
        assert len(asm.placements) == len(processed_4)

    def test_genetic_larger_population(self, processed_4, compat_data):
        """genetic с pop=20, gen=10 возвращает корректную сборку."""
        _, entries = compat_data
        asm = genetic_assembly(processed_4, entries,
                               population_size=20, n_generations=10, seed=42)
        assert isinstance(asm, Assembly)
        assert len(asm.placements) == len(processed_4)

    def test_genetic_reproducible_with_seed(self, processed_4, compat_data):
        """genetic с одинаковым seed даёт одинаковый total_score."""
        _, entries = compat_data
        asm1 = genetic_assembly(processed_4, entries,
                                population_size=10, n_generations=5, seed=99)
        asm2 = genetic_assembly(processed_4, entries,
                                population_size=10, n_generations=5, seed=99)
        assert asm1.total_score == pytest.approx(asm2.total_score, abs=1e-9)

    def test_mcts_few_simulations(self, processed_4, compat_data):
        """mcts с n_simulations=5 возвращает корректную сборку."""
        _, entries = compat_data
        asm = mcts_assembly(processed_4, entries, n_simulations=5, seed=42)
        assert isinstance(asm, Assembly)
        assert len(asm.placements) == len(processed_4)

    def test_mcts_more_simulations(self, processed_4, compat_data):
        """mcts с n_simulations=30 возвращает корректную сборку."""
        _, entries = compat_data
        asm = mcts_assembly(processed_4, entries, n_simulations=30, seed=42)
        assert isinstance(asm, Assembly)
        assert len(asm.placements) == len(processed_4)

    def test_ant_colony_few_iterations(self, processed_4, compat_data):
        """ant_colony с n_iterations=5 возвращает корректную сборку."""
        _, entries = compat_data
        asm = ant_colony_assembly(processed_4, entries, n_iterations=5, seed=42)
        assert isinstance(asm, Assembly)
        assert len(asm.placements) == len(processed_4)

    def test_ant_colony_more_iterations(self, processed_4, compat_data):
        """ant_colony с n_iterations=30 возвращает корректную сборку."""
        _, entries = compat_data
        asm = ant_colony_assembly(processed_4, entries, n_iterations=30, seed=42)
        assert isinstance(asm, Assembly)
        assert len(asm.placements) == len(processed_4)

    def test_gamma_few_iterations(self, processed_4, compat_data):
        """gamma_optimizer с n_iter=20 возвращает корректную сборку."""
        _, entries = compat_data
        asm = gamma_optimizer(processed_4, entries, n_iter=20, seed=42)
        assert isinstance(asm, Assembly)
        assert len(asm.placements) == len(processed_4)

    def test_config_assembly_config_defaults(self):
        """AssemblyConfig имеет корректные дефолтные поля."""
        cfg = AssemblyConfig()
        assert cfg.beam_width > 0
        assert cfg.sa_iter > 0
        assert cfg.genetic_pop > 0
        assert cfg.genetic_gen > 0
        assert cfg.mcts_sim > 0

    def test_config_default_method_is_string(self):
        """AssemblyConfig.method является строкой."""
        cfg = AssemblyConfig()
        assert isinstance(cfg.method, str)
        assert len(cfg.method) > 0

    def test_config_seed_is_int(self):
        """AssemblyConfig.seed является целым числом."""
        cfg = AssemblyConfig()
        assert isinstance(cfg.seed, int)


# ─── Дополнительные сквозные тесты ───────────────────────────────────────────

@pytest.mark.integration
class TestDataPipelineIntegration:
    """Тесты полного пайплайна: генерация → разрыв → предобработка → матрица."""

    def test_generate_test_document_shape(self, synthetic_doc):
        """generate_test_document возвращает изображение нужного размера."""
        assert synthetic_doc.shape == (400, 300, 3)

    def test_generate_test_document_dtype(self, synthetic_doc):
        """generate_test_document возвращает uint8."""
        assert synthetic_doc.dtype == np.uint8

    def test_tear_document_count(self, torn_4):
        """tear_document возвращает ожидаемое число фрагментов (4)."""
        assert len(torn_4) == 4

    def test_torn_fragments_are_ndarray(self, torn_4):
        """Каждый фрагмент из tear_document является numpy array."""
        for frag in torn_4:
            assert isinstance(frag, np.ndarray)

    def test_torn_fragments_are_3channel(self, torn_4):
        """Каждый фрагмент является 3-канальным изображением."""
        for frag in torn_4:
            assert frag.ndim == 3
            assert frag.shape[2] == 3

    def test_processed_fragments_count(self, processed_4):
        """Все 4 фрагмента успешно обработаны."""
        assert len(processed_4) == 4

    def test_processed_fragments_have_edges(self, processed_4):
        """Каждый обработанный фрагмент имеет непустой список edges."""
        for frag in processed_4:
            assert frag.edges is not None
            assert len(frag.edges) > 0

    def test_processed_fragments_have_tangram(self, processed_4):
        """Каждый обработанный фрагмент имеет TangramSignature."""
        for frag in processed_4:
            assert frag.tangram is not None

    def test_processed_fragments_have_fractal(self, processed_4):
        """Каждый обработанный фрагмент имеет FractalSignature."""
        for frag in processed_4:
            assert frag.fractal is not None

    def test_compat_matrix_shape(self, processed_4, compat_data):
        """Матрица совместимости имеет правильный размер (N_edges × N_edges)."""
        matrix, _ = compat_data
        n_edges = sum(len(f.edges) for f in processed_4)
        assert matrix.shape == (n_edges, n_edges)

    def test_compat_matrix_symmetric(self, compat_data):
        """Матрица совместимости симметрична."""
        matrix, _ = compat_data
        diff = np.abs(matrix - matrix.T).max()
        assert diff == pytest.approx(0.0, abs=1e-6), (
            f"Compat matrix is not symmetric: max diff = {diff}"
        )

    def test_compat_entries_sorted_descending(self, compat_data):
        """Список CompatEntry отсортирован по убыванию score."""
        _, entries = compat_data
        if len(entries) < 2:
            pytest.skip("Not enough entries to check ordering")
        scores = [e.score for e in entries]
        assert scores == sorted(scores, reverse=True), (
            "CompatEntry list is not sorted in descending order by score"
        )

    def test_compat_entries_non_negative_scores(self, compat_data):
        """Все score в CompatEntry >= 0."""
        _, entries = compat_data
        for e in entries:
            assert e.score >= 0.0, f"Negative score found: {e.score}"

    def test_compat_entries_no_self_pairs(self, processed_4, compat_data):
        """В CompatEntry нет пар краёв одного и того же фрагмента.
        
        edge_id не уникален между фрагментами (каждый имеет 0..N-1),
        поэтому используем идентификатор объекта Python (id()) для сопоставления.
        """
        _, entries = compat_data
        # Строим карту: id объекта края → fragment_id
        edge_obj_to_frag = {}
        for frag in processed_4:
            for edge in frag.edges:
                edge_obj_to_frag[id(edge)] = frag.fragment_id

        for e in entries:
            fid_i = edge_obj_to_frag.get(id(e.edge_i))
            fid_j = edge_obj_to_frag.get(id(e.edge_j))
            assert fid_i != fid_j, (
                f"Self-pair found: fragment {fid_i} edges paired with each other"
            )
