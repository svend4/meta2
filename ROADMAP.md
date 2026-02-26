# Roadmap: puzzle_reconstruction — план дальнейшего развития

> **Дата составления:** 2026-02-26
> **Текущий статус:** 43 152 юнит-теста (100 % проходят), 105 631 строк production-кода
> **Ветка разработки:** `claude/puzzle-text-docs-3tcRj`

---

## Содержание

1. [Обзор архитектуры](#1-обзор-архитектуры)
2. [Фаза 1 — Интеграционные тесты (глубокий уровень)](#2-фаза-1--интеграционные-тесты)
3. [Фаза 2 — Benchmarks и профилирование](#3-фаза-2--benchmarks)
4. [Фаза 3 — Новая функциональность алгоритмов](#4-фаза-3--новая-функциональность)
5. [Фаза 4 — Качество и надёжность](#5-фаза-4--качество-и-надёжность)
6. [Фаза 5 — UI и экспорт](#6-фаза-5--ui-и-экспорт)
7. [Фаза 6 — DevOps и CI/CD](#7-фаза-6--devops-и-cicd)
8. [Матрица приоритетов](#8-матрица-приоритетов)
9. [Поэтапный план реализации](#9-поэтапный-план-реализации)

---

## 1. Обзор архитектуры

Система строится из **6 последовательных этапов**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      puzzle_reconstruction pipeline                      │
│                                                                         │
│  [images] → preprocessing → synthesis → matching → assembly → verify   │
│                │                │           │          │          │      │
│          segmentation      fractal+CSS  compat_matrix  8 methods  OCR   │
│          contour           tangram      DTW/CSS/FD/txt  greedy/SA  bbox  │
│          color_norm        edge_sig     threshold       beam/gamma seam  │
│          orientation       CSS         consensus       GA/ACO     layout │
│                                                        MCTS/exh   text   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Ключевые модули по подсистемам:**

| Подсистема | Модулей | Строк | Покрытие |
|---|---|---|---|
| `preprocessing/` | 29 | ~9 000 | ✅ unit |
| `algorithms/` | 37 | ~12 000 | ✅ unit |
| `matching/` | 21 | ~7 500 | ✅ unit |
| `assembly/` | 27 | ~8 000 | ✅ unit |
| `scoring/` | 12 | ~4 000 | ✅ unit |
| `verification/` | 21 | ~6 500 | ✅ unit |
| `utils/` | 130 | ~55 000 | ✅ unit |
| `io/` | 3 | ~800 | ✅ unit |
| `ui/` | 1 | ~1 200 | ✅ unit |
| **Integration** | — | — | ⚠️ partial |
| **Benchmarks** | — | — | ❌ missing |
| **Property tests** | — | — | ❌ missing |

---

## 2. Фаза 1 — Интеграционные тесты

### 2.1 Что уже есть

- `test_integration.py` — базовый E2E на синтетических данных (4 фрагмента)
- `test_integration_extra.py` — расширенный вариант
- `test_integration_v2.py` — тест VerificationSuite + `main.run()`

### 2.2 Что нужно добавить

#### 2.2.1 `tests/test_integration_pipeline_class.py`

Тестирует класс `Pipeline` из `puzzle_reconstruction/pipeline.py` (1086 строк, не охвачен интеграционно).

```python
# Структура тестов:

class TestPipelinePreprocess:
    """Pipeline.preprocess(images) → List[Fragment]"""
    # - preprocess возвращает правильное число фрагментов
    # - каждый фрагмент имеет mask, contour, tangram, fractal, edges
    # - параллельная обработка (n_workers=2) даёт тот же результат
    # - graceful degradation при плохих изображениях

class TestPipelineMatch:
    """Pipeline.match(fragments) → (matrix, entries)"""
    # - матрица совместимости правильного размера N×N
    # - диагональ матрицы нулевая
    # - все значения в [0, 1]
    # - threshold_selector задействован
    # - consistency_checker запускается

class TestPipelineAssemble:
    """Pipeline.assemble(fragments, entries) → Assembly"""
    # - все 8 методов сборки через Pipeline.assemble()
    # - auto-mode выбирает лучший метод
    # - результат содержит placement для каждого фрагмента

class TestPipelineVerify:
    """Pipeline.verify(assembly) → VerificationReport"""
    # - score ∈ [0, 1]
    # - все 21 validator запущен
    # - отчёт сериализуем в JSON

class TestPipelineRunFull:
    """Pipeline.run(images) — полный прогон"""
    # - 2 / 4 / 6 фрагмента
    # - seed=42 → воспроизводимый результат
    # - PipelineResult содержит timer, cfg, consistency_report
    # - callback-хуки вызываются (on_preprocess, on_match, on_assemble)
    # - event_bus получает все события

class TestPipelineConfig:
    """Config roundtrip + override"""
    # - Config.default() → json → Config.from_json() → Pipeline.run()
    # - override assembly.method через CLI args
    # - VerificationConfig.validators roundtrip
```

---

#### 2.2.2 `tests/test_integration_assembly_methods.py`

Сравнительный тест всех 8 алгоритмов сборки на одном синтетическом наборе.

```python
# Структура:

ALL_METHODS = ["greedy", "sa", "beam", "gamma", "genetic",
               "exhaustive", "ant_colony", "mcts"]

@pytest.fixture(scope="module")
def four_fragment_data():
    """Одни и те же 4 фрагмента для всех методов."""

class TestAssemblyMethodsConsistency:
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_method_returns_assembly(self, method, four_fragment_data):
        """Каждый метод возвращает Assembly без исключения."""

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_method_covers_all_fragments(self, method, four_fragment_data):
        """В сборке присутствуют все N фрагментов."""

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_method_no_overlap(self, method, four_fragment_data):
        """Placements не перекрываются (collision_detector)."""

    def test_pick_best_selects_highest_score(self, four_fragment_data):
        """pick_best() выбирает Assembly с наибольшим score."""

    def test_run_all_methods_returns_dict(self, four_fragment_data):
        """run_all_methods() возвращает dict {method: Assembly}."""

    def test_summary_table_covers_all_methods(self, four_fragment_data):
        """summary_table() содержит строку для каждого метода."""
```

---

#### 2.2.3 `tests/test_integration_preprocessing_chain.py`

Тест цепочки предобработки (`preprocessing/chain.py`) на синтетических изображениях.

```python
class TestPreprocessingChainIntegration:
    # - chain.run(images) → List[Fragment] без исключений
    # - каждый шаг (color_norm, denoise, edge_enhance, segment, contour)
    #   вызывается в правильном порядке
    # - профиль времени chain.last_profile
    # - пропуск шагов через chain.skip(["denoise"])
    # - идемпотентность: два запуска дают одинаковый контур

class TestPreprocessingStepsIntegration:
    # - segment → contour → orientation → color_norm → patch_sampler
    # - каждый промежуточный артефакт имеет правильный dtype/shape
    # - edge_sharpener → edge_profile: профиль края не пустой
    # - gradient_analyzer: gradients.shape == contour.shape
```

---

#### 2.2.4 `tests/test_integration_matching_full.py`

Полный цикл матчинга: edge_signatures → compat_matrix → threshold → consensus.

```python
class TestCompatMatrixIntegration:
    # - build_compat_matrix(fragments) → matrix (N×N, float32)
    # - матрица симметрична (matrix[i,j] ≈ matrix[j,i])
    # - все активные матчеры (css, dtw, fd, text) вносят вклад
    # - select_threshold(matrix) возвращает разумный порог
    # - run_consistency_check(entries) не поднимает исключений
    # - global_matcher.match() использует полную матрицу

class TestMatcherCombinationIntegration:
    # - weighted combine: изменение весов меняет score
    # - rank combine: монотонный ранг по убыванию score
    # - candidate_ranker: top-K кандидатов в правильном порядке
    # - consensus.vote(): мажоритарный голос корректен
```

---

#### 2.2.5 `tests/test_integration_verification_full.py`

Полный прогон всех 21 валидаторов из `VerificationSuite`.

```python
class TestVerificationSuiteFullIntegration:
    # - suite.run_all(assembly) — все 21 validator без исключений
    # - report.summary_score ∈ [0, 1]
    # - каждый ValidatorResult имеет score ∈ [0, 1] и passed: bool
    # - validators_passed >= 0, validators_failed >= 0
    # - suite.run_selected(["boundary", "spatial"]) — подмножество
    # - all_validator_names() возвращает список ≥ 21 имён
    # - отчёт JSON-сериализуем и десериализуем
    # - report с score < 0.5 → flag low_quality

class TestVerificationMetricsIntegration:
    # - evaluate_reconstruction(ground_truth, assembly) → BenchmarkResult
    # - perfect assembly: score = 1.0
    # - shuffled assembly: score < 1.0
    # - compare_methods(assemblies_dict) строит сравнительную таблицу
```

---

#### 2.2.6 `tests/test_integration_io_export.py`

Тест roundtrip: сохранение результата → загрузка → верификация.

```python
class TestExportImportRoundtrip:
    # - export_result(assembly, ExportConfig) → файлы в tmpdir
    # - load_from_directory(tmpdir) → images
    # - metadata_writer.write() → JSON-файл читаем обратно
    # - ExportConfig: форматы PNG, JSON, CSV
    # - большие сборки (≥ 9 фрагментов) экспортируются без ошибок
```

---

#### 2.2.7 `tests/test_integration_config_overrides.py`

Тест влияния конфигурации на результат.

```python
class TestConfigImpactIntegration:
    # - AssemblyConfig(method="greedy") vs method="beam": разные scores
    # - FractalConfig(n_scales=4) vs n_scales=12: разные FD values
    # - MatchingConfig(threshold=0.1) включает больше пар, threshold=0.9 меньше
    # - MatchingConfig(combine_method="rank") vs "weighted": сравнение
    # - Config.from_json(Config.default().to_json()) → идентичные результаты
    # - SynthesisConfig(alpha=0.0): только фрактал; alpha=1.0: только танграм
```

---

#### 2.2.8 `tests/test_integration_error_recovery.py`

Тест устойчивости к плохим входным данным.

```python
class TestErrorRecoveryIntegration:
    # - пустое изображение → graceful error, не crash
    # - одноцветное изображение → FD=1.0, контур=пустой
    # - изображение 1×1 пикселей → без IndexError
    # - фрагмент без текста → text_coherence_score = 0.0, не nan
    # - матрица из одних нулей → greedy_assembly возвращает что-то
    # - очень малые фрагменты (< 10×10 пикселей)
    # - дублированные изображения → идентичные сигнатуры
```

---

## 3. Фаза 2 — Benchmarks

### 3.1 Инфраструктура бенчмарков

Создать `benchmarks/` как отдельный пакет (не в `tests/`), запускаемый отдельно:

```
benchmarks/
├── __init__.py
├── conftest.py              # общие fixture
├── data/                    # синтетические данные (генерируются при запуске)
├── bench_preprocessing.py
├── bench_descriptors.py
├── bench_compat_matrix.py
├── bench_assembly_methods.py
├── bench_verification.py
├── bench_pipeline_e2e.py
├── bench_memory.py
└── utils.py                 # вспомогательные функции
```

**Запуск:**
```bash
# Базовый прогон
python -m pytest benchmarks/ -v --benchmark-only

# С pytest-benchmark
pytest benchmarks/ --benchmark-json=results/bench_$(date +%Y%m%d).json

# Сравнение двух версий
pytest-benchmark compare results/bench_yesterday.json results/bench_today.json
```

---

### 3.2 `benchmarks/bench_preprocessing.py`

```python
# Что измерять:

@pytest.mark.benchmark
class BenchPreprocessing:
    # bench_segment_otsu(fragment_400x500)
    # bench_segment_adaptive(fragment_400x500)
    # bench_extract_contour(mask_400x500)
    # bench_color_norm(image_400x500)
    # bench_orientation_estimate(fragment_400x500)
    # bench_gradient_analyzer(fragment_400x500)
    # bench_full_chain_1_fragment
    # bench_full_chain_4_fragments
    # bench_full_chain_9_fragments
    # bench_full_chain_16_fragments   ← scaling
    # bench_full_chain_25_fragments   ← scaling

# Ожидаемые целевые показатели:
# segment_otsu:     < 5 ms  per fragment
# extract_contour:  < 2 ms  per fragment
# full_chain_4:     < 50 ms total
# full_chain_16:    < 200 ms total
```

---

### 3.3 `benchmarks/bench_descriptors.py`

```python
class BenchDescriptors:
    # bench_css_compute(contour_256pts)   ← curvature_scale_space()
    # bench_css_feature_vector(css_result)
    # bench_css_similarity(vec_a, vec_b)
    # bench_box_counting_fd(contour_256pts)
    # bench_box_counting_curve(contour_256pts)
    # bench_ifs_compute(contour_256pts)
    # bench_divider_fd(contour_256pts)
    # bench_freeman_chain_code(contour_256pts)
    # bench_dtw_pair(curve_128pts, curve_128pts)
    # bench_fourier_descriptor(contour_256pts)
    # bench_shape_context(contour_256pts)
    # bench_full_fractal_signature(contour_256pts)
    # bench_full_fractal_4_edges
    # bench_full_fractal_9_fragments

# Масштабирование по числу точек контура:
# @pytest.mark.parametrize("n_pts", [64, 128, 256, 512, 1024])
# def bench_css_scaling(n_pts)
```

---

### 3.4 `benchmarks/bench_compat_matrix.py`

```python
class BenchCompatMatrix:
    # bench_compat_matrix_2_fragments
    # bench_compat_matrix_4_fragments
    # bench_compat_matrix_9_fragments
    # bench_compat_matrix_16_fragments
    # bench_compat_matrix_25_fragments  ← O(N²) scaling visible here
    # bench_compat_matrix_matcher_css
    # bench_compat_matrix_matcher_dtw
    # bench_compat_matrix_matcher_fd
    # bench_compat_matrix_matcher_text
    # bench_compat_matrix_all_matchers
    # bench_select_threshold(matrix_25x25)
    # bench_consistency_check(entries_100)

# Строить график: N_fragments vs time_ms
# Ожидаемое O(N²) поведение
```

---

### 3.5 `benchmarks/bench_assembly_methods.py`

Самый важный бенчмарк — сравнение 8 алгоритмов.

```python
METHODS = ["greedy", "sa", "beam", "gamma",
           "genetic", "exhaustive", "ant_colony", "mcts"]

class BenchAssemblyMethods:
    # @pytest.mark.parametrize("method", METHODS)
    # @pytest.mark.parametrize("n_pieces", [4, 6, 9])
    # def bench_method(method, n_pieces, prepared_data)

# Выходная таблица (пример):
# ┌──────────────┬──────────┬──────────┬────────────┬───────────┐
# │ method       │ 4-pieces │ 6-pieces │ 9-pieces   │ accuracy  │
# ├──────────────┼──────────┼──────────┼────────────┼───────────┤
# │ greedy       │  0.3 ms  │  0.5 ms  │  0.8 ms    │  0.72     │
# │ beam(w=10)   │  2.1 ms  │  4.8 ms  │ 12.5 ms    │  0.85     │
# │ sa(5000 iter)│  45 ms   │ 120 ms   │ 310 ms     │  0.89     │
# │ gamma        │  38 ms   │  95 ms   │ 250 ms     │  0.87     │
# │ genetic(50)  │  85 ms   │ 220 ms   │ 580 ms     │  0.88     │
# │ ant_colony   │  62 ms   │ 155 ms   │ 410 ms     │  0.86     │
# │ mcts(200)    │  35 ms   │  92 ms   │ 245 ms     │  0.84     │
# │ exhaustive   │  1.2 ms  │ 180 ms   │ N/A (>9!)  │  1.00     │
# └──────────────┴──────────┴──────────┴────────────┴───────────┘
```

---

### 3.6 `benchmarks/bench_memory.py`

```python
# Использует tracemalloc / memory_profiler

class BenchMemory:
    # bench_memory_preprocessing_4_fragments
    # bench_memory_compat_matrix_25_fragments
    # bench_memory_assembly_beam_25_fragments
    # bench_memory_full_pipeline_25_fragments
    # bench_memory_bridge_registry_build

# Целевые показатели:
# full_pipeline_25_fragments: < 500 MB peak RSS
# compat_matrix_25x25: < 50 MB (float32 матрица)
```

---

### 3.7 `benchmarks/bench_pipeline_e2e.py`

```python
class BenchPipelineE2E:
    # bench_pipeline_run_4_fragments   ← < 2 s целевой показатель
    # bench_pipeline_run_9_fragments   ← < 10 s
    # bench_pipeline_run_16_fragments  ← < 30 s
    # bench_pipeline_preprocess_only
    # bench_pipeline_match_only
    # bench_pipeline_assemble_only
    # bench_pipeline_verify_only
    # bench_pipeline_with_all_validators
    # bench_pipeline_parallel_vs_sequential
```

---

### 3.8 Расширение `tools/benchmark.py`

Текущий `tools/benchmark.py` уже частично реализован. Добавить:

```python
# Новые флаги CLI:
# --scalability          : тест масштабируемости (2→4→9→16→25 фрагментов)
# --method-comparison    : полная таблица всех 8 методов
# --descriptor-timing    : замер времени каждого дескриптора
# --memory-profile       : замер памяти (tracemalloc)
# --export-plots         : matplotlib-графики в results/plots/
# --baseline FILE.json   : сравнить с предыдущим прогоном
# --html-report          : HTML-отчёт с таблицами и графиками

# Добавить класс BenchmarkSuite:
class BenchmarkSuite:
    def run_scalability_test(self, piece_counts=[2,4,6,9,16,25]) -> pd.DataFrame
    def run_method_comparison(self, n_pieces=6, trials=3) -> pd.DataFrame
    def run_descriptor_benchmark(self) -> pd.DataFrame
    def export_html_report(self, results: dict, outfile: str)
    def compare_with_baseline(self, current: dict, baseline: dict) -> str
```

---

## 4. Фаза 3 — Новая функциональность

### 4.1 Улучшение дескрипторов

#### 4.1.1 Wavelet-дескриптор края

```python
# Новый файл: algorithms/wavelet_descriptor.py
# Дополняет CSS — вейвлет-разложение профиля края

def wavelet_edge_descriptor(contour: np.ndarray,
                             wavelet: str = "db4",
                             level: int = 4) -> np.ndarray:
    """
    Многомасштабный вейвлет-дескриптор контура.
    Инвариантен к малым деформациям и шуму.
    """

def wavelet_similarity(desc_a: np.ndarray, desc_b: np.ndarray) -> float:
    """L2-расстояние в вейвлет-пространстве."""
```

#### 4.1.2 Zernike moments

```python
# algorithms/zernike_descriptor.py
# Ортогональные моменты — компактное описание формы

def zernike_moments(contour: np.ndarray, order: int = 10) -> np.ndarray:
    """
    Моменты Цернике для описания формы фрагмента.
    Инварианты к повороту, масштабу и переносу.
    """

def zernike_similarity(m_a: np.ndarray, m_b: np.ndarray) -> float:
```

#### 4.1.3 Улучшенный IFS-матчинг

```python
# Расширение algorithms/fractal/ifs.py:

def ifs_match_edges(ifs_a: np.ndarray, ifs_b: np.ndarray,
                    n_candidates: int = 5) -> float:
    """
    Сопоставление IFS-коэффициентов с учётом перестановок.
    Фрактальный аттрактор инвариантен к порядку аффинных преобразований.
    """

def ifs_to_attractor(coeffs: np.ndarray,
                     n_iter: int = 10000) -> np.ndarray:
    """Генерирует аттрактор IFS для визуализации."""
```

---

### 4.2 Новые алгоритмы сборки

#### 4.2.1 A* поиск (детерминированный)

```python
# assembly/astar.py

class AStarAssembler:
    """
    A* поиск оптимальной сборки.
    Гарантирует оптимальность при допустимой эвристике.
    Используется как upper bound для сравнения.

    Эвристика h(state) = sum(best_possible_score для незафиксированных фрагментов)
    """
    def assemble(self, fragments, compat_matrix) -> Assembly
    def _heuristic(self, state: PartialAssembly) -> float
    def _expand(self, state: PartialAssembly) -> List[PartialAssembly]
```

#### 4.2.2 Иерархический метод (bottom-up clustering)

```python
# assembly/hierarchical.py

class HierarchicalAssembler:
    """
    Агломеративная сборка: сначала объединяем пары с высшим score,
    затем группы, пока не получим одну сборку.
    Устойчив к локальным оптимумам.
    """
    def assemble(self, fragments, compat_matrix) -> Assembly
    def _merge_clusters(self, a: Cluster, b: Cluster) -> Cluster
```

#### 4.2.3 Reinforcement Learning агент (опционально)

```python
# assembly/rl_agent.py

class RLAssembler:
    """
    Policy gradient агент для сборки пазла.
    State:  (compat_matrix, current_placement_mask)
    Action: (fragment_id, position, orientation)
    Reward: delta_score после размещения

    Требует: torch (optional dependency)
    """
```

---

### 4.3 Улучшение матчинга

#### 4.3.1 Rotation-aware DTW

```python
# matching/rotation_dtw.py

def rotation_invariant_dtw(curve_a: np.ndarray,
                            curve_b: np.ndarray,
                            n_rotations: int = 36) -> tuple[float, float]:
    """
    DTW с перебором поворотов на 0..360° (шаг 10°).
    Возвращает (min_dtw_distance, best_rotation_angle).

    Решает проблему: два фрагмента подходят друг к другу,
    но один повёрнут относительно другого.
    """
```

#### 4.3.2 Spectral graph matching

```python
# matching/spectral_matcher.py (расширение существующего)

def spectral_graph_matching(G1: np.ndarray, G2: np.ndarray) -> float:
    """
    Сопоставление через спектры лапласианов.
    Инвариантно к перенумерации фрагментов.

    Применение: глобальная совместимость группы фрагментов,
    а не только попарная.
    """
```

#### 4.3.3 Text-flow matching

```python
# matching/text_flow.py (новый модуль)

class TextFlowMatcher:
    """
    Сопоставление строк текста, разорванных на границе фрагментов.

    Алгоритм:
    1. OCR обоих краёв (pytesseract/easyocr)
    2. Выравнивание Смита-Уотермана по частям слов
    3. Score = 1 - edit_distance / max_len

    Превосходит CSS для документов с видимым текстом.
    """
    def match(self, edge_a: EdgeSignature,
              edge_b: EdgeSignature) -> float
    def _smith_waterman_score(self, text_a: str, text_b: str) -> float
```

---

### 4.4 Улучшение верификации

#### 4.4.1 Геометрическая верификация через Homography

```python
# verification/homography_verifier.py

class HomographyVerifier:
    """
    Проверяет геометрическую согласованность сборки
    через оценку гомографии между соседними фрагментами.

    Если края смыкаются корректно — homography_error < threshold.
    """
    def verify(self, assembly: Assembly) -> ValidatorResult
    def _estimate_homography(self, f1: Fragment, f2: Fragment) -> np.ndarray
    def _reprojection_error(self, H: np.ndarray, pts: np.ndarray) -> float
```

#### 4.4.2 Цветовая согласованность

```python
# verification/color_continuity_verifier.py

class ColorContinuityVerifier:
    """
    Проверяет плавность перехода цвета через швы между фрагментами.

    Метрика: mean |color_left - color_right| по всем пикселям шва.
    """
    def verify(self, assembly: Assembly) -> ValidatorResult
    def _seam_color_delta(self, f1: Fragment, f2: Fragment,
                          placement1: Placement, placement2: Placement) -> float
```

#### 4.4.3 Statistical coherence scoring

```python
# verification/statistical_coherence.py

class StatisticalCoherenceVerifier:
    """
    Проверяет статистическую согласованность:
    - распределение яркости по соседним фрагментам совместимо
    - гистограммы текстуры схожи вдоль общих краёв
    - шум имеет одинаковую дисперсию (один и тот же документ)
    """
```

---

### 4.5 Preprocessing улучшения

#### 4.5.1 Illumination equalization

```python
# preprocessing/illumination_equalizer.py (улучшение существующего)

def equalize_fragments(fragments: List[Fragment],
                       method: Literal["histogram", "retinex", "clahe"]) -> List[Fragment]:
    """
    Выравнивает освещение всех фрагментов к общей базе.
    Критично для реальных (не синтетических) документов.
    """
```

#### 4.5.2 Torn-edge enhancement

```python
# preprocessing/tear_enhancer.py

class TearEdgeEnhancer:
    """
    Усиливает детали разорванного края:
    1. Sub-pixel refinement через supersampling
    2. Denoising специфически на крае (не на всём изображении)
    3. Contrast enhancement вдоль контура

    Улучшает FD и CSS точность на реальных данных.
    """
```

#### 4.5.3 Multi-scale segmentation

```python
# preprocessing/multiscale_segmenter.py

class MultiscaleSegmenter:
    """
    Сегментация в нескольких масштабах с голосованием.
    Устойчива к неравномерному фону.
    """
    def segment(self, image: np.ndarray) -> np.ndarray
    def _segment_at_scale(self, image, scale) -> np.ndarray
    def _vote(self, masks: List[np.ndarray]) -> np.ndarray
```

---

### 4.6 Новые модели данных

#### 4.6.1 `MatchingState` — состояние матчинга

```python
# models.py дополнение

@dataclass
class MatchingState:
    """Полное состояние матчинга для паузы/возобновления."""
    compat_matrix:  np.ndarray          # N×N
    entries:        List[CompatEntry]   # отфильтрованные пары
    threshold:      float
    consistency:    ConsistencyReport
    timestamp:      str
    config:         MatchingConfig

    def save(self, path: str) -> None
    def load(path: str) -> "MatchingState"
```

#### 4.6.2 `AssemblySession` — сессия сборки

```python
@dataclass
class AssemblySession:
    """
    Сохраняемая сессия сборки с историей итераций.
    Позволяет возобновить прерванный прогон SA/genetic.
    """
    method:         str
    iteration:      int
    best_assembly:  Assembly
    history:        List[float]         # score по итерациям
    config:         AssemblyConfig

    def checkpoint(self, path: str) -> None
    def resume(self, path: str) -> "AssemblySession"
```

---

## 5. Фаза 4 — Качество и надёжность

### 5.1 Property-based тесты (Hypothesis)

```python
# tests/test_properties_fractal.py

from hypothesis import given, strategies as st

class TestCSSProperties:
    @given(st.integers(min_value=10, max_value=500))
    def test_css_feature_vector_always_unit_norm(self, n_pts):
        """Для любого контура ||css_vector|| ≈ 1."""

    @given(st.floats(min_value=0.1, max_value=10.0))
    def test_css_similarity_range(self, scale):
        """Масштабирование контура не меняет css_similarity."""

class TestBoxCountingProperties:
    @given(st.integers(min_value=4, max_value=1000))
    def test_fd_always_in_range(self, n_pts):
        """FD ∈ [1.0, 2.0] для любого числа точек."""

    def test_fd_circle_near_1(self):
        """FD окружности ≈ 1.0 (гладкая кривая)."""

    def test_fd_fractal_above_1(self):
        """FD фракталоподобного контура > 1.1."""

class TestTangramProperties:
    @given(st.integers(min_value=3, max_value=20))
    def test_convex_hull_subset_of_input(self, n_pts):
        """Выпуклая оболочка ⊆ входных точек."""

    def test_normalize_polygon_centroid_at_origin(self):
        """После normalize_polygon: centroid = (0, 0)."""
```

---

### 5.2 Регрессионные тесты с фиксированными данными

```python
# tests/test_regression.py

class TestRegressionFractalSignatures:
    """
    Фиксированные эталонные значения дескрипторов.
    Гарантируют обратную совместимость алгоритмов.
    """
    def test_css_known_square(self):
        """CSS-вектор квадрата не изменился от версии к версии."""
        EXPECTED = np.array([0.125, 0.0, 0.125, 0.0, ...])  # зафиксировано
        actual = css_to_feature_vector(curvature_scale_space(SQUARE_CONTOUR))
        assert np.allclose(actual, EXPECTED, atol=1e-4)

    def test_fd_known_koch_curve(self):
        """FD кривой Коха ≈ 1.26 (теоретическое значение)."""
        fd = box_counting_fd(KOCH_CONTOUR)
        assert 1.20 < fd < 1.32

class TestRegressionAssembly:
    """Воспроизводимость сборки."""
    def test_greedy_deterministic(self):
        """greedy_assembly(seed=42) всегда возвращает одно и то же."""

    def test_beam_score_stable(self):
        """score beam_search(seed=42, w=10) = KNOWN_VALUE ± 1e-6."""
```

---

### 5.3 Type annotations и mypy

```python
# Добавить strict mypy в pyproject.toml:
# [tool.mypy]
# strict = true
# disallow_untyped_defs = true
# check_untyped_defs = true

# Приоритетные модули для аннотирования:
# pipeline.py    — центральный класс
# config.py      — уже аннотирован
# models.py      — частично аннотирован
# assembly/*.py  — 8 алгоритмов
# matching/compat_matrix.py
```

---

### 5.4 Мутационное тестирование

```bash
# Проверяет, действительно ли тесты обнаруживают баги

pip install mutmut

mutmut run --paths-to-mutate puzzle_reconstruction/algorithms/fractal/
mutmut results
# Цель: mutation score > 85%

# Если тест проходит при мутированном коде → тест слабый
# Добавить assertion'ы там, где мутации выживают
```

---

## 6. Фаза 5 — UI и экспорт

### 6.1 Улучшение `ui/viewer.py`

```python
# Текущий viewer.py — 1200 строк. Добавить:

class AssemblyViewer:
    def show_compat_matrix_heatmap(self, matrix: np.ndarray) -> None:
        """Тепловая карта матрицы совместимости."""

    def show_assembly_animated(self, history: List[Assembly]) -> None:
        """Анимация процесса сборки (SA/genetic iterations)."""

    def show_edge_comparison(self, e1: EdgeSignature,
                              e2: EdgeSignature) -> None:
        """Side-by-side сравнение двух краёв: CSS, профиль, IFS."""

    def export_svg(self, assembly: Assembly, path: str) -> None:
        """Экспорт финальной сборки в SVG."""

    def show_verification_report(self, report: VerificationReport) -> None:
        """Визуализация результатов верификации: прошедшие/проваленные."""
```

---

### 6.2 Web API (расширение `tools/server.py`)

Текущий `tools/server.py` — 568 строк. Добавить:

```python
# REST API endpoints:

POST /api/v1/reconstruct
    Request:  multipart/form-data (images[] + config.json)
    Response: { assembly_id, score, placements[] }

GET  /api/v1/assembly/{id}
    Response: full AssemblyResult JSON

GET  /api/v1/assembly/{id}/image
    Response: PNG/SVG финальной сборки

POST /api/v1/benchmark
    Request:  { method, n_pieces, trials }
    Response: BenchmarkResult JSON

GET  /api/v1/health
    Response: { status, version, algorithms[] }

# WebSocket для прогресса долгих операций:
WS   /api/v1/reconstruct/stream
    Events: preprocess_done, match_done, assembly_progress, done
```

---

### 6.3 CLI улучшения (`main.py`)

```bash
# Новые команды:

# Интерактивный режим
python main.py interactive

# Сравнение методов
python main.py compare --pieces *.png --methods all --output compare.html

# Только препроцессинг
python main.py preprocess --input *.png --output features.pkl

# Бенчмарк из CLI
python main.py benchmark --pieces 4 9 16 --trials 3 --plot

# Экспорт в разные форматы
python main.py reconstruct *.png --export svg,pdf,json
```

---

## 7. Фаза 6 — DevOps и CI/CD

### 7.1 GitHub Actions workflow

```yaml
# .github/workflows/ci.yml

name: CI

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run unit tests
        run: pytest tests/ -q --tb=short -x

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - name: Run integration tests
        run: pytest tests/ -m integration --tb=short

  benchmarks:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Run benchmarks
        run: pytest benchmarks/ --benchmark-json=bench_results.json
      - name: Store benchmark result
        uses: benchmark-action/github-action-benchmark@v1

  mypy:
    runs-on: ubuntu-latest
    steps:
      - name: Type check
        run: mypy puzzle_reconstruction/ --ignore-missing-imports

  mutation-testing:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - name: Run mutmut on changed files
        run: mutmut run --paths-to-mutate $(git diff --name-only origin/main)
```

---

### 7.2 Матрица зависимостей

```toml
# pyproject.toml дополнения:

[project.optional-dependencies]
benchmarks = ["pytest-benchmark>=4.0", "memory-profiler", "psutil"]
property   = ["hypothesis>=6.0"]
mypy       = ["mypy>=1.8", "types-Pillow", "pandas-stubs"]
dl         = ["torch>=2.0"]   # для RL-assembler
wavelet    = ["PyWavelets"]   # для wavelet_descriptor
```

---

## 8. Матрица приоритетов

| Задача | Ценность | Сложность | Приоритет |
|--------|----------|-----------|-----------|
| `test_integration_pipeline_class.py` | ⭐⭐⭐ | Низкая | 🔴 HIGH |
| `test_integration_assembly_methods.py` | ⭐⭐⭐ | Низкая | 🔴 HIGH |
| `benchmarks/bench_assembly_methods.py` | ⭐⭐⭐ | Средняя | 🔴 HIGH |
| `test_integration_error_recovery.py` | ⭐⭐⭐ | Низкая | 🔴 HIGH |
| `matching/rotation_dtw.py` | ⭐⭐⭐ | Средняя | 🟡 MED |
| `matching/text_flow.py` | ⭐⭐⭐ | Высокая | 🟡 MED |
| `test_properties_fractal.py` (Hypothesis) | ⭐⭐ | Низкая | 🟡 MED |
| `test_regression.py` | ⭐⭐ | Низкая | 🟡 MED |
| `assembly/astar.py` | ⭐⭐ | Высокая | 🟢 LOW |
| `algorithms/wavelet_descriptor.py` | ⭐⭐ | Средняя | 🟢 LOW |
| `algorithms/zernike_descriptor.py` | ⭐ | Высокая | 🟢 LOW |
| `assembly/rl_agent.py` | ⭐ | Очень высокая | 🟢 LOW |
| Web API (server.py) | ⭐⭐ | Высокая | 🟢 LOW |
| mypy strict | ⭐⭐ | Средняя | 🟢 LOW |

---

## 9. Поэтапный план реализации

### Итерация 1 (1–2 дня): Интеграционные тесты Pipeline

```
tests/
├── test_integration_pipeline_class.py   [NEW] ~80 тестов
├── test_integration_assembly_methods.py [NEW] ~40 тестов
└── test_integration_error_recovery.py   [NEW] ~30 тестов

Итого: +150 тестов → 43 302 total
```

**Шаги:**
1. Читаем `pipeline.py` полностью
2. Пишем фикстуры: `four_fragment_pipeline`, `nine_fragment_pipeline`
3. `TestPipelinePreprocess` → `TestPipelineMatch` → `TestPipelineAssemble` → `TestPipelineVerify`
4. `TestPipelineRunFull` — полный прогон с колбэками
5. `TestAssemblyMethodsConsistency` — все 8 методов параметрически
6. `TestErrorRecoveryIntegration` — edge cases

---

### Итерация 2 (1–2 дня): Matching + Verification integration

```
tests/
├── test_integration_matching_full.py    [NEW] ~50 тестов
├── test_integration_verification_full.py[NEW] ~40 тестов
└── test_integration_io_export.py        [NEW] ~25 тестов

Итого: +115 тестов → 43 417 total
```

---

### Итерация 3 (2–3 дня): Benchmarks

```
benchmarks/
├── __init__.py
├── conftest.py
├── bench_descriptors.py    ~15 bench-функций
├── bench_compat_matrix.py  ~10 bench-функций
├── bench_assembly_methods.py ~24 bench-функций (8 методов × 3 размера)
├── bench_pipeline_e2e.py   ~10 bench-функций
└── utils.py

Расширить tools/benchmark.py:
└── + scalability_test, method_comparison, html_report
```

---

### Итерация 4 (1–2 дня): Property тесты + Regression

```
tests/
├── test_properties_fractal.py   [NEW] ~20 Hypothesis-тестов
├── test_properties_geometry.py  [NEW] ~15 Hypothesis-тестов
└── test_regression.py           [NEW] ~25 регрессионных тестов

Итого: +60 тестов → 43 477 total
```

---

### Итерация 5 (2–3 дня): Rotation-aware DTW + Text flow

```python
# matching/rotation_dtw.py     [NEW]
# matching/text_flow.py        [NEW]
# config.py                    [EXTEND] — добавить MatchingConfig.rotation_n_steps
# tests/test_rotation_dtw.py   [NEW]
# tests/test_text_flow.py      [NEW]
```

---

### Итерация 6 (2–3 дня): A* assembler + Hierarchical

```python
# assembly/astar.py            [NEW]
# assembly/hierarchical.py     [NEW]
# assembly/parallel.py         [EXTEND] — добавить в ALL_METHODS
# config.py                    [EXTEND] — AssemblyConfig.method += ["astar", "hierarchical"]
# tests/...                    [NEW]
```

---

### Итерация 7: Верификация (Homography + Color continuity)

```python
# verification/homography_verifier.py      [NEW]
# verification/color_continuity_verifier.py[NEW]
# verification/suite.py                    [EXTEND] — регистрация новых валидаторов
```

---

### Итерация 8: CI/CD + mypy

```
.github/workflows/ci.yml      [NEW]
.github/workflows/bench.yml   [NEW]
pyproject.toml                [EXTEND] — mypy, optional deps
```

---

## Быстрый старт следующего шага

```bash
# 1. Проверить текущее состояние
python -m pytest --co -q | tail -2

# 2. Запустить существующие integration-тесты
python -m pytest tests/test_integration*.py -v

# 3. Запустить существующий benchmark
python tools/benchmark.py --pieces 4 --methods greedy beam sa --trials 2

# 4. Начать итерацию 1
# → Создать tests/test_integration_pipeline_class.py
```

---

*Документ создан автоматически на основе анализа кодовой базы 2026-02-26.*
*105 631 строк кода · 43 152 юнит-теста · 15 подсистем · 8 алгоритмов сборки*
