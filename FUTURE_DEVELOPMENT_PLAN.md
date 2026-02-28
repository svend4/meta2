# План дальнейшего развития: puzzle-reconstruction

> **Документ создан:** 2026-02-28
> **Текущая версия:** 1.0.0 (Production/Stable)
> **Ветка разработки:** `claude/puzzle-text-docs-3tcRj`
> **Репозиторий:** `meta2`
> **Дата последнего коммита:** 2026-02-27
> **Всего коммитов:** 449
> **Всего тестов:** 57 624 (100 % проходят)

---

## ИНСТРУКЦИЯ ДЛЯ СЛЕДУЮЩЕЙ СЕССИИ / НОВОГО ЧАТА

> **Прочти этот раздел первым — здесь всё, что нужно для мгновенного старта.**

### Шаг 1 — Ориентация

```bash
cd /home/user/meta2
git status                       # убедиться: ветка claude/puzzle-text-docs-3tcRj
git log --oneline -5             # последние коммиты
python3 -m pytest --co -q 2>/dev/null | tail -2   # кол-во тестов
```

### Шаг 2 — Прочитать ключевые файлы

| Файл | Зачем |
|------|-------|
| `FUTURE_DEVELOPMENT_PLAN.md` | **этот файл** — детальный план |
| `DEV_STATUS.md` | текущий статус всего проекта (полный) |
| `ROADMAP.md` | прежний roadmap (устарел частично, но полезен) |
| `puzzle_reconstruction/pipeline.py` | центральный класс Pipeline |
| `puzzle_reconstruction/config.py` | все конфигурации |
| `puzzle_reconstruction/models.py` | все модели данных |

### Шаг 3 — Git операции

```bash
# Убедиться что на правильной ветке
git checkout claude/puzzle-text-docs-3tcRj

# Запуск тестов (быстрая проверка)
python3 -m pytest tests/ -q --tb=short -x 2>&1 | tail -10

# Запуск отдельного файла (пример)
python3 -m pytest tests/test_properties_fractal.py -v

# После каждого изменения — коммит и пуш
git add <файлы>
git commit -m "tests: <описание>"
git push -u origin claude/puzzle-text-docs-3tcRj
```

### Шаг 4 — Структура проекта (шпаргалка)

```
puzzle_reconstruction/
├── pipeline.py          # Pipeline — 6 этапов, главный класс
├── config.py            # Config, MatchingConfig, AssemblyConfig, etc.
├── models.py            # Fragment, EdgeSignature, Assembly, Placement
├── algorithms/          # 42 модуля — дескрипторы краёв (fractal, tangram)
├── matching/            # 26 модулей — 13+ матчеров (DTW, CSS, SIFT, ...)
├── assembly/            # 27 модулей — 8 алгоритмов сборки
├── verification/        # 21 валидатор — проверка сборки
├── preprocessing/       # 42 модуля — подготовка изображений
├── scoring/             # 13 модулей — оценка качества
├── utils/               # 131 модуль — утилиты
└── ui/ io/ export.py clustering.py
tests/                   # 1087 файлов, 57 624 теста
benchmarks/              # 7 бенчмарк-файлов (инфраструктура готова)
tools/                   # 6 CLI инструментов
```

### Шаг 5 — Первая задача в новом чате

**Начинать с этапа 1 из раздела "ПРИОРИТЕТНЫЙ ПЛАН РАБОТ"** (см. ниже).
Первый приоритет: **интеграционные тесты глубокого уровня** для `Pipeline` класса.

```bash
# Создать файл и начать:
# tests/test_integration_pipeline_class.py
```

---

## РАЗДЕЛ 1: ТЕКУЩИЙ СТАТУС ПРОЕКТА

### 1.1 Общие метрики (2026-02-28)

| Метрика | Значение |
|---------|----------|
| **Версия** | 1.0.0 (Production/Stable) |
| **Строк production-кода** | ~93 000 |
| **Строк тестового кода** | ~320 000+ |
| **Модулей production** | 305 `.py` файлов |
| **Тестовых файлов** | 1 087 |
| **Тестов всего** | 57 624 |
| **Тестов проходит** | 57 624 (100 %) |
| **Провалено** | 0 |
| **Предупреждений pytest** | 0 |
| **TODO/FIXME в коде** | 0 |
| **Коммитов** | 449 |
| **Период разработки** | 8 дней (20–28 фев 2026) |
| **Merged PR** | 15 |

---

### 1.2 Что реализовано — полный список

#### Алгоритмы и подсистемы

| Подсистема | Модулей | Статус |
|-----------|---------|--------|
| `preprocessing/` | 42 | ✅ 100% |
| `algorithms/` | 42 (incl. fractal, tangram) | ✅ 100% |
| `matching/` | 26 (13+ матчеров) | ✅ 100% |
| `assembly/` | 27 (8 алгоритмов) | ✅ 100% |
| `verification/` | 21 валидатор | ✅ 100% |
| `scoring/` | 13 | ✅ 100% |
| `utils/` | 131 | ✅ 100% |
| `io/` | 3 | ✅ 100% |
| `ui/` | 1 | ✅ базовый |

#### 8 алгоритмов сборки (assembly/)

| # | Алгоритм | Файл | Статус |
|---|---------|------|--------|
| 1 | Greedy | `greedy.py` | ✅ |
| 2 | Simulated Annealing | `annealing.py` | ✅ |
| 3 | Beam Search | `beam_search.py` | ✅ |
| 4 | Gamma Optimizer | `gamma_optimizer.py` | ✅ |
| 5 | Genetic Algorithm | `genetic.py` | ✅ |
| 6 | Ant Colony | `ant_colony.py` | ✅ |
| 7 | MCTS | `mcts.py` | ✅ |
| 8 | Exhaustive (B&B) | `exhaustive.py` | ✅ |
| + | A* | `astar.py` | ✅ |
| + | Hierarchical | `hierarchical.py` | ✅ |
| + | RL Agent | `rl_agent.py` | ✅ |
| + | Auto-select | `parallel.py` | ✅ |

#### 21 валидатор (verification/)

`assembly_scorer`, `boundary_validator`, `color_continuity_verifier`,
`completeness_checker`, `confidence_scorer`, `consistency_checker`,
`edge_validator`, `fragment_validator`, `homography_verifier`,
`layout_checker`, `layout_scorer`, `layout_verifier`, `metrics`,
`ocr`, `overlap_checker`, `overlap_validator`, `placement_validator`,
`quality_reporter`, `score_reporter`, `seam_analyzer`, `spatial_validator`,
`statistical_coherence`, `text_coherence`

#### Тестовое покрытие (по типу тестов)

| Тип тестов | Файлов | Примерно тестов |
|-----------|--------|-----------------|
| Unit (базовые + extra) | 1 000+ | ~40 000 |
| Property-based (Hypothesis) | 31 | ~2 700 |
| Integration (низкое покрытие) | 9 (low_coverage) | ~500 |
| Integration (алгоритмы) | 3 | ~150 |
| Integration (E2E) | 3 | ~150 |
| Regression | 1 | ~40 |
| CLI тесты | 4 | ~130 |
| Verification Suite | 2 | ~125 |

---

## РАЗДЕЛ 2: ЧТО ЕЩЁ НЕ СДЕЛАНО — ПОЛНЫЙ СПИСОК

### 2.1 Тесты — пробелы

#### 2.1.1 Property-based тесты (Hypothesis) — НЕ ПОКРЫТЫ

Текущие 31 файл покрывают преимущественно `utils/`. Не охвачены property-тестами:

| Модуль / группа | Файлы | Приоритет |
|----------------|-------|-----------|
| `algorithms/fractal/css.py` | css_similarity, css_to_feature_vector | 🔴 HIGH |
| `algorithms/fractal/box_counting.py` | box_counting_fd | 🔴 HIGH |
| `algorithms/fractal/ifs.py` | fit_ifs_coefficients | 🟡 MED |
| `algorithms/tangram/*.py` | конвекс-оболочка, классификация форм | 🟡 MED |
| `algorithms/fourier_descriptor.py` | инварианты Фурье | 🟡 MED |
| `algorithms/shape_context.py` | инварианты к повороту/масштабу | 🟡 MED |
| `algorithms/wavelet_descriptor.py` | wavelet invariants | 🟢 LOW |
| `matching/dtw.py` | DTW метрические свойства | 🔴 HIGH |
| `matching/rotation_dtw.py` | инвариантность к повороту | 🔴 HIGH |
| `matching/compat_matrix.py` | симметричность, диапазон | 🟡 MED |
| `assembly/greedy.py` | детерминированность | 🟡 MED |
| `assembly/annealing.py` | монотонность охлаждения | 🟡 MED |
| `preprocessing/*` | идемпотентность фильтров | 🟢 LOW |
| `verification/*` | корректность диапазонов score | 🟢 LOW |

#### 2.1.2 Интеграционные тесты — глубокий уровень (Pipeline class)

Существующие integration тесты тестируют отдельные модули, но НЕ тестируют `Pipeline` класс end-to-end:

| Файл | Содержание | Приоритет |
|------|-----------|-----------|
| `test_integration_pipeline_class.py` | Pipeline.preprocess → match → assemble → verify | 🔴 HIGH |
| `test_integration_assembly_methods.py` | все 8+ методов сборки в интеграции | 🔴 HIGH |
| `test_integration_error_recovery.py` | поведение при плохих входных данных | 🔴 HIGH |
| `test_integration_matching_full.py` | полный matching pipeline | 🟡 MED |
| `test_integration_verification_full.py` | все 21 валидатор в интеграции | 🟡 MED |
| `test_integration_io_export.py` | экспорт JSON/HTML/MD/PNG | 🟡 MED |

#### 2.1.3 Регрессионные тесты

Существует 1 файл `test_regression.py`. Требуется расширение:

- Фиксированные эталоны CSS/FD для стандартных контуров (квадрат, окружность, Koch-кривая)
- Воспроизводимость сборки (`greedy(seed=42)` = KNOWN)
- Эталонные значения матрицы совместимости

#### 2.1.4 Мутационное тестирование

```bash
pip install mutmut
mutmut run --paths-to-mutate puzzle_reconstruction/algorithms/fractal/
mutmut results
# Цель: mutation score > 85%
```
— Не было сделано ни разу. Покажет слабые места тестов.

---

### 2.2 Бенчмарки и профилирование — частично сделано

#### Что есть

Инфраструктура бенчмарков готова (`benchmarks/` — 7 файлов):
- `bench_assembly_methods.py` — 8 методов × 3 размера
- `bench_compat_matrix.py` — матрица совместимости
- `bench_descriptors.py` — CSS, FD, Tangram дескрипторы
- `bench_pipeline_e2e.py` — полный pipeline
- `bench_preprocessing.py` — цепочка препроцессинга
- `bench_verification.py` — 21 валидатор
- `bench_memory.py` — потребление памяти

#### Что НЕ сделано

| Задача | Описание | Приоритет |
|--------|---------|-----------|
| Запустить бенчмарки | `pytest benchmarks/ -v -s` и зафиксировать baseline | 🔴 HIGH |
| Профилирование cProfile | Найти узкие места pipeline | 🔴 HIGH |
| Scalability tests | N=4, 9, 16, 25, 36, 64 фрагмента — время и память | 🔴 HIGH |
| HTML-отчёт бенчмарков | `tools/benchmark.py --plot` → html | 🟡 MED |
| Memory profiling | `benchmarks/bench_memory.py` + memory-profiler | 🟡 MED |
| line_profiler | Горячие пути в `compat_matrix.py`, `css.py`, `dtw.py` | 🟡 MED |
| GPU оптимизация | CuPy для матричных операций (если доступен) | 🟢 LOW |
| Сравнительный отчёт | 8 методов × 4/9/16/25 фрагментов → benchmark.html | 🟡 MED |

---

### 2.3 UI и форматы экспорта — минимальная реализация

#### Что есть

- `ui/viewer.py` — базовый OpenCV-вьювер (~364 строки)
- `export.py` — PNG, PDF, heatmap, mosaic
- `tools/server.py` — Flask REST API (6 endpoint'ов, 310 строк)

#### Что НЕ сделано

| Задача | Описание | Приоритет |
|--------|---------|-----------|
| SVG экспорт | `viewer.export_svg(assembly, path)` | 🟡 MED |
| Тепловая карта матрицы | `show_compat_matrix_heatmap(matrix)` | 🟡 MED |
| Анимация сборки | `show_assembly_animated(history)` | 🟢 LOW |
| Side-by-side сравнение краёв | `show_edge_comparison(e1, e2)` | 🟡 MED |
| Визуализация верификации | `show_verification_report(report)` | 🟡 MED |
| Web WebSocket прогресс | WS `/api/v1/reconstruct/stream` | 🟢 LOW |
| `main.py interactive` | интерактивный режим CLI | 🟢 LOW |
| `main.py compare` | сравнение методов → HTML | 🟡 MED |
| `main.py benchmark` | бенчмарк из CLI | 🟡 MED |
| Форматы экспорта | `--export svg,pdf,json` | 🟡 MED |

---

### 2.4 Автоматизация и CI/CD — частично сделано

#### Что есть

- `.github/workflows/ci.yml` — CI pipeline (unit tests, mypy, ruff)
- `Makefile` — локальные команды
- `Dockerfile` / `docker-compose.yml`

#### Что НЕ сделано

| Задача | Описание | Приоритет |
|--------|---------|-----------|
| Бенчмарки в CI | Отдельный job `benchmarks` на main-ветке | 🟡 MED |
| benchmark-action | Хранение baseline + сравнение PR vs main | 🟡 MED |
| Мутационное тестирование в CI | `mutmut run` на изменённых файлах PR | 🟢 LOW |
| macOS/Windows CI | Матрица ОС (сейчас только Linux) | 🟢 LOW |
| Auto-release | Tag → GitHub Release → CHANGELOG автоматически | 🟢 LOW |
| Coverage badge | Codecov интеграция | 🟢 LOW |
| Dependency security | `pip-audit` / Dependabot | 🟢 LOW |

---

### 2.5 Новые алгоритмы и функциональность

#### Что реализовано но не протестировано глубоко

- `matching/rotation_dtw.py` — DTW с перебором поворотов (unit-тесты есть, property нет)
- `matching/text_flow.py` — OCR + Smith-Waterman (unit-тесты есть)
- `matching/spectral_matcher.py` — спектральное совпадение (unit-тесты есть)
- `assembly/astar.py` — A* сборщик (unit-тесты есть, интеграция нет)
- `assembly/hierarchical.py` — иерархическая сборка (unit-тесты есть)
- `assembly/rl_agent.py` — RL-агент (unit-тесты есть)
- `verification/homography_verifier.py` — гомография (unit-тесты есть)
- `verification/color_continuity_verifier.py` — цветовая непрерывность (unit есть)
- `verification/statistical_coherence.py` — статистическая согласованность (unit есть)

---

## РАЗДЕЛ 3: ПРИОРИТЕТНЫЙ ПЛАН РАБОТ (ПОЭТАПНО)

### ЭТАП 1 — Интеграционные тесты Pipeline (ПЕРВЫЙ ПРИОРИТЕТ)

**Зачем:** `pipeline.py` — центральный класс (6 этапов) — не покрыт глубокими интеграционными тестами. Это главный риск.

**Что создать:**

#### 1.1 `tests/test_integration_pipeline_class.py` (~80–100 тестов)

```python
# Структура:

class TestPipelinePreprocess:
    def test_preprocess_returns_fragments(self):
        """Pipeline.preprocess(images) → List[Fragment], len = n_images"""
    def test_each_fragment_has_mask_contour_edges(self):
        """Каждый фрагмент: mask, contour, edges не None"""
    def test_preprocess_4_fragments(self):
        """4 синтетических изображения → 4 фрагмента"""
    def test_preprocess_empty_raises(self):
        """Пустой список → ValueError"""

class TestPipelineMatch:
    def test_match_returns_compat_matrix(self):
        """Pipeline.match(fragments) → N×N numpy array, range [0,1]"""
    def test_match_matrix_symmetric(self):
        """compat_matrix[i,j] == compat_matrix[j,i]"""
    def test_match_diagonal_near_one(self):
        """compat_matrix[i,i] == 1.0 для всех i"""

class TestPipelineAssemble:
    def test_assemble_greedy(self):
        """Pipeline.assemble(matrix, 'greedy') → Assembly"""
    def test_assemble_beam(self):
        """Pipeline.assemble(matrix, 'beam') → Assembly"""
    def test_assemble_all_methods_parametric(self):
        """Параметрически: все 8 методов"""
    @pytest.mark.parametrize("method", ["greedy","beam","sa","genetic","ant_colony","mcts"])
    def test_method_returns_valid_assembly(self, method):
        """Все методы возвращают Assembly с placements"""

class TestPipelineVerify:
    def test_verify_default_validators(self):
        """Pipeline.verify(assembly) запускает дефолтные валидаторы"""
    def test_verify_all_21_validators(self):
        """Pipeline.verify(assembly, validators='all') → VerificationReport"""
    def test_verify_report_serializable(self):
        """report.to_json(), to_markdown(), to_html() — без ошибок"""

class TestPipelineRunFull:
    def test_run_full_pipeline_4_pieces(self):
        """Pipeline.run(images) → PipelineResult за 1 вызов"""
    def test_run_with_callbacks(self):
        """Колбэки preprocess_done, match_done вызываются"""
    def test_result_summary(self):
        """PipelineResult.summary() возвращает строку"""
```

**Как делать:**
1. Читать `puzzle_reconstruction/pipeline.py` полностью
2. Создать фикстуры `four_fragment_images`, `nine_fragment_images` в `conftest.py`
3. Писать тест-классы последовательно
4. Запускать после каждого класса: `pytest tests/test_integration_pipeline_class.py -v`

---

#### 1.2 `tests/test_integration_assembly_methods.py` (~40–50 тестов)

```python
@pytest.mark.parametrize("method,n_pieces", [
    ("greedy", 4), ("greedy", 9), ("greedy", 16),
    ("beam", 4), ("beam", 9),
    ("sa", 4),
    ("genetic", 4),
    ("ant_colony", 4),
    ("mcts", 4),
    ("astar", 4),
    ("hierarchical", 4),
])
def test_method_n_pieces(method, n_pieces):
    """Все методы работают для разного числа фрагментов"""

class TestAssemblyConsistency:
    def test_greedy_deterministic(self):
        """greedy(seed=42) всегда даёт одинаковый результат"""
    def test_all_placements_unique(self):
        """В Assembly нет двух фрагментов в одной позиции"""
    def test_assembly_covers_all_pieces(self):
        """Все N фрагментов присутствуют в Assembly"""
```

---

#### 1.3 `tests/test_integration_error_recovery.py` (~25–30 тестов)

```python
class TestEdgeCases:
    def test_single_fragment(self):
        """1 фрагмент → Assembly с 1 размещением"""
    def test_two_fragments(self):
        """2 фрагмента → trivial assembly"""
    def test_all_black_images(self):
        """Чёрные изображения обрабатываются без краша"""
    def test_very_small_images(self):
        """8×8 пикселей → не падает"""
    def test_wrong_config_values(self):
        """Неверные параметры конфига → ValueError, не краш"""
    def test_pipeline_with_timeout(self):
        """Timeout config прерывает долгую операцию"""
```

---

### ЭТАП 2 — Запуск и валидация бенчмарков (ВТОРОЙ ПРИОРИТЕТ)

**Зачем:** Инфраструктура готова, но бенчмарки ни разу не запускались с фиксацией результатов. Нужно получить baseline.

**Шаги:**

```bash
# 2.1 Запустить все бенчмарки и сохранить результаты
python3 -m pytest benchmarks/bench_descriptors.py -v -s 2>&1 | tee benchmarks/results/descriptors.txt
python3 -m pytest benchmarks/bench_compat_matrix.py -v -s 2>&1 | tee benchmarks/results/compat_matrix.txt
python3 -m pytest benchmarks/bench_assembly_methods.py -v -s 2>&1 | tee benchmarks/results/assembly.txt
python3 -m pytest benchmarks/bench_pipeline_e2e.py -v -s 2>&1 | tee benchmarks/results/pipeline_e2e.txt
python3 -m pytest benchmarks/bench_preprocessing.py -v -s 2>&1 | tee benchmarks/results/preprocessing.txt
python3 -m pytest benchmarks/bench_verification.py -v -s 2>&1 | tee benchmarks/results/verification.txt
python3 -m pytest benchmarks/bench_memory.py -v -s 2>&1 | tee benchmarks/results/memory.txt

# 2.2 Профилирование основного пути
python3 -m cProfile -o benchmarks/results/profile_4pieces.prof \
    -c "from puzzle_reconstruction.config import Config; from puzzle_reconstruction.pipeline import Pipeline; ..."

# 2.3 Изучить профиль
python3 -c "
import pstats, io
p = pstats.Stats('benchmarks/results/profile_4pieces.prof')
p.sort_stats('cumulative').print_stats(20)
"
```

**Что создать:**

#### 2.1 `benchmarks/results/BASELINE.md` — baseline результаты

Заполнить таблицу с первыми измерениями:

| Метод | N=4 (ms) | N=9 (ms) | N=16 (ms) | RAM (MB) |
|-------|----------|----------|-----------|---------|
| greedy | ? | ? | ? | ? |
| beam | ? | ? | ? | ? |
| sa | ? | ? | ? | ? |
| genetic | ? | ? | ? | ? |
| ant_colony | ? | ? | ? | ? |
| mcts | ? | ? | ? | ? |

#### 2.2 Scalability test в `benchmarks/bench_scalability.py`

```python
SIZES = [4, 9, 16, 25, 36]  # N фрагментов
METHODS = ["greedy", "beam", "sa"]

@pytest.mark.parametrize("n,method", [(n, m) for n in SIZES for m in METHODS])
def test_scalability(n, method):
    """Время и RAM для N=4..36"""
    images = make_synthetic_images(n)
    t0 = time.perf_counter()
    run_pipeline(images, method)
    elapsed = time.perf_counter() - t0
    # Записать в results/scalability.csv
```

---

### ЭТАП 3 — Property-based тесты для алгоритмов (ТРЕТИЙ ПРИОРИТЕТ)

**Зачем:** 31 файл property-тестов уже покрывает `utils/`. Нужны аналогичные тесты для алгоритмов и матчеров — они содержат сложную математику с инвариантами.

**Что создать:**

#### 3.1 `tests/test_properties_algorithms_fractal.py`

```python
from hypothesis import given, settings, strategies as st
import numpy as np

class TestBoxCountingProperties:
    @given(st.integers(min_value=10, max_value=500))
    def test_fd_always_in_range(self, n_pts):
        """FD ∈ [1.0, 2.0] для любого числа точек контура"""

    def test_fd_circle_near_1(self):
        """FD окружности ≈ 1.0 (гладкая кривая)"""

    def test_fd_monotone_with_roughness(self):
        """Более шероховатый контур → больший FD"""

class TestCSSProperties:
    @given(st.floats(min_value=0.5, max_value=5.0))
    def test_css_scale_invariant(self, scale):
        """css_similarity(scale*C, C) > 0.95 — масштаб-инвариантность"""

    @given(st.integers(min_value=10, max_value=300))
    def test_css_vector_unit_norm(self, n_pts):
        """||css_to_feature_vector(css(C))|| ≈ 1"""

class TestTangramProperties:
    @given(st.integers(min_value=3, max_value=20))
    def test_convex_hull_subset(self, n_pts):
        """Вершины выпуклой оболочки ⊆ входным точкам"""

    def test_normalize_centroid_zero(self):
        """После normalize_polygon: centroid == (0, 0)"""
```

#### 3.2 `tests/test_properties_matching.py`

```python
class TestDTWProperties:
    def test_dtw_zero_self_distance(self):
        """DTW(A, A) == 0.0"""
    def test_dtw_symmetric(self):
        """DTW(A, B) == DTW(B, A)"""
    def test_dtw_triangle_inequality(self):
        """DTW(A, C) <= DTW(A, B) + DTW(B, C)"""

class TestRotationDTWProperties:
    @given(st.floats(min_value=0, max_value=360))
    def test_rotation_invariant(self, angle_deg):
        """rotation_dtw(C, rotate(C, angle)) ≈ 0.0"""

class TestCompatMatrixProperties:
    def test_matrix_symmetric(self):
        """compat_matrix[i,j] == compat_matrix[j,i]"""
    def test_matrix_diagonal_one(self):
        """compat_matrix[i,i] == 1.0"""
    def test_matrix_values_in_range(self):
        """Все значения ∈ [0, 1]"""
```

---

### ЭТАП 4 — Улучшение UI и форматов экспорта (ЧЕТВЁРТЫЙ ПРИОРИТЕТ)

**Зачем:** `ui/viewer.py` (364 строки) — минимальный. Нужны визуализации для анализа результатов.

#### 4.1 Расширить `puzzle_reconstruction/ui/viewer.py`

```python
class AssemblyViewer:
    # Добавить методы:

    def show_compat_matrix_heatmap(self, matrix: np.ndarray,
                                    fragment_ids: list[str]) -> None:
        """matplotlib: тепловая карта матрицы совместимости N×N"""

    def show_edge_comparison(self, e1: EdgeSignature,
                              e2: EdgeSignature,
                              score: float) -> None:
        """Side-by-side: CSS профиль, FD, IFS для двух краёв"""

    def show_assembly_animated(self, history: list[Assembly],
                                fps: int = 10) -> None:
        """matplotlib animation: эволюция сборки по итерациям SA/genetic"""

    def show_verification_report(self, report: VerificationReport) -> None:
        """Зелёный/красный статус каждого из 21 валидатора"""

    def export_svg(self, assembly: Assembly, path: str) -> None:
        """SVG: фрагменты в их финальных позициях с границами"""
```

#### 4.2 Расширить `puzzle_reconstruction/export.py`

```python
def export_svg(assembly: Assembly, path: str) -> None:
    """SVG экспорт финальной сборки (без зависимостей от matplotlib)"""

def export_comparison_html(results: dict[str, Assembly],
                            path: str) -> None:
    """HTML: сравнение результатов разных методов сборки"""
```

#### 4.3 Расширить CLI `main.py`

```bash
# Новые режимы:
python main.py compare --input *.png --methods all --output compare.html
python main.py benchmark --pieces 4 9 16 --trials 3 --plot
python main.py preprocess --input *.png --output features.pkl
python main.py reconstruct *.png --export svg,pdf,json
```

---

### ЭТАП 5 — Автоматизация CI/CD (ПЯТЫЙ ПРИОРИТЕТ)

**Зачем:** Текущий CI запускает только unit-тесты. Нужно автоматизировать полный цикл.

#### 5.1 Обновить `.github/workflows/ci.yml`

Добавить job'ы:

```yaml
jobs:
  unit-tests:         # уже есть — pytest tests/ -q
  integration-tests:  # NEW: pytest tests/test_integration*.py
  property-tests:     # NEW: pytest tests/test_properties*.py
  benchmarks:         # NEW: только на main, с хранением baseline
  mypy:               # уже есть
  mutation-testing:   # NEW: mutmut на changed files в PR
```

#### 5.2 Добавить `benchmarks/benchmark_action.yml`

```yaml
# Сохранение baseline и сравнение с PR
- uses: benchmark-action/github-action-benchmark@v1
  with:
    tool: 'customSmallerIsBetter'
    output-file-path: benchmarks/results/benchmark.json
    alert-threshold: '120%'    # алерт если регресс > 20%
    comment-on-alert: true
```

---

### ЭТАП 6 — Мутационное тестирование (ШЕСТОЙ ПРИОРИТЕТ)

**Зачем:** Проверить, действительно ли тесты обнаруживают баги (mutation score).

```bash
# Установить
pip install mutmut

# Запустить на критических модулях
mutmut run --paths-to-mutate puzzle_reconstruction/algorithms/fractal/
mutmut run --paths-to-mutate puzzle_reconstruction/matching/dtw.py
mutmut run --paths-to-mutate puzzle_reconstruction/assembly/greedy.py

# Просмотреть результаты
mutmut results
mutmut html  # → .mutmut-html/

# Цель: mutation score > 85%
# Если тест проходит при мутированном коде → усилить assertion'ы
```

---

## РАЗДЕЛ 4: ПОЛНАЯ МАТРИЦА ПРИОРИТЕТОВ

| # | Задача | Ценность | Сложность | Приоритет | Этап |
|---|--------|----------|-----------|-----------|------|
| 1 | `test_integration_pipeline_class.py` | ⭐⭐⭐ | Низкая | 🔴 P1 | 1 |
| 2 | `test_integration_assembly_methods.py` | ⭐⭐⭐ | Низкая | 🔴 P1 | 1 |
| 3 | `test_integration_error_recovery.py` | ⭐⭐⭐ | Низкая | 🔴 P1 | 1 |
| 4 | Запустить все бенчмарки + зафиксировать baseline | ⭐⭐⭐ | Низкая | 🔴 P1 | 2 |
| 5 | cProfile горячих путей pipeline | ⭐⭐⭐ | Низкая | 🔴 P1 | 2 |
| 6 | `benchmarks/bench_scalability.py` | ⭐⭐⭐ | Средняя | 🔴 P1 | 2 |
| 7 | `test_properties_algorithms_fractal.py` | ⭐⭐⭐ | Низкая | 🔴 P1 | 3 |
| 8 | `test_properties_matching.py` (DTW, compat) | ⭐⭐⭐ | Низкая | 🔴 P1 | 3 |
| 9 | `test_integration_matching_full.py` | ⭐⭐ | Средняя | 🟡 P2 | 1 |
| 10 | `test_integration_verification_full.py` | ⭐⭐ | Средняя | 🟡 P2 | 1 |
| 11 | `test_integration_io_export.py` | ⭐⭐ | Низкая | 🟡 P2 | 1 |
| 12 | Расширить `test_regression.py` | ⭐⭐ | Низкая | 🟡 P2 | 3 |
| 13 | `ui/viewer.py` — heatmap, edge comparison | ⭐⭐ | Средняя | 🟡 P2 | 4 |
| 14 | `export.py` — SVG формат | ⭐⭐ | Средняя | 🟡 P2 | 4 |
| 15 | `main.py compare` команда | ⭐⭐ | Средняя | 🟡 P2 | 4 |
| 16 | CI: integration-tests job | ⭐⭐ | Низкая | 🟡 P2 | 5 |
| 17 | CI: benchmarks job + baseline | ⭐⭐ | Средняя | 🟡 P2 | 5 |
| 18 | `mutmut` — мутационное тестирование | ⭐⭐ | Средняя | 🟡 P2 | 6 |
| 19 | `test_properties_assembly.py` | ⭐ | Низкая | 🟢 P3 | 3 |
| 20 | `ui/viewer.py` — анимация сборки | ⭐ | Высокая | 🟢 P3 | 4 |
| 21 | Web API WebSocket прогресс | ⭐ | Высокая | 🟢 P3 | 4 |
| 22 | CI: mutation testing job | ⭐ | Высокая | 🟢 P3 | 5 |
| 23 | GPU оптимизация (CuPy) | ⭐ | Очень высокая | 🟢 P3 | — |
| 24 | RL-агент — обучение на реальных данных | ⭐ | Очень высокая | 🟢 P3 | — |
| 25 | Реальный датасет документов | ⭐ | Очень высокая | 🟢 P3 | — |

---

## РАЗДЕЛ 5: КОНКРЕТНЫЕ КОМАНДЫ ДЛЯ СТАРТА

### Немедленно выполнить в новом чате (копировать как есть)

```bash
# === ШАГ 1: Ориентация ===
cd /home/user/meta2
git status
git log --oneline -3
python3 -m pytest --co -q 2>/dev/null | tail -2

# === ШАГ 2: Запустить существующие интеграционные тесты ===
python3 -m pytest tests/test_integration*.py -v --tb=short 2>&1 | tail -20

# === ШАГ 3: Запустить бенчмарки (получить baseline) ===
python3 -m pytest benchmarks/ -v -s 2>&1 | tee benchmarks/results/run_$(date +%Y%m%d).txt

# === ШАГ 4: Создать первый новый тест ===
# Создать tests/test_integration_pipeline_class.py
# Начать с TestPipelinePreprocess
python3 -m pytest tests/test_integration_pipeline_class.py -v

# === ШАГ 5: Коммит ===
git add tests/test_integration_pipeline_class.py
git commit -m "tests: add deep integration tests for Pipeline class"
git push -u origin claude/puzzle-text-docs-3tcRj
```

---

## РАЗДЕЛ 6: СПРАВОЧНИК — КЛЮЧЕВЫЕ API

### Pipeline (pipeline.py)

```python
from puzzle_reconstruction.pipeline import Pipeline
from puzzle_reconstruction.config import Config

config = Config()
pipeline = Pipeline(config)

# Отдельные этапы:
fragments = pipeline.preprocess(images)         # List[np.ndarray] → List[Fragment]
matrix = pipeline.match(fragments)              # List[Fragment] → np.ndarray (N×N)
assembly = pipeline.assemble(matrix, "beam")    # → Assembly
report = pipeline.verify(assembly)              # → VerificationReport

# Или весь pipeline за 1 вызов:
result = pipeline.run(images)                   # → PipelineResult
print(result.summary())
```

### Assembly методы

```python
from puzzle_reconstruction.assembly.greedy import greedy_assembly
from puzzle_reconstruction.assembly.beam_search import beam_search_assembly
from puzzle_reconstruction.assembly.annealing import sa_assembly
from puzzle_reconstruction.assembly.genetic import genetic_assembly
from puzzle_reconstruction.assembly.ant_colony import ant_colony_assembly
from puzzle_reconstruction.assembly.mcts import mcts_assembly
from puzzle_reconstruction.assembly.astar import astar_assembly
from puzzle_reconstruction.assembly.parallel import run_all_methods

# Параметрически (для тестов):
import importlib
mod = importlib.import_module(f"puzzle_reconstruction.assembly.{method}")
fn = getattr(mod, f"{method}_assembly")
assembly = fn(compat_matrix, config=AssemblyConfig())
```

### Верификация

```python
from puzzle_reconstruction.verification.suite import VerificationSuite

suite = VerificationSuite()
report = suite.run_all(assembly)           # все 21 валидатор
report = suite.run(assembly, ["ocr", "seam", "overlap"])  # подмножество

print(report.to_json())
print(report.to_markdown())
print(report.to_html())

names = suite.all_validator_names()        # список 21 имён
```

### Синтетические данные для тестов (conftest)

```python
import numpy as np
from puzzle_reconstruction.models import Fragment, EdgeSignature

def make_synthetic_fragment(w=64, h=64) -> Fragment:
    image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    mask = np.ones((h, w), dtype=np.uint8)
    # ... заполнить поля Fragment
    return fragment

def make_n_fragments(n: int) -> list[Fragment]:
    return [make_synthetic_fragment() for _ in range(n)]
```

---

## РАЗДЕЛ 7: ИСТОРИЯ РАЗРАБОТКИ

| Дата | Версия | Ключевые события |
|------|--------|-----------------|
| 20 фев 2026 | 0.1.0 | Первый коммит, базовая структура |
| 21–22 фев | 0.2.0 | 8 алгоритмов сборки, 13+ матчеров |
| 23–24 фев | 0.3.0 | preprocessing (38 модулей), utils (131 модуль) |
| 25 фев | 0.4.0b1 | 12 фаз интеграции, 21 валидатор, v1.0.0 beta |
| 25 фев | **1.0.0** | Production/Stable, тег v1.0.0, 42 208 тестов |
| 26 фев | 1.0.0+ | Интеграционные тесты low-coverage, ROADMAP |
| 27 фев | 1.0.0+ | Property-based тесты: 31 файл, ~2 700 тестов |
| 28 фев | 1.0.0+ | **57 624 тестов**. Этот план. Следующий этап: ↑ |

---

## РАЗДЕЛ 8: ЧЕГО НЕ ТРОГАТЬ (СТАБИЛЬНЫЕ ЧАСТИ)

Следующие части **не нужно переписывать или рефакторить**:

- ✅ `pipeline.py` — стабилен, только добавлять тесты
- ✅ `config.py` — стабилен, только добавлять параметры по необходимости
- ✅ `models.py` — стабилен, только добавлять методы
- ✅ `algorithms/fractal/` — математика выверена, property-тестами проверить
- ✅ `matching/compat_matrix.py` — core логика стабильна
- ✅ `verification/suite.py` — 21 валидатор зарегистрированы, API стабилен
- ✅ `assembly/*.py` — все 8+ алгоритмов работают
- ✅ `utils/` — 131 модуль, тщательно протестирован

**Можно улучшать:**
- `ui/viewer.py` — добавлять методы визуализации
- `export.py` — добавлять форматы
- `tools/server.py` — добавлять endpoints
- `main.py` — добавлять команды CLI
- `.github/workflows/ci.yml` — добавлять job'ы

---

*Документ создан: 2026-02-28*
*Проект: puzzle-reconstruction v1.0.0*
*57 624 тестов · 449 коммитов · 305 production-модулей*
*Следующий шаг: Этап 1 — `tests/test_integration_pipeline_class.py`*
