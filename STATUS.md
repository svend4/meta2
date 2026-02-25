# STATUS.md — Текущий статус реализации `puzzle_reconstruction`

> **Дата:** 2026-02-25
> **Ветка:** `claude/dev-status-documentation-j7YCO`
> **Версия проекта:** 0.3.0-alpha
> **Python:** 3.11.14
> **Последний коммит:** `d2fd92b` — docs: add detailed implementation status document
> **Тестовый запуск:** 2026-02-25, 3 мин 37 сек, все зависимости установлены

---

## Содержание

1. [Краткая сводка — один экран](#1-краткая-сводка)
2. [Описание проекта и алгоритм](#2-описание-проекта)
3. [Актуальные метрики кодовой базы](#3-актуальные-метрики-кодовой-базы)
4. [Результаты тестирования — ЖИВЫЕ данные](#4-результаты-тестирования)
5. [Архитектура системы](#5-архитектура-системы)
6. [API — классы, модели, конфигурация](#6-api--классы-модели-конфигурация)
7. [CLI — интерфейс командной строки](#7-cli--интерфейс-командной-строки)
8. [Статус алгоритмов сборки (8/8 активны)](#8-алгоритмы-сборки)
9. [Статус матчеров совместимости (15 зарегистрировано)](#9-матчеры-совместимости)
10. [Статус предобработки](#10-предобработка)
11. [Статус верификации](#11-верификация)
12. [Инфраструктура и утилиты](#12-инфраструктура-и-утилиты)
13. [Вспомогательные инструменты (tools/)](#13-вспомогательные-инструменты-tools)
14. [Технический стек](#14-технический-стек)
15. [История разработки](#15-история-разработки)
16. [Единственный провалившийся тест](#16-единственный-провалившийся-тест)
17. [Дорожная карта и что осталось](#17-дорожная-карта)
18. [Итоговая оценка готовности](#18-итоговая-оценка-готовности)

---

## 1. Краткая сводка

| Параметр | Значение |
|---|---|
| **Стадия разработки** | Alpha (v0.3.0) |
| **Исходных модулей** | 305 `.py` файлов |
| **Строк исходного кода** | 93 785 (только `puzzle_reconstruction/`) |
| **Строк с учётом `main.py` и `tools/`** | ~95 718 |
| **Тестовых файлов** | 824 (337 базовых + 485 `_extra`) |
| **Строк тестового кода** | 267 362 |
| **Всего тестов (живой запуск)** | **42 280** |
| **Тестов пройдено** | **42 270 (99.976%)** |
| **Тестов провалено** | **1 (0.002%)** — единственный баг |
| **Тестов xpassed** | 9 (были помечены как xfail, теперь проходят) |
| **Алгоритмов сборки** | **8/8 активны в CLI** |
| **Матчеров** | **15 зарегистрировано** в `matcher_registry` |
| **Коммитов** | 260 |
| **Время тестового прогона** | 3 мин 37 сек |

> **Ключевое изменение vs REPORT.md (24 февраля):**
> Коммит `6c98327` (feat: integrate all 8 assembly algorithms and add matcher registry)
> + три последующих `fix:` коммита снизили число провалов с **133** до **1**.

---

## 2. Описание проекта

`puzzle_reconstruction` — система **автоматической реконструкции разорванных бумажных документов** (газет, книг, архивных материалов) из отсканированных фрагментов. Реализует задачу, аналогичную пазлу, методами компьютерного зрения и комбинаторной оптимизации.

### Ключевая идея — двойной дескриптор краёв

```
         ВНУТРИ фрагмента                   СНАРУЖИ фрагмента
                │                                    │
       Танграм-контур                        Фрактальная кривая
   (геометрически правильный)            (форма «береговой линии»)
                └──────────────┬──────────────────────┘
                               │  СИНТЕЗ
                          EdgeSignature
                      (уникальная подпись края)

       B_virtual = α · B_tangram + (1-α) · B_fractal
                          ↑
               параметр --alpha (default 0.5)
```

| Дескриптор | Метод | Описывает |
|---|---|---|
| **Танграм** | Convex hull → RDP → нормализация | Крупная геометрия контура |
| **Box-counting** | Фрактальная размерность (Минковский) | Детали разрыва |
| **Divider** | Метод Ричардсона | Самоподобие кромки |
| **IFS Барнсли** | Фрактальная интерполяция | Аттрактор кромки |
| **CSS** | Curvature Scale Space (MPEG-7) | Кривизна на разных масштабах |

### Шесть этапов пайплайна

```
Этап 1  Загрузка          →  load_fragments()       — чтение PNG/JPEG/TIFF
Этап 2  Предобработка     →  process_fragment()      — сегм. + контур + ориентация
Этап 3  Дескрипторы       →  build_edge_signatures() — Танграм + Фракталы
Этап 4  Матрица           →  build_compat_matrix()   — N×N попарных оценок
Этап 5  Сборка            →  assemble()              — выбранный алгоритм
Этап 6  Верификация       →  verify_full_assembly()  — OCR-связность + экспорт
```

---

## 3. Актуальные метрики кодовой базы

### По подпакетам (живые данные, 2026-02-25)

| Подпакет | Файлов | Строк кода | Среднее строк/файл |
|---|---|---|---|
| `utils/` | 130 | 38 970 | 300 |
| `algorithms/` | 42 | 12 567 | 299 |
| `preprocessing/` | 38 | 11 662 | 307 |
| `assembly/` | 27 | 8 145 | 302 |
| `matching/` | 27 | 8 188 | 303 |
| `verification/` | 21 | 7 442 | 354 |
| `scoring/` ⚠️ спит | 12 | 4 219 | 352 |
| `io/` ⚠️ спит | 3 | 1 141 | 380 |
| `ui/` | 1 | 364 | 364 |
| *корневые (pipeline, config, models, export, clustering)* | 5 | ~1 471 | ~294 |
| **ИТОГО `puzzle_reconstruction/`** | **306** | **93 785** | **306** |
| `main.py` | 1 | 384 | — |
| `tools/*.py` | 6 | ~1 943 | ~324 |
| **ИТОГО проект** | **313** | **~96 112** | — |

> Примечание: `algorithms/` включает подпакеты `fractal/` (4 модуля) и `tangram/` (3 модуля).

### Тестовая база

| Тип | Файлов | Строк |
|---|---|---|
| Базовые тесты (`test_*.py`) | 337 | ~117 000 |
| Расширенные тесты (`test_*_extra.py`) | 485 | ~150 000 |
| `conftest.py` | 1 | ~400 |
| **Итого** | **824** | **267 362** |

### Ключевые соотношения

| Метрика | Значение |
|---|---|
| Тестовый код / исходный код | **2.85×** |
| Тестовых файлов / исходных модулей | **2.70×** |
| Тестов на исходный модуль | **~138** |
| Строк кода на модуль (исходник) | **~306** |
| Строк кода на тестовый файл | **~325** |

---

## 4. Результаты тестирования

### Живой запуск (2026-02-25, продолжительность: 3 мин 37 сек)

```
Всего собрано тестов:    42 280
─────────────────────────────────
✅  passed              42 270   (99.976%)
❌  failed                   1   (0.002%)
⚠️  xpassed                  9   (0.021%)   — были xfail, теперь проходят
🔕  warnings                81
─────────────────────────────────
```

> **Сравнение с REPORT.md (24 февраля 2026):**
>
> | Дата | Тестов | Провалов | % прохождения |
> |---|---|---|---|
> | 24 февраля | 42 217 | 133 | 99.66% |
> | 25 февраля | 42 280 | **1** | **99.976%** |
>
> Три `fix:` коммита устранили 132 провала. Добавлено 63 новых теста.

### Распределение по типам тестов

| Тип | Примерно тестов | % прохождения |
|---|---|---|
| Расширенные `_extra.py` | ~24 000 | >99.99% |
| Базовые `test_*.py` | ~18 280 | >99.95% |
| **Итого** | **42 280** | **99.976%** |

### 9 xpassed-тестов (были xfail, теперь проходят)

Тесты, помеченные `@pytest.mark.xfail`, теперь успешно проходят — код исправлен. Метка `xfail` устарела и должна быть убрана при следующем рефакторинге тестов.

---

## 5. Архитектура системы

### Полное дерево проекта

```
meta2/
├── main.py                          # Точка входа CLI (384 строки)
├── pyproject.toml                   # Сборка, зависимости, инструменты
├── requirements.txt                 # Прямые зависимости
├── README.md                        # Краткое описание проекта
├── PUZZLE_RECONSTRUCTION.md         # Полная техническая документация
├── INTEGRATION_ROADMAP.md           # Дорожная карта интеграции
├── REPORT.md                        # Отчёт о тестовом покрытии
├── STATUS.md                        # Этот файл
├── tools/                           # CLI-инструменты (6 файлов, ~1 943 строки)
│   ├── benchmark.py    (363 стр.)   # Бенчмарк всех методов сборки
│   ├── evaluate.py     (311 стр.)   # Полная оценка качества реконструкции
│   ├── mix_documents.py(271 стр.)   # Смешивание фрагментов разных документов
│   ├── profile.py      (387 стр.)   # Профилирование этапов пайплайна
│   ├── server.py       (309 стр.)   # Flask REST API сервер
│   └── tear_generator.py(303 стр.) # Генератор синтетических тестовых данных
└── puzzle_reconstruction/           # Основной пакет
    ├── __init__.py                  # Публичный API v0.3.0
    ├── pipeline.py     (~298 стр.)  # Pipeline, PipelineResult
    ├── config.py       (~340 стр.)  # Config и 6 dataclass-конфигов
    ├── models.py       (~420 стр.)  # Fragment, Assembly, EdgeSignature, ...
    ├── export.py       (~220 стр.)  # Экспорт: PNG, PDF, JSON
    ├── clustering.py   (~193 стр.)  # Кластеризация фрагментов
    ├── algorithms/                  # 42 модуля — дескрипторы и алгоритмы
    │   ├── fractal/    (4 модуля)   # box_counting, divider, ifs, css
    │   ├── tangram/    (3 модуля)   # hull, classifier, inscriber
    │   └── ...         (35 модулей) # synthesis + прочие алгоритмы
    ├── assembly/                    # 27 модулей — стратегии сборки
    │   ├── parallel.py              # ★ реестр 8 методов, run_all_methods()
    │   ├── greedy.py                # Жадный алгоритм
    │   ├── annealing.py             # Имитация отжига
    │   ├── beam_search.py           # Beam search
    │   ├── gamma_optimizer.py       # Gamma (SOTA, 2026)
    │   ├── genetic.py               # Генетический алгоритм
    │   ├── exhaustive.py            # Полный перебор
    │   ├── ant_colony.py            # Муравьиная колония (ACO)
    │   ├── mcts.py                  # Monte Carlo Tree Search
    │   └── ...         (18 модулей) # вспомогательные
    ├── matching/                    # 27 модулей — сопоставление краёв
    │   ├── matcher_registry.py      # ★ 15 зарегистрированных матчеров
    │   ├── pairwise.py              # match_score() — вычисление CompatEntry
    │   ├── compat_matrix.py         # build_compat_matrix() — матрица N×N
    │   ├── dtw.py                   # Dynamic Time Warping
    │   └── ...         (23 модуля)  # все матчеры
    ├── preprocessing/               # 38 модулей — предобработка изображений
    │   ├── segmentation.py          # ★ Otsu / Adaptive / GrabCut
    │   ├── contour.py               # ★ extract_contour, RDP
    │   ├── orientation.py           # ★ estimate_orientation, rotate_to_upright
    │   ├── color_norm.py            # ★ normalize_color
    │   └── ...         (34 модуля)  # остальная предобработка
    ├── scoring/                     # 12 модулей — оценка качества
    ├── verification/                # 21 модуль — верификация
    │   └── ocr.py                   # ★ verify_full_assembly() (Tesseract)
    ├── utils/                       # 130 модулей — инфраструктура
    │   └── logger.py                # ★ get_logger(), PipelineTimer
    ├── io/                          # 3 модуля — ввод/вывод
    └── ui/                          # 1 модуль — интерактивный просмотрщик
```
> ★ — активно используется в `main.py` / `pipeline.py`

### Поток данных

```
[PNG/JPEG фрагменты]
        │
        ▼ load_fragments()
        │  cv2.imread → grayscale normalize
        │
        ▼ process_fragment()  [параллельно, ThreadPoolExecutor]
        │  ├── segmentation  → маска фрагмента
        │  ├── contour       → контур, RDP, разбиение на края
        │  ├── orientation   → поворот к вертикали
        │  ├── tangram       → fit_tangram() → TangramSignature
        │  └── fractal       → compute_fractal_signature() → FractalSignature
        │            ↓ build_edge_signatures() → EdgeSignature[]
        │
        ▼ build_compat_matrix()
        │  ∀ пар (i,j): match_score(edge_i, edge_j)
        │    = Σ w_k · matcher_k(edge_i, edge_j)
        │    матчеры: css(0.35) + dtw(0.30) + fd(0.20) + text(0.15)
        │    (остальные 11 матчеров — конфигурируемы)
        │
        ▼ assemble()  →  parallel.py
        │  метод из: greedy | sa | beam | gamma |
        │            genetic | exhaustive | ant_colony | mcts | auto | all
        │
        ▼ verify_full_assembly()
        │  OCR-связность (Tesseract, опционально)
        │
        ▼ [result.png + metadata.json + опц. report.pdf]
```

---

## 6. API — классы, модели, конфигурация

### `puzzle_reconstruction/models.py` — модели данных

**Перечисления:**

| Enum | Значения |
|---|---|
| `ShapeClass` | TRIANGLE, RECTANGLE, TRAPEZOID, PARALLELOGRAM, PENTAGON, HEXAGON, POLYGON |
| `EdgeSide` | TOP, BOTTOM, LEFT, RIGHT, UNKNOWN |

**Dataclasses:**

| Класс | Поля | Описание |
|---|---|---|
| `FractalSignature` | fd_box, fd_divider, ifs_coeffs, css_image, chain_code, curve | Фрактальное описание края |
| `TangramSignature` | polygon, shape_class, centroid, angle, scale, area | Геометрическое описание формы |
| `EdgeSignature` | edge_id, side, virtual_curve, fd, css_vec, ifs_coeffs, length | Итоговый дескриптор края |
| `Edge` | edge_id, contour, text_hint | Физический край фрагмента |
| `Placement` | fragment_id, position, rotation | Размещение фрагмента в сборке |
| `Fragment` | fragment_id, image, mask, contour, tangram, fractal, edges, placement | Полное представление фрагмента |
| `CompatEntry` | edge_i, edge_j, score, dtw_dist, css_sim, fd_diff, text_score | Запись совместимости пары краёв |
| `Assembly` | placements, fragments, compat_matrix, total_score, ocr_score, method | Результат сборки |

### `puzzle_reconstruction/config.py` — конфигурация

Все конфиги — `@dataclass` с валидацией в `__post_init__`.

| Класс | Ключевые поля |
|---|---|
| `SegmentationConfig` | method (otsu/adaptive/grabcut), morph_kernel |
| `SynthesisConfig` | alpha (0..1), n_sides, n_points |
| `FractalConfig` | n_scales, ifs_transforms, css_n_sigmas, css_n_bins |
| `MatchingConfig` | threshold, dtw_window, **active_matchers**, **matcher_weights**, combine_method |
| `AssemblyConfig` | **method** (10 вариантов), beam_width, sa_iter, mcts_sim, genetic_pop, genetic_gen, aco_ants, aco_iter, auto_timeout |
| `VerificationConfig` | run_ocr, ocr_lang, export_pdf |
| `Config` (корневой) | seg, synthesis, fractal, matching, assembly, verification |

**Методы `Config`:** `to_dict()`, `to_json()`, `from_dict()`, `from_file()`, `default()`, `apply_overrides()`

### `puzzle_reconstruction/pipeline.py` — класс Pipeline

```python
class PipelineResult:
    assembly:   Assembly
    timer:      PipelineTimer
    cfg:        Config
    n_input:    int        # фрагментов на входе
    n_placed:   int        # фрагментов размещено
    timestamp:  str
    def summary() -> str   # текстовая сводка

class Pipeline:
    def run(images: list[np.ndarray]) -> PipelineResult
    def preprocess(images) -> list[Fragment]
    def match(fragments) -> tuple[CompatMatrix, list[CompatEntry]]
    def assemble(fragments, entries) -> Assembly
    def verify(assembly: Assembly) -> float
```

### `puzzle_reconstruction/__init__.py` — публичный API v0.3.0

```python
# Модели данных
Fragment, Assembly, CompatEntry, EdgeSignature
FractalSignature, TangramSignature, ShapeClass, EdgeSide

# Конфигурация
Config

# Кластеризация
cluster_fragments, ClusteringResult, split_by_cluster

# Пайплайн
Pipeline, PipelineResult

__version__ = "0.3.0"
```

---

## 7. CLI — интерфейс командной строки

### Основная команда

```bash
puzzle-reconstruct [OPTIONS]
# или напрямую:
python main.py [OPTIONS]
```

### Все аргументы `main.py`

| Аргумент | Тип | По умолчанию | Описание |
|---|---|---|---|
| `--input, -i` | path | **обязательный** | Директория с отсканированными фрагментами |
| `--output, -o` | path | `result.png` | Путь к выходному файлу |
| `--config, -c` | path | — | JSON/YAML файл конфигурации |
| `--method, -M` | выбор | `beam` | Алгоритм сборки (10 вариантов, см. §8) |
| `--alpha` | float | — | Вес танграма в EdgeSignature (0..1) |
| `--n-sides` | int | — | Ожидаемое число краёв на фрагмент |
| `--seg-method` | выбор | — | Сегментация: `otsu` / `adaptive` / `grabcut` |
| `--threshold` | float | — | Минимальный порог совместимости краёв |
| `--beam-width` | int | — | Ширина поиска beam search |
| `--sa-iter` | int | — | Итерации имитации отжига |
| `--mcts-sim` | int | — | Симуляции MCTS |
| `--genetic-pop` | int | — | Размер популяции генетического алгоритма |
| `--genetic-gen` | int | — | Число поколений генетического алгоритма |
| `--aco-ants` | int | — | Число агентов-муравьёв (ACO) |
| `--aco-iter` | int | — | Итерации муравьиной колонии |
| `--auto-timeout` | float | — | Таймаут на метод в режиме `auto`/`all` (сек) |
| `--visualize, -v` | flag | False | Показать результат в окне OpenCV |
| `--interactive, -I` | flag | False | Открыть интерактивный редактор |
| `--verbose` | flag | False | Уровень логирования DEBUG |
| `--log-file` | path | — | Путь к файлу лога |

### Примеры использования

```bash
# Базовый запуск (beam search по умолчанию)
python main.py --input scans/ --output result.png

# Точный алгоритм для малых наборов (≤8 фрагментов)
python main.py --input scans/ --method exhaustive

# Генетический алгоритм для средних наборов
python main.py --input scans/ --method genetic --genetic-pop 50 --genetic-gen 100

# Автовыбор алгоритма по числу фрагментов
python main.py --input scans/ --method auto --auto-timeout 60

# Запуск всех 8 алгоритмов, выбор лучшего (research mode)
python main.py --input scans/ --method all --auto-timeout 120

# С кастомными матчерами через конфиг
python main.py --input scans/ --config my_config.json

# С визуализацией и интерактивным редактором
python main.py --input scans/ --output result.png --visualize --interactive
```

---

## 8. Алгоритмы сборки

### Все 8 методов — АКТИВНЫ в CLI

Начиная с коммита `6c98327` (feat: integrate all 8 assembly algorithms),
все 8 алгоритмов доступны через параметр `--method`.

| Метод | CLI-ключ | Сложность | Качество | Детерм. | Лучший сценарий |
|---|---|---|---|---|---|
| **Полный перебор** | `exhaustive` | O(N!) | ⭐⭐⭐⭐⭐ | ✅ | ≤8 фрагментов, точный результат |
| **Beam search** | `beam` | O(W·N²) | ⭐⭐⭐⭐ | ✅ | 6–20 фрагментов, нужна скорость |
| **MCTS** | `mcts` | O(S·D) | ⭐⭐⭐⭐ | ❌ | 6–25 фрагментов, сложная топология |
| **Генетический** | `genetic` | O(G·P·N²) | ⭐⭐⭐⭐ | ❌ | 15–40 фрагментов |
| **Муравьиная колония** | `ant_colony` | O(I·A·N²) | ⭐⭐⭐⭐ | ❌ | 20–60 фрагментов |
| **Gamma** | `gamma` | O(I·N²) | ⭐⭐⭐⭐⭐ | ❌ | 20–100 фрагментов, SOTA |
| **Имитация отжига** | `sa` | O(I) | ⭐⭐⭐ | ❌ | Быстрое улучшение жадного результата |
| **Жадный** | `greedy` | O(N²) | ⭐⭐ | ✅ | Baseline, инициализация |
| **Авто** | `auto` | — | — | — | Автовыбор по числу фрагментов |
| **Все** | `all` | — | — | — | Research mode: запустить все, выбрать лучший |

### `assembly/parallel.py` — реестр алгоритмов

```python
ALL_METHODS = ["greedy", "sa", "beam", "gamma",
               "genetic", "exhaustive", "ant_colony", "mcts"]

DEFAULT_METHODS = ["greedy", "sa", "beam", "genetic"]

# Ключевые функции:
run_all_methods(fragments, entries, methods, timeout, n_workers) → list[MethodResult]
run_selected(fragments, entries, methods)  → list[MethodResult]
pick_best(results)                         → Optional[Assembly]
pick_best_k(results, k)                   → list[Assembly]
summary_table(results)                     → str  # Markdown-таблица сравнения
```

### Интеллектуальный автовыбор (`--method auto`)

```python
def auto_select_method(n_fragments: int) -> list[str]:
    if n_fragments <= 4:   return ["exhaustive"]
    elif n_fragments <= 8: return ["exhaustive", "beam"]
    elif n_fragments <= 15:return ["beam", "mcts", "sa"]
    elif n_fragments <= 30:return ["genetic", "gamma", "ant_colony"]
    else:                  return ["gamma", "sa"]
```

### Research mode (`--method all`)

Запускает все 8 методов → выводит сравнительную таблицу → возвращает лучший:

```
| Method     | Score  | Time(s) | Status  |
|------------|--------|---------|---------|
| gamma      | 0.9123 | 12.34   | OK      |
| genetic    | 0.8991 | 45.21   | OK      |
| mcts       | 0.8847 | 8.76    | OK      |
| beam       | 0.8701 | 2.13    | OK      |
| ...
```

---

## 9. Матчеры совместимости

### 15 зарегистрированных матчеров

Файл `matching/matcher_registry.py` содержит 15 матчеров, зарегистрированных через декоратор `@register`:

| № | Имя | Модуль-источник | Что измеряет | Формула |
|---|---|---|---|---|
| 1 | `css` | `algorithms.fractal.css` | Curvature Scale Space сходство | `css_similarity_mirror(e_i.css_vec, e_j.css_vec)` |
| 2 | `dtw` | `matching.dtw` | Dynamic Time Warping | `1 / (1 + dtw_distance_mirror(curve_i, curve_j))` |
| 3 | `fd` | models | Фрактальная размерность | `1 / (1 + |fd_i - fd_j|)` |
| 4 | `text` | внешний сигнал | OCR связность текста | `text_score` (0.5 fallback) |
| 5 | `icp` | `matching.icp` | ICP-выравнивание кривых | `1 / (1 + icp_error)` |
| 6 | `color` | `matching.color_match` | Цветовые гистограммы | `color_match_score(e_i, e_j)` |
| 7 | `texture` | `matching.texture_match` | LBP/Gabor текстуры | `texture_match_score(e_i, e_j)` |
| 8 | `seam` | `matching.seam_score` | Непрерывность шва | `seam_quality_score(e_i, e_j)` |
| 9 | `geometric` | `matching.geometric_match` | Геометрические инварианты | `geometric_align_score(e_i, e_j)` |
| 10 | `boundary` | `matching.boundary_matcher` | Профиль границы | `boundary_shape_score(e_i, e_j)` |
| 11 | `affine` | `matching.affine_matcher` | Аффинное преобразование | `affine_transform_score(e_i, e_j)` |
| 12 | `spectral` | `matching.spectral_matcher` | Спектральные дескрипторы | `spectral_graph_score(e_i, e_j)` |
| 13 | `shape_context` | `matching.shape_matcher` | Shape Context дескриптор | `shape_context_score(e_i, e_j)` |
| 14 | `patch` | `matching.patch_matcher` | Патч-совпадение | `patch_similarity(e_i, e_j)` |
| 15 | `feature` | `matching.feature_match` | SIFT/ORB дескрипторы | `feature_match_ratio(e_i, e_j)` |

### Конфигурация по умолчанию (`pairwise.py`)

```python
# Активные матчеры и их веса по умолчанию
DEFAULT_MATCHERS = ["css", "dtw", "fd", "text"]

DEFAULT_WEIGHTS = {
    "css":  0.35,   # Curvature Scale Space
    "dtw":  0.30,   # Dynamic Time Warping
    "fd":   0.20,   # Фрактальная размерность
    "text": 0.15,   # OCR-связность
}
```

### Конфигурируемые матчеры через `MatchingConfig`

```python
cfg.matching.active_matchers  = ["css", "dtw", "fd", "text", "color", "icp"]
cfg.matching.matcher_weights  = {"css": 0.25, "dtw": 0.25, "fd": 0.15,
                                 "text": 0.10, "color": 0.15, "icp": 0.10}
cfg.matching.combine_method   = "weighted"   # или: rank | min | max
```

### Агрегация оценок — готовая инфраструктура

| Модуль | Функции |
|---|---|
| `matching/score_combiner.py` | `weighted_combine()`, `rank_combine()`, `min_combine()`, `max_combine()` |
| `matching/score_aggregator.py` | Агрегация от N матчеров → единый `CompatEntry` |
| `matching/consensus.py` | Голосование между результатами нескольких методов сборки |

---

## 10. Предобработка

### Активные модули (в пайплайне по умолчанию)

| Модуль | Вызывается из | Функция |
|---|---|---|
| `segmentation.py` | `main.py`, `pipeline.py` | Выделение маски (Otsu/Adaptive/GrabCut) |
| `contour.py` | `main.py`, `pipeline.py` | Извлечение контура, RDP-упрощение, разбиение на края |
| `orientation.py` | `main.py`, `pipeline.py` | Оценка ориентации текста, поворот фрагмента |
| `color_norm.py` | `pipeline.py` | Нормализация цвета (CLAHE + Gray World) |
| `tangram/inscriber.py` | `main.py`, `pipeline.py` | `fit_tangram()` — вписывание геометрической фигуры |
| `fractal/*.py` | `algorithms/synthesis.py` | 4 метода фрактального описания |

### Спящие модули предобработки (не подключены по умолчанию, но реализованы)

| Группа | Модули (всего 32) |
|---|---|
| **Шумоподавление** | `noise_filter`, `noise_analyzer`, `noise_reduction`, `frequency_filter` |
| **Цвет и контраст** | `contrast`, `contrast_enhancer`, `color_normalizer`, `channel_splitter` |
| **Коррекция геометрии** | `deskewer`, `skew_correction`, `perspective`, `warp_corrector` |
| **Освещение** | `illumination_corrector`, `illumination_normalizer` |
| **Морфология** | `morphology_ops`, `edge_enhancer`, `edge_sharpener` |
| **Бинаризация** | `binarizer`, `adaptive_threshold` |
| **Очистка** | `document_cleaner`, `background_remover`, `fragment_cropper` |
| **Качество** | `quality_assessor`, `frequency_analyzer` |
| **Патчи и текстура** | `patch_normalizer`, `patch_sampler`, `texture_analyzer` |
| **Аугментация** | `augment` |

**Готовая инфраструктура подключения** (из INTEGRATION_ROADMAP.md):

```yaml
# Будущий config.yaml
preprocessing:
  chain: ["quality_assessor", "denoise", "contrast", "deskewer", "binarizer"]
  quality_threshold: 0.4
  auto_enhance: true
```

---

## 11. Верификация

### Активный модуль (1 из 21)

| Модуль | Функция | Зависимость |
|---|---|---|
| `verification/ocr.py` | `verify_full_assembly()` | Tesseract (опционально; fallback = 0.5) |
| `verification/ocr.py` | `render_assembly_image()` | OpenCV |

### Спящие верификаторы (20 из 21 — реализованы и покрыты тестами)

| Модуль | Что проверяет | Метрика |
|---|---|---|
| `metrics.py` | IoU, Kendall τ, RMSE позиций, угловая ошибка | Количественные метрики |
| `text_coherence.py` | Связность текста (N-gram модель, bigrams) | Семантическая корректность |
| `confidence_scorer.py` | Уверенность в каждом размещении | Per-fragment [0..1] |
| `consistency_checker.py` | Глобальная согласованность сборки | bool + score |
| `layout_checker.py` | Gap uniformity, column/row alignment | Геометрические метрики |
| `overlap_checker.py` | Пересечения между фрагментами | IoU пересечений |
| `seam_analyzer.py` | Gradient continuity через швы | Визуальная непрерывность |
| `boundary_validator.py` | Корректность граничных условий | bool |
| `fragment_validator.py` | Валидность каждого фрагмента до сборки | bool |
| `assembly_scorer.py` | Суммарный score всей сборки | A–F оценка |
| `completeness_checker.py` | Все фрагменты размещены? | % покрытия |
| `overlap_validator.py` | Детальная проверка перекрытий | float |
| `spatial_validator.py` | Пространственные связи, топология | bool |
| `placement_validator.py` | Корректность каждого Placement | bool |
| `layout_scorer.py` | Оценка 2D-компоновки | float [0..1] |
| `score_reporter.py` | Формирование отчёта по оценкам | dict |
| `edge_validator.py` | Совместимость краёв на стыках | per-seam score |
| `quality_reporter.py` | Полный качественный отчёт | JSON/Markdown |
| `layout_verifier.py` | Итоговая верификация компоновки | bool |
| `report.py` | Генерация финального отчёта | str/PDF |

---

## 12. Инфраструктура и утилиты

### `utils/` — 130 модулей, 38 970 строк

**Активные утилиты** (используются в `main.py` / `pipeline.py`):

| Модуль | Используется | Функция |
|---|---|---|
| `logger.py` | `main.py`, `pipeline.py` | `get_logger()`, `stage()`, `PipelineTimer` |
| `event_bus.py` | `pipeline.py` | `EventBus`, `make_event_bus()` — pub/sub прогресс-события |

**Готовая инфраструктура** (не используется напрямую, но реализована):

| Категория | Ключевые модули |
|---|---|
| **Пайплайн** | `pipeline_runner`, `batch_processor`, `progress_tracker`, `config_manager`, `result_cache`, `cache_manager`, `metric_tracker`, `profiler` |
| **Геометрия** | `geometry_utils`, `transform_utils`, `rotation_utils`, `icp_utils`, `polygon_ops_utils`, `bbox_utils`, `distance_utils`, `contour_utils` |
| **Изображения** | `image_io`, `image_transform_utils`, `image_pipeline_utils`, `mask_layout_utils`, `patch_extractor`, `patch_score_utils` |
| **Граф** | `graph_utils`, `graph_cache_utils`, `spatial_index` |
| **Сигналы** | `signal_utils`, `smoothing_utils`, `interpolation_utils`, `frequency_filter` |
| **Метрики** | `metrics`, `metric_tracker`, `score_matrix_utils`, `score_norm_utils`, `quality_score_utils` |
| **Цвет** | `color_utils`, `histogram_utils`, `color_edge_export_utils` |
| **Визуализация** | `visualizer`, `render_utils` |
| **Событийная система** | `event_bus`, `event_log` |
| **Трекинг** | `tracker_utils`, `progress_tracker`, `scoring_pipeline_utils` |

### `scoring/` — 12 модулей, 4 219 строк — СПЯЩИЙ СУБПАКЕТ

> ⚠️ **СПЯЩИЙ:** Не импортируется ни в `main.py`, ни в `pipeline.py`, ни в каком-либо
> производственном модуле. Используется **только в тестах** (`tests/test_scoring_*.py`,
> `tests/test_*_extra.py`). Готов к подключению — вся инфраструктура реализована.

Все 12 модулей полностью реализованы и покрыты тестами (>99% прохождения):

| Модуль | Строк | Функция |
|---|---|---|
| `score_normalizer.py` | 341 | z-score, min-max, rank нормализация, clip нормализация |
| `threshold_selector.py` | 403 | Otsu, адаптивные пороги, Kapur, Triangle |
| `match_evaluator.py` | 349 | Оценка качества матчинга, precision/recall пар |
| `boundary_scorer.py` | 340 | Оценка качества граничных совпадений |
| `consistency_checker.py` | 345 | Проверка глобальной согласованности оценок |
| `match_scorer.py` | 324 | Итоговый скор пары фрагментов |
| `gap_scorer.py` | 337 | Анализ и оценка зазоров между фрагментами |
| `pair_filter.py` | 335 | Фильтрация пар по порогу совместимости |
| `pair_ranker.py` | 330 | Ранжирование пар фрагментов |
| `global_ranker.py` | 331 | Глобальное ранжирование всех кандидатов |
| `evidence_aggregator.py` | 289 | Агрегация доказательств от нескольких матчеров |
| `rank_fusion.py` | 192 | Ранговое слияние (RRF, Borda count) |
| `__init__.py` | 303 | Публичный API субпакета |

**Потенциал при подключении:** замена жёстких весов в `pairwise.py` на
полноценный конфигурируемый scoring pipeline с нормализацией, фильтрацией и ранжированием.

### `io/` — 3 модуля, 1 141 строка — СПЯЩИЙ СУБПАКЕТ

> ⚠️ **СПЯЩИЙ:** Не импортируется в производственном коде
> (`main.py`, `pipeline.py`, весь `puzzle_reconstruction/`).
> **Важно:** `from .io import ...` в `utils/__init__.py` ссылается на
> `puzzle_reconstruction/utils/io.py` — это **другой** модуль внутри `utils/`, не этот субпакет.
> `puzzle_reconstruction.io` используется **только в тестах** (9 тестовых файлов, ~450 тестов).

| Модуль | Строк | Функция |
|---|---|---|
| `image_loader.py` | 325 | Пакетная загрузка PNG/JPEG/TIFF: `load_image()`, `load_image_dir()`, `filter_by_extension()`, `parse_fragment_id()`, `ImageRecord` |
| `metadata_writer.py` | 338 | Сохранение метаданных реконструкции: `write_json()`, `write_yaml()`, `AssemblyMetadata` |
| `result_exporter.py` | 397 | Экспорт результатов: PNG, PDF, JSON, HTML отчёт, `ExportConfig` |
| `__init__.py` | 81 | Публичный API субпакета |

**Потенциал при подключении:** `image_loader.py` может заменить ручной `cv2.imread` в `main.py`,
`result_exporter.py` — расширить форматы экспорта за рамки одного PNG.

### `ui/` — 1 модуль, 364 строки

- Интерактивный просмотрщик результатов (OpenCV/Qt). Открывается через `--interactive`.

---

## 13. Вспомогательные инструменты (tools/)

| Скрипт | Строк | CLI-команда | Назначение |
|---|---|---|---|
| `benchmark.py` | 363 | `puzzle-benchmark` | Генерирует документы, рвёт их, запускает все методы, сравнивает с ground truth, выводит JSON-отчёт |
| `evaluate.py` | 311 | `puzzle-evaluate` | Полная оценка: генерация → разрыв → реконструкция → метрики (NA, DC, RMSE) → HTML/JSON/Markdown |
| `profile.py` | 387 | `puzzle-profile` | Профилирует каждый этап пайплайна: segmentation, denoise, color_norm, descriptor, synthesis, compat_matrix, assembly. Поддерживает cProfile + pstats |
| `server.py` | 309 | `puzzle-server` | Flask REST API: `/health`, `/config`, `/api/reconstruct`, `/api/cluster`, `/api/report/<job_id>` |
| `tear_generator.py` | 303 | `puzzle-generate` | Синтетический генератор: рвёт изображения на N фрагментов с реалистичными рваными краями (Perlin-like шум) |
| `mix_documents.py` | 271 | `puzzle-mix` | Смешивает фрагменты нескольких документов, создаёт `ground_truth.json` для тестирования кластеризации |

---

## 14. Технический стек

### Обязательные зависимости (`pyproject.toml[dependencies]`)

| Библиотека | Версия | Назначение |
|---|---|---|
| **Python** | ≥3.11 | Язык реализации |
| **numpy** | ≥1.24 | Массивы, матрицы, числовые вычисления |
| **scipy** | ≥1.11 | Оптимизация, интерполяция, FFT, spatial |
| **opencv-python** | ≥4.8 | Компьютерное зрение, работа с изображениями |
| **scikit-image** | ≥0.22 | Алгоритмы обработки изображений |
| **Pillow** | ≥10.0 | Загрузка/сохранение изображений |
| **scikit-learn** | ≥1.3 | Кластеризация, PCA, метрики |

### Опциональные зависимости

| Группа | Библиотека | Назначение |
|---|---|---|
| `[ocr]` | pytesseract ≥0.3 | OCR верификация (+ нужен системный Tesseract-OCR) |
| `[yaml]` | pyyaml ≥6.0 | YAML конфигурация |
| `[pdf]` | reportlab ≥4.0, fpdf2 ≥2.7 | Экспорт отчётов в PDF |
| `[api]` | flask ≥3.0 | REST API сервер |
| `[all]` | все выше | Полная установка |

> **Примечание:** `shapely`, `networkx`, `matplotlib` указаны в `requirements.txt`, но отсутствуют в `pyproject.toml[dependencies]`.

### Dev-инструменты

| Инструмент | Версия | Назначение |
|---|---|---|
| pytest | ≥7.4 | Фреймворк тестирования |
| pytest-cov | ≥4.1 | Покрытие кода |
| ruff | ≥0.3 | Linter + formatter (100 символов/строка) |
| mypy | ≥1.8 | Статическая типизация (ignore_missing_imports) |

---

## 15. История разработки

### Временная шкала

```
20 февраля 2026  →  Фаза 1: Создание кодовой базы (iter-1 — iter-36)
                    Параллельное создание 305 модулей + базовые тесты
                    Алгоритмы, предобработка, матчинг, верификация, utils

21–23 февраля    →  Фаза 2: Расширение тестового покрытия (iter-37 — iter-191)
                    Систематическое добавление _extra.py тестов
                    2 модуля за итерацию, все подпакеты

23–24 февраля    →  Фаза 3: Комплексные тесты (iter-192 — iter-249)
                    4 модуля за итерацию
                    Особое внимание: граничные случаи, числовая стабильность

24 февраля       →  Фаза 4: Финализация тестов
                    iter-249 → добавлены REPORT.md, исправления моделей
                    Коммиты: 2a4c0bb, c3c44c3, d290633

24 февраля       →  Фаза 5: Интеграция алгоритмов
                    6c98327: feat: integrate all 8 assembly algorithms
                             + add matcher registry
                    Три fix: коммита снизили провалы с 133 до 1

25 февраля       →  STATUS.md — текущий документ
```

### Статистика коммитов (всего 260)

| Тип | Кол-во | Примеры |
|---|---|---|
| `iter-NNN:` | 212 | Итеративное добавление тестов |
| `feat:` | ~15 | Новые функции, алгоритмы |
| `fix:` | ~23 | Исправления (в т.ч. 88 провалившихся тестов) |
| `docs:` | ~10 | Документация |

### Последние коммиты

| Хэш | Сообщение |
|---|---|
| `d2fd92b` | docs: add detailed implementation status document (STATUS.md) |
| `47d5fbe` | fix: resolve 4 more failing tests |
| `445b7f6` | fix: resolve 26 more failing tests (Phase 6 continued) |
| `aa47360` | fix: resolve 58 failing tests (19 new + 39 pre-existing) |
| `6c98327` | **feat: integrate all 8 assembly algorithms and add matcher registry** |
| `d290633` | docs: update REPORT.md with accurate statistics |
| `c3c44c3` | fix: add Placement/Edge models and fix test collection errors |
| `2a4c0bb` | docs: add comprehensive test coverage report (REPORT.md) |
| `ddbea00` | iter-249: add extra tests for report, seam_analyzer, spatial_validator, text_coherence |

---

## 16. Единственный провалившийся тест

Из **42 280** тестов провалился ровно **один**:

```
FAILED tests/test_scoring_gap_scorer.py::TestFilterGapMeasures::test_filter_all_out

Причина:
  tests/test_scoring_gap_scorer.py:338:
    assert filter_gap_measures(r, 1.1) == []

  puzzle_reconstruction/scoring/gap_scorer.py:276:
    raise ValueError("min_score должен быть в [0, 1], получено 1.1")
```

**Суть:** Тест ожидает, что при `min_score=1.1` (больше допустимого диапазона [0, 1])
функция вернёт пустой список. Функция вместо этого выбрасывает `ValueError`.

**Исправление:** Два варианта:
1. Изменить тест: `with pytest.raises(ValueError): filter_gap_measures(r, 1.1)`
2. Изменить функцию: не выбрасывать исключение при `min_score > 1`, возвращать `[]`

**Влияние на систему:** нулевое (edge case в scoring утилите, не в основном пайплайне).

---

## 17. Дорожная карта

### Что уже выполнено (ПОЛНОСТЬЮ)

| Задача | Статус |
|---|---|
| Реализация всех 305 модулей | ✅ 100% |
| Тестовое покрытие всех модулей | ✅ 100% (824 файла, 42 280 тестов) |
| Подключение всех 8 алгоритмов сборки в CLI | ✅ `6c98327` |
| Реестр 15 матчеров (`matcher_registry.py`) | ✅ `6c98327` |
| Конфигурируемые матчеры через `MatchingConfig` | ✅ |
| REST API сервер | ✅ `tools/server.py` |
| Бенчмарк и профилировщик | ✅ `tools/benchmark.py`, `tools/profile.py` |
| Синтетический генератор данных | ✅ `tools/tear_generator.py` |

### Что осталось

| Задача | Приоритет | Оценка работ |
|---|---|---|
| Исправить 1 провалившийся тест | 🔴 Высокий | 1–2 строки |
| Убрать метку `xfail` с 9 xpassed-тестов | 🟡 Средний | ~9 строк |
| Подключить `PreprocessingChain` (35 модулей) | 🟡 Средний | ~60 строк |
| Подключить `VerificationSuite` (20 модулей) | 🟡 Средний | ~80 строк |
| Подключить инфраструктурные утилиты (`event_bus`, `result_cache`, etc.) | 🟢 Низкий | ~30 строк |
| Подключить `scoring/` в пайплайн (заменить жёсткие веса) | 🟡 Средний | ~40 строк |
| Подключить `io/image_loader.py` вместо ручного `cv2.imread` в `main.py` | 🟡 Средний | ~20 строк |
| Подключить `io/result_exporter.py` для расширения форматов вывода | 🟡 Средний | ~15 строк |
| Добавить YAML-конфигурацию | 🟢 Низкий | ~20 строк |
| Устранить расхождение между `requirements.txt` и `pyproject.toml` (shapely, networkx, matplotlib) | 🟢 Низкий | 3 строки |

---

## 18. Итоговая оценка готовности

| Компонент | Реализован | Активен в CLI | Тесты | Готовность |
|---|---|---|---|---|
| Алгоритм (Танграм + Фрактал + Синтез) | 100% | 100% | ✅ | **Production** |
| Пайплайн (6 этапов, класс Pipeline) | 100% | 100% | ✅ | **Production** |
| CLI-интерфейс (`main.py`, все аргументы) | 100% | 100% | ✅ | **Production** |
| Алгоритмы сборки (8/8) | 100% | **100%** | ✅ | **Production** |
| Матчеры (15 зарегистрировано) | 100% | 4 по умолч. | ✅ | **Beta** |
| Предобработка (38 модулей) | 100% | 5 активных | ✅ | **Beta** |
| Верификация (21 модуль) | 100% | 1 активный | ✅ | **Beta** |
| Scoring (12 модулей) ⚠️ | 100% | ❌ **не подключён** | ✅ | **Alpha** |
| IO-субпакет (3 модуля) ⚠️ | 100% | ❌ **не подключён** | ✅ | **Alpha** |
| Utils (130 модулей) | 100% | ~3 активных | ✅ | **Alpha** |
| Tools (6 скриптов) | 100% | 6/6 CLI | ✅ | **Beta** |
| REST API сервер | 100% | `puzzle-server` | — | **Alpha** |
| Документация | 100% | — | — | **Production** |
| Тестовое покрытие | 100% модулей | 99.976% pass | — | **Production** |

### Общий вывод

Проект находится в стадии **позднего Alpha / раннего Beta**.

- **Ядро** (алгоритм + пайплайн + CLI + все 8 методов сборки) — **готово к Production**.
- **Расширенные возможности** (все 15 матчеров, полная предобработка, полная верификация) — реализованы, покрыты тестами, но не подключены по умолчанию.
- **Единственная известная ошибка** — тривиальный edge case в одном тестовом файле.
- **Переход в Beta** требует подключения `PreprocessingChain` и `VerificationSuite` (~140 строк изменений).
- **Переход в Stable** — полная интеграция Research Mode и стабилизация всех матчеров.

---

*Документ создан 2026-02-25 на основе живого тестового прогона и детального анализа кодовой базы.*
*Обновлять при каждом значимом изменении статуса компонентов.*
