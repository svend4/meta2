# INTEGRATION_ROADMAP.md
## Дорожная карта интеграции: от «мёртвого» кода к живой экосистеме

> Версия: 1.0.0
> Дата: 2026-02-24
> Ветка: `claude/puzzle-text-docs-3tcRj`
> Принцип: **не удалять — находить правильное место и роль**

---

## Содержание

1. [Ключевое открытие](#1-ключевое-открытие)
2. [Анатомия проекта: дерево модулей](#2-анатомия-проекта-дерево-модулей)
3. [Таксономия «спящих» модулей](#3-таксономия-спящих-модулей)
4. [Матрица: алгоритм → сценарий применения](#4-матрица-алгоритм--сценарий-применения)
5. [Архитектурные мосты: что изменить](#5-архитектурные-мосты-что-изменить)
6. [Фазы интеграции](#6-фазы-интеграции)
7. [Research Mode: исследовательский режим](#7-research-mode-исследовательский-режим)
8. [Карта зависимостей](#8-карта-зависимостей)
9. [Критерии успеха](#9-критерии-успеха)

---

## 1. Ключевое открытие

Анализ кодовой базы показал: так называемый «мёртвый» код (~48 200 строк) **не является мёртвым по своей природе**. Он реализован, экспортирован, покрыт тестами — но не подключён к точке входа `main.py`.

### Почему код «спит», а не «умер»

| Характеристика | Мёртвый код | Спящий код (наш случай) |
|---|---|---|
| Реализован | нет / частично | ✅ полностью |
| Протестирован | нет | ✅ 25 000+ тестов |
| Экспортирован в `__init__.py` | нет | ✅ везде |
| Подключён к `main.py` | — | ❌ только 4/8 алгоритмов |
| Нужны ли изменения кода? | большие | ✅ минимальные (мосты) |

### Три точки разрыва (всё остальное работает)

```
main.py:assemble()    ──── if/elif (4 метода) ─────▶  parallel.py (8 методов)
                                                        ^^^^^^^^^^^^^ МОСТ №1
main.py:process_fragment() ── 5 модулей из 38 ──▶  preprocessing/* (38 модулей)
                                                        ^^^^^^^^^^^^^ МОСТ №2
matching/pairwise.py  ──── жёсткие веса (4) ──────▶  matcher_registry.py (20+)
                                                        ^^^^^^^^^^^^^ МОСТ №3
```

**Вывод:** инфраструктура уже существует. `parallel.py` содержит реестр всех 8 алгоритмов. `consensus.py` умеет голосовать между результатами. `score_combiner.py` агрегирует оценки от любого числа матчеров. Нужно лишь подключить провода.

---

## 2. Анатомия проекта: дерево модулей

Метафора «дерево» хорошо отражает структуру: каждая часть выполняет свою роль, все части — единый живой организм.

```
puzzle-reconstruction
│
├── 🌱 КОРНИ (Preprocessing — 38 модулей)
│   Подготавливают сырой материал: очищают, нормализуют,
│   выделяют значимые характеристики фрагментов.
│   Сейчас используется: segmentation, contour, orientation (3/38)
│
├── 🌳 СТВОЛ (Pipeline Core — 4 файла)
│   main.py, pipeline.py, config.py, models.py
│   Оркестрирует 6-этапный процесс восстановления.
│
├── 🌿 ВЕТВИ (Assembly — 8 алгоритмов + 15 вспомогательных)
│   Стратегии сборки документа. Каждая — насадка на одну рукоятку (parallel.py).
│   Сейчас в CLI: greedy, sa, beam, gamma (4/8)
│   Ждут подключения: genetic, exhaustive, ant_colony, mcts (4/8)
│
├── 🍃 ЛИСТЬЯ (Matching — 26 модулей)
│   Алгоритмы оценки совместимости краёв фрагментов.
│   13 различных «чувств» распознавания: форма, цвет, текстура, геометрия.
│   Сейчас в scorer: CSS, DTW, FD, TEXT (4/13+)
│
├── 🍎 ПЛОДЫ (Verification — 21 модуль)
│   Проверка качества итоговой сборки. A–F оценка.
│   Сейчас используется: ocr (1/21)
│
└── 🌍 ПОЧВА (Utils — 103 модуля)
    Инфраструктура: кэш, шина событий, пайплайн-раннер,
    геометрия, метрики, визуализация.
    Сейчас используется: logger (1/103)
```

---

## 3. Таксономия «спящих» модулей

### 3.1 Корни — Preprocessing (35 модулей ждут)

Каждый модуль — специализированный фильтр/трансформер изображения.
Роль: **предобработка фрагмента до извлечения EdgeSignature**.

| Модуль | Роль | Когда нужен |
|---|---|---|
| `denoise` | Шумоподавление (Gaussian/Median/Bilateral/NLM) | Плохие сканы, шумные изображения |
| `contrast` | CLAHE, гистограммное выравнивание, гамма | Низкий контраст, пожелтевшие документы |
| `binarizer` | Отсу, Саувола, Нибэк, Бернсен | Рукописный текст, неравномерное освещение |
| `document_cleaner` | Удаление теней, рамок, пятен | Архивные документы, копии |
| `noise_analyzer` | σ, SNR, JPEG-артефакты, зернистость | Автодиагностика качества входных данных |
| `fragment_cropper` | Автообрезка к содержательной области | Фрагменты с полями/фоном |
| `noise_reducer` | Лёгкое шумоподавление с оценкой | Быстрая предобработка |
| `deskewer` | Коррекция наклона (Hough/FFT/projection) | Перекошенные сканы |
| `background_remover` | Удаление фона (thresh/edges/GrabCut) | Фрагменты на цветном фоне |
| `patch_normalizer` | Нормализация патчей для матчинга | Перед извлечением дескрипторов |
| `quality_assessor` | Оценка blur/noise/contrast/completeness | Фильтрация плохих фрагментов |
| `illumination_corrector` | Ретинекс, гомоморфная фильтрация | Неравномерное освещение |
| `morphology_ops` | Эрозия, дилатация, opening/closing | Улучшение маски сегментации |
| `frequency_filter` | FFT-фильтрация (low/high/band pass) | Удаление периодических артефактов |
| `channel_splitter` | Разделение каналов RGB/HSV/LAB | Цветовой анализ фрагментов |
| `adaptive_threshold` | Адаптивная бинаризация | Документы с неравномерным фоном |
| `noise_filter` | Целенаправленное подавление шума | До контурной обработки |
| `edge_enhancer` | Усиление краёв (Laplacian/unsharp) | Размытые границы фрагментов |
| `color_normalizer` | Нормализация цвета (Gray World/LAB) | Цветовая консистентность между фрагментами |
| `image_enhancer` | Комплексное улучшение | Универсальный авто-улучшитель |
| `texture_analyzer` | Текстурные дескрипторы (LBP, Gabor) | Классификация типа документа |
| `gradient_analyzer` | Анализ градиентов | Характеристики краёв для EdgeSignature |
| `patch_sampler` | Случайная/регулярная выборка патчей | Обучение/обогащение дескрипторов |
| `frequency_analyzer` | Спектральный анализ | Детектирование регулярных паттернов |
| `warp_corrector` | Коррекция искажений | Изогнутые/деформированные фрагменты |
| `contrast_enhancer` | Целенаправленное улучшение контраста | Перед OCR |
| `contour_processor` | Обработка контуров | Уточнение после extract_contour |
| `illumination_normalizer` | Нормализация освещённости | Batch-нормализация коллекций |
| `edge_sharpener` | Усиление резкости краёв | Повышение качества EdgeSignature |
| `skew_correction` | Коррекция наклона (Hough) | Повёрнутые фрагменты |
| `perspective` | Коррекция перспективы | Сфотографированные (не сканированные) фрагменты |
| `augment` | Аугментация данных | Тренировочный режим / генерация датасетов |
| `color_norm` | CLAHE + Gray World + гамма | Базовая цветовая нормализация |
| `noise_reduction` | Расширенное шумоподавление | Архивные документы |
| `background_remover` | Удаление фона | Фрагменты на пёстром фоне |

**Интеграционный план:** создать `PreprocessingChain` — конфигурируемый список фильтров.

```python
# config.yaml (будущий)
preprocessing:
  chain: ["quality_assessor", "denoise", "contrast", "deskewer", "binarizer"]
  quality_threshold: 0.4   # отфильтровать фрагменты ниже порога
  auto_enhance: true        # автоматически выбирать фильтры
```

### 3.2 Ветви — Assembly: 4 спящих алгоритма

Все 4 реализованы и зарегистрированы в `parallel.py`. Нужно только добавить в CLI.

| Алгоритм | Модуль | Сложность | Лучший сценарий | Статус |
|---|---|---|---|---|
| `genetic` | `assembly/genetic.py` | O(G·P·N²) | 15–30 фрагментов, высокое качество | 🔴 не в CLI |
| `exhaustive` | `assembly/exhaustive.py` | O(N!) | ≤8 фрагментов, точный результат | 🔴 не в CLI |
| `ant_colony` | `assembly/ant_colony.py` | O(I·A·N²) | 20–50 фрагментов, хорошее покрытие | 🔴 не в CLI |
| `mcts` | `assembly/mcts.py` | O(S·D) | 6–25 фрагментов, exploration/exploitation | 🔴 не в CLI |

**Вспомогательные модули ветви** (15 модулей, экспортированы но не используются напрямую из pipeline):

| Модуль | Роль в будущей архитектуре |
|---|---|
| `layout_builder` | Построение 2D-компоновки → визуализация в research mode |
| `collision_detector` | AABB-проверка коллизий → валидация сборки |
| `gap_analyzer` | Анализ зазоров → метрика качества сборки |
| `canvas_builder` | Рендер финального холста → замена `render_assembly_image` |
| `position_estimator` | Оценка позиций → уточнение после assembly |
| `fragment_mapper` | Маппинг в зоны → структурированный отчёт |
| `sequence_planner` | Планирование порядка → оптимизация beam_search |
| `layout_refiner` | Итеративное уточнение → post-processing |
| `overlap_resolver` | Разрешение перекрытий → финальная полировка |
| `fragment_sorter` | Сортировка перед сборкой → preprocessing |
| `fragment_sequencer` | Определение порядка → input для beam/MCTS |
| `candidate_filter` | Фильтрация кандидатов → ускорение матрицы |
| `cost_matrix` | Матрицы стоимостей → альтернатива compat_matrix |
| `fragment_arranger` | Расстановка на холсте → визуализация |
| `score_tracker` | Трекинг эволюции → research mode plots |
| `placement_optimizer` | Оптимизация порядка → улучшение greedy |

### 3.3 Листья — Matching: 9+ спящих матчеров

Текущий scorer (`pairwise.py`) использует 4 матчера с жёсткими весами.
Остальные реализованы и экспортированы.

| Матчер | Модуль | Что измеряет | Лучший для |
|---|---|---|---|
| `icp` | `matching/icp.py` | Геометрическое выравнивание (ICP) | Точные геометрические совпадения |
| `color_match` | `matching/color_match.py` | Цветовые гистограммы | Цветные документы, фото |
| `texture_match` | `matching/texture_match.py` | Текстурные дескрипторы | Документы с текстурой |
| `shape_matcher` | `matching/shape_matcher.py` | Shape Context | Сложные формы краёв |
| `geometric_match` | `matching/geometric_match.py` | Геометрические инварианты | Повёрнутые фрагменты |
| `seam_score` | `matching/seam_score.py` | Непрерывность швов | Точная стыковка |
| `boundary_matcher` | `matching/boundary_matcher.py` | Профиль границы | Рваные края |
| `affine_matcher` | `matching/affine_matcher.py` | Аффинное преобразование | Деформированные фрагменты |
| `spectral_matcher` | `matching/spectral_matcher.py` | Спектральные дескрипторы | Периодические паттерны |
| `graph_match` | `matching/graph_match.py` | Граф совместимости | Глобальная согласованность |
| `feature_match` | `matching/feature_match.py` | Feature descriptors | Точечные совпадения |
| `patch_matcher` | `matching/patch_matcher.py` | Патч-совпадение | Текстурные регионы |
| `edge_comparator` | `matching/edge_comparator.py` | Попарное сравнение краёв | Базовое сравнение |
| `orient_matcher` | `matching/orient_matcher.py` | Ориентация | Совпадение направлений |
| `curve_descriptor` | `matching/curve_descriptor.py` | Дескриптор кривой | Вычисление базовых признаков |

**Инфраструктура агрегации** (уже готова, не используется):

| Модуль | Функция |
|---|---|
| `score_combiner.py` | `weighted_combine()`, `rank_combine()`, `min_combine()`, `max_combine()` |
| `score_aggregator.py` | Агрегация от N матчеров → единый `CompatEntry` |
| `consensus.py` | Голосование по результатам нескольких методов сборки |
| `score_normalizer.py` | Нормализация оценок перед комбинацией |
| `global_matcher.py` | Глобальный матчинг с учётом всех пар |
| `candidate_ranker.py` | Ранжирование кандидатов |
| `pair_scorer.py` | Итоговый скор пары |
| `patch_validator.py` | Валидация патч-совпадений |

### 3.4 Плоды — Verification: 20 спящих модулей

Сейчас из 21 модуля используется только `ocr.py`. Остальные реализованы.

| Модуль | Что проверяет | Оценка |
|---|---|---|
| `metrics` | IoU, Kendall τ, permutation distance, placement accuracy | Количественные метрики |
| `text_coherence` | Связность текста между фрагментами | Семантическая корректность |
| `confidence_scorer` | Уверенность в каждом размещении | Per-fragment score |
| `consistency_checker` | Глобальная согласованность сборки | Системная проверка |
| `layout_checker` | Корректность 2D-компоновки | Геометрия |
| `overlap_checker` | Пересечения фрагментов | Физическая корректность |
| `seam_analyzer` | Качество швов | Визуальная непрерывность |
| `boundary_validator` | Корректность граничных условий | Граница документа |
| `fragment_validator` | Валидность каждого фрагмента | Предварительная проверка |
| `assembly_scorer` | Суммарный score всей сборки | Итоговая оценка A-F |
| `completeness_checker` | Полнота — все фрагменты размещены? | Покрытие |
| `overlap_validator` | Детальная проверка перекрытий | Физическая корректность |
| `spatial_validator` | Пространственные связи | Топология |
| `placement_validator` | Корректность каждого размещения | Per-placement |
| `layout_scorer` | Оценка 2D-компоновки | Геометрический score |
| `score_reporter` | Формирование отчёта по оценкам | Отчётность |
| `edge_validator` | Проверка совместимости краёв | Edge-level QA |
| `quality_reporter` | Полный качественный отчёт | Документация результата |
| `layout_verifier` | Итоговая верификация компоновки | Финальный контроль |
| `report` | Генерация отчёта | Экспорт результатов |

**Интеграционный план:** создать `VerificationSuite` — запуск выбранных верификаторов и агрегация оценок.

### 3.5 Почва — Utils: 102 спящих модуля

Инфраструктурные «строительные блоки» для будущей архитектуры.
Три категории:

**А. Инфраструктура пайплайна** (готова к немедленному использованию):

| Модуль | Роль | Где использовать |
|---|---|---|
| `event_bus` | Pub/sub шина событий | pipeline.py → трекинг прогресса |
| `pipeline_runner` | Multi-step runner с retry | Заменить ручной код в main.py |
| `batch_processor` | Пакетная обработка | Обработка нескольких документов |
| `progress_tracker` | Трекер прогресса | UI/лог прогресс-бар |
| `config_manager` | Управление конфигурацией | Расширение Config |
| `result_cache` | Кэш с TTL | Кэш дескрипторов и матриц |
| `cache_manager` | LRU-кэш | Кэш промежуточных результатов |
| `metric_tracker` | Трекинг метрик | Research mode |
| `event_log` | Журнал событий | Детальный аудит пайплайна |

**Б. Геометрия и компьютерное зрение** (для улучшения алгоритмов):

| Модуль | Роль |
|---|---|
| `geometry` | rotation_matrix_2d, polygon_area, poly_iou |
| `transform_utils` | rotate, flip, scale, affine_from_params |
| `color_utils` | to_gray, to_lab, compute_histogram |
| `mask_utils` | create_alpha_mask, erode_mask, crop_to_mask |
| `array_utils` | normalize_array, sliding_window, pairwise_norms |
| `contour_utils` | simplify_contour, contour_iou, align_orientation |
| `histogram_utils` | earth_mover_distance, chi_squared_distance |
| `keypoint_utils` | detect_keypoints, match_descriptors, filter_ransac |
| `signal_utils` | smooth_signal, cross_correlation, resample |
| `bbox_utils` | BBox, bbox_iou, merge_overlapping |
| `sparse_utils` | SparseEntry, sparse_top_k, threshold_matrix |
| `clustering_utils` | kmeans_cluster, hierarchical, silhouette_score |
| `distance_utils` | hausdorff, chamfer, cosine, pairwise |
| `smoothing_utils` | moving_average, savgol, smooth_contour |
| `feature_selector` | variance_selection, pca_reduce, rank_features |
| `graph_utils` | build_graph, dijkstra, minimum_spanning_tree |

**В. Предметные утилиты** (специфичные для задачи):

| Модуль | Роль |
|---|---|
| `image_stats` | ImageStats, энтропия, резкость, compare_images |
| `patch_extractor` | Patch, PatchSet, grid/sliding/random/border |
| `image_io` | ImageRecord, load_directory, batch_resize |
| `profiler` | StepProfile, PipelineProfiler, @timed |
| `visualizer` | word boxes, contours, matches, confidence bar |
| `metrics` | ReconstructionMetrics, IoU, Kendall τ |
| `cache` | LRU-кэш и дисковый кэш дескрипторов |

---

## 4. Матрица: алгоритм → сценарий применения

### 4.1 Алгоритмы сборки

| Алгоритм | N фрагментов | Время | Качество | Детерминирован | Лучший случай |
|---|---|---|---|---|---|
| `exhaustive` | ≤ 8 | O(N!) ~мин | ⭐⭐⭐⭐⭐ | ✅ да | Маленький точный пазл |
| `beam` | 6–20 | O(W·N²) ~сек | ⭐⭐⭐⭐ | ✅ да | Средний пазл, нужна скорость |
| `mcts` | 6–25 | O(S·D) ~сек | ⭐⭐⭐⭐ | ❌ нет | Сложная топология, много локальных минимумов |
| `genetic` | 15–40 | O(G·P·N²) ~мин | ⭐⭐⭐⭐ | ❌ нет | Много фрагментов, нет ограничений по времени |
| `ant_colony` | 20–60 | O(I·A·N²) ~мин | ⭐⭐⭐⭐ | ❌ нет | Крупный пазл, параллельный поиск |
| `gamma` | 20–100 | O(I·N²) ~мин | ⭐⭐⭐⭐⭐ | ❌ нет | Крупный пазл, SOTA качество |
| `sa` | любой | O(I) настр. | ⭐⭐⭐ | ❌ нет | Быстрое улучшение жадного решения |
| `greedy` | любой | O(N²) <1 сек | ⭐⭐ | ✅ да | Baseline, инициализация для других |

### 4.2 Режим `auto` — интеллектуальный выбор

```python
def auto_select_method(n_fragments: int) -> list[str]:
    if n_fragments <= 4:
        return ["exhaustive"]
    elif n_fragments <= 8:
        return ["exhaustive", "beam"]      # exhaustive точный, beam для сравнения
    elif n_fragments <= 15:
        return ["beam", "mcts", "sa"]      # скорость + качество
    elif n_fragments <= 30:
        return ["genetic", "gamma", "ant_colony"]  # эволюционные методы
    else:
        return ["gamma", "sa"]             # масштабируемые методы
```

### 4.3 Режим `all` — полное сравнение (research)

```python
# Все 8 методов → summary_table → consensus сборка
results = run_all_methods(fragments, entries, methods=ALL_METHODS)
print(summary_table(results))
# | Method     | Score  | Time(s) | Status  |
# |------------|--------|---------|---------|
# | gamma      | 0.9123 | 12.34   | OK      |
# | genetic    | 0.8991 | 45.21   | OK      |
# | mcts       | 0.8847 | 8.76    | OK      |
# ...
final = build_consensus([r.assembly for r in results if r.success])
```

### 4.4 Матчеры: когда что использовать

| Матчер | Тип документа | Состояние | Уникальность |
|---|---|---|---|
| CSS (контурные сигнатуры) | любой | хорошее | форма края |
| DTW (динамическое выравнивание) | любой | потёртые края | временны́е ряды |
| FD (фрактальная размерность) | рукопись | любое | текстура края |
| TEXT (OCR связность) | печатный | читаемый | семантика |
| ICP (итеративное выравнивание) | геометрический | точные края | точность позиции |
| Color (цветовые гистограммы) | цветной | яркий | цвет |
| Texture (LBP/Gabor) | документ с текстурой | любое | паттерн |
| Shape Context | сложные формы | любое | глобальная форма |
| Seam (непрерывность шва) | линейный | хорошее | стык |
| Boundary (профиль границы) | рваные края | плохое | граница |
| Affine (аффинное) | деформированный | любое | инвариантность |
| Spectral (спектральный) | с периодикой | любое | частота |
| Graph (граф) | много фрагментов | любое | глобальная структура |

---

## 5. Архитектурные мосты: что изменить

### Мост №1: Assembly Registry (2 файла, ~30 строк)

**Файл:** `puzzle_reconstruction/config.py`

```python
# БЫЛО (строка 54):
method: Literal["greedy", "sa", "beam", "gamma"] = "beam"

# СТАНЕТ:
from puzzle_reconstruction.assembly.parallel import ALL_METHODS
AssemblyMethod = Literal[
    "greedy", "sa", "beam", "gamma",
    "genetic", "exhaustive", "ant_colony", "mcts",
    "auto", "all"
]
method: AssemblyMethod = "beam"

# Новые параметры (добавить в AssemblyConfig):
genetic_pop:       int   = 50        # Размер популяции
genetic_gen:       int   = 100       # Число поколений
aco_ants:          int   = 20        # Число агентов-муравьёв
aco_iter:          int   = 100       # Итерации ACO
mcts_sim:          int   = 200       # Симуляции MCTS
exhaustive_max_n:  int   = 9         # Макс. N для exhaustive
auto_timeout:      float = 60.0      # Таймаут на метод в auto/all режиме
```

**Файл:** `main.py` (функция `assemble`, строки 159–190)

```python
# БЫЛО: 4 ветки if/elif с ручными вызовами

# СТАНЕТ: делегирование в parallel.py
from puzzle_reconstruction.assembly.parallel import (
    run_all_methods, run_selected, pick_best, ALL_METHODS, summary_table
)

def assemble(fragments, entries, cfg, log):
    method = cfg.assembly.method

    if method == "all":
        results = run_all_methods(
            fragments, entries, methods=ALL_METHODS,
            timeout=cfg.assembly.auto_timeout,
            n_workers=min(4, len(ALL_METHODS))
        )
        log.info("\n" + summary_table(results))
        return pick_best(results)

    if method == "auto":
        methods = _auto_methods(len(fragments))
        log.info(f"  Auto-selected: {methods}")
        results = run_all_methods(
            fragments, entries, methods=methods,
            timeout=cfg.assembly.auto_timeout
        )
        log.info("\n" + summary_table(results))
        return pick_best(results)

    # Одиночный метод — через реестр (backward-compatible)
    results = run_selected(fragments, entries, methods=[method])
    if not results or not results[0].success:
        log.error(f"Метод {method!r} завершился с ошибкой")
        sys.exit(1)
    return results[0].assembly
```

### Мост №2: Preprocessing Chain (1 файл, ~60 строк)

**Новый файл:** `puzzle_reconstruction/preprocessing/chain.py`

```python
"""
Конфигурируемая цепочка предобработки фрагментов.
Позволяет подключать любые из 38 модулей preprocessing через config.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

AVAILABLE_FILTERS = {
    "denoise":               "noise_reduction.smart_denoise",
    "contrast":              "contrast.auto_enhance",
    "deskew":                "deskewer.auto_deskew",
    "background_remove":     "background_remover.auto_remove_background",
    "quality_assess":        "quality_assessor.assess_quality",
    "illumination_correct":  "illumination_corrector.correct_by_retinex",
    "binarize":              "binarizer.auto_binarize",
    "document_clean":        "document_cleaner.clean",
    "noise_analyze":         "noise_analyzer.analyze",
    # ... все 35 модулей
}

@dataclass
class PreprocessingChain:
    """Выполняет последовательность фильтров предобработки."""
    filters: List[str] = field(default_factory=list)
    quality_threshold: float = 0.0    # 0.0 = без фильтрации по качеству
    auto_enhance: bool = False

    def apply(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Применяет цепочку фильтров. Возвращает None если качество < порога."""
        result = image
        for filter_name in self.filters:
            result = self._apply_one(filter_name, result)
            if result is None:
                return None
        return result
```

### Мост №3: Matcher Registry (1 новый файл + изменение pairwise.py)

**Новый файл:** `puzzle_reconstruction/matching/matcher_registry.py`

```python
"""
Реестр всех матчеров с единым интерфейсом.
Каждый матчер: (EdgeSignature, EdgeSignature) -> float [0..1]
"""
from typing import Dict, Callable, Optional
from ..models import EdgeSignature

# Тип матчера: принимает два края, возвращает score 0..1
MatcherFn = Callable[[EdgeSignature, EdgeSignature], float]

MATCHER_REGISTRY: Dict[str, MatcherFn] = {}

def register(name: str):
    """Декоратор для регистрации матчера."""
    def decorator(fn: MatcherFn) -> MatcherFn:
        MATCHER_REGISTRY[name] = fn
        return fn
    return decorator

def get_matcher(name: str) -> Optional[MatcherFn]:
    return MATCHER_REGISTRY.get(name)

def list_matchers() -> list[str]:
    return sorted(MATCHER_REGISTRY.keys())

# Базовые матчеры (уже работают в pairwise.py)
# Регистрация происходит при импорте модулей
```

**Изменение:** `puzzle_reconstruction/matching/pairwise.py`

```python
# БЫЛО: жёсткие веса константами
W_CSS  = 0.35
W_DTW  = 0.30
W_FD   = 0.20
W_TEXT = 0.15

# СТАНЕТ: веса из конфига
def match_score(e_i, e_j, text_score=0.0, cfg=None) -> CompatEntry:
    weights = cfg.matching.matcher_weights if cfg else DEFAULT_WEIGHTS
    active  = cfg.matching.active_matchers if cfg else DEFAULT_MATCHERS
    # Вычислить только активные матчеры, применить веса из конфига
```

**Изменение:** `puzzle_reconstruction/config.py` — расширить `MatchingConfig`

```python
@dataclass
class MatchingConfig:
    threshold:       float = 0.3
    dtw_window:      int   = 20
    # Новые поля:
    active_matchers: list = field(default_factory=lambda: ["css","dtw","fd","text"])
    matcher_weights: dict = field(default_factory=lambda: {
        "css": 0.35, "dtw": 0.30, "fd": 0.20, "text": 0.15
    })
    combine_method:  str  = "weighted"   # weighted | rank | min | max
```

---

## 6. Фазы интеграции

### Фаза 1 ✅ (выполнено) — Документация

Этот документ.

### Фаза 2 — Assembly Registry (высокий приоритет, малый риск)

**Что делать:**
1. Расширить `AssemblyConfig.method` Literal на 10 значений
2. Добавить параметры genetic/aco/mcts/exhaustive в `AssemblyConfig`
3. Переписать `main.py:assemble()` через `run_all_methods()`/`run_selected()`
4. Обновить argparse choices в `build_parser()`

**Риск:** минимальный. `parallel.py` уже реализован и протестирован.
**Обратная совместимость:** полная. `--method beam` работает как раньше.

**Файлы:**
- `puzzle_reconstruction/config.py` — 15 строк
- `main.py` — 30 строк (замена assemble())

### Фаза 3 — Matcher Registry (средний приоритет)

**Что делать:**
1. Создать `matching/matcher_registry.py`
2. Расширить `MatchingConfig` в `config.py`
3. Обновить `pairwise.py` — читать веса из конфига
4. Зарегистрировать все матчеры через декоратор `@register`

**Риск:** средний. Нужна осторожность с backward-compatibility scorer'а.
**Проверка:** `build_compat_matrix()` должна давать тот же результат с дефолтным конфигом.

**Файлы:**
- `matching/matcher_registry.py` — новый файл, ~100 строк
- `matching/pairwise.py` — ~20 строк изменений
- `config.py` — ~10 строк

### Фаза 4 — Preprocessing Chain (средний приоритет)

**Что делать:**
1. Создать `preprocessing/chain.py` с `PreprocessingChain`
2. Добавить `PreprocessingConfig` в `config.py`
3. Подключить цепочку в `main.py:process_fragment()`
4. Добавить в конфиг возможность задавать список фильтров

**Риск:** средний. Базовый поток (segmentation+contour+orientation) остаётся неизменным.

### Фаза 5 — Verification Suite (средний приоритет)

**Что делать:**
1. Создать `verification/suite.py` с `VerificationSuite`
2. Подключить к этапу верификации в `main.py`
3. Добавить `VerificationConfig.validators` список

**Риск:** низкий. Верификация — постпроцессинг, не влияет на сборку.

### Фаза 6 — Infrastructure Utils (низкий приоритет, высокая ценность)

**Что делать:**
1. Подключить `event_bus` в `pipeline.py` → прогресс-события
2. Использовать `result_cache` для кэша дескрипторов (между запусками)
3. Использовать `batch_processor` для обработки нескольких документов
4. Использовать `metric_tracker` в research mode

**Риск:** низкий. Всё аддитивно.

### Фаза 7 — Research Mode (исследовательский, низкий приоритет)

См. раздел 7.

---

## 7. Research Mode: исследовательский режим

### Концепция

Два режима работы одной системы:

```
production  (быстро, один лучший метод)
     ↕
research    (медленно, все методы, сравнение, consensus)
```

### CLI-интерфейс

```bash
# Production (текущее поведение)
python main.py --input scans/ --method beam

# Auto (интеллектуальный выбор по числу фрагментов)
python main.py --input scans/ --method auto

# Research (все 8 методов, benchmark)
python main.py --input scans/ --method all --research

# Benchmark (запуск уже объявлен в pyproject.toml)
puzzle-benchmark --input scans/ --methods all --output benchmark.json
```

### `--method all` + `--research` даёт:

1. **Таблица сравнения** (уже есть `summary_table()`):
   ```
   | Method     | Score  | Time(s) | Status  |
   |------------|--------|---------|---------|
   | gamma      | 0.9123 | 12.34   | OK      |
   | genetic    | 0.8991 | 45.21   | OK      |
   ```

2. **Консенсусная сборка** (уже есть `consensus.py`):
   - Запускаем все методы
   - Голосуем: пара (i,j) считается верной если ≥50% методов её выбрали
   - Строим финальную сборку из консенсусных пар

3. **Ablation study** (отключение отдельных матчеров):
   ```bash
   puzzle-benchmark --ablate-matchers css,dtw  # без CSS и DTW
   puzzle-benchmark --ablate-matchers fd        # без фрактальной размерности
   ```

4. **Score evolution plots** (через `score_tracker.py`):
   - График изменения score по итерациям для SA, GA, ACO, MCTS
   - Convergence detection

5. **Per-fragment quality map** (через `confidence_scorer.py`):
   - Тепловая карта уверенности по каждому фрагменту
   - Выделение «слабых мест» сборки

### Конфигурация research mode

```yaml
# research_config.yaml
mode: research
assembly:
  method: all
  auto_timeout: 120.0
matching:
  active_matchers: ["css", "dtw", "fd", "text", "color", "texture", "seam"]
  combine_method: weighted
preprocessing:
  chain: ["quality_assessor", "denoise", "contrast", "deskewer"]
verification:
  validators: ["metrics", "text_coherence", "layout_checker",
               "seam_analyzer", "assembly_scorer", "completeness_checker"]
  run_ocr: true
research:
  consensus: true
  consensus_threshold: 0.5
  score_evolution: true
  ablation: false
  export_comparison: true
```

---

## 8. Карта зависимостей

### Поток данных (текущий + будущий)

```
[images/]
    │
    ▼ PreprocessingChain (Фаза 4)
    │  ├── quality_assessor → фильтрация плохих фрагментов
    │  ├── denoise / contrast / deskew → улучшение
    │  └── background_remover / binarizer → по типу документа
    │
    ▼ process_fragment() [main.py]
    │  ├── segmentation (✅ активно)
    │  ├── orientation  (✅ активно)
    │  ├── contour      (✅ активно)
    │  ├── tangram      (✅ активно)
    │  └── fractal      (✅ активно)
    │
    ▼ build_compat_matrix() [matching/compat_matrix.py]
    │  └── match_score() → MatcherRegistry (Фаза 3)
    │       ├── css    (✅ активно)
    │       ├── dtw    (✅ активно)
    │       ├── fd     (✅ активно)
    │       ├── text   (✅ активно)
    │       ├── color      🔴 спит
    │       ├── texture    🔴 спит
    │       ├── icp        🔴 спит
    │       ├── seam       🔴 спит
    │       └── ...9 других 🔴 спят
    │
    ▼ assemble() → parallel.py (Фаза 2)
    │  ├── greedy    (✅ в CLI)
    │  ├── sa        (✅ в CLI)
    │  ├── beam      (✅ в CLI)
    │  ├── gamma     (✅ в CLI)
    │  ├── genetic   🔴 спит
    │  ├── exhaustive 🔴 спит
    │  ├── ant_colony 🔴 спит
    │  └── mcts      🔴 спит
    │
    ▼ VerificationSuite (Фаза 5)
    │  ├── ocr (✅ активно)
    │  ├── metrics        🔴 спит
    │  ├── text_coherence 🔴 спит
    │  ├── layout_checker 🔴 спит
    │  └── ...17 других   🔴 спят
    │
    ▼ [result.png / report.json]
```

### Модули без зависимостей (готовы к немедленному использованию)

Эти модули не требуют никаких изменений в зависимостях, только вызова:

```
assembly/parallel.py        → run_all_methods(), AssemblyRacer
matching/consensus.py       → build_consensus(), vote_on_pairs()
matching/score_combiner.py  → weighted_combine(), rank_combine()
matching/score_aggregator.py → aggregate_scores()
verification/metrics.py     → ReconstructionMetrics, iou_score()
utils/event_bus.py          → EventBus, make_event_bus()
utils/pipeline_runner.py    → run_pipeline(), make_step()
utils/result_cache.py       → ResultCache, cached_result()
utils/metric_tracker.py     → MetricTracker, export_metrics()
```

---

## 9. Критерии успеха

### После Фазы 2 (Assembly Registry):

- [ ] `python main.py --method genetic` работает
- [ ] `python main.py --method exhaustive` работает для N≤8
- [ ] `python main.py --method ant_colony` работает
- [ ] `python main.py --method mcts` работает
- [ ] `python main.py --method auto` выбирает методы по числу фрагментов
- [ ] `python main.py --method all` запускает все 8, выводит таблицу
- [ ] Все существующие тесты `pytest tests/ -x -q` проходят
- [ ] `--method beam` даёт идентичный результат с до-интеграционной версией

### После Фазы 3 (Matcher Registry):

- [ ] `config.yaml` с `active_matchers: [css, dtw, color]` работает
- [ ] Дефолтные веса дают идентичный результат с до-интеграционной версией
- [ ] `config.yaml` с `combine_method: rank` работает
- [ ] Добавление нового матчера через `@register("my_matcher")` работает

### После Фазы 7 (Research Mode):

- [ ] `--method all --research` выводит сравнительную таблицу
- [ ] Консенсусная сборка лучше или равна лучшему одиночному методу
- [ ] `puzzle-benchmark --input scans/` генерирует `benchmark.json`
- [ ] Score evolution plot сохраняется для итерационных методов

---

*Этот документ является живым — обновлять при каждой завершённой фазе.*
