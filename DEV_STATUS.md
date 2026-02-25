# Текущий статус разработки — puzzle-reconstruction (meta2)

> Дата формирования отчёта: 2026-02-25 (обновлено — v1.0.0 Stable, Phase 8: +329 unit-тестов, 42 934 тестов)
> Предыдущие версии: 2026-02-23, 2026-02-24
> Версия: **1.0.0** (Production/Stable, тег `v1.0.0`)
> Текущая ветка: `claude/puzzle-text-docs-3tcRj`

---

## 1. Общие сведения о проекте

| Параметр | Значение |
|---|---|
| **Название пакета** | `puzzle-reconstruction` |
| **Репозиторий** | `meta2` |
| **Версия** | **1.0.0** |
| **Стадия** | `Development Status :: 5 - Production/Stable` |
| **Язык** | Python 3.11+ |
| **Лицензия** | MIT |
| **Первый коммит** | 2026-02-20 |
| **Последний коммит (текущая ветка)** | 2026-02-25 (`aec8b77` — feat: /api/validators endpoint, README verification docs, CHANGELOG fix) |
| **Всего коммитов** | 315+ |
| **Контрибьюторы** | 2 (Claude: ~310 коммитов, svend4: ~5 коммитов) |
| **Ветки** | `master`, `claude/puzzle-text-docs-3tcRj` (текущая), `origin/main` |
| **Merged PR** | 15 (PR #1–#15 из веток claude/*) |

**Назначение**: автоматическая реконструкция (сборка) разорванных, разрезанных
газет, книг и документов из отсканированных фрагментов. Два ключевых алгоритма —
Танграм (геометрическая аппроксимация) и Фрактальная кромка (фрактальная
размерность + CSS-дескриптор) — создают уникальную «подпись» каждого края для
точного сопоставления фрагментов.

---

## 1a. Итог интеграции — все 12 фаз выполнены (2026-02-25)

Ветки синхронизированы. Все запланированные «мосты» построены и включены
в текущую ветку `claude/puzzle-text-docs-3tcRj`.

### Что выполнено (12 фаз интеграции):

**Фаза 1 — Assembly Registry (Мост №1):**
- `AssemblyConfig.method`: `Literal["greedy","sa","beam","gamma","genetic","exhaustive","ant_colony","mcts","auto","all"]`
- `--method auto` выбирает методы по числу фрагментов; `--method all` запускает все 8
- `parallel.py:run_all_methods()` / `run_selected()` — параллельный запуск

**Фаза 2 — Preprocessing Chain (Мост №2):**
- `PreprocessingChain` подключает все 38 модулей через пайплайн
- Все preprocessing-модули активны через цепочку

**Фаза 3 — Matcher Registry (Мост №3):**
- `matching/matcher_registry.py`: `MATCHER_REGISTRY`, `@register`, `get_matcher()`, `list_matchers()`
- Все 13+ матчеров зарегистрированы через `@register`
- `MatchingConfig`: `active_matchers`, `matcher_weights`, `combine_method`

**Фаза 4 — Verification Suite первая волна:**
- `VerificationSuite` активирует 9 из 21 верификаторов

**Фазы 5–7 — scoring/, io/, infrastructure/research utils:**
- `scoring/` подключён к pipeline
- `io/` подключён к pipeline
- Infrastructure Utils + Research Mode реализованы

**Фаза 8 — Verification 21/21 (2026-02-25):**
- `_build_validator_registry()` расширен 12 новыми валидаторами: `boundary`, `layout_verify`,
  `overlap_validate`, `spatial`, `placement`, `layout_score`, `fragment_valid`, `quality_report`,
  `score_report`, `full_report`, `metrics`, `overlap_area`
- Добавлены: `run_all()` → запускает все 21, `all_validator_names()` → список всех 21 имён
- Исправлен `completeness` валидатор (неверный вызов API → правильная сигнатура)

**Фаза 9 — mypy coverage (2026-02-25):**
- `pyproject.toml`: 7 новых `[[tool.mypy.overrides]]` секций
- Строгая типизация: 50+ модулей (verification/*, assembly/*, matching/*, algorithms/*, корневые)
- Мягкая проверка: utils/*, preprocessing/*
- Глобально: `warn_unreachable=true`, `no_implicit_optional=true`

**Фаза 10 — CLI верификации (2026-02-25):**
- `--validators LIST` — запуск подмножества или `--validators all` (21 валидатор)
- `--export-report PATH` — экспорт отчёта: `.json`, `.md`/`.txt`, `.html`
- `_export_verification_report()` — три-форматный экспортёр с graceful fallback

**Фаза 11 — E2E тесты (2026-02-25):**
- `tests/test_suite_extended.py` — 82 теста для 12 новых валидаторов
- `tests/test_main_export_report.py` — 31 тест для `--validators` / `--export-report`
- `tests/test_integration_v2.py` — 20 `@integration` E2E тестов всего пайплайна

**Фаза 12 — v1.0.0 Stable (2026-02-25):**
- `VerificationReport.as_dict()` / `to_json(indent)` / `to_markdown()` / `to_html()` — сериализация отчёта
- `Pipeline.verify_suite(assembly, validators)` — интеграция Suite в Pipeline; `PipelineResult.verification_report`
- `PipelineResult.summary()` — показывает строку верификации Suite при наличии отчёта
- `--list-validators` CLI — список 21 валидатора без `--input`; обработка до argparse
- `_export_verification_report()` рефакторирован: делегирует методам отчёта (DRY)
- `tests/test_verification_report_methods.py` — 43 теста методов сериализации и Pipeline.verify_suite
- `tests/test_main_list_validators.py` — 20 тестов `--list-validators` и `all_validator_names()`
- `pyproject.toml`: `0.4.0b1` → `1.0.0`, classifier → `Production/Stable`
- `CHANGELOG.md`: раздел `[1.0.0]`; git-тег `v1.0.0`

### Метрики после интеграции:

| Метрика | Значение |
|---|---|
| Production .py файлов | **305** |
| Utils-модулей | **131** |
| Тестовых файлов | **827** (↑5 новых суммарно) |
| Всего тестов | **42 476** (↑72 за xfail-снятие + sklearn-разблокировку) |
| Assembly methods в CLI | **10** (greedy, sa, beam, gamma, genetic, exhaustive, ant_colony, mcts, auto, all) |
| Активных матчеров | **13+** (через `matcher_registry`) |
| Активных preprocessing-модулей | **38 из 38** (через PreprocessingChain) |
| Активных верификаторов | **21 из 21** (через VerificationSuite) |
| mypy coverage | **50+ модулей** строгой типизации |
| CLI-опции верификации | `--validators`, `--export-report`, `--list-validators` |
| VerificationReport API | `as_dict()`, `to_json()`, `to_markdown()`, `to_html()` |
| Pipeline.verify_suite() | ✅ Интегрирован; PipelineResult.verification_report |
| Edge/Placement в models.py | **Есть** (4 ImportError исправлены) |
| Документы | `DEV_STATUS.md`, `STATUS.md`, `INTEGRATION_ROADMAP.md`, `REPORT.md`, `IMPLEMENTATION_STATUS.md`, `CHANGELOG.md` |

---

## 2. Объём кодовой базы

| Компонент | Файлов (`.py`) | Строк кода | Классов | Функций |
|---|---:|---:|---:|---:|
| **preprocessing/** | 38 | ~11 655 | ~50 | ~300 |
| **algorithms/** (вкл. tangram/ + fractal/) | 42 | ~12 553 | ~150 | ~350 |
| **assembly/** | 27 | ~8 141 | ~45 | ~150 |
| **matching/** | 26 | ~7 825 | ~40 | ~120 |
| **verification/** | 21 | ~7 395 | ~30 | ~100 |
| **scoring/** | 12 | ~3 908 | ~20 | ~80 |
| **utils/** | 131 | ~38 976 | ~215 | ~1 398 |
| **io/** | 3 | ~1 060 | ~10 | ~30 |
| **ui/** | 1 | ~364 | ~1 | ~15 |
| **Корневые** (config, models, pipeline, clustering, export) | 5 | ~1 402 | ~15 | ~40 |
| **Точка входа** `main.py` | 1 | 377 | 0 | 6 |
| **CLI-утилиты** `tools/` | 6 | ~1 640 | 2 | ~40 |
| **ИТОГО production** | **305** | **~93 279** | **~578** | **~2 629** |
| **Тесты** `tests/` | 827 | ~269 000+ | — | ~42 404+ |
| **ИТОГО** | **1 129** | **~362 000** | — | — |

---

## 3. Архитектура — модули и их статус

### 3.1 Структура пакета

```
puzzle_reconstruction/
├── __init__.py
├── config.py              # Централизованная конфигурация (7 dataclass-секций)
├── models.py              # Структуры данных (Fragment, EdgeSignature, ...)
├── pipeline.py            # Унифицированный пайплайн (6 этапов, ThreadPoolExecutor)
├── clustering.py          # Кластеризация фрагментов (GMM/KMeans/Spectral)
├── export.py              # Canvas/PDF/Heatmap/Mosaic экспорт
│
├── preprocessing/         # 39 модулей — подготовка изображений
│   ├── segmentation.py       # Otsu / Adaptive / GrabCut
│   ├── contour.py            # Контур, RDP, разбиение на 4 края
│   ├── orientation.py        # Ориентация по тексту
│   ├── color_norm.py         # CLAHE, white balance, gamma
│   ├── denoise.py            # Gaussian, Median, Bilateral, NLM
│   ├── adaptive_threshold.py # Otsu, Niblack, Sauvola, Bernsen (6 методов)
│   ├── binarizer.py          # Расширенная бинаризация с энтропийным автовыбором
│   ├── deskewer.py           # Исправление наклона (проекции/Hough)
│   ├── perspective.py        # Коррекция перспективы
│   ├── frequency_filter.py   # FFT: low/high/band-pass, notch
│   ├── gradient_analyzer.py  # Sobel/Scharr/Prewitt + HOG-гистограммы
│   ├── ... (ещё 28 модулей)  # augment, background_remover, channel_splitter,
│   │                         # contrast, denoise, document_cleaner, edge_detector,
│   │                         # edge_enhancer, edge_sharpener, fragment_cropper,
│   │                         # frequency_analyzer, illumination_corrector,
│   │                         # image_enhancer, morphology_ops, noise_analyzer,
│   │                         # noise_filter, noise_reducer, patch_normalizer,
│   │                         # patch_sampler, quality_assessor, skew_correction,
│   │                         # texture_analyzer, warp_corrector, ...
│   └── __init__.py           # 774 строки, 140+ экспортов
│
├── algorithms/            # 46 модулей — вычисление дескрипторов
│   ├── tangram/
│   │   ├── hull.py           # Convex hull → RDP → PCA-нормализация
│   │   ├── classifier.py     # Классификация: triangle/rect/trapezoid/pentagon/...
│   │   └── inscriber.py      # Вписывание и извлечение танграм-краёв
│   ├── fractal/
│   │   ├── box_counting.py   # FD = slope(log N vs log 1/r), диапазон [1.0, 2.0]
│   │   ├── divider.py        # Richardson: L(s) ~ s^(1-FD), FD = 1 - slope
│   │   ├── css.py            # MPEG-7 CSS + Freeman chain code
│   │   └── ifs.py            # Барнсли IFS: {d_n} ∈ [-0.95, 0.95]
│   ├── synthesis.py          # B_virtual(t) = α·B_tangram(t) + (1-α)·B_fractal(t)
│   ├── shape_context.py      # Log-polar histogram (Belongie 2002)
│   ├── fourier_descriptor.py # FFT-дескрипторы формы
│   ├── sift_matcher.py       # SIFT + Lowe ratio test + RANSAC
│   ├── homography_estimator.py # DLT + RANSAC + Hartley normalization
│   ├── path_planner.py       # Dijkstra, Floyd-Warshall, Prim MST
│   ├── ... (ещё 25 модулей)
│   └── __init__.py           # 122 строки, 70+ экспортов
│
├── matching/              # 27 модулей — сопоставление краёв
│   ├── dtw.py                # O(N·M·w) с окном Сакое-Чибы
│   ├── pairwise.py           # Попарная оценка совместимости
│   ├── compat_matrix.py      # Матрица N×N
│   ├── color_match.py        # BGR/HSV/LAB гистограммы + моменты
│   ├── feature_match.py      # SIFT/ORB ключевые точки
│   ├── icp.py                # Iterative Closest Point
│   ├── geometric_match.py    # Hu моменты, aspect ratio
│   ├── spectral_matcher.py   # Magnitude/phase корреляция
│   ├── graph_match.py        # MST, spectral ordering, random walk
│   ├── consensus.py          # Голосование нескольких методов
│   ├── affine_matcher.py     # RANSAC/LMEDS аффинные преобразования
│   ├── boundary_matcher.py   # Hausdorff/Chamfer distance
│   ├── ... (ещё 15 модулей)
│   └── __init__.py           # 150+ строк, 100+ экспортов
│
├── assembly/              # 28 модулей — сборка документа
│   ├── greedy.py             # O(E): жадная сборка по CompatEntry
│   ├── annealing.py          # SA: Metropolis exp(dE/T), геом. охлаждение
│   ├── beam_search.py        # B гипотез × N расширений на глубину
│   ├── gamma_optimizer.py    # MLE Gamma(k,θ), MCMC Metropolis-Hastings
│   ├── genetic.py            # Order Crossover, tournament selection, elitism
│   ├── ant_colony.py         # τ^α·η^β, elite reinforcement, evaporation ρ
│   ├── mcts.py               # UCB1 = mean + c·√(ln(parent)/visits)
│   ├── exhaustive.py         # Branch & Bound, fallback при N>8
│   ├── parallel.py           # ThreadPoolExecutor, AssemblyRacer
│   ├── assembly_state.py     # Immutable state (copy-on-write)
│   ├── canvas_builder.py     # Overwrite/average blending
│   ├── collision_detector.py # AABB O(N²) + greedy resolution
│   ├── cost_matrix.py        # minmax/zscore/rank нормализация
│   ├── layout_refiner.py     # Gradient-descent позиций
│   ├── ... (ещё 14 модулей)
│   └── __init__.py           # 506 строк, 8 алгоритмов + утилиты
│
├── verification/          # 22 модуля — верификация результата
│   ├── ocr.py                # Tesseract: strip OCR + quality scoring
│   ├── assembly_scorer.py    # geometry + coverage + seam + uniqueness
│   ├── confidence_scorer.py  # 5 компонент → оценка A/B/C/D/F
│   ├── boundary_validator.py # Экспоненциальный штраф за gap/overlap
│   ├── completeness_checker.py # Fragment + spatial coverage
│   ├── consistency_checker.py  # LINE_SPACING, CHAR_HEIGHT, TEXT_ANGLE
│   ├── metrics.py            # NA, DC, RMSE, angular error, edge match rate
│   ├── text_coherence.py     # N-gram языковая модель
│   ├── seam_analyzer.py      # Color/gradient/texture continuity
│   ├── ... (ещё 13 модулей)
│   └── __init__.py           # 467 строк, 100+ экспортов
│
├── scoring/               # 13 модулей — оценка и ранжирование
│   ├── score_normalizer.py   # minmax, zscore, rank, softmax, sigmoid
│   ├── global_ranker.py      # Глобальная нормализация + ранжирование
│   ├── pair_filter.py        # По score, inlier, top-K, дедупликация
│   ├── rank_fusion.py        # RRF (Reciprocal Rank Fusion), Borda count
│   ├── evidence_aggregator.py # Взвешенная агрегация сигналов
│   ├── threshold_selector.py # Fixed, percentile, Otsu, F1, adaptive
│   ├── match_evaluator.py    # Precision, Recall, F-score
│   ├── ... (ещё 6 модулей)
│   └── __init__.py           # 304 строки, 80+ экспортов
│
├── io/                    # 4 модуля
│   ├── image_loader.py       # Загрузка с resize, color mode, batch
│   ├── metadata_writer.py    # JSON/CSV экспорт метаданных
│   ├── result_exporter.py    # JSON/CSV/Image/Text/Summary экспорт
│   └── __init__.py           # 81 строка
│
├── ui/                    # 2 модуля
│   ├── viewer.py             # OpenCV: drag, rotate (R), auto-SA (A),
│   │                         # save (S), undo (Z), zoom (+/-)
│   │                         # Цвета: green>0.85, yellow>0.65, red<0.65
│   └── __init__.py
│
└── utils/                 # 103 модуля (32 590 строк, 215 классов, 1398 функций)
    ├── [Геометрия]           # geometry, bbox_utils, polygon_utils, contour_utils,
    │                         # rotation_utils, transform_utils, alignment_utils (8 файлов)
    ├── [Метрики]             # metrics, distance_utils, curve_metrics, edge_scorer,
    │                         # *_score_utils (16 файлов)
    ├── [Сигналы]             # array_utils, frequency_utils, histogram_utils,
    │                         # interpolation_utils, normalization_utils, sampling_utils,
    │                         # signal_utils, smoothing_utils, stats_utils (10 файлов)
    ├── [Изображения]         # image_io, image_stats, mask_utils, render_utils,
    │                         # blend_utils, patch_utils (8 файлов)
    ├── [Кэширование]         # cache, cache_manager, result_cache, config_manager (5 файлов)
    ├── [Событийная шина]     # event_bus, event_log, logger, progress_tracker (5 файлов)
    ├── [Контуры/дескрипторы] # contour_sampler, curvature_utils, descriptor_utils,
    │                         # keypoint_utils, shape_match_utils (6 файлов)
    ├── [Цвет/рёбра]          # color_utils, color_hist_utils, edge_profiler,
    │                         # gradient_utils (6 файлов)
    ├── [Кластеризация]       # clustering_utils, spatial_index, tile_utils,
    │                         # patch_extractor, fragment_stats (6 файлов)
    ├── [Ранжирование]        # candidate_rank_utils, voting_utils, score_aggregator,
    │                         # score_matrix_utils, score_norm_utils (7 файлов)
    ├── [Пайплайн]            # pipeline_runner, batch_processor, profiler,
    │                         # filter_pipeline_utils, path_plan_utils (5 файлов)
    ├── [Спец. утилиты]       # icp_utils, morph_utils, sparse_utils,
    │                         # topology_utils, threshold_utils, window_utils, ... (14 файлов)
    ├── [Визуализация]        # visualizer, text_utils, io, annealing_schedule (4 файлов)
    └── __init__.py           # Экспорт всех символов
```

### 3.2 Детальный статус каждого модуля

| Модуль | Файлов | LOC | Паттерн | Статус |
|--------|-------:|----:|---------|--------|
| `preprocessing/` | 39 | ~11 000 | Config/Result dataclass + batch | **100% реализован** |
| `algorithms/tangram/` | 4 | ~249 | hull→classify→inscribe | **100% реализован** |
| `algorithms/fractal/` | 5 | ~541 | 4 метода FD + chain code | **100% реализован** |
| `algorithms/` (остальные) | 37 | ~12 600 | Дескрипторы + matchers | **100% реализован** |
| `matching/` | 27 | ~8 300 | Score/Result dataclass | **100% реализован** |
| `assembly/` | 28 | ~8 650 | State + algorithms | **100% реализован** |
| `verification/` | 22 | ~7 860 | Violation/Report | **100% реализован** |
| `scoring/` | 13 | ~4 210 | Config/Entry/Summary | **100% реализован** |
| `utils/` | 103 | ~32 590 | Config/Entry/Summary | **100% реализован** |
| `io/` | 4 | ~1 140 | Config/Record | **100% реализован** |
| `ui/` | 2 | ~370 | OpenCV callbacks | **100% реализован** (минимально) |
| **Корневые** | 5 | ~1 700 | Dataclass/Pipeline | **100% реализован** |

---

## 4. Алгоритмический конвейер (Pipeline)

```
              ┌─────────────┐
              │  Изображения │
              │  фрагментов  │
              └──────┬───────┘
                     │
           ┌─────────▼──────────┐
  Этап 1   │  Загрузка и        │  normalize_color()
           │  нормализация цвета│  CLAHE + white balance + gamma
           └─────────┬──────────┘
                     │
           ┌─────────▼──────────┐
  Этап 2   │  Сегментация       │  segment_fragment()    — маска
           │  + контуры         │  extract_contour()     — контур + 4 края
           │  + ориентация      │  estimate_orientation() — угол текста
           └─────────┬──────────┘
                     │
           ┌─────────▼──────────┐
  Этап 3   │  Описание краёв    │  fit_tangram()           — TangramSignature
           │                    │  compute_fractal_signature() — FractalSignature
           │  Танграм:          │  build_edge_signatures()  — EdgeSignature[]
           │   hull→RDP→PCA     │
           │  Фрактал:          │  Формула синтеза:
           │   FD_box, FD_div,  │  B(t) = α·B_T(t) + (1-α)·B_F(t)
           │   IFS, CSS         │  α ∈ [0.3, 0.7], default 0.5
           └─────────┬──────────┘
                     │
           ┌─────────▼──────────┐
  Этап 4   │  Матрица           │  build_compat_matrix()
           │  совместимости     │  Score = w₁·CSS_sim + w₂·exp(-DTW) +
           │  N_edges × N_edges │         w₃·FD_score + w₄·OCR_coherence
           └─────────┬──────────┘
                     │
           ┌─────────▼──────────┐
  Этап 5   │  Сборка            │  8 алгоритмов (см. §4.1)
           │  (Assembly)        │  AssemblyRacer для параллельного запуска
           └─────────┬──────────┘
                     │
           ┌─────────▼──────────┐
  Этап 6   │  Верификация       │  verify_full_assembly()
           │  + отчёт           │  5 компонент → оценка A-F
           └─────────┬──────────┘
                     │
              ┌──────▼───────┐
              │  Результат:  │
              │  Assembly +  │
              │  PipelineResult │
              └──────────────┘
```

Pipeline реализован как класс `Pipeline` в `pipeline.py` (333 строки):
- Параллельная предобработка через `ThreadPoolExecutor` (настраиваемое число workers)
- Callback-хуки для мониторинга прогресса
- Воспроизводимость: конфиг сохраняется вместе с результатом
- `PipelineResult` содержит: `assembly`, `timer`, `config`, `n_input`, `n_output`, `timestamp`

### 4.1 Методы сборки (assembly) — подробно

| Метод | Модуль | LOC | Сложность | Подход |
|-------|--------|----:|-----------|--------|
| **Greedy** | `greedy.py` | 147 | O(E) | Nearest-edge: присоединяет по лучшему CompatEntry |
| **Simulated Annealing** | `annealing.py` | 143 | O(N·iter) | Metropolis exp(dE/T), swap/rotate/shift, геом. охлаждение |
| **Beam Search** | `beam_search.py` | 193 | O(B²·N·E) | B гипотез, расширение по лучшим CompatEntry |
| **Gamma Optimizer** | `gamma_optimizer.py` | 278 | O(iter·E·N) | MLE Gamma(k,θ) на DTW-дистанциях, MCMC |
| **Genetic Algorithm** | `genetic.py` | 297 | O(gen·pop·N²) | Order Crossover (OX), tournament, segment reversal |
| **Ant Colony** | `ant_colony.py` | 271 | O(ants·iter·N²) | τ^α·η^β, evaporation ρ, elite reinforcement |
| **MCTS** | `mcts.py` | ~100 | O(sim·rollout) | UCB1 = mean + c·√(ln(parent)/visits) |
| **Exhaustive** | `exhaustive.py` | 224 | O(N!·4^N) | Branch & Bound с pruning, N ≤ 8 |

Зависимости: `annealing`, `genetic`, `gamma_optimizer` используют `greedy` для начальной сборки.
`exhaustive` fallback на `beam_search` при N > 8.
`parallel.py` содержит `AssemblyRacer` для параллельного запуска всех методов.

### 4.2 Методы сопоставления краёв (matching) — подробно

| Метод | Модуль | Алгоритм | Метрика |
|-------|--------|----------|---------|
| **DTW** | `dtw.py` | Dynamic Time Warping, Sakoe-Chiba | dtw[n,m]/(n+m) |
| **CSS** | `fractal/css.py` | Gaussian smoothing × σ, zero crossings | Cosine distance |
| **SIFT** | `sift_matcher.py` | Lowe ratio test + RANSAC homography | Inlier count |
| **ICP** | `icp.py` | Iterative Closest Point | Mean point distance |
| **Color** | `color_match.py` | BGR/HSV/LAB histograms + moments | χ², Bhattacharyya |
| **Texture** | `texture_match.py` | LBP, Gabor filters | Histogram intersection |
| **Geometric** | `geometric_match.py` | Hu moments, aspect ratio, area ratio | L2 distance |
| **Spectral** | `spectral_matcher.py` | FFT magnitude/phase | Correlation |
| **Graph** | `graph_match.py` | MST, spectral ordering, random walk | Graph distance |
| **Boundary** | `boundary_matcher.py` | Hausdorff, Chamfer distance | exp(-dist) |
| **Affine** | `affine_matcher.py` | RANSAC/LMEDS affine estimation | Reprojection error |
| **Shape Context** | `shape_context.py` | Log-polar histograms, Hungarian matching | χ² distance |
| **Consensus** | `consensus.py` | Majority/weighted vote нескольких методов | Combined score |

### 4.3 Предобработка (preprocessing) — подробно

39 модулей, ~11 000 строк. Архитектурный паттерн: `XxxConfig` (dataclass) → функции → `XxxResult` (dataclass) + `batch_*()`.

**Ключевые алгоритмы**:

| Категория | Модули | Методы |
|-----------|--------|--------|
| Сегментация | `segmentation.py` | Otsu, Adaptive (mean/gaussian), GrabCut |
| Бинаризация | `adaptive_threshold.py`, `binarizer.py` | Otsu, Niblack (μ+k·σ), Sauvola (μ·(1+k·(σ/R-1))), Bernsen, энтропийный автовыбор |
| Контуры | `contour.py`, `contour_processor.py` | RDP (рекурсивный), arc-length resampling, Shoelace area |
| Денойзинг | `denoise.py` | Gaussian, Median, Bilateral, Non-Local Means, автовыбор по Laplacian σ |
| Цвет | `color_norm.py`, `color_normalizer.py` | CLAHE (LAB L-channel), Grey World, Max RGB, gamma LUT |
| Контраст | `contrast.py`, `contrast_enhancer.py` | CLAHE, HisEQ, Gamma, Percentile stretch, Single-Scale Retinex |
| Частоты | `frequency_filter.py`, `frequency_analyzer.py` | FFT: low/high/band-pass Gaussian, notch rejection, spectral entropy |
| Морфология | `morphology_ops.py` | Erosion, dilation, opening, closing, top-hat, black-hat |
| Перспектива | `perspective.py`, `warp_corrector.py` | Гомографические преобразования |
| Наклон | `deskewer.py`, `skew_correction.py` | Projection profile (variance max), HoughLinesP |
| Шум | `noise_analyzer.py`, `noise_filter.py`, `noise_reducer.py` | Laplacian variance, Wiener filter |
| Освещение | `illumination_corrector.py`, `illumination_normalizer.py` | Background subtraction, Homomorphic filtering, Multi-Scale Retinex |
| Рёбра | `edge_detector.py`, `edge_enhancer.py`, `edge_sharpener.py` | Adaptive Canny (σ-method), Sobel, LoG, Unsharp mask, Laplacian |
| Текстура | `texture_analyzer.py` | LBP, GLCM (contrast, homogeneity, energy, correlation) |
| Аугментация | `augment.py` | Random crop/rotate, Gaussian/salt-pepper noise, JPEG compress, scan simulate |

### 4.4 Верификация — система оценок

Модуль `confidence_scorer.py` вычисляет 5 компонент → итоговая оценка (A–F):

| Компонент | Вес | Метод |
|-----------|-----|-------|
| Edge compatibility | 0.30 | Средняя CSS + DTW + FD совместимость |
| Layout quality | 0.25 | Overlap + gap penalties |
| Coverage | 0.20 | Доля использованных фрагментов |
| Uniqueness | 0.15 | Штраф за дубликаты |
| Assembly score | 0.10 | Global assembly metric |

**Грейды**: A ≥ 0.85, B ≥ 0.70, C ≥ 0.55, D ≥ 0.40, F < 0.40

**Метрики для benchmark** (`metrics.py`):
- Neighbor Accuracy (NA) — процент правильных соседей
- Direct Comparison (DC) — точное совпадение позиций
- Position RMSE — среднеквадратичная ошибка позиций
- Angular Error — средняя ошибка поворота (градусы)
- Edge Match Rate — доля правильно совмещённых рёбер

---

## 5. CLI-инструменты — подробно

### 5.1 Основной CLI (`main.py`, 320 строк)

6-этапный pipeline: загрузка → обработка → матрица → сборка → OCR → экспорт.

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|:------------:|----------|
| `--input` | str | (обязательный) | Директория со сканами |
| `--output` | str | (обязательный) | Путь к выходному файлу |
| `--alpha` | float | 0.5 | Вес танграма vs фрактала (0..1) |
| `--n-sides` | int | 4 | Ожидаемое число краёв |
| `--seg-method` | str | `otsu` | otsu / adaptive / grabcut |
| `--threshold` | float | 0.3 | Минимальная совместимость |
| `--sa-iter` | int | 5000 | Итерации SA |
| `--visualize` | flag | — | OpenCV preview |

### 5.2 Инструменты (`tools/`, 6 файлов, ~1 640 строк)

| Инструмент | Файл | LOC | Назначение |
|------------|------|----:|------------|
| `puzzle-benchmark` | `benchmark.py` | 355 | Синтетические документы → разрыв → сборка → метрики (NA, DC, RMSE) |
| `puzzle-generate` | `tear_generator.py` | 303 | Генерация фрагментов с fBm noise (fractional Brownian motion) |
| `puzzle-evaluate` | `evaluate.py` | 312 | Полная оценка через Pipeline + HTML/JSON/Markdown отчёты |
| `puzzle-mix` | `mix_documents.py` | 272 | Смешивание фрагментов + оценка кластеризации (Purity, Rand, ARI) |
| `puzzle-server` | `server.py` | 310 | Flask REST API (6 endpoints, thread-safe jobs) |
| `puzzle-profile` | `profile.py` | 388 | Stage-level timing + cProfile для 7 этапов |

---

## 6. REST API (Flask-сервер) — подробно

Реализовано в `tools/server.py` (310 строк). Thread-safe хранение задач.

| Метод | Эндпоинт | Описание |
|-------|----------|----------|
| GET | `/health` | Healthcheck: версия, uptime, число задач |
| GET | `/config` | Текущая конфигурация (JSON) |
| POST | `/api/reconstruct` | Upload multipart/form-data → реконструкция → placements JSON |
| POST | `/api/cluster` | Upload → кластеризация → document membership |
| GET | `/api/report/<job_id>` | JSON отчёт по задаче |
| GET | `/api/report/<job_id>/html` | HTML отчёт с визуализацией |

---

## 7. Конфигурационная система — подробно

`puzzle_reconstruction/config.py` (151 строка) — 7 `dataclass`-секций:

| Секция | Поля | Default | Описание |
|--------|------|---------|----------|
| `SegmentationConfig` | `method`, `morph_kernel` | otsu, 3 | Сегментация |
| `SynthesisConfig` | `alpha`, `n_sides`, `n_points` | 0.5, 4, 128 | Синтез дескрипторов |
| `FractalConfig` | `n_scales`, `ifs_transforms`, `css_n_sigmas`, `css_n_bins` | 8, 8, 7, 32 | Фрактальный анализ |
| `MatchingConfig` | `threshold`, `dtw_window` | 0.3, 20 | Сопоставление |
| `AssemblyConfig` | `method`, `beam_width`, `sa_iter`, `sa_T_max/min`, `sa_cooling`, `gamma_iter`, `seed` | beam, 10, 5000, 1000/0.1, 0.995, 3000, 42 | Сборка |
| `VerificationConfig` | `run_ocr`, `ocr_lang`, `export_pdf` | True, "rus+eng", False | OCR |

Поддержка: JSON + YAML (pyyaml), override через `apply_overrides(**kwargs)`.

---

## 8. Модели данных (`models.py`) — подробно

| Класс/Enum | Поля | Описание |
|-------------|------|----------|
| `ShapeClass` | 7 значений | triangle, rectangle, trapezoid, parallelogram, pentagon, hexagon, polygon |
| `EdgeSide` | 5 значений | top, bottom, left, right, unknown |
| `FractalSignature` | `fd_box`, `fd_divider`, `ifs_coeffs` (M,), `css_image`, `chain_code`, `curve` (N,2) | Фрактальное описание края |
| `TangramSignature` | `polygon` (K,2), `shape_class`, `centroid` (2,), `angle`, `scale`, `area` | Геометрическое описание |
| `EdgeSignature` | `edge_id`, `side`, `virtual_curve` (N,2), `fd`, `css_vec`, `ifs_coeffs`, `length` | Уникальная подпись края |
| `Fragment` | `fragment_id`, `image` (H,W,3), `mask` (H,W), `contour` (N,2), `tangram?`, `fractal?`, `edges[]` | Физический фрагмент |

---

## 9. Тестирование — подробно

### 9.1 Результаты запуска тестов (2026-02-25, текущая ветка — после Фаз 8–12 / v1.0.0)

```
Собрано тестов:      42 476
Ошибки сбора:            0
Пройдено:            42 476  (100%)
Провалено:               0   (0%)
Пропущено:               0   (scikit-learn 1.8.0 установлен — clustering тесты активны)
xpassed:                 0   (xfail-маркеры Laplacian удалены)
Предупреждений:         ~9   (RankWarning/RuntimeWarning устранены)

Новые тесты (Фазы 8–11):
  test_suite_extended.py       +82  (12 новых валидаторов)
  test_main_export_report.py   +31  (--validators / --export-report)
  test_integration_v2.py       +20  (@integration E2E)
  Итого новых:                +133
```

**Примечание:** 4 ImportError (Edge/Placement) исправлены на текущей ветке
в коммите `c3c44c3`. Все ранее провальные тесты исправлены в серии fix-коммитов
(`b8b15f4`, `c03f6b4`, `d896c56`, `b9b8e36`).

### 9.2 Категоризация 822 тестовых файлов

| Категория | Тестовых файлов | Покрываемый модуль |
|-----------|----------------:|-------------------|
| `algorithms/` | ~84 | Все дескрипторы, tangram, fractal |
| `assembly/` | ~47 | Все 8 алгоритмов + утилиты |
| `matching/` | ~48 | DTW, color, SIFT, ICP, consensus, ... |
| `preprocessing/` | ~54 | Сегментация, контуры, денойз, ... |
| `scoring/` | ~24 | Все 12 scoring-модулей |
| `verification/` | ~28 | OCR, boundary, completeness, confidence |
| `utils/` | ~169 | Геометрия, кэш, метрики, events, ... |
| `io/` | ~5 | Загрузка, экспорт, метаданные |
| Кросс-модульные | ~361 | config, models, main, pipeline, export, ... |
| **Интеграционные** | 4 | `test_integration.py`, `test_integration_v2.py`, `test_pipeline.py`, ... |
| **Верификация (новые)** | 1 | `test_suite_extended.py` — 82 теста 12 новых валидаторов |
| **CLI (новые)** | 1 | `test_main_export_report.py` — 31 тест --validators/--export-report |

Значительная часть тестовых файлов — `_extra` варианты (дополнительные edge-case тесты).

### 9.3 Организация тестов

- **Паттерн**: `class Test*` с группировкой по функции, docstrings на русском
- **Фикстуры**: синтетические numpy-массивы (изображения, маски, контуры), seeded RNG
- **Assertions**: `pytest.approx()`, `np.testing.assert_allclose()`, `isinstance`, `.shape`, `.dtype`
- **Нет внешних тестовых данных** — всё генерируется в коде

### 9.4 Ошибки импорта — исправлены

Ранее 4 файла не загружались из-за отсутствия `Edge`/`Placement` в `models.py`.
Исправлено в коммите `c3c44c3`:
- `Edge(edge_id, contour, text_hint)` добавлен в `models.py`
- `Placement(fragment_id, position, rotation)` добавлен в `models.py`

Все 4 файла (`test_confidence_scorer.py`, `test_graph_match.py`,
`test_layout_verifier.py`, `test_parallel.py`) теперь загружаются корректно.

### 9.5 Провальные тесты — устранены

Все ранее провальные тесты исправлены серией fix-коммитов:

| Коммит | Что исправлено |
|--------|----------------|
| `b8b15f4` | Устранены противоречивые тесты `TestFilterGapMeasures` |
| `c03f6b4` | Устранён нестабильный тест `TestGaussianFilter::test_constant_image_unchanged` |
| `d896c56` | Устранены 3 источника `RuntimeWarning`/`DeprecationWarning` |
| `b9b8e36` | Устранены оставшиеся `RankWarning` и `RuntimeWarning` |

**Итог: 0 провальных тестов из 42 404+.**

### 9.6 Интеграционные тесты

**`test_integration.py`** — генерирует синтетический документ, рвёт на 4 фрагмента, тестирует:
- Preprocessing: сегментация + контуры (4 теста)
- Algorithms: tangram + fractal + edges (4 теста)
- Matching: матрица совместимости (3 теста)
- Assembly: greedy + SA + beam (4 теста)
- Metrics: perfect vs random reconstruction (3 теста)
- Config: serialization round-trip (3 теста)

**`test_integration_v2.py`** *(NEW — Фаза 11)* — E2E тесты для Фаз 8–10:
- `TestSuiteE2EOnRealAssembly` — run_all() на реальных фрагментах (8 тестов)
- `TestExportReportE2E` — export JSON/MD/HTML на реальном отчёте (4 теста)
- `TestVerificationConfigE2E` — Config.verification roundtrip + all_validator_names() (5 тестов)
- `TestMultiMethodVerificationE2E` — completeness=1.0, zero-fragment edge case (3 теста)

**`test_verification_report_methods.py`** *(NEW — Фаза 12)* — сериализация VerificationReport:
- `TestAsDictMethod` — 8 тестов `as_dict()`
- `TestToJsonMethod` — 7 тестов `to_json()`
- `TestToMarkdownMethod` — 8 тестов `to_markdown()`
- `TestToHtmlMethod` — 10 тестов `to_html()`
- `TestPipelineVerifySuite` — 5 тестов `Pipeline.verify_suite()`
- `TestPipelineResultVerificationReport` — 5 тестов `PipelineResult.verification_report`

**`test_main_list_validators.py`** *(NEW — Фаза 12)* — `--list-validators` и `all_validator_names()`:
- `TestListValidatorsParserFlag` — 4 теста (регистрация флага в parser)
- `TestListValidatorsOutput` — 8 тестов (вывод main(), не вызывает run())
- `TestAllValidatorNamesContract` — 8 тестов (stable, 21, no dups, 9+12)

**`test_pipeline.py`** — тестирует класс `Pipeline`:
- Init: config, workers, logger, callbacks (5 тестов)
- Preprocess: параллелизм, edge/fractal computation (8 тестов)
- Match: matrix shape, entries sorted (3 теста)
- Assemble: greedy/beam/exhaustive, invalid method (5 тестов)
- Verify: OCR disabled → 0.0 (2 теста)
- Run: full e2e → PipelineResult (6 тестов)

### 9.7 CI/CD (GitHub Actions)

Файл: `.github/workflows/ci.yml`

| Job | Описание | Блокирует CI? |
|-----|----------|:-------------:|
| `test` | Python 3.11 + 3.12 на Ubuntu — unit tests (`-x` fail-fast) | **Да** |
| `test` (integration) | `test_integration.py` + `test_pipeline.py` | Нет (`continue-on-error`) |
| `lint` | `ruff check` + `mypy` (3 файла) | Нет (`continue-on-error`) |
| `build` | `python -m build` → wheel + sdist + `twine check` | **Да** (зависит от test) |
| Coverage | pytest-cov → XML → Codecov | Нет (informational) |

---

## 10. Качество кода — подробно

### 10.1 Линтинг и типизация

| Инструмент | Конфигурация | Статус |
|------------|-------------|--------|
| **ruff** | line-length=100, target=py311, select=[E,F,W,I,N,UP,RUF], ignore=[E501] | Настроен, `continue-on-error` |
| **mypy** | python_version=3.11, warn_unreachable, no_implicit_optional | **50+ модулей** (Фаза 9): verification/*(21), assembly/*(10), matching/*(4), algorithms/*(8), корневые(5), utils/*.check, preprocessing/*.check, tools.* |
| **isort** (ruff) | known-first-party: puzzle_reconstruction, tools | Настроен |

### 10.2 Маркеры технического долга

| Маркер | Количество в production-коде |
|--------|:---:|
| `TODO` / `FIXME` / `HACK` / `XXX` / `NotImplemented` | **0** |

### 10.3 Архитектурные паттерны

| Паттерн | Где используется | Описание |
|---------|-----------------|----------|
| **Config/Result dataclass** | preprocessing, algorithms, matching, scoring, utils | Входные параметры в dataclass с `__post_init__` validation |
| **Config/Entry/Summary triple** | utils/*_score_utils, scoring/* | Config → list[Entry] → Summary с агрегацией |
| **Batch processing** | Все модули | `batch_*()` функции для пакетной обработки |
| **Dispatcher pattern** | preprocessing (auto_*), assembly, matching | Одна функция выбирает алгоритм по конфигу |
| **Immutable state** | assembly_state.py | Copy-on-write через deepcopy |
| **Pub/Sub** | event_bus.py | Topic-based with wildcard subscriptions |
| **LRU Cache** | cache.py, cache_manager.py | SHA-256 keying, TTL support, thread-safe |

### 10.4 Язык кодовой базы

- Все docstrings, комментарии и документация — на **русском языке**
- Имена переменных, функций, классов — на **английском языке**
- README.md, PUZZLE_RECONSTRUCTION.md — на **русском языке**

---

## 11. Зависимости

### 11.1 Основные (`dependencies` в pyproject.toml)

| Пакет | Минимальная версия | Использование |
|-------|:------------------:|---------------|
| numpy | >=1.24 | Массивы, линалг, FFT |
| scipy | >=1.11 | gaussian_filter, gamma.fit, optimize |
| opencv-python | >=4.8 | cv2: SIFT, contours, morphology, drawing |
| scikit-image | >=0.22 | Segmentation, filters |
| Pillow | >=10.0 | Image I/O, PDF fallback |
| scikit-learn | >=1.3 | KMeans, GMM, SpectralClustering, silhouette |

### 11.2 Дополнительные (`requirements.txt`)

| Пакет | Использование |
|-------|---------------|
| shapely >=2.0 | Polygon intersection, IoU |
| matplotlib >=3.7 | Визуализация (в tools) |
| networkx >=3.1 | Graph match, MST |
| pytesseract >=0.3.10 | OCR verification |

### 11.3 Опциональные группы (pyproject.toml)

| Группа | Пакеты |
|--------|--------|
| `ocr` | pytesseract >=0.3 |
| `yaml` | pyyaml >=6.0 |
| `pdf` | reportlab >=4.0, fpdf2 >=2.7 |
| `api` | flask >=3.0 |
| `dev` | pytest >=7.4, pytest-cov >=4.1, ruff >=0.3, mypy >=1.8, types-Pillow |
| `all` | Все вышеуказанные |

---

## 12. Кластеризация фрагментов (`clustering.py`)

Разделение фрагментов из смешанных документов. 316 строк.

**28-мерный вектор признаков на фрагмент**:
- Fractal dimensions (2D)
- CSS-дескрипторы (8D)
- Brightness histogram (8D)
- Edge gradient histogram (8D)
- Shape factors (2D: area, scale)

**Методы кластеризации**:
- **GMM** (Gaussian Mixture Model) — вероятностный
- **KMeans** — дистанционный
- **Spectral** — RBF affinity kernel

**Автовыбор K**: BIC + silhouette score (порог ≥ 0.10)

---

## 13. Экспорт результатов (`export.py`)

474 строки. Форматы:

| Функция | Выход | Описание |
|---------|-------|----------|
| `render_canvas()` | np.ndarray | Рендер сборки с rotation + mask blending |
| `render_heatmap()` | np.ndarray | Gaussian-kernel confidence visualization |
| `render_mosaic()` | np.ndarray | Grid thumbnails, sorted by (y, x) |
| `save_png()` | файл | PNG/JPEG с настройкой quality |
| `save_pdf()` | файл | PDF через reportlab (text layer) или Pillow (fallback) |
| `comparison_strip()` | np.ndarray | Side-by-side: fragments / assembly / heatmap |

---

## 14. Utils — подробная структура (103 модуля, 32 590 LOC)

| Функциональная область | Файлов | LOC | Классов | Функций |
|------------------------|-------:|----:|--------:|--------:|
| Геометрия и координаты | 8 | 2 680 | 7 | 107 |
| Метрики и scoring | 16 | 4 877 | 38 | 231 |
| Сигналы и данные | 10 | 3 286 | 6 | 99 |
| Изображения и I/O | 8 | 2 493 | 14 | 90 |
| Кэширование и state | 5 | 1 621 | 13 | 98 |
| Событийная шина и логи | 5 | 1 583 | 20 | 97 |
| Контуры и дескрипторы | 6 | 1 991 | 11 | 81 |
| Цвет и рёбра | 6 | 1 892 | 14 | 77 |
| Кластеризация и группы | 6 | 2 185 | 12 | 76 |
| Ранжирование и voting | 7 | 2 007 | 22 | 102 |
| Пайплайн и обработка | 5 | 1 544 | 18 | 88 |
| Спец. утилиты | 14 | 4 837 | 34 | 207 |
| Визуализация и вывод | 4 | 1 591 | 6 | 45 |
| **ИТОГО** | **103** | **32 590** | **215** | **1 398** |

---

## 15. Техническая документация (`PUZZLE_RECONSTRUCTION.md`)

802 строки. Полная техническая документация системы:

| Раздел | Содержание |
|--------|-----------|
| §1 Постановка задачи | N фрагментов → O(N!·4^N) NP-complete задача |
| §2 Архитектура | 5-модульный pipeline (ASCII-диаграмма) |
| §3 Алгоритм 1: Танграм | Segmentation → Hull → RDP → Classification → Fitting → Normalization |
| §4 Алгоритм 2: Фрактал | Box-counting, Divider (Richardson), IFS Барнсли, CSS (MPEG-7) |
| §5 Синтез подписи | B_virtual(t) = α·B_T(t) + (1-α)·B_F(t), α ∈ [0.3, 0.7] |
| §6 Сопоставление | Score = w₁·CSS + w₂·DTW + w₃·FD + w₄·OCR; зеркальная комплементарность |
| §7 Глобальная сборка | Greedy O(N²K), SA, Gamma, Beam Search |
| §8 Псевдокод | Полный pseudocode: RECONSTRUCT_DOCUMENT, fit_tangram, compute_fractal_signature |
| §9 Реализация | Структура проекта + рабочие примеры кода (box-counting, CSS) |
| §10 Диаграмма | ASCII-art пайплайн от сканера до монитора |
| §11 Аналоги | Unshredder, Fraggler (JOSS 2024), RePAIR (Horizon 2020), DARPA 2011 |
| §12 Исторический контекст | Свитки Мёртвого моря, Оксиринхские папирусы, архив Штази (~600 мешков), нейросети |
| §13 Литература | 8 ключевых статей (CVPR 2020, arXiv 2024, JMIV 2022) + открытые датасеты |
| Приложение А | Таблица параметров с дефолтами |

---

## 16. История разработки — подробная хронология

| Дата | Коммиты | Событие |
|------|---------|---------|
| 2026-02-20 | `866f90d` | Initial commit: базовая структура |
| 2026-02-20–21 | iter-83 → iter-128 | 46 итераций: по 4 теста + 1 production-модуль |
| 2026-02-21 | `a217d9e` | **Merge PR #1** (svend4): checkpoint iter-128 |
| 2026-02-21–22 | iter-129 → iter-141 | 13 итераций |
| 2026-02-22 | `59e19a1` | **Merge PR #2** (svend4): checkpoint iter-141 |
| 2026-02-22–23 | iter-142 → iter-169 | 28 итераций |
| 2026-02-23 | `cfce33b` | **Merge PR #3** (svend4): checkpoint iter-169 |
| 2026-02-23 | `cdceb07` | docs: DEV_STATUS.md создан |
| 2026-02-23–24 | iter-170 → iter-249 | **80 итераций на `main`**: +318 тестов, +28 utils |
| 2026-02-24 | **PR #4–#8** | Merge дополнительных тестов и документации |
| 2026-02-24 | `c3c44c3` | **fix: добавлены Edge/Placement** в models.py |
| 2026-02-24 | `6c98327` | **feat: интеграция всех 8 алгоритмов + matcher_registry** (Мост №1) |
| 2026-02-24 | **PR #9, #10** | Merge интеграции и INTEGRATION_ROADMAP.md |
| 2026-02-25 | `e6149ec` | docs: обновление DEV_STATUS.md (спящий код → мосты) |
| 2026-02-25 | `31fae10` | feat: release v0.4.0-beta — Docker, CI blocking, Makefile, CHANGELOG, OpenAPI |
| 2026-02-25 | `b9b8e36` | fix: eliminate remaining RankWarning and RuntimeWarning |
| 2026-02-25 | `c03f6b4` + `b8b15f4` | fix: resolve flaky/contradictory tests |
| 2026-02-25 | `d896c56` | fix: eliminate 3 RuntimeWarning/DeprecationWarning sources |
| 2026-02-25 | `f33d429` | docs: mark all 7 integration phases complete in INTEGRATION_ROADMAP.md |
| 2026-02-25 | `63a4332` | feat: Phases 6+7 — Infrastructure Utils + Research Mode |
| 2026-02-25 | `a9962c6` | Merge branch 'main' into claude/puzzle-text-docs-3tcRj |
| 2026-02-25 | `d186260` | feat: integrate sleeping modules scoring/ and io/ |
| 2026-02-25 | **PR #15** | Merge PR #15 — финальная интеграция |
| 2026-02-25 | `b1ac4de` | **feat: integrate all 7 phases — full pipeline connection** |
| 2026-02-25 | `629bf44` | docs: fix incorrect merge conflict resolution (DEV_STATUS + STATUS) |
| 2026-02-25 | `fe2b0d9` | docs: restore correct STATUS.md v0.4.0-beta after repeated incorrect merge |
| 2026-02-25 | `369524b` | **feat: activate all 21 verifiers + expand mypy coverage (Фазы 8–9)** |
| 2026-02-25 | `a411c54` | test+fix: 82 тестов для 12 новых валидаторов + fix boundary validator |
| 2026-02-25 | `bedfa4f` | **feat: --validators и --export-report CLI options (Фаза 10)** |
| 2026-02-25 | `a591621` | **test+fix: E2E integration tests (Фаза 11) + completeness validator fix** |
| 2026-02-25 | `c3029be` | docs: update STATUS.md / DEV_STATUS.md — Phases 8–11 |
| 2026-02-25 | `821ccee` | **feat(v1.0.0): VerificationReport API, Pipeline.verify_suite, --list-validators (Фаза 12)** 🏷️ |
| 2026-02-25 | `5064075` | docs+test: --list-validators tests, STATUS/DEV_STATUS → v1.0.0 |
| 2026-02-25 | `aec8b77` | **feat: /api/validators endpoint, README verification docs, CHANGELOG fix** ✅ |

**Паттерн разработки**: каждая из ~250 итераций добавляла:
1. 4 новых тестовых файла (по одному на подсистему)
2. 1 новый production-модуль (обычно в `utils/`)
3. Обновление `utils/__init__.py` с новыми экспортами

**Workflow**: Claude генерирует код на feature-ветке → svend4 ревьюит и мёржит через PR.

---

## 17. Оценка зрелости — подробно

### 17.1 Сильные стороны

- **Алгоритмическое ядро**: два полных алгоритма описания краёв (Танграм + Фрактал) с математически обоснованным синтезом
- **8 методов сборки**: от Greedy O(E) до Exhaustive O(N!), включая метаэвристики (SA, Genetic, ACO, MCTS)
- **13 методов сопоставления**: DTW, SIFT, ICP, Color, Texture, Spectral, Graph, Shape Context, Consensus
- **Обширная предобработка**: 39 модулей, 6 методов бинаризации, 4 денойзера, частотная фильтрация, коррекция перспективы
- **Тестовое покрытие**: 42 208+ тестов, 100% pass rate, отношение тест/код = 2.87:1
- **0 маркеров TODO/FIXME** в production-коде
- **Полная документация**: 802 строки техдока с формулами и pseudocode
- **REST API**: 6 endpoints с thread-safe job storage
- **7 CLI-инструментов**: reconstruct, benchmark, generate, mix, server, evaluate, profile
- **Кластеризация**: разделение смешанных документов (28D feature vector, 3 метода)
- **Интерактивный UI**: OpenCV viewer с drag/rotate/undo/zoom/auto-assembly

### 17.2 Проблемы и технический долг

| Аспект | Текущее состояние | Серьёзность | Рекомендация |
|--------|-------------------|:-----------:|--------------|
| **Windows/macOS CI** | Закомментировано | Низкая | Включить при необходимости |
| **UI минимальный** | Только OpenCV viewer | Низкая | Рассмотреть веб-интерфейс |
| **E2E на реальных данных** | Только синтетические фрагменты | Средняя | Набор реальных сканов для валидации |

> **Закрытые долги (Фазы 8–12):**
> - ~~mypy частичный (3/305)~~ → **50+ модулей** строгой типизации
> - ~~Верификаторов активно 9/21~~ → **21/21** + `run_all()` + CLI-опции
> - ~~нет API сериализации VerificationReport~~ → `as_dict/to_json/to_markdown/to_html`
> - ~~Pipeline не имеет verify_suite()~~ → **интегрировано** + `PipelineResult.verification_report`
> - ~~версия Beta~~ → **v1.0.0 Production/Stable** 🎉

### 17.3 Итоговая оценка (обновлено 2026-02-25)

#### Текущая ветка `claude/puzzle-text-docs-3tcRj` (v0.4.0-beta):

```
Стадия:            Beta (0.4.0)

Реализация ядра:   ██████████ 100%  — 2 алгоритма описания + 8 методов сборки + 13 matchers
Подключение к CLI: ██████████ 100%  — 10/10 (8 алгоритмов + auto + all)
Matching:          ██████████ 100%  — 13+ матчеров через matcher_registry
Preprocessing:     ██████████ 100%  — 38/38 через PreprocessingChain
Verification:      ██████████ 100%  — 21/21 активны (VerificationSuite, run_all(), --validators all)
Тестирование:      ██████████ 100%  — 42 290/42 290 pass (100%), 0 ImportError
Документация:      █████████░  90%  — техдок + README + DEV_STATUS + STATUS + ROADMAP + REPORT
CI/CD:             █████████░  90%  — test блокирует, Docker, Makefile, CHANGELOG
Инструментарий:    █████████░  90%  — 7 CLI + REST API + benchmark + profiler
Деплой:            ████████░░  80%  — Docker + CI готовы, нет production-деплоя
```

**Общая стадия**: Проект выпущен в **Production/Stable (v1.0.0)** 🎉
Все 12 фаз интеграции выполнены. **42 476 тестов проходят**, 0 провалено.
21/21 верификаторов активны. mypy 50+ модулей. Git-тег `v1.0.0`.
Docker, CI/CD, CHANGELOG, OpenAPI, Makefile, E2E-тесты добавлены.
VerificationReport API + Pipeline.verify_suite() + --list-validators в составе релиза.

> Закрыты ~~Активировать 12 верификаторов~~, ~~Расширить mypy покрытие~~,
> ~~Выпустить v1.0.0~~ — **всё выполнено!**

---

## 18. Глубокий анализ реализации (верификация запуском)

### 18.1 Результаты runtime-проверки

Каждый ключевой модуль был импортирован и выполнен с реальными данными:

| Проверка | Результат | Детали |
|----------|:---------:|--------|
| `Config()` → `to_dict()` | **PASS** | Все 6 секций конфигурации создаются корректно |
| `Pipeline(Config())` | **PASS** | Импорт и создание экземпляра без ошибок |
| `Fragment(image, mask, contour)` | **PASS** | Модели данных работают |
| `segment_fragment(image)` | **PASS** | Возвращает маску (200,200) uint8, значения {0, 255} |
| `box_counting_fd(curve)` | **PASS** | Возвращает FD = 1.0 (для синусоиды) |
| `dtw_distance(a, b)` | **PASS** | Возвращает 0.683 (для случайных кривых 50×2 и 60×2) |
| `import greedy_assembly` | **PASS** | Все 8 алгоритмов сборки импортируются |
| `import simulated_annealing` | **PASS** | |
| `import beam_search` | **PASS** | |
| pytest (5 файлов, 208 тестов) | **207 pass, 1 fail** | Единственный провал: DTW triangle inequality (DTW не метрика) |

### 18.2 Классификация реализации ядра (18 ключевых модулей)

Каждый файл прочитан полностью и классифицирован:

| Модуль | LOC | Вердикт | Обоснование |
|--------|----:|:-------:|-------------|
| `tangram/hull.py` | 62 | **THIN+REAL** | `convex_hull` и `rdp_simplify` — обёртки cv2 (3-4 строки); `normalize_polygon` — настоящий PCA через SVD (20 строк) |
| `fractal/box_counting.py` | 87 | **REAL** | Учебная реализация: нормализация [0,1], подсчёт ячеек через set(zip), polyfit log-log, clip [1.0, 2.0] |
| `fractal/ifs.py` | 149 | **REAL** | Настоящий IFS Барнсли: проекция на перпендикуляр хорды, сегментация, регрессия, |d_k| < 1 |
| `fractal/css.py` | 179 | **REAL** | Полный CSS (MPEG-7): Gaussian smoothing × σ, вычисление кривизны, zero-crossing detection |
| `fractal/divider.py` | 99 | **REAL** | Richardson compass walk с интерполяцией |
| `synthesis.py` | 140 | **REAL** | Синтез: ресемплинг обеих кривых, нормализация, α-взвешивание, CSS per edge |
| `matching/dtw.py` | 59 | **REAL** | Ручной DTW с Sakoe-Chiba band, O(n·w), не обёртка библиотеки |
| `matching/pairwise.py` | 78 | **REAL** | 4 сигнала с явными весами (CSS 0.35, DTW 0.30, FD 0.20, text 0.15) + length penalty |
| `assembly/greedy.py` | 147 | **REAL** | Жадная сборка с геометрическим размещением |
| `assembly/annealing.py` | 143 | **REAL** | SA: swap/rotate/shift, Metropolis exp(dE/T), геом. охлаждение |
| `assembly/beam_search.py` | 193 | **REAL** | Hypothesis dataclass, expand по лучшим CompatEntry, 2D rotation/translation |
| `assembly/genetic.py` | 297 | **REAL** | Order Crossover (OX), tournament selection, 3 вида мутаций, elitism |
| `assembly/ant_colony.py` | 271 | **REAL** | τ^α·η^β, evaporation ρ, elite reinforcement, вероятностный переход |
| `assembly/mcts.py` | 293 | **REAL** | UCB1, 4 фазы (Selection, Expansion, Simulation, Backpropagation) |
| `verification/ocr.py` | 164 | **REAL** | Strip extraction + pytesseract + quality scoring; graceful degradation |
| `pipeline.py` | 334 | **REAL** | ThreadPoolExecutor, 7 шагов предобработки, 5 методов сборки, PipelineResult |
| `clustering.py` | 317 | **REAL** | 28D features, BIC+silhouette для K, GMM/KMeans/Spectral |
| `main.py` | 321 | **REAL** | Полный argparse CLI, 6-этапный pipeline, config file + override |

**Итог ядра: 17 из 18 модулей = REAL (настоящие алгоритмы, написанные вручную)**

### 18.3 Классификация вспомогательного кода (utils/, 103 модуля)

Анализ 27 модулей (20 случайных + 7 целевых):

| Категория | Доля | Примеры |
|-----------|-----:|---------|
| **SUBSTANTIAL** (настоящие алгоритмы, 10+ строк логики) | 15% | `geometry.py` (Sutherland-Hodgman, winding number), `icp_utils.py` (SVD, nearest-neighbor), `topology_utils.py` (Graham scan, flood-fill BFS), `spatial_index.py` (grid-based kNN), `graph_utils.py` (Dijkstra, Prim MST), `cache.py` (LRU+SHA-256), `feature_selector.py` (PCA via SVD) |
| **MODERATE** (реальные, но простые, 5-15 строк) | 60% | `event_bus.py` (pub/sub), `profiler.py` (step timing), `visualizer.py` (compositing), `rotation_utils.py` (SVD Procrustes), `threshold_utils.py` (Otsu), `contour_utils.py` (arc-length resampling) |
| **THIN** (обёртки 1-4 строки) | 25% | `mask_utils.py` (cv2.erode/dilate), `normalization_utils.py` (np.linalg.norm), `sampling_utils.py` (rng.choice), `config_manager.py` (dict ops), `sparse_utils.py` (np.where) |

### 18.4 Признаки шаблонной генерации в utils/

| Индикатор | Значение |
|-----------|----------|
| Файлов с идентичным dataclass-шаблоном (`Config` + `__post_init__` + функции + `batch_`) | **49/102 (48%)** |
| Функций с ≤ 3 statements (без docstring) | **59.6%** (852/1429) |
| Однострочных return-функций | **31.1%** (445/1429) |
| `batch_*` функций (тривиальные `[f(x) for x in items]`) | **78 шт. в 66 файлах** |
| Дублирование между файлами | `symmetrize_matrix`, `threshold_matrix`, `diagonal_zeros` — реализованы в 2-3 файлах |
| Разброс размеров файлов | 194–471 строк, медиана 326 — аномально однородно |
| Файлов с идентичными `# ─── Section ───` разделителями | **81%** |

#### 18.4.1 Конкретные примеры дублирования

**`cosine_distance`** — две разные реализации:

```python
# descriptor_utils.py:67 — возвращает [0, 1]
def cosine_distance(a, b, eps=1e-8):
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return (1.0 - cos) / 2.0

# distance_utils.py:77 — возвращает [0, 2]
def cosine_distance(a, b):
    return float(np.clip(1.0 - cosine_similarity(a, b), 0.0, 2.0))
```

**`normalize_contour`** — две разные реализации:

```python
# geometry.py:311 — нормализация по диагонали bbox
def normalize_contour(pts, target_scale=1.0): ...

# contour_sampler.py:401 — нормализация в [-1, 1] по max abs
def normalize_contour(contour): ...
```

**`normalize_profile`** — дублирование в `contour_profile_utils.py:141` и `edge_profiler.py:208`.

#### 18.4.2 Конкретные примеры batch-инфляции

```python
# blend_utils.py:354 — 13 строк (docstring) ради 1 строки логики:
def batch_blend(pairs, alpha=0.5):
    return [alpha_blend(src, dst, alpha) for src, dst in pairs]

# descriptor_utils.py:270 — 6 строк ради 1 строки:
def batch_nn_match(query_sets, train_set, metric="l2"):
    return [nn_match(q, train_set, metric) for q in query_sets]
```

#### 18.4.3 Конкретные примеры thin-обёрток

```python
# descriptor_utils.py:62 — 1 строка логики
def l2_distance(a, b):
    return float(np.linalg.norm(a - b))

# color_utils.py:61 — 2 строки логики
def to_hsv(img):
    bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

# descriptor_utils.py:84 — 1 строка логики
def l1_distance(a, b):
    return float(np.sum(np.abs(a - b)))
```

#### 18.4.4 Идентичный шаблон валидации в 15+ файлах

```python
# annealing_schedule.py:66
raise ValueError(f"n_steps должен быть >= 1, получено {self.n_steps}")
# batch_processor.py
raise ValueError(f"max_retries должен быть >= 0, получено {self.max_retries}")
# graph_utils.py:36
raise ValueError(f"src должен быть >= 0, получено {self.src}")
# contour_sampler.py
raise ValueError(f"corner_threshold должен быть >= 0, получено {corner_threshold}")
# edge_scorer.py
raise ValueError(f"length_tol должен быть >= 0, получено {self.length_tol}")
```

Все — одна и та же f-string-формула: `"{name} должен быть >= {N}, получено {value}"`.

### 18.5 Реальный объём уникальной логики (переоценка)

| Слой | Номинально | Реально уникального кода | Коэффициент |
|------|:----------:|:------------------------:|:-----------:|
| **Ядро** (algorithms, matching, assembly, verification, pipeline) | ~38 800 LOC | ~25 000–30 000 LOC | 0.70 |
| **Предобработка** (preprocessing) | ~11 000 LOC | ~6 000–8 000 LOC | 0.65 |
| **Утилиты** (utils) | ~32 590 LOC | ~5 000–8 000 LOC | 0.20 |
| **Scoring** | ~4 210 LOC | ~2 000–3 000 LOC | 0.60 |
| **IO + UI + корневые** | ~4 850 LOC | ~3 000–4 000 LOC | 0.70 |
| **ИТОГО production** | **~91 180 LOC** | **~41 000–53 000 LOC** | **0.50** |

### 18.6 Инструменты (tools/) — все REAL

| Файл | LOC | Вердикт | Обоснование |
|------|----:|:-------:|-------------|
| `server.py` | 309 | **REAL** | Flask, 6 endpoints, thread-safe jobs, multipart upload |
| `benchmark.py` | 355 | **REAL** | Ground-truth tracking, NA/DC/RMSE метрики |
| `tear_generator.py` | 303 | **REAL** | fBm noise (fractional Brownian motion), multi-octave |
| `evaluate.py` | 312 | **REAL** | Pipeline + HTML/JSON/Markdown отчёты |
| `mix_documents.py` | 272 | **REAL** | Кластеризация + Purity/Rand/ARI оценка |
| `profile.py` | 388 | **REAL** | cProfile, 7-этапное профилирование |

### 18.7 Проблемы тестирования

#### Ошибки импорта (4 файла не загружаются)

```
ImportError: cannot import name 'Edge' from 'puzzle_reconstruction.models'
ImportError: cannot import name 'Placement' from 'puzzle_reconstruction.models'
```

Тесты ссылаются на классы `Edge` и `Placement`, которых нет в `models.py`.
Это означает рассинхрон между тестами и кодом — тесты генерировались
отдельно или классы планировались, но не были реализованы.

#### Некорректный тест DTW

```
FAILED test_dtw.py::test_triangle_inequality
```

DTW **не является метрикой** — неравенство треугольника для DTW не выполняется
(это хорошо известный факт). Тест проверяет математическое свойство,
которым DTW не обладает. Это указывает на автоматическую генерацию тестов
без проверки математической корректности утверждений.

### 18.8 Итоговый диагноз

Проект содержит **два слоя разного качества**:

1. **Ядро** (algorithms/, matching/, assembly/, verification/, pipeline, tools/) —
   настоящий код, написанный с пониманием предметной области. Реализации DTW,
   IFS, CSS, box-counting, SA, beam search, genetic, ACO, MCTS — полноценные
   алгоритмы, верифицированные запуском. Плотность логики 27-34%.

2. **Утилиты** (utils/, 103 модуля) — сгенерированный по шаблону код.
   Признаки: идентичная структура файлов, одинаковые f-string валидации,
   механические batch-обёртки, дублирование между файлами, аномально
   однородные размеры. Плотность логики 9-15%.

**Реальный объём проекта: ~45 000 LOC уникального кода (не 91 000).**
Коэффициент раздутия: ×2.0 (в основном за счёт utils/).

---

## 19. Анализ импортов utils: спящий код

> **Обновление 2026-02-24:** Анализ в `INTEGRATION_ROADMAP.md` установил, что
> код, ранее классифицированный как «мёртвый», в действительности является
> **«спящим»** — полностью реализованным, экспортированным, покрытым тестами,
> но не подключённым к точке входа `main.py`. Подробности — см. §22.

### 19.1 Факт: utils минимально подключён к текущему pipeline

Полный AST-анализ **всех** файлов в `algorithms/`, `matching/`, `assembly/`,
`verification/`, `preprocessing/`, `scoring/`, `tools/`, `pipeline.py`,
`main.py`, `models.py`, `config.py`, `clustering.py`, `export.py` выявил:

```
Всего строк с импортом из utils во всём production-коде: 2

matching/icp.py:32:    from ..utils.geometry import resample_curve, rotate_points, align_centroids
pipeline.py:51:        from .utils.logger import get_logger, PipelineTimer
```

**Из 102 модулей utils (32 590 LOC) текущий pipeline использует ровно 2 модуля
и 5 функций.** Однако это не означает, что остальные модули бесполезны —
см. §22 для плана интеграции.

### 19.2 Классификация 102 модулей utils по текущему использованию

| Категория | Кол-во модулей | LOC | Описание |
|-----------|:--------------:|:---:|----------|
| **Используются текущим pipeline** | 2 | ~700 | `geometry`, `logger` — подключены сейчас |
| **Тестируются, ожидают интеграции** | 59 | ~19 300 | Реализованы и протестированы, но не подключены к `main.py` |
| **Полные орфаны (даже тесты не ссылаются)** | 41 | ~12 565 | Не упоминаются ни кодом, ни тестами |

### 19.3 Орфаны — 41 файл, 12 565 LOC без тестовых ссылок

```
alignment_utils.py (326)    annealing_score_utils.py (245)   assembly_score_utils.py (290)
candidate_rank_utils.py (222)  canvas_build_utils.py (199)   color_edge_export_utils.py (293)
color_hist_utils.py (256)   config_utils.py (328)            consensus_score_utils.py (221)
contour_profile_utils.py (359)  curve_metrics.py (262)       descriptor_utils.py (292)
edge_profile_utils.py (357)  event_affine_utils.py (323)     filter_pipeline_utils.py (204)
fragment_filter_utils.py (323)  icp_utils.py (274)           image_cluster_utils.py (431)
image_transform_utils.py (290)  noise_stats_utils.py (300)   orient_skew_utils.py (309)
overlap_score_utils.py (354)  pair_score_utils.py (251)      patch_score_utils.py (304)
path_plan_utils.py (334)    placement_metrics_utils.py (435) placement_score_utils.py (335)
polygon_ops_utils.py (386)  quality_score_utils.py (234)     rank_result_utils.py (264)
ranking_layout_utils.py (415)  region_score_utils.py (364)   render_utils.py (277)
rotation_hist_utils.py (414)  rotation_score_utils.py (391)  score_matrix_utils.py (292)
score_norm_utils.py (194)   segment_utils.py (262)           seq_gap_utils.py (311)
shape_match_utils.py (205)  tracker_utils.py (439)
```

Средний размер орфана: **306 LOC** — аномально однородно, что указывает
на шаблонную генерацию.

### 19.4 Влияние на тестовую базу

| Метрика | Значение |
|---------|:--------:|
| Тестовых файлов, тестирующих utils | **90 / 504 (18%)** |
| LOC тестов для utils | **35 008 / 167 838 (21%)** |

21% всего тестового кода (~35 000 LOC) тестирует модули, которые
текущий pipeline не импортирует. Тесты верифицируют корректность
модулей в изоляции, что ценно для будущей интеграции.

### 19.5 __init__.py — 3 277 строк реэкспорта

Файл `utils/__init__.py` содержит **3 277 строк** и реэкспортирует
всё из всех 102 модулей. Это самый большой `__init__.py` в проекте.
Текущий pipeline использует прямые импорты
(`from ..utils.geometry import ...`), а не `from ..utils import ...`.

---

## 20. Анализ preprocessing/ и scoring/ — спящие модули

### 20.1 preprocessing/ (38 модулей, 11 693 LOC)

**Шаблонные признаки:**

| Индикатор | Значение |
|-----------|:--------:|
| С `@dataclass` | **32/38 (84%)** |
| С `__post_init__` | **19/38 (50%)** |
| С `batch_*` | **32/38 (84%)** |
| С русской валидацией «должен быть» | **14/38 (37%)** |
| Размеры файлов | min=89, max=457, **медиана=316** |

**Использование ядром:** из 38 модулей production-код импортирует только **4**:

```
segmentation  — pipeline.py, algorithms/__init__.py
contour       — pipeline.py, algorithms/, matching/
color_norm    — pipeline.py
orientation   — pipeline.py
```

**Спящие модули: 31 из 38 (82%), ~9 700 LOC, ожидающих интеграции через PreprocessingChain:**

```
augment (340)  background_remover (296)  binarizer (388)  channel_splitter (234)
color_normalizer (316)  contour_processor (456)  contrast_enhancer (321)  denoise (231)
deskewer (271)  document_cleaner (300)  edge_detector (384)  edge_enhancer (298)
edge_sharpener (328)  fragment_cropper (275)  frequency_analyzer (385)  frequency_filter (292)
gradient_analyzer (315)  illumination_corrector (304)  illumination_normalizer (335)
image_enhancer (277)  morphology_ops (389)  noise_analyzer (254)  noise_filter (291)
noise_reducer (253)  noise_reduction (371)  patch_normalizer (287)  patch_sampler (439)
quality_assessor (269)  skew_correction (370)  texture_analyzer (349)  warp_corrector (352)
```

Характерно: **массовое дублирование по «синонимам»** — генератор создавал
отдельный файл для каждого варианта формулировки одной и той же задачи:

| Задача | Файлы-дубликаты | Дублированные функции |
|--------|:-:|:-:|
| Шумоподавление | `denoise`, `noise_filter`, `noise_reducer`, `noise_reduction` **(4)** | `gaussian_denoise` ≈ `gaussian_filter`, `bilateral_denoise` ≈ `bilateral_filter`, `nlmeans_denoise` ≈ `nlm_filter` |
| Нормализация цвета | `color_norm`, `color_normalizer` **(2)** | `white_balance()`, `gamma_correction()`, `normalize_brightness()` |
| Контраст | `contrast`, `contrast_enhancer` **(2)** | CLAHE, histeq, gamma, stretch |
| Детекция краёв | `edge_detector`, `edge_enhancer`, `edge_sharpener` **(3)** | Варианты Canny/Sobel/Laplacian |
| Коррекция освещения | `illumination_corrector`, `illumination_normalizer` **(2)** | Retinex, background subtraction |

Итого **~13 файлов** (≈4 000 LOC) сводятся к **~5 задачам** — коэффициент
дублирования ×2.6.

**Наиболее качественные (не-шаблонные) модули:**
- `contour.py` (158 LOC) — RDP, corner detection, edge splitting. Минимум boilerplate.
- `orientation.py` (88 LOC) — Hough + PCA fallback. Самый компактный.
- `segmentation.py` (98 LOC) — Otsu + adaptive + GrabCut. Чистая структура.

### 20.2 scoring/ (12 модулей, 3 920 LOC)

**Шаблонные признаки:**

| Индикатор | Значение |
|-----------|:--------:|
| С `@dataclass` | **11/12 (92%)** |
| С `__post_init__` | **11/12 (92%)** |
| С `batch_*` | **7/12 (58%)** |
| С русской валидацией «должен быть» | **12/12 (100%)** |
| Размеры файлов | min=193, max=400, **медиана=338** |

**Scoring — наиболее шаблонный модуль проекта**: 100% файлов содержат
русские валидационные строки, 92% следуют dataclass-шаблону.

**Использование ядром:** только **2 из 12**:

```
consistency_checker — verification/__init__.py
score_normalizer    — matching/__init__.py
```

**Спящие модули: 9 из 12 (75%), ~3 030 LOC, ожидающих интеграции через MatcherRegistry:**

```
boundary_scorer (333)  evidence_aggregator (289)  gap_scorer (337)
global_ranker (331)  match_evaluator (349)  match_scorer (324)
pair_filter (338)  pair_ranker (330)  threshold_selector (399)
```

**100% файлов следуют шаблону:** Config → Result → core_function → batch_function.

```python
# Шаблон повторяется дословно во всех 12 файлах:
@dataclass
class ScorerConfig:
    weights: Dict[str, float] = field(default_factory=lambda: {...})
    min_score: float = 0.0
    max_score: float = 1.0
    def __post_init__(self):
        for name, w in self.weights.items():
            if w < 0.0:
                raise ValueError(f"Вес канала '{name}' должен быть >= 0, получено {w}")
```

**Наиболее качественный модуль:** `rank_fusion.py` (192 LOC) — RRF, Borda count,
score fusion. Чистый алгоритмический код без dataclass-шаблона.

### 20.3 Совокупный спящий код проекта

| Источник | Не подключено к pipeline (LOC) | Статус |
|----------|:-------------------------------:|--------|
| utils/ | ~31 890 (из 32 590) | Инфраструктура для будущих фаз (event_bus, cache, metrics, geometry) |
| preprocessing/ | ~9 700 | 35 модулей для PreprocessingChain (Фаза 4 INTEGRATION_ROADMAP) |
| scoring/ | ~3 330 | Модули для MatcherRegistry (Фаза 3) |
| assembly/ (4 алгоритма) | ~960 | genetic, exhaustive, ant_colony, mcts — ожидают CLI (Фаза 2) |
| matching/ (9+ матчеров) | ~2 700 | icp, color, texture, seam и др. — ожидают registry (Фаза 3) |
| verification/ (20 модулей) | ~7 500 | Ожидают VerificationSuite (Фаза 5) |
| **Итого** | **~48 200** | **Три точки разрыва (мосты) — см. §22** |

Из **91 180 LOC production-кода** около **48 200 LOC (53%)** реализовано,
экспортировано и протестировано, но не подключено к текущему pipeline
через `main.py`. Инфраструктура для подключения уже существует:
`parallel.py` (реестр 8 алгоритмов), `consensus.py` (голосование),
`score_combiner.py` (агрегация).

**Текущий активный pipeline: ~43 000 LOC. Спящий код: ~48 200 LOC.**

---

## 21. Детальная категоризация 129 провальных тестов + 4 ошибки сбора

### 21.1 Ошибки сбора (4 теста)

```
ImportError: cannot import name 'Edge' from 'puzzle_reconstruction.models'
ImportError: cannot import name 'Placement' from 'puzzle_reconstruction.models'
```

**Файлы:** `test_confidence_scorer.py`, `test_graph_match.py`,
`test_layout_verifier.py`, `test_parallel.py`

**Диагноз:** Тесты генерировались под API, включающий классы `Edge` и
`Placement`, которые не были реализованы в `models.py`. Рассинхрон
генератора тестов и генератора кода.

### 21.2 Классификация 129 FAILED тестов (уточнённая)

| Категория | Кол-во | % | Описание |
|-----------|:------:|:-:|----------|
| **E. Неверное утверждение** | 55 | 42.6% | Тест ожидает неверный результат |
| **B. API-рассинхрон** (TypeError) | 28 | 21.7% | Неверная сигнатура вызова |
| **D. Реальные баги в коде** | 18 | 14.0% | Крэши и ошибки в production-коде |
| **C. Несуществующий атрибут** | 13 | 10.1% | Мок несуществующей функции |
| **F. Допуск/tolerance** | 11 | 8.5% | Неверные ожидания под видом approx |
| **G. Невалидный test setup** | 4 | 3.1% | Тест подаёт данные, отклоняемые валидацией |

### 21.3 Категория B: API-рассинхрон (28 тестов)

```python
# test_algorithms_edge_profile.py (16 тестов):
extract_intensity_profile(image)  # тест
extract_intensity_profile(image, side)  # реальная сигнатура — нет 'side'

# test_match_scorer.py (7 тестов):
# передают int-ключи, где ожидаются строки → "keywords must be strings"

# test_match_scorer.py (3 теста):
actual >= pytest.approx(x)  # TypeError: '>=' и ApproxScalar несовместимы

# test_fragment_validator.py, test_illumination_corrector.py (2 теста):
# несуществующие keyword-аргументы (value=, ksize=)
```

### 21.4 Категория C: Несуществующие атрибуты (13 тестов)

Все 13 — `test_synthesis.py::TestBuildEdgeSignatures`. Тесты пытаются
замокать `synthesis.curvature_scale_space` — функцию, которой нет в модуле.

### 21.5 Категория D: **Реальные баги в production-коде** (18 тестов)

| Баг | Тестов | Файл | Проблема |
|-----|:------:|------|----------|
| **Laplacian ddepth** | 10 | `edge_detector.py:244` | float32 → Laplacian с неверным `ddepth`, крэш OpenCV |
| **uint8 overflow** | 2 | `visualizer.py:336` | Присвоение 382 и -127 в uint8 без `np.clip()` |
| **Gamma NaN** | 2 | `gamma.py` | scipy solver получает inf/NaN, крэш |
| **Empty array** | 2 | `edge_validator.py`, `score_normalizer.py` | `max()` на пустом массиве |
| **Texture unpack** | 1 | `texture_match.py:101` | Неверное число значений при unpacking |
| **Strip shape** | 1 | `boundary_scorer.py:157` | `(64, 4)` вместо `(64, 4, 3)` — grayscale/BGR путаница |

**Это единственные 18 тестов, выявившие настоящие ошибки в коде.**

### 21.6 Категория E: Неверные утверждения (55 тестов)

Крупнейшая категория. Тесты работают без крэша, но assert ложный:

| Подтип | Примеры |
|--------|---------|
| Неверное ожидаемое значение (18) | `assert 64 == 256` (color_utils), `assert 4 == 200` (contour), `assert [0,3] == [0,1,2,3]` (graph_utils) |
| Нереалистичный порог (13) | `assert 0.33 >= 0.8` (edge_scorer), `assert 1.0 < 0.5` (boundary_scorer) |
| Неверная логика (10) | `assert False is True`, `assert 'noisy' == 'clean'` |
| Математически неверно (6) | DTW triangle inequality, `chamfer <= hausdorff` |
| i18n-рассинхрон (1) | `assert 'Score' in '=== Результат пайплайна ...'` — English vs Russian |
| Перепутанное направление (2) | `test_gamma_gt_one_darker` — гамма-коррекция наоборот |
| Слишком строгое (3) | ICP RMSE, fractal similarity ≈0.997 vs ==1.0 |

### 21.7 Категория F: Tolerance (11 тестов)

Только 2-3 — настоящие проблемы с точностью (`0.996 ≈ 1.0 ± 1e-4`).
Остальные 8 — **полностью неверные значения** под видом `pytest.approx()`:
- `5.0 == 4.0 ± 4e-6` (path_planner)
- `80.0 == 0.0 ± 1e-12` (consistency_checker)
- `1681.0 == 1600.0 ± 80` (overlap_checker: 41×41 vs 40×40)

### 21.8 Категория G: Невалидный setup (4 теста)

Тесты подают данные, которые код правильно отклоняет:
- `n_pairs=0`, а код требует `>= 1`
- `color_weight=2.0`, а код требует `[0, 1]`
- `min_score=1.1`, а код требует `[0, 1]`

### 21.9 Выводы о тестах

**Из 129 провалов:**

- **111 (86%)** — ошибки в самих тестах (категории B, C, E, F, G).
  Классические признаки AI-генерации: придуманные значения (`64 == 256`),
  несуществующие функции, English-ассерты для Russian-вывода,
  неверные математические свойства.

- **18 (14%)** — **реальные баги в production-коде**. Стоит исправить:
  1. `edge_detector.py:244` — исправить `ddepth` для float32 (10 тестов)
  2. `visualizer.py:336` — добавить `np.clip(0, 255)` (2 теста)
  3. `gamma.py` — обработка inf/NaN (2 теста)
  4. `edge_validator.py`, `score_normalizer.py` — проверка на пустой массив (2 теста)
  5. `texture_match.py:101` — исправить unpacking (1 тест)
  6. `boundary_scorer.py:157` — исправить shape mismatch (1 тест)

Также 4 файла (≈246 тестов) не загружаются из-за ImportError (`Edge`, `Placement`).

---

## 22. Рекомендации и дорожная карта

> **Обновление 2026-02-24:** Раздел полностью переработан на основе
> `INTEGRATION_ROADMAP.md`. Ключевое изменение: вместо удаления «мёртвого»
> кода — **интеграция спящего кода через три архитектурных моста**.

### 22.0 Ключевое открытие

Анализ показал: ~48 200 LOC «неподключённого» кода **не является мёртвым**.
Он реализован, экспортирован, покрыт тестами (25 000+) — но не подключён
к точке входа `main.py`.

| Характеристика | Мёртвый код | Спящий код (наш случай) |
|---|---|---|
| Реализован | нет / частично | полностью |
| Протестирован | нет | 25 000+ тестов |
| Экспортирован в `__init__.py` | нет | везде |
| Подключён к `main.py` | — | только 4/8 алгоритмов, 4/38 preprocessing, 4/13 matchers |
| Нужны ли изменения кода? | большие | минимальные (3 моста) |

### 22.0.1 Три точки разрыва

Вся инфраструктура уже работает. Разрыв — в трёх местах:

```
main.py:assemble()         ──── if/elif (4 метода) ─────▶  parallel.py (8 методов)
                                                             ^^^^^^^^^^^^^ МОСТ №1
main.py:process_fragment() ──── 5 модулей из 38 ───────▶  preprocessing/* (38 модулей)
                                                             ^^^^^^^^^^^^^ МОСТ №2
matching/pairwise.py       ──── жёсткие веса (4) ──────▶  matcher_registry (20+)
                                                             ^^^^^^^^^^^^^ МОСТ №3
```

`parallel.py` содержит реестр всех 8 алгоритмов. `consensus.py` умеет
голосовать между результатами. `score_combiner.py` агрегирует оценки
от любого числа матчеров. Нужно лишь подключить провода.

### 22.1 Фаза 2 — Assembly Registry (высокий приоритет, малый риск)

**Мост №1:** подключить 4 спящих алгоритма сборки + режимы `auto`/`all`.

**Что делать:**
1. Расширить `AssemblyConfig.method` Literal на 10 значений
   (`genetic`, `exhaustive`, `ant_colony`, `mcts`, `auto`, `all`)
2. Добавить параметры genetic/aco/mcts/exhaustive в `AssemblyConfig`
3. Переписать `main.py:assemble()` — делегировать в `parallel.py`
   (`run_all_methods()`/`run_selected()`)
4. Обновить argparse choices в `build_parser()`

**Риск:** минимальный. `parallel.py` уже реализован и протестирован.
**Обратная совместимость:** полная. `--method beam` работает как раньше.

**Файлы:**
- `puzzle_reconstruction/config.py` — ~15 строк
- `main.py` — ~30 строк (замена `assemble()`)

**Критерии успеха:**
- [ ] `python main.py --method genetic` работает
- [ ] `python main.py --method exhaustive` работает для N<=8
- [ ] `python main.py --method ant_colony` работает
- [ ] `python main.py --method mcts` работает
- [ ] `python main.py --method auto` выбирает методы по числу фрагментов
- [ ] `python main.py --method all` запускает все 8, выводит таблицу
- [ ] `--method beam` даёт идентичный результат с до-интеграционной версией
- [ ] Все существующие тесты `pytest tests/ -x -q` проходят

### 22.2 Фаза 3 — Matcher Registry (средний приоритет)

**Мост №3:** подключить 9+ спящих матчеров через конфигурируемые веса.

**Что делать:**
1. Создать `matching/matcher_registry.py` — реестр с декоратором `@register`
2. Расширить `MatchingConfig` — `active_matchers`, `matcher_weights`,
   `combine_method`
3. Обновить `pairwise.py` — читать веса из конфига вместо жёстких констант
4. Зарегистрировать все матчеры через декоратор

**Риск:** средний. Нужна осторожность с backward-compatibility scorer'а.
**Проверка:** `build_compat_matrix()` должна давать тот же результат
с дефолтным конфигом.

**Критерии успеха:**
- [ ] `config.yaml` с `active_matchers: [css, dtw, color]` работает
- [ ] Дефолтные веса дают идентичный результат с до-интеграционной версией
- [ ] `config.yaml` с `combine_method: rank` работает
- [ ] Добавление нового матчера через `@register("my_matcher")` работает

### 22.3 Фаза 4 — Preprocessing Chain (средний приоритет)

**Мост №2:** подключить 35 спящих preprocessing-модулей через конфигурируемую цепочку.

**Что делать:**
1. Создать `preprocessing/chain.py` с `PreprocessingChain`
2. Добавить `PreprocessingConfig` в `config.py`
3. Подключить цепочку в `main.py:process_fragment()`
4. Добавить в конфиг возможность задавать список фильтров

**Риск:** средний. Базовый поток (segmentation+contour+orientation)
остаётся неизменным.

### 22.4 Фаза 5 — Verification Suite (средний приоритет)

**Что делать:**
1. Создать `verification/suite.py` с `VerificationSuite`
2. Подключить к этапу верификации в `main.py`
3. Добавить `VerificationConfig.validators` список

**Риск:** низкий. Верификация — постпроцессинг, не влияет на сборку.

### 22.5 Исправление тестов (параллельно с фазами 2–5)

**Проблема:** 129 провальных тестов (111 — ошибки тестов, 18 — реальные баги).

**Действия:**

1. **Исправить 18 реальных багов в production-коде:**
   - `edge_detector.py:244` — добавить корректный `ddepth` для float32 (10 тестов)
   - `visualizer.py:336` — добавить `np.clip(0, 255)` перед uint8 (2 теста)
   - `gamma.py` — обработка inf/NaN перед вызовом scipy solver (2 теста)
   - `edge_validator.py`, `score_normalizer.py` — проверка на пустой массив (2 теста)
   - `texture_match.py:101` — исправить распаковку возвращаемого значения (1 тест)
   - `boundary_scorer.py:157` — исправить shape mismatch grayscale/BGR (1 тест)
2. **Исправить 111 ошибок в тестах:**
   - 28 API-рассинхронов — обновить сигнатуры вызовов
   - 13 AttributeError — убрать моки несуществующих функций
   - 55 AssertionError — исправить ожидаемые значения
   - 11 Tolerance — исправить или ослабить допуски
   - 4 Setup — привести входные данные к валидным
3. **Добавить `Edge` и `Placement` в `models.py`** или удалить 4 тестовых файла
   (246 тестов)

### 22.6 Фаза 6 — Infrastructure Utils (низкий приоритет, высокая ценность)

**Что делать:**
1. Подключить `event_bus` в `pipeline.py` — прогресс-события
2. Использовать `result_cache` для кэша дескрипторов (между запусками)
3. Использовать `batch_processor` для обработки нескольких документов
4. Использовать `metric_tracker` в research mode

**Риск:** низкий. Всё аддитивно.

### 22.7 Фаза 7 — Research Mode (исследовательский, низкий приоритет)

Два режима работы одной системы:
- **production** — быстро, один лучший метод
- **research** — медленно, все методы, сравнение, consensus

```bash
# Auto (интеллектуальный выбор по числу фрагментов)
python main.py --input scans/ --method auto

# Research (все 8 методов, benchmark)
python main.py --input scans/ --method all --research
```

### 22.8 Консолидация utils (после интеграции)

**Проблема:** дублирование и шаблонный boilerplate в utils/.

**Действия (после завершения фаз 2–6):**
1. Устранить дублирование (`cosine_distance`, `normalize_contour`,
   `normalize_profile` — оставить по одной канонической реализации)
2. Объединить оставшиеся модули в укрупнённые по функциональным областям
3. Удалить 41 орфанный модуль (12 565 LOC), не востребованный
   ни интеграцией, ни тестами

### 22.9 Итоговый результат интеграции — ВЫПОЛНЕНО (2026-02-25)

| Метрика | Было (v0.3.0) | Стало (v0.4.0-beta) | Изменение |
|---------|:------:|:---------------------------:|:---------:|
| Активных алгоритмов сборки | 4 | **8 (+auto, +all)** | ✅ +4 |
| Активных матчеров | 4 | **13+** | ✅ +9 |
| Активных preprocessing-модулей | 4 | **38** | ✅ +34 |
| Активных верификаторов | 1 | **21** | ✅ +20 |
| Провальных тестов | 129 + 4 ошибки | **0** | ✅ |
| Production-баги | 18 | **0** | ✅ |
| CLI-режимы assembly | 4 | **10** | ✅ +6 |
| Всего тестов | 42 208 | **42 476** | ✅ +268 (↑ фазы 8–12 + sklearn + xfail) |
| mypy-модулей (строгий) | 3 | **50+** | ✅ |
| CLI-опции верификации | 0 | **3** (`--validators`, `--export-report`, `--list-validators`) | ✅ |
| VerificationReport API | нет | **as_dict/to_json/to_markdown/to_html** | ✅ |
| Pipeline.verify_suite() | нет | **Интегрирован** | ✅ |
| Версия | 0.3.0 | **1.0.0 Stable** | ✅ |

**Принцип:** не удалять — находить правильное место и роль.
Подробная дорожная карта — см. `INTEGRATION_ROADMAP.md`.

---

*Документ сгенерирован автоматическим аудитом 2026-02-23.*
*Обновлён 2026-02-24: интеграция выводов из INTEGRATION_ROADMAP.md (спящий код, три моста).*
*Обновлён 2026-02-25 (итерация 1): анализ расхождения веток (§1a), свежий pytest (§9.1), уточнение зрелости (§17.3).*
*Обновлён 2026-02-25 (итерация 2, Фазы 8–11): верификация 21/21, mypy 50+, --export-report, E2E-тесты.*
*Обновлён 2026-02-25 (итерация 3, Фаза 12 / v1.0.0): VerificationReport API, Pipeline.verify_suite, --list-validators.*
*Обновлён 2026-02-25 (итерация 4): scikit-learn 1.8.0 активирован (+63 clustering тестов), xfail-маркеры Laplacian удалены (+9); итого 42 476 тестов.*
*Методы: AST-анализ импортов, pytest --tb=short, wc -l, git diff origin/main..HEAD, анализ кода.*
