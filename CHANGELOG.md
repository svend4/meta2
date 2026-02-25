# Changelog — puzzle-reconstruction

Все существенные изменения в этом проекте документируются в данном файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.0.0/).
Версионирование следует [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.4.0-beta] — 2026-02-25

### Добавлено

- **Dockerfile** — многоэтапный образ (builder → runtime, непривилегированный пользователь, healthcheck)
- **docker-compose.yml** — стек с профилями `cli`, `bench`, `dev`; тома для input/output
- **.dockerignore** — исключение тестов, кэша, IDE-файлов из образа
- **Makefile** — команды `install`, `lint`, `format`, `typecheck`, `qa`, `test`, `test-fast`, `test-cov`, `test-warn`, `profile`, `benchmark`, `server`, `build`, `docker`, `docker-up`, `docker-down`, `clean`
- **CHANGELOG.md** — данный файл; история всех релизов
- **tools/server.py**: новый endpoint `GET /spec` — OpenAPI 3.0 JSON-спецификация всех эндпоинтов
- **pyproject.toml**: новые optional-dependencies `[geometry]` (shapely≥2.0), `[graph]` (networkx≥3.0), `[viz]` (matplotlib≥3.7); группа `[all]` обновлена
- **pyproject.toml**: версия → `0.4.0b1`, classifier → `Development Status :: 4 - Beta`
- **pyproject.toml**: mypy-опции `check_untyped_defs = true`, `warn_return_any = false`

### Улучшено

- **CI (ci.yml)**: lint (ruff) переведён из `continue-on-error: true` в **blocking**
- **CI (ci.yml)**: integration тесты переведены в **blocking** (были `continue-on-error: true`)
- **CI (ci.yml)**: mypy расширен с 3 до 13 ключевых модулей (добавлены `export.py`, `pipeline.py`, `synthesis.py`, `box_counting.py`, `css.py`, `divider.py`, `hull.py`, `pairwise.py`, `compat_matrix.py`, `matcher_registry.py`)
- **STATUS.md**: таблица алгоритмов обновлена — все 8 методов показывают `✅` в CLI-колонке
- **STATUS.md**: история предупреждений дополнена 2 новыми исправлениями

### Исправлено

- **`fractal/box_counting.py:57`** — `RankWarning: Polyfit may be poorly conditioned` при `n_scales=1`: добавлен guard → возврат `1.0` при менее 2 точках для регрессии
- **`algorithms/edge_scorer.py:171`** — `RuntimeWarning: invalid value encountered in divide` из `np.corrcoef` при нулевом std: обёрнут в `np.errstate(invalid='ignore')`, результат проверяется через `np.isfinite`

---

## [0.3.0] — 2026-02-25

### Добавлено (Фаза 7 — Research Mode)

- `--method all` запускает все 8 алгоритмов и выбирает лучший по score
- `--method auto` автоматически выбирает алгоритм по числу фрагментов
- `--research` добавляет сравнительную таблицу и консенсус-голосование
- `--export-json` экспортирует отчёт в JSON с метриками каждого метода

### Добавлено (Фаза 6 — Infrastructure Utils)

- `utils/event_bus.py` — Pub/sub шина событий подключена к Pipeline
- `utils/result_cache.py` — LRU-кэш с TTL для матрицы совместимости
- `utils/metric_tracker.py` — трекинг метрик производительности
- `utils/batch_processor.py` — BatchProcessor для параллельной обработки

### Добавлено (Фазы 2–5 — Integration Bridges)

- **Мост №1**: `assembly/parallel.py` — реестр всех 8 алгоритмов; `main.py` делегирует через `run_selected()` / `run_all_methods()`
- **Мост №2**: `preprocessing/chain.py` — `PreprocessingChain`; 38 модулей активируются через `PreprocessingConfig.chain`
- **Мост №3**: `matching/matcher_registry.py` — `MATCHER_REGISTRY`, `@register`, `get_matcher()`; 13+ матчеров через `MatchingConfig`
- **Мост №4**: `verification/suite.py` — `VerificationSuite`; 21 верификатор через `VerificationConfig`

### Исправлено

- 133 ранее провалившихся теста устранены → **0 failures**
- 3 RuntimeWarning/DeprecationWarning в `gamma_optimizer`, `graph_match`, `classifier`
- Все `xfail`-маркеры проверены и актуализированы

### Тестирование

- 42 219 тестов, 42 208 passed (99.97%), 0 failures, 2 skipped (архитектурно-зависимые), 9 xpassed
- 822 тестовых файла, 267 359 строк тестового кода
- Соотношение тест/prod код: **2.87:1**

---

## [0.2.0] — 2026-02-24

### Добавлено

- `assembly/ant_colony.py` — алгоритм муравьиных колоний (ACO)
- `assembly/mcts.py` — Monte Carlo Tree Search
- `assembly/genetic.py` — генетический алгоритм
- `assembly/exhaustive.py` — полный перебор (≤8 фрагментов)
- `matching/dtw.py` — Dynamic Time Warping
- `verification/text_coherence.py` — N-gram модель связности текста
- `verification/layout_checker.py` — проверка 2D-компоновки
- `scoring/` — 12 модулей оценки качества
- REPORT.md — детальный отчёт о покрытии тестами

### Исправлено

- Добавлены модели `Placement`, `Edge`, `CompatEntry` в `models.py`
- Устранены ошибки сбора тестов (`test collection errors`)
- 58 провалившихся тестов исправлены (включая 39 предсуществующих)

---

## [0.1.0] — 2026-02-20 — 2026-02-23

### Добавлено (начальная разработка, iter-1 → iter-249)

- Основная архитектура: `Pipeline`, `Config`, `Fragment`, `EdgeSignature`, `Assembly`
- Алгоритм Танграм: `tangram/hull.py`, `tangram/classifier.py`, `tangram/inscriber.py`
- Алгоритмы Фракталов: `fractal/box_counting.py`, `fractal/divider.py`, `fractal/ifs.py`, `fractal/css.py`
- Синтез подписей: `algorithms/synthesis.py` — `build_edge_signatures()`, `compute_fractal_signature()`
- Базовые алгоритмы сборки: `greedy.py`, `annealing.py`, `beam_search.py`, `gamma_optimizer.py`
- Матчеры: CSS, DTW, FD, TEXT (OCR-связность)
- Предобработка: 38 модулей (segmentation, denoise, color_norm, deskew, ...)
- Верификация: 21 модуль (ocr, metrics, seam_analyzer, ...)
- Утилиты: 130 модулей (geometry, transform, cache, graph, distance, ...)
- Инструменты: `tools/benchmark.py`, `tools/evaluate.py`, `tools/mix_documents.py`, `tools/profile.py`, `tools/server.py`, `tools/tear_generator.py`
- CLI: `puzzle-reconstruct`, `puzzle-benchmark`, `puzzle-generate`, `puzzle-mix`, `puzzle-server`, `puzzle-evaluate`, `puzzle-profile`
- Тесты: 822 файла, 42 219 тестов, покрытие 100% модулей
- Документация: README.md, PUZZLE_RECONSTRUCTION.md, INTEGRATION_ROADMAP.md

---

[0.4.0-beta]: https://github.com/svend4/meta2/compare/v0.3.0...HEAD
[0.3.0]:      https://github.com/svend4/meta2/releases/tag/v0.3.0
[0.2.0]:      https://github.com/svend4/meta2/compare/v0.1.0...v0.2.0
[0.1.0]:      https://github.com/svend4/meta2/releases/tag/v0.1.0
