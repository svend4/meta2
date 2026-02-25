# STATUS.md — Полный статус реализации `puzzle_reconstruction`

> **Дата:** 2026-02-25
> **Ветка:** `claude/dev-status-documentation-j7YCO`
> **Версия:** 0.3.0-alpha · Python 3.11.14
> **Последний коммит:** `07fd44f`
> **Тест-прогон:** 42 280 тестов · 2 мин 58 сек · **1 провал**
> **Документ:** версия 5 (полный аудит импортов)

---

## Содержание

1. [Сводка](#1-сводка)
2. [Как Python загружает модули — цепочка активации](#2-цепочка-активации)
3. [Живые результаты тестирования](#3-живые-результаты-тестирования)
4. [Метрики кодовой базы](#4-метрики-кодовой-базы)
5. [preprocessing/ — 38 модулей](#5-preprocessing--38-модулей)
6. [algorithms/ — 42 модуля](#6-algorithms--42-модуля)
7. [matching/ — 27 модулей](#7-matching--27-модулей)
8. [assembly/ — 27 модулей](#8-assembly--27-модулей)
9. [verification/ — 21 модуль](#9-verification--21-модуль)
10. [scoring/ — 12 модулей ⚠️ СПЯЩИЙ](#10-scoring--12-модулей--спящий)
11. [io/ — 3 модуля ⚠️ СПЯЩИЙ](#11-io--3-модуля--спящий)
12. [utils/ — 130 модулей](#12-utils--130-модулей)
13. [ui/ — 1 модуль](#13-ui--1-модуль)
14. [Корневые модули пакета](#14-корневые-модули-пакета)
15. [CLI — интерфейс командной строки](#15-cli)
16. [tools/ — 6 вспомогательных скриптов](#16-tools--6-скриптов)
17. [Технический стек](#17-технический-стек)
18. [История разработки](#18-история-разработки)
19. [Единственный провалившийся тест — внутреннее противоречие](#19-единственный-провалившийся-тест)
20. [Дорожная карта](#20-дорожная-карта)
21. [Итоговая оценка](#21-итоговая-оценка)

---

## 1. Сводка

| Параметр | Значение |
|---|---|
| **Стадия** | Alpha (v0.3.0) |
| **Всего исходных модулей** | **306** `.py` (без `__init__.py`) |
| **Строк исходного кода** | ~94 000 в пакете + 384 `main.py` + ~1 943 `tools/` |
| **Тестовых файлов** | 824 |
| **Строк тестов** | 267 362 |
| **Тестов (живой прогон)** | **42 280** |
| **Пройдено** | **42 270 (99.976%)** |
| **Провалено** | **1 (0.002%)** |
| **Коммитов** | 260 |

### Статус по субпакетам — точный аудит

| Субпакет | Модулей | ✅ Всегда активен | 🔄 Лениво активен | 💤 Спит |
|---|---|---|---|---|
| Корневые (`pipeline`, `config`, ...) | 5 | **4** | — | **1** (`export.py`) |
| `preprocessing/` | 38 | **4** | — | 34 |
| `algorithms/` | 42 | **5** (synth+CSS+tangram×3) | **3** (fractal×3) | 34 |
| `matching/` | 27 | **3** | **1** (registry, опц.) | 23 |
| `assembly/` | 27 | **1** (parallel) | **8** (все алгоритмы) | 18 |
| `verification/` | 21 | **1** (ocr) | — | 20 |
| `scoring/` ⚠️ | 12 | — | — | **12** |
| `io/` ⚠️ | 3 | — | — | **3** |
| `utils/` | 130 | **2** (logger, event_bus) | — | 128 |
| `ui/` | 1 | — | **1** (`--interactive`) | — |
| **ИТОГО** | **306** | **20** | **13** | **273** |

> **273 из 306 модулей (89.2%) — реализованы, покрыты тестами, но не подключены к основному потоку выполнения.**

---

## 2. Цепочка активации

Механизм загрузки: когда `main.py` выполняет первый импорт
`from puzzle_reconstruction.config import Config`, Python запускает
`puzzle_reconstruction/__init__.py`, который в свою очередь статически
загружает `models`, `config`, `clustering`, `pipeline`.

### Полное дерево загрузки (с уровнями)

```
python main.py
│
├── [статически при запуске]
│     puzzle_reconstruction/__init__.py
│       ├── models.py              ✅ всегда активен
│       ├── config.py              ✅ всегда активен
│       ├── clustering.py          ✅ всегда активен  ← через __init__.py
│       └── pipeline.py            ✅ всегда активен  ← через __init__.py
│             └── preprocessing/color_norm.py  ✅
│             └── utils/event_bus.py           ✅
│
├── [статически, явные импорты main.py]
│     preprocessing/segmentation.py  ✅
│     preprocessing/contour.py       ✅
│     preprocessing/orientation.py   ✅
│     algorithms/tangram/inscriber.py ✅
│       ├── tangram/hull.py         ✅
│       └── tangram/classifier.py   ✅
│     algorithms/synthesis.py       ✅
│       └── fractal/css.py          ✅ (статический в synthesis)
│     matching/compat_matrix.py     ✅
│       └── matching/pairwise.py    ✅
│             └── matching/dtw.py   ✅
│     assembly/parallel.py          ✅
│     verification/ocr.py           ✅
│     utils/logger.py               ✅
│
├── [лениво — при вызове compute_fractal_signature()]
│     fractal/box_counting.py  🔄
│     fractal/divider.py       🔄
│     fractal/ifs.py           🔄
│
├── [лениво — при вызове parallel.dispatch(method)]
│     assembly/greedy.py         🔄
│     assembly/annealing.py      🔄
│     assembly/beam_search.py    🔄
│     assembly/gamma_optimizer.py 🔄
│     assembly/exhaustive.py     🔄
│     assembly/genetic.py        🔄
│     assembly/ant_colony.py     🔄
│     assembly/mcts.py           🔄
│
├── [лениво — только если cfg.active_matchers содержит нестандартные]
│     matching/matcher_registry.py  🔄
│       └── конкретные матчеры (icp, color, texture…)  🔄
│
└── [лениво — только при --interactive]
      ui/viewer.py  🔄
```

### Что НЕ загружается никогда в production

```
export.py          ← только в docstring __init__.py, НЕ в коде
scoring/*          ← 12 модулей, только тесты
io/*               ← 3 модуля, только тесты

assembly/canvas_builder, gap_analyzer, layout_builder, …  (18 модулей)
matching/consensus, score_combiner, global_matcher, …     (23 модуля)
verification/metrics, layout_checker, seam_analyzer, …   (20 модулей)
preprocessing/noise_filter, binarizer, deskewer, …       (34 модуля)
algorithms/shape_context, sift_matcher, edge_profile, …  (34 модуля)
utils/* кроме logger + event_bus                         (128 модулей)
```

### Ловушки и неочевидные факты

| Артефакт | Ожидание | Реальность |
|---|---|---|
| `export.py` (корень) | Активен (в __init__.py упомянут) | 💤 **СПИТ** — строки 11–14 в `__init__.py` это docstring-пример, не код |
| `clustering.py` (корень) | Может спать | ✅ **АКТИВЕН** — `__init__.py` строка 31: `from .clustering import …` |
| `puzzle_reconstruction/utils/io.py` vs `puzzle_reconstruction/io/` | Один пакет | **Два разных файла**: `utils/io.py` активен через `utils/__init__.py`; `io/` субпакет — полностью спит |
| `ui/viewer.py` | Спит | 🔄 **ЛЕНИВО АКТИВЕН** — `main.py:358` при `--interactive` |
| `matcher_registry.py` | Всегда активен раз зарегистрирован | 🔄 **ЛЕНИВО** — загружается только если `cfg.active_matchers` содержит не-базовые матчеры |
| `fractal/box_counting, divider, ifs` | Спят (нет прямого импорта) | 🔄 **ЛЕНИВО** — импортируются внутри `compute_fractal_signature()` |
| Docstring main.py | «4 метода: greedy/sa/beam/gamma» | ⚠️ **УСТАРЕЛ** — реально поддерживается 10 методов (+ auto, all) |

---

## 3. Живые результаты тестирования

Прогон 2026-02-25, **2 мин 58 сек**:

```
Собрано:    42 280 тестов
──────────────────────────────────
✅  passed   42 270   (99.976%)
❌  failed        1   (0.002%)
⚠️  xpassed       9   (0.021%)  — были xfail, теперь проходят
💬  warnings     81
──────────────────────────────────
```

### Динамика (история исправлений)

| Коммит | Дата | Провалов |
|---|---|---|
| REPORT.md baseline | 24 фев | **133** |
| `aa47360` fix: −58 | 24 фев | 75 |
| `445b7f6` fix: −26 | 24 фев | 49 |
| `47d5fbe` fix: −4 | 24 фев | 45 |
| `6c98327` feat+fix: −44 | 24 фев | **1** |
| Сейчас | 25 фев | **1** ← стабильно |

### 9 xpassed-тестов

Тесты, ранее помеченные `@pytest.mark.xfail`, теперь проходят.
Метки устарели — нужно убрать `xfail` при следующем цикле.

---

## 4. Метрики кодовой базы

### Исходный код

| Субпакет | Файлов (без __init__) | Строк (без __init__) | __init__ строк | Итого |
|---|---|---|---|---|
| `utils/` | 130 | 38 970 | — | 38 970 |
| `algorithms/` | 42 | 11 817 | 817 | 12 634 |
| `preprocessing/` | 38 | 11 662 | 773 | 12 435 |
| `assembly/` | 27 | 8 145 | 505 | 8 650 |
| `matching/` | 27 | 8 188 | 484 | 8 672 |
| `verification/` | 21 | 7 442 | 466 | 7 908 |
| `scoring/` ⚠️ | 12 | 3 916 | 303 | 4 219 |
| `io/` ⚠️ | 3 | 1 060 | 81 | 1 141 |
| `ui/` | 1 | 364 | — | 364 |
| корневые (5 модулей) | 5 | ~1 471 | 57 | ~1 528 |
| **Пакет итого** | **306** | ~92 035 | — | ~96 521 |
| `main.py` | 1 | 384 | — | 384 |
| `tools/*.py` | 6 | ~1 943 | — | ~1 943 |
| **Проект итого** | **313** | — | — | **~98 848** |

### Тесты

| Тип | Файлов | Строк | Pass% |
|---|---|---|---|
| Базовые (`test_*.py`) | 337 | ~117 400 | >99.95% |
| Расширенные (`test_*_extra.py`) | 485 | ~149 900 | >99.99% |
| Итого | **824** | **267 362** | **99.976%** |

### Соотношения

| Метрика | Значение |
|---|---|
| Строк тестов / строк источника | **2.85×** |
| Тестовых файлов / исходных модулей | **2.69×** |
| Тестов на модуль (среднее) | **~138** |

---

## 5. `preprocessing/` — 38 модулей

Строк всего: **12 435** (вкл. `__init__.py` — 773 стр.)

### ✅ Всегда активные (4 модуля — 573 строки)

| Модуль | Стр. | Загружается из | Ключевые функции |
|---|---|---|---|
| `segmentation.py` | 98 | `main.py`, `pipeline.py` | `segment_fragment()` — Otsu/Adaptive/GrabCut |
| `contour.py` | 163 | `main.py`, `pipeline.py`, `synthesis.py` | `extract_contour()`, `resample_curve()`, `split_contour_to_edges()` |
| `orientation.py` | 88 | `main.py`, `pipeline.py` | `estimate_orientation()`, `rotate_to_upright()` |
| `color_norm.py` | 224 | `pipeline.py` | `normalize_color()` — CLAHE + Gray World |

### 💤 Спящие (34 модуля — ~10 316 строк)

| Модуль | Стр. | Назначение |
|---|---|---|
| `contour_processor.py` | 456 | Уточнение и постобработка контуров |
| `patch_sampler.py` | 439 | Выборка патчей: регулярная/случайная/по краям |
| `morphology_ops.py` | 391 | Эрозия, дилатация, opening, closing |
| `binarizer.py` | 388 | Otsu, Sauvola, Niblack, Bernsen |
| `perspective.py` | 387 | Коррекция перспективы (для фото) |
| `frequency_analyzer.py` | 385 | FFT-спектр, детект. периодических паттернов |
| `edge_detector.py` | 384 | Canny, Laplacian, Sobel |
| `contrast.py` | 376 | CLAHE, гистограммное выравнивание, гамма |
| `noise_reduction.py` | 371 | Расширенное шумоподавление (NLM, bilateral) |
| `skew_correction.py` | 370 | Hough-based коррекция наклона |
| `adaptive_threshold.py` | 354 | Адаптивная бинаризация |
| `warp_corrector.py` | 352 | Коррекция геометрических искажений |
| `texture_analyzer.py` | 349 | LBP, Gabor текстурные дескрипторы |
| `augment.py` | 340 | Flip, rotate, noise аугментация (для обучения) |
| `illumination_normalizer.py` | 335 | Batch-нормализация освещённости |
| `edge_sharpener.py` | 328 | Повышение резкости (unsharp mask) |
| `contrast_enhancer.py` | 321 | Целенаправленное улучшение контраста |
| `color_normalizer.py` | 316 | Gray World, LAB цветовая нормализация |
| `gradient_analyzer.py` | 315 | Градиентное поле (Sobel, Scharr) |
| `illumination_corrector.py` | 304 | Ретинекс, гомоморфная фильтрация |
| `document_cleaner.py` | 300 | Удаление теней, рамок, пятен |
| `edge_enhancer.py` | 298 | Усиление границ (Laplacian + blending) |
| `background_remover.py` | 296 | GrabCut, threshold-based удаление фона |
| `frequency_filter.py` | 292 | Low/high/band-pass FFT-фильтрация |
| `noise_filter.py` | 291 | Gaussian, median, bilateral |
| `patch_normalizer.py` | 287 | Нормализация патчей для матчинга |
| `image_enhancer.py` | 277 | Комплексный автоулучшитель |
| `fragment_cropper.py` | 275 | Автообрезка к содержательной области |
| `deskewer.py` | 271 | Hough/FFT/projection коррекция наклона |
| `quality_assessor.py` | 269 | Blur, noise, contrast, completeness |
| `noise_analyzer.py` | 254 | σ, SNR, JPEG-артефакты, зернистость |
| `noise_reducer.py` | 253 | Лёгкое шумоподавление с авто-оценкой |
| `channel_splitter.py` | 234 | Разделение RGB/HSV/LAB каналов |
| `denoise.py` | 231 | Gaussian/Median/Bilateral/NLM |

---

## 6. `algorithms/` — 42 модуля

Строк всего: **12 634** (вкл. `__init__.py` — 817 стр.)

### ✅ Всегда активные (5 модулей — статические импорты)

| Модуль | Стр. | Загружается из |
|---|---|---|
| `synthesis.py` | 137 | `main.py` напрямую |
| `fractal/css.py` | 184 | `synthesis.py` (строка 17) |
| `tangram/inscriber.py` | 95 | `main.py` напрямую |
| `tangram/hull.py` | 61 | `inscriber.py` |
| `tangram/classifier.py` | 78 | `inscriber.py` |

### 🔄 Лениво активные (3 модуля — внутри `compute_fractal_signature()`)

```python
# algorithms/synthesis.py строки 26-27:
def compute_fractal_signature(contour):
    from .fractal.box_counting import box_counting_fd   # ← ленивый
    from .fractal.divider       import divider_fd        # ← ленивый
    from .fractal.ifs           import fit_ifs_coefficients  # ← ленивый
```

| Модуль | Стр. | Функция |
|---|---|---|
| `fractal/box_counting.py` | 86 | `box_counting_fd()` — фрактальная размерность методом Минковского |
| `fractal/divider.py` | 98 | `divider_fd()` — метод Ричардсона |
| `fractal/ifs.py` | 148 | `fit_ifs_coefficients()` — фрактальная интерполяция Барнсли |

### 💤 Спящие (34 модуля — ~10 320 строк)

| Модуль | Стр. | Назначение |
|---|---|---|
| `edge_profile.py` | 446 | Профиль края: направление, кривизна, длина |
| `line_detector.py` | 438 | Детектирование прямых (Hough) |
| `word_segmentation.py` | 427 | Сегментация слов |
| `boundary_descriptor.py` | 410 | Дескриптор границы фрагмента |
| `fragment_classifier.py` | 407 | Классификация типа фрагмента |
| `region_segmenter.py` | 404 | Сегментация регионов |
| `patch_aligner.py` | 375 | Выравнивание патчей |
| `contour_smoother.py` | 374 | Сглаживание контура (spline, Bezier) |
| `homography_estimator.py` | 371 | Оценка гомографии |
| `edge_comparator.py` | 371 | Попарное сравнение краёв |
| `fragment_quality.py` | 369 | Качество отдельного фрагмента |
| `shape_context.py` | 366 | Shape Context (Belongie 2002) |
| `fragment_aligner.py` | 361 | Геометрическое выравнивание |
| `path_planner.py` | 360 | Планирование маршрута обхода |
| `patch_matcher.py` | 359 | Сопоставление патчей |
| `descriptor_combiner.py` | 353 | Комбинирование дескрипторов |
| `descriptor_aggregator.py` | 349 | Агрегация по фрагменту |
| `seam_evaluator.py` | 347 | Оценка качества шва |
| `region_scorer.py` | 342 | Оценка совпадения регионов |
| `rotation_estimator.py` | 333 | Угол поворота фрагмента |
| `color_palette.py` | 333 | Цветовая палитра (k-means) |
| `sift_matcher.py` | 332 | SIFT + FLANN матчинг |
| `gradient_flow.py` | 326 | Градиентный поток |
| `edge_scorer.py` | 317 | Оценка совпадения краёв |
| `edge_extractor.py` | 310 | Извлечение граничных пикселей |
| `texture_descriptor.py` | 306 | Gabor, LBP, GLCM |
| `fourier_descriptor.py` | 304 | Дескрипторы Фурье |
| `contour_tracker.py` | 302 | Трекинг контура |
| `color_space.py` | 297 | RGB↔HSV↔LAB↔YCrCb |
| `overlap_resolver.py` | 282 | Устранение перекрытий |
| `region_splitter.py` | 281 | Разбиение регионов |
| `position_estimator.py` | 274 | 2D-позиция фрагмента |
| `score_aggregator.py` | 264 | Агрегация оценок |
| `edge_filter.py` | 190 | Фильтрация краёв |

---

## 7. `matching/` — 27 модулей

Строк всего: **8 672** (вкл. `__init__.py` — 484 стр.)

> **Почему 27, а не 26?** Модуль `matcher_registry.py` добавлен коммитом `6c98327` —
> он не существовал когда писался REPORT.md (305 модулей → стало 306).

### ✅ Всегда активные (3 модуля)

| Модуль | Стр. | Загружается из |
|---|---|---|
| `compat_matrix.py` | 64 | `main.py` напрямую |
| `pairwise.py` | 161 | `compat_matrix.py` |
| `dtw.py` | 58 | `pairwise.py` |

### 🔄 Лениво активный (1 модуль — при нестандартном конфиге)

| Модуль | Стр. | Условие |
|---|---|---|
| `matcher_registry.py` | 260 | `cfg.active_matchers` содержит матчер не из `["css","dtw","fd","text"]` |

`matcher_registry.py` при загрузке регистрирует **15 матчеров** через `try/except`:

| # | Имя | Из модуля | Стр. |
|---|---|---|---|
| 1 | `css` | `fractal/css.py` | 184 |
| 2 | `dtw` | `matching/dtw.py` | 58 |
| 3 | `fd` | из модели | — |
| 4 | `text` | внешний сигнал | — |
| 5 | `icp` | `matching/icp.py` | 291 |
| 6 | `color` | `matching/color_match.py` | 400 |
| 7 | `texture` | `matching/texture_match.py` | 354 |
| 8 | `seam` | `matching/seam_score.py` | 310 |
| 9 | `geometric` | `matching/geometric_match.py` | 292 |
| 10 | `boundary` | `matching/boundary_matcher.py` | 372 |
| 11 | `affine` | `matching/affine_matcher.py` | 329 |
| 12 | `spectral` | `matching/spectral_matcher.py` | 279 |
| 13 | `shape_context` | `matching/shape_matcher.py` | 317 |
| 14 | `patch` | `matching/patch_matcher.py` | 313 |
| 15 | `feature` | `matching/feature_match.py` | 324 |

**Активные по умолчанию** (без конфига): `css`, `dtw`, `fd`, `text` с весами 0.35/0.30/0.20/0.15.

### 💤 Спящие (23 модуля)

| Модуль | Стр. | Назначение |
|---|---|---|
| `graph_match.py` | 407 | Граф: random walk similarity |
| `color_match.py` | 400 | Цветовые гистограммы (chi-squared) |
| `global_matcher.py` | 383 | Глобальный матчинг всех пар |
| `curve_descriptor.py` | 379 | Кривизна кривой края |
| `patch_validator.py` | 376 | Валидация патч-совпадений |
| `score_normalizer.py` | 375 | z-score, rank, min-max нормализация |
| `boundary_matcher.py` | 372 | Профиль границы фрагмента |
| `orient_matcher.py` | 359 | Матчинг по ориентации |
| `edge_comparator.py` | 355 | Попарное сравнение краёв |
| `texture_match.py` | 354 | LBP/Gabor текстурный матчинг |
| `score_aggregator.py` | 346 | Агрегация от N матчеров |
| `affine_matcher.py` | 329 | Аффинное выравнивание краёв |
| `feature_match.py` | 324 | SIFT/ORB точечный матчинг |
| `score_combiner.py` | 320 | `weighted_combine()`, `rank_combine()` |
| `shape_matcher.py` | 317 | Shape Context матчинг |
| `patch_matcher.py` | 313 | Патч-совпадение |
| `seam_score.py` | 310 | Непрерывность шва |
| `spectral_matcher.py` | 279 | Спектральные дескрипторы |
| `consensus.py` | 275 | Голосование между сборками |
| `icp.py` | 291 | Iterative Closest Point |
| `geometric_match.py` | 292 | Геометрические инварианты |
| `pair_scorer.py` | 290 | Итоговый скор пары |
| `candidate_ranker.py` | 199 | Ранжирование кандидатов |

---

## 8. `assembly/` — 27 модулей

Строк всего: **8 650** (вкл. `__init__.py` — 505 стр.)

### ✅ Всегда активный (1 модуль)

| Модуль | Стр. | Функции |
|---|---|---|
| `parallel.py` | 380 | `run_all_methods()`, `run_selected()`, `pick_best()`, `summary_table()` |

`ALL_METHODS = ["greedy","sa","beam","gamma","genetic","exhaustive","ant_colony","mcts"]`

### 🔄 Лениво активные (8 алгоритмов — dispatch внутри parallel.py)

| Модуль | Стр. | CLI-ключ | Лучший для |
|---|---|---|---|
| `greedy.py` | 146 | `greedy` | Baseline, O(N²) |
| `annealing.py` | 142 | `sa` | Быстрое улучшение |
| `beam_search.py` | 192 | `beam` | 6–20 фрагментов |
| `gamma_optimizer.py` | 281 | `gamma` | 20–100, SOTA |
| `exhaustive.py` | 223 | `exhaustive` | ≤8, точный O(N!) |
| `genetic.py` | 296 | `genetic` | 15–40 фрагментов |
| `ant_colony.py` | 270 | `ant_colony` | 20–60 фрагментов |
| `mcts.py` | 292 | `mcts` | 6–25 фрагментов |

### 💤 Спящие (18 вспомогательных модулей)

| Модуль | Стр. | Назначение |
|---|---|---|
| `canvas_builder.py` | 390 | Рендер итогового холста |
| `sequence_planner.py` | 383 | Планирование порядка сборки |
| `fragment_arranger.py` | 374 | Расстановка на холсте |
| `gap_analyzer.py` | 367 | Анализ зазоров между фрагментами |
| `layout_refiner.py` | 361 | Уточнение 2D-компоновки |
| `cost_matrix.py` | 358 | Матрица стоимостей |
| `overlap_resolver.py` | 346 | Разрешение перекрытий |
| `position_estimator.py` | 344 | 2D-позиции в сборке |
| `placement_optimizer.py` | 327 | Порядок размещения |
| `fragment_scorer.py` | 323 | Вклад фрагмента в сборку |
| `fragment_mapper.py` | 323 | Маппинг в документные зоны |
| `layout_builder.py` | 317 | Построение 2D-компоновки |
| `assembly_state.py` | 313 | Состояние процесса сборки |
| `fragment_sequencer.py` | 307 | Оптимальный порядок |
| `collision_detector.py` | 300 | AABB-детектирование коллизий |
| `candidate_filter.py` | 300 | Фильтрация пар-кандидатов |
| `fragment_sorter.py` | 281 | Сортировка перед сборкой |
| `score_tracker.py` | 209 | Трекинг эволюции score |

---

## 9. `verification/` — 21 модуль

Строк всего: **7 908** (вкл. `__init__.py` — 466 стр.)

### ✅ Всегда активный (1 модуль)

| Модуль | Стр. | Функции |
|---|---|---|
| `ocr.py` | 163 | `verify_full_assembly()`, `render_assembly_image()` |

При отсутствии Tesseract возвращает fallback-оценку `0.5`.

### 💤 Спящие (20 модулей — 7 279 строк)

| Модуль | Стр. | Что проверяет |
|---|---|---|
| `consistency_checker.py` | 551 | Глобальная согласованность сборки |
| `layout_verifier.py` | 485 | Итоговая верификация 2D-компоновки |
| `layout_checker.py` | 476 | Gap uniformity, column/row alignment |
| `text_coherence.py` | 400 | N-gram связность текста |
| `spatial_validator.py` | 391 | Пространственные связи, топология |
| `layout_scorer.py` | 373 | Оценка 2D-компоновки |
| `edge_validator.py` | 367 | Совместимость краёв на стыках |
| `confidence_scorer.py` | 361 | Уверенность в каждом Placement |
| `report.py` | 360 | Генерация финального отчёта |
| `fragment_validator.py` | 360 | Валидность фрагмента |
| `placement_validator.py` | 356 | Корректность Placement |
| `quality_reporter.py` | 354 | Полный качественный отчёт |
| `score_reporter.py` | 342 | Отчёт по оценкам |
| `assembly_scorer.py` | 338 | Суммарный score (A–F) |
| `overlap_validator.py` | 334 | Пересечения фрагментов |
| `seam_analyzer.py` | 325 | Gradient continuity через швы |
| `boundary_validator.py` | 306 | Граничные условия документа |
| `completeness_checker.py` | 292 | Все фрагменты размещены? |
| `overlap_checker.py` | 257 | IoU пересечений |
| `metrics.py` | 251 | IoU, Kendall τ, RMSE, угловая ошибка |

---

## 10. `scoring/` — 12 модулей ⚠️ СПЯЩИЙ

Строк всего: **4 219** (вкл. `__init__.py` — 303 стр.)

> ⚠️ **Полностью спящий субпакет.** Не импортируется ни в одном
> производственном модуле. Виден только тестам.

| Модуль | Стр. | Назначение |
|---|---|---|
| `threshold_selector.py` | 403 | Otsu, Kapur, Triangle, адаптивные пороги |
| `consistency_checker.py` | 345 | Согласованность оценок |
| `pair_filter.py` | 335 | Фильтрация пар по порогу |
| `gap_scorer.py` | 337 | Анализ зазоров ← **содержит единственный провалившийся тест** |
| `score_normalizer.py` | 341 | z-score, min-max, rank нормализация |
| `match_evaluator.py` | 349 | Precision/recall пар |
| `boundary_scorer.py` | 340 | Граничные совпадения |
| `match_scorer.py` | 324 | Итоговый скор пары |
| `pair_ranker.py` | 330 | Ранжирование пар |
| `global_ranker.py` | 331 | Глобальное ранжирование |
| `evidence_aggregator.py` | 289 | Агрегация доказательств |
| `rank_fusion.py` | 192 | RRF и Borda count |

---

## 11. `io/` — 3 модуля ⚠️ СПЯЩИЙ

Строк всего: **1 141** (вкл. `__init__.py` — 81 стр.)

> ⚠️ **Полностью спящий субпакет.** Не импортируется нигде в production.
>
> **Ловушка:** `from .io import ...` в `utils/__init__.py` — это `utils/io.py`,
> **не** этот субпакет. `puzzle_reconstruction.io` виден только тестам (9 файлов).
>
> **Ловушка №2:** `export.py` в корне пакета тоже спит. В `__init__.py` строки 11–14
> выглядят как импорт, но находятся внутри строки документации — это пример кода,
> не исполняемый Python.

| Модуль | Стр. | Функции |
|---|---|---|
| `result_exporter.py` | 397 | Экспорт PNG/PDF/JSON/HTML, `ExportConfig` |
| `metadata_writer.py` | 338 | `write_json()`, `write_yaml()`, `AssemblyMetadata` |
| `image_loader.py` | 325 | `load_image()`, `load_image_dir()`, `ImageRecord` |

---

## 12. `utils/` — 130 модулей

Строк всего: **38 970**

### ✅ Всегда активные (2 модуля)

| Модуль | Загружается из | Функции |
|---|---|---|
| `logger.py` | `main.py`, `pipeline.py` | `get_logger()`, `stage()`, `PipelineTimer` |
| `event_bus.py` | `pipeline.py` | `EventBus`, `make_event_bus()` |

### 💤 Спящие (~128 модулей)

**Пайплайн-инфраструктура (готова к подключению):**

| Модуль | Функции |
|---|---|
| `pipeline_runner.py` | Multi-step runner с retry |
| `batch_processor.py` | Пакетная обработка |
| `progress_tracker.py` | Трекер прогресса с ETA |
| `config_manager.py` | YAML/JSON конфиг |
| `result_cache.py` | LRU + TTL кэш |
| `cache_manager.py` | Дисковый кэш |
| `metric_tracker.py` | Трекинг метрик |
| `event_log.py` | Детальный журнал |
| `profiler.py` | `@timed`, `PipelineProfiler` |

**Геометрия и CV:** `geometry_utils`, `transform_utils`, `icp_utils`, `polygon_ops_utils`,
`bbox_utils`, `distance_utils`, `contour_utils`, `rotation_utils`, `spatial_index`

**Изображения:** `image_io`, `image_transform_utils`, `mask_layout_utils`, `patch_extractor`,
`color_utils`, `histogram_utils`, `keypoint_utils`

**Математика:** `signal_utils`, `smoothing_utils`, `interpolation_utils`, `stats_utils`,
`array_utils`, `sparse_utils`, `clustering_utils`, `distance_shape_utils`

**Граф:** `graph_utils`, `graph_cache_utils`

**Метрики:** `metrics`, `score_matrix_utils`, `score_norm_utils`, `quality_score_utils`,
`score_seam_utils`, `scoring_pipeline_utils`

**Визуализация:** `visualizer`, `render_utils`

---

## 13. `ui/` — 1 модуль

| Модуль | Стр. | Статус | Условие загрузки |
|---|---|---|---|
| `viewer.py` | 364 | 🔄 Лениво активен | `main.py:358` при флаге `--interactive` |

```python
# main.py строка 356-360:
if args.interactive:
    from puzzle_reconstruction.ui.viewer import show
    assembly = show(assembly, output_path=str(output_path))
```

---

## 14. Корневые модули пакета

`puzzle_reconstruction/__init__.py` (57 строк) — публичный API v0.3.0:

```python
# Фактически импортируется (строки 20-32):
from .models    import Fragment, Assembly, CompatEntry, EdgeSignature, …
from .config    import Config
from .clustering import cluster_fragments, ClusteringResult, split_by_cluster
from .pipeline  import Pipeline, PipelineResult
```

| Модуль | Стр. | Статус | Причина |
|---|---|---|---|
| `models.py` | 129 | ✅ Всегда активен | `__init__.py` строка 20 |
| `config.py` | 175 | ✅ Всегда активен | `__init__.py` строка 30 |
| `clustering.py` | 316 | ✅ Всегда активен | `__init__.py` строка 31 |
| `pipeline.py` | 377 | ✅ Всегда активен | `__init__.py` строка 32 |
| `export.py` | 474 | 💤 **СПИТ** | В `__init__.py` только в docstring (строки 11–14), не в коде |

---

## 15. CLI

### Все параметры `main.py` (384 строки)

| Параметр | Тип | Default | Описание |
|---|---|---|---|
| `--input, -i` | path | **обязательный** | Директория со сканами |
| `--output, -o` | path | `result.png` | Результат |
| `--config, -c` | path | — | JSON/YAML конфиг |
| `--method, -M` | str | `beam` | 10 вариантов (см. ниже) |
| `--alpha` | float | — | Вес танграма 0..1 |
| `--n-sides` | int | — | Краёв на фрагмент |
| `--seg-method` | str | — | otsu/adaptive/grabcut |
| `--threshold` | float | — | Порог совместимости |
| `--beam-width` | int | — | Ширина beam search |
| `--sa-iter` | int | — | Итерации отжига |
| `--mcts-sim` | int | — | Симуляции MCTS |
| `--genetic-pop` | int | — | Популяция генетика |
| `--genetic-gen` | int | — | Поколений генетика |
| `--aco-ants` | int | — | Агентов-муравьёв |
| `--aco-iter` | int | — | Итерации ACO |
| `--auto-timeout` | float | — | Таймаут (сек) для auto/all |
| `--visualize, -v` | flag | False | Окно OpenCV |
| `--interactive, -I` | flag | False | Интерактивный редактор |
| `--verbose` | flag | False | DEBUG лог |
| `--log-file` | path | — | Файл лога |

> ⚠️ **Несоответствие в коде:** докстринг `main.py` (строки 11–15) перечисляет
> только 4 метода (greedy/sa/beam/gamma). Реально поддерживается **10**:
> greedy, sa, beam, gamma, genetic, exhaustive, ant_colony, mcts, auto, all.

### Поддерживаемые методы сборки

| Метод | Сложность | Качество | Детерм. | Лучший сценарий |
|---|---|---|---|---|
| `exhaustive` | O(N!) | ⭐⭐⭐⭐⭐ | ✅ | ≤8 фрагментов |
| `beam` | O(W·N²) | ⭐⭐⭐⭐ | ✅ | 6–20 фрагментов |
| `mcts` | O(S·D) | ⭐⭐⭐⭐ | ❌ | 6–25 фрагментов |
| `genetic` | O(G·P·N²) | ⭐⭐⭐⭐ | ❌ | 15–40 фрагментов |
| `ant_colony` | O(I·A·N²) | ⭐⭐⭐⭐ | ❌ | 20–60 фрагментов |
| `gamma` | O(I·N²) | ⭐⭐⭐⭐⭐ | ❌ | 20–100, SOTA |
| `sa` | O(I) | ⭐⭐⭐ | ❌ | Быстрое улучшение |
| `greedy` | O(N²) | ⭐⭐ | ✅ | Baseline |
| `auto` | — | — | — | Авто по N фрагментов |
| `all` | — | — | — | Research: все 8 + таблица |

---

## 16. `tools/` — 6 скриптов

| Скрипт | Стр. | Команда | Назначение |
|---|---|---|---|
| `benchmark.py` | 363 | `puzzle-benchmark` | Все методы → JSON-отчёт с ground truth |
| `evaluate.py` | 311 | `puzzle-evaluate` | NA, DC, RMSE → HTML/JSON/Markdown |
| `profile.py` | 387 | `puzzle-profile` | cProfile+pstats по этапам |
| `server.py` | 309 | `puzzle-server` | Flask REST: `/api/reconstruct`, `/api/cluster`, ... |
| `tear_generator.py` | 303 | `puzzle-generate` | Perlin-разрыв → N фрагментов + ground_truth.json |
| `mix_documents.py` | 271 | `puzzle-mix` | Смешивание фрагментов N документов |

---

## 17. Технический стек

### Обязательные зависимости

| | Версия | Роль |
|---|---|---|
| Python | ≥3.11 (3.11.14) | Язык |
| numpy | ≥1.24 (2.4.2) | Массивы |
| scipy | ≥1.11 | FFT, spatial, оптимизация |
| opencv-python | ≥4.8 | CV, `cv2.imread` |
| scikit-image | ≥0.22 | Алгоритмы изображений |
| Pillow | ≥10.0 | Чтение/запись |
| scikit-learn | ≥1.3 | Кластеризация, PCA |

### Опциональные

| Группа | Пакет | Роль |
|---|---|---|
| `[ocr]` | pytesseract | OCR верификация |
| `[yaml]` | pyyaml | YAML конфиг |
| `[pdf]` | reportlab, fpdf2 | PDF экспорт |
| `[api]` | flask | REST сервер |

> **Несоответствие:** `shapely`, `networkx`, `matplotlib` — в `requirements.txt`,
> но не в `pyproject.toml[dependencies]`.

---

## 18. История разработки

```
20 фев   iter-1…36    Фаза 1: создание 305 модулей + базовые тесты
21–23 фев iter-37…191 Фаза 2: _extra.py тесты (2 модуля/итерацию)
23–24 фев iter-192…249 Фаза 3: utils полностью (4 модуля/итерацию)
24 фев   2a4c0bb      REPORT.md (133 провала)
24 фев   c3c44c3      Placement/Edge models исправлены
24 фев   d290633      REPORT.md обновлён
24 фев   6c98327 ★    feat: все 8 алгоритмов + matcher_registry (306-й модуль)
24 фев   aa47360      fix: -58 провалов
24 фев   445b7f6      fix: -26 провалов
24 фев   47d5fbe      fix: -4 провала → **1 провал**
25 фев   d2fd92b      docs: STATUS.md v1
25 фев   1b2881c      docs: STATUS.md v2 (live data)
25 фев   4d9ee84      docs: STATUS.md v3 (scoring/io sleeping)
25 фев   07fd44f      docs: STATUS.md v4 (per-module audit)
25 фев   (текущий)    docs: STATUS.md v5 (полный аудит + исправления)
```

**260 коммитов:** 212 `iter-NNN:` + ~15 `feat:` + ~23 `fix:` + ~10 `docs:`

---

## 19. Единственный провалившийся тест

```
FAILED tests/test_scoring_gap_scorer.py::TestFilterGapMeasures::test_filter_all_out
```

### Суть — внутреннее противоречие в тест-файле

В **одном и том же файле** два теста проверяют противоположное поведение
`filter_gap_measures(r, 1.1)`:

```python
# ТЕСТ 1 — ПРОВАЛЕН (строка ~338):
def test_filter_all_out(self):
    r = self._report()
    assert filter_gap_measures(r, 1.1) == []        # ожидает []

# ТЕСТ 2 — ПРОХОДИТ (строка ~343):
def test_min_score_above_one_raises(self):
    r = self._report()
    with pytest.raises(ValueError):                 # ожидает ValueError
        filter_gap_measures(r, 1.1)                # тот же вызов, другое ожидание!
```

Функция выбрасывает `ValueError` при `min_score > 1.0` → `test_filter_all_out` падает,
`test_min_score_above_one_raises` проходит. Это противоречие внутри одного тест-класса.

### Исправление (1 строка)

```python
# Вариант А — исправить тест (правильно, ∵ функция документирует ValueError):
def test_filter_all_out(self):
    r = self._report()
    with pytest.raises(ValueError):        # ← заменить assert на raises
        filter_gap_measures(r, 1.1)

# Вариант Б — исправить функцию (если нужна мягкая обрезка):
# scoring/gap_scorer.py:276 — заменить raise ValueError на clip:
min_score = min(max(min_score, 0.0), 1.0)
```

**Влияние:** нулевое — `scoring/gap_scorer.py` спит (не вызывается из production).

---

## 20. Дорожная карта

### Выполнено полностью ✅

| Задача | Коммит |
|---|---|
| Реализация 306 модулей | iter-1…249 + `6c98327` |
| 100% покрытие модулей тестами | iter-1…249 |
| Все 8 алгоритмов сборки в CLI | `6c98327` |
| 15 матчеров в `matcher_registry` | `6c98327` |
| REST API сервер | `tools/server.py` |
| Бенчмарк, профилировщик, генератор | `tools/` |

### Осталось

| Задача | Приоритет | Оценка |
|---|---|---|
| Исправить 1 провалившийся тест | 🔴 Высокий | 1 строка |
| Обновить docstring `main.py` (4→10 методов) | 🔴 Высокий | 2 строки |
| Убрать `xfail` с 9 xpassed-тестов | 🟡 Средний | 9 строк |
| Подключить `PreprocessingChain` (34 модуля) | 🟡 Средний | ~60 строк |
| Подключить `VerificationSuite` (20 модулей) | 🟡 Средний | ~80 строк |
| Подключить `scoring/` в `pairwise.py` | 🟡 Средний | ~40 строк |
| Подключить `io/image_loader.py` в `main.py` | 🟡 Средний | ~20 строк |
| Подключить `io/result_exporter.py` | 🟡 Средний | ~15 строк |
| Добавить `export.py` в `__init__.py` | 🟡 Средний | 1 строка |
| Утилиты: `event_bus`, `result_cache`, ... | 🟢 Низкий | ~30 строк |
| YAML-конфигурация | 🟢 Низкий | ~20 строк |
| Согласовать `requirements.txt` / `pyproject.toml` | 🟢 Низкий | 3 строки |

---

## 21. Итоговая оценка

### По компонентам

| Компонент | Реализован | В production | Тесты | Статус |
|---|---|---|---|---|
| Алгоритм (Танграм+Фракталы+Синтез) | 100% | ✅ всегда | ✅ | **Production** |
| Пайплайн (6 этапов, `Pipeline`) | 100% | ✅ всегда | ✅ | **Production** |
| CLI (`main.py`, 20 параметров) | 100% | ✅ всегда | ✅ | **Production** |
| 8 алгоритмов сборки | 100% | 🔄 лениво | ✅ | **Production** |
| 15 матчеров (4 дефолт + 11 опц.) | 100% | 🔄 лениво | ✅ | **Beta** |
| Preprocessing активная (4 модуля) | 100% | ✅ всегда | ✅ | **Production** |
| Preprocessing спящая (34 модуля) | 100% | 💤 не подключена | ✅ | **Ready** |
| Verification/ocr | 100% | ✅ всегда | ✅ | **Production** |
| Verification спящая (20 модулей) | 100% | 💤 не подключена | ✅ | **Ready** |
| `scoring/` (12 модулей) ⚠️ | 100% | 💤 **не подключён** | ✅ | **Ready** |
| `io/` (3 модуля) ⚠️ | 100% | 💤 **не подключён** | ✅ | **Ready** |
| `export.py` (корень) ⚠️ | 100% | 💤 **не подключён** | ✅ | **Ready** |
| `clustering.py` (корень) | 100% | ✅ всегда (API) | ✅ | **Production** |
| `ui/viewer.py` | 100% | 🔄 `--interactive` | ✅ | **Beta** |
| `utils/` (~128 спящих) | 100% | 💤 не подключена | ✅ | **Alpha** |
| `tools/` (6 скриптов) | 100% | ✅ 6 CLI-команд | ✅ | **Beta** |
| REST API | 100% | `puzzle-server` | — | **Alpha** |
| Документация | 100% | — | — | **Production** |
| Тесты | 100% модулей | 99.976% pass | — | **Production** |

### Общий вывод

```
Реализовано:      306 модулей  (100%)
Покрыто тестами:  306 модулей  (100%)
Активно:           33 модуля   (10.8%) — всегда(20) + лениво(13)
Ждут подключения: 273 модуля   (89.2%)

Провалившихся тестов: 1 из 42 280 (0.002%)
```

Проект находится в стадии **позднего Alpha**:

- **Ядро системы** (алгоритм + CLI + 8 алгоритмов сборки + матчеры) → **Production ready**
- **89% кодовой базы** реализованы, протестированы, но не подключены к основному потоку
- **Переход в Beta** требует `PreprocessingChain` + `VerificationSuite` (~140 строк кода)
- **Переход в Stable** — полная интеграция `scoring/`, `io/`, `utils/` инфраструктуры

---

*STATUS.md v5 · 2026-02-25 · Основан на полном аудите 306 модулей и живом тест-прогоне.*
*Следующее обновление — при изменении статуса компонентов.*
