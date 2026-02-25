# meta2 — Восстановление разорванных документов

Программа для автоматической сборки разорванных газет, книг и документов
из отсканированных фрагментов.

> **Статус**: **v1.0.0 Production/Stable** — 305 модулей, 42 384 теста, 100% покрытие модулей.
> Последнее обновление: февраль 2026 г.

## Алгоритм

Два взаимодополняющих описания каждого края:

```
         ВНУТРИ фрагмента          СНАРУЖИ фрагмента
                 │                        │
        Танграм-контур              Фрактальная кривая
        (геометрически              (форма «береговой линии»)
         правильный)
                 └───────────┬────────────┘
                             │ СИНТЕЗ
                        EdgeSignature
                    (уникальная подпись края)
```

**Алгоритм 1 — Танграм**: вписываем фрагмент в геометрическую фигуру (выпуклая
оболочка → упрощение RDP → нормализация). Описывает крупную геометрию.

**Алгоритм 2 — Фрактальная кромка**: вычисляем фрактальную размерность края
тремя методами (Box-counting, Divider, IFS Барнсли) + CSS-дескриптор (MPEG-7).
Описывает мелкие детали разрыва.

**Синтез**: `B_virtual = α·B_tangram + (1-α)·B_fractal` — уникальный «отпечаток»
каждого края, инвариантный к освещению и сканированию.

---

## Быстрый старт

```bash
pip install -r requirements.txt

# Для OCR (опционально)
sudo apt install tesseract-ocr tesseract-ocr-rus

# Одиночный документ — метод beam (по умолчанию)
python main.py --input scans/ --output result.png

# Автовыбор алгоритма по числу фрагментов
python main.py --input scans/ --method auto

# Все 8 алгоритмов, выбрать лучший
python main.py --input scans/ --method all

# Исследовательский режим: сравнение + консенсус + JSON-экспорт
python main.py --input scans/ --method all --research --export-json report.json

# Пакетная обработка нескольких документов
python main.py --input-list dirs.txt --output results/

# Интерактивный просмотр с визуализацией
python main.py --input scans/ --visualize --interactive

# Верификация: показать список валидаторов
python main.py --list-validators

# Верификация: запустить все 21 валидатор и экспортировать отчёт
python main.py --input scans/ --validators all --export-report report.html

# Верификация: выбранные валидаторы + JSON-отчёт
python main.py --input scans/ --validators boundary,metrics,completeness --export-report report.json
```

---

## Алгоритмы сборки (`--method`)

| Метод | Фрагментов | Качество | Описание |
|---|---|---|---|
| `exhaustive` | ≤ 8 | ⭐⭐⭐⭐⭐ | Полный перебор, гарантированный оптимум |
| `beam` | 6–20 | ⭐⭐⭐⭐ | Beam search, скорость + качество (по умолчанию) |
| `mcts` | 6–25 | ⭐⭐⭐⭐ | Monte Carlo Tree Search |
| `genetic` | 15–40 | ⭐⭐⭐⭐ | Генетический алгоритм |
| `ant_colony` | 20–60 | ⭐⭐⭐⭐ | Оптимизация муравьиной колонией |
| `gamma` | 20–100 | ⭐⭐⭐⭐⭐ | Гамма-распределение отклонений, SOTA |
| `sa` | любой | ⭐⭐⭐ | Имитация отжига, быстрое улучшение |
| `greedy` | любой | ⭐⭐ | Жадный, базовая линия |
| `auto` | любой | — | Автовыбор по числу фрагментов |
| `all` | любой | — | Все 8 методов, победитель по score |

---

## Параметры командной строки

```
Основные:
  --input DIR            Директория с фрагментами
  --input-list FILE      Файл-список директорий (пакетная обработка)
  --output PATH          Путь результата (по умолчанию: result.png)
  --config FILE          JSON/YAML файл конфигурации
  --method METHOD        Алгоритм сборки (см. таблицу выше, default: beam)

Признаки фрагментов:
  --alpha FLOAT          Вес танграма в синтезе EdgeSignature (0..1)
  --n-sides INT          Ожидаемое число краёв на фрагмент
  --seg-method STR       Метод сегментации: otsu / adaptive / grabcut
  --threshold FLOAT      Минимальная оценка совместимости краёв

Параметры алгоритмов:
  --beam-width INT       Ширина луча для beam search
  --sa-iter INT          Итерации имитации отжига
  --mcts-sim INT         Симуляции MCTS
  --genetic-pop INT      Размер популяции GA
  --genetic-gen INT      Число поколений GA
  --aco-ants INT         Агенты-муравьи ACO
  --aco-iter INT         Итерации ACO
  --auto-timeout FLOAT   Таймаут на один метод в режиме auto/all (сек)

Режимы вывода:
  --visualize            Показать результат в окне OpenCV
  --interactive          Интерактивный просмотрщик (Minority Report style)
  --verbose              Подробный лог
  --log-file FILE        Записать лог в файл

Research mode:
  --research             Включить режим сравнения методов и консенсуса
  --no-consensus         Отключить консенсусную сборку в research mode
  --export-json FILE     Экспортировать сравнительный отчёт в JSON

Кэш:
  --cache-dir DIR        Директория для кэша дескрипторов

Верификация (VerificationSuite — 21 валидатор):
  --list-validators      Показать список всех 21 валидаторов и выйти
  --validators LIST      Запустить валидаторы (через запятую) или 'all' для всех 21
                           Примеры: --validators all
                                    --validators boundary,metrics,placement
  --export-report PATH   Экспортировать отчёт верификации:
                           report.json  — структурированный JSON
                           report.md    — Markdown-таблица
                           report.html  — HTML-страница с CSS
```

---

## Структура проекта

```
main.py                                  # Точка входа (CLI)
requirements.txt
puzzle_reconstruction/
├── models.py                            # Данные: Fragment, EdgeSignature, ...
├── pipeline.py                          # Конвейер обработки
├── config.py                            # Конфигурация (dataclass)
├── export.py                            # Экспорт результатов
├── clustering.py                        # Кластеризация фрагментов
│
├── preprocessing/   (38 модулей)        # Предобработка изображений
│   ├── segmentation.py                  # Выделение маски фрагмента
│   ├── contour.py                       # Контур, RDP, разбиение на края
│   ├── orientation.py                   # Ориентация по тексту
│   ├── chain.py                         # PreprocessingChain (конфигурируемый)
│   ├── denoise.py / contrast.py / ...   # 35 специализированных фильтров
│   └── ...
│
├── algorithms/      (42 модуля)         # Алгоритмы описания формы
│   ├── tangram/                         # Танграм-аппроксимация
│   │   ├── hull.py                      # Convex hull, RDP, нормализация
│   │   ├── classifier.py                # Классификация формы полигона
│   │   └── inscriber.py                 # Вписывание танграм-фигуры
│   ├── fractal/                         # Фрактальная кромка
│   │   ├── box_counting.py              # FD методом Box-counting
│   │   ├── divider.py                   # FD методом Ричардсона
│   │   ├── ifs.py                       # Фрактальная интерполяция Барнсли
│   │   └── css.py                       # Curvature Scale Space (MPEG-7)
│   └── synthesis.py                     # Синтез EdgeSignature
│
├── matching/        (26 модулей)        # Попарное сопоставление краёв
│   ├── pairwise.py                      # Попарная оценка (CSS+DTW+FD+TEXT)
│   ├── matcher_registry.py              # Реестр всех матчеров (@register)
│   ├── compat_matrix.py                 # Матрица совместимости N×N
│   ├── dtw.py                           # Dynamic Time Warping
│   ├── icp.py / color_match.py / ...    # 13+ специализированных матчеров
│   ├── score_combiner.py                # weighted/rank/min/max комбинация
│   ├── consensus.py                     # Голосование между методами
│   └── ...
│
├── assembly/        (27 модулей)        # Глобальная сборка
│   ├── greedy.py                        # Жадный алгоритм
│   ├── annealing.py                     # Имитация отжига (SA)
│   ├── beam_search.py                   # Beam search
│   ├── gamma_optimizer.py               # Гамма-оптимизация (SOTA)
│   ├── genetic.py                       # Генетический алгоритм
│   ├── exhaustive.py                    # Полный перебор (N≤8)
│   ├── ant_colony.py                    # Муравьиная оптимизация
│   ├── mcts.py                          # Monte Carlo Tree Search
│   └── parallel.py                      # Реестр + параллельный запуск всех
│
├── verification/    (21 модуль)         # Верификация результата
│   ├── ocr.py                           # OCR-верификация (Tesseract)
│   ├── suite.py                         # VerificationSuite (конфигурируемый)
│   ├── metrics.py                       # IoU, Kendall τ, RMSE
│   ├── seam_analyzer.py                 # Качество швов
│   ├── text_coherence.py                # Текстовая связность
│   └── ...
│
├── scoring/         (12 модулей)        # Оценка качества
├── io/              (3 модуля)          # Ввод/вывод, метаданные
├── ui/              (1 модуль)          # Интерактивный просмотрщик
└── utils/           (130 модулей)       # Геометрия, кэш, визуализация, ...
```

**Всего: 305 модулей, ~93 000 строк кода.**

---

## Документация

| Файл | Содержание |
|---|---|
| [PUZZLE_RECONSTRUCTION.md](PUZZLE_RECONSTRUCTION.md) | Полная техническая документация с математикой и псевдокодом |
| [INTEGRATION_ROADMAP.md](INTEGRATION_ROADMAP.md) | Архитектурные решения, карта интеграции всех модулей |
| [REPORT.md](REPORT.md) | Отчёт о тестовом покрытии (42 208 тестов, 0 сбоев) |
