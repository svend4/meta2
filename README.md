# meta2 — Восстановление разорванных документов

Программа для автоматической сборки разорванных газет, книг и документов
из отсканированных фрагментов.

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

## Быстрый старт

```bash
pip install -r requirements.txt

# Для OCR (опционально)
sudo apt install tesseract-ocr tesseract-ocr-rus

# Запуск
python main.py --input scans/ --output result.png
python main.py --input scans/ --output result.png --visualize
```

## Параметры

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--alpha` | 0.5 | Вес танграма vs фрактала (0..1) |
| `--n-sides` | 4 | Ожидаемое число краёв на фрагмент |
| `--seg-method` | otsu | Метод сегментации: otsu / adaptive / grabcut |
| `--threshold` | 0.3 | Минимальная оценка совместимости краёв |
| `--sa-iter` | 5000 | Итерации имитации отжига |
| `--visualize` | — | Показать результат в окне OpenCV |

## Структура проекта

```
main.py                                  # Точка входа
requirements.txt
PUZZLE_RECONSTRUCTION.md                 # Полная техническая документация
puzzle_reconstruction/
├── models.py                            # Данные: Fragment, EdgeSignature, ...
├── preprocessing/
│   ├── segmentation.py                  # Выделение маски фрагмента
│   ├── contour.py                       # Контур, RDP, разбиение на края
│   └── orientation.py                   # Ориентация по тексту
├── algorithms/
│   ├── tangram/
│   │   ├── hull.py                      # Convex hull, RDP, нормализация
│   │   ├── classifier.py                # Классификация формы полигона
│   │   └── inscriber.py                 # Вписывание танграм-фигуры
│   ├── fractal/
│   │   ├── box_counting.py              # FD методом Box-counting
│   │   ├── divider.py                   # FD методом Ричардсона (Divider)
│   │   ├── ifs.py                       # Фрактальная интерполяция Барнсли
│   │   └── css.py                       # Curvature Scale Space (MPEG-7)
│   └── synthesis.py                     # Синтез EdgeSignature
├── matching/
│   ├── dtw.py                           # Dynamic Time Warping
│   ├── pairwise.py                      # Попарная оценка совместимости
│   └── compat_matrix.py                 # Матрица совместимости N×N
├── assembly/
│   ├── greedy.py                        # Жадная начальная сборка
│   └── annealing.py                     # Оптимизация: имитация отжига
└── verification/
    └── ocr.py                           # Верификация через OCR
```

## Документация

Подробная техническая документация с псевдокодом, математическими формулами
и ссылками на научные работы: [PUZZLE_RECONSTRUCTION.md](PUZZLE_RECONSTRUCTION.md)
