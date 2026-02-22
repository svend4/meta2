"""Экспорт результатов сборки пазла в различные форматы.

Модуль предоставляет функции для сохранения результатов сборки:
экспорт в JSON, CSV, сохранение изображения с разметкой, генерация
текстового отчёта, сборка сводной таблицы и пакетный экспорт.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── ExportConfig ─────────────────────────────────────────────────────────────

_VALID_FORMATS = frozenset({"json", "csv", "image", "text", "summary"})


@dataclass
class ExportConfig:
    """Конфигурация экспорта результатов.

    Атрибуты:
        fmt:          Формат экспорта ('json', 'csv', 'image', 'text', 'summary').
        output_path:  Путь к файлу (None → только в памяти).
        indent:       Отступ JSON (>= 0).
        draw_ids:     Рисовать идентификаторы фрагментов на изображении.
        draw_bboxes:  Рисовать ограничивающие прямоугольники.
        font_scale:   Масштаб шрифта (> 0).
        params:       Дополнительные параметры.
    """

    fmt: str = "json"
    output_path: Optional[str] = None
    indent: int = 2
    draw_ids: bool = True
    draw_bboxes: bool = True
    font_scale: float = 0.5
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fmt not in _VALID_FORMATS:
            raise ValueError(
                f"Неизвестный формат '{self.fmt}'. "
                f"Допустимые: {sorted(_VALID_FORMATS)}"
            )
        if self.indent < 0:
            raise ValueError(
                f"indent должен быть >= 0, получено {self.indent}"
            )
        if self.font_scale <= 0.0:
            raise ValueError(
                f"font_scale должен быть > 0, получено {self.font_scale}"
            )


# ─── AssemblyResult ───────────────────────────────────────────────────────────

@dataclass
class AssemblyResult:
    """Результат сборки пазла для экспорта.

    Атрибуты:
        fragment_ids: Идентификаторы фрагментов.
        positions:    Позиции (x, y) каждого фрагмента.
        sizes:        Размеры (w, h) каждого фрагмента.
        scores:       Оценки совместимости (опционально, None → все 0).
        canvas_w:     Ширина холста (>= 1).
        canvas_h:     Высота холста (>= 1).
        metadata:     Произвольные метаданные.
    """

    fragment_ids: List[int]
    positions: List[Tuple[int, int]]
    sizes: List[Tuple[int, int]]
    canvas_w: int
    canvas_h: int
    scores: Optional[List[float]] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.fragment_ids)
        if len(self.positions) != n:
            raise ValueError(
                f"Длины fragment_ids ({n}) и positions "
                f"({len(self.positions)}) не совпадают"
            )
        if len(self.sizes) != n:
            raise ValueError(
                f"Длины fragment_ids ({n}) и sizes "
                f"({len(self.sizes)}) не совпадают"
            )
        if self.scores is not None and len(self.scores) != n:
            raise ValueError(
                f"Длины fragment_ids ({n}) и scores "
                f"({len(self.scores)}) не совпадают"
            )
        if self.canvas_w < 1:
            raise ValueError(
                f"canvas_w должен быть >= 1, получено {self.canvas_w}"
            )
        if self.canvas_h < 1:
            raise ValueError(
                f"canvas_h должен быть >= 1, получено {self.canvas_h}"
            )

    def __len__(self) -> int:
        return len(self.fragment_ids)


# ─── to_json ──────────────────────────────────────────────────────────────────

def to_json(result: AssemblyResult, indent: int = 2) -> str:
    """Сериализовать результат сборки в JSON-строку.

    Аргументы:
        result: AssemblyResult.
        indent: Отступ (>= 0).

    Возвращает:
        Строка JSON.

    Исключения:
        ValueError: Если indent < 0.
    """
    if indent < 0:
        raise ValueError(f"indent должен быть >= 0, получено {indent}")
    data: Dict[str, Any] = {
        "fragment_ids": result.fragment_ids,
        "positions": [list(p) for p in result.positions],
        "sizes": [list(s) for s in result.sizes],
        "canvas_w": result.canvas_w,
        "canvas_h": result.canvas_h,
        "scores": result.scores if result.scores is not None else [],
        "metadata": result.metadata,
    }
    return json.dumps(data, ensure_ascii=False, indent=indent)


# ─── from_json ────────────────────────────────────────────────────────────────

def from_json(json_str: str) -> AssemblyResult:
    """Десериализовать AssemblyResult из JSON-строки.

    Аргументы:
        json_str: JSON-строка.

    Возвращает:
        AssemblyResult.

    Исключения:
        ValueError: Если строка не является корректным JSON.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Некорректный JSON: {exc}") from exc

    return AssemblyResult(
        fragment_ids=data["fragment_ids"],
        positions=[tuple(p) for p in data["positions"]],
        sizes=[tuple(s) for s in data["sizes"]],
        canvas_w=data["canvas_w"],
        canvas_h=data["canvas_h"],
        scores=data.get("scores") or None,
        metadata=data.get("metadata", {}),
    )


# ─── to_csv ───────────────────────────────────────────────────────────────────

def to_csv(result: AssemblyResult) -> str:
    """Сериализовать результат сборки в CSV-строку.

    Столбцы: fragment_id, x, y, width, height[, score].

    Аргументы:
        result: AssemblyResult.

    Возвращает:
        CSV-строка.
    """
    output = io.StringIO()
    has_scores = result.scores is not None
    fieldnames = ["fragment_id", "x", "y", "width", "height"]
    if has_scores:
        fieldnames.append("score")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for i, (fid, (x, y), (w, h)) in enumerate(
        zip(result.fragment_ids, result.positions, result.sizes)
    ):
        row: Dict[str, Any] = {
            "fragment_id": fid, "x": x, "y": y, "width": w, "height": h
        }
        if has_scores:
            row["score"] = f"{result.scores[i]:.6f}"
        writer.writerow(row)
    return output.getvalue()


# ─── to_text_report ───────────────────────────────────────────────────────────

def to_text_report(result: AssemblyResult) -> str:
    """Сгенерировать читаемый текстовый отчёт о сборке.

    Аргументы:
        result: AssemblyResult.

    Возвращает:
        Многострочная строка отчёта.
    """
    lines = [
        "=" * 60,
        "ОТЧЁТ О СБОРКЕ ПАЗЛА",
        "=" * 60,
        f"Холст: {result.canvas_w} x {result.canvas_h} px",
        f"Фрагментов: {len(result)}",
    ]
    if result.metadata:
        lines.append("Метаданные:")
        for k, v in result.metadata.items():
            lines.append(f"  {k}: {v}")
    lines.append("-" * 60)
    lines.append(f"{'ID':>4}  {'X':>6}  {'Y':>6}  {'W':>6}  {'H':>6}"
                 + ("  {'Score':>8}" if result.scores else ""))
    for i, (fid, (x, y), (w, h)) in enumerate(
        zip(result.fragment_ids, result.positions, result.sizes)
    ):
        line = f"{fid:>4}  {x:>6}  {y:>6}  {w:>6}  {h:>6}"
        if result.scores is not None:
            line += f"  {result.scores[i]:>8.4f}"
        lines.append(line)
    lines.append("=" * 60)
    return "\n".join(lines)


# ─── render_annotated_image ────────────────────────────────────────────────────

def render_annotated_image(
    result: AssemblyResult,
    canvas: Optional[np.ndarray] = None,
    draw_ids: bool = True,
    draw_bboxes: bool = True,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Нарисовать ограничивающие прямоугольники и ID фрагментов на холсте.

    Аргументы:
        result:      AssemblyResult.
        canvas:      Фоновое изображение (uint8, 3-D). None → белый холст.
        draw_ids:    Рисовать идентификаторы.
        draw_bboxes: Рисовать рамки.
        font_scale:  Масштаб шрифта (> 0).

    Возвращает:
        Изображение (uint8, H×W×3) с разметкой.

    Исключения:
        ValueError: Если font_scale <= 0 или canvas некорректен.
    """
    if font_scale <= 0.0:
        raise ValueError(f"font_scale должен быть > 0, получено {font_scale}")

    H, W = result.canvas_h, result.canvas_w
    if canvas is None:
        img = np.full((H, W, 3), 255, dtype=np.uint8)
    else:
        canvas = np.asarray(canvas)
        if canvas.ndim != 3:
            raise ValueError(
                f"canvas должен быть 3-D (H×W×3), получено ndim={canvas.ndim}"
            )
        img = canvas.copy()

    color_bbox = (0, 120, 255)  # BGR синий
    color_text = (200, 50, 50)  # BGR тёмно-синий

    for fid, (x, y), (w, h) in zip(
        result.fragment_ids, result.positions, result.sizes
    ):
        if draw_bboxes:
            cv2.rectangle(img, (x, y), (x + w, y + h), color_bbox, 1)
        if draw_ids:
            cv2.putText(
                img, str(fid),
                (x + 2, y + max(12, int(14 * font_scale))),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                color_text, 1, cv2.LINE_AA,
            )
    return img


# ─── summary_table ────────────────────────────────────────────────────────────

def summary_table(results: List[AssemblyResult]) -> Dict[str, List]:
    """Построить сводную таблицу по списку результатов сборки.

    Аргументы:
        results: Список AssemblyResult.

    Возвращает:
        Словарь {имя_столбца: список_значений} для каждого результата.
    """
    table: Dict[str, List] = {
        "n_fragments": [],
        "canvas_w": [],
        "canvas_h": [],
        "mean_score": [],
        "min_score": [],
        "max_score": [],
    }
    for r in results:
        table["n_fragments"].append(len(r))
        table["canvas_w"].append(r.canvas_w)
        table["canvas_h"].append(r.canvas_h)
        if r.scores:
            arr = np.array(r.scores, dtype=np.float64)
            table["mean_score"].append(float(arr.mean()))
            table["min_score"].append(float(arr.min()))
            table["max_score"].append(float(arr.max()))
        else:
            table["mean_score"].append(None)
            table["min_score"].append(None)
            table["max_score"].append(None)
    return table


# ─── export_result ────────────────────────────────────────────────────────────

def export_result(
    result: AssemblyResult, config: ExportConfig
) -> Optional[str]:
    """Экспортировать результат согласно конфигурации.

    Аргументы:
        result: AssemblyResult.
        config: ExportConfig.

    Возвращает:
        Строку (JSON/CSV/текст/сводка) или None для формата 'image'.
    """
    if config.fmt == "json":
        content = to_json(result, indent=config.indent)
    elif config.fmt == "csv":
        content = to_csv(result)
    elif config.fmt == "text":
        content = to_text_report(result)
    elif config.fmt == "summary":
        tbl = summary_table([result])
        content = json.dumps(tbl, ensure_ascii=False, indent=config.indent)
    else:  # image
        img = render_annotated_image(
            result,
            draw_ids=config.draw_ids,
            draw_bboxes=config.draw_bboxes,
            font_scale=config.font_scale,
        )
        if config.output_path:
            cv2.imwrite(config.output_path, img)
        return None

    if config.output_path and config.fmt != "image":
        Path(config.output_path).write_text(content, encoding="utf-8")
    return content


# ─── batch_export ─────────────────────────────────────────────────────────────

def batch_export(
    results: List[AssemblyResult], config: ExportConfig
) -> List[Optional[str]]:
    """Пакетный экспорт нескольких результатов.

    Аргументы:
        results: Список AssemblyResult.
        config:  ExportConfig (output_path игнорируется для пакетного режима).

    Возвращает:
        Список строк (или None для image).
    """
    no_path = ExportConfig(
        fmt=config.fmt,
        output_path=None,
        indent=config.indent,
        draw_ids=config.draw_ids,
        draw_bboxes=config.draw_bboxes,
        font_scale=config.font_scale,
        params=dict(config.params),
    )
    return [export_result(r, no_path) for r in results]
