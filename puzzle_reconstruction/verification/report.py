"""
Генератор отчётов о результатах восстановления документа.

Поддерживаемые форматы:
    JSON      — машиночитаемый, подходит для CI-пайплайнов
    Markdown  — для GitHub / документации
    HTML      — полноценный отчёт с встроенными изображениями (base64)

Использование:
    report = build_report(assembly, pipeline_result, metrics)
    report.save_json("report.json")
    report.save_markdown("report.md")
    report.save_html("report.html")
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class FragmentInfo:
    fragment_id:  int
    shape_class:  str
    fd_box:       float
    fd_divider:   float
    n_edges:      int
    placed:       bool
    position:     Optional[list]   # [x, y] или None
    angle_deg:    float


@dataclass
class ReportData:
    """Полная структура отчёта (сериализуется в JSON)."""
    timestamp:          str
    method:             str
    n_input:            int
    n_placed:           int
    assembly_score:     float
    ocr_score:          float
    neighbor_accuracy:  Optional[float] = None
    direct_comparison:  Optional[float] = None
    position_rmse:      Optional[float] = None
    angular_error_deg:  Optional[float] = None
    perfect:            Optional[bool]  = None
    runtime_sec:        Optional[float] = None
    fragments:          list = field(default_factory=list)
    config:             dict = field(default_factory=dict)
    notes:              str  = ""


class Report:
    """
    Объект отчёта с методами экспорта в JSON / Markdown / HTML.
    """

    def __init__(self, data: ReportData,
                  canvas: Optional[np.ndarray] = None,
                  heatmap: Optional[np.ndarray] = None,
                  mosaic: Optional[np.ndarray] = None):
        self.data    = data
        self.canvas  = canvas
        self.heatmap = heatmap
        self.mosaic  = mosaic

    # ── JSON ─────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self.data)

    def save_json(self, path: str | Path, indent: int = 2) -> None:
        """Сохраняет отчёт в JSON."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)

    # ── Markdown ─────────────────────────────────────────────────────────

    def to_markdown(self) -> str:
        d  = self.data
        ok = "✓" if d.perfect else "✗"
        lines = [
            f"# Отчёт: восстановление документа",
            f"",
            f"**Дата:** {d.timestamp}  |  **Метод:** `{d.method}`",
            f"",
            f"## Сводка",
            f"",
            f"| Показатель | Значение |",
            f"|---|---|",
            f"| Фрагментов (вход) | {d.n_input} |",
            f"| Размещено | {d.n_placed} / {d.n_input} |",
            f"| Уверенность сборки | {d.assembly_score:.1%} |",
            f"| OCR-связность | {d.ocr_score:.1%} |",
        ]

        if d.neighbor_accuracy is not None:
            lines += [
                f"| Neighbor Accuracy | {d.neighbor_accuracy:.1%} |",
                f"| Direct Comparison | {d.direct_comparison:.1%} |",
                f"| Position RMSE | {d.position_rmse:.1f} px |",
                f"| Angular Error | {d.angular_error_deg:.1f}° |",
                f"| Perfect | {ok} |",
            ]

        if d.runtime_sec is not None:
            lines.append(f"| Время | {d.runtime_sec:.2f} с |")

        lines += [
            f"",
            f"## Фрагменты ({len(d.fragments)})",
            f"",
            f"| ID | Форма | FD (box) | FD (div) | Краёв | Размещён |",
            f"|---|---|---|---|---|---|",
        ]
        for fr in d.fragments:
            placed = "✓" if fr["placed"] else "✗"
            lines.append(
                f"| {fr['fragment_id']} | {fr['shape_class']} | "
                f"{fr['fd_box']:.3f} | {fr['fd_divider']:.3f} | "
                f"{fr['n_edges']} | {placed} |"
            )

        if d.notes:
            lines += ["", f"## Примечания", "", d.notes]

        return "\n".join(lines)

    def save_markdown(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(self.to_markdown(), encoding="utf-8")

    # ── HTML ─────────────────────────────────────────────────────────────

    def to_html(self) -> str:
        d   = self.data
        md_table = self._md_to_html_table()

        # Встраиваем изображения как base64
        canvas_tag  = self._img_tag(self.canvas,  "Восстановленный документ")
        heatmap_tag = self._img_tag(self.heatmap, "Тепловая карта уверенности")
        mosaic_tag  = self._img_tag(self.mosaic,  "Мозаика фрагментов")

        frag_rows = ""
        for fr in d.fragments:
            placed = '<span style="color:green">✓</span>' \
                if fr["placed"] else '<span style="color:red">✗</span>'
            frag_rows += (
                f"<tr><td>{fr['fragment_id']}</td>"
                f"<td>{fr['shape_class']}</td>"
                f"<td>{fr['fd_box']:.3f}</td>"
                f"<td>{fr['fd_divider']:.3f}</td>"
                f"<td>{fr['n_edges']}</td>"
                f"<td>{placed}</td></tr>\n"
            )

        na_row  = ""
        if d.neighbor_accuracy is not None:
            perfect = "ДА" if d.perfect else "нет"
            na_row = f"""
            <tr><td>Neighbor Accuracy</td><td>{d.neighbor_accuracy:.1%}</td></tr>
            <tr><td>Direct Comparison</td><td>{d.direct_comparison:.1%}</td></tr>
            <tr><td>Position RMSE</td><td>{d.position_rmse:.1f} px</td></tr>
            <tr><td>Angular Error</td><td>{d.angular_error_deg:.1f}°</td></tr>
            <tr><td>Perfect</td><td>{perfect}</td></tr>
            """

        rt_row = ""
        if d.runtime_sec is not None:
            rt_row = f"<tr><td>Время</td><td>{d.runtime_sec:.2f} с</td></tr>"

        html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <title>Отчёт: восстановление документа</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto;
            padding: 20px; background: #f5f5f5; color: #333; }}
    h1   {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; }}
    h2   {{ color: #34495e; margin-top: 30px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0;
             background: white; border-radius: 6px; overflow: hidden;
             box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
    th   {{ background: #3498db; color: white; padding: 10px 14px; text-align: left; }}
    td   {{ padding: 8px 14px; border-bottom: 1px solid #eee; }}
    tr:hover td {{ background: #f0f7ff; }}
    .meta {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 20px; }}
    .score-high {{ color: #27ae60; font-weight: bold; }}
    .score-low  {{ color: #e74c3c; font-weight: bold; }}
    .imgs {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }}
    .img-block {{ text-align: center; }}
    .img-block img {{ max-width: 500px; max-height: 400px;
                       border: 1px solid #ddd; border-radius: 4px;
                       box-shadow: 0 2px 6px rgba(0,0,0,0.15); }}
    .img-block p {{ font-size: 0.85em; color: #7f8c8d; margin: 4px 0; }}
    pre {{ background: #2c3e50; color: #ecf0f1; padding: 16px;
            border-radius: 6px; overflow-x: auto; font-size: 0.85em; }}
  </style>
</head>
<body>
  <h1>Отчёт: восстановление разорванного документа</h1>
  <p class="meta">Дата: {d.timestamp} &nbsp;|&nbsp; Метод: <code>{d.method}</code></p>

  <h2>Сводные показатели</h2>
  <table>
    <tr><th>Показатель</th><th>Значение</th></tr>
    <tr><td>Фрагментов (вход)</td><td>{d.n_input}</td></tr>
    <tr><td>Размещено</td><td>{d.n_placed} / {d.n_input}</td></tr>
    <tr><td>Уверенность сборки</td>
        <td class="{'score-high' if d.assembly_score > 0.7 else 'score-low'}">
            {d.assembly_score:.1%}</td></tr>
    <tr><td>OCR-связность</td>
        <td class="{'score-high' if d.ocr_score > 0.6 else 'score-low'}">
            {d.ocr_score:.1%}</td></tr>
    {na_row}
    {rt_row}
  </table>

  <h2>Визуализация</h2>
  <div class="imgs">
    {canvas_tag}
    {heatmap_tag}
    {mosaic_tag}
  </div>

  <h2>Фрагменты</h2>
  <table>
    <tr>
      <th>ID</th><th>Форма</th><th>FD (box)</th>
      <th>FD (div)</th><th>Краёв</th><th>Размещён</th>
    </tr>
    {frag_rows}
  </table>

  <h2>Конфигурация</h2>
  <pre>{json.dumps(d.config, indent=2, ensure_ascii=False)}</pre>

  {"<h2>Примечания</h2><p>" + d.notes + "</p>" if d.notes else ""}
</body>
</html>"""
        return html

    def save_html(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(self.to_html(), encoding="utf-8")

    # ── Вспомогательные ──────────────────────────────────────────────────

    @staticmethod
    def _img_tag(img: Optional[np.ndarray], caption: str) -> str:
        if img is None:
            return ""
        success, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            return ""
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return (
            f'<div class="img-block">'
            f'<img src="data:image/jpeg;base64,{b64}" alt="{caption}">'
            f'<p>{caption}</p></div>'
        )

    def _md_to_html_table(self) -> str:
        """Возвращает HTML-таблицу из Markdown-источника (заглушка)."""
        return ""


# ─── Фабричная функция ────────────────────────────────────────────────────

def build_report(assembly,
                  pipeline_result=None,
                  metrics=None,
                  notes: str = "",
                  canvas: Optional[np.ndarray] = None,
                  heatmap: Optional[np.ndarray] = None,
                  mosaic: Optional[np.ndarray] = None) -> Report:
    """
    Создаёт объект Report из результатов пайплайна.

    Args:
        assembly:         Assembly — результат сборки.
        pipeline_result:  PipelineResult (опционально) — статистика этапов.
        metrics:          ReconstructionMetrics (опционально) — метрики качества.
        notes:            Дополнительный текст.
        canvas:           Изображение сборки для встраивания в HTML.
        heatmap:          Тепловая карта для встраивания в HTML.
        mosaic:           Мозаика фрагментов для встраивания в HTML.

    Returns:
        Report готовый к экспорту.
    """
    # Метаданные метода
    method = "unknown"
    cfg_dict = {}
    runtime = None
    n_input = len(assembly.fragments)

    if pipeline_result is not None:
        method   = pipeline_result.cfg.assembly.method
        cfg_dict = pipeline_result.cfg.to_dict()
        runtime  = sum(pipeline_result.timer._stages.values())
        n_input  = pipeline_result.n_input

    # Информация о фрагментах
    frag_infos = []
    for frag in assembly.fragments:
        fid   = frag.fragment_id
        placed = fid in assembly.placements
        pos_raw, angle = assembly.placements.get(fid, (None, 0.0))
        pos_list = pos_raw.tolist() if pos_raw is not None else None

        info = FragmentInfo(
            fragment_id  = fid,
            shape_class  = frag.tangram.shape_class.value if frag.tangram else "unknown",
            fd_box       = frag.fractal.fd_box      if frag.fractal  else 0.0,
            fd_divider   = frag.fractal.fd_divider  if frag.fractal  else 0.0,
            n_edges      = len(frag.edges),
            placed       = placed,
            position     = pos_list,
            angle_deg    = float(np.degrees(angle)),
        )
        frag_infos.append(asdict(info))

    # Метрики качества
    na, dc, rmse, ang, perfect = None, None, None, None, None
    if metrics is not None:
        na      = metrics.neighbor_accuracy
        dc      = metrics.direct_comparison
        rmse    = metrics.position_rmse
        ang     = metrics.angular_error_deg
        perfect = metrics.perfect

    data = ReportData(
        timestamp          = time.strftime("%Y-%m-%d %H:%M:%S"),
        method             = method,
        n_input            = n_input,
        n_placed           = len(assembly.placements),
        assembly_score     = float(assembly.total_score),
        ocr_score          = float(assembly.ocr_score),
        neighbor_accuracy  = na,
        direct_comparison  = dc,
        position_rmse      = rmse,
        angular_error_deg  = ang,
        perfect            = perfect,
        runtime_sec        = runtime,
        fragments          = frag_infos,
        config             = cfg_dict,
        notes              = notes,
    )

    return Report(data, canvas=canvas, heatmap=heatmap, mosaic=mosaic)
