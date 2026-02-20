#!/usr/bin/env python3
"""
Лёгкий REST API сервер для системы восстановления пазлов.

Позволяет загружать фрагменты документа и получать результат сборки
через HTTP. Используется для демонстрации и интеграции с внешними системами.

Эндпоинты:
    GET  /health                   — состояние сервера
    GET  /config                   — текущая конфигурация
    POST /api/reconstruct          — загрузить фрагменты → JSON с результатом
    POST /api/cluster              — определить принадлежность к документам
    GET  /api/report/<job_id>      — получить отчёт по завершённому заданию
    GET  /api/report/<job_id>/html — HTML-версия отчёта

Использование:
    pip install flask
    python tools/server.py

    # Или с настройками:
    python tools/server.py --host 0.0.0.0 --port 8080 --method beam

Пример запроса (curl):
    curl -X POST http://localhost:5000/api/reconstruct \\
         -F "files=@scan1.png" -F "files=@scan2.png" \\
         -F "method=beam" -F "n_sides=4"
"""
import io
import json
import os
import sys
import time
import uuid
import argparse
import tempfile
import threading
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
try:
    import cv2
except ImportError:
    print("Нужен opencv-python: pip install opencv-python")
    sys.exit(1)

try:
    from flask import Flask, request, jsonify, Response
except ImportError:
    print("Нужен Flask: pip install flask")
    sys.exit(1)

from puzzle_reconstruction.config import Config
from puzzle_reconstruction.pipeline import Pipeline
from puzzle_reconstruction.clustering import cluster_fragments
from puzzle_reconstruction.verification.report import build_report
from puzzle_reconstruction.export import render_canvas, render_heatmap, render_mosaic


# ─── Приложение ───────────────────────────────────────────────────────────

app  = Flask(__name__)
JOBS: Dict[str, dict] = {}   # job_id → {status, result, report, ts}
LOCK = threading.Lock()


# ─── /health ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Проверка работоспособности сервера."""
    return jsonify({
        "status":    "ok",
        "version":   "0.2.0",
        "jobs":      len(JOBS),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


# ─── /config ──────────────────────────────────────────────────────────────

@app.get("/config")
def get_config():
    """Возвращает конфигурацию по умолчанию."""
    return jsonify(Config.default().to_dict())


# ─── POST /api/reconstruct ───────────────────────────────────────────────

@app.post("/api/reconstruct")
def reconstruct():
    """
    Принимает multipart/form-data с полями:
        files[]   — PNG/JPEG изображения фрагментов
        method    — метод сборки (greedy|sa|beam|gamma|exhaustive)
        alpha     — вес танграма (0..1)
        n_sides   — ожидаемое число краёв на фрагмент
        threshold — минимальная оценка совместимости

    Возвращает JSON:
        job_id    — идентификатор задания (для получения отчёта)
        n_placed  — число размещённых фрагментов
        score     — уверенность сборки
        ocr       — OCR-связность
        placements — {frag_id: {x, y, angle_deg}}
    """
    files = request.files.getlist("files") or request.files.getlist("files[]")
    if not files:
        return jsonify({"error": "Нет файлов. Передайте files[] как multipart/form-data"}), 400

    # ── Читаем параметры ──────────────────────────────────────────────────
    cfg = Config.default()
    cfg.apply_overrides(
        method    = request.form.get("method",    None),
        alpha     = _float_or_none(request.form.get("alpha")),
        n_sides   = _int_or_none(request.form.get("n_sides")),
        threshold = _float_or_none(request.form.get("threshold")),
    )
    cfg.verification.run_ocr = False   # OCR отключён по умолчанию для скорости

    # ── Декодируем изображения ────────────────────────────────────────────
    images = []
    for f in files:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": f"Не удалось декодировать: {f.filename}"}), 400
        images.append(img)

    # ── Запуск пайплайна ──────────────────────────────────────────────────
    t0       = time.perf_counter()
    pipeline = Pipeline(cfg=cfg, n_workers=4)
    result   = pipeline.run(images)
    elapsed  = time.perf_counter() - t0

    assembly = result.assembly

    # ── Формируем ответ ───────────────────────────────────────────────────
    placements_dict = {}
    for fid, (pos, angle) in assembly.placements.items():
        placements_dict[str(fid)] = {
            "x":         float(pos[0]),
            "y":         float(pos[1]),
            "angle_deg": float(np.degrees(angle)),
        }

    job_id = str(uuid.uuid4())[:8]
    with LOCK:
        JOBS[job_id] = {
            "status":    "done",
            "assembly":  assembly,
            "result":    result,
            "ts":        time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed":   elapsed,
        }

    return jsonify({
        "job_id":     job_id,
        "n_input":    result.n_input,
        "n_placed":   result.n_placed,
        "score":      float(assembly.total_score),
        "ocr":        float(assembly.ocr_score),
        "elapsed_sec": round(elapsed, 3),
        "method":     cfg.assembly.method,
        "placements": placements_dict,
    })


# ─── POST /api/cluster ────────────────────────────────────────────────────

@app.post("/api/cluster")
def cluster():
    """
    Разбивает смешанные фрагменты по документам.

    Параметры:
        files[]   — изображения фрагментов
        k         — число документов (0 = авто)
        method    — kmeans|gmm|spectral
    """
    files = request.files.getlist("files") or request.files.getlist("files[]")
    if not files:
        return jsonify({"error": "Нет файлов"}), 400

    k_req  = _int_or_none(request.form.get("k"))
    method = request.form.get("method", "gmm")

    cfg = Config.default()
    cfg.verification.run_ocr = False

    images = []
    for f in files:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)

    if not images:
        return jsonify({"error": "Нет читаемых изображений"}), 400

    pipeline  = Pipeline(cfg=cfg, n_workers=4)
    fragments = pipeline.preprocess(images)

    if not fragments:
        return jsonify({"error": "Ни один фрагмент не обработан"}), 422

    clust_result = cluster_fragments(
        fragments, k=k_req, method=method, seed=42
    )

    return jsonify({
        "n_fragments": len(fragments),
        "n_clusters":  clust_result.n_clusters,
        "silhouette":  round(float(clust_result.silhouette), 4),
        "labels":      clust_result.labels.tolist(),
        "clusters":    [{"doc_id": i, "frag_ids": group}
                         for i, group in enumerate(clust_result.cluster_groups)],
    })


# ─── GET /api/report/<job_id> ─────────────────────────────────────────────

@app.get("/api/report/<job_id>")
def report_json(job_id: str):
    """Возвращает JSON-отчёт по завершённому заданию."""
    job = _get_job(job_id)
    if job is None:
        return jsonify({"error": "Задание не найдено"}), 404

    report = build_report(job["assembly"], pipeline_result=job["result"])
    return jsonify(report.to_dict())


@app.get("/api/report/<job_id>/html")
def report_html(job_id: str):
    """Возвращает HTML-отчёт по завершённому заданию."""
    job = _get_job(job_id)
    if job is None:
        return Response("Задание не найдено", status=404, mimetype="text/plain")

    assembly = job["assembly"]
    canvas   = render_canvas(assembly)
    heatmap  = render_heatmap(assembly, canvas.shape)
    mosaic   = render_mosaic(assembly)

    report = build_report(job["assembly"], pipeline_result=job["result"],
                           canvas=canvas, heatmap=heatmap, mosaic=mosaic)
    return Response(report.to_html(), status=200, mimetype="text/html; charset=utf-8")


# ─── Утилиты ──────────────────────────────────────────────────────────────

def _get_job(job_id: str) -> dict | None:
    with LOCK:
        return JOBS.get(job_id)


def _float_or_none(s) -> float | None:
    if s is None:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _int_or_none(s) -> int | None:
    if s is None:
        return None
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


# ─── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="REST API сервер восстановления документов",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("--port",   type=int, default=5000)
    parser.add_argument("--debug",  action="store_true")
    parser.add_argument("--method", default="beam",
                         choices=["greedy", "sa", "beam", "gamma", "exhaustive"])
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════╗
║   Puzzle Reconstruction API  v0.2.0             ║
║   http://{args.host}:{args.port:<5}                         ║
║                                                  ║
║   Эндпоинты:                                     ║
║     GET  /health                                 ║
║     POST /api/reconstruct                        ║
║     POST /api/cluster                            ║
║     GET  /api/report/<job_id>/html               ║
╚══════════════════════════════════════════════════╝
""")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
