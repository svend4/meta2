#!/usr/bin/env python3
"""
Лёгкий REST API сервер для системы восстановления пазлов.

Позволяет загружать фрагменты документа и получать результат сборки
через HTTP. Используется для демонстрации и интеграции с внешними системами.

Эндпоинты:
    GET  /health                   — состояние сервера
    GET  /config                   — текущая конфигурация
    GET  /api/methods              — список методов сборки
    GET  /api/validators           — список валидаторов VerificationSuite (21 шт.)
    POST /api/reconstruct          — загрузить фрагменты → JSON с результатом
    POST /api/cluster              — определить принадлежность к документам
    GET  /api/report/<job_id>      — получить отчёт по завершённому заданию
    GET  /api/report/<job_id>/html — HTML-версия отчёта
    GET  /spec                     — OpenAPI 3.0 JSON-спецификация

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

app         = Flask(__name__)
JOBS: Dict[str, dict] = {}   # job_id → {status, result, report, ts}
LOCK        = threading.Lock()
_START_TIME = time.perf_counter()


# ─── /health ──────────────────────────────────────────────────────────────

_VERSION = "1.0.0"


@app.get("/health")
def health():
    """Проверка работоспособности сервера."""
    return jsonify({
        "status":    "ok",
        "version":   _VERSION,
        "jobs":      len(JOBS),
        "uptime_s":  round(time.perf_counter() - _START_TIME, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


# ─── /spec (OpenAPI 3.0) ──────────────────────────────────────────────────

@app.get("/spec")
def openapi_spec():
    """OpenAPI 3.0 JSON-спецификация всех эндпоинтов."""
    spec = {
        "openapi": "3.0.3",
        "info": {
            "title":       "puzzle-reconstruction API",
            "version":     _VERSION,
            "description": "REST API для реконструкции разорванных документов из отсканированных фрагментов.",
            "license":     {"name": "MIT"},
        },
        "servers": [{"url": "/", "description": "Текущий сервер"}],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Проверка работоспособности",
                    "operationId": "health",
                    "responses": {
                        "200": {
                            "description": "Сервер работает",
                            "content": {"application/json": {"schema": {
                                "type": "object",
                                "properties": {
                                    "status":    {"type": "string", "example": "ok"},
                                    "version":   {"type": "string"},
                                    "jobs":      {"type": "integer"},
                                    "uptime_s":  {"type": "number"},
                                    "timestamp": {"type": "string"},
                                },
                            }}},
                        }
                    },
                }
            },
            "/config": {
                "get": {
                    "summary": "Конфигурация по умолчанию",
                    "operationId": "getConfig",
                    "responses": {
                        "200": {"description": "Объект конфигурации", "content": {"application/json": {"schema": {"type": "object"}}}},
                    },
                }
            },
            "/api/methods": {
                "get": {
                    "summary": "Список доступных методов сборки",
                    "operationId": "listMethods",
                    "responses": {
                        "200": {"description": "Методы", "content": {"application/json": {"schema": {"type": "object"}}}},
                    },
                }
            },
            "/api/validators": {
                "get": {
                    "summary": "Список доступных валидаторов VerificationSuite",
                    "operationId": "listValidators",
                    "responses": {
                        "200": {
                            "description": "21 валидатор с группировкой original/extended",
                            "content": {"application/json": {"schema": {
                                "type": "object",
                                "properties": {
                                    "total":      {"type": "integer", "example": 21},
                                    "validators": {"type": "array", "items": {"type": "string"}},
                                    "groups":     {"type": "object"},
                                    "usage":      {"type": "object"},
                                },
                            }}},
                        },
                        "500": {"description": "Внутренняя ошибка"},
                    },
                }
            },
            "/api/reconstruct": {
                "post": {
                    "summary": "Реконструировать документ из фрагментов",
                    "operationId": "reconstruct",
                    "requestBody": {
                        "required": True,
                        "content": {"multipart/form-data": {"schema": {
                            "type": "object",
                            "properties": {
                                "files":      {"type": "array", "items": {"type": "string", "format": "binary"}, "description": "PNG/JPEG изображения фрагментов"},
                                "method":     {"type": "string", "enum": ["greedy", "sa", "beam", "gamma", "genetic", "exhaustive", "ant_colony", "mcts", "auto", "all"], "default": "beam"},
                                "alpha":      {"type": "number", "minimum": 0, "maximum": 1, "description": "Вес алгоритма Танграм"},
                                "n_sides":    {"type": "integer", "minimum": 2, "description": "Ожидаемое число краёв на фрагмент"},
                                "threshold":  {"type": "number", "minimum": 0, "maximum": 1, "description": "Минимальная оценка совместимости"},
                                "validators": {"type": "string", "description": "'all' или список через запятую. Активирует VerificationSuite — поле 'verification' появится в ответе."},
                            },
                            "required": ["files"],
                        }}},
                    },
                    "responses": {
                        "200": {
                            "description": "Результат сборки",
                            "content": {"application/json": {"schema": {
                                "type": "object",
                                "properties": {
                                    "job_id":      {"type": "string"},
                                    "n_input":     {"type": "integer"},
                                    "n_placed":    {"type": "integer"},
                                    "score":       {"type": "number"},
                                    "ocr":         {"type": "number"},
                                    "elapsed_sec": {"type": "number"},
                                    "method":      {"type": "string"},
                                    "placements":  {"type": "object"},
                                },
                            }}},
                        },
                        "400": {"description": "Некорректный запрос"},
                    },
                }
            },
            "/api/cluster": {
                "post": {
                    "summary": "Разбить фрагменты по документам",
                    "operationId": "cluster",
                    "requestBody": {
                        "required": True,
                        "content": {"multipart/form-data": {"schema": {
                            "type": "object",
                            "properties": {
                                "files":  {"type": "array", "items": {"type": "string", "format": "binary"}},
                                "k":      {"type": "integer", "minimum": 0, "description": "Число документов (0 = авто)"},
                                "method": {"type": "string", "enum": ["kmeans", "gmm", "spectral"], "default": "gmm"},
                            },
                            "required": ["files"],
                        }}},
                    },
                    "responses": {
                        "200": {"description": "Результат кластеризации"},
                        "400": {"description": "Некорректный запрос"},
                        "422": {"description": "Нет читаемых фрагментов"},
                    },
                }
            },
            "/api/report/{job_id}": {
                "get": {
                    "summary": "JSON-отчёт по завершённому заданию",
                    "operationId": "reportJson",
                    "parameters": [{"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "responses": {
                        "200": {"description": "Отчёт"},
                        "404": {"description": "Задание не найдено"},
                    },
                }
            },
            "/api/report/{job_id}/html": {
                "get": {
                    "summary": "HTML-отчёт по завершённому заданию",
                    "operationId": "reportHtml",
                    "parameters": [{"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "responses": {
                        "200": {"description": "HTML-страница отчёта", "content": {"text/html": {}}},
                        "404": {"description": "Задание не найдено"},
                    },
                }
            },
            "/spec": {
                "get": {
                    "summary": "OpenAPI спецификация",
                    "operationId": "spec",
                    "responses": {
                        "200": {"description": "OpenAPI 3.0 JSON"},
                    },
                }
            },
        },
    }
    return jsonify(spec)


# ─── /api/methods ─────────────────────────────────────────────────────────

@app.get("/api/methods")
def list_methods():
    """Список доступных методов сборки с описаниями."""
    return jsonify({
        "methods": [
            {"id": "greedy",     "desc": "Жадный алгоритм (O(N²), детерминирован, быстрый)",       "n_range": "любой"},
            {"id": "sa",         "desc": "Имитация отжига (O(I), стохастический)",                   "n_range": "любой"},
            {"id": "beam",       "desc": "Beam search (O(W·N²), детерминирован) — рекомендуется",   "n_range": "6–20"},
            {"id": "gamma",      "desc": "Gamma optimizer (O(I·N²), SOTA качество)",                 "n_range": "20–100"},
            {"id": "genetic",    "desc": "Генетический алгоритм (O(G·P·N²))",                        "n_range": "15–40"},
            {"id": "exhaustive", "desc": "Полный перебор — гарантированный оптимум",                 "n_range": "≤8"},
            {"id": "ant_colony", "desc": "Муравьиные колонии ACO (O(I·A·N²))",                       "n_range": "20–60"},
            {"id": "mcts",       "desc": "Monte Carlo Tree Search (O(S·D))",                         "n_range": "6–25"},
            {"id": "auto",       "desc": "Автовыбор метода по числу фрагментов",                     "n_range": "любой"},
            {"id": "all",        "desc": "Все 8 методов — выбор лучшего по score",                   "n_range": "любой"},
        ],
        "default": "beam",
    })


# ─── /api/validators ──────────────────────────────────────────────────────

@app.get("/api/validators")
def list_validators():
    """Список всех 21 доступных валидаторов VerificationSuite."""
    try:
        from puzzle_reconstruction.verification.suite import (
            all_validator_names, _build_validator_registry,
        )
        names = all_validator_names()
        # Разбиваем на оригинальные 9 и новые 12
        original_9 = ["assembly_score", "layout", "completeness", "seam",
                       "overlap", "text_coherence", "confidence",
                       "consistency", "edge_quality"]
        return jsonify({
            "total":      len(names),
            "validators": names,
            "groups": {
                "original": [n for n in names if n in original_9],
                "extended": [n for n in names if n not in original_9],
            },
            "usage": {
                "cli_all":    "--validators all",
                "cli_subset": "--validators boundary,metrics,placement",
                "api_all":    "POST /api/reconstruct + form field: validators=all",
            },
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


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

    # Параметр validators: "all" или список через запятую
    validators_param = request.form.get("validators", None)
    if validators_param:
        from puzzle_reconstruction.verification.suite import all_validator_names
        if validators_param.strip().lower() == "all":
            cfg.verification.validators = all_validator_names()
        else:
            cfg.verification.validators = [
                v.strip() for v in validators_param.split(",") if v.strip()
            ]

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

    response = {
        "job_id":      job_id,
        "n_input":     result.n_input,
        "n_placed":    result.n_placed,
        "score":       float(assembly.total_score),
        "ocr":         float(assembly.ocr_score),
        "elapsed_sec": round(elapsed, 3),
        "method":      cfg.assembly.method,
        "placements":  placements_dict,
    }

    # Если VerificationSuite был запущен — включаем отчёт
    if result.verification_report is not None:
        response["verification"] = result.verification_report.as_dict()

    return jsonify(response)


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
                         choices=["greedy", "sa", "beam", "gamma",
                                  "genetic", "exhaustive", "ant_colony", "mcts",
                                  "auto", "all"])
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════╗
║   Puzzle Reconstruction API  {_VERSION:<18} ║
║   http://{args.host}:{args.port:<5}                         ║
║                                                  ║
║   Эндпоинты:                                     ║
║     GET  /health          — статус сервера       ║
║     GET  /config          — конфигурация         ║
║     GET  /spec            — OpenAPI 3.0 JSON     ║
║     GET  /api/methods     — список методов       ║
║     GET  /api/validators  — список валидаторов   ║
║     POST /api/reconstruct — реконструкция        ║
║     POST /api/cluster     — кластеризация        ║
║     GET  /api/report/<id> — JSON-отчёт           ║
║     GET  /api/report/<id>/html — HTML-отчёт      ║
╚══════════════════════════════════════════════════╝
""")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
