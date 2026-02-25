# ─────────────────────────────────────────────────────────────────────────────
# puzzle-reconstruction — многоэтапный Docker-образ
# ─────────────────────────────────────────────────────────────────────────────
# Стадия 1: builder — устанавливаем зависимости и собираем пакет
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Системные зависимости для OpenCV
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем только файлы зависимостей — кэш слоёв при пересборке
COPY requirements.txt pyproject.toml ./
COPY puzzle_reconstruction/__init__.py ./puzzle_reconstruction/

# Устанавливаем зависимости (без исходников пакета)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
      numpy scipy opencv-python-headless scikit-image Pillow scikit-learn \
      flask pyyaml shapely networkx matplotlib

# ─────────────────────────────────────────────────────────────────────────────
# Стадия 2: runtime — лёгкий финальный образ
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем установленные пакеты из builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копируем исходный код
COPY puzzle_reconstruction/ ./puzzle_reconstruction/
COPY tools/              ./tools/
COPY main.py             ./

# Создаём директории для данных и результатов
RUN mkdir -p /data/input /data/output /data/temp

# Непривилегированный пользователь
RUN useradd -m -u 1000 puzzle && chown -R puzzle:puzzle /app /data
USER puzzle

# ── Метаданные ────────────────────────────────────────────────────────────────
LABEL org.opencontainers.image.title="puzzle-reconstruction"
LABEL org.opencontainers.image.version="0.4.0-beta"
LABEL org.opencontainers.image.description="Automatic reconstruction of torn/shredded documents"
LABEL org.opencontainers.image.licenses="MIT"

# ── Переменные окружения ──────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PUZZLE_INPUT_DIR=/data/input \
    PUZZLE_OUTPUT_DIR=/data/output \
    PUZZLE_METHOD=auto

# ── Порт REST API ──────────────────────────────────────────────────────────────
EXPOSE 5000

# ── Точка входа по умолчанию: REST API ────────────────────────────────────────
CMD ["python", "tools/server.py", "--host", "0.0.0.0", "--port", "5000"]

# ── Альтернативные точки входа (переопределяются через docker run) ─────────────
# CLI-режим:
#   docker run puzzle-reconstruction python main.py --input /data/input --output /data/output/result.png
# Бенчмарк:
#   docker run puzzle-reconstruction python tools/benchmark.py
# Профилирование:
#   docker run puzzle-reconstruction python tools/profile.py --fragments 8
