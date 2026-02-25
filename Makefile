# ─────────────────────────────────────────────────────────────────────────────
# Makefile — puzzle-reconstruction v0.4.0-beta
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: help install install-all lint format typecheck test test-fast test-cov \
        profile benchmark server clean build docker docker-up docker-down

PYTHON   := python3
PIP      := pip3
PKG      := puzzle_reconstruction
TESTS    := tests/
PORT     := 5000

# ── Цвета ────────────────────────────────────────────────────────────────────
BOLD   := \033[1m
RESET  := \033[0m
GREEN  := \033[32m
YELLOW := \033[33m

help:  ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BOLD)%-20s$(RESET) %s\n", $$1, $$2}'

# ── Установка ────────────────────────────────────────────────────────────────
install:  ## Установить базовые зависимости (editable)
	$(PIP) install -e ".[dev]"

install-all:  ## Установить все зависимости включая опциональные
	$(PIP) install -e ".[all]"

# ── Качество кода ─────────────────────────────────────────────────────────────
lint:  ## Проверить код ruff (лайтер)
	ruff check $(PKG)/ tools/ \
	  --ignore E501,F401,N803,N806,UP006,UP007,UP035,RUF012

format:  ## Отформатировать код ruff
	ruff format $(PKG)/ tools/ main.py

typecheck:  ## Проверить типы mypy (ключевые модули)
	mypy \
	  $(PKG)/config.py \
	  $(PKG)/models.py \
	  $(PKG)/clustering.py \
	  $(PKG)/export.py \
	  $(PKG)/pipeline.py \
	  $(PKG)/algorithms/synthesis.py \
	  $(PKG)/algorithms/fractal/box_counting.py \
	  $(PKG)/algorithms/fractal/css.py \
	  $(PKG)/algorithms/fractal/divider.py \
	  $(PKG)/algorithms/tangram/hull.py \
	  $(PKG)/matching/pairwise.py \
	  $(PKG)/matching/compat_matrix.py \
	  $(PKG)/matching/matcher_registry.py \
	  --ignore-missing-imports \
	  --no-strict-optional \
	  --check-untyped-defs

qa: lint typecheck  ## Запустить lint + typecheck вместе

# ── Тесты ────────────────────────────────────────────────────────────────────
test:  ## Запустить все тесты
	$(PYTHON) -m pytest $(TESTS) -v --tb=short -q

test-fast:  ## Быстрые тесты (без интеграционных, -x)
	$(PYTHON) -m pytest $(TESTS) \
	  --ignore=$(TESTS)test_integration.py \
	  --ignore=$(TESTS)test_pipeline.py \
	  -q --tb=short -x

test-cov:  ## Тесты + отчёт о покрытии кода
	$(PYTHON) -m pytest $(TESTS) \
	  --ignore=$(TESTS)test_integration.py \
	  --ignore=$(TESTS)test_pipeline.py \
	  --cov=$(PKG) \
	  --cov-report=term-missing \
	  --cov-report=html:htmlcov \
	  -q --tb=short
	@echo "$(GREEN)Coverage report: htmlcov/index.html$(RESET)"

test-warn:  ## Тесты со строгой проверкой предупреждений (RuntimeWarning → error)
	$(PYTHON) -m pytest $(TESTS) \
	  -W error::RuntimeWarning \
	  -q --tb=short

# ── Инструменты ──────────────────────────────────────────────────────────────
profile:  ## Профилировать производительность пайплайна
	$(PYTHON) tools/profile.py --fragments 8 --iters 3

profile-full:  ## Полное профилирование с cProfile
	$(PYTHON) tools/profile.py --fragments 16 --cprofile --cprofile-out profile.stats --json profile.json

benchmark:  ## Запустить бенчмарк алгоритмов сборки
	$(PYTHON) tools/benchmark.py

server:  ## Запустить REST API сервер (порт $(PORT))
	$(PYTHON) tools/server.py --host 0.0.0.0 --port $(PORT)

# ── Сборка и Docker ──────────────────────────────────────────────────────────
build:  ## Собрать Python-пакет (wheel + sdist)
	$(PYTHON) -m build
	twine check dist/*

docker:  ## Собрать Docker-образ
	docker build -t puzzle-reconstruction:0.4.0 .

docker-up:  ## Запустить стек через docker-compose
	docker compose up --build -d

docker-down:  ## Остановить docker-compose стек
	docker compose down

# ── Очистка ──────────────────────────────────────────────────────────────────
clean:  ## Удалить временные файлы сборки
	rm -rf dist/ build/ *.egg-info/ htmlcov/ .coverage coverage.xml profile.stats __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
