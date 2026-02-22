"""
Структурированное логирование для пайплайна восстановления пазлов.

Поддерживает:
  - Консольный вывод с цветами (уровни: DEBUG / INFO / WARNING / ERROR)
  - Запись в файл (опционально)
  - Таймер для измерения времени этапов
  - Прогресс-бар для длительных операций
"""
import sys
import time
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


# ANSI-цвета (отключаются автоматически при перенаправлении в файл)
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREY   = "\033[90m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_BLUE   = "\033[94m"


class ColorFormatter(logging.Formatter):
    """Форматтер с цветовой маркировкой уровней."""

    LEVEL_COLORS = {
        logging.DEBUG:   _GREY,
        logging.INFO:    _CYAN,
        logging.WARNING: _YELLOW,
        logging.ERROR:   _RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        use_color = _supports_color()
        color = self.LEVEL_COLORS.get(record.levelno, "")
        reset = _RESET if use_color else ""
        bold  = _BOLD  if use_color else ""

        level = record.levelname[:4]
        ts    = time.strftime("%H:%M:%S")

        if use_color:
            prefix = f"{_GREY}{ts}{_RESET} {color}{bold}{level}{reset}"
        else:
            prefix = f"{ts} {level}"

        return f"{prefix}  {record.getMessage()}"


def get_logger(name: str = "puzzle",
               level: int = logging.INFO,
               log_file: Optional[str] = None) -> logging.Logger:
    """
    Возвращает настроенный логгер.

    Args:
        name:     Имя логгера.
        level:    Уровень логирования (logging.DEBUG / INFO / WARNING / ERROR).
        log_file: Путь к файлу лога (если None — только консоль).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Уже настроен

    logger.setLevel(level)

    # Консольный хендлер
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(ColorFormatter())
    logger.addHandler(console)

    # Файловый хендлер (опционально)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s  %(message)s"))
        logger.addHandler(fh)

    logger.propagate = False
    return logger


# Глобальный логгер по умолчанию
log = get_logger("puzzle")


@contextmanager
def stage(name: str, logger: logging.Logger = None):
    """
    Контекстный менеджер — обёртка для именованного этапа пайплайна.
    Выводит имя этапа и время выполнения.

    Использование:
        with stage("Сегментация"):
            ... код этапа ...
    """
    _log = logger or log
    use_color = _supports_color()
    bold  = _BOLD  if use_color else ""
    green = _GREEN if use_color else ""
    reset = _RESET if use_color else ""

    _log.info(f"{bold}▶ {name}{reset}")
    t0 = time.perf_counter()
    try:
        yield
        elapsed = time.perf_counter() - t0
        _log.info(f"{green}✓ {name}  ({elapsed:.2f}с){reset}")
    except Exception as e:
        elapsed = time.perf_counter() - t0
        _log.error(f"✗ {name}  ({elapsed:.2f}с)  — {e}")
        raise


class ProgressBar:
    """
    Простой текстовый прогресс-бар для консольного вывода.

    Использование:
        with ProgressBar("Обработка", total=100) as pb:
            for i in range(100):
                pb.update(i + 1)
    """

    def __init__(self, label: str, total: int, width: int = 30,
                 logger: logging.Logger = None):
        self.label  = label
        self.total  = total
        self.width  = width
        self._log   = logger or log
        self._t0    = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self._t0
        self._print(self.total, elapsed, done=True)
        print()  # Перенос строки после бара

    def update(self, current: int) -> None:
        elapsed = time.perf_counter() - (self._t0 or time.perf_counter())
        self._print(current, elapsed, done=False)

    def _print(self, current: int, elapsed: float, done: bool) -> None:
        frac  = min(1.0, current / max(1, self.total))
        filled = int(self.width * frac)
        bar   = "█" * filled + "░" * (self.width - filled)
        pct   = f"{frac:.0%}"
        end   = "\n" if done else "\r"
        if _supports_color():
            color = _GREEN if done else _BLUE
            reset = _RESET
        else:
            color = reset = ""
        print(f"  {color}{bar}{reset} {pct:>4}  {self.label}  [{elapsed:.1f}с]",
              end=end, flush=True)


class PipelineTimer:
    """
    Накапливает время выполнения каждого этапа пайплайна.
    Используется для профилирования.
    """

    def __init__(self):
        self._stages: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._stages[name] = time.perf_counter() - t0

    def report(self) -> str:
        if not self._stages:
            return "(нет данных)"
        total = sum(self._stages.values())
        lines = ["Время выполнения:"]
        for name, t in self._stages.items():
            bar = "█" * int(20 * t / total)
            lines.append(f"  {name:<28} {t:5.2f}с  {bar}")
        lines.append(f"  {'ИТОГО':<28} {total:5.2f}с")
        return "\n".join(lines)


# ─── Утилиты ──────────────────────────────────────────────────────────────

def _supports_color() -> bool:
    """Проверяет, поддерживает ли терминал ANSI-цвета."""
    return (
        hasattr(sys.stdout, "isatty") and
        sys.stdout.isatty() and
        sys.platform != "win32"
    )
