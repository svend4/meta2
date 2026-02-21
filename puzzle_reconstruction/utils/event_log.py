"""Журнал событий пайплайна восстановления пазла.

Модуль предоставляет структуры и функции для записи, фильтрации
и экспорта событий пайплайна: информационных сообщений, предупреждений,
ошибок и отладочной информации.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─── константы уровней ────────────────────────────────────────────────────────

DEBUG = "debug"
INFO = "info"
WARNING = "warning"
ERROR = "error"

_LEVELS = (DEBUG, INFO, WARNING, ERROR)
_LEVEL_ORDER = {lvl: i for i, lvl in enumerate(_LEVELS)}


# ─── EventLogConfig ───────────────────────────────────────────────────────────

@dataclass
class EventLogConfig:
    """Параметры журнала событий.

    Атрибуты:
        max_events:     Максимальное число хранимых событий (0 = без ограничений; >= 0).
        default_level:  Уровень по умолчанию ('debug'|'info'|'warning'|'error').
        namespace:      Пространство имён журнала.
    """

    max_events: int = 0
    default_level: str = INFO
    namespace: str = "default"

    def __post_init__(self) -> None:
        if self.max_events < 0:
            raise ValueError(
                f"max_events должен быть >= 0, получено {self.max_events}"
            )
        if self.default_level not in _LEVELS:
            raise ValueError(
                f"default_level должен быть одним из {_LEVELS}, "
                f"получено '{self.default_level}'"
            )
        if not self.namespace:
            raise ValueError("namespace не должен быть пустым")


# ─── EventRecord ──────────────────────────────────────────────────────────────

@dataclass
class EventRecord:
    """Одна запись журнала событий.

    Атрибуты:
        event_id:  Порядковый номер (>= 0).
        name:      Название события.
        level:     Уровень ('debug'|'info'|'warning'|'error').
        timestamp: Unix-время создания (>= 0).
        meta:      Дополнительные данные.
    """

    event_id: int
    name: str
    level: str
    timestamp: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_id < 0:
            raise ValueError(
                f"event_id должен быть >= 0, получено {self.event_id}"
            )
        if not self.name:
            raise ValueError("name не должен быть пустым")
        if self.level not in _LEVELS:
            raise ValueError(
                f"level должен быть одним из {_LEVELS}, получено '{self.level}'"
            )
        if self.timestamp < 0.0:
            raise ValueError(
                f"timestamp должен быть >= 0, получено {self.timestamp}"
            )

    @property
    def is_error(self) -> bool:
        """True если уровень == 'error'."""
        return self.level == ERROR

    @property
    def level_order(self) -> int:
        """Числовой порядок уровня (0=debug, 1=info, 2=warning, 3=error)."""
        return _LEVEL_ORDER[self.level]


# ─── EventSummary ─────────────────────────────────────────────────────────────

@dataclass
class EventSummary:
    """Сводка по журналу событий.

    Атрибуты:
        total:    Общее число событий (>= 0).
        n_debug:  Число debug-событий (>= 0).
        n_info:   Число info-событий (>= 0).
        n_warn:   Число warning-событий (>= 0).
        n_error:  Число error-событий (>= 0).
        names:    Уникальные имена событий.
    """

    total: int
    n_debug: int
    n_info: int
    n_warn: int
    n_error: int
    names: List[str]

    def __post_init__(self) -> None:
        for attr in ("total", "n_debug", "n_info", "n_warn", "n_error"):
            val = getattr(self, attr)
            if val < 0:
                raise ValueError(f"{attr} должен быть >= 0, получено {val}")

    @property
    def has_errors(self) -> bool:
        """True если есть хотя бы одно error-событие."""
        return self.n_error > 0

    @property
    def error_ratio(self) -> float:
        """Доля error-событий (0 если total == 0)."""
        if self.total == 0:
            return 0.0
        return float(self.n_error) / float(self.total)


# ─── EventLog ─────────────────────────────────────────────────────────────────

class EventLog:
    """Журнал событий пайплайна.

    Аргументы:
        cfg: Параметры журнала (None → EventLogConfig()).
    """

    def __init__(self, cfg: Optional[EventLogConfig] = None) -> None:
        if cfg is None:
            cfg = EventLogConfig()
        self._cfg = cfg
        self._records: List[EventRecord] = []
        self._counter: int = 0

    # ── запись ────────────────────────────────────────────────────────────────

    def log(
        self,
        name: str,
        level: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> EventRecord:
        """Записать событие в журнал.

        Аргументы:
            name:      Название события.
            level:     Уровень (None → default_level).
            meta:      Метаданные события.
            timestamp: Время (None → текущее время).

        Возвращает:
            Созданный EventRecord.

        Исключения:
            ValueError: Если name пуст.
        """
        if not name:
            raise ValueError("name не должен быть пустым")
        lvl = level if level is not None else self._cfg.default_level
        ts = timestamp if timestamp is not None else time.time()
        rec = EventRecord(
            event_id=self._counter,
            name=name,
            level=lvl,
            timestamp=ts,
            meta=meta or {},
        )
        self._records.append(rec)
        self._counter += 1
        if self._cfg.max_events > 0 and len(self._records) > self._cfg.max_events:
            self._records = self._records[-self._cfg.max_events:]
        return rec

    # ── чтение ────────────────────────────────────────────────────────────────

    def filter_by_level(self, level: str) -> List[EventRecord]:
        """Вернуть события с уровнем >= level.

        Аргументы:
            level: Минимальный уровень ('debug'|'info'|'warning'|'error').

        Исключения:
            ValueError: Если level не является допустимым значением.
        """
        if level not in _LEVELS:
            raise ValueError(
                f"level должен быть одним из {_LEVELS}, получено '{level}'"
            )
        min_order = _LEVEL_ORDER[level]
        return [r for r in self._records if r.level_order >= min_order]

    def filter_by_name(self, name: str) -> List[EventRecord]:
        """Вернуть события с заданным именем."""
        return [r for r in self._records if r.name == name]

    def since(self, timestamp: float) -> List[EventRecord]:
        """Вернуть события с timestamp >= заданного.

        Аргументы:
            timestamp: Unix-время (>= 0).

        Исключения:
            ValueError: Если timestamp < 0.
        """
        if timestamp < 0.0:
            raise ValueError(
                f"timestamp должен быть >= 0, получено {timestamp}"
            )
        return [r for r in self._records if r.timestamp >= timestamp]

    def to_list(self) -> List[EventRecord]:
        """Вернуть все события в хронологическом порядке."""
        return list(self._records)

    # ── статистика и управление ───────────────────────────────────────────────

    def size(self) -> int:
        """Число хранимых событий."""
        return len(self._records)

    def clear(self) -> int:
        """Очистить журнал; вернуть число удалённых записей."""
        n = len(self._records)
        self._records.clear()
        return n

    @property
    def namespace(self) -> str:
        """Пространство имён журнала."""
        return self._cfg.namespace

    @property
    def config(self) -> EventLogConfig:
        """Конфигурация журнала."""
        return self._cfg


# ─── make_event_log ───────────────────────────────────────────────────────────

def make_event_log(
    max_events: int = 0,
    default_level: str = INFO,
    namespace: str = "default",
) -> EventLog:
    """Создать EventLog с заданными параметрами.

    Аргументы:
        max_events:    Максимальное число событий (0 = без ограничений).
        default_level: Уровень по умолчанию.
        namespace:     Пространство имён.

    Возвращает:
        EventLog.
    """
    return EventLog(EventLogConfig(
        max_events=max_events,
        default_level=default_level,
        namespace=namespace,
    ))


# ─── log_event ────────────────────────────────────────────────────────────────

def log_event(
    log: EventLog,
    name: str,
    level: str = INFO,
    meta: Optional[Dict[str, Any]] = None,
) -> EventRecord:
    """Добавить событие в журнал (сокращённый интерфейс).

    Аргументы:
        log:   EventLog.
        name:  Название события.
        level: Уровень.
        meta:  Метаданные.

    Возвращает:
        EventRecord.
    """
    return log.log(name, level=level, meta=meta)


# ─── filter_events ────────────────────────────────────────────────────────────

def filter_events(
    records: List[EventRecord],
    level: Optional[str] = None,
    name: Optional[str] = None,
) -> List[EventRecord]:
    """Фильтровать список EventRecord по уровню и/или имени.

    Аргументы:
        records: Список событий.
        level:   Минимальный уровень (None = без фильтрации).
        name:    Имя события (None = без фильтрации).

    Возвращает:
        Отфильтрованный список.
    """
    result = records
    if level is not None:
        if level not in _LEVELS:
            raise ValueError(
                f"level должен быть одним из {_LEVELS}, получено '{level}'"
            )
        min_order = _LEVEL_ORDER[level]
        result = [r for r in result if r.level_order >= min_order]
    if name is not None:
        result = [r for r in result if r.name == name]
    return result


# ─── summarize_events ─────────────────────────────────────────────────────────

def summarize_events(records: List[EventRecord]) -> EventSummary:
    """Построить сводку по списку событий.

    Аргументы:
        records: Список EventRecord.

    Возвращает:
        EventSummary.
    """
    n_debug = sum(1 for r in records if r.level == DEBUG)
    n_info = sum(1 for r in records if r.level == INFO)
    n_warn = sum(1 for r in records if r.level == WARNING)
    n_error = sum(1 for r in records if r.level == ERROR)
    names = sorted({r.name for r in records})
    return EventSummary(
        total=len(records),
        n_debug=n_debug,
        n_info=n_info,
        n_warn=n_warn,
        n_error=n_error,
        names=names,
    )


# ─── merge_event_logs ─────────────────────────────────────────────────────────

def merge_event_logs(
    target: EventLog,
    source: EventLog,
) -> int:
    """Добавить все события из source в target.

    Аргументы:
        target: Целевой журнал.
        source: Источник событий.

    Возвращает:
        Число добавленных событий.
    """
    added = 0
    for rec in source.to_list():
        target.log(rec.name, level=rec.level,
                   meta=dict(rec.meta), timestamp=rec.timestamp)
        added += 1
    return added


# ─── export_event_log ─────────────────────────────────────────────────────────

def export_event_log(
    log: EventLog,
) -> List[Dict[str, Any]]:
    """Экспортировать журнал в список словарей.

    Аргументы:
        log: EventLog.

    Возвращает:
        Список {'event_id', 'name', 'level', 'timestamp', 'meta'}.
    """
    return [
        {
            "event_id": r.event_id,
            "name": r.name,
            "level": r.level,
            "timestamp": r.timestamp,
            "meta": dict(r.meta),
        }
        for r in log.to_list()
    ]
