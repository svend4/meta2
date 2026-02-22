"""Лёгкая шина событий (pub/sub) для пайплайна восстановления.

Модуль предоставляет простой механизм публикации и подписки на события,
позволяя компонентам пайплайна слабо связываться между собой.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ─── BusConfig ────────────────────────────────────────────────────────────────

@dataclass
class BusConfig:
    """Параметры шины событий.

    Атрибуты:
        max_history:    Максимальное число событий в истории (>= 0, 0 = без лимита).
        raise_on_error: Выбрасывать исключение при ошибке обработчика.
        allow_wildcard: Разрешить подписку на все события через "*".
    """

    max_history: int = 0
    raise_on_error: bool = False
    allow_wildcard: bool = True

    def __post_init__(self) -> None:
        if self.max_history < 0:
            raise ValueError(
                f"max_history должен быть >= 0, получено {self.max_history}"
            )


# ─── EventRecord ──────────────────────────────────────────────────────────────

@dataclass
class EventRecord:
    """Запись об одном событии в шине.

    Атрибуты:
        topic:    Имя топика (непустая строка).
        payload:  Данные события (произвольный объект).
        seq:      Порядковый номер события (>= 0).
        n_handlers: Число вызванных обработчиков.
    """

    topic: str
    payload: Any
    seq: int
    n_handlers: int = 0

    def __post_init__(self) -> None:
        if not self.topic:
            raise ValueError("topic не должен быть пустой строкой")
        if self.seq < 0:
            raise ValueError(
                f"seq должен быть >= 0, получено {self.seq}"
            )

    @property
    def has_handlers(self) -> bool:
        """True если событие было обработано хотя бы одним обработчиком."""
        return self.n_handlers > 0


# ─── BusSummary ───────────────────────────────────────────────────────────────

@dataclass
class BusSummary:
    """Сводка по работе шины событий.

    Атрибуты:
        total_published: Общее число опубликованных событий (>= 0).
        total_handled:   Общее число успешных обработок (>= 0).
        total_errors:    Общее число ошибок обработчиков (>= 0).
        n_topics:        Число уникальных топиков (>= 0).
        n_subscriptions: Общее число активных подписок (>= 0).
    """

    total_published: int
    total_handled: int
    total_errors: int
    n_topics: int
    n_subscriptions: int

    def __post_init__(self) -> None:
        for name, val in (
            ("total_published", self.total_published),
            ("total_handled", self.total_handled),
            ("total_errors", self.total_errors),
            ("n_topics", self.n_topics),
            ("n_subscriptions", self.n_subscriptions),
        ):
            if val < 0:
                raise ValueError(f"{name} должен быть >= 0, получено {val}")

    @property
    def error_ratio(self) -> float:
        """Доля обработок, завершившихся ошибкой (0 если нет обработок)."""
        total = self.total_handled + self.total_errors
        if total == 0:
            return 0.0
        return float(self.total_errors) / float(total)


# ─── EventBus ─────────────────────────────────────────────────────────────────

class EventBus:
    """Шина событий с подпиской, публикацией и историей.

    Использование::
        bus = EventBus()
        sub_id = bus.subscribe("topic", handler)
        bus.publish("topic", payload)
        bus.unsubscribe(sub_id)
    """

    def __init__(self, cfg: Optional[BusConfig] = None) -> None:
        self._cfg = cfg or BusConfig()
        self._handlers: Dict[str, Dict[str, Callable[[EventRecord], None]]] = {}
        self._history: List[EventRecord] = []
        self._seq = 0
        self._total_handled = 0
        self._total_errors = 0
        self._sub_counter = 0
        # subscription_id -> topic
        self._sub_topics: Dict[str, str] = {}

    # ── публичный API ─────────────────────────────────────────────────────────

    def subscribe(
        self,
        topic: str,
        handler: Callable[[EventRecord], None],
    ) -> str:
        """Подписать обработчик на топик.

        Аргументы:
            topic:   Имя топика или "*" (если allow_wildcard).
            handler: Функция, принимающая EventRecord.

        Возвращает:
            Уникальный subscription_id.

        Исключения:
            ValueError: Если топик — "*" при allow_wildcard=False.
        """
        if not topic:
            raise ValueError("topic не должен быть пустой строкой")
        if topic == "*" and not self._cfg.allow_wildcard:
            raise ValueError("wildcard подписка отключена в конфигурации")

        self._sub_counter += 1
        sub_id = f"sub_{self._sub_counter}"

        if topic not in self._handlers:
            self._handlers[topic] = {}
        self._handlers[topic][sub_id] = handler
        self._sub_topics[sub_id] = topic
        return sub_id

    def unsubscribe(self, sub_id: str) -> bool:
        """Отменить подписку по subscription_id.

        Аргументы:
            sub_id: Идентификатор подписки.

        Возвращает:
            True если подписка была найдена и удалена, иначе False.
        """
        topic = self._sub_topics.pop(sub_id, None)
        if topic is None:
            return False
        bucket = self._handlers.get(topic, {})
        bucket.pop(sub_id, None)
        return True

    def publish(self, topic: str, payload: Any = None) -> EventRecord:
        """Опубликовать событие на топик.

        Аргументы:
            topic:   Имя топика (непустая строка).
            payload: Данные события.

        Возвращает:
            EventRecord с числом вызванных обработчиков.

        Исключения:
            ValueError:   Если topic пустая строка.
            RuntimeError: Если обработчик бросил исключение
                          и raise_on_error=True.
        """
        if not topic:
            raise ValueError("topic не должен быть пустой строкой")

        record = EventRecord(topic=topic, payload=payload, seq=self._seq)
        self._seq += 1

        n_called = 0
        for handler_topic in (topic, "*"):
            if handler_topic == "*" and not self._cfg.allow_wildcard:
                continue
            for sub_id, handler in list(self._handlers.get(handler_topic, {}).items()):
                try:
                    handler(record)
                    n_called += 1
                    self._total_handled += 1
                except Exception as exc:
                    self._total_errors += 1
                    if self._cfg.raise_on_error:
                        raise RuntimeError(
                            f"Ошибка в обработчике {sub_id}: {exc}"
                        ) from exc

        record.n_handlers = n_called

        self._history.append(record)
        if self._cfg.max_history > 0:
            while len(self._history) > self._cfg.max_history:
                self._history.pop(0)

        return record

    def history(self, topic: Optional[str] = None) -> List[EventRecord]:
        """Вернуть историю событий.

        Аргументы:
            topic: Фильтр по топику (None = все события).

        Возвращает:
            Список EventRecord в порядке публикации.
        """
        if topic is None:
            return list(self._history)
        return [r for r in self._history if r.topic == topic]

    def clear_history(self) -> None:
        """Очистить историю событий."""
        self._history.clear()

    def topics(self) -> List[str]:
        """Вернуть список топиков с активными подписками."""
        return [t for t, subs in self._handlers.items() if subs]

    def summary(self) -> BusSummary:
        """Вернуть сводку по работе шины."""
        n_subs = sum(len(subs) for subs in self._handlers.values())
        return BusSummary(
            total_published=self._seq,
            total_handled=self._total_handled,
            total_errors=self._total_errors,
            n_topics=len(self.topics()),
            n_subscriptions=n_subs,
        )


# ─── make_event_bus ───────────────────────────────────────────────────────────

def make_event_bus(
    max_history: int = 0,
    raise_on_error: bool = False,
    allow_wildcard: bool = True,
) -> EventBus:
    """Создать EventBus с заданными параметрами.

    Аргументы:
        max_history:    Максимальный размер истории.
        raise_on_error: Выбрасывать исключение при ошибке обработчика.
        allow_wildcard: Разрешить "*" подписку.

    Возвращает:
        EventBus.
    """
    return EventBus(BusConfig(
        max_history=max_history,
        raise_on_error=raise_on_error,
        allow_wildcard=allow_wildcard,
    ))


# ─── collect_events ───────────────────────────────────────────────────────────

def collect_events(
    bus: EventBus,
    topic: str,
) -> List[EventRecord]:
    """Вернуть все события топика из истории шины.

    Аргументы:
        bus:   EventBus.
        topic: Имя топика.

    Возвращает:
        Список EventRecord для данного топика.
    """
    return bus.history(topic=topic)


# ─── drain_bus ────────────────────────────────────────────────────────────────

def drain_bus(bus: EventBus) -> List[EventRecord]:
    """Извлечь всю историю событий и очистить её.

    Аргументы:
        bus: EventBus.

    Возвращает:
        Список всех EventRecord.
    """
    events = bus.history()
    bus.clear_history()
    return events
