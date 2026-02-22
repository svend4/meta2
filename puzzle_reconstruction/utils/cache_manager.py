"""Менеджер кэша промежуточных результатов пайплайна.

Модуль предоставляет LRU-кэш для хранения результатов вычислений
(дескрипторов, матриц оценок и т.д.), контроля размера кэша,
инвалидации записей, сериализации и восстановления состояния.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Hashable, Iterator, List, Optional, TypeVar

import numpy as np


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


# ─── CacheEntry ───────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    """Одна запись в кэше.

    Атрибуты:
        key:       Ключ записи.
        value:     Хранимое значение.
        created:   Время создания (unix timestamp).
        hits:      Число обращений.
        ttl:       Время жизни в секундах (None = бесконечно).
    """

    key: Any
    value: Any
    created: float = field(default_factory=time.time)
    hits: int = 0
    ttl: Optional[float] = None

    def is_expired(self) -> bool:
        """True если запись устарела (TTL истёк)."""
        if self.ttl is None:
            return False
        return (time.time() - self.created) > self.ttl

    def touch(self) -> None:
        """Зафиксировать обращение к записи."""
        self.hits += 1


# ─── CacheStats ───────────────────────────────────────────────────────────────

@dataclass
class CacheStats:
    """Статистика работы кэша.

    Атрибуты:
        hits:       Число попаданий.
        misses:     Число промахов.
        evictions:  Число вытесненных записей.
        size:       Текущее число записей.
        capacity:   Максимальная ёмкость (0 = неограничена).
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    capacity: int = 0

    @property
    def hit_rate(self) -> float:
        """Доля попаданий (float в [0, 1])."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total


# ─── LRUCache ─────────────────────────────────────────────────────────────────

class LRUCache:
    """LRU-кэш фиксированной ёмкости.

    Аргументы:
        capacity: Максимальное число записей (>= 1).
        default_ttl: Время жизни записей в секундах (None = бесконечно).

    Исключения:
        ValueError: Если capacity < 1.
    """

    def __init__(
        self,
        capacity: int = 128,
        default_ttl: Optional[float] = None,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity должен быть >= 1, получено {capacity}")
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._store: OrderedDict = OrderedDict()
        self._stats = CacheStats(capacity=capacity)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def stats(self) -> CacheStats:
        self._stats.size = len(self._store)
        return self._stats

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._store and not self._store[key].is_expired()

    def __iter__(self) -> Iterator:
        return iter(self._store)

    # ── Core operations ───────────────────────────────────────────────────────

    def get(self, key: Hashable, default: Any = None) -> Any:
        """Получить значение по ключу.

        Аргументы:
            key:     Ключ записи.
            default: Значение по умолчанию (если ключа нет).

        Возвращает:
            Хранимое значение или default.
        """
        entry = self._store.get(key)
        if entry is None or entry.is_expired():
            if entry is not None:
                # Удалить устаревшую запись
                del self._store[key]
            self._stats.misses += 1
            return default

        entry.touch()
        self._store.move_to_end(key)
        self._stats.hits += 1
        return entry.value

    def put(
        self,
        key: Hashable,
        value: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """Добавить или обновить запись в кэше.

        Аргументы:
            key:   Ключ записи.
            value: Хранимое значение.
            ttl:   Время жизни (None → default_ttl).
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl

        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = CacheEntry(key=key, value=value, ttl=effective_ttl)
            return

        if len(self._store) >= self._capacity:
            self._evict()

        self._store[key] = CacheEntry(key=key, value=value, ttl=effective_ttl)

    def delete(self, key: Hashable) -> bool:
        """Удалить запись по ключу.

        Возвращает:
            True если запись была удалена, False если ключ не найден.
        """
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        """Очистить весь кэш."""
        self._store.clear()

    def keys(self) -> List:
        """Список ключей (от старейшего к новейшему)."""
        return list(self._store.keys())

    def values(self) -> List:
        """Список значений (от старейшего к новейшему)."""
        return [e.value for e in self._store.values()]

    def purge_expired(self) -> int:
        """Удалить все устаревшие записи.

        Возвращает:
            Число удалённых записей.
        """
        expired = [k for k, e in self._store.items() if e.is_expired()]
        for k in expired:
            del self._store[k]
        return len(expired)

    def resize(self, new_capacity: int) -> None:
        """Изменить ёмкость кэша.

        Если новая ёмкость меньше текущего размера, лишние (старейшие)
        записи вытесняются.

        Аргументы:
            new_capacity: Новая ёмкость (>= 1).

        Исключения:
            ValueError: Если new_capacity < 1.
        """
        if new_capacity < 1:
            raise ValueError(
                f"new_capacity должен быть >= 1, получено {new_capacity}"
            )
        self._capacity = new_capacity
        self._stats.capacity = new_capacity
        while len(self._store) > self._capacity:
            self._evict()

    def to_dict(self) -> Dict:
        """Сериализовать кэш в словарь (только не-устаревшие записи).

        Значения, которые не поддерживают JSON-сериализацию (например,
        ndarray), конвертируются в списки.
        """
        result = {}
        for k, entry in self._store.items():
            if not entry.is_expired():
                v = entry.value
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                result[k] = {"value": v, "hits": entry.hits}
        return result

    # ── Private ───────────────────────────────────────────────────────────────

    def _evict(self) -> None:
        """Вытеснить наименее недавно используемую запись."""
        if self._store:
            self._store.popitem(last=False)
            self._stats.evictions += 1


# ─── Convenience functions ────────────────────────────────────────────────────

def make_cache(capacity: int = 128, ttl: Optional[float] = None) -> LRUCache:
    """Создать LRU-кэш с заданной ёмкостью.

    Аргументы:
        capacity: Максимальное число записей (>= 1).
        ttl:      Время жизни записей (None = бесконечно).

    Возвращает:
        LRUCache.
    """
    return LRUCache(capacity=capacity, default_ttl=ttl)


def cached_call(
    cache: LRUCache,
    key: Hashable,
    fn: Any,
    *args: Any,
    ttl: Optional[float] = None,
    **kwargs: Any,
) -> Any:
    """Вызвать fn(*args, **kwargs) с кэшированием результата.

    Если результат для key уже в кэше — возвращает кэшированное значение.
    Иначе вызывает fn и кэширует результат.

    Аргументы:
        cache: LRUCache.
        key:   Ключ кэша.
        fn:    Вызываемый объект.
        ttl:   Время жизни для этой записи.
        *args, **kwargs: Аргументы для fn.

    Возвращает:
        Результат fn или кэшированное значение.
    """
    _sentinel = object()
    result = cache.get(key, default=_sentinel)
    if result is not _sentinel:
        return result
    result = fn(*args, **kwargs)
    cache.put(key, result, ttl=ttl)
    return result


def merge_caches(caches: List[LRUCache], capacity: int = 256) -> LRUCache:
    """Объединить несколько кэшей в один.

    Записи из более поздних кэшей перезаписывают более ранние.

    Аргументы:
        caches:   Список LRUCache.
        capacity: Ёмкость итогового кэша (>= 1).

    Возвращает:
        Новый LRUCache с объединёнными записями.
    """
    merged = LRUCache(capacity=capacity)
    for c in caches:
        for k, entry in c._store.items():
            if not entry.is_expired():
                merged.put(k, entry.value)
    return merged
