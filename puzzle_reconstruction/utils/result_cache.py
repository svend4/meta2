"""Кэш промежуточных результатов с поддержкой TTL и явной инвалидации.

Модуль предоставляет простой in-memory кэш на основе словаря с временем
жизни (TTL) записей, счётчиком обращений и поддержкой именованных
пространств имён (namespace).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple


# ─── CachePolicy ──────────────────────────────────────────────────────────────

@dataclass
class CachePolicy:
    """Политика кэширования.

    Атрибуты:
        ttl:         Время жизни записи в секундах (> 0, или 0 — бесконечно).
        max_size:    Максимальное число записей (> 0, или 0 — без ограничений).
        namespace:   Пространство имён (непустое).
    """

    ttl: float = 0.0
    max_size: int = 0
    namespace: str = "default"

    def __post_init__(self) -> None:
        if self.ttl < 0.0:
            raise ValueError(
                f"ttl должен быть >= 0, получено {self.ttl}"
            )
        if self.max_size < 0:
            raise ValueError(
                f"max_size должен быть >= 0, получено {self.max_size}"
            )
        if not self.namespace:
            raise ValueError("namespace не должен быть пустым")


# ─── CacheRecord ──────────────────────────────────────────────────────────────

@dataclass
class CacheRecord:
    """Одна запись в кэше.

    Атрибуты:
        key:        Ключ записи (непустой).
        value:      Кэшированное значение.
        created_at: Время создания (Unix time, >= 0).
        ttl:        Время жизни (0 — бесконечно, > 0 — истекает).
        hits:       Число обращений (>= 0).
    """

    key: str
    value: Any
    created_at: float
    ttl: float = 0.0
    hits: int = 0

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("key не должен быть пустым")
        if self.created_at < 0.0:
            raise ValueError(
                f"created_at должен быть >= 0, получено {self.created_at}"
            )
        if self.ttl < 0.0:
            raise ValueError(
                f"ttl должен быть >= 0, получено {self.ttl}"
            )
        if self.hits < 0:
            raise ValueError(
                f"hits должен быть >= 0, получено {self.hits}"
            )

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Проверить, истёк ли TTL записи."""
        if self.ttl == 0.0:
            return False
        t = now if now is not None else time.time()
        return t > self.created_at + self.ttl


# ─── CacheSummary ─────────────────────────────────────────────────────────────

@dataclass
class CacheSummary:
    """Сводная статистика по кэшу.

    Атрибуты:
        namespace:   Пространство имён.
        n_entries:   Текущее число записей (>= 0).
        n_expired:   Число устаревших записей (>= 0).
        total_hits:  Суммарное число попаданий (>= 0).
    """

    namespace: str
    n_entries: int
    n_expired: int
    total_hits: int

    def __post_init__(self) -> None:
        if self.n_entries < 0:
            raise ValueError(
                f"n_entries должен быть >= 0, получено {self.n_entries}"
            )
        if self.n_expired < 0:
            raise ValueError(
                f"n_expired должен быть >= 0, получено {self.n_expired}"
            )
        if self.total_hits < 0:
            raise ValueError(
                f"total_hits должен быть >= 0, получено {self.total_hits}"
            )

    @property
    def hit_ratio(self) -> float:
        """Отношение попаданий к числу записей (0.0 если пусто)."""
        if self.n_entries == 0:
            return 0.0
        return float(self.total_hits) / float(self.n_entries)


# ─── ResultCache ──────────────────────────────────────────────────────────────

class ResultCache:
    """In-memory кэш результатов с TTL и ограничением размера.

    Аргументы:
        policy: Политика кэширования (None → CachePolicy()).
    """

    def __init__(self, policy: Optional[CachePolicy] = None) -> None:
        if policy is None:
            policy = CachePolicy()
        self._policy = policy
        self._store: Dict[str, CacheRecord] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def put(self, key: str, value: Any) -> None:
        """Добавить/обновить запись в кэше.

        Аргументы:
            key:   Ключ (непустой).
            value: Значение.

        Исключения:
            ValueError: Если key пустой.
        """
        if not key:
            raise ValueError("key не должен быть пустым")

        # Evict expired first
        self._evict_expired()

        # Enforce max_size (remove oldest entry by insertion order)
        if self._policy.max_size > 0:
            while len(self._store) >= self._policy.max_size:
                oldest = next(iter(self._store))
                del self._store[oldest]

        self._store[key] = CacheRecord(
            key=key,
            value=value,
            created_at=time.time(),
            ttl=self._policy.ttl,
        )

    def get(self, key: str) -> Optional[Any]:
        """Получить значение из кэша.

        Аргументы:
            key: Ключ.

        Возвращает:
            Значение или None (промах / устаревшая запись).
        """
        record = self._store.get(key)
        if record is None:
            return None
        if record.is_expired():
            del self._store[key]
            return None
        record.hits += 1
        return record.value

    def has(self, key: str) -> bool:
        """Проверить наличие актуальной записи."""
        record = self._store.get(key)
        if record is None:
            return False
        if record.is_expired():
            del self._store[key]
            return False
        return True

    def invalidate(self, key: str) -> bool:
        """Удалить запись по ключу.

        Возвращает:
            True если запись существовала.
        """
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> int:
        """Очистить кэш.

        Возвращает:
            Число удалённых записей.
        """
        n = len(self._store)
        self._store.clear()
        return n

    def keys(self) -> List[str]:
        """Список актуальных ключей (без истёкших)."""
        self._evict_expired()
        return list(self._store.keys())

    def size(self) -> int:
        """Число актуальных записей."""
        self._evict_expired()
        return len(self._store)

    def summarize(self) -> CacheSummary:
        """Сводная статистика по кэшу."""
        now = time.time()
        n_expired = sum(1 for r in self._store.values() if r.is_expired(now))
        total_hits = sum(r.hits for r in self._store.values())
        return CacheSummary(
            namespace=self._policy.namespace,
            n_entries=len(self._store),
            n_expired=n_expired,
            total_hits=total_hits,
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _evict_expired(self) -> int:
        """Удалить все истёкшие записи. Возвращает число удалённых."""
        now = time.time()
        expired = [k for k, r in self._store.items() if r.is_expired(now)]
        for k in expired:
            del self._store[k]
        return len(expired)


# ─── make_cache ───────────────────────────────────────────────────────────────

def make_cache(
    ttl: float = 0.0,
    max_size: int = 0,
    namespace: str = "default",
) -> ResultCache:
    """Создать ResultCache с заданными параметрами.

    Аргументы:
        ttl:       Время жизни записи (0 — бесконечно).
        max_size:  Лимит числа записей (0 — без ограничений).
        namespace: Пространство имён.

    Возвращает:
        Новый ResultCache.
    """
    return ResultCache(CachePolicy(ttl=ttl, max_size=max_size, namespace=namespace))


# ─── cached_result ────────────────────────────────────────────────────────────

def cached_result(
    cache: ResultCache,
    key: str,
    compute_fn,
) -> Any:
    """Получить значение из кэша или вычислить и сохранить.

    Аргументы:
        cache:      Кэш.
        key:        Ключ.
        compute_fn: Callable без аргументов, вычисляющий значение.

    Возвращает:
        Кэшированное или вновь вычисленное значение.
    """
    value = cache.get(key)
    if value is None:
        value = compute_fn()
        cache.put(key, value)
    return value


# ─── merge_caches ─────────────────────────────────────────────────────────────

def merge_caches(
    target: ResultCache,
    source: ResultCache,
) -> int:
    """Скопировать актуальные записи из source в target.

    Аргументы:
        target: Целевой кэш.
        source: Исходный кэш.

    Возвращает:
        Число скопированных записей.
    """
    count = 0
    for key in source.keys():
        value = source.get(key)
        if value is not None:
            target.put(key, value)
            count += 1
    return count


# ─── evict_expired ────────────────────────────────────────────────────────────

def evict_expired(cache: ResultCache) -> int:
    """Принудительно удалить все истёкшие записи из кэша.

    Аргументы:
        cache: Кэш.

    Возвращает:
        Число удалённых записей.
    """
    return cache._evict_expired()
