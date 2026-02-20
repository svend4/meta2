"""
Кэширование вычисленных дескрипторов фрагментов.

Вычисление фрактальных дескрипторов, CSS и IFS для одного фрагмента занимает
десятки миллисекунд. При повторных запусках (разные методы сборки, тюнинг
параметров, интерактивная работа) один и тот же фрагмент обрабатывается
несколько раз. Кэш устраняет избыточные вычисления.

Классы:
    DescriptorCache — LRU-кэш в оперативной памяти с настраиваемым размером
    DiskCache       — постоянный кэш на диске (shelve), переживает перезапуск

Функции:
    descriptor_key  — строит ключ (SHA256) по изображению и контуру фрагмента
    cached          — декоратор-кэш для функций вычисления дескрипторов

Использование:
    from puzzle_reconstruction.utils.cache import DescriptorCache, cached

    cache = DescriptorCache(max_size=256)

    @cached(cache)
    def compute_fractal(fragment: Fragment) -> FractalSignature:
        ...  # Дорогое вычисление

    sig = compute_fractal(fragment)   # Вычисляется один раз
    sig = compute_fractal(fragment)   # Берётся из кэша
"""
from __future__ import annotations

import functools
import hashlib
import logging
import shelve
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Generic, Optional, TypeVar

import numpy as np

from ..models import Fragment

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")


# ─── Ключи ────────────────────────────────────────────────────────────────────

def descriptor_key(fragment: Fragment,
                    include_image: bool = True,
                    precision: int = 3) -> str:
    """
    Строит строковый ключ (SHA-256, 16 символов) для фрагмента.

    Ключ зависит от:
        - fragment_id
        - Формы изображения (H, W)
        - Округлённого контура (точность *precision* знаков)
        - Хэша пикселей если include_image=True

    Args:
        fragment:      Объект Fragment.
        include_image: True → хешировать пиксели (надёжнее, медленнее).
        precision:     Число знаков после запятой для контура.

    Returns:
        16-символьный hex-дайджест.
    """
    h = hashlib.sha256()
    h.update(str(fragment.fragment_id).encode())
    h.update(str(fragment.image.shape).encode())

    # Контур (округлённый)
    contour_round = np.round(fragment.contour, decimals=precision)
    h.update(contour_round.tobytes())

    if include_image:
        # Быстрый хэш: только каждый 8-й пиксель (достаточно для отпечатка)
        sampled = fragment.image.ravel()[::8]
        h.update(sampled.tobytes())

    return h.hexdigest()[:16]


# ─── LRU-кэш в памяти ─────────────────────────────────────────────────────────

class DescriptorCache:
    """
    LRU-кэш фиксированного размера для дескрипторов фрагментов.

    Потокобезопасен (threading.Lock).

    Args:
        max_size: Максимальное число записей (по умолчанию 512).

    Пример:
        cache = DescriptorCache(max_size=256)
        cache.set("key1", my_descriptor)
        descriptor = cache.get("key1")   # None если отсутствует
    """

    def __init__(self, max_size: int = 512) -> None:
        self._max  = max(1, max_size)
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._hits  = 0
        self._misses = 0

    # ── Базовый интерфейс ─────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """Возвращает значение или None если ключ отсутствует."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._hits += 1
                return self._data[key]
            self._misses += 1
            return None

    def set(self, key: str, value: Any) -> None:
        """Сохраняет значение. При переполнении удаляет самую старую запись."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            if len(self._data) > self._max:
                self._data.popitem(last=False)

    def clear(self) -> None:
        """Очищает весь кэш."""
        with self._lock:
            self._data.clear()
            self._hits  = 0
            self._misses = 0

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    # ── Удобные методы ────────────────────────────────────────────────────

    def get_or_compute(self,
                        key:        str,
                        compute_fn: Callable[[], Any]) -> Any:
        """
        Возвращает кэшированное значение или вычисляет его.

        Args:
            key:        Строковый ключ.
            compute_fn: Callable без аргументов, возвращающий значение.

        Returns:
            Значение из кэша или результат compute_fn().
        """
        cached_val = self.get(key)
        if cached_val is not None:
            return cached_val
        value = compute_fn()
        self.set(key, value)
        return value

    def get_for_fragment(self,
                          fragment:   Fragment,
                          compute_fn: Callable[[Fragment], Any],
                          **key_kwargs) -> Any:
        """
        Кэшированное вычисление для объекта Fragment.

        Args:
            fragment:   Фрагмент.
            compute_fn: Функция fragment → результат.
            **key_kwargs: Дополнительные аргументы для descriptor_key.

        Returns:
            Кэшированный или вновь вычисленный дескриптор.
        """
        key = descriptor_key(fragment, **key_kwargs)
        return self.get_or_compute(key, lambda: compute_fn(fragment))

    # ── Статистика ────────────────────────────────────────────────────────

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        with self._lock:
            return {
                "size":    len(self._data),
                "max":     self._max,
                "hits":    self._hits,
                "misses":  self._misses,
                "hit_rate": self.hit_rate,
            }

    def __repr__(self) -> str:
        s = self.stats()
        return (f"DescriptorCache(size={s['size']}/{s['max']}, "
                f"hit_rate={s['hit_rate']:.1%})")


# ─── Кэш на диске ─────────────────────────────────────────────────────────────

class DiskCache:
    """
    Постоянный кэш дескрипторов на диске (через shelve / dbm).

    Переживает перезапуск Python. Полезен при повторных запусках pipeline
    на одном наборе данных.

    Args:
        path: Путь к файлу базы данных (без расширения, shelve добавит .db).

    Пример:
        cache = DiskCache("./cache/descriptors")
        cache["key1"] = my_descriptor
        descriptor = cache["key1"]
        cache.close()
    """

    def __init__(self, path: str) -> None:
        self._path = str(path)
        self._lock = threading.Lock()
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._db   = shelve.open(self._path, flag="c", writeback=False)

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._db[key]

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            self._db[key] = value
            self._db.sync()

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._db

    def __len__(self) -> int:
        with self._lock:
            return len(self._db)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def get_or_compute(self,
                        key:        str,
                        compute_fn: Callable[[], Any]) -> Any:
        if key in self:
            return self[key]
        value = compute_fn()
        self[key] = value
        return value

    def clear(self) -> None:
        with self._lock:
            self._db.clear()
            self._db.sync()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> "DiskCache":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"DiskCache(path={self._path!r}, size={len(self)})"


# ─── Декоратор @cached ────────────────────────────────────────────────────────

def cached(cache: DescriptorCache,
            key_fn: Optional[Callable[..., str]] = None):
    """
    Декоратор: кэширует результат функции в переданный DescriptorCache.

    Ключ вычисляется из первого аргумента-Fragment (если key_fn не задан).
    Для остальных случаев нужно передать key_fn явно.

    Args:
        cache:  Экземпляр DescriptorCache.
        key_fn: Функция (*args, **kwargs) → str. None → auto от первого Fragment.

    Пример:
        cache = DescriptorCache(512)

        @cached(cache)
        def compute_fractal(fragment: Fragment) -> FractalSignature:
            ...

        sig = compute_fractal(frag)   # Вычислится единожды
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Строим ключ
            if key_fn is not None:
                key = key_fn(*args, **kwargs)
            elif args and isinstance(args[0], Fragment):
                key = descriptor_key(args[0])
            else:
                # Fallback: repr первого аргумента
                key = hashlib.sha256(
                    repr(args[0] if args else kwargs).encode()
                ).hexdigest()[:16]

            hit = cache.get(key)
            if hit is not None:
                return hit

            result = fn(*args, **kwargs)
            cache.set(key, result)
            return result

        # Прикрепляем кэш к функции для интроспекции
        wrapper.cache = cache   # type: ignore[attr-defined]
        wrapper.clear_cache = cache.clear  # type: ignore[attr-defined]
        return wrapper

    return decorator


# ─── Глобальный кэш по умолчанию ─────────────────────────────────────────────

_default_cache: Optional[DescriptorCache] = None


def get_default_cache(max_size: int = 1024) -> DescriptorCache:
    """
    Возвращает (или создаёт) глобальный кэш дескрипторов.

    Позволяет использовать @cached(get_default_cache()) без явного создания.

    Args:
        max_size: Размер кэша при первом вызове (игнорируется после инициализации).

    Returns:
        Синглтон DescriptorCache.
    """
    global _default_cache
    if _default_cache is None:
        _default_cache = DescriptorCache(max_size=max_size)
    return _default_cache


def clear_default_cache() -> None:
    """Очищает глобальный кэш."""
    if _default_cache is not None:
        _default_cache.clear()
