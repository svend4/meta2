"""Управление конфигурацией модулей пазловой реконструкции.

Модуль предоставляет инструменты для описания схем конфигурации,
их валидации, слияния нескольких источников настроек и создания
фиксированных (frozen) снимков текущего состояния.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ─── ConfigField ──────────────────────────────────────────────────────────────

@dataclass
class ConfigField:
    """Описание одного поля конфигурации.

    Атрибуты:
        name:        Имя поля (непустое).
        default:     Значение по умолчанию (None → поле обязательное если required=True).
        required:    Обязательное поле (True / False).
        type_name:   Строковое имя ожидаемого типа ('int', 'float', 'str', 'bool', 'any').
        description: Описание назначения поля.
    """

    name: str
    default: Any = None
    required: bool = False
    type_name: str = "any"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name не должно быть пустым")
        if self.type_name not in {"int", "float", "str", "bool", "any"}:
            raise ValueError(
                f"type_name должен быть одним из "
                f"{{'int','float','str','bool','any'}}, получено '{self.type_name}'"
            )


# ─── ConfigSpec ───────────────────────────────────────────────────────────────

@dataclass
class ConfigSpec:
    """Схема конфигурации — набор описанных полей.

    Атрибуты:
        name:   Имя схемы (непустое).
        fields: Список ConfigField.
    """

    name: str
    fields: List[ConfigField] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name не должно быть пустым")

    @property
    def required_fields(self) -> List[ConfigField]:
        """Только обязательные поля."""
        return [f for f in self.fields if f.required]

    @property
    def optional_fields(self) -> List[ConfigField]:
        """Только необязательные поля."""
        return [f for f in self.fields if not f.required]

    def field_names(self) -> List[str]:
        """Список имён всех полей."""
        return [f.name for f in self.fields]


# ─── ConfigSnapshot ───────────────────────────────────────────────────────────

@dataclass
class ConfigSnapshot:
    """Фиксированный снимок конфигурации.

    Атрибуты:
        name:      Имя источника конфигурации (непустое).
        data:      Словарь значений.
        timestamp: Время создания снимка (Unix time, >= 0).
    """

    name: str
    data: Dict[str, Any]
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name не должно быть пустым")
        if self.timestamp < 0.0:
            raise ValueError(
                f"timestamp должен быть >= 0, получено {self.timestamp}"
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Получить значение поля."""
        return self.data.get(key, default)

    def has(self, key: str) -> bool:
        """Проверить наличие поля."""
        return key in self.data


# ─── validate_field_type ──────────────────────────────────────────────────────

def validate_field_type(value: Any, type_name: str) -> bool:
    """Проверить соответствие значения ожидаемому типу.

    Аргументы:
        value:     Проверяемое значение.
        type_name: Ожидаемый тип ('int', 'float', 'str', 'bool', 'any').

    Возвращает:
        True если тип совпадает.

    Исключения:
        ValueError: Если type_name неизвестен.
    """
    if type_name == "any":
        return True
    type_map = {"int": int, "float": (int, float), "str": str, "bool": bool}
    if type_name not in type_map:
        raise ValueError(f"Неизвестный type_name: '{type_name}'")
    return isinstance(value, type_map[type_name])


# ─── validate_config ──────────────────────────────────────────────────────────

def validate_config(
    data: Dict[str, Any],
    spec: ConfigSpec,
) -> List[str]:
    """Валидировать словарь конфигурации по схеме.

    Аргументы:
        data: Словарь значений.
        spec: Схема конфигурации.

    Возвращает:
        Список строк с ошибками (пустой — если всё в порядке).
    """
    errors: List[str] = []
    for cf in spec.required_fields:
        if cf.name not in data:
            errors.append(f"Отсутствует обязательное поле '{cf.name}'")
        elif not validate_field_type(data[cf.name], cf.type_name):
            errors.append(
                f"Поле '{cf.name}': ожидается тип '{cf.type_name}', "
                f"получено {type(data[cf.name]).__name__}"
            )
    for cf in spec.optional_fields:
        if cf.name in data and not validate_field_type(data[cf.name], cf.type_name):
            errors.append(
                f"Поле '{cf.name}': ожидается тип '{cf.type_name}', "
                f"получено {type(data[cf.name]).__name__}"
            )
    return errors


# ─── load_config ──────────────────────────────────────────────────────────────

def load_config(
    data: Dict[str, Any],
    spec: ConfigSpec,
) -> Dict[str, Any]:
    """Загрузить конфигурацию из словаря, подставив значения по умолчанию.

    Аргументы:
        data: Входной словарь (может быть неполным).
        spec: Схема конфигурации.

    Возвращает:
        Словарь с заполненными полями (дефолты для отсутствующих).

    Исключения:
        ValueError: Если отсутствует обязательное поле.
    """
    result: Dict[str, Any] = {}
    for cf in spec.fields:
        if cf.name in data:
            result[cf.name] = data[cf.name]
        elif cf.required:
            raise ValueError(f"Отсутствует обязательное поле '{cf.name}'")
        else:
            result[cf.name] = cf.default
    return result


# ─── merge_configs ────────────────────────────────────────────────────────────

def merge_configs(
    configs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Слить несколько словарей конфигурации (более поздние перекрывают ранние).

    Аргументы:
        configs: Список словарей (от базового к более специфичному).

    Возвращает:
        Объединённый словарь.
    """
    result: Dict[str, Any] = {}
    for cfg in configs:
        result.update(cfg)
    return result


# ─── diff_configs ─────────────────────────────────────────────────────────────

def diff_configs(
    base: Dict[str, Any],
    other: Dict[str, Any],
) -> Dict[str, Tuple[Any, Any]]:
    """Вычислить разницу между двумя конфигурациями.

    Аргументы:
        base:  Базовая конфигурация.
        other: Сравниваемая конфигурация.

    Возвращает:
        Словарь {key: (base_value, other_value)} для отличающихся ключей.
        Ключи присутствующие только в одной из конфигураций включаются
        со значением None для отсутствующей стороны.
    """
    diff: Dict[str, Tuple[Any, Any]] = {}
    all_keys = set(base.keys()) | set(other.keys())
    for key in all_keys:
        bv = base.get(key)
        ov = other.get(key)
        if bv != ov:
            diff[key] = (bv, ov)
    return diff


# ─── make_config_snapshot ─────────────────────────────────────────────────────

def make_config_snapshot(
    name: str,
    data: Dict[str, Any],
) -> ConfigSnapshot:
    """Создать снимок конфигурации с текущим временем.

    Аргументы:
        name: Имя источника.
        data: Словарь значений.

    Возвращает:
        ConfigSnapshot.
    """
    return ConfigSnapshot(name=name, data=dict(data), timestamp=time.time())


# ─── batch_validate ───────────────────────────────────────────────────────────

def batch_validate(
    data_list: List[Dict[str, Any]],
    spec: ConfigSpec,
) -> List[List[str]]:
    """Валидировать несколько конфигураций по одной схеме.

    Аргументы:
        data_list: Список словарей.
        spec:      Схема.

    Возвращает:
        Список списков ошибок (по одному на конфигурацию).
    """
    return [validate_config(d, spec) for d in data_list]
