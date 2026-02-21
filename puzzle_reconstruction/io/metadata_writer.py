"""Запись метаданных реконструкции в структурированные форматы.

Модуль сохраняет информацию о фрагментах, компоновке и оценках
в JSON, CSV и plain-text форматы для дальнейшего анализа.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─── WriterConfig ─────────────────────────────────────────────────────────────

@dataclass
class WriterConfig:
    """Параметры записи метаданных.

    Атрибуты:
        indent:       Отступ JSON (None = компактный формат).
        sort_keys:    Сортировать ключи JSON.
        csv_dialect:  Диалект CSV.
        float_prec:   Число знаков после запятой для float.
    """

    indent: Optional[int] = 2
    sort_keys: bool = True
    csv_dialect: str = "excel"
    float_prec: int = 6

    def __post_init__(self) -> None:
        if self.indent is not None and self.indent < 0:
            raise ValueError(
                f"indent должен быть >= 0 или None, получено {self.indent}"
            )
        if self.float_prec < 0:
            raise ValueError(
                f"float_prec должен быть >= 0, получено {self.float_prec}"
            )
        if not self.csv_dialect:
            raise ValueError("csv_dialect не должен быть пустым")


# ─── MetadataRecord ───────────────────────────────────────────────────────────

@dataclass
class MetadataRecord:
    """Запись метаданных для одного фрагмента.

    Атрибуты:
        fragment_id:  ID фрагмента.
        position:     (x, y) в компоновке.
        rotation_deg: Угол поворота в градусах.
        score:        Оценка размещения [0, 1].
        extra:        Дополнительные поля.
    """

    fragment_id: int
    position: tuple  # (x, y)
    rotation_deg: float = 0.0
    score: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.position) != 2:
            raise ValueError(
                f"position должен быть кортежем (x, y), получено {self.position}"
            )
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"score должен быть в [0, 1], получено {self.score}"
            )

    def to_dict(self, float_prec: int = 6) -> Dict[str, Any]:
        """Конвертировать в словарь."""
        fmt = f".{float_prec}f"
        return {
            "fragment_id": self.fragment_id,
            "x": float(format(self.position[0], fmt)),
            "y": float(format(self.position[1], fmt)),
            "rotation_deg": float(format(self.rotation_deg, fmt)),
            "score": float(format(self.score, fmt)),
            **self.extra,
        }


# ─── MetadataCollection ───────────────────────────────────────────────────────

@dataclass
class MetadataCollection:
    """Коллекция метаданных реконструкции.

    Атрибуты:
        records:     Список MetadataRecord.
        global_meta: Глобальные метаданные (название, дата и т.п.).
    """

    records: List[MetadataRecord] = field(default_factory=list)
    global_meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_fragments(self) -> int:
        """Число фрагментов."""
        return len(self.records)

    @property
    def fragment_ids(self) -> List[int]:
        """Список всех fragment_id."""
        return [r.fragment_id for r in self.records]

    @property
    def mean_score(self) -> float:
        """Средняя оценка по всем записям."""
        if not self.records:
            return 0.0
        return float(sum(r.score for r in self.records) / len(self.records))

    def get_record(self, fragment_id: int) -> Optional[MetadataRecord]:
        """Найти запись по fragment_id или None."""
        for r in self.records:
            if r.fragment_id == fragment_id:
                return r
        return None

    def add(self, record: MetadataRecord) -> None:
        """Добавить запись."""
        self.records.append(record)

    def to_list(self, float_prec: int = 6) -> List[Dict[str, Any]]:
        """Конвертировать в список словарей."""
        return [r.to_dict(float_prec) for r in self.records]


# ─── write_json ───────────────────────────────────────────────────────────────

def write_json(
    collection: MetadataCollection,
    cfg: Optional[WriterConfig] = None,
) -> str:
    """Сериализовать коллекцию в JSON-строку.

    Аргументы:
        collection: Коллекция метаданных.
        cfg:        Параметры.

    Возвращает:
        JSON-строка.
    """
    if cfg is None:
        cfg = WriterConfig()

    payload: Dict[str, Any] = {
        "meta": collection.global_meta,
        "n_fragments": collection.n_fragments,
        "mean_score": collection.mean_score,
        "fragments": collection.to_list(cfg.float_prec),
    }
    return json.dumps(payload, indent=cfg.indent, sort_keys=cfg.sort_keys)


# ─── read_json ────────────────────────────────────────────────────────────────

def read_json(json_str: str) -> MetadataCollection:
    """Десериализовать коллекцию из JSON-строки.

    Аргументы:
        json_str: JSON-строка.

    Возвращает:
        MetadataCollection.

    Исключения:
        ValueError: при некорректном JSON или отсутствующих полях.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Некорректный JSON: {exc}") from exc

    global_meta = data.get("meta", {})
    records = []
    for item in data.get("fragments", []):
        try:
            fid = item["fragment_id"]
            x = item["x"]
            y = item["y"]
        except KeyError as exc:
            raise ValueError(f"Отсутствует обязательное поле: {exc}") from exc

        extra = {k: v for k, v in item.items()
                 if k not in ("fragment_id", "x", "y", "rotation_deg", "score")}
        rec = MetadataRecord(
            fragment_id=fid,
            position=(float(x), float(y)),
            rotation_deg=float(item.get("rotation_deg", 0.0)),
            score=float(item.get("score", 0.0)),
            extra=extra,
        )
        records.append(rec)

    return MetadataCollection(records=records, global_meta=global_meta)


# ─── write_csv ────────────────────────────────────────────────────────────────

def write_csv(
    collection: MetadataCollection,
    cfg: Optional[WriterConfig] = None,
    extra_columns: Optional[List[str]] = None,
) -> str:
    """Сериализовать коллекцию в CSV-строку.

    Аргументы:
        collection:     Коллекция метаданных.
        cfg:            Параметры.
        extra_columns:  Дополнительные столбцы из extra.

    Возвращает:
        CSV-строка.
    """
    if cfg is None:
        cfg = WriterConfig()
    if extra_columns is None:
        extra_columns = []

    base_fields = ["fragment_id", "x", "y", "rotation_deg", "score"]
    fieldnames = base_fields + extra_columns

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames,
                            dialect=cfg.csv_dialect,
                            extrasaction="ignore")
    writer.writeheader()
    for rec in collection.records:
        row = rec.to_dict(cfg.float_prec)
        writer.writerow(row)

    return output.getvalue()


# ─── write_summary ────────────────────────────────────────────────────────────

def write_summary(
    collection: MetadataCollection,
    cfg: Optional[WriterConfig] = None,
) -> str:
    """Создать текстовый отчёт-резюме.

    Аргументы:
        collection: Коллекция метаданных.
        cfg:        Параметры.

    Возвращает:
        Многострочная строка.
    """
    if cfg is None:
        cfg = WriterConfig()

    lines = [
        "=== Metadata Summary ===",
        f"Fragments : {collection.n_fragments}",
        f"Mean score: {collection.mean_score:.{cfg.float_prec}f}",
    ]

    if collection.global_meta:
        lines.append("Global metadata:")
        for k, v in sorted(collection.global_meta.items()):
            lines.append(f"  {k}: {v}")

    lines.append("Fragments:")
    for rec in collection.records:
        lines.append(
            f"  [{rec.fragment_id:4d}]  "
            f"pos=({rec.position[0]:.2f}, {rec.position[1]:.2f})  "
            f"rot={rec.rotation_deg:.1f}°  "
            f"score={rec.score:.4f}"
        )

    return "\n".join(lines)


# ─── filter_by_score ──────────────────────────────────────────────────────────

def filter_by_score(
    collection: MetadataCollection,
    min_score: float,
) -> MetadataCollection:
    """Вернуть новую коллекцию с записями, у которых score >= min_score.

    Аргументы:
        collection: Исходная коллекция.
        min_score:  Минимальная оценка [0, 1].

    Возвращает:
        Новая MetadataCollection.

    Исключения:
        ValueError: если min_score вне [0, 1].
    """
    if not (0.0 <= min_score <= 1.0):
        raise ValueError(
            f"min_score должен быть в [0, 1], получено {min_score}"
        )
    filtered = [r for r in collection.records if r.score >= min_score]
    return MetadataCollection(records=filtered,
                              global_meta=dict(collection.global_meta))


# ─── merge_collections ────────────────────────────────────────────────────────

def merge_collections(
    *collections: MetadataCollection,
) -> MetadataCollection:
    """Объединить несколько коллекций в одну.

    Записи с одинаковым fragment_id из более поздних коллекций
    перезаписывают более ранние.

    Аргументы:
        collections: Одна или несколько MetadataCollection.

    Возвращает:
        Объединённая MetadataCollection.
    """
    merged: Dict[int, MetadataRecord] = {}
    global_meta: Dict[str, Any] = {}

    for col in collections:
        global_meta.update(col.global_meta)
        for rec in col.records:
            merged[rec.fragment_id] = rec

    return MetadataCollection(
        records=list(merged.values()),
        global_meta=global_meta,
    )
