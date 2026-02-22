"""
Утилиты конфигурации и управления параметрами.

Configuration management utilities: schema validation, section merging,
environment-variable overrides, profile switching, and CLI helpers used
throughout the reconstruction pipeline.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

# Recognised scalar types for validation
_SCALAR_TYPES = (bool, int, float, str)

SchemaType = Dict[str, type]


def validate_section(
    section: object,
    schema: SchemaType,
) -> List[str]:
    """
    Validate a config section against *schema*.

    Parameters
    ----------
    section : dataclass instance or plain dict.
    schema  : {field_name: expected_type} mapping.

    Returns
    -------
    List of error messages (empty = valid).
    """
    errors: List[str] = []
    if hasattr(section, "__dict__"):
        data = vars(section)
    elif isinstance(section, dict):
        data = section
    else:
        return [f"Unsupported section type: {type(section)}"]

    for field_name, expected_type in schema.items():
        if field_name not in data:
            errors.append(f"Missing field: {field_name!r}")
            continue
        val = data[field_name]
        if val is not None and not isinstance(val, expected_type):
            errors.append(
                f"Field {field_name!r}: expected {expected_type.__name__}, "
                f"got {type(val).__name__}"
            )
    return errors


def validate_range(
    value: float,
    lo: float,
    hi: float,
    name: str = "value",
) -> Optional[str]:
    """
    Validate that *value* is in [lo, hi].

    Returns None if valid, else an error string.
    """
    if not (lo <= value <= hi):
        return f"{name} = {value} is outside [{lo}, {hi}]"
    return None


# ---------------------------------------------------------------------------
# Section merging
# ---------------------------------------------------------------------------

def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge *override* into *base*, returning a new dict.

    Parameters
    ----------
    base     : base configuration dict.
    override : values to apply on top of base.

    Returns
    -------
    Merged dict (base is not mutated).
    """
    result = dict(base)
    for key, val in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(val, dict)
        ):
            result[key] = merge_dicts(result[key], val)
        else:
            result[key] = val
    return result


def flatten_dict(
    d: Dict[str, Any],
    sep: str = ".",
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Flatten a nested dict to a single level using *sep* as key separator.

    Example
    -------
    >>> flatten_dict({"a": {"b": 1, "c": 2}})
    {"a.b": 1, "a.c": 2}
    """
    result: Dict[str, Any] = {}
    for key, val in d.items():
        full_key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(val, dict):
            result.update(flatten_dict(val, sep=sep, prefix=full_key))
        else:
            result[full_key] = val
    return result


def unflatten_dict(
    flat: Dict[str, Any],
    sep: str = ".",
) -> Dict[str, Any]:
    """
    Reconstruct a nested dict from a flat dict produced by :func:`flatten_dict`.

    Example
    -------
    >>> unflatten_dict({"a.b": 1, "a.c": 2})
    {"a": {"b": 1, "c": 2}}
    """
    result: Dict[str, Any] = {}
    for key, val in flat.items():
        parts = key.split(sep)
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = val
    return result


# ---------------------------------------------------------------------------
# Environment-variable overrides
# ---------------------------------------------------------------------------

def _try_cast(val: str) -> Any:
    """Attempt to cast a string env-var value to bool/int/float/str."""
    if val.lower() in ("true", "yes", "1"):
        return True
    if val.lower() in ("false", "no", "0"):
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def overrides_from_env(
    prefix: str = "PUZZLE_",
) -> Dict[str, Any]:
    """
    Read environment variables with *prefix* and return an override dict.

    Variable names are lowercased and *prefix* is stripped.
    Double-underscore ``__`` is treated as a section separator.

    Example
    -------
    ``PUZZLE_ASSEMBLY__METHOD=sa`` → ``{"assembly": {"method": "sa"}}``
    """
    flat: Dict[str, Any] = {}
    for key, raw in os.environ.items():
        if key.startswith(prefix):
            stripped = key[len(prefix):].lower()
            flat_key = stripped.replace("__", ".")
            flat[flat_key] = _try_cast(raw)
    return unflatten_dict(flat, sep=".")


# ---------------------------------------------------------------------------
# Profile switching
# ---------------------------------------------------------------------------

@dataclass
class ConfigProfile:
    """Named preset of config overrides."""

    name: str
    description: str
    overrides: Dict[str, Any] = field(default_factory=dict)

    def apply_to(self, cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Return *cfg_dict* with this profile's overrides merged in."""
        return merge_dicts(cfg_dict, self.overrides)


# Built-in profiles
PROFILES: Dict[str, ConfigProfile] = {
    "fast": ConfigProfile(
        name="fast",
        description="Faster but less accurate — small IFS transforms, greedy assembly",
        overrides={
            "fractal":  {"ifs_transforms": 4, "n_scales": 4, "css_n_bins": 16},
            "assembly": {"method": "greedy"},
        },
    ),
    "accurate": ConfigProfile(
        name="accurate",
        description="Slower, higher quality — more IFS transforms, beam search",
        overrides={
            "fractal":  {"ifs_transforms": 16, "n_scales": 12, "css_n_bins": 64},
            "assembly": {"method": "beam", "beam_width": 20, "sa_iter": 10000},
        },
    ),
    "debug": ConfigProfile(
        name="debug",
        description="Minimal settings for rapid debugging",
        overrides={
            "fractal":  {"ifs_transforms": 2, "n_scales": 2, "css_n_bins": 8},
            "synthesis": {"n_points": 32, "n_sides": 3},
            "assembly": {"method": "greedy"},
        },
    ),
}


def apply_profile(
    cfg_dict: Dict[str, Any],
    profile_name: str,
) -> Dict[str, Any]:
    """
    Apply a named :class:`ConfigProfile` to *cfg_dict*.

    Parameters
    ----------
    cfg_dict     : serialised config dict.
    profile_name : one of the keys in :data:`PROFILES`.

    Returns
    -------
    New dict with the profile's overrides applied.

    Raises
    ------
    ValueError if *profile_name* is unknown.
    """
    if profile_name not in PROFILES:
        available = ", ".join(sorted(PROFILES))
        raise ValueError(
            f"Unknown profile {profile_name!r}. Available: {available}"
        )
    return PROFILES[profile_name].apply_to(cfg_dict)


def list_profiles() -> List[Tuple[str, str]]:
    """Return list of (name, description) for all built-in profiles."""
    return [(p.name, p.description) for p in PROFILES.values()]


# ---------------------------------------------------------------------------
# JSON I/O helpers (lower-level than Config class)
# ---------------------------------------------------------------------------

def load_json_config(path: Path) -> Dict[str, Any]:
    """
    Load a JSON config file, returning a plain dict.

    Raises :class:`FileNotFoundError` if *path* does not exist.
    Raises :class:`json.JSONDecodeError` if content is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_config(cfg_dict: Dict[str, Any], path: Path, indent: int = 2) -> None:
    """
    Save *cfg_dict* as a formatted JSON file at *path*.

    Creates parent directories if they do not exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cfg_dict, indent=indent, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def diff_configs(
    cfg_a: Dict[str, Any],
    cfg_b: Dict[str, Any],
) -> Dict[str, Tuple[Any, Any]]:
    """
    Return a flat dict of keys where *cfg_a* and *cfg_b* differ.

    Returns ``{key: (value_in_a, value_in_b)}``.
    Compares after flattening both dicts.
    """
    flat_a = flatten_dict(cfg_a)
    flat_b = flatten_dict(cfg_b)
    all_keys = set(flat_a) | set(flat_b)
    diffs: Dict[str, Tuple[Any, Any]] = {}
    for key in all_keys:
        va = flat_a.get(key)
        vb = flat_b.get(key)
        if va != vb:
            diffs[key] = (va, vb)
    return diffs
