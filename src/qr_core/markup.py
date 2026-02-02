from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_markup(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def extract_expected_value(markup: dict[str, Any]) -> str | None:
    try:
        value = markup["props"]["barcode"]["value"]
    except (KeyError, TypeError):
        return None

    if value is None:
        return None
    return _safe_str(value)


def extract_module_size_px(markup: dict[str, Any]) -> float | None:
    try:
        module_size = markup["props"]["barcode"]["module_size"]
    except (KeyError, TypeError):
        return None

    if not isinstance(module_size, (list, tuple)) or len(module_size) != 2:
        return None

    values: list[float] = []
    for pair in module_size:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        try:
            values.append(float(pair[0]))
            values.append(float(pair[1]))
        except (TypeError, ValueError):
            return None

    if not values:
        return None

    return sum(values) / float(len(values))


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)
