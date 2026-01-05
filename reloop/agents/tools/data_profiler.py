from __future__ import annotations

import numbers
from typing import Any, Dict, List

from ..schemas import DataFieldProfile, DataProfile


def _type_name(value: Any) -> str:
    if isinstance(value, numbers.Number):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _sample_keys(mapping: Dict[Any, Any]) -> List[str]:
    keys: List[str] = []
    for k in mapping.keys():
        keys.append(str(k))
        if len(keys) >= 5:
            break
    return keys


def _profile(value: Any, path: str, fields: List[DataFieldProfile]) -> None:
    if isinstance(value, dict):
        fields.append(
            DataFieldProfile(
                path=path or "<root>",
                kind="dict",
                type="dict",
                key_types="mixed" if not value else type(next(iter(value.keys()))).__name__,
                sample_keys=_sample_keys(value),
            )
        )
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            _profile(child, child_path, fields)
        return

    if isinstance(value, list):
        elem_type = _type_name(value[0]) if value else "unknown"
        fields.append(
            DataFieldProfile(
                path=path or "<root>",
                kind="list",
                type="list",
                length=len(value),
                element_type=elem_type,
            )
        )
        if value and isinstance(value[0], (dict, list)):
            _profile(value[0], f"{path}[0]" if path else "[0]", fields)
        return

    fields.append(
        DataFieldProfile(
            path=path or "<root>", kind="scalar", type=_type_name(value)
        )
    )


def profile_data(data: Any) -> DataProfile:
    """Profile the structure (type/indexing) of scenario data without numeric values."""
    fields: List[DataFieldProfile] = []
    _profile(data, "", fields)
    summary = "type/indexing profile generated from scenario data"
    return DataProfile(summary=summary, fields=fields)