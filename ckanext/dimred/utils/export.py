from __future__ import annotations

import csv
import io
from typing import Any

import numpy as np

_FORMULA_PREFIXES = ("=", "+", "-", "@")


def _safe_csv_cell(value: Any) -> Any:
    """Prevent spreadsheet applications from treating untrusted text as a formula."""
    if isinstance(value, str) and value.lstrip().startswith(_FORMULA_PREFIXES):
        return f"'{value}"
    return value


def embedding_to_csv(embedding: list[list[float]] | np.ndarray, meta: dict[str, Any]) -> str:
    """Convert embedding + meta into CSV string."""
    arr = np.array(embedding)
    n_dims = arr.shape[1] if arr.ndim > 1 else 1

    labels = ["x", "y", "z"]
    headers = [labels[i] if i < len(labels) else f"dim_{i + 1}" for i in range(n_dims)]

    prepare_info = meta.get("prepare_info", {}) or {}
    source_row_ids = prepare_info.get("source_row_ids") or []
    color_by = prepare_info.get("color_by")
    color_values = prepare_info.get("color_values") or []
    display_fields = prepare_info.get("display_fields") or []
    include_source_row_id = len(source_row_ids) == len(arr)
    include_color = bool(color_by) and len(color_values) == len(arr)
    included_display_fields = [
        field
        for field in display_fields
        if isinstance(field, dict)
        and field.get("name") != color_by
        and isinstance(field.get("values"), list)
        and len(field["values"]) == len(arr)
    ]

    if include_source_row_id:
        headers.append("source_row_id")
    if include_color:
        headers.append(_safe_csv_cell(color_by))
    headers.extend(_safe_csv_cell(field["name"]) for field in included_display_fields)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)

    for idx, coords in enumerate(arr):
        row = list(coords[:n_dims])
        if include_source_row_id:
            row.append(_safe_csv_cell(source_row_ids[idx]))
        if include_color:
            row.append(_safe_csv_cell(color_values[idx]))
        row.extend(_safe_csv_cell(field["values"][idx]) for field in included_display_fields)
        writer.writerow(row)

    return buf.getvalue()
