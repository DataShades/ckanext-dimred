from __future__ import annotations

import csv
import io
from typing import Any

import numpy as np


def embedding_to_csv(embedding: list[list[float]] | np.ndarray, meta: dict[str, Any]) -> str:
    """Convert embedding + meta into CSV string."""
    arr = np.array(embedding)
    n_dims = arr.shape[1] if arr.ndim > 1 else 1

    headers = [f"dim_{i + 1}" for i in range(n_dims)]

    prepare_info = meta.get("prepare_info", {}) or {}
    color_by = prepare_info.get("color_by")
    color_values = prepare_info.get("color_values") or []
    include_color = bool(color_by) and len(color_values) == len(arr)

    if include_color:
        headers.append(color_by)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)

    for idx, coords in enumerate(arr):
        row = list(coords[:n_dims])
        if include_color:
            row.append(color_values[idx])
        writer.writerow(row)

    return buf.getvalue()
