from __future__ import annotations

import base64
import io
import logging
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import ckan.plugins.toolkit as tk

from ckanext.dimred.adapters import BaseAdapter, adapter_registry
from ckanext.dimred.exception import DimredEmbeddingError

log = logging.getLogger(__name__)


collect_adapters_signal = tk.signals.ckanext.signal(
    "dimred:register_format_adapters",
    "Collect adapters from subscribers",
)
get_adapter_for_resource_signal = tk.signals.ckanext.signal(
    "dimred:get_adapter_for_resource",
    "Get adapter for a given resource",
)


def printable_file_size(size_bytes: int) -> str:
    """Return a human-readable file size string."""
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(float(size_bytes) / p, 1)
    return f"{s} {size_name[i]}"


def get_adapter_for_resource(
    resource: dict[str, Any],
) -> type[BaseAdapter] | None:
    """Resolve an adapter class for the given resource via signals or registry."""
    res_format = (resource.get("format") or "").lower()

    for _, adapter in get_adapter_for_resource_signal.send(resource):
        if adapter is None:
            continue
        if adapter is False:
            return None
        return adapter

    return adapter_registry.get(res_format)


def embedding_to_png_data_url(embedding: np.ndarray, meta: dict[str, Any]) -> str:
    """Render a 2D/3D scatter plot for the embedding and return a data URL."""
    if embedding.shape[1] < 2:  # noqa PLR2004
        raise DimredEmbeddingError

    xs = embedding[:, 0]
    ys = embedding[:, 1]
    is_3d = embedding.shape[1] >= 3  # noqa PLR2004
    zs = embedding[:, 2] if is_3d else None

    info = meta.get("prepare_info", {}) or {}
    color_by = info.get("color_by")
    color_values = info.get("color_values") or []

    if color_by and len(color_values) == len(xs):
        palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        color_map: dict[str, str] = {}
        colors = []
        for label in color_values:
            if label not in color_map:
                idx = len(color_map) % len(palette)
                color_map[label] = palette[idx]
            colors.append(color_map[label])
    else:
        colors = "#333333"

    if is_3d:
        fig = plt.figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xs, ys, zs, s=10, c=colors, depthshade=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.scatter(xs, ys, s=10, c=colors)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    b64 = base64.b64encode(buf.read()).decode("ascii")
    return "data:image/png;base64," + b64
