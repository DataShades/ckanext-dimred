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

    colors = _compute_colors(color_by, color_values, len(xs))

    if is_3d:
        fig, ax = _make_3d_figure(xs, ys, zs, colors)
    else:
        fig, ax = _make_2d_figure(xs, ys, colors)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    b64 = base64.b64encode(buf.read()).decode("ascii")
    return "data:image/png;base64," + b64


def _compute_colors(color_by: str | None, color_values: list[Any], n_points: int) -> list[str] | str:
    """Return color mapping for points."""
    if color_by and len(color_values) == n_points:
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
        colors: list[str] = []
        for label in color_values:
            if label not in color_map:
                idx = len(color_map) % len(palette)
                color_map[label] = palette[idx]
            colors.append(color_map[label])
        return colors
    return "#333333"


def _axis_ticks(values: np.ndarray, n: int = 5) -> tuple[list[float], tuple[float, float]]:
    """Return nice tick positions and limits for an array."""
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5
    ticks = np.linspace(vmin, vmax, n).tolist()
    return ticks, (vmin, vmax)


def _make_3d_figure(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, colors: list[str] | str):
    """Build a styled 3D matplotlib figure."""
    fig = plt.figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, s=10, c=colors, depthshade=True)
    xticks, xlim = _axis_ticks(xs)
    yticks, ylim = _axis_ticks(ys)
    zticks, zlim = _axis_ticks(zs)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_zticks(zticks)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.tick_params(labelsize=8, colors="#444444")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = "#dddddd"  # type: ignore[attr-defined]
        axis._axinfo["grid"]["linewidth"] = 0.5  # type: ignore[attr-defined]
    ax.grid(True)
    fig.tight_layout()
    return fig, ax


def _make_2d_figure(xs: np.ndarray, ys: np.ndarray, colors: list[str] | str):
    """Build a styled 2D matplotlib figure."""
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.scatter(xs, ys, s=10, c=colors)
    xticks, xlim = _axis_ticks(xs)
    yticks, ylim = _axis_ticks(ys)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.tick_params(labelsize=8, colors="#444444")
    ax.grid(True, color="#dddddd", linewidth=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    return fig, ax

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    b64 = base64.b64encode(buf.read()).decode("ascii")
    return "data:image/png;base64," + b64


def embedding_summary(embedding: np.ndarray | None, meta: dict[str, Any], top_n: int = 5) -> dict[str, Any]:
    """Compute simple summary stats for an embedding."""
    if embedding is None:
        return {}

    arr = np.asarray(embedding)
    if arr.ndim != 2 or arr.size == 0:  # noqa PLR2004
        return {}

    n_points, n_dims = arr.shape
    dim_stats = []
    labels = ["x", "y", "z"]
    for idx in range(n_dims):
        col = arr[:, idx]
        name = labels[idx] if idx < len(labels) else f"dim_{idx + 1}"
        dim_stats.append(
            {
                "name": name,
                "min": float(np.nanmin(col)),
                "max": float(np.nanmax(col)),
            }
        )

    info = (meta or {}).get("prepare_info") or {}
    color_by = info.get("color_by") or None
    color_values = info.get("color_values") or []
    n_classes = None
    top_classes: list[dict[str, Any]] = []

    if color_by and len(color_values) == n_points:
        counts: dict[str, int] = {}
        for val in color_values:
            key = str(val)
            counts[key] = counts.get(key, 0) + 1
        n_classes = len(counts)
        sorted_counts = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top_classes = [{"label": label, "count": count} for label, count in sorted_counts[:top_n]]

    return {
        "n_points": int(n_points),
        "n_dims": int(n_dims),
        "dim_stats": dim_stats,
        "color_by": color_by,
        "n_classes": n_classes,
        "top_classes": top_classes,
    }


def build_display_summary(
    meta: dict[str, Any],
    summary: dict[str, Any],
    top_n_columns: int = 5,
) -> dict[str, Any]:
    """Combine embedding summary and prepare info for display."""
    info = (meta or {}).get("prepare_info") or {}
    method_params = (meta or {}).get("method_params") or {}

    color_by = summary.get("color_by") or info.get("color_by")
    n_points = summary.get("n_points")
    rows_used = info.get("n_rows_used")
    rows_original = info.get("n_rows_original")

    numeric_used = info.get("numeric_used") or []
    categorical_used = info.get("categorical_used") or []

    method_components = summary.get("n_dims") or method_params.get("n_components")

    return {
        "method": meta.get("method"),
        "components": method_components,
        "rows_used": rows_used,
        "rows_original": rows_original,
        "points": n_points,
        "color_by": color_by,
        "classes": summary.get("n_classes"),
        "top_classes": summary.get("top_classes") or [],
        "ranges": summary.get("dim_stats") or [],
        "features": info.get("n_features"),
        "numeric_count": len(numeric_used),
        "categorical_count": len(categorical_used),
        "numeric_sample": numeric_used[:top_n_columns],
        "numeric_more": max(len(numeric_used) - top_n_columns, 0),
        "categorical_sample": categorical_used[:top_n_columns],
        "categorical_more": max(len(categorical_used) - top_n_columns, 0),
    }
