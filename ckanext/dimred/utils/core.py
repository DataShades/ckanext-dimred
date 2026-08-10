from __future__ import annotations

import base64
import io
import logging
import math
from importlib import import_module
from typing import Any, cast

import numpy as np

import ckan.plugins.toolkit as tk

from ckanext.dimred.adapters import BaseAdapter, adapter_registry
from ckanext.dimred.exception import DimredEmbeddingError

log = logging.getLogger(__name__)

CATEGORICAL_PALETTE = [
    "#5470c6",
    "#91cc75",
    "#fac858",
    "#ee6666",
    "#73c0de",
    "#3ba272",
    "#fc8452",
    "#9a60b4",
    "#ea7ccc",
]
MISSING_COLOR = "#999999"
NUMERIC_COLORMAP = "Blues"
MAX_COLOR_LEGEND_ITEMS = 30


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
    plt = import_module("matplotlib.pyplot")

    if embedding.shape[1] < 2:  # noqa PLR2004
        raise DimredEmbeddingError

    xs = embedding[:, 0]
    ys = embedding[:, 1]
    is_3d = embedding.shape[1] >= 3  # noqa PLR2004

    info = meta.get("prepare_info", {}) or {}
    color_spec = _color_spec(info, len(xs))

    if is_3d:
        fig, ax, scatter = _make_3d_figure(plt, xs, ys, embedding[:, 2], color_spec)
    else:
        fig, ax, scatter = _make_2d_figure(plt, xs, ys, color_spec)

    if color_spec["kind"] == "numeric":
        fig.colorbar(scatter, ax=ax, label=info.get("color_by") or "")
    elif color_spec["legend"]:
        for item in color_spec["legend"]:
            ax.scatter([], [], s=30, c=item["color"], label=item["label"])
        ax.legend(title=info.get("color_by") or None, loc="best", fontsize=8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    b64 = base64.b64encode(buf.read()).decode("ascii")
    return "data:image/png;base64," + b64


def _color_spec(info: dict[str, Any], n_points: int) -> dict[str, Any]:
    """Build shared categorical or numeric colour semantics for Matplotlib."""
    color_values = info.get("color_values") or []
    color_kind = info.get("color_kind")
    if not info.get("color_by") or len(color_values) != n_points:
        return {"kind": "uniform", "values": "#333333", "legend": []}

    if color_kind == "numeric":
        values = []
        for value in color_values:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = math.nan
            values.append(numeric if math.isfinite(numeric) else math.nan)
        numeric_values = np.asarray(values)
        if not np.isfinite(numeric_values).any():
            return {"kind": "uniform", "values": "#333333", "legend": []}
        return {"kind": "numeric", "values": numeric_values, "legend": []}

    known_labels = _color_legend_labels(info, color_values)
    color_map = {
        label: CATEGORICAL_PALETTE[index % len(CATEGORICAL_PALETTE)] for index, label in enumerate(known_labels)
    }
    colors: list[str] = []
    has_missing = False
    for value in color_values:
        if value is None or value == "":
            colors.append(MISSING_COLOR)
            has_missing = True
            continue
        label = str(value)
        if label not in color_map:
            color_map[label] = CATEGORICAL_PALETTE[len(color_map) % len(CATEGORICAL_PALETTE)]
        colors.append(color_map[label])
    legend = [{"label": label, "color": color_map[label]} for label in known_labels]
    if has_missing:
        legend.append({"label": str(tk._("Missing")), "color": MISSING_COLOR})
    return {"kind": "categorical", "values": colors, "legend": legend}


def _color_legend_labels(info: dict[str, Any], color_values: list[Any]) -> list[str]:
    """Return the bounded candidate labels in the same order as the frontend."""
    color_by = info.get("color_by")
    for candidate in info.get("color_candidates") or []:
        if candidate.get("name") == color_by and candidate.get("kind") == "categorical":
            return [str(value) for value in candidate.get("unique_values") or []]

    labels: list[str] = []
    for value in color_values:
        if value is None or value == "":
            continue
        label = str(value)
        if label not in labels:
            labels.append(label)
        if len(labels) == MAX_COLOR_LEGEND_ITEMS:
            break
    return labels


def _axis_ticks(values: np.ndarray, n: int = 5) -> tuple[list[float], tuple[float, float]]:
    """Return nice tick positions and limits for an array."""
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5
    ticks = np.linspace(vmin, vmax, n).tolist()
    return ticks, (vmin, vmax)


def _make_3d_figure(plt: Any, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, color_spec: dict[str, Any]):
    """Build a styled 3D matplotlib figure."""
    fig = plt.figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    scatter = _scatter(ax, xs, ys, zs, color_spec, depthshade=True)
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
    return fig, ax, scatter


def _make_2d_figure(plt: Any, xs: np.ndarray, ys: np.ndarray, color_spec: dict[str, Any]):
    """Build a styled 2D matplotlib figure."""
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    for spine in ax.spines.values():
        spine.set_visible(False)
    scatter = _scatter(ax, xs, ys, None, color_spec)
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
    return fig, ax, scatter


def _scatter(
    ax: Any,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray | None,
    color_spec: dict[str, Any],
    **kwargs: Any,
) -> Any:
    """Draw a scatter plot with the selected colour contract."""
    if color_spec["kind"] == "numeric":
        values = cast(np.ndarray, color_spec["values"])
        finite = np.isfinite(values)
        cmap = import_module("matplotlib").colormaps[NUMERIC_COLORMAP].copy()
        numeric_options: dict[str, Any] = {"s": 10, "c": values[finite], "cmap": cmap, **kwargs}
        if zs is None:
            scatter = ax.scatter(xs[finite], ys[finite], **numeric_options)
            if not finite.all():
                ax.scatter(xs[~finite], ys[~finite], s=10, c=MISSING_COLOR, **kwargs)
        else:
            scatter = ax.scatter(xs[finite], ys[finite], zs[finite], **numeric_options)
            if not finite.all():
                ax.scatter(xs[~finite], ys[~finite], zs[~finite], s=10, c=MISSING_COLOR, **kwargs)
        return scatter

    options: dict[str, Any] = {"s": 10, "c": color_spec["values"], **kwargs}
    if zs is None:
        return ax.scatter(xs, ys, **options)
    return ax.scatter(xs, ys, zs, **options)


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

    if color_by and info.get("color_kind") != "numeric" and len(color_values) == n_points:
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
    skipped_columns = info.get("skipped_columns") or []
    skipped_reasons = _skipped_reason_counts(skipped_columns)
    dropped_rows = info.get("n_rows_dropped", max((rows_original or 0) - (rows_used or 0), 0))
    warnings = _analysis_warning_codes(meta.get("method"), dropped_rows, numeric_used, categorical_used)
    projection_info = (meta or {}).get("projection_info") or {}

    return {
        "method": meta.get("method"),
        "components": method_components,
        "rows_used": rows_used,
        "rows_original": rows_original,
        "rows_dropped": dropped_rows,
        "row_limit": info.get("row_limit"),
        "sampling_method": info.get("sampling_method", "all_rows"),
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
        "skipped_count": len(skipped_columns),
        "skipped_reasons": skipped_reasons,
        "pca_variance": projection_info.get("explained_variance_ratio") or [],
        "pca_variance_cumulative": projection_info.get("explained_variance_cumulative"),
        "warnings": warnings,
    }


def _skipped_reason_counts(skipped_columns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count automatic feature skips by their stable pipeline reason."""
    counts: dict[str, int] = {}
    for column in skipped_columns:
        reason = str(column.get("reason") or "other")
        counts[reason] = counts.get(reason, 0) + 1
    return [{"reason": reason, "count": count} for reason, count in counts.items()]


def _analysis_warning_codes(
    method: Any,
    dropped_rows: int,
    numeric_used: list[str],
    categorical_used: list[str],
) -> list[str]:
    """Return concise, display-ready warning codes for a preview."""
    warnings: list[str] = []
    if method in {"umap", "tsne"}:
        warnings.append("projection_interpretation")
    if dropped_rows:
        warnings.append("sampled_rows")
    if numeric_used:
        warnings.append("numeric_standardized")
    if categorical_used:
        warnings.append("categorical_one_hot")
    return warnings
