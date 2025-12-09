from __future__ import annotations

from typing import Any

import ckan.plugins.toolkit as tk

from ckanext.dimred import config as dimred_config
from ckanext.dimred import utils as dimred_utils
from ckanext.dimred.exception import DimredError
from ckanext.dimred.methods import get_projection_method


def dimred_allowed_methods() -> list[str]:
    """Return the list of enabled dimred methods from config."""
    return dimred_config.allowed_methods()


def dimred_default_method() -> str:
    """Return the default dimred method from config."""
    return dimred_config.default_method()


def dimred_allowed_methods_options() -> list[dict[str, str]]:
    """Return method options formatted for form.select macro with friendly labels."""
    labels = dimred_method_labels()
    options = []
    for m in dimred_config.allowed_methods():
        text = labels.get(m, m)
        options.append({"value": m, "text": text})
    return options


def dimred_color_options(fields: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return color_by select options from resource fields."""
    options = [{"value": "", "text": tk._("Not selected")}]
    for field in fields or []:
        field_id = field.get("id")
        if field_id:
            options.append({"value": field_id, "text": field_id})
    return options


def dimred_color_options_from_resource(
    resource: dict[str, Any], resource_view: dict[str, Any] | None = None
) -> list[dict[str, str]]:
    """Return color_by options derived from resource columns via adapter."""
    options = [{"value": "", "text": tk._("Not selected")}]

    try:
        adapter_cls = dimred_utils.get_adapter_for_resource(resource)
        if not adapter_cls:
            return options
        adapter = adapter_cls(resource, resource_view or {})
        cols = adapter.get_columns()
        options.extend({"value": col, "text": col} for col in cols)
    except DimredError:
        return options

    return options


def dimred_feature_options_from_resource(
    resource: dict[str, Any], resource_view: dict[str, Any] | None = None
) -> list[dict[str, str]]:
    """Return feature selection options derived from resource columns."""
    options: list[dict[str, str]] = []

    try:
        adapter_cls = dimred_utils.get_adapter_for_resource(resource)
        if not adapter_cls:
            return options
        adapter = adapter_cls(resource, resource_view or {})
        cols = adapter.get_columns()
        options.extend({"value": col, "text": col} for col in cols)

    except DimredError:
        return options

    return options


def dimred_export_enabled() -> bool:
    """Check whether export button/endpoint is enabled."""
    return dimred_config.export_enabled()


def dimred_method_default_params(method_name: str) -> dict[str, Any]:
    """Return default params for a given dimred method."""
    try:
        method = get_projection_method(method_name)
    except KeyError:
        return {}

    return method.default_params()


def dimred_methods_defaults() -> dict[str, dict[str, Any]]:
    """Return defaults for all allowed methods keyed by method name."""
    defaults: dict[str, dict[str, Any]] = {}
    for name in dimred_config.allowed_methods():
        defaults[name] = dimred_method_default_params(name)
    return defaults


def dimred_method_labels() -> dict[str, str]:
    """Return mapping of method names to display labels."""
    return {
        "umap": "UMAP",
        "tsne": "t-SNE",
        "pca": "PCA",
    }


def dimred_method_label(method: str) -> str:
    """Return a friendly label for a method name."""
    return dimred_method_labels().get(method, method)


def dimred_render_backend_default() -> str:
    """Return default render backend from config."""
    return dimred_config.render_backend()


def dimred_render_backend_options() -> list[dict[str, str]]:
    """Return select options for render backend."""
    labels = {
        "echarts": tk._("ECharts (interactive)"),
        "matplotlib": tk._("Matplotlib (PNG)"),
    }
    return [{"value": key, "text": labels.get(key, key)} for key in ("echarts", "matplotlib")]


def dimred_render_asset(render_backend: str | None = None) -> str | None:
    """Return asset bundle for render backend (customizable)."""
    backend = render_backend or dimred_config.render_backend()
    custom = dimred_config.render_asset()
    if custom:
        return custom
    if _use_echarts(backend):
        return "dimred/dimred-echarts-js"
    return None


def dimred_render_module(render_backend: str | None = None) -> str | None:
    """Return CKAN module name for render backend (customizable)."""
    backend = render_backend or dimred_config.render_backend()
    custom = dimred_config.render_module()
    if custom:
        return custom
    if _use_echarts(backend):
        return "dimred-view-echarts"
    return None


def _use_echarts(render_backend: str | None = None) -> bool:
    """True if echarts backend is selected."""
    backend = render_backend or dimred_config.render_backend()
    return backend == "echarts"
