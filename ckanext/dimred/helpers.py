from __future__ import annotations

import json
from typing import Any, Protocol, cast

import ckan.plugins.toolkit as tk

from ckanext.dimred import config as dimred_config
from ckanext.dimred import utils as dimred_utils
from ckanext.dimred.exception import DimredError
from ckanext.dimred.methods import get_projection_method


class ColumnAdapter(Protocol):
    """Minimal adapter interface needed to populate form column options."""

    def get_columns(self) -> list[str]: ...


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
    """Return color_by options derived from the resource's active data source."""
    options = [{"value": "", "text": tk._("Not selected")}]
    options.extend({"value": col, "text": col} for col in _resource_columns(resource, resource_view))
    return options


def dimred_feature_options_from_resource(
    resource: dict[str, Any], resource_view: dict[str, Any] | None = None
) -> list[dict[str, str]]:
    """Return feature selection options derived from the active data source."""
    return [
        option
        for group in dimred_feature_option_groups_from_resource(resource, resource_view)
        for option in group["options"]
    ]


def dimred_feature_option_groups_from_resource(
    resource: dict[str, Any], resource_view: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Return feature options grouped by DataStore schema type when available."""
    try:
        if resource.get("datastore_active"):
            return _datastore_feature_option_groups(_datastore_fields(resource))

        columns = _resource_columns(resource, resource_view)
        if not columns:
            return []
        return [{"label": tk._("Columns"), "options": [{"value": col, "text": col} for col in columns]}]
    except (DimredError, KeyError, tk.NotAuthorized, tk.ObjectNotFound, tk.ValidationError):
        return []


def _resource_columns(resource: dict[str, Any], resource_view: dict[str, Any] | None) -> list[str]:
    """Return selectable columns from DataStore or the resource adapter."""
    try:
        if resource.get("datastore_active"):
            return [field["id"] for field in _datastore_fields(resource) if field.get("id") != "_id"]

        adapter_cls = dimred_utils.get_adapter_for_resource(resource)
        if not adapter_cls:
            return []
        adapter = cast(ColumnAdapter, adapter_cls(resource, resource_view or {}))
        return adapter.get_columns()
    except (DimredError, KeyError, tk.NotAuthorized, tk.ObjectNotFound, tk.ValidationError):
        return []


def _datastore_fields(resource: dict[str, Any]) -> list[dict[str, Any]]:
    """Return DataStore fields through CKAN's authorized action."""
    info = tk.get_action("datastore_info")({}, {"id": resource["id"]})
    return [field for field in info.get("fields", []) if field.get("id") != "_id"]


def _datastore_feature_option_groups(fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group DataStore fields without inferring pipeline feature eligibility."""
    numeric_options: list[dict[str, str]] = []
    other_options: list[dict[str, str]] = []

    for field in fields:
        field_id = field.get("id")
        if not field_id:
            continue
        option = {"value": field_id, "text": field_id}
        if _is_numeric_datastore_type(field.get("type")):
            numeric_options.append(option)
        else:
            other_options.append(option)

    groups: list[dict[str, Any]] = []
    if numeric_options:
        groups.append({"label": tk._("Numeric columns"), "options": numeric_options})
    if other_options:
        groups.append({"label": tk._("Other columns"), "options": other_options})
    return groups


def _is_numeric_datastore_type(value: Any) -> bool:
    """Recognize numeric DataStore schema types without examining row values."""
    type_name = str(value or "").lower()
    return type_name in {
        "bigint",
        "decimal",
        "double precision",
        "float",
        "float4",
        "float8",
        "int",
        "int2",
        "int4",
        "int8",
        "integer",
        "numeric",
        "real",
        "smallint",
    }


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


def dimred_method_default_params_form(method_name: str) -> dict[str, Any]:
    """Return default params for a method, excluding n_components (handled separately)."""
    params = dict(dimred_method_default_params(method_name))
    params.pop("n_components", None)
    return params


def dimred_methods_defaults_form() -> dict[str, dict[str, Any]]:
    """Return defaults for all methods excluding the dedicated n_components field."""
    defaults: dict[str, dict[str, Any]] = {}
    for name in dimred_config.allowed_methods():
        defaults[name] = dimred_method_default_params_form(name)
    return defaults


def dimred_method_params_form_values(raw_params: Any, method_name: str) -> dict[str, Any]:
    """Return displayable method parameters merged with the configured defaults."""
    defaults = dimred_method_default_params_form(method_name)
    if isinstance(raw_params, str):
        try:
            raw_params = json.loads(raw_params)
        except (TypeError, ValueError):
            raw_params = {}

    if not isinstance(raw_params, dict):
        return defaults

    values = dict(defaults)
    for name in values:
        if raw_params.get(name) is not None:
            values[name] = raw_params[name]
    return values


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
        "matplotlib": tk._("Matplotlib (static)"),
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
