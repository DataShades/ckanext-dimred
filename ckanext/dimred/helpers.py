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
    """Return method options formatted for form.select macro."""
    return [{"value": m, "text": m} for m in dimred_config.allowed_methods()]


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
        df = adapter.get_dataframe()
        options.extend({"value": col, "text": col} for col in df.columns)
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
        df = adapter.get_dataframe()
        options.extend({"value": col, "text": col} for col in df.columns)

    except DimredError:
        return options

    return options


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
