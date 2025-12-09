from __future__ import annotations

import json
import logging
from typing import Any

import ckan.plugins.toolkit as tk
from ckan import types

from ckanext.dimred import config as dimred_config

log = logging.getLogger(__name__)


def dimred_allowed_method(value: Any, context: types.Context) -> str:
    """Validate that method belongs to allowed dimred methods."""
    if value in (None, ""):
        return value

    method = str(value).strip()
    if not method:
        return method

    allowed_raw = dimred_config.allowed_methods()
    if isinstance(allowed_raw, str):
        allowed = {part for part in (v.strip() for v in allowed_raw.split()) if part}
    else:
        allowed = {str(v).strip() for v in allowed_raw if str(v).strip()}
    if method not in allowed:
        raise tk.Invalid(tk._("Method '{method}' is not allowed.").format(method=method))

    return method


def dimred_feature_columns_list(value: Any, context: types.Context) -> list[str]:
    """Validate/normalize feature_columns to a list of strings."""
    if value in (None, ""):
        return value

    parsed: list[str] | None = None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            loaded = json.loads(text)
            if isinstance(loaded, list):
                parsed = [str(v) for v in loaded]
        except json.JSONDecodeError:
            parsed = [v.strip() for v in text.split(",") if v.strip()]
    elif isinstance(value, (list, tuple, set)):
        parsed = [str(v) for v in value]

    if parsed is None:
        raise tk.Invalid(tk._("feature_columns must be a list or comma-separated string."))

    return parsed


def dimred_method_params_object(value: Any, context: types.Context) -> dict[str, Any]:
    """Validate that method_params is a JSON object or dict."""
    if value in (None, ""):
        return value

    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as err:
            raise tk.Invalid(tk._("Invalid JSON in method_params.")) from err
        if isinstance(parsed, dict):
            return parsed
        raise tk.Invalid(tk._("method_params must be a JSON object."))

    raise tk.Invalid(tk._("method_params must be a JSON object."))


def dimred_allowed_backend(value: Any, context: types.Context) -> str:
    """Validate that render_backend is one of the supported values."""
    if value in (None, ""):
        return value

    backend = str(value).strip()
    if not backend:
        return backend

    allowed = {"echarts", "matplotlib"}
    if backend not in allowed:
        raise tk.Invalid(tk._("Render backend '{backend}' is not supported.").format(backend=backend))

    return backend
