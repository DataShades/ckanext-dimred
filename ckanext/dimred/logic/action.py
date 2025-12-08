from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ckan import types
from ckan.logic import validate
from ckan.plugins import toolkit as tk

from ckanext.dimred import config as dimred_config
from ckanext.dimred import utils as dimred_utils
from ckanext.dimred.exception import (
    DimredAdapterNotFoundError,
    DimredFeatureError,
    DimredNumericColumnError,
)
from ckanext.dimred.logic import schema
from ckanext.dimred.methods import BaseProjectionMethod, get_projection_method
from ckanext.dimred.utils import cache as dimred_cache
from ckanext.dimred.utils.export import embedding_to_csv


@tk.side_effect_free
@validate(schema.dimred_get_dimred_preview_schema)
def dimred_get_dimred_preview(context: types.Context, data_dict: types.DataDict) -> types.ActionResult:
    """Return embedding and metadata for a given resource + view pair.

    Expected data_dict keys:
    - id: resource id
    - view_id: resource_view id
    """
    resource = tk.get_action("resource_show")(context, {"id": data_dict["id"]})
    resource_view = tk.get_action("resource_view_show")(context, {"id": data_dict["view_id"]})

    return dimred_run_dimred_pipeline(
        context,
        {
            "resource": resource,
            "resource_view": resource_view,
        },
    )


@tk.side_effect_free
def dimred_run_dimred_pipeline(context: types.Context, data_dict: types.DataDict) -> types.ActionResult:
    """Execute the dimred pipeline and return embedding + metadata.

    Accepts either pre-fetched resource/resource_view or ids.
    """
    resource = data_dict.get("resource")
    resource_view = data_dict.get("resource_view")

    if resource is None:
        resource = tk.get_action("resource_show")(context, {"id": data_dict["id"]})
    if resource_view is None:
        resource_view = tk.get_action("resource_view_show")(context, {"id": data_dict["view_id"]})

    resource_id = resource["id"]
    resource_view_id = resource_view["id"]

    method_params = _parse_method_params(resource_view.get("method_params"))
    resource_view = dict(resource_view)
    resource_view["method_params"] = method_params

    settings = _cache_settings(resource_view)
    cache = dimred_cache.get_cache()
    settings_sig = cache.settings_signature(settings)

    cached = cache.get(resource_id, resource_view_id, settings_sig)
    if cached:
        return cached

    embedding, meta = _build_dimred_preview(resource, resource_view)
    embedding_serializable = embedding.tolist() if hasattr(embedding, "tolist") else embedding

    result = {"embedding": embedding_serializable, "meta": meta}
    cache.save(resource_id, resource_view_id, settings_sig, result)

    return result


@tk.side_effect_free
@validate(schema.dimred_export_embedding_schema)
def dimred_export_embedding(context: types.Context, data_dict: types.DataDict) -> types.ActionResult:
    """Return CSV export for a dimred preview."""
    if not dimred_config.export_enabled():
        raise tk.ValidationError({"export": ["Dimred export is disabled."]})

    result = tk.get_action("dimred_get_dimred_preview")(context, data_dict)
    if not result or "embedding" not in result:
        raise DimredFeatureError

    csv_content = embedding_to_csv(result["embedding"], result["meta"])
    resource_id = data_dict["id"]
    view_id = data_dict["view_id"]
    filename = f"dimred-{resource_id}-{view_id}.csv"

    return {
        "filename": filename,
        "content": csv_content,
        "content_type": "text/csv; charset=utf-8",
    }


def _build_dimred_preview(
    resource: dict[str, Any],
    resource_view: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the dimred pipeline for a given resource + view."""
    method_name = (resource_view.get("method") or "").strip() or dimred_config.default_method()
    allowed_methods = set(dimred_config.allowed_methods())

    if method_name not in allowed_methods:
        raise tk.ValidationError({"method": [f"Method '{method_name}' is not allowed."]})

    method_cls = get_projection_method(method_name)
    method_params = _parse_method_params(resource_view.get("method_params"))

    reducer: BaseProjectionMethod = method_cls(**method_params)

    x_matrix, prepare_info = _prepare_matrix_from_resource(resource, resource_view)

    embedding = reducer.fit_transform(x_matrix)

    meta: dict[str, Any] = {
        "method": method_name,
        "method_params": reducer.params,
        "prepare_info": prepare_info,
    }

    return embedding, meta


def _cache_settings(resource_view: dict[str, Any]) -> dict[str, Any]:
    """Build settings dict that affects cache identity."""
    method_name = (resource_view.get("method") or "").strip() or dimred_config.default_method()
    return {
        "method": method_name,
        "method_params": resource_view.get("method_params"),
        "feature_columns": resource_view.get("feature_columns"),
        "color_by": resource_view.get("color_by"),
        "max_rows": dimred_config.max_rows(),
        "enable_categorical": dimred_config.enable_categorical(),
        "max_categories_for_ohe": dimred_config.max_categories_for_ohe(),
    }


def _parse_method_params(raw_params: str | dict[str, Any] | None) -> dict[str, Any]:
    """Parse method_params JSON string or dict into a dict."""
    if raw_params is None:
        return {}
    if isinstance(raw_params, dict):
        return raw_params
    if not isinstance(raw_params, str):
        raise tk.ValidationError({"method_params": ["method_params must be a JSON object."]})

    raw_params = raw_params.strip()
    if not raw_params:
        return {}

    try:
        parsed = json.loads(raw_params)
    except ValueError as err:
        raise tk.ValidationError({"method_params": ["Invalid JSON in method_params."]}) from err
    if not isinstance(parsed, dict):
        raise tk.ValidationError({"method_params": ["method_params must be a JSON object."]})
    return parsed


def _prepare_matrix_from_resource(
    resource: dict[str, Any],
    resource_view: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a tabular resource, select suitable columns and return a feature matrix.

    Features:
    - numeric columns are always included;
    - optional low-cardinality categorical columns are one-hot encoded
      if enabled in config;
    - optional 'color_by' column is passed through to metadata.
    """
    df = _load_dataframe(resource, resource_view)
    df, n_rows_original = _maybe_limit_rows(df)

    color_by, color_values = _extract_color_info(df, resource_view)
    selected_features = _extract_selected_features(df, resource_view)

    numeric_cols = _select_numeric_columns(df, selected_features)
    categorical_cols = _select_categorical_columns(df, numeric_cols, color_by, selected_features)

    df_features = _build_feature_frame(df, numeric_cols, categorical_cols)

    scaler = StandardScaler()
    x_matrix = scaler.fit_transform(df_features.values)

    info: dict[str, Any] = {
        "n_rows_original": n_rows_original,
        "n_rows_used": len(df),
        "n_features": x_matrix.shape[1],
        "numeric_used": numeric_cols,
        "categorical_used": categorical_cols,
        "color_by": color_by or None,
        "color_values": color_values,
        "feature_columns": selected_features or None,
    }

    return x_matrix, info


def _load_dataframe(resource: dict[str, Any], resource_view: dict[str, Any]) -> pd.DataFrame:
    """Load dataframe via adapter with validation."""
    adapter_cls = dimred_utils.get_adapter_for_resource(resource)
    if adapter_cls is None:
        res_format = (resource.get("format") or "").lower()
        raise DimredAdapterNotFoundError(res_format)

    adapter = adapter_cls(resource, resource_view)
    df = adapter.get_dataframe()

    if df.empty:
        raise DimredFeatureError

    return df


def _maybe_limit_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Apply max_rows sampling if configured."""
    n_rows_original = len(df)
    max_rows = dimred_config.max_rows()
    if max_rows and n_rows_original > max_rows:
        df = df.sample(max_rows, random_state=42).reset_index(drop=True)
    return df, n_rows_original


def _extract_color_info(df: pd.DataFrame, resource_view: dict[str, Any]) -> tuple[str, list[str] | None]:
    """Extract color_by and corresponding values."""
    color_by = (resource_view.get("color_by") or "").strip()
    if color_by and color_by in df.columns:
        return color_by, df[color_by].astype(str).tolist()
    return "", None


def _extract_selected_features(df: pd.DataFrame, resource_view: dict[str, Any]) -> list[str]:
    """Parse feature selection from resource_view and validate against df columns."""
    raw_features = resource_view.get("feature_columns") or []
    selected: list[str] = []
    if raw_features:
        if isinstance(raw_features, str):
            try:
                parsed = json.loads(raw_features)
                if isinstance(parsed, list):
                    selected = [str(v) for v in parsed]
            except json.JSONDecodeError:
                selected = [f.strip() for f in raw_features.split(",") if f.strip()]
        elif isinstance(raw_features, (list, tuple, set)):
            selected = [str(v) for v in raw_features]

    return [c for c in selected if c in df.columns]


def _select_numeric_columns(df: pd.DataFrame, selected_features: list[str]) -> list[str]:
    """Return numeric columns, optionally filtered by selected_features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if selected_features:
        numeric_cols = [c for c in numeric_cols if c in selected_features]
    if not numeric_cols:
        raise DimredNumericColumnError
    return numeric_cols


def _select_categorical_columns(
    df: pd.DataFrame,
    numeric_cols: list[str],
    color_by: str,
    selected_features: list[str],
) -> list[str]:
    """Return low-cardinality categorical columns to include."""
    categorical_cols: list[str] = []
    if not dimred_config.enable_categorical():
        return categorical_cols

    max_cat = dimred_config.max_categories_for_ohe()
    for col in df.columns:
        if col in numeric_cols:
            continue
        if col == color_by:
            continue
        if selected_features and col not in selected_features:
            continue
        n_unique = df[col].nunique(dropna=True)
        if 1 < n_unique <= max_cat:
            categorical_cols.append(col)
    return categorical_cols


def _build_feature_frame(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    """Assemble feature frame with one-hot encoding and basic cleaning."""
    feature_cols = numeric_cols + categorical_cols
    df_features = df[feature_cols].copy()

    if categorical_cols:
        df_features = pd.get_dummies(df_features, columns=categorical_cols, dummy_na=False, drop_first=False)

    df_features = df_features.astype(float)
    df_features = df_features.fillna(df_features.mean())
    df_features = df_features.fillna(0.0)

    if df_features.shape[1] < 2:  # noqa PLR2004
        raise DimredFeatureError

    return df_features
