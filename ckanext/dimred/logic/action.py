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

__all__ = ["dimred_get_dimred_preview", "dimred_export_embedding"]

METHOD_PARAM_NAMES = {
    "pca": {"n_components", "random_state", "whiten"},
    "tsne": {"n_components", "perplexity", "random_state"},
    "umap": {"min_dist", "n_components", "n_neighbors", "random_state"},
}


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
    _validate_resource_view_resource(resource, resource_view)

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
    decimals = dimred_config.embedding_decimals()
    embedding = np.round(np.asarray(embedding, dtype=float), decimals)
    embedding_serializable = embedding.tolist()

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

    try:
        method_cls = get_projection_method(method_name)
    except KeyError as err:
        raise tk.ValidationError({"method": [f"Method '{method_name}' is not supported."]}) from err

    method_params = _parse_method_params(resource_view.get("method_params"))
    n_components = _parse_n_components(resource_view.get("n_components"))
    if n_components is not None:
        method_params = dict(method_params)
        method_params["n_components"] = n_components

    method_params = _validate_method_params(method_name, method_params)

    try:
        reducer: BaseProjectionMethod = method_cls(**method_params)
    except (TypeError, ValueError) as err:
        raise tk.ValidationError({"method_params": [f"Invalid {method_name} parameters: {err}"]}) from err

    x_matrix, prepare_info = _prepare_matrix_from_resource(resource, resource_view)
    _validate_matrix_compatibility(method_name, reducer.params, x_matrix)

    try:
        embedding = reducer.fit_transform(x_matrix)
    except (TypeError, ValueError) as err:
        raise tk.ValidationError({"data": [f"Cannot run {method_name}: {err}"]}) from err

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
        "n_components": resource_view.get("n_components"),
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


def _validate_resource_view_resource(resource: dict[str, Any], resource_view: dict[str, Any]) -> None:
    """Ensure a resource view is used only with its own resource."""
    view_resource_id = resource_view.get("resource_id")
    if view_resource_id != resource.get("id"):
        raise tk.ValidationError({"view_id": ["Resource view does not belong to the specified resource."]})


def _validate_method_params(method_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize method-specific parameters from resource view JSON."""
    allowed = METHOD_PARAM_NAMES[method_name]
    unknown = sorted(set(params).difference(allowed))
    if unknown:
        names = ", ".join(unknown)
        raise tk.ValidationError({"method_params": [f"Unsupported {method_name} parameter(s): {names}."]})

    normalized = dict(params)
    if "n_components" in normalized:
        normalized["n_components"] = _parse_n_components(normalized["n_components"])
    if "random_state" in normalized:
        normalized["random_state"] = _parse_non_negative_int(normalized["random_state"], "random_state")

    if method_name == "pca" and "whiten" in normalized:
        normalized["whiten"] = _parse_bool(normalized["whiten"], "whiten")
    if method_name == "tsne" and "perplexity" in normalized:
        normalized["perplexity"] = _parse_positive_float(normalized["perplexity"], "perplexity")
    if method_name == "umap":
        if "n_neighbors" in normalized:
            normalized["n_neighbors"] = _parse_int_at_least(normalized["n_neighbors"], "n_neighbors", 2)
        if "min_dist" in normalized:
            normalized["min_dist"] = _parse_float_between(normalized["min_dist"], "min_dist", 0, 1)

    return normalized


def _validate_matrix_compatibility(
    method_name: str,
    params: dict[str, Any],
    x_matrix: np.ndarray,
) -> None:
    """Reject method settings that cannot work with the prepared feature matrix."""
    n_rows, n_features = x_matrix.shape
    if n_rows < 2:  # noqa PLR2004
        raise tk.ValidationError({"data": ["At least 2 data rows are required for dimensionality reduction."]})

    n_components = params["n_components"]
    if method_name == "pca" and n_components > min(n_rows, n_features):
        raise tk.ValidationError(
            {"n_components": ["PCA n_components cannot exceed the number of rows or features."]}
        )
    if method_name == "tsne" and params["perplexity"] >= n_rows:
        raise tk.ValidationError({"method_params": ["t-SNE perplexity must be smaller than the number of rows."]})
    if method_name == "umap" and params["n_neighbors"] >= n_rows:
        raise tk.ValidationError({"method_params": ["UMAP n_neighbors must be smaller than the number of rows."]})


def _parse_bool(value: Any, name: str) -> bool:
    """Parse a strict JSON boolean parameter."""
    if isinstance(value, bool):
        return value
    raise tk.ValidationError({"method_params": [f"{name} must be a boolean."]})


def _parse_non_negative_int(value: Any, name: str) -> int:
    """Parse a non-negative integer method parameter."""
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise tk.ValidationError({"method_params": [f"{name} must be a non-negative integer."]})
    try:
        parsed = int(value)
    except (TypeError, ValueError) as err:
        raise tk.ValidationError({"method_params": [f"{name} must be a non-negative integer."]}) from err
    if parsed < 0:
        raise tk.ValidationError({"method_params": [f"{name} must be a non-negative integer."]})
    return parsed


def _parse_int_at_least(value: Any, name: str, minimum: int) -> int:
    """Parse an integer method parameter with a lower bound."""
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise tk.ValidationError({"method_params": [f"{name} must be an integer of at least {minimum}."]})
    try:
        parsed = int(value)
    except (TypeError, ValueError) as err:
        raise tk.ValidationError({"method_params": [f"{name} must be an integer of at least {minimum}."]}) from err
    if parsed < minimum:
        raise tk.ValidationError({"method_params": [f"{name} must be an integer of at least {minimum}."]})
    return parsed


def _parse_positive_float(value: Any, name: str) -> float:
    """Parse a strictly positive float method parameter."""
    parsed = _parse_float(value, name)
    if parsed <= 0:
        raise tk.ValidationError({"method_params": [f"{name} must be greater than 0."]})
    return parsed


def _parse_float_between(value: Any, name: str, minimum: float, maximum: float) -> float:
    """Parse a finite float method parameter within an inclusive range."""
    parsed = _parse_float(value, name)
    if not minimum <= parsed <= maximum:
        raise tk.ValidationError({"method_params": [f"{name} must be between {minimum:g} and {maximum:g}."]})
    return parsed


def _parse_float(value: Any, name: str) -> float:
    """Parse a finite float method parameter."""
    if isinstance(value, bool):
        raise tk.ValidationError({"method_params": [f"{name} must be a number."]})
    try:
        parsed = float(value)
    except (TypeError, ValueError) as err:
        raise tk.ValidationError({"method_params": [f"{name} must be a number."]}) from err
    if not np.isfinite(parsed):
        raise tk.ValidationError({"method_params": [f"{name} must be a finite number."]})
    return parsed


def _parse_n_components(raw_value: Any) -> int | None:
    """Parse n_components value (allow only 2 or 3)."""
    if raw_value in (None, ""):
        return None
    if isinstance(raw_value, bool) or (isinstance(raw_value, float) and not raw_value.is_integer()):
        raise tk.ValidationError({"n_components": ["n_components must be 2 or 3."]})
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError) as err:
        raise tk.ValidationError({"n_components": ["n_components must be an integer."]}) from err

    if parsed not in (2, 3):  # noqa PLR2004
        raise tk.ValidationError({"n_components": ["n_components must be 2 or 3."]})

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

    numeric_cols = _select_numeric_columns(df, color_by, selected_features)
    categorical_cols = _select_categorical_columns(df, numeric_cols, color_by, selected_features)
    if not numeric_cols:
        raise DimredNumericColumnError
    color_candidates = _build_color_candidates(df, color_by, numeric_cols, categorical_cols)

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
        "feature_columns": numeric_cols + categorical_cols,
        "color_candidates": color_candidates,
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
    if not color_by:
        return "", None
    if color_by not in df.columns:
        raise tk.ValidationError({"color_by": [f"Unknown color column: {color_by}."]})

    series = df[color_by]
    kind = _infer_color_kind(series)
    if kind == "numeric":
        values, _, _ = _serialize_numeric_values(series)
    else:
        values, _ = _serialize_categorical_values(series, dimred_config.max_categories_for_ohe())
    return color_by, values


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

    unknown = sorted(set(selected).difference(df.columns))
    if unknown:
        raise tk.ValidationError({"feature_columns": [f"Unknown feature column(s): {', '.join(unknown)}."]})
    return selected


def _select_numeric_columns(df: pd.DataFrame, color_by: str, selected_features: list[str]) -> list[str]:
    """Return numeric columns, excluding color_by unless explicitly selected."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if selected_features:
        numeric_cols = [c for c in numeric_cols if c in selected_features]
    elif color_by:
        numeric_cols = [c for c in numeric_cols if c != color_by]
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
        _raise_if_categorical_features_disabled(selected_features, numeric_cols)
        return categorical_cols

    max_cat = dimred_config.max_categories_for_ohe()
    for col in df.columns:
        if col in numeric_cols:
            continue
        if selected_features and col not in selected_features:
            continue
        if not selected_features and col == color_by:
            continue
        n_unique = df[col].nunique(dropna=True)
        if not 1 < n_unique <= max_cat:
            if selected_features:
                _raise_invalid_categorical_feature(col, n_unique, max_cat)
            continue
        categorical_cols.append(col)
    return categorical_cols


def _raise_if_categorical_features_disabled(selected_features: list[str], numeric_cols: list[str]) -> None:
    """Raise when an explicit categorical selection cannot be encoded."""
    selected_categorical = [col for col in selected_features if col not in numeric_cols]
    if selected_categorical:
        names = ", ".join(selected_categorical)
        raise tk.ValidationError({"feature_columns": [f"Categorical feature columns are disabled: {names}."]})


def _raise_invalid_categorical_feature(column: str, n_unique: int, max_categories: int) -> None:
    """Raise a field-specific error for an explicit unusable categorical column."""
    if n_unique <= 1:
        message = f"Feature column '{column}' must contain at least 2 distinct values."
    else:
        message = f"Feature column '{column}' has {n_unique} categories; maximum is {max_categories}."
    raise tk.ValidationError({"feature_columns": [message]})


def _build_color_candidates(
    df: pd.DataFrame,
    color_by: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> list[dict[str, Any]]:
    """Prepare color candidates metadata for the frontend dropdown."""
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    max_categories = max(dimred_config.max_categories_for_ohe(), 1)

    def add_candidate(name: str, force: bool = False) -> None:
        if not name or name in seen:
            return
        if name not in df.columns:
            return

        series = df[name]
        kind = _infer_color_kind(series)

        if kind == "categorical":
            n_unique = series.nunique(dropna=True)
            if not force and (n_unique <= 1 or n_unique > max_categories):
                return
            values, unique_values = _serialize_categorical_values(series, max_categories)
            candidates.append(
                {
                    "name": name,
                    "kind": "categorical",
                    "values": values,
                    "unique_values": unique_values,
                }
            )
        else:
            values, min_val, max_val = _serialize_numeric_values(series)
            if min_val is None or max_val is None:
                if not force:
                    return
            candidates.append(
                {
                    "name": name,
                    "kind": "numeric",
                    "values": values,
                    "min": min_val,
                    "max": max_val,
                }
            )

        seen.add(name)

    add_candidate(color_by, force=True)

    for col in categorical_cols:
        add_candidate(col)

    for col in numeric_cols:
        add_candidate(col)

    return candidates


def _infer_color_kind(series: pd.Series) -> str:
    """Return 'categorical' or 'numeric' for a pandas Series."""
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _serialize_categorical_values(series: pd.Series, unique_limit: int) -> tuple[list[Any], list[str]]:
    """Return safe values + limited unique list for categorical series."""
    values: list[Any] = []
    unique_values: list[str] = []
    seen_values: set[str] = set()

    for raw in series.tolist():
        if pd.isna(raw):
            values.append(None)
            continue

        val_str = str(raw)
        values.append(val_str)

        if val_str in seen_values:
            continue
        if len(unique_values) >= unique_limit:
            continue
        unique_values.append(val_str)
        seen_values.add(val_str)

    return values, unique_values


def _serialize_numeric_values(series: pd.Series) -> tuple[list[Any], float | None, float | None]:
    """Return numeric values + min/max for a series."""
    numeric = pd.to_numeric(series, errors="coerce")
    values = [None if pd.isna(v) else float(v) for v in numeric.tolist()]

    finite_values = [v for v in values if v is not None and np.isfinite(v)]
    if not finite_values:
        return values, None, None

    min_val = float(np.min(finite_values))
    max_val = float(np.max(finite_values))
    return values, min_val, max_val


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
