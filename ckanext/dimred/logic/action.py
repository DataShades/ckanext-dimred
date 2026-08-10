from __future__ import annotations

import json
import re
from hashlib import sha256
from typing import Any, cast

import numpy as np
import pandas as pd
from redis import exceptions as redis_exc
from sklearn.preprocessing import StandardScaler

from ckan.lib import jobs as ckan_jobs
from ckan.logic import validate
from ckan.plugins import toolkit as tk
from ckan.types import Context, DataDict

from ckanext.dimred import config as dimred_config
from ckanext.dimred import utils as dimred_utils
from ckanext.dimred.exception import (
    DimredAdapterNotFoundError,
    DimredDatastoreError,
    DimredError,
    DimredFeatureError,
    DimredNumericColumnError,
    DimredPreviewError,
    DimredPreviewPayloadError,
    DimredRemoteFetchError,
)
from ckanext.dimred.logic import schema
from ckanext.dimred.methods import BaseProjectionMethod, get_projection_method
from ckanext.dimred.utils import cache as dimred_cache
from ckanext.dimred.utils.export import embedding_to_csv

__all__ = [
    "dimred_start_preview",
    "dimred_get_preview_status",
    "dimred_get_dimred_color_values",
    "dimred_export_embedding",
]

METHOD_PARAM_NAMES = {
    "pca": {"n_components", "random_state", "whiten"},
    "tsne": {"n_components", "perplexity", "random_state"},
    "umap": {"min_dist", "n_components", "n_neighbors", "random_state"},
}

PIPELINE_SCHEMA_VERSION = 1
PREVIEW_QUEUE = "dimred"
PREVIEW_JOB_RESULT_TTL = 3600
PREVIEW_ENQUEUE_LOCK_TTL = 60

DATE_LIKE_PATTERN = re.compile(
    r"(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
    r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})(?:[T\s].*)?"
)


@validate(schema.dimred_get_dimred_preview_schema)
def _compute_dimred_preview(context: Context, data_dict: DataDict) -> dict[str, Any]:
    """Build a preview for the background worker; this is not a public action."""
    resource, resource_view = _get_preview_resource_and_view(context, data_dict)
    return dimred_run_dimred_pipeline(
        context,
        {
            "resource": resource,
            "resource_view": resource_view,
        },
    )


@validate(schema.dimred_start_preview_schema)
def dimred_start_preview(context: Context, data_dict: DataDict) -> dict[str, Any]:
    """Return a cached preview or enqueue a CKAN background job to build it."""
    resource, resource_view = _get_preview_resource_and_view(context, data_dict)
    resource_id = resource["id"]
    resource_view_id = resource_view["id"]
    cache = dimred_cache.get_cache()
    settings_sig = cache.settings_signature(_cache_settings(resource, resource_view))
    cacheable = _is_cacheable_resource(resource)

    cached = cache.get(resource_id, resource_view_id, settings_sig) if cacheable else None
    if cached:
        return {"status": "ready", "result": cached}

    if cache.client is None:
        raise DimredPreviewError

    job_id = _preview_job_id(resource_id, resource_view_id, settings_sig)
    existing = _preview_job(job_id, resource_id, resource_view_id, settings_sig)
    if existing is not None:
        response = _preview_job_response(existing)
        if response["status"] != "failed" or not response.get("retryable"):
            return response
        try:
            existing.delete()
        except redis_exc.RedisError as err:
            raise DimredPreviewError from err

    try:
        lock_acquired = cache.acquire_job_lock(job_id, PREVIEW_ENQUEUE_LOCK_TTL)
    except redis_exc.RedisError as err:
        raise DimredPreviewError from err
    if not lock_acquired:
        return {"status": "pending", "job_id": job_id}

    try:
        job = tk.enqueue_job(
            _run_preview_job,
            args=[resource_id, resource_view_id, str(context.get("user") or "")],
            title="Dimensionality reduction preview",
            queue=PREVIEW_QUEUE,
            rq_kwargs={
                "job_id": job_id,
                "result_ttl": PREVIEW_JOB_RESULT_TTL,
                "failure_ttl": PREVIEW_JOB_RESULT_TTL,
                "meta": {
                    "dimred_resource_id": resource_id,
                    "dimred_view_id": resource_view_id,
                    "dimred_settings_signature": settings_sig,
                },
            },
        )
    except Exception as err:
        cache.release_job_lock(job_id)
        raise DimredPreviewError from err

    cache.release_job_lock(job_id)
    return _preview_job_response(job)


@tk.side_effect_free
@validate(schema.dimred_get_preview_status_schema)
def dimred_get_preview_status(context: Context, data_dict: DataDict) -> dict[str, Any]:
    """Return the authorized application-level status of a dimred preview job."""
    resource, resource_view = _get_preview_resource_and_view(context, data_dict)
    resource_id = resource["id"]
    resource_view_id = resource_view["id"]
    job_id = data_dict["job_id"]
    cache = dimred_cache.get_cache()
    settings_sig = cache.settings_signature(_cache_settings(resource, resource_view))
    job = _preview_job(job_id, resource_id, resource_view_id, settings_sig)
    if job is None:
        # A parallel request can observe the enqueue lock before RQ stores the job.
        return {"status": "pending", "job_id": job_id}

    return _preview_job_response(job)


@tk.side_effect_free
@validate(schema.dimred_get_dimred_color_values_schema)
def dimred_get_dimred_color_values(context: Context, data_dict: DataDict) -> dict[str, Any]:
    """Return values for one color candidate aligned with a dimred embedding."""
    resource = tk.get_action("resource_show")(context, {"id": data_dict["id"]})
    resource_view = tk.get_action("resource_view_show")(context, {"id": data_dict["view_id"]})
    _validate_resource_view_resource(resource, resource_view)

    method_name, _, _ = _resolve_projection_settings(resource_view)
    row_limit = dimred_config.effective_max_rows(method_name)
    df, source_row_ids, _ = _load_dataframe(context, resource, resource_view, row_limit)
    df, source_row_ids = _maybe_limit_rows(df, source_row_ids, row_limit)

    candidate = _color_candidate_for_column(df, resource_view, data_dict["column"])
    if candidate is None:
        raise tk.ValidationError({"column": [f"Column '{data_dict['column']}' is not available for coloring."]})

    series = cast(pd.Series, df[candidate["name"]])
    values = _serialize_color_values(series, candidate["kind"])
    return {
        "column": candidate["name"],
        "kind": candidate["kind"],
        "values": values,
        "source_row_ids": source_row_ids,
    }


@tk.side_effect_free
def dimred_run_dimred_pipeline(context: Context, data_dict: DataDict) -> dict[str, Any]:
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

    settings = _cache_settings(resource, resource_view)
    cache = dimred_cache.get_cache()
    settings_sig = cache.settings_signature(settings)
    cacheable = _is_cacheable_resource(resource)

    cached = cache.get(resource_id, resource_view_id, settings_sig) if cacheable else None
    if cached:
        return cached

    embedding, meta = _build_dimred_preview(resource, resource_view, context)
    decimals = dimred_config.embedding_decimals()
    embedding = np.round(np.asarray(embedding, dtype=float), decimals)
    embedding_serializable = embedding.tolist()

    result = {"embedding": embedding_serializable, "meta": meta}
    _validate_preview_payload_size(result)
    if cacheable:
        cache.save(resource_id, resource_view_id, settings_sig, result)

    return result


def _get_preview_resource_and_view(
    context: Context,
    data_dict: DataDict,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate the resource/view pair before creating a preview job."""
    resource = tk.get_action("resource_show")(context, {"id": data_dict["id"]})
    resource_view = tk.get_action("resource_view_show")(context, {"id": data_dict["view_id"]})
    _validate_resource_view_resource(resource, resource_view)

    resource_view = dict(resource_view)
    resource_view["method_params"] = _parse_method_params(resource_view.get("method_params"))
    _resolve_projection_settings(resource_view)
    return resource, resource_view


def _preview_job_id(resource_id: str, resource_view_id: str, settings_sig: str) -> str:
    """Build a deterministic RQ job ID scoped to the current CKAN site."""
    site_id = str(tk.config.get("ckan.site_id", "default"))
    value = ":".join((site_id, resource_id, resource_view_id, settings_sig))
    return f"dimred-preview-{sha256(value.encode('utf-8')).hexdigest()}"


def _preview_job(job_id: str, resource_id: str, resource_view_id: str, settings_sig: str) -> Any | None:
    """Return a matching dimred job without exposing unrelated CKAN jobs."""
    try:
        job = ckan_jobs.job_from_id(job_id)
    except KeyError:
        return None
    except redis_exc.RedisError as err:
        raise DimredPreviewError from err

    meta = job.meta or {}
    if (
        meta.get("dimred_resource_id") != resource_id
        or meta.get("dimred_view_id") != resource_view_id
        or meta.get("dimred_settings_signature") != settings_sig
    ):
        return None
    return job


def _preview_job_response(job: Any) -> dict[str, Any]:
    """Map RQ state to the small public status contract used by the view."""
    try:
        status = str(job.get_status(refresh=True))
    except redis_exc.RedisError as err:
        raise DimredPreviewError from err
    if status == "finished":
        try:
            return_value = job.return_value
            result = return_value() if callable(return_value) else return_value
        except redis_exc.RedisError as err:
            raise DimredPreviewError from err
        if isinstance(result, dict) and "embedding" in result and "meta" in result:
            return {"status": "ready", "job_id": job.id, "result": result}
        if isinstance(result, dict) and isinstance(result.get("error"), str):
            return {
                "status": "failed",
                "job_id": job.id,
                "error": result["error"],
                "retryable": bool(result.get("retryable")),
            }
        return {"status": "failed", "job_id": job.id, "error": "Dimred preview failed.", "retryable": True}
    if status in {"failed", "stopped", "canceled"}:
        return {"status": "failed", "job_id": job.id, "error": "Dimred preview failed.", "retryable": True}
    if status == "started":
        return {"status": "running", "job_id": job.id}
    return {"status": "pending", "job_id": job.id}


def _run_preview_job(resource_id: str, resource_view_id: str, user: str) -> dict[str, Any]:
    """Execute a preview in an RQ worker using the submitter's CKAN identity."""
    try:
        return _compute_dimred_preview({"user": user}, {"id": resource_id, "view_id": resource_view_id})
    except tk.ValidationError as err:
        return {"error": _validation_error_message(err), "retryable": False}
    except DimredError as err:
        return {"error": str(err), "retryable": isinstance(err, DimredRemoteFetchError)}
    except tk.NotAuthorized:
        return {"error": "Dimred preview failed.", "retryable": False}


def _validation_error_message(error: tk.ValidationError) -> str:
    """Flatten a validation error without relying on CKAN's UI-only summary."""
    messages: list[str] = []
    for values in error.error_dict.values():
        if isinstance(values, list):
            messages.extend(str(value) for value in values)
        else:
            messages.append(str(values))
    return " ".join(messages) or "Dimred preview failed."


@tk.side_effect_free
@validate(schema.dimred_export_embedding_schema)
def dimred_export_embedding(context: Context, data_dict: DataDict) -> dict[str, Any]:
    """Return CSV export for a dimred preview."""
    if not dimred_config.export_enabled():
        raise tk.ValidationError({"export": ["Dimred export is disabled."]})

    preview = _existing_preview(context, data_dict)
    if preview["status"] != "ready":
        raise tk.ValidationError({"preview": ["Dimred preview is still being prepared."]})

    result = preview["result"]

    csv_content = embedding_to_csv(result["embedding"], result["meta"])
    resource_id = data_dict["id"]
    view_id = data_dict["view_id"]
    filename = f"dimred-{resource_id}-{view_id}.csv"

    return {
        "filename": filename,
        "content": csv_content,
        "content_type": "text/csv; charset=utf-8",
    }


def _existing_preview(context: Context, data_dict: DataDict) -> dict[str, Any]:
    """Read a cached or completed preview without enqueueing background work."""
    resource, resource_view = _get_preview_resource_and_view(context, data_dict)
    resource_id = resource["id"]
    resource_view_id = resource_view["id"]
    cache = dimred_cache.get_cache()
    settings_sig = cache.settings_signature(_cache_settings(resource, resource_view))
    cacheable = _is_cacheable_resource(resource)

    cached = cache.get(resource_id, resource_view_id, settings_sig) if cacheable else None
    if cached:
        return {"status": "ready", "result": cached}

    job_id = _preview_job_id(resource_id, resource_view_id, settings_sig)
    job = _preview_job(job_id, resource_id, resource_view_id, settings_sig)
    if job is None:
        return {"status": "pending", "job_id": job_id}
    return _preview_job_response(job)


def _build_dimred_preview(
    resource: dict[str, Any],
    resource_view: dict[str, Any],
    context: Context,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the dimred pipeline for a given resource + view."""
    method_name, method_cls, method_params = _resolve_projection_settings(resource_view)

    try:
        reducer: BaseProjectionMethod = method_cls(**method_params)
    except (TypeError, ValueError) as err:
        raise tk.ValidationError({"method_params": [f"Invalid {method_name} parameters: {err}"]}) from err

    row_limit = dimred_config.effective_max_rows(method_name)
    x_matrix, prepare_info = _prepare_matrix_from_resource(context, resource, resource_view, row_limit)
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


def _resolve_projection_settings(
    resource_view: dict[str, Any],
) -> tuple[str, type[BaseProjectionMethod], dict[str, Any]]:
    """Validate projection settings and merge explicit parameters with defaults."""
    method_name = (resource_view.get("method") or "").strip() or dimred_config.default_method()
    if method_name not in set(dimred_config.allowed_methods()):
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
    effective_params = {
        **method_cls.default_params(),
        **{key: value for key, value in method_params.items() if value is not None},
    }
    return method_name, method_cls, effective_params


def _cache_settings(resource: dict[str, Any], resource_view: dict[str, Any]) -> dict[str, Any]:
    """Build settings dict that affects cache identity."""
    method_name, _, effective_params = _resolve_projection_settings(resource_view)
    return {
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "method": method_name,
        "method_params": effective_params,
        "feature_columns": resource_view.get("feature_columns"),
        "color_by": resource_view.get("color_by"),
        "effective_max_rows": dimred_config.effective_max_rows(method_name),
        "enable_categorical": dimred_config.enable_categorical(),
        "max_categories_for_ohe": dimred_config.max_categories_for_ohe(),
        "max_preview_payload_bytes": dimred_config.max_preview_payload_bytes(),
        "max_color_candidates": dimred_config.max_color_candidates(),
        "embedding_decimals": dimred_config.embedding_decimals(),
        "resource_fingerprint": _resource_cache_fingerprint(resource),
    }


def _is_cacheable_resource(resource: dict[str, Any]) -> bool:
    """Return whether a resource has a reliable cache revision fingerprint."""
    return resource.get("url_type") == "upload" and not resource.get("datastore_active")


def _resource_cache_fingerprint(resource: dict[str, Any]) -> dict[str, Any]:
    """Return CKAN resource metadata that changes when an upload source changes."""
    return {
        "id": resource["id"],
        "url_type": resource.get("url_type"),
        "url": resource.get("url"),
        "format": resource.get("format"),
        "last_modified": resource.get("last_modified"),
        "size": resource.get("size"),
        "hash": resource.get("hash"),
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
    context: Context,
    resource: dict[str, Any],
    resource_view: dict[str, Any],
    row_limit: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a tabular resource, select suitable columns and return a feature matrix.

    Features:
    - numeric columns are always included;
    - optional low-cardinality categorical columns are one-hot encoded
      if enabled in config;
    - optional 'color_by' column is passed through to metadata.
    """
    df, source_row_ids, n_rows_original = _load_dataframe(context, resource, resource_view, row_limit)
    df, source_row_ids = _maybe_limit_rows(df, source_row_ids, row_limit)

    color_by, color_values = _extract_color_info(df, resource_view)
    selected_features = _extract_selected_features(df, resource_view)

    numeric_cols, categorical_cols, skipped_columns = _select_feature_columns(df, color_by, selected_features)
    if not numeric_cols:
        raise DimredNumericColumnError
    all_color_candidates = _build_color_candidates(df, color_by, numeric_cols, categorical_cols)
    color_candidates, omitted_color_candidates = _limit_color_candidates(all_color_candidates)

    df_features = _build_feature_frame(df, numeric_cols, categorical_cols)

    try:
        x_matrix = StandardScaler().fit_transform(df_features.values)
    except (TypeError, ValueError) as err:
        raise tk.ValidationError({"data": [f"Cannot prepare feature data: {err}"]}) from err
    if not np.isfinite(x_matrix).all():
        raise tk.ValidationError({"data": ["Feature data must contain only finite values."]})

    info: dict[str, Any] = {
        "n_rows_original": n_rows_original,
        "n_rows_used": len(df),
        "row_limit": row_limit,
        "source_row_ids": source_row_ids,
        "n_features": x_matrix.shape[1],
        "numeric_used": numeric_cols,
        "categorical_used": categorical_cols,
        "color_by": color_by or None,
        "color_values": color_values,
        "feature_columns": numeric_cols + categorical_cols,
        "skipped_columns": skipped_columns,
        "color_candidates": color_candidates,
        "color_candidates_total": len(all_color_candidates),
        "color_candidates_truncated": omitted_color_candidates,
    }

    return x_matrix, info


def _load_dataframe(
    context: Context,
    resource: dict[str, Any],
    resource_view: dict[str, Any],
    row_limit: int,
) -> tuple[pd.DataFrame, list[int], int]:
    """Load a dataframe and stable source row IDs through CKAN's supported APIs."""
    if resource.get("datastore_active"):
        return _load_datastore_dataframe(context, resource, row_limit)

    adapter_cls = dimred_utils.get_adapter_for_resource(resource)
    if adapter_cls is None:
        res_format = (resource.get("format") or "").lower()
        raise DimredAdapterNotFoundError(res_format)

    adapter = adapter_cls(resource, resource_view)
    df, source_row_ids, n_rows_original = adapter.get_sampled_dataframe(row_limit)
    df = df.reset_index(drop=True)

    if df.empty:
        raise DimredFeatureError

    return df, source_row_ids, n_rows_original


def _load_datastore_dataframe(
    context: Context,
    resource: dict[str, Any],
    row_limit: int,
) -> tuple[pd.DataFrame, list[int], int]:
    """Load DataStore records through its authorized action and preserve CKAN `_id`."""
    try:
        search = tk.get_action("datastore_search")
    except KeyError as err:
        raise DimredDatastoreError from err

    result = search(
        context,
        {
            "resource_id": resource["id"],
            "limit": row_limit,
            "include_total": True,
        },
    )
    records = result.get("records", [])
    if not records:
        raise DimredFeatureError

    source_row_ids = [record.get("_id") for record in records]
    if any(isinstance(row_id, bool) or not isinstance(row_id, int) for row_id in source_row_ids):
        raise DimredDatastoreError

    df = pd.DataFrame([{key: value for key, value in record.items() if key != "_id"} for record in records])
    return df.reset_index(drop=True), source_row_ids, result.get("total", len(df))


def _maybe_limit_rows(
    df: pd.DataFrame,
    source_row_ids: list[int],
    row_limit: int,
) -> tuple[pd.DataFrame, list[int]]:
    """Apply the effective row limit while retaining each sampled row's source ID."""
    if len(df) <= row_limit:
        return df, source_row_ids

    sampled = df.sample(row_limit, random_state=42)
    sampled_row_ids = [source_row_ids[position] for position in sampled.index]
    return sampled.reset_index(drop=True), sampled_row_ids


def _extract_color_info(df: pd.DataFrame, resource_view: dict[str, Any]) -> tuple[str, list[Any] | None]:
    """Extract color_by and corresponding values."""
    color_by = _color_by_name(resource_view)
    if not color_by:
        return "", None
    if color_by not in df.columns:
        raise tk.ValidationError({"color_by": [f"Unknown color column: {color_by}."]})

    series = cast(pd.Series, df[color_by])
    kind = _infer_color_kind(series)
    return color_by, _serialize_color_values(series, kind)


def _color_by_name(resource_view: dict[str, Any]) -> str:
    """Return the normalized color column selected in the resource view."""
    return (resource_view.get("color_by") or "").strip()


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
        elif isinstance(raw_features, list | tuple | set):
            selected = [str(v) for v in raw_features]

    unknown = sorted(set(selected).difference(df.columns))
    if unknown:
        raise tk.ValidationError({"feature_columns": [f"Unknown feature column(s): {', '.join(unknown)}."]})
    return selected


def _select_feature_columns(
    df: pd.DataFrame,
    color_by: str,
    selected_features: list[str],
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """Classify selected columns and return usable features plus automatic skips."""
    explicit_selection = bool(selected_features)
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    skipped_columns: list[dict[str, str]] = []

    for col in df.columns:
        if explicit_selection and col not in selected_features:
            continue
        if not explicit_selection and col == color_by:
            skipped_columns.append({"name": col, "reason": "used for color only"})
            continue

        series = cast(pd.Series, df[col])
        kind = _classify_feature_column(series)
        if kind == "numeric":
            numeric_cols.append(col)
        elif kind == "categorical":
            _select_categorical_feature(series, col, explicit_selection, categorical_cols, skipped_columns)
        else:
            _skip_or_raise_feature(
                col,
                kind,
                explicit_selection,
                skipped_columns,
                _unsupported_feature_message(col, kind),
            )

    return numeric_cols, categorical_cols, skipped_columns


def _classify_feature_column(series: pd.Series) -> str:
    """Classify a column for feature selection without coercing mixed data."""
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    non_null = series.dropna()
    if non_null.empty:
        return "empty"

    numeric = cast(pd.Series, pd.to_numeric(series, errors="coerce"))
    finite = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    non_null_numeric = cast(pd.Series, pd.to_numeric(non_null, errors="coerce"))
    if pd.api.types.is_numeric_dtype(series) or non_null_numeric.notna().all():
        return "numeric" if not finite.empty else "empty"
    if non_null_numeric.notna().any():
        return "mixed"
    return "datetime" if _looks_like_datetime(non_null) else "categorical"


def _looks_like_datetime(values: pd.Series) -> bool:
    """Recognize common date strings without guessing arbitrary labels."""
    date_strings = values.astype(str).str.strip()
    if not date_strings.str.fullmatch(DATE_LIKE_PATTERN).all():
        return False
    return bool(pd.to_datetime(date_strings, errors="coerce", format="mixed").notna().all())


def _select_categorical_feature(
    series: pd.Series,
    column: str,
    explicit_selection: bool,
    categorical_cols: list[str],
    skipped_columns: list[dict[str, str]],
) -> None:
    """Add an eligible categorical feature or report why it cannot be used."""
    if not dimred_config.enable_categorical():
        _skip_or_raise_feature(
            column,
            "categorical-disabled",
            explicit_selection,
            skipped_columns,
            f"Feature column '{column}' cannot be used because categorical features are disabled.",
        )
        return

    n_unique = series.nunique(dropna=True)
    max_categories = dimred_config.max_categories_for_ohe()
    if n_unique <= 1:
        _skip_or_raise_feature(
            column,
            "fewer than 2 distinct values",
            explicit_selection,
            skipped_columns,
            f"Feature column '{column}' must contain at least 2 distinct values.",
        )
        return
    if n_unique > max_categories:
        _skip_or_raise_feature(
            column,
            "too many categories",
            explicit_selection,
            skipped_columns,
            f"Feature column '{column}' has {n_unique} categories; maximum is {max_categories}.",
        )
        return

    categorical_cols.append(column)


def _skip_or_raise_feature(
    column: str,
    reason: str,
    explicit_selection: bool,
    skipped_columns: list[dict[str, str]],
    message: str,
) -> None:
    """Raise for an explicit unusable column, otherwise record an automatic skip."""
    if explicit_selection:
        raise tk.ValidationError({"feature_columns": [message]})
    skipped_columns.append({"name": column, "reason": reason})


def _unsupported_feature_message(column: str, kind: str) -> str:
    """Return a short field-specific error for unsupported feature data."""
    messages = {
        "empty": f"Feature column '{column}' has no finite values.",
        "mixed": f"Feature column '{column}' contains mixed numeric and text values.",
        "datetime": f"Feature column '{column}' is a datetime column, which is not supported.",
    }
    return messages[kind]


def _build_color_candidates(
    df: pd.DataFrame,
    color_by: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> list[dict[str, Any]]:
    """Prepare compact color candidate descriptors for the frontend dropdown."""
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    max_categories = max(dimred_config.max_categories_for_ohe(), 1)

    def add_candidate(name: str, force: bool = False) -> None:
        if not name or name in seen:
            return
        if name not in df.columns:
            return

        series = cast(pd.Series, df[name])
        kind = _infer_color_kind(series)

        if kind == "categorical":
            n_unique = series.nunique(dropna=True)
            if not force and (n_unique <= 1 or n_unique > max_categories):
                return
            candidates.append(
                {
                    "name": name,
                    "kind": "categorical",
                    "unique_values": _categorical_unique_values(series, max_categories),
                }
            )
        else:
            min_val, max_val = _numeric_range(series)
            if (min_val is None or max_val is None) and not force:
                return
            candidates.append(
                {
                    "name": name,
                    "kind": "numeric",
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


def _color_candidate_for_column(
    df: pd.DataFrame,
    resource_view: dict[str, Any],
    column: str,
) -> dict[str, Any] | None:
    """Return a color descriptor only when the column is available in the preview."""
    selected_features = _extract_selected_features(df, resource_view)
    numeric_cols, categorical_cols, _ = _select_feature_columns(df, _color_by_name(resource_view), selected_features)
    candidates, _ = _limit_color_candidates(
        _build_color_candidates(df, _color_by_name(resource_view), numeric_cols, categorical_cols)
    )
    return next((candidate for candidate in candidates if candidate["name"] == column), None)


def _limit_color_candidates(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep the configured number of ordered color candidates for the preview."""
    limit = dimred_config.max_color_candidates()
    return candidates[:limit], max(0, len(candidates) - limit)


def _validate_preview_payload_size(result: dict[str, Any]) -> None:
    """Reject a completed result that cannot fit within the browser output budget."""
    payload = dimred_cache.serialize_preview_result(result).encode("utf-8")
    if len(payload) > dimred_config.max_preview_payload_bytes():
        raise DimredPreviewPayloadError


def _infer_color_kind(series: pd.Series) -> str:
    """Return 'categorical' or 'numeric' for a pandas Series."""
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _serialize_color_values(series: pd.Series, kind: str) -> list[Any]:
    """Return safe color values for a single selected column."""
    if kind == "numeric":
        return _serialize_numeric_values(series)

    return [None if pd.isna(raw) else str(raw) for raw in series.tolist()]


def _categorical_unique_values(series: pd.Series, unique_limit: int) -> list[str]:
    """Return a bounded ordered list of categorical values without a row vector."""
    unique_values: list[str] = []
    seen_values: set[str] = set()

    for raw in series.tolist():
        if pd.isna(raw):
            continue

        val_str = str(raw)
        if val_str in seen_values:
            continue
        if len(unique_values) >= unique_limit:
            continue
        unique_values.append(val_str)
        seen_values.add(val_str)

    return unique_values


def _serialize_numeric_values(series: pd.Series) -> list[Any]:
    """Return safe numeric values for a single selected color column."""
    numeric = cast(pd.Series, pd.to_numeric(series, errors="coerce"))
    return [None if pd.isna(v) or not np.isfinite(v) else float(v) for v in numeric.tolist()]


def _numeric_range(series: pd.Series) -> tuple[float | None, float | None]:
    """Return finite numeric bounds for a candidate descriptor."""
    numeric = cast(pd.Series, pd.to_numeric(series, errors="coerce"))
    finite_values = [float(v) for v in numeric.tolist() if not pd.isna(v) and np.isfinite(v)]
    if not finite_values:
        return None, None

    return float(np.min(finite_values)), float(np.max(finite_values))


def _build_feature_frame(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    """Assemble a finite feature frame with numeric imputation and categorical encoding."""
    try:
        feature_cols = numeric_cols + categorical_cols
        df_features = cast(pd.DataFrame, df[feature_cols].copy())
        if numeric_cols:
            numeric_features = cast(pd.DataFrame, df_features[numeric_cols])
            numeric = cast(pd.DataFrame, numeric_features.apply(pd.to_numeric, errors="coerce"))
            df_features[numeric_cols] = numeric.replace([np.inf, -np.inf], np.nan)
        if categorical_cols:
            df_features = cast(
                pd.DataFrame,
                pd.get_dummies(df_features, columns=categorical_cols, dummy_na=False, drop_first=False),
            )
        df_features = df_features.astype(float)
        df_features = df_features.fillna(df_features.mean())
        df_features = df_features.fillna(0.0)
    except (TypeError, ValueError) as err:
        raise tk.ValidationError({"data": [f"Cannot prepare feature data: {err}"]}) from err

    if df_features.shape[1] < 2:  # noqa PLR2004
        raise DimredFeatureError
    if not np.isfinite(df_features.to_numpy(dtype=float)).all():
        raise tk.ValidationError({"data": ["Feature data must contain only finite values."]})

    return df_features
