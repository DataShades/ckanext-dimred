from __future__ import annotations

from typing import Any

import numpy as np

import ckan.plugins as p
import ckan.plugins.toolkit as tk
from ckan import types
from ckan.common import CKANConfig

from ckanext.dimred import config as dimred_config
from ckanext.dimred import utils as dimred_utils
from ckanext.dimred.adapters import adapter_registry
from ckanext.dimred.exception import DimredError, DimredPreviewError
from ckanext.dimred.logic import schema


@tk.blanket.actions
@tk.blanket.config_declarations
@tk.blanket.helpers
@tk.blanket.validators
@tk.blanket.blueprints
class DimredPlugin(p.SingletonPlugin):
    p.implements(p.IConfigurable)
    p.implements(p.IConfigurer)
    p.implements(p.IResourceView, inherit=True)

    # IConfigurable

    def configure(self, config_: CKANConfig) -> None:
        self._register_adapters()

    @classmethod
    def _register_adapters(cls) -> None:
        """Allow other extensions to extend or override adapters via signals."""
        dimred_utils.collect_adapters_signal.send(adapter_registry)

    # IConfigurer

    def update_config(self, config_: CKANConfig):
        tk.add_template_directory(config_, "templates")
        tk.add_resource("assets", "dimred")

    # IResourceView

    def info(self) -> dict[str, Any]:
        return {
            "name": "dimred_view",
            "title": tk._("Dimensionality reduction"),
            "default_title": tk._("Dimensionality reduction"),
            "icon": "project-diagram",
            "iframed": False,
            "schema": schema.dimred_form_schema(),
        }

    def can_view(self, data_dict: types.DataDict) -> bool:
        """Determine whether dimred_view is applicable to a given resource."""
        resource = data_dict["resource"]
        return bool(resource.get("datastore_active")) or dimred_utils.get_adapter_for_resource(resource) is not None

    def setup_template_variables(self, context: types.Context, data_dict: types.DataDict) -> dict[str, Any]:
        """Prepare variables for the template."""
        resource = data_dict["resource"]
        resource_view = data_dict["resource_view"]
        render_backend = _render_backend(resource_view)

        return {
            "render_backend": render_backend,
            "resource": resource,
            "resource_view": resource_view,
            "package": data_dict["package"],
            **_preview_template_state(context, resource, resource_view, render_backend),
        }

    def view_template(self, context: types.Context, data_dict: types.DataDict) -> str:
        return "dimred/dimred_preview.html"

    def form_template(self, context: types.Context, data_dict: types.DataDict) -> str:
        return "dimred/dimred_form.html"


def _raise_if_error(result: dict[str, Any] | None) -> None:
    """Normalize and raise error from dimred_get_dimred_preview result."""
    if not result:
        raise DimredPreviewError
    if result.get("error"):
        raise DimredPreviewError(str(result["error"]))


def _render_backend(resource_view: types.DataDict) -> str:
    """Resolve the configured render backend for a resource view."""
    raw_backend = resource_view.get("render_backend") or dimred_config.render_backend()
    backend = str(raw_backend).strip() if raw_backend is not None else ""
    return backend or dimred_config.render_backend()


def _preview_template_state(
    context: types.Context,
    resource: types.DataDict,
    resource_view: types.DataDict,
    render_backend: str,
) -> dict[str, Any]:
    """Build the result or pending state without running a reducer in the request."""
    if not resource_view.get("id"):
        return _empty_preview_template_state()

    try:
        preview = tk.get_action("dimred_start_preview")(
            context,
            {"id": resource["id"], "view_id": resource_view["id"]},
        )
    except tk.ValidationError as exc:
        return _empty_preview_template_state(error=_format_validation_error(exc))
    except (DimredError, tk.NotAuthorized) as exc:
        return _empty_preview_template_state(error=str(exc))

    if preview["status"] == "ready":
        return _ready_preview_template_state(preview["result"], render_backend)
    if preview["status"] == "failed":
        return _empty_preview_template_state(error=str(preview.get("error") or tk._("Dimred preview failed.")))
    return _empty_preview_template_state(status=preview["status"], job_id=preview.get("job_id"))


def _ready_preview_template_state(result: dict[str, Any], render_backend: str) -> dict[str, Any]:
    """Format a completed job result for the resource-view template."""
    _raise_if_error(result)
    embedding = result["embedding"]
    meta = result["meta"]
    summary_raw = dimred_utils.embedding_summary(np.array(embedding), meta)
    summary = dimred_utils.build_display_summary(meta, summary_raw)
    image_data_url = None
    if render_backend == "matplotlib":
        image_data_url = dimred_utils.embedding_to_png_data_url(np.array(embedding), meta)
        embedding = None
    return {
        "image_data_url": image_data_url,
        "embedding": embedding,
        "meta": meta,
        "summary": summary,
        "error": None,
        "preview_status": "ready",
        "preview_job_id": None,
    }


def _empty_preview_template_state(
    status: str | None = None,
    job_id: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Return a template state with no embedding payload."""
    return {
        "image_data_url": None,
        "embedding": None,
        "meta": {},
        "summary": {},
        "error": error,
        "preview_status": "failed" if error else status,
        "preview_job_id": job_id,
    }


def _format_validation_error(error: tk.ValidationError) -> str:
    """Return validation messages suitable for rendering in the resource view."""
    messages: list[str] = []
    for field_errors in error.error_dict.values():
        if isinstance(field_errors, list):
            messages.extend(str(message) for message in field_errors)
        else:
            messages.append(str(field_errors))
    return " ".join(messages) or tk._("Invalid dimensionality reduction settings.")
