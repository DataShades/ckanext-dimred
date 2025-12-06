from __future__ import annotations

from typing import Any

import numpy as np

import ckan.plugins as p
import ckan.plugins.toolkit as tk
from ckan import types
from ckan.common import CKANConfig

from ckanext.dimred import utils as dimred_utils
from ckanext.dimred.adapters import adapter_registry
from ckanext.dimred.exception import DimredError, DimredPreviewError
from ckanext.dimred.logic import schema
from ckanext.dimred.utils import cache as dimred_cache


@tk.blanket.actions
@tk.blanket.config_declarations
@tk.blanket.helpers
@tk.blanket.auth_functions
@tk.blanket.validators
class DimredPlugin(p.SingletonPlugin):
    p.implements(p.IConfigurable)
    p.implements(p.IConfigurer)
    p.implements(p.IResourceView, inherit=True)
    p.implements(p.IResourceController, inherit=True)

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
        tk.add_public_directory(config_, "public")
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
        return dimred_utils.get_adapter_for_resource(resource) is not None

    def setup_template_variables(self, context: types.Context, data_dict: types.DataDict) -> dict[str, Any]:
        """Prepare variables for the template."""
        resource = data_dict["resource"]
        resource_view = data_dict["resource_view"]

        try:
            result = tk.get_action("dimred_get_dimred_preview")(
                context,
                {"id": resource["id"], "view_id": resource_view["id"]},
            )

            _raise_if_error(result)

            embedding = np.array(result["embedding"])
            meta = result["meta"]

            image_data_url = dimred_utils.embedding_to_png_data_url(embedding, meta)

            error = None
        except (DimredError, tk.ValidationError) as exc:
            image_data_url = None
            meta = {}
            error = str(exc)

        return {
            "image_data_url": image_data_url,
            "meta": meta,
            "error": error,
            "resource": resource,
            "resource_view": resource_view,
            "package": data_dict["package"],
        }

    def view_template(self, context: types.Context, data_dict: types.DataDict) -> str:
        return "dimred/dimred_preview.html"

    def form_template(self, context: types.Context, data_dict: types.DataDict) -> str:
        return "dimred/dimred_form.html"

    # IResourceController

    def before_resource_update(self, context: types.Context, current: dict[str, Any], resource: dict[str, Any]):
        if _resource_data_changed(current, resource):
            cache = dimred_cache.get_cache()
            cache.delete_for_resource(current["id"])

    def before_resource_delete(self, context: types.Context, resource: dict[str, Any]):
        cache = dimred_cache.get_cache()
        cache.delete_for_resource(resource["id"])


def _raise_if_error(result: dict[str, Any] | None) -> None:
    """Normalize and raise error from dimred_get_dimred_preview result."""
    if not result:
        raise DimredPreviewError
    if result.get("error"):
        raise DimredPreviewError(str(result["error"]))


def _resource_data_changed(current: dict[str, Any] | None, resource: dict[str, Any]) -> bool:
    """Return True if the resource file/URL changed (not just metadata)."""
    if resource.get("upload") or resource.get("upload_file"):
        return True
    current_url = (current or {}).get("url")
    new_url = resource.get("url")
    if new_url is None and current_url is None:
        return False
    return new_url != current_url
