from __future__ import annotations

from importlib import import_module

from ckanext.dimred.methods.base import BaseProjectionMethod

PROJECTION_METHODS = {
    "umap": ("ckanext.dimred.methods.umap", "UMAPProjection"),
    "tsne": ("ckanext.dimred.methods.tsne", "TSNEProjection"),
    "pca": ("ckanext.dimred.methods.pca", "PCAProjection"),
}


def get_projection_method(name: str) -> type[BaseProjectionMethod]:
    """Import and return only the requested projection method class."""
    try:
        module_name, class_name = PROJECTION_METHODS[name]
    except KeyError:
        raise KeyError from None
    return getattr(import_module(module_name), class_name)


__all__ = [
    "BaseProjectionMethod",
    "PROJECTION_METHODS",
    "get_projection_method",
]
