from __future__ import annotations

from ckanext.dimred.methods.base import BaseProjectionMethod
from ckanext.dimred.methods.pca import PCAProjection
from ckanext.dimred.methods.tsne import TSNEProjection
from ckanext.dimred.methods.umap import UMAPProjection

PROJECTION_METHODS: dict[str, type[BaseProjectionMethod]] = {
    UMAPProjection.name: UMAPProjection,
    TSNEProjection.name: TSNEProjection,
    PCAProjection.name: PCAProjection,
}


def get_projection_method(name: str) -> type[BaseProjectionMethod]:
    """Return the projection method class for the given name."""
    try:
        return PROJECTION_METHODS[name]
    except KeyError:
        raise KeyError from None


__all__ = [
    "BaseProjectionMethod",
    "UMAPProjection",
    "TSNEProjection",
    "PCAProjection",
    "PROJECTION_METHODS",
    "get_projection_method",
]
