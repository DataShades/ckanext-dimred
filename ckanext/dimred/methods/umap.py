from __future__ import annotations

import logging
from typing import Any

import numpy as np
import umap

from ckanext.dimred import config as dimred_config
from ckanext.dimred.methods.base import BaseProjectionMethod

log = logging.getLogger(__name__)


class UMAPProjection(BaseProjectionMethod):
    """Wrapper around umap-learn."""

    name = "umap"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

        self._reducer = umap.UMAP(
            n_neighbors=self.params["n_neighbors"],
            min_dist=self.params["min_dist"],
            n_components=self.params["n_components"],
            random_state=self.params.get("random_state", 42),
        )

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        """Return default parameters for the UMAP method."""
        return {
            "n_neighbors": dimred_config.umap_n_neighbors(),
            "min_dist": dimred_config.umap_min_dist(),
            "n_components": dimred_config.umap_n_components(),
            "random_state": 42,
        }

    def fit_transform(self, x_matrix: np.ndarray):
        """Run UMAP and return the embedding matrix."""
        return self._reducer.fit_transform(x_matrix)
