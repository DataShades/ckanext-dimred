from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from ckanext.dimred import config as dimred_config
from ckanext.dimred.methods.base import BaseProjectionMethod


class PCAProjection(BaseProjectionMethod):
    """Wrapper around sklearn.decomposition.PCA."""

    name = "pca"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._reducer = PCA(
            n_components=self.params["n_components"],
            whiten=self.params.get("whiten", False),
            random_state=self.params.get("random_state", 42),
        )

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        """Return default parameters for PCA."""
        return {
            "n_components": dimred_config.pca_n_components(),
            "whiten": dimred_config.pca_whiten(),
            "random_state": 42,
        }

    def fit_transform(self, x_matrix: np.ndarray):
        """Run PCA and return the embedding matrix."""
        return self._reducer.fit_transform(x_matrix)
