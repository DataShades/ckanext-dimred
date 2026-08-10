from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

from ckanext.dimred import config as dimred_config
from ckanext.dimred.methods.base import BaseProjectionMethod


class PCAProjection(BaseProjectionMethod):
    """Wrapper around sklearn.decomposition.PCA."""

    name = "pca"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        pca = import_module("sklearn.decomposition").PCA
        self._reducer = pca(
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

    def result_metadata(self) -> dict[str, Any]:
        """Return explained variance for the fitted PCA components."""
        ratios = self._reducer.explained_variance_ratio_
        if not np.isfinite(ratios).all():
            return {}
        values = [float(value) for value in ratios]
        return {
            "explained_variance_ratio": values,
            "explained_variance_cumulative": float(sum(values)),
        }
