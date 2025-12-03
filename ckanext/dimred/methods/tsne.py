from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.manifold import TSNE

from ckanext.dimred import config as dimred_config
from ckanext.dimred.methods.base import BaseProjectionMethod

log = logging.getLogger(__name__)


class TSNEProjection(BaseProjectionMethod):
    """Wrapper around sklearn.manifold.TSNE."""

    name = "tsne"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

        self._reducer = TSNE(
            n_components=self.params["n_components"],
            perplexity=self.params["perplexity"],
            random_state=self.params.get("random_state", 42),
            init="random",
        )

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        """Return default parameters for the t-SNE method."""
        return {
            "perplexity": dimred_config.tsne_perplexity(),
            "n_components": dimred_config.tsne_n_components(),
            "random_state": 42,
        }

    def fit_transform(self, x_matrix: np.ndarray):
        """Run t-SNE and return the embedding matrix."""
        return self._reducer.fit_transform(x_matrix)
