from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseProjectionMethod(ABC):
    """Base class for all dimensionality reduction methods (UMAP, t-SNE, PCA, ...).

    - default_params() should return method-specific default parameters.
    - __init__ merges defaults with the parameters passed from the caller.
    """

    name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = self._merge_with_defaults(params)

    @classmethod
    @abstractmethod
    def default_params(cls) -> dict[str, Any]:
        """Return a dictionary of default parameters for the method."""
        raise NotImplementedError

    def _merge_with_defaults(self, params: dict[str, Any]) -> dict[str, Any]:
        """Merge the method's default parameters with the given params.

        Explicitly provided params override defaults, but None values are ignored.
        """
        defaults = dict(self.default_params())
        return {
            **defaults,
            **{key: value for key, value in params.items() if value is not None},
        }

    @abstractmethod
    def fit_transform(self, x_matrix: np.ndarray):
        """Run dimensionality reduction and return the embedding matrix.

        :param x_matrix: Input data matrix.
        :return: Embedding matrix.
        """
        raise NotImplementedError
