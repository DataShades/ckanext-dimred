from __future__ import annotations

import ckan.plugins.toolkit as tk

DEFAULT_METHOD = "ckanext.dimred.default_method"
ALLOWED_METHODS = "ckanext.dimred.allowed_methods"

MAX_FILE_SIZE_MB = "ckanext.dimred.max_file_size_mb"
MAX_ROWS = "ckanext.dimred.max_rows"

ENABLE_CATEGORICAL = "ckanext.dimred.enable_categorical"
MAX_CATEGORIES_FOR_OHE = "ckanext.dimred.max_categories_for_ohe"

UMAP_N_NEIGHBORS = "ckanext.dimred.umap.n_neighbors"
UMAP_MIN_DIST = "ckanext.dimred.umap.min_dist"
UMAP_N_COMPONENTS = "ckanext.dimred.umap.n_components"

TSNE_PERPLEXITY = "ckanext.dimred.tsne.perplexity"
TSNE_N_COMPONENTS = "ckanext.dimred.tsne.n_components"


def default_method() -> str:
    """Default dimensionality reduction method (e.g. 'umap')."""
    return tk.config[DEFAULT_METHOD]


def allowed_methods() -> list[str]:
    """List of enabled dimred methods.

    With a proper config_declaration this will already be a list, but
    we also handle a space-separated string just in case.
    """
    return tk.config[ALLOWED_METHODS]


def max_file_size_mb() -> int:
    """Maximum resource file size (in megabytes) for which dimred is attempted."""
    return int(tk.config[MAX_FILE_SIZE_MB])


def max_rows() -> int:
    """Maximum number of rows to load from the resource."""
    return int(tk.config[MAX_ROWS])


def enable_categorical() -> bool:
    """Whether to include low-cardinality categorical columns via one-hot encoding."""
    return tk.config[ENABLE_CATEGORICAL]


def max_categories_for_ohe() -> int:
    """Maximum distinct values for a categorical column to be one-hot encoded."""
    return int(tk.config[MAX_CATEGORIES_FOR_OHE])


def umap_n_neighbors() -> int:
    """Default UMAP n_neighbors value."""
    return int(tk.config[UMAP_N_NEIGHBORS])


def umap_min_dist() -> float:
    """Default UMAP min_dist value (parsed from config as float)."""
    value = tk.config[UMAP_MIN_DIST]
    return float(value)


def umap_n_components() -> int:
    """Number of output components for UMAP."""
    return tk.config[UMAP_N_COMPONENTS]


def tsne_perplexity() -> int:
    """Default t-SNE perplexity."""
    return tk.config[TSNE_PERPLEXITY]


def tsne_n_components() -> int:
    """Number of output components for t-SNE."""
    return tk.config[TSNE_N_COMPONENTS]
