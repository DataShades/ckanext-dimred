from __future__ import annotations

import ckan.plugins.toolkit as tk

DEFAULT_METHOD = "ckanext.dimred.default_method"
ALLOWED_METHODS = "ckanext.dimred.allowed_methods"

MAX_FILE_SIZE_MB = "ckanext.dimred.max_file_size_mb"
MAX_ROWS = "ckanext.dimred.max_rows"

ENABLE_CATEGORICAL = "ckanext.dimred.enable_categorical"
MAX_CATEGORIES_FOR_OHE = "ckanext.dimred.max_categories_for_ohe"
CACHE_ENABLED = "ckanext.dimred.cache_enabled"
CACHE_TTL = "ckanext.dimred.cache_ttl"
EXPORT_ENABLED = "ckanext.dimred.export_enabled"
RENDER_BACKEND = "ckanext.dimred.render_backend"
RENDER_ASSET = "ckanext.dimred.render_asset"
RENDER_MODULE = "ckanext.dimred.render_module"
EMBEDDING_DECIMALS = "ckanext.dimred.embedding_decimals"
UMAP_N_NEIGHBORS = "ckanext.dimred.umap.n_neighbors"
UMAP_MIN_DIST = "ckanext.dimred.umap.min_dist"
UMAP_N_COMPONENTS = "ckanext.dimred.umap.n_components"

TSNE_PERPLEXITY = "ckanext.dimred.tsne.perplexity"
TSNE_N_COMPONENTS = "ckanext.dimred.tsne.n_components"

PCA_N_COMPONENTS = "ckanext.dimred.pca.n_components"
PCA_WHITEN = "ckanext.dimred.pca.whiten"


def default_method() -> str:
    """Default dimensionality reduction method (e.g. 'umap')."""
    return tk.config[DEFAULT_METHOD]


def allowed_methods() -> list[str]:
    """List of enabled dimred methods."""
    return tk.config[ALLOWED_METHODS]


def max_file_size_mb() -> int:
    """Maximum resource file size (in megabytes) for which dimred is attempted."""
    return tk.config[MAX_FILE_SIZE_MB]


def max_rows() -> int:
    """Maximum number of rows to load from the resource."""
    return tk.config[MAX_ROWS]


def enable_categorical() -> bool:
    """Whether to include low-cardinality categorical columns via one-hot encoding."""
    return tk.config[ENABLE_CATEGORICAL]


def max_categories_for_ohe() -> int:
    """Maximum distinct values for a categorical column to be one-hot encoded."""
    return tk.config[MAX_CATEGORIES_FOR_OHE]


def cache_enabled() -> bool:
    """Whether caching for dimred previews is enabled."""
    return tk.config[CACHE_ENABLED]


def cache_ttl() -> int:
    """TTL for cached dimred previews in seconds."""
    return tk.config[CACHE_TTL]


def export_enabled() -> bool:
    """Whether embedding export is enabled."""
    return tk.config[EXPORT_ENABLED]


def render_backend() -> str:
    """Return the render backend ('echarts' or 'matplotlib')."""
    return tk.config[RENDER_BACKEND]


def render_asset() -> str | None:
    """Optional webasset bundle name for the selected backend."""
    return tk.config[RENDER_ASSET]


def render_module() -> str | None:
    """Optional CKAN module name for the selected backend."""
    return tk.config[RENDER_MODULE]


def embedding_decimals() -> int:
    """Decimal places to round embedding coordinates."""
    return tk.config[EMBEDDING_DECIMALS]


def umap_n_neighbors() -> int:
    """Default UMAP n_neighbors value."""
    return tk.config[UMAP_N_NEIGHBORS]


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


def pca_n_components() -> int:
    """Number of output components for PCA."""
    return tk.config[PCA_N_COMPONENTS]


def pca_whiten() -> bool:
    """Whether to whiten PCA output."""
    return tk.config[PCA_WHITEN]
