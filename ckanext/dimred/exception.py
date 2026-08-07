from __future__ import annotations


class DimredError(Exception):
    """Domain-specific exception for dimred failures."""

    default_message = "Dimred error"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.default_message)


class DimredEmbeddingError(DimredError):
    """Raised when embedding cannot be plotted."""

    default_message = "Embedding must have at least 2 dimensions to plot."


class DimredNumericColumnError(DimredError):
    """Raised when no numeric columns are available for dimred."""

    default_message = "No numeric columns found for dimred processing."


class DimredFeatureError(DimredError):
    """Raised when not enough features are available for dimred."""

    default_message = "Not enough features for dimred."


class DimredDatastoreError(DimredError):
    """Raised when a DataStore resource cannot be loaded for dimred."""

    default_message = "DataStore resource cannot be loaded for dimred processing."


class DimredResourceUrlError(DimredError):
    """Raised when a resource URL cannot be used safely."""

    default_message = "Resource URL is invalid."


class DimredAdapterNotFoundError(DimredError):
    """Raised when no adapter is available for a resource format."""

    default_message = "No tabular adapter available for this format."


class DimredPreviewError(DimredError):
    """Raised when dimred preview action fails."""

    default_message = "Dimred preview failed."


class DimredResourceSizeError(DimredError):
    """Raised when resource exceeds configured size limit."""

    default_message = "Resource exceeds maximum allowed size for dimred processing."


class DimredRemoteFetchError(DimredError):
    """Raised when remote resource fetching fails."""

    default_message = "Error fetching remote resource."


class DimredTabularLoadError(DimredError):
    """Raised when tabular data cannot be loaded."""

    default_message = "Failed to load tabular data."
