from __future__ import annotations

import logging
from typing import Any

import requests

import ckan.plugins.toolkit as tk
from ckan.lib.uploader import get_resource_uploader

from ckanext.dimred import config as dimred_config
from ckanext.dimred import utils as dimred_utils
from ckanext.dimred.exception import (
    DimredRemoteFetchError,
    DimredResourceSizeError,
    DimredResourceUrlError,
)

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60


class BaseAdapter:
    """Base adapter for dimred resource handling.

    It encapsulates:
    - local vs remote resource detection
    - resource file path resolution
    - file size validation
    - HTTP fetching for remote resources
    """

    def __init__(
        self,
        resource: dict[str, Any],
        resource_view: dict[str, Any],
        filepath: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.resource = resource
        self.resource_view = resource_view
        self.kwargs = kwargs

        if filepath:
            self.remote = False
            self.filepath = filepath
        else:
            self.remote = self._is_remote()
            self.filepath = self._get_filepath()

    def _get_filepath(self) -> str:
        """Resolve local or remote path/URL for the resource."""
        resource_url = self.resource.get("url", "")

        if not resource_url:
            raise DimredResourceUrlError

        if self.remote:
            return resource_url

        uploader = get_resource_uploader(self.resource)
        return uploader.get_path(self.resource["id"])

    def _is_remote(self) -> bool:
        """Determine whether the resource is remote or stored locally."""
        resource_type = self.resource.get("type", "")
        resource_url = self.resource.get("url", "")

        if not resource_url:
            raise DimredResourceUrlError

        if resource_type == "upload":
            return False

        if resource_type == "url":
            return True

        site_url = tk.config.get("ckan.site_url", "")
        return not resource_url.startswith(site_url)

    def validate_size_limit(self) -> None:
        """Ensure the resource does not exceed the configured max file size."""
        size = self.resource.get("size")

        if size and isinstance(size, str):
            try:
                size = int(size)
            except (ValueError, TypeError):
                size = None

        if size is None:
            return

        max_size_mb = dimred_config.max_file_size_mb()
        max_size_bytes = max_size_mb * 1024 * 1024

        if size <= max_size_bytes:
            return

        readable_size = dimred_utils.printable_file_size(max_size_bytes)
        raise DimredResourceSizeError(readable_size)

    def make_request(self, url: str) -> bytes:
        """Make a GET request to the specified URL and return the raw content."""
        try:
            with requests.get(url, timeout=DEFAULT_TIMEOUT, stream=True) as resp:
                resp.raise_for_status()
                content = resp.content
        except requests.RequestException as e:
            raise DimredRemoteFetchError(str(e)) from e

        return content

    def get_dataframe(self):
        """Return a pandas.DataFrame representing the tabular data.

        Subclasses must implement this method and handle both local and
        remote resources based on the `self.remote` attribute.
        """
        raise NotImplementedError
