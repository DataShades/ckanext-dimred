from __future__ import annotations

import http.client
import ipaddress
import logging
import os
import socket
import ssl
from typing import Any
from urllib.parse import SplitResult, urljoin, urlsplit

import ckan.plugins.toolkit as tk
from ckan.lib.uploader import get_resource_uploader

from ckanext.dimred import config as dimred_config
from ckanext.dimred import utils as dimred_utils
from ckanext.dimred.exception import (
    DimredRemoteFetchError,
    DimredResourceSizeError,
    DimredResourceUrlError,
    DimredTabularLoadError,
)

log = logging.getLogger(__name__)

CONNECT_TIMEOUT = 5
READ_TIMEOUT = 30
MAX_REDIRECTS = 3
USER_AGENT = "ckanext-dimred/0.0.1"
CHUNK_SIZE = 8192
HTTP_ERROR_STATUS = 400


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
        """Ensure metadata and local files do not exceed the configured size limit."""
        size = self.resource.get("size")

        if size and isinstance(size, str):
            try:
                size = int(size)
            except (ValueError, TypeError):
                size = None

        max_size_bytes = self._max_size_bytes()
        if size is not None and size > max_size_bytes:
            self._raise_size_error(max_size_bytes)

        if self.remote:
            return

        try:
            local_size = os.path.getsize(self.filepath)
        except OSError as err:
            raise DimredTabularLoadError from err
        if local_size > max_size_bytes:
            self._raise_size_error(max_size_bytes)

    def _max_size_bytes(self) -> int:
        """Return the extension-specific resource size limit in bytes."""
        return dimred_config.max_file_size_mb() * 1024 * 1024

    def _raise_size_error(self, max_size_bytes: int) -> None:
        """Raise a size error without exposing resource paths or URLs."""
        readable_size = dimred_utils.printable_file_size(max_size_bytes)
        raise DimredResourceSizeError(readable_size)

    def fetch_remote(self, url: str, max_bytes: int | None = None) -> bytes:
        """Fetch a remote resource with SSRF checks, redirect validation and size limits."""
        current_url = url
        for redirect_count in range(MAX_REDIRECTS + 1):
            parsed, addresses = _validate_remote_url(current_url)
            connection, response = self._request_remote(parsed, addresses)
            try:
                if response.status in {301, 302, 303, 307, 308}:
                    location = response.getheader("Location")
                    if not location or redirect_count == MAX_REDIRECTS:
                        raise DimredRemoteFetchError
                    current_url = urljoin(current_url, location)
                    continue

                if response.status >= HTTP_ERROR_STATUS:
                    raise DimredRemoteFetchError
                return self._read_response(response, max_bytes)
            except _REMOTE_REQUEST_ERRORS as err:
                raise DimredRemoteFetchError from err
            finally:
                response.close()
                connection.close()

        raise DimredRemoteFetchError

    def _request_remote(
        self,
        parsed: SplitResult,
        addresses: list[str],
    ) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
        """Fetch from a validated address without performing another DNS lookup."""
        request_path = parsed.path or "/"
        if parsed.query:
            request_path = f"{request_path}?{parsed.query}"

        for address in addresses:
            connection = _open_remote_connection(parsed, address)
            try:
                connection.request(
                    "GET",
                    request_path,
                    headers={
                        "Host": _host_header(parsed),
                        "User-Agent": USER_AGENT,
                        "Accept-Encoding": "identity",
                    },
                )
                return connection, connection.getresponse()
            except _REMOTE_REQUEST_ERRORS:
                connection.close()

        raise DimredRemoteFetchError

    def _read_response(self, response: http.client.HTTPResponse, max_bytes: int | None) -> bytes:
        """Read an HTTP response without exceeding the configured in-memory limit."""
        max_size_bytes = self._max_size_bytes()
        content_length = _content_length(response.getheader("Content-Length"))
        if content_length is not None and content_length > max_size_bytes:
            self._raise_size_error(max_size_bytes)

        read_limit = min(max_bytes, max_size_bytes) if max_bytes is not None else max_size_bytes
        content = bytearray()
        while chunk := response.read(CHUNK_SIZE):
            if not chunk:
                continue
            remaining = read_limit + 1 - len(content)
            content.extend(chunk[:remaining])
            if len(content) > read_limit:
                if max_bytes is not None:
                    return bytes(content[:max_bytes])
                self._raise_size_error(max_size_bytes)

        if max_bytes is None and content_length is not None and len(content) != content_length:
            raise DimredRemoteFetchError
        return bytes(content)

    def get_dataframe(self):
        """Return a pandas.DataFrame representing the tabular data.

        Subclasses must implement this method and handle both local and
        remote resources based on the `self.remote` attribute.
        """
        raise NotImplementedError

    def get_sampled_dataframe(self, row_limit: int):
        """Return a bounded dataframe with stable one-based source row IDs.

        Adapters that can stream their input should override this method. The
        fallback preserves the existing behavior for formats that require a
        complete read, such as Excel workbooks.
        """
        df = self.get_dataframe().reset_index(drop=True)
        n_rows_original = len(df)
        source_row_ids = list(range(1, n_rows_original + 1))
        if n_rows_original <= row_limit:
            return df, source_row_ids, n_rows_original

        sampled = df.sample(row_limit, random_state=42)
        sampled_row_ids = [source_row_ids[position] for position in sampled.index]
        return sampled.reset_index(drop=True), sampled_row_ids, n_rows_original


_REMOTE_REQUEST_ERRORS = (http.client.HTTPException, OSError, ssl.SSLError)


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection pinned to a DNS result validated by the caller."""

    def connect(self) -> None:
        """Connect with a short connect timeout and a separate read timeout."""
        super().connect()
        self.sock.settimeout(READ_TIMEOUT)


class _PinnedHTTPSConnection(_PinnedHTTPConnection):
    """HTTPS connection pinned to an IP while preserving hostname verification."""

    def __init__(self, address: str, port: int, server_hostname: str) -> None:
        super().__init__(address, port, timeout=CONNECT_TIMEOUT)
        self.server_hostname = server_hostname
        self.context = ssl.create_default_context()

    def connect(self) -> None:
        """Connect to the validated IP and verify TLS for the original hostname."""
        super().connect()
        self.sock = self.context.wrap_socket(self.sock, server_hostname=self.server_hostname)
        self.sock.settimeout(READ_TIMEOUT)


def _validate_remote_url(url: str) -> tuple[SplitResult, list[str]]:
    """Reject URLs that cannot be fetched safely by the server."""
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise DimredResourceUrlError
        hostname = _hostname(parsed)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as err:
        raise DimredResourceUrlError from err

    try:
        results = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as err:
        raise DimredRemoteFetchError from err
    addresses = list(dict.fromkeys(address for result in results if isinstance(address := result[4][0], str)))
    if not addresses or any(not _is_public_address(address) for address in addresses):
        raise DimredRemoteFetchError
    return parsed, addresses


def _open_remote_connection(parsed: SplitResult, address: str) -> http.client.HTTPConnection:
    """Return a direct connection to a validated address without proxy support."""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    hostname = _hostname(parsed)
    if parsed.scheme == "https":
        return _PinnedHTTPSConnection(address, port, hostname)
    return _PinnedHTTPConnection(address, port, timeout=CONNECT_TIMEOUT)


def _host_header(parsed: SplitResult) -> str:
    """Return the original hostname for HTTP routing, adding a non-default port."""
    hostname = _hostname(parsed)
    if ":" in hostname:
        hostname = f"[{hostname}]"
    port = parsed.port
    if port is not None and port != (443 if parsed.scheme == "https" else 80):
        return f"{hostname}:{port}"
    return hostname


def _hostname(parsed: SplitResult) -> str:
    """Return a hostname suitable for DNS, TLS SNI, and HTTP headers."""
    hostname = parsed.hostname
    if not hostname:
        raise DimredResourceUrlError
    return hostname.encode("idna").decode("ascii")


def _is_public_address(value: str) -> bool:
    """Return whether an IP is safe for a server-side outbound request."""
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return address.is_global


def _content_length(value: str | None) -> int | None:
    """Parse a non-negative Content-Length header without trusting invalid values."""
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None
