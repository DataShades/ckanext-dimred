from __future__ import annotations

import socket

import pandas as pd
import pytest

from ckanext.dimred.adapters import base, tabular
from ckanext.dimred.exception import (
    DimredRemoteFetchError,
    DimredResourceSizeError,
    DimredResourceUrlError,
    DimredTabularLoadError,
)


class FakeResponse:
    def __init__(self, status=200, headers=None, chunks=()):
        self.status = status
        self.headers = headers or {}
        self.content = b"".join(chunks)
        self.offset = 0
        self.closed = False

    def getheader(self, name):
        return self.headers.get(name)

    def read(self, size):
        chunk = self.content[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.request_args = None
        self.closed = False

    def request(self, method, path, headers):
        self.request_args = (method, path, headers)
        if self.error:
            raise self.error

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


def _dns_result(address, port):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]


@pytest.mark.usefixtures("with_plugins")
def test_tabular_adapter_reads_csv(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    adapter = tabular.TabularAdapter(
        {"format": "csv", "size": 10},
        {},
        filepath=str(csv_path),
    )

    df = adapter.get_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["a", "b"]


@pytest.mark.usefixtures("with_plugins")
def test_tabular_adapter_samples_csv_incrementally_and_deterministically(tmp_path):
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text(
        "x,y\n" + "".join(f"{number},{number * 10}\n" for number in range(1, 21)),
        encoding="utf-8",
    )
    adapter = tabular.TabularAdapter(
        {"format": "csv", "size": csv_path.stat().st_size},
        {},
        filepath=str(csv_path),
    )

    first_df, first_ids, first_total = adapter.get_sampled_dataframe(4)
    second_df, second_ids, second_total = adapter.get_sampled_dataframe(4)

    assert first_total == second_total == 20
    assert len(first_df) == len(first_ids) == 4
    assert first_ids == second_ids
    assert first_df.equals(second_df)
    assert all(1 <= row_id <= first_total for row_id in first_ids)


@pytest.mark.usefixtures("with_plugins")
def test_tabular_adapter_reads_xlsx(tmp_path):
    xlsx_path = tmp_path / "data.xlsx"
    pd.DataFrame({"a": [1, 3], "b": [2, 4]}).to_excel(xlsx_path, index=False)

    adapter = tabular.TabularAdapter(
        {"format": "xlsx", "size": xlsx_path.stat().st_size},
        {},
        filepath=str(xlsx_path),
    )

    df = adapter.get_dataframe()

    assert df.to_dict(orient="list") == {"a": [1, 3], "b": [2, 4]}


@pytest.mark.usefixtures("with_plugins")
def test_tabular_adapter_get_columns(tmp_path):
    csv_path = tmp_path / "cols.csv"
    csv_path.write_text("col1,col2,col3\n1,2,3\n", encoding="utf-8")

    adapter = tabular.TabularAdapter(
        {"format": "csv", "size": 10},
        {},
        filepath=str(csv_path),
    )

    cols = adapter.get_columns()

    assert cols == ["col1", "col2", "col3"]


@pytest.mark.usefixtures("with_plugins")
@pytest.mark.ckan_config("ckanext.dimred.max_file_size_mb", "1")
def test_adapter_checks_actual_local_file_size(tmp_path):
    csv_path = tmp_path / "large.csv"
    csv_path.write_bytes(b"x" * (1024 * 1024 + 1))

    adapter = tabular.TabularAdapter(
        {"format": "csv", "size": 1},
        {},
        filepath=str(csv_path),
    )

    with pytest.raises(DimredResourceSizeError):
        adapter.validate_size_limit()


@pytest.mark.usefixtures("with_plugins")
def test_adapter_rejects_unstatable_local_file(monkeypatch, tmp_path):
    adapter = tabular.TabularAdapter({"format": "csv"}, {}, filepath=str(tmp_path / "missing.csv"))
    monkeypatch.setattr(base.os.path, "getsize", lambda path: (_ for _ in ()).throw(OSError()))

    with pytest.raises(DimredTabularLoadError):
        adapter.validate_size_limit()


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("8.8.8.8", True),
        ("127.0.0.1", False),
        ("10.0.0.1", False),
        ("169.254.169.254", False),
        ("::1", False),
        ("fc00::1", False),
    ],
)
def test_public_address_check(address, expected):
    assert base._is_public_address(address) is expected


def test_remote_url_rejects_credentials():
    with pytest.raises(DimredResourceUrlError):
        base._validate_remote_url("https://user:secret@example.com/data.csv")


def test_https_connection_uses_validated_address_and_original_hostname():
    connection = base._open_remote_connection(base.urlsplit("https://public.example/data.csv"), "8.8.8.8")

    assert connection.host == "8.8.8.8"
    assert connection.server_hostname == "public.example"


@pytest.mark.usefixtures("with_plugins")
def test_remote_fetch_revalidates_redirect_target(monkeypatch):
    def fake_getaddrinfo(host, port, type):
        address = "8.8.8.8" if host == "public.example" else "127.0.0.1"
        return _dns_result(address, port)

    response = FakeResponse(302, {"Location": "http://internal.example/data.csv"})
    connection = FakeConnection(response)
    monkeypatch.setattr(base.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(base, "_open_remote_connection", lambda parsed, address: connection)

    adapter = base.BaseAdapter({"type": "url", "url": "http://public.example/data.csv"}, {})

    with pytest.raises(DimredRemoteFetchError):
        adapter.fetch_remote(adapter.filepath)

    assert response.closed
    assert connection.closed


@pytest.mark.usefixtures("with_plugins")
@pytest.mark.ckan_config("ckanext.dimred.max_file_size_mb", "1")
def test_remote_fetch_stops_after_configured_size_limit(monkeypatch):
    response = FakeResponse(
        headers={"Content-Length": "1"},
        chunks=[b"x" * (1024 * 1024 + 1)],
    )
    connection = FakeConnection(response)

    monkeypatch.setattr(
        base.socket,
        "getaddrinfo",
        lambda host, port, type: _dns_result("8.8.8.8", port),
    )
    monkeypatch.setattr(base, "_open_remote_connection", lambda parsed, address: connection)
    adapter = base.BaseAdapter({"type": "url", "url": "https://public.example/data.csv"}, {})

    with pytest.raises(DimredResourceSizeError):
        adapter.fetch_remote(adapter.filepath)

    assert response.closed
    assert connection.request_args == (
        "GET",
        "/data.csv",
        {
            "Host": "public.example",
            "User-Agent": base.USER_AGENT,
            "Accept-Encoding": "identity",
        },
    )


@pytest.mark.usefixtures("with_plugins")
def test_remote_fetch_hides_transport_and_truncation_details(monkeypatch):
    monkeypatch.setattr(
        base.socket,
        "getaddrinfo",
        lambda host, port, type: _dns_result("8.8.8.8", port),
    )
    adapter = base.BaseAdapter({"type": "url", "url": "https://public.example/data.csv"}, {})

    monkeypatch.setattr(
        base,
        "_open_remote_connection",
        lambda parsed, address: FakeConnection(FakeResponse(headers={"Content-Length": "10"}, chunks=[b"short"])),
    )
    with pytest.raises(DimredRemoteFetchError):
        adapter.fetch_remote(adapter.filepath)

    monkeypatch.setattr(base, "_open_remote_connection", lambda parsed, address: FakeConnection(error=OSError()))
    with pytest.raises(DimredRemoteFetchError):
        adapter.fetch_remote(adapter.filepath)
