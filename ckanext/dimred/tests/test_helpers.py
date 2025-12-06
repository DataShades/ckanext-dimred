from __future__ import annotations

import pytest

from ckanext.dimred import helpers


class DummyAdapter:
    def __init__(self, resource, resource_view):
        self.resource = resource
        self.resource_view = resource_view
        self.columns_called = False
        self.dataframe_called = False

    def get_columns(self):
        self.columns_called = True
        return ["c1", "c2"]

    def get_dataframe(self):
        self.dataframe_called = True
        raise AssertionError


@pytest.mark.usefixtures("with_plugins")
def test_color_options_use_columns(monkeypatch):
    adapter = DummyAdapter({"id": "1"}, {})

    monkeypatch.setattr(
        "ckanext.dimred.helpers.dimred_utils.get_adapter_for_resource",
        lambda resource: lambda *args, **kwargs: adapter,
    )

    opts = helpers.dimred_color_options_from_resource({"id": "1", "format": "csv"})

    assert adapter.columns_called is True
    assert adapter.dataframe_called is False
    assert [o["value"] for o in opts] == ["", "c1", "c2"]


@pytest.mark.usefixtures("with_plugins")
def test_feature_options_use_columns(monkeypatch):
    adapter = DummyAdapter({"id": "1"}, {})

    monkeypatch.setattr(
        "ckanext.dimred.helpers.dimred_utils.get_adapter_for_resource",
        lambda resource: lambda *args, **kwargs: adapter,
    )

    opts = helpers.dimred_feature_options_from_resource({"id": "1", "format": "csv"})

    assert adapter.columns_called is True
    assert adapter.dataframe_called is False
    assert [o["value"] for o in opts] == ["c1", "c2"]
