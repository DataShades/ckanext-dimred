from __future__ import annotations

import pytest

from ckanext.dimred import config as dimred_config
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


def test_render_asset_default_echarts(monkeypatch):
    monkeypatch.setattr(dimred_config, "render_backend", lambda: "echarts")
    monkeypatch.setattr(dimred_config, "render_asset", lambda: "")

    assert helpers.dimred_render_asset() == "dimred/dimred-echarts-js"


def test_render_asset_custom(monkeypatch):
    monkeypatch.setattr(dimred_config, "render_backend", lambda: "custom")
    monkeypatch.setattr(dimred_config, "render_asset", lambda: "my/custom.js")

    assert helpers.dimred_render_asset() == "my/custom.js"


def test_render_module_default_echarts(monkeypatch):
    monkeypatch.setattr(dimred_config, "render_backend", lambda: "echarts")
    monkeypatch.setattr(dimred_config, "render_module", lambda: "")

    assert helpers.dimred_render_module() == "dimred-view-echarts"


def test_render_module_custom(monkeypatch):
    monkeypatch.setattr(dimred_config, "render_backend", lambda: "custom")
    monkeypatch.setattr(dimred_config, "render_module", lambda: "my-module")

    assert helpers.dimred_render_module() == "my-module"


def test_render_asset_respects_backend_arg(monkeypatch):
    monkeypatch.setattr(dimred_config, "render_backend", lambda: "echarts")
    monkeypatch.setattr(dimred_config, "render_asset", lambda: "")

    assert helpers.dimred_render_asset("matplotlib") is None


def test_render_backend_default(monkeypatch):
    monkeypatch.setattr(dimred_config, "render_backend", lambda: "matplotlib")

    assert helpers.dimred_render_backend_default() == "matplotlib"
