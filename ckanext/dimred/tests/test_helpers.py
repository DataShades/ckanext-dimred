from __future__ import annotations

import pytest

from ckan.tests.helpers import call_action

from ckanext.dimred import config as dimred_config
from ckanext.dimred import helpers

pytest_plugins = ["ckanext.datastore.tests.conftest"]


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


@pytest.mark.usefixtures("with_plugins")
def test_method_params_form_values_merge_known_saved_values_with_defaults():
    values = helpers.dimred_method_params_form_values(
        '{"perplexity": 12, "unexpected": "ignored", "n_components": 3}',
        "tsne",
    )

    assert values["perplexity"] == 12
    assert values["random_state"] == 42
    assert "unexpected" not in values
    assert "n_components" not in values


@pytest.mark.usefixtures("with_plugins")
def test_method_params_form_values_fall_back_to_defaults_for_invalid_json():
    assert helpers.dimred_method_params_form_values("not json", "pca") == {
        "whiten": False,
        "random_state": 42,
    }


@pytest.mark.usefixtures("with_plugins")
def test_methods_defaults_include_n_components_for_method_switching():
    assert all("n_components" in defaults for defaults in helpers.dimred_methods_defaults().values())


@pytest.mark.usefixtures("clean_db", "clean_datastore", "with_plugins")
@pytest.mark.ckan_config("ckan.plugins", "datastore dimred")
def test_resource_options_use_datastore_columns(app, package):
    call_action(
        "datastore_create",
        {},
        resource={"package_id": package["id"], "name": "DataStore rows", "format": "csv"},
        records=[{"store_x": 1, "store_y": 2, "label": "first"}],
    )
    dataset = call_action("package_show", {}, id=package["id"])
    resource = dataset["resources"][0]

    assert resource["datastore_active"] is True
    assert resource["url_type"] == "datastore"
    assert [option["value"] for option in helpers.dimred_feature_options_from_resource(resource)] == [
        "store_x",
        "store_y",
        "label",
    ]
    assert [option["value"] for option in helpers.dimred_color_options_from_resource(resource)] == [
        "",
        "store_x",
        "store_y",
        "label",
    ]


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


def test_method_defaults_form_strips_n_components():
    params = helpers.dimred_method_default_params_form("umap")

    assert "n_components" not in params
