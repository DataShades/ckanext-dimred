import pytest

from ckan.plugins import plugin_loaded

from ckanext.dimred.plugin import DimredPlugin


@pytest.mark.usefixtures("with_plugins")
def test_plugin():
    assert plugin_loaded("dimred")


@pytest.mark.usefixtures("with_plugins")
def test_setup_template_variables_returns_error(monkeypatch, sysadmin):
    plugin = DimredPlugin()

    dummy_resource = {"id": "res-1", "format": "csv"}
    dummy_view = {"id": "view-1"}

    def fake_get_action(name):
        if name == "dimred_get_dimred_preview":
            return lambda ctx, data: {"error": "bad"}
        return lambda *a, **k: None

    monkeypatch.setattr("ckanext.dimred.plugin.tk.get_action", fake_get_action)

    out = plugin.setup_template_variables(
        {"user": sysadmin["id"]},
        {"resource": dummy_resource, "resource_view": dummy_view, "package": {}},
    )

    assert out["error"] == "bad"
    assert out["image_data_url"] is None
