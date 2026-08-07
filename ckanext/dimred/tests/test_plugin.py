import pytest

from ckan.plugins import plugin_loaded
from ckan.plugins import toolkit as tk

from ckanext.dimred.plugin import DimredPlugin


@pytest.mark.usefixtures("with_plugins")
def test_plugin():
    assert plugin_loaded("dimred")


@pytest.mark.usefixtures("with_plugins")
def test_plugin_exports_only_public_actions():
    actions = DimredPlugin().get_actions()

    assert "dimred_get_dimred_preview" in actions
    assert "dimred_export_embedding" in actions
    assert "dimred_run_dimred_pipeline" not in actions


@pytest.mark.usefixtures("with_plugins")
def test_can_view_datastore_only_resource():
    assert DimredPlugin().can_view({"resource": {"datastore_active": True, "format": ""}})


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


@pytest.mark.usefixtures("with_plugins")
@pytest.mark.parametrize(
    "field_errors",
    ["perplexity must be greater than 0.", ["perplexity must be greater than 0."]],
)
def test_setup_template_variables_formats_validation_error(monkeypatch, sysadmin, field_errors):
    plugin = DimredPlugin()

    def fake_get_action(name):
        if name == "dimred_get_dimred_preview":
            def preview(context, data_dict):
                raise tk.ValidationError({"method_params": field_errors})

            return preview
        return lambda *args, **kwargs: None

    monkeypatch.setattr("ckanext.dimred.plugin.tk.get_action", fake_get_action)

    out = plugin.setup_template_variables(
        {"user": sysadmin["id"]},
        {"resource": {"id": "res-1", "format": "csv"}, "resource_view": {"id": "view-1"}, "package": {}},
    )

    assert out["error"] == "perplexity must be greater than 0."
