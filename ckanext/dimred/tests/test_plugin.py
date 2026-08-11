from io import BytesIO

import pytest
from werkzeug.datastructures import FileStorage

from ckan.lib.helpers import url_for
from ckan.plugins import plugin_loaded
from ckan.plugins import toolkit as tk
from ckan.tests import factories
from ckan.tests.helpers import call_action

from ckanext.dimred.plugin import DimredPlugin


@pytest.mark.usefixtures("with_plugins")
def test_plugin():
    assert plugin_loaded("dimred")


@pytest.mark.usefixtures("with_plugins")
def test_plugin_exports_only_public_actions():
    actions = DimredPlugin().get_actions()

    assert "dimred_start_preview" in actions
    assert "dimred_get_preview_status" in actions
    assert "dimred_get_dimred_color_values" in actions
    assert "dimred_export_embedding" in actions
    assert "dimred_run_dimred_pipeline" not in actions
    assert "dimred_get_dimred_preview" not in actions


@pytest.mark.usefixtures("with_plugins")
def test_can_view_datastore_only_resource():
    assert DimredPlugin().can_view({"resource": {"datastore_active": True, "format": ""}})


@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_form_renders_method_parameter_controls(app):
    user = factories.UserWithToken()
    organization = factories.Organization(users=[{"name": user["name"], "capacity": "admin"}])
    package = factories.Dataset(owner_org=organization["id"])
    resource = factories.Resource(package_id=package["id"], format="csv")
    url = url_for(
        "dataset_resource.edit_view",
        id=package["name"],
        resource_id=resource["id"],
        view_type="dimred_view",
    )

    response = app.get(url, headers={"Authorization": user["token"]})

    assert 'id="field-method-params"' in response
    assert 'type="hidden"' in response
    assert 'id="field-method-param-random-state"' in response
    assert 'id="field-method-param-perplexity"' in response
    assert 'id="field-method-param-n-neighbors"' in response
    assert 'data-dimred-param="perplexity" step="any"' in response
    assert 'data-dimred-param="min_dist" step="any"' in response
    assert 'data-module-defaults=' in response
    assert "Method-specific parameters (JSON)" not in response


@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_form_renders_autocomplete_feature_multi_select(app):
    user = factories.UserWithToken()
    organization = factories.Organization(users=[{"name": user["name"], "capacity": "admin"}])
    package = factories.Dataset(owner_org=organization["id"])
    resource = call_action(
        "resource_create",
        package_id=package["id"],
        name="Columns",
        format="csv",
        upload=FileStorage(
            stream=BytesIO(b"x,y,label\n1,2,first\n"),
            filename="columns.csv",
            content_type="text/csv",
        ),
    )
    url = url_for(
        "dataset_resource.edit_view",
        id=package["name"],
        resource_id=resource["id"],
        view_type="dimred_view",
    )

    response = app.get(url, headers={"Authorization": user["token"]})

    assert 'id="field-feature-columns"' in response
    assert 'name="feature_columns"' in response
    assert 'multiple="multiple"' in response
    assert 'data-module="autocomplete"' in response
    assert 'id="dimred-select-all-features"' in response
    assert 'id="dimred-clear-features"' in response
    assert '<option value="x">x</option>' in response
    assert '<option value="y">y</option>' in response
    assert '<option value="label">label</option>' in response


@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_form_submission_persists_method_params_and_restores_controls(app):
    user = factories.UserWithToken()
    organization = factories.Organization(users=[{"name": user["name"], "capacity": "admin"}])
    package = factories.Dataset(owner_org=organization["id"])
    resource = factories.Resource(package_id=package["id"], format="csv")
    new_view_url = url_for(
        "dataset_resource.edit_view",
        id=package["name"],
        resource_id=resource["id"],
    )

    app.post(
        new_view_url,
        headers={"Authorization": user["token"]},
        query_string={"view_type": "dimred_view"},
        status=200,
        data={
            "title": "Dimred",
            "view_type": "dimred_view",
            "method": "tsne",
            "n_components": "2",
            "method_params": '{"random_state":42,"perplexity":12.5}',
            "dimred_param_random_state": "42",
            "dimred_param_perplexity": "12.5",
        },
    )
    view = call_action("resource_view_list", id=resource["id"])[0]
    edit_view_url = url_for(
        "dataset_resource.edit_view",
        id=package["name"],
        resource_id=resource["id"],
        view_id=view["id"],
    )
    response = app.get(edit_view_url, headers={"Authorization": user["token"]})

    assert view["method_params"] == {"random_state": 42, "perplexity": 12.5}
    assert 'id="field-method-param-perplexity"' in response
    assert 'value="12.5"' in response


@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_form_submission_persists_native_feature_multi_select(app):
    user = factories.UserWithToken()
    organization = factories.Organization(users=[{"name": user["name"], "capacity": "admin"}])
    package = factories.Dataset(owner_org=organization["id"])
    resource = call_action(
        "resource_create",
        package_id=package["id"],
        name="Columns",
        format="csv",
        upload=FileStorage(
            stream=BytesIO(b"x,y,label\n1,2,first\n"),
            filename="columns.csv",
            content_type="text/csv",
        ),
    )
    new_view_url = url_for(
        "dataset_resource.edit_view",
        id=package["name"],
        resource_id=resource["id"],
    )

    app.post(
        new_view_url,
        headers={"Authorization": user["token"]},
        query_string={"view_type": "dimred_view"},
        status=200,
        data={
            "title": "Dimred",
            "view_type": "dimred_view",
            "method": "pca",
            "feature_columns": ["x", "y"],
            "color_by": "label",
        },
    )
    view = call_action("resource_view_list", id=resource["id"])[0]
    edit_view_url = url_for(
        "dataset_resource.edit_view",
        id=package["name"],
        resource_id=resource["id"],
        view_id=view["id"],
    )
    response = app.get(edit_view_url, headers={"Authorization": user["token"]})

    assert view["feature_columns"] == ["x", "y"]
    assert '<option value="x" selected>x</option>' in response
    assert '<option value="y" selected>y</option>' in response
    assert '<option value="label">label</option>' in response


@pytest.mark.usefixtures("with_plugins")
def test_setup_template_variables_returns_error(monkeypatch, sysadmin):
    plugin = DimredPlugin()

    dummy_resource = {"id": "res-1", "format": "csv"}
    dummy_view = {"id": "view-1"}

    def fake_get_action(name):
        if name == "dimred_start_preview":
            return lambda ctx, data: {"status": "failed", "error": "bad"}
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
        if name == "dimred_start_preview":
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
