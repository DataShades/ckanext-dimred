from __future__ import annotations

import pytest

import ckan.plugins.toolkit as tk


@pytest.mark.ckan_config("ckanext.dimred.allowed_methods", "umap")
@pytest.mark.usefixtures("with_plugins")
def test_allowed_method_accepts_configured_method():
    validator = tk.get_validator("dimred_allowed_method")

    assert validator("umap", {}) == "umap"
    assert validator(" umap ", {}) == "umap"


@pytest.mark.ckan_config("ckanext.dimred.allowed_methods", "umap")
@pytest.mark.usefixtures("with_plugins")
def test_allowed_method_rejects_unknown():
    validator = tk.get_validator("dimred_allowed_method")

    with pytest.raises(tk.Invalid):
        validator("tsne", {})


@pytest.mark.usefixtures("with_plugins")
def test_feature_columns_list_parses_json_and_csv():
    validator = tk.get_validator("dimred_feature_columns_list")

    assert validator('["a", "b"]', {}) == ["a", "b"]
    assert validator("a, b ,c", {}) == ["a", "b", "c"]
    assert validator(["x", "y"], {}) == ["x", "y"]


@pytest.mark.usefixtures("with_plugins")
def test_feature_columns_list_invalid_type():
    validator = tk.get_validator("dimred_feature_columns_list")

    with pytest.raises(tk.Invalid):
        validator(123, {})


@pytest.mark.usefixtures("with_plugins")
def test_method_params_object_accepts_dict_and_json():
    validator = tk.get_validator("dimred_method_params_object")

    assert validator({"a": 1}, {}) == {"a": 1}
    assert validator('{"a": 1}', {}) == {"a": 1}
    assert validator("", {}) == ""


@pytest.mark.usefixtures("with_plugins")
def test_method_params_object_rejects_invalid_json():
    validator = tk.get_validator("dimred_method_params_object")

    with pytest.raises(tk.Invalid):
        validator("{bad json}", {})
    with pytest.raises(tk.Invalid):
        validator("[1, 2]", {})
