from __future__ import annotations

import pathlib

import pytest

import ckan.plugins.toolkit as tk
from ckan.tests.helpers import call_action

IRIS_CSV = pathlib.Path(__file__).parent / "data" / "iris.csv"


@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_dimred_get_dimred_preview_runs_pipeline(package, create_with_upload):
    with open(IRIS_CSV, "rb") as csv:
        resource = create_with_upload(csv.read(), "iris.csv", format="csv", package_id=package["id"])

    view = call_action(
        "resource_view_create",
        {},
        resource_id=resource["id"],
        view_type="dimred_view",
        title="Dimred",
        method="umap",
    )

    result = call_action("dimred_get_dimred_preview", id=resource["id"], view_id=view["id"])

    assert "embedding" in result
    assert "meta" in result
    assert result["meta"]["method"] == "umap"
    prepare = result["meta"]["prepare_info"]
    assert prepare["n_rows_used"] > 0
    assert prepare["n_features"] >= 2


@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_dimred_get_dimred_preview_color_and_features(package, create_with_upload):
    with open(IRIS_CSV, "rb") as csv:
        resource = create_with_upload(csv.read(), "iris.csv", format="csv", package_id=package["id"])

    view = call_action(
        "resource_view_create",
        {},
        resource_id=resource["id"],
        view_type="dimred_view",
        title="Dimred",
        method="umap",
        color_by="Species",
        feature_columns=["Sepal.Length", "Sepal.Width"],
    )

    result = call_action("dimred_get_dimred_preview", id=resource["id"], view_id=view["id"])

    info = result["meta"]["prepare_info"]
    assert info["color_by"] == "Species"
    assert info["feature_columns"] == ["Sepal.Length", "Sepal.Width"]
    assert info["numeric_used"] == ["Sepal.Length", "Sepal.Width"]


@pytest.mark.usefixtures("with_plugins")
@pytest.mark.ckan_config("ckan.plugins", "dimred")
def test_dimred_get_dimred_preview_validation_error():
    with pytest.raises(tk.ValidationError):
        call_action("dimred_get_dimred_preview", {})
