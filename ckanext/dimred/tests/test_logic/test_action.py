from __future__ import annotations

import pathlib

import pytest

import ckan.plugins.toolkit as tk
from ckan.tests.helpers import call_action

from ckanext.dimred.adapters.tabular import TabularAdapter
from ckanext.dimred.exception import (
    DimredFeatureError,
    DimredNumericColumnError,
    DimredResourceSizeError,
)
from ckanext.dimred.logic import action as dimred_action

IRIS_CSV = pathlib.Path(__file__).resolve().parent.parent / "data" / "iris.csv"


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
def test_dimred_get_dimred_preview_pca(package, create_with_upload):
    with open(IRIS_CSV, "rb") as csv:
        resource = create_with_upload(csv.read(), "iris.csv", format="csv", package_id=package["id"])

    view = call_action(
        "resource_view_create",
        {},
        resource_id=resource["id"],
        view_type="dimred_view",
        title="Dimred",
        method="pca",
    )

    result = call_action("dimred_get_dimred_preview", id=resource["id"], view_id=view["id"])

    assert result["meta"]["method"] == "pca"


@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_dimred_get_dimred_preview_respects_n_components(package, create_with_upload):
    with open(IRIS_CSV, "rb") as csv:
        resource = create_with_upload(csv.read(), "iris.csv", format="csv", package_id=package["id"])

    view = call_action(
        "resource_view_create",
        {},
        resource_id=resource["id"],
        view_type="dimred_view",
        title="Dimred",
        method="pca",
        n_components=3,
    )

    result = call_action("dimred_get_dimred_preview", id=resource["id"], view_id=view["id"])

    assert result["meta"]["method_params"]["n_components"] == 3
    assert len(result["embedding"][0]) == 3


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


@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_dimred_pipeline_no_numeric_columns(package, create_with_upload):
    csv_content = "a,b\nfoo,bar\nbaz,qux\n"
    resource = create_with_upload(
        csv_content.encode("utf-8"), "non_numeric.csv", format="csv", package_id=package["id"]
    )

    view = call_action(
        "resource_view_create",
        {},
        resource_id=resource["id"],
        view_type="dimred_view",
        title="Dimred",
        method="umap",
    )

    with pytest.raises(DimredNumericColumnError):
        call_action("dimred_get_dimred_preview", id=resource["id"], view_id=view["id"])


@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_dimred_pipeline_feature_filter_removes_numeric(package, create_with_upload):
    csv_content = "num1,num2,cat\n1,2,x\n3,4,y\n"
    resource = create_with_upload(
        csv_content.encode("utf-8"), "feature_filter.csv", format="csv", package_id=package["id"]
    )

    view = call_action(
        "resource_view_create",
        {},
        resource_id=resource["id"],
        view_type="dimred_view",
        title="Dimred",
        method="umap",
        feature_columns=["cat"],
    )

    with pytest.raises(DimredNumericColumnError):
        call_action("dimred_get_dimred_preview", id=resource["id"], view_id=view["id"])


@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_dimred_pipeline_single_numeric_feature(package, create_with_upload):
    csv_content = "value\n1\n2\n3\n"
    resource = create_with_upload(
        csv_content.encode("utf-8"), "single_numeric.csv", format="csv", package_id=package["id"]
    )

    view = call_action(
        "resource_view_create",
        {},
        resource_id=resource["id"],
        view_type="dimred_view",
        title="Dimred",
        method="umap",
    )

    with pytest.raises(DimredFeatureError):
        call_action("dimred_get_dimred_preview", id=resource["id"], view_id=view["id"])


@pytest.mark.usefixtures("with_plugins")
@pytest.mark.ckan_config("ckanext.dimred.max_file_size_mb", "1")
def test_tabular_adapter_respects_size_limit(tmp_path):
    csv_path = tmp_path / "big.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    adapter = TabularAdapter(
        {"format": "csv", "size": 2 * 1024 * 1024},
        {},
        filepath=str(csv_path),
    )

    with pytest.raises(DimredResourceSizeError) as excinfo:
        adapter.get_dataframe()

    assert str(excinfo.value) == "1.0 MB"


@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_dimred_pipeline_disallowed_method(package, create_with_upload):
    csv_content = "x,y\n1,2\n3,4\n"
    resource = create_with_upload(csv_content.encode("utf-8"), "data.csv", format="csv", package_id=package["id"])

    with pytest.raises(tk.ValidationError) as excinfo:
        call_action(
            "resource_view_create",
            {},
            resource_id=resource["id"],
            view_type="dimred_view",
            title="Dimred",
            method="abc",
        )

    assert "Method 'abc' is not allowed." in excinfo.value.error_dict["method"][0]


@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_dimred_export_embedding(package, create_with_upload):
    with open(IRIS_CSV, "rb") as csv:
        resource = create_with_upload(csv.read(), "iris.csv", format="csv", package_id=package["id"])

    view = call_action(
        "resource_view_create",
        {},
        resource_id=resource["id"],
        view_type="dimred_view",
        title="Dimred",
        method="pca",
    )

    result = call_action("dimred_export_embedding", id=resource["id"], view_id=view["id"])

    assert result["filename"].endswith(".csv")
    assert "x" in result["content"]


@pytest.mark.usefixtures("clean_db", "with_plugins")
@pytest.mark.ckan_config("ckanext.dimred.export_enabled", "false")
def test_dimred_export_disabled(package, create_with_upload):
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

    with pytest.raises(tk.ValidationError):
        dimred_action.dimred_export_embedding({}, {"id": resource["id"], "view_id": view["id"]})
