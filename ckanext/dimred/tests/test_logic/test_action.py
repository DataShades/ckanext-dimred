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


def _create_dimred_view(resource_id, **settings):
    return call_action(
        "resource_view_create",
        {},
        resource_id=resource_id,
        view_type="dimred_view",
        title="Dimred",
        **settings,
    )


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
def test_dimred_get_dimred_preview_tsne(package, create_with_upload):
    with open(IRIS_CSV, "rb") as csv:
        resource = create_with_upload(csv.read(), "iris.csv", format="csv", package_id=package["id"])

    view = _create_dimred_view(resource["id"], method="tsne", method_params={"perplexity": 5})

    result = call_action("dimred_get_dimred_preview", id=resource["id"], view_id=view["id"])

    assert result["meta"]["method"] == "tsne"


@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_dimred_get_dimred_preview_rejects_view_from_another_resource(package, create_with_upload):
    csv_content = b"x,y\n1,2\n3,4\n4,5\n"
    resource = create_with_upload(csv_content, "first.csv", format="csv", package_id=package["id"])
    other_resource = create_with_upload(csv_content, "second.csv", format="csv", package_id=package["id"])
    view = _create_dimred_view(other_resource["id"], method="pca")

    with pytest.raises(tk.ValidationError) as excinfo:
        call_action("dimred_get_dimred_preview", id=resource["id"], view_id=view["id"])

    assert excinfo.value.error_dict["view_id"] == ["Resource view does not belong to the specified resource."]


@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_dimred_get_dimred_preview_rejects_unknown_method_parameter(package, create_with_upload):
    csv_content = b"x,y\n1,2\n3,4\n4,5\n"
    resource = create_with_upload(csv_content, "data.csv", format="csv", package_id=package["id"])
    view = _create_dimred_view(resource["id"], method="pca", method_params={"unexpected": 1})

    with pytest.raises(tk.ValidationError) as excinfo:
        call_action("dimred_get_dimred_preview", id=resource["id"], view_id=view["id"])

    assert excinfo.value.error_dict["method_params"] == ["Unsupported pca parameter(s): unexpected."]


@pytest.mark.usefixtures("clean_db", "with_plugins")
@pytest.mark.parametrize(
    ("method", "method_params", "message"),
    [
        ("pca", {"whiten": "true"}, "whiten must be a boolean."),
        ("tsne", {"perplexity": 0}, "perplexity must be greater than 0."),
        ("umap", {"n_neighbors": 2.5}, "n_neighbors must be an integer of at least 2."),
        ("umap", {"min_dist": 1.1}, "min_dist must be between 0 and 1."),
    ],
)
def test_dimred_get_dimred_preview_rejects_invalid_method_parameters(
    package, create_with_upload, method, method_params, message
):
    csv_content = b"x,y\n1,2\n3,4\n4,5\n"
    resource = create_with_upload(csv_content, "data.csv", format="csv", package_id=package["id"])
    view = _create_dimred_view(resource["id"], method=method, method_params=method_params)

    with pytest.raises(tk.ValidationError) as excinfo:
        call_action("dimred_get_dimred_preview", id=resource["id"], view_id=view["id"])

    assert excinfo.value.error_dict["method_params"] == [message]


@pytest.mark.usefixtures("clean_db", "with_plugins")
@pytest.mark.parametrize(
    ("method", "method_params", "message"),
    [
        ("pca", {"n_components": 3}, "PCA n_components cannot exceed the number of rows or features."),
        ("tsne", {}, "t-SNE perplexity must be smaller than the number of rows."),
        ("umap", {"n_neighbors": 3}, "UMAP n_neighbors must be smaller than the number of rows."),
    ],
)
def test_dimred_get_dimred_preview_rejects_parameters_incompatible_with_data(
    package, create_with_upload, method, method_params, message
):
    csv_content = b"x,y\n1,2\n3,4\n4,5\n"
    resource = create_with_upload(csv_content, "data.csv", format="csv", package_id=package["id"])
    view = _create_dimred_view(resource["id"], method=method, method_params=method_params)

    with pytest.raises(tk.ValidationError) as excinfo:
        call_action("dimred_get_dimred_preview", id=resource["id"], view_id=view["id"])

    assert excinfo.value.error_dict["method_params" if method != "pca" else "n_components"] == [message]


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


@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_prepare_info_color_candidates(package, create_with_upload):
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
    )

    result = call_action("dimred_get_dimred_preview", id=resource["id"], view_id=view["id"])

    prepare = result["meta"]["prepare_info"]
    candidates = prepare.get("color_candidates") or []
    assert candidates

    names = {c["name"] for c in candidates if c.get("name")}
    assert "Species" in names
    assert "Sepal.Length" in names

    species = next(c for c in candidates if c.get("name") == "Species")
    assert species["kind"] == "categorical"
    assert len(species["values"]) == prepare["n_rows_used"]
    assert set(species["unique_values"]) == {"setosa", "versicolor", "virginica"}

    sepal_length = next(c for c in candidates if c.get("name") == "Sepal.Length")
    assert sepal_length["kind"] == "numeric"
    assert sepal_length["min"] < sepal_length["max"]


@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_color_candidates_respect_cardinality_limits(package, create_with_upload):
    rows = ["num1,num2,color_col,low_cat,high_cat"]
    rows.extend(f"{idx},{idx * 2},label{idx},group{idx % 2},skip{idx}" for idx in range(60))
    csv_content = "\n".join(rows)

    resource = create_with_upload(csv_content.encode("utf-8"), "colors.csv", format="csv", package_id=package["id"])

    view = call_action(
        "resource_view_create",
        {},
        resource_id=resource["id"],
        view_type="dimred_view",
        title="Dimred",
        method="umap",
        color_by="color_col",
    )

    result = call_action("dimred_get_dimred_preview", id=resource["id"], view_id=view["id"])
    prepare = result["meta"]["prepare_info"]
    candidates = prepare.get("color_candidates") or []

    names = {c["name"] for c in candidates if c.get("name")}
    assert "color_col" in names
    assert "low_cat" in names
    assert "high_cat" not in names

    color_candidate = next(c for c in candidates if c.get("name") == "color_col")
    assert color_candidate["kind"] == "categorical"
    assert len(color_candidate["values"]) == prepare["n_rows_used"]
    assert len(color_candidate["unique_values"]) <= dimred_action.dimred_config.max_categories_for_ohe()

    low_cat_candidate = next(c for c in candidates if c.get("name") == "low_cat")
    assert low_cat_candidate["kind"] == "categorical"
    assert set(low_cat_candidate["unique_values"]) == {"group0", "group1"}


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
