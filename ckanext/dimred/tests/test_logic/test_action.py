from __future__ import annotations

import csv
import io
import json
import pathlib

import numpy as np
import pytest
from redis import exceptions as redis_exc

import ckan.plugins.toolkit as tk
from ckan.lib import jobs as ckan_jobs
from ckan.tests import factories
from ckan.tests.helpers import call_action

from ckanext.dimred.adapters.tabular import TabularAdapter
from ckanext.dimred.exception import (
    DimredFeatureError,
    DimredNumericColumnError,
    DimredPreviewError,
    DimredResourceSizeError,
)
from ckanext.dimred.logic import action as dimred_action

pytest_plugins = ["ckanext.datastore.tests.conftest"]

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


def _run_dimred_worker():
    """Run queued dimred work inline through CKAN's no-fork RQ test worker."""
    worker = ckan_jobs.Worker([dimred_action.PREVIEW_QUEUE])
    worker.work(burst=True)


def _build_dimred_preview(resource_id, view_id):
    """Exercise the internal reducer directly in focused pipeline tests."""
    return dimred_action._compute_dimred_preview({}, {"id": resource_id, "view_id": view_id})


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

    result = _build_dimred_preview(resource["id"], view["id"])

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

    result = _build_dimred_preview(resource["id"], view["id"])

    assert result["meta"]["method"] == "pca"


@pytest.mark.usefixtures("clean_db", "clean_redis", "with_plugins", "with_test_worker")
def test_dimred_background_preview_is_deduplicated_and_becomes_ready(package, create_with_upload):
    with open(IRIS_CSV, "rb") as csv:
        resource = create_with_upload(csv.read(), "iris.csv", format="csv", package_id=package["id"])
    view = _create_dimred_view(resource["id"], method="pca")
    data_dict = {"id": resource["id"], "view_id": view["id"]}

    first = tk.get_action("dimred_start_preview")({}, data_dict)
    second = tk.get_action("dimred_start_preview")({}, data_dict)

    assert first["status"] == "pending"
    assert second == first
    assert len(ckan_jobs.get_queue(dimred_action.PREVIEW_QUEUE).jobs) == 1

    _run_dimred_worker()

    status = tk.get_action("dimred_get_preview_status")({}, {**data_dict, "job_id": first["job_id"]})
    assert status["status"] == "ready"
    assert status["result"]["meta"]["method"] == "pca"

    cached = tk.get_action("dimred_start_preview")({}, data_dict)
    assert cached["status"] == "ready"
    assert cached["result"] == status["result"]
    assert not ckan_jobs.get_queue(dimred_action.PREVIEW_QUEUE).jobs


@pytest.mark.usefixtures("clean_db", "clean_redis", "with_plugins", "with_test_worker")
def test_dimred_background_preview_reports_sanitized_failure(package, create_with_upload):
    resource = create_with_upload(b"label\na\nb\nc\n", "labels.csv", format="csv", package_id=package["id"])
    view = _create_dimred_view(resource["id"], method="pca")
    data_dict = {"id": resource["id"], "view_id": view["id"]}

    started = tk.get_action("dimred_start_preview")({}, data_dict)
    _run_dimred_worker()
    status = tk.get_action("dimred_get_preview_status")({}, {**data_dict, "job_id": started["job_id"]})

    assert status == {
        "status": "failed",
        "job_id": started["job_id"],
        "error": "No numeric columns found for dimred processing.",
        "retryable": False,
    }
    assert tk.get_action("dimred_start_preview")({}, data_dict) == status
    assert not ckan_jobs.get_queue(dimred_action.PREVIEW_QUEUE).jobs


@pytest.mark.usefixtures("clean_db", "clean_redis", "with_plugins")
def test_dimred_background_preview_reports_redis_errors(package, create_with_upload, monkeypatch):
    resource = create_with_upload(b"x,y\n1,2\n3,4\n", "rows.csv", format="csv", package_id=package["id"])
    view = _create_dimred_view(resource["id"], method="pca")

    def unavailable_job_store(job_id):
        raise redis_exc.ConnectionError

    monkeypatch.setattr(dimred_action.ckan_jobs, "job_from_id", unavailable_job_store)

    with pytest.raises(DimredPreviewError):
        tk.get_action("dimred_start_preview")({}, {"id": resource["id"], "view_id": view["id"]})


@pytest.mark.parametrize("status", ["pending", "finished"])
def test_preview_job_response_normalizes_redis_errors(status):
    class UnavailableJob:
        id = "dimred-preview-test"

        def get_status(self, refresh):
            if status == "pending":
                raise redis_exc.ConnectionError
            return status

        def return_value(self):
            raise redis_exc.ConnectionError

    with pytest.raises(DimredPreviewError):
        dimred_action._preview_job_response(UnavailableJob())


def test_preview_job_response_supports_legacy_rq_return_value_property():
    result = {"embedding": [[1.0, 2.0]], "meta": {"method": "pca"}}

    class LegacyJob:
        id = "dimred-preview-test"
        return_value = result

        def get_status(self, refresh):
            return "finished"

    assert dimred_action._preview_job_response(LegacyJob()) == {
        "status": "ready",
        "job_id": LegacyJob.id,
        "result": result,
    }


@pytest.mark.usefixtures("clean_db", "clean_redis", "with_plugins", "with_test_worker")
def test_dimred_background_preview_keeps_private_resource_authorization(create_with_upload):
    owner = factories.User()
    outsider = factories.User()
    organization = factories.Organization(user=owner)
    package = factories.Dataset(user=owner, owner_org=organization["id"], private=True)
    context = {"user": owner["name"]}
    resource = create_with_upload(
        b"x,y\n1,2\n3,4\n",
        "private.csv",
        format="csv",
        package_id=package["id"],
        context=context,
    )
    view = call_action(
        "resource_view_create",
        context,
        resource_id=resource["id"],
        view_type="dimred_view",
        title="Dimred",
        method="pca",
    )
    data_dict = {"id": resource["id"], "view_id": view["id"]}

    started = tk.get_action("dimred_start_preview")(context, data_dict)
    _run_dimred_worker()

    owner_status = tk.get_action("dimred_get_preview_status")(context, {**data_dict, "job_id": started["job_id"]})
    assert owner_status["status"] == "ready"
    with pytest.raises(tk.NotAuthorized):
        tk.get_action("dimred_get_preview_status")(
            {"user": outsider["name"]},
            {**data_dict, "job_id": started["job_id"]},
        )


@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_dimred_get_dimred_preview_tsne(package, create_with_upload):
    with open(IRIS_CSV, "rb") as csv:
        resource = create_with_upload(csv.read(), "iris.csv", format="csv", package_id=package["id"])

    view = _create_dimred_view(resource["id"], method="tsne", method_params={"perplexity": 5})

    result = _build_dimred_preview(resource["id"], view["id"])

    assert result["meta"]["method"] == "tsne"


@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_dimred_get_dimred_preview_rejects_view_from_another_resource(package, create_with_upload):
    csv_content = b"x,y\n1,2\n3,4\n4,5\n"
    resource = create_with_upload(csv_content, "first.csv", format="csv", package_id=package["id"])
    other_resource = create_with_upload(csv_content, "second.csv", format="csv", package_id=package["id"])
    view = _create_dimred_view(other_resource["id"], method="pca")

    with pytest.raises(tk.ValidationError) as excinfo:
        _build_dimred_preview(resource["id"], view["id"])

    assert excinfo.value.error_dict["view_id"] == ["Resource view does not belong to the specified resource."]


@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_dimred_get_dimred_preview_rejects_unknown_method_parameter(package, create_with_upload):
    csv_content = b"x,y\n1,2\n3,4\n4,5\n"
    resource = create_with_upload(csv_content, "data.csv", format="csv", package_id=package["id"])
    view = _create_dimred_view(resource["id"], method="pca", method_params={"unexpected": 1})

    with pytest.raises(tk.ValidationError) as excinfo:
        _build_dimred_preview(resource["id"], view["id"])

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
        _build_dimred_preview(resource["id"], view["id"])

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
        _build_dimred_preview(resource["id"], view["id"])

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

    result = _build_dimred_preview(resource["id"], view["id"])

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

    result = _build_dimred_preview(resource["id"], view["id"])

    info = result["meta"]["prepare_info"]
    assert info["color_by"] == "Species"
    assert info["feature_columns"] == ["Sepal.Length", "Sepal.Width"]
    assert info["numeric_used"] == ["Sepal.Length", "Sepal.Width"]


@pytest.mark.usefixtures("clean_db", "with_plugins")
@pytest.mark.parametrize(
    ("view_settings", "field", "message"),
    [
        ({"feature_columns": ["x", "missing"]}, "feature_columns", "Unknown feature column(s): missing."),
        ({"color_by": "missing"}, "color_by", "Unknown color column: missing."),
    ],
)
def test_dimred_get_dimred_preview_rejects_unknown_columns(
    package, create_with_upload, view_settings, field, message
):
    csv_content = b"x,y,label\n1,2,a\n3,4,b\n5,6,c\n"
    resource = create_with_upload(csv_content, "data.csv", format="csv", package_id=package["id"])
    view = _create_dimred_view(resource["id"], method="pca", **view_settings)

    with pytest.raises(tk.ValidationError) as excinfo:
        _build_dimred_preview(resource["id"], view["id"])

    assert excinfo.value.error_dict[field] == [message]


@pytest.mark.usefixtures("clean_db", "with_plugins")
@pytest.mark.parametrize(
    ("csv_content", "expected_default", "expected_explicit"),
    [
        (
            b"x,y,color\n1,2,10\n3,4,20\n5,6,30\n",
            {"numeric_used": ["x", "y"], "categorical_used": []},
            {"numeric_used": ["x", "y", "color"], "categorical_used": []},
        ),
        (
            b"x,y,color\n1,2,a\n3,4,b\n5,6,a\n",
            {"numeric_used": ["x", "y"], "categorical_used": []},
            {"numeric_used": ["x", "y"], "categorical_used": ["color"]},
        ),
    ],
)
def test_dimred_color_by_is_included_only_when_explicitly_selected(
    package, create_with_upload, csv_content, expected_default, expected_explicit
):
    resource = create_with_upload(csv_content, "data.csv", format="csv", package_id=package["id"])
    default_view = _create_dimred_view(resource["id"], method="pca", color_by="color")
    explicit_view = _create_dimred_view(
        resource["id"], method="pca", color_by="color", feature_columns=["x", "y", "color"]
    )

    default_info = _build_dimred_preview(resource["id"], default_view["id"])["meta"][
        "prepare_info"
    ]
    explicit_info = _build_dimred_preview(resource["id"], explicit_view["id"])["meta"][
        "prepare_info"
    ]

    assert {key: default_info[key] for key in expected_default} == expected_default
    assert {key: explicit_info[key] for key in expected_explicit} == expected_explicit
    assert default_info["feature_columns"] == ["x", "y"]
    assert explicit_info["feature_columns"] == ["x", "y", "color"]


@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_dimred_get_dimred_preview_rejects_explicit_high_cardinality_categorical_feature(
    package, create_with_upload
):
    max_categories = dimred_action.dimred_config.max_categories_for_ohe()
    rows = ["x,y,color"]
    rows.extend(f"{index},{index * 2},category-{index}" for index in range(max_categories + 1))
    resource = create_with_upload("\n".join(rows).encode(), "data.csv", format="csv", package_id=package["id"])
    view = _create_dimred_view(
        resource["id"], method="pca", feature_columns=["x", "y", "color"], color_by="color"
    )

    with pytest.raises(tk.ValidationError) as excinfo:
        _build_dimred_preview(resource["id"], view["id"])

    assert excinfo.value.error_dict["feature_columns"] == [
        f"Feature column 'color' has {max_categories + 1} categories; maximum is {max_categories}."
    ]


@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_dimred_get_dimred_preview_normalizes_numeric_gaps_and_skips_unsupported_columns(
    package, create_with_upload
):
    csv_content = (
        b"x,y,color,empty,mixed,when,flag\n"
        b"1,2,10,,1,2024-01-01,true\n"
        b",inf,inf,,oops,2024-01-02,false\n"
        b"3,4,30,,3,2024-01-03,true\n"
    )
    resource = create_with_upload(csv_content, "data.csv", format="csv", package_id=package["id"])
    view = _create_dimred_view(resource["id"], method="pca", color_by="color")

    result = _build_dimred_preview(resource["id"], view["id"])

    info = result["meta"]["prepare_info"]
    assert np.isfinite(np.asarray(result["embedding"])).all()
    json.dumps(result, allow_nan=False)
    assert info["numeric_used"] == ["x", "y"]
    assert info["categorical_used"] == ["flag"]
    assert info["color_values"] == [10.0, None, 30.0]
    assert info["skipped_columns"] == [
        {"name": "color", "reason": "used for color only"},
        {"name": "empty", "reason": "empty"},
        {"name": "mixed", "reason": "mixed"},
        {"name": "when", "reason": "datetime"},
    ]


@pytest.mark.usefixtures("clean_db", "with_plugins")
@pytest.mark.parametrize(
    ("csv_content", "column", "message"),
    [
        (
            b"x,y,mixed\n1,2,1\n3,4,oops\n5,6,3\n",
            "mixed",
            "Feature column 'mixed' contains mixed numeric and text values.",
        ),
        (
            b"x,y,when\n1,2,31/01/2024\n3,4,01/02/2024\n5,6,02/03/2024\n",
            "when",
            "Feature column 'when' is a datetime column, which is not supported.",
        ),
        (
            b"x,y,empty\n1,2,\n3,4,\n5,6,\n",
            "empty",
            "Feature column 'empty' has no finite values.",
        ),
    ],
)
def test_dimred_get_dimred_preview_rejects_explicit_unsupported_feature_columns(
    package, create_with_upload, csv_content, column, message
):
    resource = create_with_upload(csv_content, "data.csv", format="csv", package_id=package["id"])
    view = _create_dimred_view(resource["id"], method="pca", feature_columns=["x", "y", column])

    with pytest.raises(tk.ValidationError) as excinfo:
        _build_dimred_preview(resource["id"], view["id"])

    assert excinfo.value.error_dict["feature_columns"] == [message]


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

    result = _build_dimred_preview(resource["id"], view["id"])

    prepare = result["meta"]["prepare_info"]
    candidates = prepare.get("color_candidates") or []
    assert candidates

    names = {c["name"] for c in candidates if c.get("name")}
    assert "Species" in names
    assert "Sepal.Length" in names

    species = next(c for c in candidates if c.get("name") == "Species")
    assert species["kind"] == "categorical"
    assert "values" not in species
    assert set(species["unique_values"]) == {"setosa", "versicolor", "virginica"}
    assert len(prepare["color_values"]) == prepare["n_rows_used"]

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

    result = _build_dimred_preview(resource["id"], view["id"])
    prepare = result["meta"]["prepare_info"]
    candidates = prepare.get("color_candidates") or []

    names = {c["name"] for c in candidates if c.get("name")}
    assert "color_col" in names
    assert "low_cat" in names
    assert "high_cat" not in names

    color_candidate = next(c for c in candidates if c.get("name") == "color_col")
    assert color_candidate["kind"] == "categorical"
    assert "values" not in color_candidate
    assert len(color_candidate["unique_values"]) <= dimred_action.dimred_config.max_categories_for_ohe()

    low_cat_candidate = next(c for c in candidates if c.get("name") == "low_cat")
    assert low_cat_candidate["kind"] == "categorical"
    assert set(low_cat_candidate["unique_values"]) == {"group0", "group1"}


@pytest.mark.usefixtures("with_plugins")
@pytest.mark.ckan_config("ckan.plugins", "dimred")
def test_dimred_get_dimred_preview_validation_error():
    with pytest.raises(tk.ValidationError):
        dimred_action._compute_dimred_preview({}, {})


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
        _build_dimred_preview(resource["id"], view["id"])


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
        _build_dimred_preview(resource["id"], view["id"])


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
        _build_dimred_preview(resource["id"], view["id"])


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


@pytest.mark.usefixtures("clean_db", "clean_redis", "with_plugins", "with_test_worker")
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

    with pytest.raises(tk.ValidationError) as excinfo:
        call_action("dimred_export_embedding", id=resource["id"], view_id=view["id"])

    assert excinfo.value.error_dict["preview"] == ["Dimred preview is still being prepared."]
    assert not ckan_jobs.get_queue(dimred_action.PREVIEW_QUEUE).jobs
    tk.get_action("dimred_start_preview")({}, {"id": resource["id"], "view_id": view["id"]})
    _run_dimred_worker()

    result = call_action("dimred_export_embedding", id=resource["id"], view_id=view["id"])

    assert result["filename"].endswith(".csv")
    assert "x" in result["content"]


@pytest.mark.usefixtures("clean_db", "with_plugins")
@pytest.mark.ckan_config("ckanext.dimred.max_rows", 4)
@pytest.mark.ckan_config("ckanext.dimred.pca.max_rows", 2)
def test_dimred_preview_preserves_file_row_ids_through_sampling_and_export(package, create_with_upload):
    csv_content = b"x,y,label\n1,11,row-1\n2,12,row-2\n3,13,row-3\n4,14,row-4\n"
    resource = create_with_upload(csv_content, "rows.csv", format="csv", package_id=package["id"])
    view = _create_dimred_view(resource["id"], method="pca", color_by="label")

    result = _build_dimred_preview(resource["id"], view["id"])
    prepare_info = result["meta"]["prepare_info"]

    assert prepare_info["n_rows_original"] == 4
    assert prepare_info["n_rows_used"] == 2
    assert prepare_info["row_limit"] == 2
    assert set(zip(prepare_info["source_row_ids"], prepare_info["color_values"], strict=True)) == {
        (2, "row-2"),
        (4, "row-4"),
    }

    export = call_action("dimred_export_embedding", id=resource["id"], view_id=view["id"])
    rows = list(csv.reader(io.StringIO(export["content"])))

    assert rows[0] == ["x", "y", "source_row_id", "label"]
    assert {(int(row[2]), row[3]) for row in rows[1:]} == {(2, "row-2"), (4, "row-4")}


@pytest.mark.usefixtures("clean_db", "with_plugins")
@pytest.mark.ckan_config("ckanext.dimred.max_rows", 4)
@pytest.mark.ckan_config("ckanext.dimred.pca.max_rows", 2)
def test_dimred_color_values_are_aligned_with_sampled_file_rows(package, create_with_upload):
    csv_content = b"x,y,label\n1,11,row-1\n2,12,row-2\n3,13,row-3\n4,14,row-4\n"
    resource = create_with_upload(csv_content, "rows.csv", format="csv", package_id=package["id"])
    view = _create_dimred_view(resource["id"], method="pca", color_by="label")

    preview = _build_dimred_preview(resource["id"], view["id"])
    colors = call_action("dimred_get_dimred_color_values", id=resource["id"], view_id=view["id"], column="label")

    assert colors == {
        "column": "label",
        "kind": "categorical",
        "values": ["row-2", "row-4"],
        "source_row_ids": preview["meta"]["prepare_info"]["source_row_ids"],
    }


@pytest.mark.usefixtures("clean_datastore", "clean_redis", "with_plugins", "with_test_worker")
@pytest.mark.ckan_config("ckan.plugins", "datastore dimred")
def test_dimred_preview_uses_datastore_ids_and_export(package, create_with_upload):
    resource = create_with_upload(b"x,y,label\n", "rows.csv", format="csv", package_id=package["id"])
    call_action(
        "datastore_create",
        {},
        resource_id=resource["id"],
        force=True,
        records=[
            {"x": 1, "y": 11, "label": "row-1"},
            {"x": 2, "y": 12, "label": "row-2"},
            {"x": 3, "y": 13, "label": "row-3"},
        ],
    )
    view = _create_dimred_view(resource["id"], method="pca", color_by="label")

    started = tk.get_action("dimred_start_preview")({}, {"id": resource["id"], "view_id": view["id"]})
    _run_dimred_worker()
    result = tk.get_action("dimred_get_preview_status")(
        {},
        {"id": resource["id"], "view_id": view["id"], "job_id": started["job_id"]},
    )["result"]
    prepare_info = result["meta"]["prepare_info"]

    assert prepare_info["source_row_ids"] == [1, 2, 3]
    assert prepare_info["color_values"] == ["row-1", "row-2", "row-3"]

    export = call_action("dimred_export_embedding", id=resource["id"], view_id=view["id"])
    rows = list(csv.reader(io.StringIO(export["content"])))

    assert rows[0] == ["x", "y", "source_row_id", "label"]
    assert [row[2:] for row in rows[1:]] == [["1", "row-1"], ["2", "row-2"], ["3", "row-3"]]

    colors = call_action("dimred_get_dimred_color_values", id=resource["id"], view_id=view["id"], column="label")

    assert colors == {
        "column": "label",
        "kind": "categorical",
        "values": ["row-1", "row-2", "row-3"],
        "source_row_ids": [1, 2, 3],
    }


@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_dimred_color_values_reject_unavailable_column(package, create_with_upload):
    resource = create_with_upload(b"x,y\n1,2\n3,4\n", "rows.csv", format="csv", package_id=package["id"])
    view = _create_dimred_view(resource["id"], method="pca")

    with pytest.raises(tk.ValidationError) as excinfo:
        call_action(
            "dimred_get_dimred_color_values",
            id=resource["id"],
            view_id=view["id"],
            column="missing",
        )

    assert excinfo.value.error_dict["column"] == ["Column 'missing' is not available for coloring."]


@pytest.mark.usefixtures("clean_datastore", "with_plugins")
@pytest.mark.ckan_config("ckan.plugins", "datastore dimred")
@pytest.mark.ckan_config("ckanext.dimred.max_rows", 4)
@pytest.mark.ckan_config("ckanext.dimred.pca.max_rows", 2)
def test_dimred_preview_passes_method_limit_to_datastore(package, create_with_upload):
    resource = create_with_upload(b"x,y\n", "rows.csv", format="csv", package_id=package["id"])
    call_action(
        "datastore_create",
        {},
        resource_id=resource["id"],
        force=True,
        records=[{"x": 1, "y": 11}, {"x": 2, "y": 12}, {"x": 3, "y": 13}],
    )
    view = _create_dimred_view(resource["id"], method="pca")

    result = _build_dimred_preview(resource["id"], view["id"])
    prepare_info = result["meta"]["prepare_info"]

    assert prepare_info["n_rows_original"] == 3
    assert prepare_info["n_rows_used"] == 2
    assert prepare_info["row_limit"] == 2
    assert prepare_info["source_row_ids"] == [1, 2]


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
