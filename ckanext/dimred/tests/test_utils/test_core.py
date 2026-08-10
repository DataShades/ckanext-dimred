from __future__ import annotations

import numpy as np
import pytest

from ckanext.dimred.utils import core


def test_embedding_summary_with_classes():
    embedding = np.array([[0.0, 1.0], [2.0, 3.0], [2.0, -1.0]])
    meta = {"prepare_info": {"color_by": "label", "color_values": ["a", "b", "a"]}}

    summary = core.embedding_summary(embedding, meta, top_n=2)

    assert summary["n_points"] == 3
    assert summary["n_dims"] == 2
    assert summary["dim_stats"][0]["min"] == 0.0
    assert summary["dim_stats"][0]["max"] == 2.0
    assert summary["n_classes"] == 2
    assert summary["top_classes"][0] == {"label": "a", "count": 2}


def test_embedding_summary_without_color():
    embedding = np.array([[1.0, 2.0]])
    meta = {"prepare_info": {}}

    summary = core.embedding_summary(embedding, meta)

    assert summary["n_classes"] is None
    assert summary["top_classes"] == []


def test_embedding_summary_does_not_treat_numeric_colors_as_classes():
    summary = core.embedding_summary(
        np.array([[0.0, 1.0], [2.0, 3.0]]),
        {"prepare_info": {"color_by": "score", "color_kind": "numeric", "color_values": [1.0, 2.0]}},
    )

    assert summary["n_classes"] is None
    assert summary["top_classes"] == []


def test_build_display_summary_combines_info_and_summary():
    meta = {
        "method": "pca",
        "prepare_info": {
            "n_rows_used": 10,
            "n_rows_original": 12,
            "n_rows_dropped": 2,
            "row_limit": 10,
            "sampling_method": "reservoir",
            "n_features": 3,
            "numeric_used": ["a", "b"],
            "categorical_used": ["c"],
            "color_by": "label",
            "skipped_columns": [
                {"name": "empty", "reason": "empty"},
                {"name": "mixed", "reason": "mixed"},
                {"name": "another-empty", "reason": "empty"},
            ],
        },
        "method_params": {"n_components": 3},
        "projection_info": {"explained_variance_ratio": [0.6, 0.3], "explained_variance_cumulative": 0.9},
    }
    summary = {
        "n_points": 10,
        "n_classes": 2,
        "color_by": "label",
        "top_classes": [{"label": "x", "count": 6}],
        "dim_stats": [{"name": "x", "min": 0, "max": 1}],
        "n_dims": 3,
    }

    display = core.build_display_summary(meta, summary)

    assert display["points"] == 10
    assert display["classes"] == 2
    assert display["method"] == "pca"
    assert display["components"] == 3
    assert display["rows_used"] == 10
    assert display["numeric_sample"] == ["a", "b"]
    assert display["categorical_sample"] == ["c"]
    assert display["sampling_method"] == "reservoir"
    assert display["rows_dropped"] == 2
    assert display["skipped_reasons"] == [{"reason": "empty", "count": 2}, {"reason": "mixed", "count": 1}]
    assert display["pca_variance"] == [0.6, 0.3]
    assert display["pca_variance_cumulative"] == 0.9
    assert display["warnings"] == ["sampled_rows", "numeric_standardized", "categorical_one_hot"]


def test_color_spec_uses_shared_categorical_palette_and_missing_marker():
    spec = core._color_spec(
        {
            "color_by": "label",
            "color_kind": "categorical",
            "color_values": ["first", None, "second", "first"],
            "color_candidates": [{"name": "label", "kind": "categorical", "unique_values": ["first", "second"]}],
        },
        4,
    )

    assert spec["kind"] == "categorical"
    assert spec["values"] == ["#5470c6", "#999999", "#91cc75", "#5470c6"]
    assert spec["legend"] == [
        {"label": "first", "color": "#5470c6"},
        {"label": "second", "color": "#91cc75"},
        {"label": "Missing", "color": "#999999"},
    ]


def test_color_spec_keeps_numeric_values_continuous_and_marks_missing():
    spec = core._color_spec(
        {"color_by": "score", "color_kind": "numeric", "color_values": [1.5, None, 3.5]},
        3,
    )

    assert spec["kind"] == "numeric"
    assert spec["values"][0] == 1.5
    assert np.isnan(spec["values"][1])
    assert spec["values"][2] == 3.5


def test_color_spec_falls_back_to_uniform_when_numeric_values_are_all_missing():
    spec = core._color_spec(
        {"color_by": "score", "color_kind": "numeric", "color_values": [None, float("nan")]},
        2,
    )

    assert spec == {"kind": "uniform", "values": "#333333", "legend": []}


def test_embedding_to_png_supports_numeric_color_scale_with_missing_values():
    image = core.embedding_to_png_data_url(
        np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
        {"prepare_info": {"color_by": "score", "color_kind": "numeric", "color_values": [1.0, None, 3.0]}},
    )

    assert image.startswith("data:image/png;base64,")


@pytest.mark.parametrize("zs", [None, np.array([0.0, 1.0, 2.0])], ids=["2d", "3d"])
def test_numeric_scatter_keeps_missing_color_values_visible(zs):
    plt = core.import_module("matplotlib.pyplot")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d") if zs is not None else fig.add_subplot(111)
    try:
        scatter = core._scatter(
            ax,
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0]),
            zs,
            core._color_spec(
                {"color_by": "score", "color_kind": "numeric", "color_values": [1.0, None, 3.0]},
                3,
            ),
        )
        fig.canvas.draw()

        assert len(scatter.get_offsets()) == 2
        assert len(ax.collections) == 2
        assert len(ax.collections[1].get_offsets()) == 1
    finally:
        plt.close(fig)
