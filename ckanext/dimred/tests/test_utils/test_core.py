from __future__ import annotations

import numpy as np

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


def test_build_display_summary_combines_info_and_summary():
    meta = {
        "method": "umap",
        "prepare_info": {
            "n_rows_used": 10,
            "n_rows_original": 12,
            "n_features": 3,
            "numeric_used": ["a", "b"],
            "categorical_used": ["c"],
            "color_by": "label",
        },
        "method_params": {"n_components": 3},
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
    assert display["method"] == "umap"
    assert display["components"] == 3
    assert display["rows_used"] == 10
    assert display["numeric_sample"] == ["a", "b"]
    assert display["categorical_sample"] == ["c"]
