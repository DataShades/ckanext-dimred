from __future__ import annotations

import csv
import io

import pytest

from ckanext.dimred.utils.export import embedding_to_csv


def test_embedding_to_csv_basic():
    csv_text = embedding_to_csv([[1.0, 2.0], [3.0, 4.0]], {"prepare_info": {}})
    lines = csv_text.strip().splitlines()

    assert lines[0] == "x,y"
    assert lines[1] == "1.0,2.0"
    assert lines[2] == "3.0,4.0"


def test_embedding_to_csv_with_color():
    meta = {"prepare_info": {"color_by": "label", "color_values": ["a", "b"]}}
    csv_text = embedding_to_csv([[1, 2], [3, 4]], meta)
    lines = csv_text.strip().splitlines()

    assert lines[0] == "x,y,label"
    assert lines[1].endswith(",a")
    assert lines[2].endswith(",b")


@pytest.mark.parametrize("value", ["=SUM(A1:A2)", "+cmd", "-formula", "@formula", " \t=SUM(A1:A2)"])
def test_embedding_to_csv_neutralizes_formula_like_color_values(value):
    csv_text = embedding_to_csv([[1.0, 2.0]], {"prepare_info": {"color_by": "label", "color_values": [value]}})

    rows = list(csv.reader(io.StringIO(csv_text)))

    assert rows == [["x", "y", "label"], ["1.0", "2.0", f"'{value}"]]


def test_embedding_to_csv_neutralizes_formula_like_color_header_but_preserves_coordinates():
    csv_text = embedding_to_csv(
        [[-1.5, 2.0]],
        {"prepare_info": {"color_by": "=label", "color_values": ["safe"]}},
    )

    rows = list(csv.reader(io.StringIO(csv_text)))

    assert rows == [["x", "y", "'=label"], ["-1.5", "2.0", "safe"]]
