from __future__ import annotations

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
