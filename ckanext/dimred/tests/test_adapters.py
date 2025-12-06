from __future__ import annotations

import pandas as pd
import pytest

from ckanext.dimred.adapters import tabular


@pytest.mark.usefixtures("with_plugins")
def test_tabular_adapter_reads_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    adapter = tabular.TabularAdapter(
        {"format": "csv", "size": 10},
        {},
        filepath=str(csv_path),
    )

    df = adapter.get_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["a", "b"]


@pytest.mark.usefixtures("with_plugins")
def test_tabular_adapter_get_columns(tmp_path):
    csv_path = tmp_path / "cols.csv"
    csv_path.write_text("col1,col2,col3\n1,2,3\n", encoding="utf-8")

    adapter = tabular.TabularAdapter(
        {"format": "csv", "size": 10},
        {},
        filepath=str(csv_path),
    )

    cols = adapter.get_columns()

    assert cols == ["col1", "col2", "col3"]
