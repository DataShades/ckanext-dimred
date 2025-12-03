from __future__ import annotations

import io
import logging

import pandas as pd

from ckanext.dimred.adapters.base import BaseAdapter
from ckanext.dimred.exception import DimredError

log = logging.getLogger(__name__)


class TabularAdapter(BaseAdapter):
    """Adapter for tabular resources (CSV, TSV, spreadsheets).

    It returns a pandas.DataFrame that will be further processed by the
    dimred pipeline.
    """

    def get_dataframe(self) -> pd.DataFrame:
        """Load the resource content into a pandas.DataFrame."""
        self.validate_size_limit()

        res_format = (self.resource.get("format") or "").lower()

        if self.remote:
            raw = self.make_request(self.filepath)
            buffer = io.BytesIO(raw)
        else:
            buffer = self.filepath

        try:
            if res_format in ("csv", "tsv"):
                sep = "," if res_format == "csv" else "\t"
                df = pd.read_csv(buffer, sep=sep, low_memory=False)
            elif res_format in ("xls", "xlsx"):
                df = pd.read_excel(buffer)
            else:
                df = pd.read_csv(buffer, low_memory=False)
        except Exception as e:
            raise DimredError(str(e)) from e

        return df
