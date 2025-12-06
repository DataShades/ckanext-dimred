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
            raw = self.fetch_remote(self.filepath)
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

    def get_columns(self) -> list[str]:
        """Return column names without loading the full dataset where possible."""
        self.validate_size_limit()

        res_format = (self.resource.get("format") or "").lower()
        sep = "," if res_format == "csv" else "\t"

        buffer: io.BytesIO | str

        if self.remote and res_format in ("csv", "tsv"):
            sample = self.fetch_remote(self.filepath, max_bytes=128 * 1024)
            buffer = io.BytesIO(sample)
        elif self.remote:
            raw = self.fetch_remote(self.filepath)
            buffer = io.BytesIO(raw)
        else:
            buffer = self.filepath

        try:
            if res_format in ("csv", "tsv"):
                df = pd.read_csv(buffer, sep=sep, nrows=0, low_memory=False)
            elif res_format in ("xls", "xlsx"):
                df = pd.read_excel(buffer, nrows=0)
            else:
                df = pd.read_csv(buffer, nrows=0, low_memory=False)
        except (pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError) as err:
            log.warning("Column read fallback to full load due to %s", err)
            df = self.get_dataframe()

        return df.columns.tolist()
