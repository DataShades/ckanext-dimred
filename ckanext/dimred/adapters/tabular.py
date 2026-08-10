from __future__ import annotations

import io
import logging
import random
from collections.abc import Iterable

import pandas as pd

from ckanext.dimred.adapters.base import BaseAdapter
from ckanext.dimred.exception import DimredError

log = logging.getLogger(__name__)

CSV_CHUNK_ROWS = 10_000
SAMPLE_RANDOM_SEED = 42


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

    def get_sampled_dataframe(self, row_limit: int) -> tuple[pd.DataFrame, list[int], int]:
        """Read CSV/TSV incrementally and retain a deterministic reservoir sample."""
        res_format = (self.resource.get("format") or "").lower()
        if res_format not in {"csv", "tsv"}:
            return super().get_sampled_dataframe(row_limit)

        self.validate_size_limit()
        buffer: io.BytesIO | str = io.BytesIO(self.fetch_remote(self.filepath)) if self.remote else self.filepath

        try:
            reader = pd.read_csv(
                buffer,
                sep="," if res_format == "csv" else "\t",
                chunksize=CSV_CHUNK_ROWS,
                low_memory=False,
            )
            return _reservoir_sample(reader, row_limit)
        except Exception as err:
            raise DimredError(str(err)) from err

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


def _reservoir_sample(
    chunks: Iterable[pd.DataFrame],
    row_limit: int,
) -> tuple[pd.DataFrame, list[int], int]:
    """Build a deterministic reservoir sample without retaining all input rows."""
    rng = random.Random(SAMPLE_RANDOM_SEED)  # noqa: S311 - deterministic sampling is not security-sensitive.
    records: list[tuple[object, ...]] = []
    source_row_ids: list[int] = []
    columns: list[str] | None = None
    n_rows_original = 0

    for chunk in chunks:
        columns = [str(column) for column in chunk.columns]
        for row in chunk.itertuples(index=False, name=None):
            n_rows_original += 1
            if len(records) < row_limit:
                records.append(row)
                source_row_ids.append(n_rows_original)
                continue

            replacement_index = rng.randrange(n_rows_original)
            if replacement_index < row_limit:
                records[replacement_index] = row
                source_row_ids[replacement_index] = n_rows_original

    return pd.DataFrame.from_records(records, columns=columns), source_row_ids, n_rows_original
