from __future__ import annotations

from ckanext.dimred.adapters.base import BaseAdapter
from ckanext.dimred.adapters.tabular import TabularAdapter
from ckanext.dimred.types import Registry

ADAPTERS: dict[str, type[BaseAdapter]] = {
    "csv": TabularAdapter,
    "tsv": TabularAdapter,
    "xls": TabularAdapter,
    "xlsx": TabularAdapter,
}

adapter_registry: Registry[str, type[BaseAdapter]] = Registry(ADAPTERS)

__all__ = ["adapter_registry", "BaseAdapter", "TabularAdapter", "Registry"]
