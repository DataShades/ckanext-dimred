from __future__ import annotations

from collections.abc import ItemsView, Iterable, Iterator, Mapping
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class Registry(Generic[K, V]):
    """Simple registry wrapper to store and extend mappings, similar to ckanext-unfold.

    It can be used as a central place to collect adapters via signals.
    """

    def __init__(self, mapping: Mapping[K, V] | None = None) -> None:
        self._mapping: dict[K, V] = dict(mapping or {})

    def get(self, key: K, default: V | None = None) -> V | None:
        return self._mapping.get(key, default)

    def register(self, key: K, value: V) -> None:
        self._mapping[key] = value

    def items(self) -> ItemsView[K, V]:
        return self._mapping.items()

    def update(self, other: Mapping[K, V] | Iterable[tuple[K, V]]) -> None:
        if isinstance(other, Mapping):
            self._mapping.update(other)
        else:
            self._mapping.update(dict(other))

    def __contains__(self, key: object) -> bool:
        return key in self._mapping

    def __iter__(self) -> Iterator[K]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)
