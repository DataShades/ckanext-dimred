from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from typing import Any

from redis import exceptions as redis_exc

from ckan.lib.redis import connect_to_redis

from ckanext.dimred import config as dimred_config

log = logging.getLogger(__name__)


def _stable_dumps(data: dict[str, Any]) -> str:
    """Serialize data deterministically for hashing."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


class DimredCacheManager:
    """Small Redis-backed cache for dimred previews."""

    prefix = "ckanext:dimred:preview"

    def __init__(self) -> None:
        try:
            self.client = connect_to_redis()
        except (redis_exc.RedisError, OSError) as err:
            log.warning("Dimred cache disabled: cannot connect to redis (%s)", err)
            self.client = None

    @property
    def enabled(self) -> bool:
        return bool(self.client) and dimred_config.cache_enabled()

    @property
    def ttl(self) -> int:
        return dimred_config.cache_ttl()

    def settings_signature(self, settings: dict[str, Any]) -> str:
        payload = _stable_dumps(settings)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _key(self, resource_id: str, view_id: str, settings_sig: str) -> str:
        return f"{self.prefix}:{resource_id}:{view_id}:{settings_sig}"

    def get(self, resource_id: str, view_id: str, settings_sig: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        try:
            raw = self.client.get(self._key(resource_id, view_id, settings_sig))
            if not raw:
                return None
            data = json.loads(raw)
            if isinstance(data, dict) and "embedding" in data and "meta" in data:
                return data
        except (redis_exc.RedisError, json.JSONDecodeError, TypeError) as err:
            log.warning("Dimred cache get failed: %s", err)
        return None

    def save(self, resource_id: str, view_id: str, settings_sig: str, result: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            key = self._key(resource_id, view_id, settings_sig)
            payload = json.dumps(result)
            self.client.setex(key, self.ttl, payload)
        except (redis_exc.RedisError, TypeError, ValueError) as err:
            log.warning("Dimred cache save failed: %s", err)

    def delete_for_resource(self, resource_id: str) -> None:
        if not self.enabled:
            return
        pattern = f"{self.prefix}:{resource_id}:*"
        try:
            keys = list(self.client.scan_iter(match=pattern))
            if keys:
                self.client.delete(*keys)
        except redis_exc.RedisError as err:
            log.warning("Dimred cache delete failed: %s", err)


@lru_cache(maxsize=1)
def get_cache() -> DimredCacheManager:
    return DimredCacheManager()
