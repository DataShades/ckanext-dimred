from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from typing import Any
from urllib.parse import quote

from redis import exceptions as redis_exc

from ckan.lib.redis import connect_to_redis
from ckan.plugins import toolkit as tk

from ckanext.dimred import config as dimred_config

log = logging.getLogger(__name__)


def _stable_dumps(data: dict[str, Any]) -> str:
    """Serialize data deterministically for hashing."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def serialize_preview_result(result: dict[str, Any]) -> str:
    """Serialize a preview result in the compact format used for its payload budget."""
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


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
        site_id = quote(str(tk.config.get("ckan.site_id", "default")).strip() or "default", safe="")
        return f"{self.prefix}:{site_id}:{resource_id}:{view_id}:{settings_sig}"

    def _job_lock_key(self, job_id: str) -> str:
        site_id = quote(str(tk.config.get("ckan.site_id", "default")).strip() or "default", safe="")
        return f"{self.prefix}:{site_id}:job-lock:{quote(job_id, safe='')}"

    def get(self, resource_id: str, view_id: str, settings_sig: str) -> dict[str, Any] | None:
        client = self.client
        if not self.enabled or client is None:
            return None
        try:
            raw = client.get(self._key(resource_id, view_id, settings_sig))
            if not raw:
                return None
            if not isinstance(raw, str | bytes | bytearray):
                return None
            data = json.loads(raw)
            if isinstance(data, dict) and "embedding" in data and "meta" in data:
                return data
        except (redis_exc.RedisError, json.JSONDecodeError, TypeError) as err:
            log.warning("Dimred cache get failed: %s", err)
        return None

    def save(self, resource_id: str, view_id: str, settings_sig: str, result: dict[str, Any]) -> None:
        client = self.client
        if not self.enabled or client is None:
            return
        try:
            key = self._key(resource_id, view_id, settings_sig)
            payload = serialize_preview_result(result)
            client.setex(key, self.ttl, payload)
        except (redis_exc.RedisError, TypeError, ValueError) as err:
            log.warning("Dimred cache save failed: %s", err)

    def acquire_job_lock(self, job_id: str, ttl: int) -> bool:
        """Atomically reserve preview job creation for a short period."""
        client = self.client
        if client is None:
            return False
        try:
            return bool(client.set(self._job_lock_key(job_id), "1", nx=True, ex=ttl))
        except redis_exc.RedisError as err:
            log.warning("Dimred job lock failed: %s", err)
            raise

    def release_job_lock(self, job_id: str) -> None:
        """Release a reservation only when enqueueing the job failed."""
        client = self.client
        if client is None:
            return
        try:
            client.delete(self._job_lock_key(job_id))
        except redis_exc.RedisError as err:
            log.warning("Dimred job lock release failed: %s", err)

@lru_cache(maxsize=1)
def get_cache() -> DimredCacheManager:
    return DimredCacheManager()
