from __future__ import annotations

import numpy as np
import pytest

from ckanext.dimred.logic import action as dimred_action
from ckanext.dimred.plugin import DimredPlugin


class FakeCache:
    def __init__(self):
        self.store = {}
        self.deleted: list[str] = []
        self.enabled = True

    def settings_signature(self, settings):
        return settings.get("method", "sig")

    def get(self, resource_id, view_id, sig):
        return self.store.get((resource_id, view_id, sig))

    def save(self, resource_id, view_id, sig, result):
        self.store[(resource_id, view_id, sig)] = result

    def delete_for_resource(self, resource_id):
        self.deleted.append(resource_id)


@pytest.mark.usefixtures("with_plugins")
@pytest.mark.ckan_config("ckanext.dimred.allowed_methods", "umap tsne")
def test_pipeline_uses_cache(monkeypatch):
    fake_cache = FakeCache()
    monkeypatch.setattr("ckanext.dimred.utils.cache.get_cache", lambda: fake_cache)

    calls = {"count": 0}

    def fake_build(resource, resource_view):
        calls["count"] += 1
        return np.array([[1.0, 2.0]]), {"method": resource_view["method"], "prepare_info": {}}

    monkeypatch.setattr(dimred_action, "_build_dimred_preview", fake_build)

    ctx = {"ignore_auth": True}
    resource = {"id": "r1", "format": "csv"}
    view = {"id": "v1", "method": "umap"}

    result1 = dimred_action.dimred_run_dimred_pipeline(ctx, {"resource": resource, "resource_view": view})
    result2 = dimred_action.dimred_run_dimred_pipeline(ctx, {"resource": resource, "resource_view": view})

    assert calls["count"] == 1
    assert result1 == result2
    assert fake_cache.get("r1", "v1", "umap") == result1


@pytest.mark.usefixtures("with_plugins")
@pytest.mark.ckan_config("ckanext.dimred.allowed_methods", "umap tsne")
def test_cache_signature_changes_with_method(monkeypatch):
    fake_cache = FakeCache()
    monkeypatch.setattr("ckanext.dimred.utils.cache.get_cache", lambda: fake_cache)

    calls = {"count": 0}

    def fake_build(resource, resource_view):
        calls["count"] += 1
        val = float(calls["count"])
        return np.array([[val, val]]), {"method": resource_view["method"], "prepare_info": {}}

    monkeypatch.setattr(dimred_action, "_build_dimred_preview", fake_build)

    ctx = {"ignore_auth": True}
    resource = {"id": "r1", "format": "csv"}
    view_umap = {"id": "v1", "method": "umap"}
    view_tsne = {"id": "v1", "method": "tsne"}

    res_umap = dimred_action.dimred_run_dimred_pipeline(ctx, {"resource": resource, "resource_view": view_umap})
    res_tsne = dimred_action.dimred_run_dimred_pipeline(ctx, {"resource": resource, "resource_view": view_tsne})

    assert calls["count"] == 2
    assert res_umap != res_tsne
    assert len(fake_cache.store) == 2


@pytest.mark.usefixtures("with_plugins")
def test_resource_cache_invalidation_on_update(monkeypatch):
    fake_cache = FakeCache()
    monkeypatch.setattr("ckanext.dimred.utils.cache.get_cache", lambda: fake_cache)

    plugin = DimredPlugin()

    plugin.before_resource_update({}, {"id": "r1", "url": "http://old"}, {"id": "r1", "url": "http://new"})
    assert fake_cache.deleted == ["r1"]

    fake_cache.deleted.clear()
    plugin.before_resource_update({}, {"id": "r1", "url": "http://same"}, {"id": "r1", "url": "http://same"})
    assert fake_cache.deleted == []


@pytest.mark.usefixtures("with_plugins")
def test_resource_cache_invalidation_on_delete(monkeypatch):
    fake_cache = FakeCache()
    monkeypatch.setattr("ckanext.dimred.utils.cache.get_cache", lambda: fake_cache)

    plugin = DimredPlugin()
    plugin.before_resource_delete({}, {"id": "r1"})

    assert fake_cache.deleted == ["r1"]
