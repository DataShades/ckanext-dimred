from __future__ import annotations

import json

import numpy as np
import pytest

from ckan.plugins import toolkit as tk

from ckanext.dimred import config as dimred_config
from ckanext.dimred.logic import action as dimred_action
from ckanext.dimred.utils.cache import DimredCacheManager


class FakeCache:
    def __init__(self):
        self.store = {}
        self.enabled = True

    def settings_signature(self, settings):
        return json.dumps(settings, sort_keys=True, default=str)

    def get(self, resource_id, view_id, sig):
        return self.store.get((resource_id, view_id, sig))

    def save(self, resource_id, view_id, sig, result):
        self.store[(resource_id, view_id, sig)] = result


def _upload_resource(**overrides):
    resource = {
        "id": "r1",
        "format": "csv",
        "url_type": "upload",
        "url": "rows.csv",
        "last_modified": "2026-08-07T12:00:00",
        "size": 100,
        "hash": "rows-v1",
    }
    resource.update(overrides)
    return resource


@pytest.mark.usefixtures("with_plugins")
def test_cache_settings_include_pipeline_schema_version():
    settings = dimred_action._cache_settings(_upload_resource(), {"method": "umap"})

    assert settings["pipeline_schema_version"] == 1
    assert settings["method_params"]["n_neighbors"] == 15
    assert settings["effective_max_rows"] == 10000
    assert settings["embedding_decimals"] == dimred_config.embedding_decimals()


@pytest.mark.usefixtures("with_plugins")
@pytest.mark.ckan_config("ckanext.dimred.allowed_methods", "umap tsne")
def test_pipeline_uses_cache(monkeypatch):
    fake_cache = FakeCache()
    monkeypatch.setattr("ckanext.dimred.utils.cache.get_cache", lambda: fake_cache)

    calls = {"count": 0}

    def fake_build(resource, resource_view, context):
        calls["count"] += 1
        return np.array([[1.0, 2.0]]), {"method": resource_view["method"], "prepare_info": {}}

    monkeypatch.setattr(dimred_action, "_build_dimred_preview", fake_build)

    ctx = {"ignore_auth": True}
    resource = _upload_resource()
    view = {"id": "v1", "resource_id": "r1", "method": "umap"}

    result1 = dimred_action.dimred_run_dimred_pipeline(ctx, {"resource": resource, "resource_view": view})
    result2 = dimred_action.dimred_run_dimred_pipeline(ctx, {"resource": resource, "resource_view": view})

    assert calls["count"] == 1
    assert result1 == result2
    assert len(fake_cache.store) == 1


@pytest.mark.usefixtures("with_plugins")
@pytest.mark.ckan_config("ckanext.dimred.allowed_methods", "umap tsne")
def test_cache_signature_changes_with_method(monkeypatch):
    fake_cache = FakeCache()
    monkeypatch.setattr("ckanext.dimred.utils.cache.get_cache", lambda: fake_cache)

    calls = {"count": 0}

    def fake_build(resource, resource_view, context):
        calls["count"] += 1
        val = float(calls["count"])
        return np.array([[val, val]]), {"method": resource_view["method"], "prepare_info": {}}

    monkeypatch.setattr(dimred_action, "_build_dimred_preview", fake_build)

    ctx = {"ignore_auth": True}
    resource = _upload_resource()
    view_umap = {"id": "v1", "resource_id": "r1", "method": "umap"}
    view_tsne = {"id": "v1", "resource_id": "r1", "method": "tsne"}

    res_umap = dimred_action.dimred_run_dimred_pipeline(ctx, {"resource": resource, "resource_view": view_umap})
    res_tsne = dimred_action.dimred_run_dimred_pipeline(ctx, {"resource": resource, "resource_view": view_tsne})

    assert calls["count"] == 2
    assert res_umap != res_tsne
    assert len(fake_cache.store) == 2


@pytest.mark.usefixtures("with_plugins")
@pytest.mark.ckan_config("ckanext.dimred.allowed_methods", "umap tsne")
@pytest.mark.parametrize(
    "resource",
    [
        {"id": "r1", "format": "csv", "url_type": "upload", "datastore_active": True},
        {"id": "r1", "format": "csv", "url_type": "url", "url": "https://example.test/rows.csv"},
    ],
    ids=["datastore", "remote"],
)
def test_pipeline_bypasses_cache_for_unversioned_resources(monkeypatch, resource):
    fake_cache = FakeCache()
    monkeypatch.setattr("ckanext.dimred.utils.cache.get_cache", lambda: fake_cache)

    calls = {"count": 0}

    def fake_build(resource, resource_view, context):
        calls["count"] += 1
        return np.array([[1.0, 2.0]]), {"method": resource_view["method"], "prepare_info": {}}

    monkeypatch.setattr(dimred_action, "_build_dimred_preview", fake_build)

    ctx = {"ignore_auth": True}
    view = {"id": "v1", "resource_id": "r1", "method": "umap"}

    dimred_action.dimred_run_dimred_pipeline(ctx, {"resource": resource, "resource_view": view})
    dimred_action.dimred_run_dimred_pipeline(ctx, {"resource": resource, "resource_view": view})

    assert calls["count"] == 2
    assert fake_cache.store == {}


@pytest.mark.usefixtures("with_plugins")
def test_cache_signature_changes_with_effective_method_defaults(monkeypatch):
    fake_cache = FakeCache()
    monkeypatch.setattr("ckanext.dimred.utils.cache.get_cache", lambda: fake_cache)

    calls = {"count": 0}

    def fake_build(resource, resource_view, context):
        calls["count"] += 1
        return np.array([[1.0, 2.0]]), {"method": resource_view["method"], "prepare_info": {}}

    monkeypatch.setattr(dimred_action, "_build_dimred_preview", fake_build)

    context = {"ignore_auth": True}
    resource = _upload_resource()
    view = {"id": "v1", "resource_id": "r1", "method": "umap"}
    dimred_action.dimred_run_dimred_pipeline(context, {"resource": resource, "resource_view": view})
    monkeypatch.setattr("ckanext.dimred.logic.action.dimred_config.umap_n_neighbors", lambda: 20)
    dimred_action.dimred_run_dimred_pipeline(context, {"resource": resource, "resource_view": view})

    assert calls["count"] == 2
    assert len(fake_cache.store) == 2


@pytest.mark.usefixtures("with_plugins")
def test_cache_signature_changes_with_embedding_decimals(monkeypatch):
    fake_cache = FakeCache()
    monkeypatch.setattr("ckanext.dimred.utils.cache.get_cache", lambda: fake_cache)

    calls = {"count": 0}

    def fake_build(resource, resource_view, context):
        calls["count"] += 1
        return np.array([[1.0, 2.0]]), {"method": resource_view["method"], "prepare_info": {}}

    monkeypatch.setattr(dimred_action, "_build_dimred_preview", fake_build)

    context = {"ignore_auth": True}
    resource = _upload_resource()
    view = {"id": "v1", "resource_id": "r1", "method": "umap"}
    dimred_action.dimred_run_dimred_pipeline(context, {"resource": resource, "resource_view": view})
    monkeypatch.setattr("ckanext.dimred.logic.action.dimred_config.embedding_decimals", lambda: 4)
    dimred_action.dimred_run_dimred_pipeline(context, {"resource": resource, "resource_view": view})

    assert calls["count"] == 2
    assert len(fake_cache.store) == 2


@pytest.mark.usefixtures("with_plugins")
def test_cache_signature_changes_with_effective_row_limit(monkeypatch):
    fake_cache = FakeCache()
    monkeypatch.setattr("ckanext.dimred.utils.cache.get_cache", lambda: fake_cache)

    def fake_build(resource, resource_view, context):
        return np.array([[1.0, 2.0]]), {"method": resource_view["method"], "prepare_info": {}}

    monkeypatch.setattr(dimred_action, "_build_dimred_preview", fake_build)

    context = {"ignore_auth": True}
    resource = _upload_resource()
    view = {"id": "v1", "resource_id": "r1", "method": "tsne"}
    dimred_action.dimred_run_dimred_pipeline(context, {"resource": resource, "resource_view": view})
    monkeypatch.setitem(tk.config, dimred_config.TSNE_MAX_ROWS, 1000)
    dimred_action.dimred_run_dimred_pipeline(context, {"resource": resource, "resource_view": view})

    assert len(fake_cache.store) == 2


@pytest.mark.usefixtures("with_plugins")
def test_cache_signature_changes_with_resource_fingerprint(monkeypatch):
    fake_cache = FakeCache()
    monkeypatch.setattr("ckanext.dimred.utils.cache.get_cache", lambda: fake_cache)

    calls = {"count": 0}

    def fake_build(resource, resource_view, context):
        calls["count"] += 1
        return np.array([[1.0, 2.0]]), {"method": resource_view["method"], "prepare_info": {}}

    monkeypatch.setattr(dimred_action, "_build_dimred_preview", fake_build)

    context = {"ignore_auth": True}
    resource = _upload_resource()
    view = {"id": "v1", "resource_id": "r1", "method": "umap"}
    dimred_action.dimred_run_dimred_pipeline(context, {"resource": resource, "resource_view": view})
    resource["last_modified"] = "2026-08-07T12:01:00"
    dimred_action.dimred_run_dimred_pipeline(context, {"resource": resource, "resource_view": view})

    assert calls["count"] == 2
    assert len(fake_cache.store) == 2


@pytest.mark.usefixtures("with_plugins")
def test_cache_key_is_scoped_to_site_id(monkeypatch):
    manager = object.__new__(DimredCacheManager)

    monkeypatch.setitem(tk.config, "ckan.site_id", "site-one")
    first = manager._key("r1", "v1", "settings")
    monkeypatch.setitem(tk.config, "ckan.site_id", "site-two")
    second = manager._key("r1", "v1", "settings")

    assert first != second
    assert ":site-one:" in first
    assert ":site-two:" in second
