# Configuration

Add settings to the final applicable CKAN configuration layer described in
[Installation](installation.md#choose-the-right-configuration-layer). Defaults
come from `ckanext/dimred/config_declaration.yaml`.

## General settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `ckanext.dimred.default_method` | `umap` | Method selected for a new view. |
| `ckanext.dimred.allowed_methods` | `umap tsne pca` | Enabled projection methods. |
| `ckanext.dimred.max_file_size_mb` | `50` | Per-resource size budget for processing. |
| `ckanext.dimred.max_rows` | `50000` | Global maximum rows processed by any method. |
| `ckanext.dimred.max_preview_payload_mb` | `8` | Maximum serialized result held by RQ/cache and sent to the browser. |
| `ckanext.dimred.max_color_candidates` | `100` | Maximum colour-column descriptors in a result; the selected colour stays included. |
| `ckanext.dimred.enable_categorical` | `true` | Allow low-cardinality categorical feature fields. |
| `ckanext.dimred.max_categories_for_ohe` | `30` | Maximum distinct values for categorical one-hot encoding. |
| `ckanext.dimred.export_enabled` | `true` | Enable CSV embedding export. |
| `ckanext.dimred.render_backend` | `echarts` | Default backend: `echarts` or `matplotlib`. |
| `ckanext.dimred.render_asset` | empty | Optional webassets bundle override for the backend. |
| `ckanext.dimred.render_module` | empty | Optional CKAN JavaScript module override for the backend. |
| `ckanext.dimred.embedding_decimals` | `3` | Coordinate decimal places in result and export. |

## Cache and background work

| Setting | Default | Purpose |
| --- | --- | --- |
| `ckanext.dimred.cache_enabled` | `true` | Cache completed eligible previews in CKAN Redis. |
| `ckanext.dimred.cache_ttl` | `3600` | Cache lifetime in seconds. |

The extension uses CKAN's Redis connection. It does not define a separate Redis
URL. Uncached work runs on CKAN's `dimred` RQ queue and uses the standard
`ckan.jobs.timeout` setting.

## Method defaults and effective row limits

| Method | Settings | Default row budget |
| --- | --- | --- |
| UMAP | `n_neighbors=15`, `min_dist=0.1`, `n_components=2` | `ckanext.dimred.umap.max_rows = 10000` |
| t-SNE | `perplexity=30`, `n_components=2` | `ckanext.dimred.tsne.max_rows = 2000` |
| PCA | `n_components=2`, `whiten=false` | `ckanext.dimred.pca.max_rows = 50000` |

The effective row limit is always the lower of `ckanext.dimred.max_rows` and
the selected method's `*.max_rows`. Method values are defaults: a view can
choose valid method parameters in its form. Changing a default does not turn an
invalid view value into a valid one.

For example, restrict UMAP to 10,000 rows:

```ini
ckanext.dimred.allowed_methods = umap
ckanext.dimred.max_rows = 10000
ckanext.dimred.umap.max_rows = 10000
ckanext.dimred.enable_categorical = true
```

The limits are compute budgets, not a latency guarantee. Re-measure after
changing CKAN, Python, projection libraries, CPU allocation, or representative
feature counts. See [Workload budgets](benchmarks/workload-budgets.md).
