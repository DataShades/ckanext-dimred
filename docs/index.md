# ckanext-dimred

`ckanext-dimred` adds a CKAN resource view for dimensionality-reduction previews
of tabular data. Editors can create PCA, t-SNE, or UMAP projections, choose
feature, colour, and tooltip fields, and render a 2D or 3D result with ECharts
or Matplotlib.

The extension is designed for bounded preview workloads rather than general
purpose batch analytics. It reads CSV, TSV, XLS, XLSX, and active DataStore
resources; records are sampled when configured limits apply, and the resulting
embedding retains stable source-row identifiers.

## Start here

- [Installation and compatibility](installation.md) covers CKAN/Python support,
  plugin enablement, and the CKAN configuration layer to edit.
- [Usage](usage.md) explains the resource-view form, feature selection, preview
  states, and export.
- [Configuration](configuration.md) lists all `ckanext.dimred.*` settings and
  their effective limits.
- [Operations and troubleshooting](operations.md) covers Redis/RQ workers,
  resource loading, routine checks, and common failures.
- [Workload budgets](benchmarks/workload-budgets.md) records the reproducible
  benchmark baseline behind the default limits.

## What it does

- **Bounded ingestion:** uploaded resources are checked against an extension
  size limit; CSV/TSV files are read incrementally and sampled deterministically.
  DataStore reads use CKAN's DataStore API.
- **Column-aware preparation:** numeric fields are eligible by default;
  low-cardinality categorical fields can be one-hot encoded when enabled.
- **Asynchronous previews:** uncached projections run in CKAN's standard RQ
  queue. The browser polls a status action instead of keeping a web request open.
- **Safe sharing:** normal CKAN resource/view authorization remains in force.
  The CSV export contains the embedding and source-row IDs, with optional
  contextual fields.

!!! note

    UMAP and t-SNE are exploratory projections. Visible distance or apparent
    clusters alone are not evidence of a statistical cluster structure,
    especially when a preview uses a sample.
