# DimRed workload-budget baseline

Measured 2026-08-11 on the local development host before changing any row-limit
defaults. The figures are a calibration baseline, not a performance guarantee
for CKAN deployments.

## Environment

- CPU: 8 logical CPUs, 11th Gen Intel Core i5-1135G7 @ 2.40 GHz.
- CKAN: 2.11.4; Python: 3.12.3.
- NumPy: 2.3.5; scikit-learn: 1.7.2; umap-learn: 0.5.9.post2.
- Input: deterministic Gaussian numeric data, seed `42`, with 10 or 50
  features. Matrix generation and `DataFrame` construction happen before the
  timer; resource loading, sampling and color/display metadata are deliberately
  outside this Python-compute benchmark.
- Preprocessing: the production `_prepare_feature_matrix` helper (feature-frame
  construction and `StandardScaler`); projection: the production
  `PCAProjection`, `UMAPProjection`, or `TSNEProjection` wrapper.
- Effective parameters: PCA `n_components=2`, `whiten=false`,
  `random_state=42`; UMAP `n_neighbors=15`, `min_dist=0.1`,
  `n_components=2`, `random_state=42`; t-SNE `perplexity=30`,
  `n_components=2`, `random_state=42`.
- Each measurement runs in its own forked CKAN process. Peak RSS is the full
  process high-water mark, including the inherited CKAN runtime; it is useful
  for comparing cases on one host, but not as an incremental matrix-only cost.
- `preview_payload_bytes` is compact UTF-8 JSON for rounded coordinates plus
  source row IDs. It intentionally excludes color and display-field vectors, so
  the production `max_preview_payload_mb` check remains the final guard.

## Full calibration matrix

| Method | Rows × features | Wall time | Peak RSS | Preview payload |
| --- | ---: | ---: | ---: | ---: |
| PCA | 1,000 × 10 | 0.10 s | 198 MiB | 18.8 KB |
| PCA | 5,000 × 10 | 0.10 s | 199 MiB | 98.0 KB |
| PCA | 10,000 × 10 | 0.10 s | 202 MiB | 197 KB |
| PCA | 50,000 × 10 | 0.13 s | 216 MiB | 1.03 MB |
| PCA | 10,000 × 50 | 0.14 s | 212 MiB | 197 KB |
| UMAP | 1,000 × 10 | 16.50 s | 446 MiB | 42.6 KB |
| UMAP | 5,000 × 10 | 33.34 s | 533 MiB | 216 KB |
| UMAP | 10,000 × 10 | 41.87 s | 551 MiB | 431 KB |
| UMAP | 10,000 × 50 | 44.25 s | 568 MiB | 431 KB |
| t-SNE | 500 × 10 | 1.42 s | 205 MiB | 21.8 KB |
| t-SNE | 1,000 × 10 | 2.80 s | 208 MiB | 43.9 KB |
| t-SNE | 2,000 × 10 | 5.74 s | 213 MiB | 88.7 KB |
| t-SNE | 2,000 × 50 | 7.12 s | 217 MiB | 88.7 KB |

The current defaults remain appropriate for this baseline:

- PCA at 50,000 rows is comfortably within the compute and 8 MiB payload
  budgets for this representative numeric input.
- UMAP at 10,000 rows is the dominant configured workload and remains bounded
  to roughly 42 seconds and 551 MiB on this host. Raising this limit would need
  fresh measurements on the intended worker class.
- t-SNE at 2,000 rows remains deliberately conservative relative to the global
  50,000-row cap.

## Re-running

From `src/ckanext-dimred` with the CKAN development environment active:

```bash
pytest -m benchmark -s -q ckanext/dimred/tests/benchmarks
DIMRED_BENCHMARK_PROFILE=pca-limit pytest -m benchmark -s -q ckanext/dimred/tests/benchmarks
DIMRED_BENCHMARK_PROFILE=umap-limit pytest -m benchmark -s -q ckanext/dimred/tests/benchmarks
DIMRED_BENCHMARK_PROFILE=tsne-limit pytest -m benchmark -s -q ckanext/dimred/tests/benchmarks
DIMRED_BENCHMARK_PROFILE=full pytest -m benchmark -s -q ckanext/dimred/tests/benchmarks
```

`smoke` is the default three-case profile. `full` adds intermediate row counts
and 50-feature cases; it is intentionally manual and must not become a
speed-sensitive CI assertion.
