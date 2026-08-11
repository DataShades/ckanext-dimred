# Operations and troubleshooting

## Deployment architecture

Enable the `dimred` plugin in the CKAN web process and run a standard CKAN RQ
worker for the `dimred` queue:

```bash
ckan -c /path/to/ckan.ini jobs worker dimred
```

The extension uses CKAN's configured Redis connection for both queueing and
eligible preview caching. It uses `ckan.jobs.timeout` for job execution; there
is no separate Dimred queue server, Redis URL, or timeout setting to configure.
Run the worker in every deployment where users can request uncached previews.

Completed upload-based previews can be read from the Redis cache. DataStore and
remote resources are deliberately not cacheable because the extension cannot
derive a reliable source revision fingerprint for them. RQ retains completed or
failed job results for one hour.

## Workload and payload budgets

| Budget | Control | Operational effect |
| --- | --- | --- |
| Resource size | `max_file_size_mb` | Rejects local and remote resources above the extension-specific processing budget. |
| Rows | global `max_rows` and per-method `*.max_rows` | Uses the lower value; CSV/TSV uses deterministic reservoir sampling. |
| Preview payload | `max_preview_payload_mb` | Rejects oversized compact JSON before cache/RQ storage and browser rendering. |
| Colour descriptors | `max_color_candidates` | Bounds choices returned for wide data; the selected colour remains available. |
| Cache lifetime | `cache_ttl` | Controls completed-preview cache expiry. |
| Method defaults | method-specific settings | Supply valid form defaults; they do not bypass input validation. |

Active DataStore resources are read through CKAN's `datastore_search` action,
so CKAN remains responsible for access control and the DataStore backend.

## Resource loading policy

Uploaded files are read through CKAN's uploader/storage path. For remote
resources, the extension only accepts `http` and `https` URLs without embedded
credentials. It resolves and connects to validated public IP addresses, uses
short connect/read timeouts, enforces the file-size budget while reading, and
revalidates every redirect target. Private, loopback, link-local, and otherwise
non-public addresses are intentionally blocked.

This policy protects the CKAN server; it is not an exception list mechanism for
internal data services. Use a CKAN upload or DataStore resource for data that
must remain on private infrastructure.

## Routine checks

After deployment or a configuration change:

1. Confirm the web process loads `dimred` and the intended final configuration
   layer contains the expected values.
2. Confirm the CKAN worker process is running with queue `dimred` and can reach
   the same configured Redis service as the web process.
3. Create a small public CSV resource and preview it; observe a pending state
   followed by a ready result.
4. Check worker logs for a validation, resource-loading, or projection error if
   the preview fails.
5. Re-run a completed upload preview to confirm cache behavior when caching is
   enabled.

Do not flush the shared Redis database to diagnose a preview. Dimred cache keys
are namespaced under `ckanext:dimred:preview:<ckan.site_id>:`; inspect the
specific namespace first and restrict any operational cleanup to it.

## Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Preview stays pending | No `dimred` RQ worker, or it cannot use Redis. | Start/check the CKAN worker and its Redis connectivity; inspect worker logs. |
| Preview fails immediately | Invalid method parameters, removed fields, too few usable rows, or no eligible features. | Review the inline field error, restore/select valid fields, and choose values compatible with the dataset. |
| File format cannot be read | Unsupported/corrupt input or a missing Excel parser dependency. | Use CSV/TSV/XLS/XLSX and confirm the installed extension environment includes `openpyxl` and `xlrd`. |
| Resource is too large | `max_file_size_mb`, `max_rows`, or payload budget is exceeded. | Reduce the source data or deliberately adjust the relevant deployment budget after measurement. |
| Remote URL is blocked | Non-HTTP(S), credentials, DNS failure, redirect to a private address, or size limit. | Use a public safe URL, upload the data, or load it into DataStore. |
| Result does not reflect new view settings | Unsaved view changes or a stale browser page. | Reload the resource view and confirm the saved settings. If the persisted view still produces an unexpected result, retain its IDs and relevant worker logs for investigation; do not flush Redis. |
| Export is unavailable | CSV export has been disabled. | Set `ckanext.dimred.export_enabled = true` and restart the web process. |

## Upgrade policy

The extension has no database migrations. CKAN validates resource-view fields
through its normal schema/NAVL flow. Cache identity includes the projection
settings and a pipeline schema version, so incompatible cache content is not
reused after an intentional pipeline change. Before a future change to view or
cache schema, document its compatibility behavior and any targeted operator
action here; do not prescribe a whole-database or whole-Redis reset.
