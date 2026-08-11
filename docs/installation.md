# Installation and compatibility

## Compatibility

The service-backed CI suite verifies the following combinations using official
CKAN development images.

| CKAN | Python runtime tested in CI | Status |
| --- | --- | --- |
| 2.10.x | 3.10 | Supported |
| 2.11.x | 3.10 | Supported |
| 2.9 and earlier | — | Not supported |
| 2.12+ | — | Not tested yet |

The package metadata allows Python 3.10–3.12. That expresses the extension's
Python requirement; it does not extend the service-backed CI matrix to every
CKAN/Python combination. PostgreSQL and Solr versions follow the compatible
CKAN deployment and are not separately claimed here.

## Install the extension

Activate the virtual environment used by your CKAN deployment, then install the
extension from source or from its built wheel.

```bash
git clone https://github.com/DataShades/ckanext-dimred.git
cd ckanext-dimred
pip install .
```

Enable the plugin in CKAN:

```ini
ckan.plugins = ... dimred
```

Restart the CKAN web process after changing the plugin list.

## Choose the right configuration layer

In DataShades deployments, CKAN configuration is layered as follows:

```text
default.ini → project.ini → ckan.ini (local) | production.ini (server)
```

Each layer on the right overrides the preceding one. Put shared defaults in the
appropriate shared layer and deployment-specific values in the final local or
production configuration. Do not add an extension setting to an earlier layer
when a later layer already overrides it.

See [Configuration](configuration.md) for the available settings and
[Operations](operations.md) for the worker requirement.

## Development and documentation setup

For extension development:

```bash
pip install -e '.[dev]'
```

For this documentation site:

```bash
pip install -e '.[docs]'
mkdocs serve
```

Build the site without serving it:

```bash
mkdocs build --strict
```
