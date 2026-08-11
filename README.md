[![Tests](https://github.com/DataShades/ckanext-dimred/actions/workflows/test.yml/badge.svg)](https://github.com/DataShades/ckanext-dimred/actions/workflows/test.yml)

# ckanext-dimred

CKAN resource view for bounded dimensionality-reduction previews of tabular
data. It supports PCA, t-SNE, and UMAP; interactive ECharts and static
Matplotlib rendering; configurable feature, colour, and tooltip fields; CSV
export; Redis caching; and CKAN RQ background jobs.

![Tabular CKAN resource transformed into an interactive dimensionality-reduction preview](https://raw.githubusercontent.com/DataShades/ckanext-dimred/master/docs/assets/images/dimred-overview.png)

Read the documentation in this repository:

- [Installation and compatibility](https://github.com/DataShades/ckanext-dimred/blob/master/docs/installation.md)
- [Usage](https://github.com/DataShades/ckanext-dimred/blob/master/docs/usage.md)
- [Configuration](https://github.com/DataShades/ckanext-dimred/blob/master/docs/configuration.md)
- [Operations and troubleshooting](https://github.com/DataShades/ckanext-dimred/blob/master/docs/operations.md)
- [Workload budgets](https://github.com/DataShades/ckanext-dimred/blob/master/docs/benchmarks/workload-budgets.md)
- [Changelog](https://github.com/DataShades/ckanext-dimred/blob/master/CHANGELOG.md)

## Install

Activate the CKAN virtual environment and install the extension:

```bash
git clone https://github.com/DataShades/ckanext-dimred.git
cd ckanext-dimred
pip install .
```

Enable the plugin in the final applicable CKAN configuration layer:

```ini
ckan.plugins = ... dimred
```

Run a CKAN worker for uncached previews:

```bash
ckan -c /path/to/ckan.ini jobs worker dimred
```

## Development

```bash
pip install -e '.[dev]'
pytest --ckan-ini=test_config/test.ini
```

Run the local static checks after installing JavaScript dependencies:

```bash
npm ci
ruff check ckanext/dimred
python scripts/typecheck.py
npm run lint:js
```

Browser tests additionally require Chromium and three terminals:

```bash
playwright install --with-deps chromium
```

In the first terminal, start CKAN:

```bash
ckan -c test_config/test.ini run -t
```

In the second terminal, start the DimRed worker:

```bash
ckan -c test_config/test.ini jobs worker dimred
```

In the third terminal, run Playwright:

```bash
pytest --ckan-ini=test_config/test.ini -m playwright --browser chromium \
  --base-url=http://127.0.0.1:5000 ckanext/dimred/tests/e2e
```

Build the documentation locally:

```bash
pip install -e '.[docs]'
mkdocs build --strict
```

Regenerate the changelog after installing the development dependencies:

```bash
make changelog
```

## License

[AGPL-3.0-or-later](LICENSE)
