# Changelog

All notable user-facing changes are generated from the Git history.

## [Unreleased]

### Features

- Add frontend i18n and refresh screenshots
- Add preflight workload context to DimRed forms
- Add contextual tooltip fields to DimRed views
- Improve feature selection for wide datasets
- Add visual context to DimRed previews
- Add method parameter controls to resource view form
- Process dimred previews with CKAN background jobs
- Preserve source row identity in dimred output
- Add color-by dropdown with backend candidates
- Add per-view n_components selection
- Allow selecting render backend per view and validate choices
- Add 3d scatter rendering support across backends
- Add pluggable render backend with echarts support
- Use friendly method labels
- Make dimred preview template extensible with override blocks
- Add configurable embedding CSV export with download button
- Add PCA projection support
- Add redis caching for dimred previews
- Add validators
- Add initial dimred resource view for tabular data

### Security

- Secure dimred tooltip and CSV output
- Secure dimred resource loading

### Fixes

- Harden dimred data preprocessing
- Enforce dimred column selection contract
- Validate dimred preview inputs
- Replace tk.validate with ckan.logic.validate for CKAN 2.10/2.11 compatibility
- Register plugin interfaces

### Performance

- Benchmark and calibrate projection workload budgets
- Bound DimRed preview payloads
- Sample CSV and TSV resources incrementally
- Lazy-load color values for dimred previews
- Bound projection workloads and lazy-load backends
- Make dimred cache identity revision-aware
- Speed up column option loading without full dataset reads

### Documentation

- Add dimensionality reduction overview to README
- Add MkDocs site and operations guide
- Refresh README images
- Refresh README

### Build

- Harden release packaging and metadata

### Refactoring

- Add configurable embedding rounding
- Enhance embedding summaries and align matplotlib visuals
