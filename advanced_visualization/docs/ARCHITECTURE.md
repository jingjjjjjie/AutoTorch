# Architecture

## Design goals

This package is organized around a strict dependency direction:

```text
browser/static ──HTTP──> web/routes ──> web services ──> core
Streamlit views ──────────────────────────────────────> core
CLI adapters ─────────────────────────────────────────> core
model engines ────────────────────────────────────────> model contract + core config
```

`core` must not import `web`, `views`, or `cli`. UI modules translate input and render output; reusable calculations, data preparation, path handling, and projection logic belong in `core`. CLI modules parse arguments only and call a core service. Model-specific code is behind the model registry and typed contracts.

## Package map

```text
advanced_visualization/
├── core/
│   ├── analysis.py              notebook-style ori/crop/merged evaluation
│   ├── comparison.py            stable-ID alignment and A/B outcome logic
│   ├── dataframe_filters.py     shared literal/category/range filtering
│   ├── evaluation.py            binary truth/prediction semantics
│   ├── feature_data.py          feature discovery, merge rules, sampling
│   ├── gradcam_generation.py    batch Grad-CAM generation service
│   ├── images.py                path aliases, validation, loading, cache keys
│   ├── preparation.py           standardized prepared CSV + manifest
│   ├── projection.py            PCA/t-SNE/UMAP/LDA implementation
│   ├── pipeline.py              preparation workflow orchestration
│   └── settings.py              persisted configuration contracts
├── models/
│   ├── base.py                  typed engine protocol and GradcamBundle
│   ├── registry.py              model-engine selection
│   └── <family>/gradcam.py      family-specific loading/scoring/CAM details
├── web/
│   ├── app.py                   application factory and router assembly only
│   ├── routes/                  HTTP boundary grouped by capability
│   ├── repository.py            source discovery and modification-aware cache
│   ├── filtering.py             review request orchestration and paging
│   ├── projections.py           projection cache/application service
│   ├── comparisons.py           comparison response/application service
│   ├── analysis.py              results-analysis application service
│   └── static/                  one browser module per page/component
├── views/                       legacy Streamlit adapters and renderers
├── ui/                          reusable legacy Streamlit components
├── cli/                         thin argument-parsing entrypoints
└── tests/                       core, API/service, and experiment tests
```

## Main request flows

Image review:

```text
GET /api/sources
  -> repository discovers configured artifacts
POST /api/review
  -> repository loads review columns only
  -> core evaluation + shared filters
  -> stable paging + artifact URLs
GET /api/image or /api/prepared-gradcam
  -> validate requested row/column/path
  -> cached, bounded image encoding
```

Feature projection:

```text
POST /api/projection
  -> repository loads full feature frame lazily
  -> categorical selection + deterministic sampling
  -> core.project_matrix
  -> cached coordinates and inspectable row metadata
```

Model comparison:

```text
POST /api/comparison
  -> load A and B review frames
  -> align repeated IDs by occurrence
  -> evaluate each model with its own threshold
  -> classify A-only-correct/B-only-correct/both/mismatch/missing
  -> page cards with independent image/CAM URLs
```

Results analysis:

```text
POST /api/analysis
  -> apply notebook cleanup rules
  -> evaluate original, crop, and mean-merged scores
  -> build metrics and grouped breakdowns
  -> page image pairs for the chosen outcome/drilldown
```

## State and caching

- `DatasetRepository` owns CSV caches and invalidates them with file modification time and size.
- Review data excludes high-dimensional feature columns; feature data loads only for projections.
- Projection cache keys include source version and projection/filter parameters.
- Image caching includes resolved path, modification time, and file size.
- Browser page state lives in the page module or shared `state.js`; detached viewers encode required state in their URL.

## Boundary rules

When changing code, preserve these rules:

1. Do not place dataframe algorithms in route functions, Streamlit controls, or JavaScript.
2. Do not import CLI modules from core services.
3. Do not return anonymous tuple-shaped model bundles; use `GradcamBundle`.
4. Do not duplicate projection, failure, filtering, image-path, or A/B alignment logic.
5. Route modules validate HTTP concerns and delegate; `web/app.py` only assembles the application.
6. Keep optional heavy model libraries lazy where possible so the viewer can start without loading a model.
7. Add a focused module when a page/component gains an independent responsibility; avoid generic `utils.py` dumping grounds.

## Legacy policy

The FastAPI browser application is primary. Streamlit is retained for compatibility and model-specific workspaces. Legacy views may own Streamlit state and presentation, but their calculations should call `core`. Avoid recreating browser-grade custom workspaces inside Streamlit when the primary web viewer already provides the capability.
