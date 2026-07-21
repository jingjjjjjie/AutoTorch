# Extending Advanced Visualization

## Start with the reusable contract

Before adding a page-specific implementation, decide which layer owns the behavior:

- dataframe/math/path behavior: `core`;
- source caching and response assembly: a named `web` service;
- HTTP validation and status mapping: `web/routes`;
- browser interaction/rendering: one module under `web/static`;
- Streamlit-only compatibility state: `views` or `ui`;
- CLI argument parsing: `cli`;
- checkpoint/backbone specifics: `models/<family>`.

This keeps one implementation of each business rule.

## Add a new configured experiment

Usually no code is required. Add an enabled model entry to `settings.json` with a unique key and valid prediction/feature/artifact paths. Set preferred `image_column`, `prediction_column`, and an optional `review_preset`.

Export stable sample IDs if the source will be compared with another model. Export numeric, consistently named feature columns if it will use Feature Space.

## Add a new PyTorch model family

1. Create `models/<family>/gradcam.py` implementing the protocol in `models/base.py`.
2. Return a `GradcamBundle(model, transform, target_layer, device)`.
3. Keep checkpoint construction, preprocessing, target selection, scoring, and family-specific CAM behavior in that module.
4. Register the engine in `models/registry.py`.
5. Add a settings model entry whose `model_type` matches the registry key.
6. Test bundle caching, score shape, checkpoint errors, and at least one small CAM calculation when feasible.

Viewer-only exports from another framework should normally use `model_type: "artifact_only"` and provide prepared images/CAM paths. The browser does not require a model engine to display exported artifacts.

## Add a new browser page

1. Put reusable calculation in a focused `core/<capability>.py` service with unit tests.
2. Define request/response contracts in `web/models.py` when a typed body is useful.
3. Add response assembly/cache behavior to `web/<capability>.py`.
4. Add a small router under `web/routes/<capability>.py` and include it in `web/app.py`.
5. Create `web/static/<capability>.js`; keep orchestration in `app.js` and page behavior in the page module.
6. Add only the necessary markup to `index.html` and styles to a clearly named CSS section.
7. Add service tests plus an HTTP-boundary smoke test.

Do not import `web` from `core`, perform CSV work in the browser, or copy failure/filter/projection logic into the new module.

## Add a new comparison outcome

Outcome classification belongs in `core/comparison.py`. Add the outcome to its contract and tests first, then expose it through `web/comparisons.py`, and finally add the browser label/control. Truth mismatches and unscored rows must remain separate from model correctness.

## Add notebook-style analysis

Translate notebook calculations into named, pure functions under `core`; keep row-cleanup rules explicit and test them with small dataframes. A notebook may then call the same functions, while the web service handles source lookup, paging, and artifact URLs. Do not make a notebook the only implementation of evaluation semantics.

## Add a legacy launchable workspace

Prefer configuration under `extra_view_configs` when the workspace is another combination of image review, feature projection, and layered artifacts. Add new Streamlit code only for behavior that cannot be expressed by configuration. Reuse `core` services and existing `ui` components.

## Definition of done

- the dependency direction in `ARCHITECTURE.md` is preserved;
- focused unit tests cover new reusable behavior;
- route/browser smoke tests cover integration;
- no large data, checkpoint, or image artifact is committed accidentally;
- docs and settings examples describe any new configuration;
- `python -m pytest -q advanced_visualization/tests` and JavaScript syntax checks pass.
