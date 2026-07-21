# Codex / Maintainer Handoff

## Current product shape

The FastAPI browser viewer is the primary product. It has image review, feature projection, bidirectional A/B comparison, independent model viewers, and notebook-compatible results analysis. Streamlit remains a compatibility interface and configuration editor.

The codebase was refactored so that reusable behavior lives in `core`, HTTP endpoints are grouped under `web/routes`, `web/app.py` is application assembly, CLI files are parsers/adapters, and model loaders return the typed `GradcamBundle` contract.

## Invariants to protect

- `core` never imports `web`, `views`, or `cli`.
- Model/checkpoint libraries are not loaded merely to serve a normal viewer request.
- Binary evaluation semantics come from `core/evaluation.py`.
- A/B alignment and outcomes come from `core/comparison.py`.
- Projection algorithms come from `core/projection.py`.
- Common category/text/range filters come from `core/dataframe_filters.py`.
- Image mount aliases and cache identity come from `core/images.py`.
- CLI `argparse.Namespace` objects never cross into core services.
- Truth mismatch, missing-in-source, and unscored comparison rows are not counted as ordinary model failures.

## Verification commands

Run from the repository root with the active Python environment:

```bash
python -m compileall -q advanced_visualization
python -m pytest -q advanced_visualization/tests
node --check advanced_visualization/web/static/app.js
for file in advanced_visualization/web/static/*.js; do node --check "$file"; done
```

Start the server and smoke-test:

```bash
python -m advanced_visualization.web.app
curl -fsS http://localhost:8000/api/health
```

Also load each of the four pages in a browser, run one review filter, one projection, one comparison, and one results-analysis request. Open A and B detached viewers and confirm their URL/state independence.

## Where to make common changes

- New evaluation rule: `core/evaluation.py`, then its consumers/tests.
- New comparison behavior: `core/comparison.py` and `web/comparisons.py`.
- New analysis metric: `core/analysis.py` and `web/analysis.py`.
- New projection: `core/projection.py`, the projection request contract, and browser selector.
- New API endpoint: focused route module; do not grow `web/app.py`.
- New browser behavior: page/component JavaScript module; do not grow a single global controller.
- New model family: its own engine plus registry entry.
- New persisted setting: typed settings dataclass, JSON roundtrip, UI/defaults, and docs.

## Maintenance checklist

1. Inspect `git status` and preserve unrelated user changes.
2. Read `README.md`, `docs/ARCHITECTURE.md`, and the closest tests before editing.
3. Put behavior at the lowest reusable layer and keep adapters thin.
4. Add a regression test for every bug fixed during a refactor.
5. Run the focused test first, then the full suite and browser smoke test.
6. Update the user guide or extension guide if behavior/configuration changes.

## Intentionally retained compatibility code

- `views/` and `ui/` support the legacy Streamlit application.
- `ui/zoom.py` is an isolated embedded component because Streamlit does not provide the browser viewer's full pan/zoom behavior.
- CLI files retain a small repository-path bootstrap so they still work when invoked by file path; documentation prefers `python -m ...`.
- Model-specific engines may add `src` to the module path because parts of the training tree use top-level imports.

These exceptions should stay isolated. Do not use them as patterns for new core or web code.
