# AutoTorch Advanced Visualization

The recommended interface is the FastAPI browser application:

```bash
python -m advanced_visualization.web.app
```

Open <http://localhost:8000>. The application provides four focused pages:

- **Image review** — filter predictions, inspect failures, and compare originals with prepared Grad-CAM artifacts.
- **Feature space** — PCA, t-SNE, UMAP, or LDA projections with point-level image inspection.
- **Compare models** — align two experiment outputs and show what A gets right that B gets wrong, and vice versa.
- **Results analysis** — reproduce the reporting and image-inspection workflow from `src/eval/idfraud/annotation/visualize_and_analyze_results.ipynb`.

The comparison page can launch independent A and B image viewers in separate tabs. Each viewer receives its own source, columns, threshold, image, and CAM settings in the URL, so its state is isolated.

## Documentation

- [User guide](docs/USER_GUIDE.md) — configuration, launch commands, workflows, and troubleshooting.
- [Architecture](docs/ARCHITECTURE.md) — module boundaries, dependency direction, and request/data flow.
- [Extending the system](docs/EXTENDING.md) — add a source, model engine, page, or API capability without copying logic.
- [Codex/session handoff](docs/CODEX_HANDOFF.md) — invariants, verification commands, and maintenance checklist.
- [Model integration handoffs](handoff_docs/README.md) — model/export-specific background documents.

## Configure data

Settings live in `advanced_visualization/settings.json`. The legacy Streamlit settings page can edit this file:

```bash
streamlit run advanced_visualization/unified_app.py
```

The browser viewer reads enabled prediction/feature CSVs and artifact directories from the same settings. A viewer-only model can use `model_type: "artifact_only"`; checkpoint fields are required only for preparation commands that load a model.

## Prepare artifacts

Run the complete configured pipeline:

```bash
python -m advanced_visualization.cli.prepare_all
```

Or run the independent stages:

```bash
python -m advanced_visualization.cli.preparation \
  --artifact-dir /path/to/artifacts \
  --pred-csv /path/to/predictions.csv \
  --weights-epoch 11 \
  --model-key my_model

python -m advanced_visualization.cli.pregenerate_gradcam \
  --csv /path/to/artifacts/prepared_predictions.csv \
  --cam-method gradcam \
  --cam-target fraud
```

Preparation is deliberately separate from browsing. The viewer serves existing data and artifacts; it does not load model checkpoints during ordinary page requests.

## Legacy compatibility interface

The Streamlit interface remains available for existing workflows:

```bash
streamlit run advanced_visualization/unified_app.py
```

It offers image review, feature space, launchable model workspaces, and settings. New user-facing functionality should normally be implemented in the browser app first; legacy views should reuse framework-neutral `core` services.

## Container

```bash
docker compose -f advanced_visualization/docker-compose.yml up --build advanced-visualization
```

Add bind mounts for any configured image/artifact paths outside the repository.
