# AutoTorch Advanced Visualization

The recommended interface is the FastAPI browser application:

```bash
python -m advanced_visualization.web.app
```

Open <http://localhost:8000>. The application provides five focused pages:

- **Image review** — filter predictions, inspect failures, and compare originals with prepared Grad-CAM artifacts.
- **Feature space** — PCA, t-SNE, UMAP, or LDA projections with point-level image inspection.
- **Compare models** — align two experiment outputs and show what A gets right that B gets wrong, and vice versa.
- **Results analysis** — reproduce the reporting and image-inspection workflow from `src/eval/idfraud/annotation/visualize_and_analyze_results.ipynb`.
- **Live inference** — upload one image, run a configured checkpoint, and inspect the prediction alongside genuine and fraud Grad-CAM explanations.

Live inference lists enabled, checkpoint-backed models from `settings.json`. The first request for a model loads and caches its checkpoint; later requests reuse it. Uploads are processed in memory and are not added to prediction CSVs or artifact directories.

TensorFlow Ench21 VAN Small live inference runs in a separate TensorFlow 2.8
container because the main visualization service uses Python 3.12. The Compose
stack exposes two additional live models, `ench21_vansmall_ori` and
`ench21_vansmall_crop`. Each returns pre-sigmoid-logit Grad-CAM evidence for
both Genuine and Fraud at `norm3`, `block3.3`, `block4.1`, and `norm4`.

The service builds on the existing local `idrecapture_automl:v1.0.0` image so
its TensorFlow 2.8 and CUDA environment stays aligned with IDRecapture.

```bash
IDRECAPTURE_REPO_PATH=/home/jingjie/Dev/automl_platform/idrecapture_server \
VANSMALL_CUDA_VISIBLE_DEVICES=0 \
docker compose -f advanced_visualization/docker-compose.yml up --build
```

GPU indices use PCI-bus order. On the current visualization host, the default
device `0` is an NVIDIA GeForce RTX 3090.

Open `http://localhost:8001`, choose **Live inference**, select the original or
crop VAN Small model, and upload the corresponding image view. The first call
loads the selected 156 MB checkpoint; later calls reuse the cached Keras model.
The TensorFlow service health endpoint is available at `http://localhost:8701/health`.

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
The web image installs CPU-only PyTorch by default, matching the compose service's
portable configuration. Override the `TORCH_INDEX_URL` build argument and add a
Compose GPU reservation when deploying on an NVIDIA Container Toolkit host.
