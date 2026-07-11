# Visualization Handoff Docs

This folder collects the markdown files needed to hand off visualization
integration work to another repo, infra owner, or Codex session.

## Files

- `01_autotorch_model_integration_guide.md`
  - Explains how AutoTorch advanced visualization supports new model runs, new
    PyTorch backbones, new visualization engines, and artifact-only TensorFlow
    outputs.

- `02_idrecapture_tf_export_to_mnt5_request.md`
  - The direct request to send to the IDRecapture TensorFlow infra/Codex team.
    It specifies the exact `/mnt5/temp_jj/...` output folder, required CSV
    columns, Grad-CAM PNG paths, manifest shape, and AutoTorch settings entry.

- `03_idrecapture_repo_context_ench21.md`
  - Context copied from the IDRecapture repo understanding note. Use this when
    the receiving Codex session needs repository-specific details about Ench21,
    model paths, evaluation outputs, preprocessing, and existing Grad-CAM
    behavior.

## Recommended Handoff

Send these two files first:

```text
02_idrecapture_tf_export_to_mnt5_request.md
03_idrecapture_repo_context_ench21.md
```

Keep `01_autotorch_model_integration_guide.md` as the AutoTorch-side reference.
