"""HTTP endpoints for live upload inference."""

from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Query

from advanced_visualization.core.config import all_model_runs
from advanced_visualization.core.settings import load_settings
from advanced_visualization.web.live_inference import (
    RemoteInferenceError,
    live_inference_service,
    tensorflow_live_inference_client,
)


router = APIRouter(prefix="/api/live-inference", tags=["live-inference"])


@router.get("/models")
def live_models() -> list[dict[str, object]]:
    settings = load_settings()
    configured = {model.key: model for model in settings.models}
    models = []
    for key, config in all_model_runs().items():
        user_config = configured.get(key)
        preset = user_config.review_preset if user_config else {}
        models.append(
            {
                "key": key,
                "label": key.replace("_", " "),
                "model_name": config.model_name,
                "image_size": config.image_size,
                "available": config.checkpoint.is_file(),
                "threshold": float(preset.get("threshold", 0.5)),
            }
        )
    try:
        remote_models = tensorflow_live_inference_client.models()
    except RemoteInferenceError as exc:
        remote_models = []
        models.append(
            {
                "key": "tensorflow_vansmall_unavailable",
                "label": "TensorFlow VAN Small",
                "model_name": "vansmall",
                "framework": "tensorflow",
                "image_size": 512,
                "available": False,
                "threshold": 0.5,
                "status": str(exc),
            }
        )
    known = {str(model["key"]) for model in models}
    models.extend(model for model in remote_models if str(model.get("key")) not in known)
    return models


@router.post("/predict")
def live_predict(
    model_key: str = Query(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    method: str = Query("gradcam++", pattern="^gradcam\\+\\+$"),
    payload: bytes = Body(..., media_type="application/octet-stream"),
) -> dict[str, object]:
    config = all_model_runs().get(model_key)
    if config is None:
        try:
            return tensorflow_live_inference_client.predict(
                model_key,
                payload,
                threshold=threshold,
                method=method,
            )
        except RemoteInferenceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    if not config.checkpoint.is_file():
        raise HTTPException(status_code=409, detail="The selected model checkpoint is unavailable.")
    try:
        result = live_inference_service.infer(
            config, payload, threshold=threshold, method=method
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result.to_dict()
