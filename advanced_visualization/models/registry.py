"""Registry for model-specific Grad-CAM builders."""
from __future__ import annotations

from advanced_visualization.models.base import GradcamEngine
from advanced_visualization.models.unireplknet.gradcam import UniRepLKNetGradcamEngine


_ENGINES: dict[str, GradcamEngine] = {
    UniRepLKNetGradcamEngine.name: UniRepLKNetGradcamEngine(),
}


def get_gradcam_engine(name: str) -> GradcamEngine:
    try:
        return _ENGINES[name]
    except KeyError as exc:
        known = ", ".join(sorted(_ENGINES))
        raise ValueError(f"Unknown Grad-CAM engine {name!r}. Available engines: {known}") from exc
