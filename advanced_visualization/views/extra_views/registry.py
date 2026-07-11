"""Registry for launchable model-specific workspaces."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from advanced_visualization.core.settings import DEFAULT_EXTRA_VIEW_CONFIGS, load_settings
from advanced_visualization.views.extra_views import workspace


RenderFn = Callable[[pd.DataFrame, dict, dict], None]


@dataclass(frozen=True)
class ExtraView:
    key: str
    label: str
    model_type: str
    description: str
    config: dict
    required_columns: tuple[str, ...]
    render: RenderFn

    def missing_columns(self, df: pd.DataFrame) -> list[str]:
        return [column for column in self.required_columns if column not in df.columns]


def _configs() -> list[dict]:
    settings = load_settings()
    return settings.extra_view_configs or DEFAULT_EXTRA_VIEW_CONFIGS


def _view_key(config: dict) -> str:
    model_type = str(config.get("model_type", "model")).strip() or "model"
    view = str(config.get("view", "layered_gradcam")).strip() or "layered_gradcam"
    return f"{model_type}:{view}"


def _render_fn(config: dict) -> RenderFn:
    return workspace.render


def _build_view(config: dict) -> ExtraView:
    return ExtraView(
        key=_view_key(config),
        label=str(config.get("label") or config.get("model_type") or "Extra view"),
        model_type=str(config.get("model_type", "")),
        description=str(config.get("description", "")),
        config=config,
        required_columns=tuple(str(column) for column in config.get("required_columns", [])),
        render=_render_fn(config),
    )


def extra_view_options() -> list[ExtraView]:
    return [_build_view(config) for config in _configs()]


def get_extra_view(key: str) -> ExtraView:
    for view in extra_view_options():
        if view.key == key:
            return view
    raise KeyError(key)
