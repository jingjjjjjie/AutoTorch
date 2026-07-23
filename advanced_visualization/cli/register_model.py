"""Register a model-local router using only model_id and data_dir."""

from __future__ import annotations

import argparse
from pathlib import Path

from advanced_visualization.core.model_router import load_model_route
from advanced_visualization.core.settings import UserModelConfig, load_settings, save_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id")
    parser.add_argument("data_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    route = load_model_route(args.model_id, args.data_dir)
    settings = load_settings()
    registration = UserModelConfig(
        key=route.model_id,
        data_dir=str(route.data_dir),
        enabled=True,
    )
    settings.models = [
        model for model in settings.models if model.key != route.model_id
    ] + [registration]
    save_settings(settings)
    print(
        f"Registered {route.model_id}: data={route.prediction_data} "
        f"artifact={route.artifact_dir} router={route.router_path}"
    )


if __name__ == "__main__":
    main()
