"""Cached Grad-CAM path helpers used by the lightweight viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from advanced_visualization.core.config import (
    DEFAULT_GRADCAM_ROOT,
    gradcam_artifact_root,
)
from advanced_visualization.core.images import image_cache_digests, valid_image


def gradcam_cache_candidates(
    root: Path,
    image_path: Path,
    method: str = "",
    space: str = "original",
    target: str = "fraud",
    layer: str = "",
) -> list[Path]:
    digests = image_cache_digests(image_path)
    if not digests:
        return []
    space_marker = "_model_input" if space == "model-input" else ""
    target_marker = "_genuine" if target == "genuine" else ""
    if method == "gradcam":
        return [
            candidate
            for digest in digests
            for candidate in (
                root / f"{digest}_gradcam{target_marker}_logit{space_marker}.png",
                root / f"{digest}_gradcam{target_marker}{space_marker}.png",
            )
        ]
    if method in {"gradcam++", "gradcampp"}:
        return [
            candidate
            for digest in digests
            for candidate in (
                *(
                    (root / layer / target / f"{digest}_gradcampp_logit.webp",)
                    if layer
                    else ()
                ),
                root / f"{digest}_gradcampp{target_marker}_logit{space_marker}.png",
                root / f"{digest}_gradcampp{target_marker}{space_marker}.png",
            )
        ]
    if target == "genuine":
        return [
            candidate
            for digest in digests
            for candidate in (
                *(
                    (root / layer / target / f"{digest}_gradcampp_logit.webp",)
                    if layer
                    else ()
                ),
                root / f"{digest}_gradcam_genuine_logit.png",
                root / f"{digest}_gradcam_genuine_logit_model_input.png",
                root / f"{digest}_gradcampp_genuine_logit.png",
                root / f"{digest}_gradcampp_genuine_logit_model_input.png",
                root / f"{digest}_gradcam_genuine.png",
                root / f"{digest}_gradcam_genuine_model_input.png",
                root / f"{digest}_gradcampp_genuine.png",
                root / f"{digest}_gradcampp_genuine_model_input.png",
            )
        ]
    return [
        candidate
        for digest in digests
        for candidate in (
            *(
                (root / layer / target / f"{digest}_gradcampp_logit.webp",)
                if layer
                else ()
            ),
            root / f"{digest}_gradcam_logit.png",
            root / f"{digest}_gradcam_logit_model_input.png",
            root / f"{digest}_gradcampp_logit.png",
            root / f"{digest}_gradcampp_logit_model_input.png",
            root / f"{digest}_gradcam.png",
            root / f"{digest}_gradcam_model_input.png",
            root / f"{digest}_gradcampp.png",
            root / f"{digest}_gradcampp_model_input.png",
        )
    ]


GRADCAM_PRIORITY = (
    "_gradcam_logit.png",
    "_gradcam_logit_model_input.png",
    "_gradcampp_logit.png",
    "_gradcampp_logit_model_input.png",
    "_gradcam.png",
    "_gradcam_model_input.png",
    "_gradcampp.png",
    "_gradcampp_model_input.png",
)

GRADCAM_METHOD_PRIORITY = {
    "gradcam": (
        "_gradcam_logit.png",
        "_gradcam_logit_model_input.png",
        "_gradcam.png",
        "_gradcam_model_input.png",
    ),
    "gradcam++": (
        "_gradcampp_logit.png",
        "_gradcampp_logit_model_input.png",
        "_gradcampp.png",
        "_gradcampp_model_input.png",
    ),
    "gradcampp": (
        "_gradcampp_logit.png",
        "_gradcampp_logit_model_input.png",
        "_gradcampp.png",
        "_gradcampp_model_input.png",
    ),
}

GENUINE_GRADCAM_METHOD_PRIORITY = {
    "gradcam": (
        "_gradcam_genuine_logit.png",
        "_gradcam_genuine_logit_model_input.png",
        "_gradcam_genuine.png",
        "_gradcam_genuine_model_input.png",
    ),
    "gradcam++": (
        "_gradcampp_genuine_logit.png",
        "_gradcampp_genuine_logit_model_input.png",
        "_gradcampp_genuine.png",
        "_gradcampp_genuine_model_input.png",
    ),
    "gradcampp": (
        "_gradcampp_genuine_logit.png",
        "_gradcampp_genuine_logit_model_input.png",
        "_gradcampp_genuine.png",
        "_gradcampp_genuine_model_input.png",
    ),
}


def priority_index(name: str, priority: tuple[str, ...]) -> int:
    for index, marker in enumerate(priority):
        if marker in name:
            return index
    return len(priority)


def gradcam_file_index(
    root: str, method: str = "", target: str = "fraud"
) -> dict[str, str]:
    root_path = Path(root).expanduser()
    if not root_path.exists():
        return {}
    if target == "genuine":
        priority = GENUINE_GRADCAM_METHOD_PRIORITY.get(
            method,
            tuple(
                marker
                for markers in GENUINE_GRADCAM_METHOD_PRIORITY.values()
                for marker in markers
            ),
        )
    else:
        priority = GRADCAM_METHOD_PRIORITY.get(method, GRADCAM_PRIORITY)
    index: dict[str, str] = {}
    for path in root_path.glob("*.png"):
        is_genuine = "_genuine" in path.name.lower()
        if is_genuine != (target == "genuine"):
            continue
        digest = path.name.split("_", 1)[0]
        if len(digest) != 18:
            continue
        if priority_index(path.name, priority) >= len(priority):
            continue
        current = index.get(digest)
        if current is None:
            index[digest] = str(path)
            continue
        current_name = Path(current).name
        if priority_index(path.name, priority) < priority_index(current_name, priority):
            index[digest] = str(path)
    return index


def gradcam_roots(active_stem: Optional[str], gradcam_dir: str = "") -> list[Path]:
    roots: list[Path] = []
    if gradcam_dir:
        roots.append(Path(gradcam_dir).expanduser())
    if active_stem:
        artifact_root = gradcam_artifact_root(active_stem)
        if artifact_root is not None:
            roots.append(artifact_root)
        roots.append(DEFAULT_GRADCAM_ROOT / active_stem)

    unique_roots = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            unique_roots.append(root)
            seen.add(key)
    return unique_roots


def resolve_gradcam_path(row, controls: dict):
    gradcam_column = controls["gradcam_column"]
    gradcam_dir = controls["gradcam_dir"]
    image_column = controls["image_column"]
    if gradcam_column and gradcam_column in row.index:
        path = valid_image(row[gradcam_column])
        if path is not None:
            return path

    if not image_column or image_column not in row.index:
        return None

    image_path = valid_image(row[image_column])
    if image_path is None:
        return None

    active_stem = controls.get("active_csv_stem")
    roots = gradcam_roots(active_stem, gradcam_dir)
    method = controls.get("cam_method", "")
    space = controls.get("cam_space", "original")

    for root in roots:
        for candidate in gradcam_cache_candidates(
            root,
            image_path,
            method=method,
            space=space,
            target=controls.get("cam_target", "fraud"),
        ):
            if candidate.is_file():
                return candidate

    root = roots[0] if roots else DEFAULT_GRADCAM_ROOT / str(active_stem or "")
    candidates = [
        root / f"{image_path.stem}.png",
        root / f"{image_path.stem}.jpg",
        root / f"{image_path.stem}_gradcam.png",
        root / f"{image_path.stem}_overlay.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
