"""Image path, loading, and cache-key helpers."""
from __future__ import annotations

import base64
import hashlib
import io
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image

from advanced_visualization.core.config import IMAGE_EXTENSIONS


def valid_image(path_value) -> Optional[Path]:
    if pd.isna(path_value):
        return None
    path = Path(str(path_value)).expanduser()
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    return path


def load_image(path_value) -> Optional[Image.Image]:
    path = valid_image(path_value)
    if path is None:
        return None
    try:
        return Image.open(path).convert("RGB")
    except OSError:
        return None


def image_to_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _alternate_cache_paths(path: Path) -> list[Path]:
    paths = [path]
    raw = str(path)
    if raw.startswith("/routine_data/"):
        paths.append(Path("/mnt5") / raw.lstrip("/"))
    return paths


@lru_cache(maxsize=500_000)
def _cached_image_cache_digest(raw_path: str) -> Optional[str]:
    image_path = Path(raw_path).expanduser()
    if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    resolved = image_path.expanduser().resolve()
    try:
        stamp = f"{resolved}:{resolved.stat().st_mtime_ns}"
    except OSError:
        return None
    return hashlib.sha1(stamp.encode("utf-8")).hexdigest()[:18]


def image_cache_digests(path_value) -> list[str]:
    if pd.isna(path_value):
        return []
    digests: list[str] = []
    for path in _alternate_cache_paths(Path(str(path_value)).expanduser()):
        digest = _cached_image_cache_digest(str(path))
        if digest and digest not in digests:
            digests.append(digest)
    return digests


def image_cache_digest(path_value) -> Optional[str]:
    digests = image_cache_digests(path_value)
    return digests[-1] if digests else None
