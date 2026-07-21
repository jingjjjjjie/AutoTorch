"""Image path, loading, and cache-key helpers."""
from __future__ import annotations

import base64
import hashlib
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

from advanced_visualization.core.config import IMAGE_EXTENSIONS

DEFAULT_PREVIEW_MAX_SIDE = int(os.environ.get("AUTOTORCH_IMAGE_PREVIEW_MAX_SIDE", "900"))
DEFAULT_ZOOM_MAX_SIDE = int(os.environ.get("AUTOTORCH_IMAGE_ZOOM_MAX_SIDE", "0"))
DEFAULT_JPEG_QUALITY = int(os.environ.get("AUTOTORCH_IMAGE_PREVIEW_JPEG_QUALITY", "86"))


def valid_image(path_value) -> Optional[Path]:
    if pd.isna(path_value):
        return None
    path = Path(str(path_value)).expanduser()
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    return path


def _image_signature(path_value) -> Optional[tuple[str, int, int]]:
    path = valid_image(path_value)
    if path is None:
        return None
    try:
        resolved = path.resolve()
        stat = resolved.stat()
        return str(resolved), int(stat.st_mtime_ns), int(stat.st_size)
    except OSError:
        return None


@lru_cache(maxsize=4096)
def _load_image_cached(raw_path: str, mtime_ns: int, size_bytes: int, max_side: int) -> Optional[Image.Image]:
    del mtime_ns, size_bytes
    try:
        with Image.open(raw_path) as image:
            loaded = ImageOps.exif_transpose(image).convert("RGB")
            if max_side > 0:
                loaded.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            return loaded.copy()
    except (OSError, UnidentifiedImageError):
        return None


def load_image(path_value, max_side: int = DEFAULT_PREVIEW_MAX_SIDE) -> Optional[Image.Image]:
    signature = _image_signature(path_value)
    if signature is None:
        return None
    image = _load_image_cached(*signature, int(max_side))
    return image.copy() if image is not None else None


def image_to_data_uri(image: Image.Image, quality: int = DEFAULT_JPEG_QUALITY) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=False)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@lru_cache(maxsize=4096)
def _image_path_to_data_uri_cached(
    raw_path: str,
    mtime_ns: int,
    size_bytes: int,
    max_side: int,
    quality: int,
) -> Optional[str]:
    image = _load_image_cached(raw_path, mtime_ns, size_bytes, max_side)
    if image is None:
        return None
    return image_to_data_uri(image, quality=quality)


def image_path_to_data_uri(
    path_value,
    max_side: int = DEFAULT_PREVIEW_MAX_SIDE,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> Optional[str]:
    signature = _image_signature(path_value)
    if signature is None:
        return None
    return _image_path_to_data_uri_cached(*signature, int(max_side), int(quality))


def _alternate_cache_paths(path: Path) -> list[Path]:
    paths = [path]
    raw = str(path)
    if raw.startswith("/routine_data/"):
        paths.append(Path("/mnt5") / raw.lstrip("/"))
    return paths


def _cached_image_cache_digest(raw_path: str) -> Optional[str]:
    image_path = Path(raw_path).expanduser()
    if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    resolved = image_path.expanduser().resolve()
    try:
        stat = resolved.stat()
    except OSError:
        return None
    stamp = f"{resolved}:{stat.st_mtime_ns}:{stat.st_size}"
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
