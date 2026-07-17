"""Validated image lookup and thumbnail generation."""
from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from advanced_visualization.core.images import valid_image


def _render_image(path: str, max_side: int) -> bytes:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if max_side > 0:
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=86, optimize=False)
            return output.getvalue()
    except (OSError, UnidentifiedImageError) as exc:
        raise FileNotFoundError("Image could not be decoded.") from exc


@lru_cache(maxsize=1024)
def _thumbnail(path: str, modified_ns: int, size: int, max_side: int) -> bytes:
    del modified_ns, size
    return _render_image(path, max_side)


def image_bytes(path_value, max_side: int = 900) -> tuple[bytes, str]:
    path = valid_image(path_value)
    if path is None:
        raise FileNotFoundError("Image does not exist or has an unsupported format.")
    resolved = Path(path).resolve()
    stat = resolved.stat()
    if max_side == 0:
        return _render_image(str(resolved), max_side), str(stat.st_mtime_ns)
    return _thumbnail(str(resolved), stat.st_mtime_ns, stat.st_size, max_side), str(stat.st_mtime_ns)
