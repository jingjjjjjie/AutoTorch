"""JSON-safe scalar conversion shared by API response builders."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def json_value(value: Any) -> str | int | float | bool | None:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, int):
        return value
    return str(value)

