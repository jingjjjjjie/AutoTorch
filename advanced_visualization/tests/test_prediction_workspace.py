from __future__ import annotations

import pandas as pd
import pytest

from advanced_visualization.cli.finalize_prediction_workspace import (
    complete_prediction_table,
)


def test_complete_prediction_table_keeps_metadata_and_drops_incomplete_rows() -> None:
    merged = pd.DataFrame(
        {
            "uuid": ["one", "two", "three"],
            "label": [0, 1, 0],
            "absolute_ori_path": ["a.jpg", "b.jpg", "c.jpg"],
            "model_a": [0.1, 0.9, None],
            "model_b": [0.2, None, 0.3],
        }
    )

    result = complete_prediction_table(
        merged,
        ["uuid", "label", "absolute_ori_path"],
        ["model_a", "model_b"],
    )

    assert result.to_dict("records") == [
        {
            "uuid": "one",
            "label": 0,
            "absolute_ori_path": "a.jpg",
            "model_a": 0.1,
            "model_b": 0.2,
        }
    ]


def test_complete_prediction_table_rejects_missing_contract_columns() -> None:
    with pytest.raises(ValueError, match="model_b"):
        complete_prediction_table(
            pd.DataFrame({"uuid": ["one"], "model_a": [0.1]}),
            ["uuid"],
            ["model_a", "model_b"],
        )
