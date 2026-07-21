from pathlib import Path

import pandas as pd
import pytest

from advanced_visualization.core.dataframe_filters import (
    apply_categorical_filters,
    apply_numeric_ranges,
    apply_text_search,
)
from advanced_visualization.core.feature_data import (
    apply_merge_mapping,
    feature_columns,
    limit_named_group,
    parse_merge_mapping,
)
from advanced_visualization.core.gradcam_generation import (
    GradcamGenerationOptions,
    apply_filters as apply_gradcam_filters,
    output_root_for_config,
)
from advanced_visualization.core.preparation import prepare_dataframe


def test_shared_dataframe_filters_ignore_unknown_columns() -> None:
    frame = pd.DataFrame(
        {
            "group": ["a", "b", None],
            "score": [0.1, 0.5, 0.9],
            "name": ["Alpha", "Beta", "Gamma"],
        }
    )

    searched = apply_text_search(frame, "alp", ["name", "unknown"])
    categorical = apply_categorical_filters(frame, {"group": ["Missing"]})
    ranged = apply_numeric_ranges(frame, {"score": (0.4, 0.8), "unknown": (0, 1)})

    assert searched.index.tolist() == [0]
    assert categorical.index.tolist() == [2]
    assert ranged.index.tolist() == [1]


def test_feature_helpers_parse_merge_rules_and_limit_one_group() -> None:
    frame = pd.DataFrame(
        {
            "feature_0": [1.0, 2.0, 3.0, 4.0],
            "group": ["Genuine", "Genuine", "Genuine", "Fraud"],
        }
    )

    assert feature_columns(frame) == ["feature_0"]
    assert parse_merge_mapping("printed=print,copy") == {
        "print": "printed",
        "copy": "printed",
    }
    assert apply_merge_mapping(
        pd.Series(["print", "other"]), "printed=print,copy"
    ).tolist() == ["printed", "other"]

    limited = limit_named_group(
        frame,
        group_column="group",
        group_name="genuine",
        max_rows=1,
        random_state=7,
    )
    assert limited["group"].value_counts().to_dict() == {"Genuine": 1, "Fraud": 1}
    assert "sampling_note" in limited.attrs


def test_prepare_dataframe_adds_only_known_standard_columns() -> None:
    frame = pd.DataFrame({"uuid": ["a"], "pred": [0.7]})
    prepared = prepare_dataframe(
        frame,
        {
            "item_id_column": "uuid",
            "prediction_column": "pred",
            "image_column": "missing",
        },
    )

    assert prepared["__item_id_column"].tolist() == ["a"]
    assert prepared["__prediction_column"].tolist() == [0.7]
    assert "__image_column" not in prepared
    assert frame.columns.tolist() == ["uuid", "pred"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"offset": -1}, "offset"),
        ({"limit": -1}, "limit"),
        ({"num_shards": 0}, "num_shards"),
        ({"shard_index": 2, "num_shards": 2}, "shard_index"),
        ({"batch_size": 0}, "batch_size"),
        ({"prefetch_factor": 0}, "prefetch_factor"),
        ({"cam_methods": ("unknown",)}, "cam_methods"),
        ({"cam_targets": ()}, "cam_targets"),
    ],
)
def test_gradcam_options_validate(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GradcamGenerationOptions(**overrides).validate()


def test_gradcam_filters_apply_offset_limit_and_shard() -> None:
    frame = pd.DataFrame(
        {
            "group": ["a", "a", "b", "a", "a", "a"],
            "value": list(range(6)),
        }
    )
    options = GradcamGenerationOptions(
        filters=("group=a",),
        offset=1,
        limit=4,
        num_shards=2,
        shard_index=1,
    )

    assert apply_gradcam_filters(frame, options)["value"].tolist() == [3, 5]


def test_gradcam_output_root_uses_explicit_root(tmp_path: Path) -> None:
    options = GradcamGenerationOptions(output_root=tmp_path)
    assert output_root_for_config("experiment", options) == tmp_path / "experiment"
