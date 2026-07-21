"""Pure dataframe alignment and model-comparison logic."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from advanced_visualization.core.evaluation import BinaryEvaluation, evaluate_binary, evaluation_metrics


COMPARISON_OUTCOMES = (
    "both_correct",
    "a_only_correct",
    "b_only_correct",
    "both_wrong",
    "unscored",
    "truth_mismatch",
    "only_in_a",
    "only_in_b",
)


@dataclass(frozen=True)
class ComparisonResult:
    frame: pd.DataFrame
    evaluation_a: BinaryEvaluation
    evaluation_b: BinaryEvaluation
    alignment: dict[str, int]

    def summary(self, mask: pd.Series | None = None) -> dict:
        selected = pd.Series(True, index=self.frame.index, dtype=bool) if mask is None else mask.reindex(
            self.frame.index, fill_value=False
        )
        comparable = selected & self.frame["__comparison_outcome"].isin(COMPARISON_OUTCOMES[:4])
        outcomes = self.frame.loc[selected, "__comparison_outcome"].value_counts().to_dict()
        return {
            "rows": int(selected.sum()),
            "comparable_rows": int(comparable.sum()),
            "prediction_disagreements": int((comparable & self.frame["__prediction_disagreement"]).sum()),
            "a": evaluation_metrics(self.evaluation_a, comparable),
            "b": evaluation_metrics(self.evaluation_b, comparable),
            "outcomes": {outcome: int(outcomes.get(outcome, 0)) for outcome in COMPARISON_OUTCOMES},
            "alignment": self.alignment,
        }


def _require_columns(frame: pd.DataFrame, columns: list[str], experiment: str) -> None:
    missing = [column for column in columns if not column or column not in frame.columns]
    if missing:
        raise ValueError(f"Experiment {experiment} is missing required column(s): {missing}")


def _comparison_keys(series: pd.Series) -> pd.Series:
    keys = series.astype("string").str.strip()
    return keys.mask(keys.eq(""))


def compare_frames(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    item_id_column_a: str,
    item_id_column_b: str,
    truth_column_a: str,
    truth_column_b: str,
    prediction_column_a: str,
    prediction_column_b: str,
    threshold_a: float = 0.5,
    threshold_b: float = 0.5,
    metadata_columns_a: tuple[str, ...] = (),
    metadata_columns_b: tuple[str, ...] = (),
) -> ComparisonResult:
    """Align two experiment frames by stable ID and classify every outcome."""
    _require_columns(
        frame_a,
        [item_id_column_a, truth_column_a, prediction_column_a, "__row_id"],
        "A",
    )
    _require_columns(
        frame_b,
        [item_id_column_b, truth_column_b, prediction_column_b, "__row_id"],
        "B",
    )

    keys_a = _comparison_keys(frame_a[item_id_column_a])
    keys_b = _comparison_keys(frame_b[item_id_column_b])
    duplicate_ids_a = int(keys_a[keys_a.notna() & keys_a.duplicated(keep=False)].nunique())
    duplicate_ids_b = int(keys_b[keys_b.notna() & keys_b.duplicated(keep=False)].nunique())

    columns_a = list(dict.fromkeys([
        "__row_id", item_id_column_a, truth_column_a, prediction_column_a, *metadata_columns_a,
    ]))
    columns_b = list(dict.fromkeys([
        "__row_id", item_id_column_b, truth_column_b, prediction_column_b, *metadata_columns_b,
    ]))
    left = frame_a.loc[keys_a.notna(), [column for column in columns_a if column in frame_a]].copy()
    right = frame_b.loc[keys_b.notna(), [column for column in columns_b if column in frame_b]].copy()
    left.insert(0, "__comparison_id", keys_a.loc[left.index].astype(str))
    right.insert(0, "__comparison_id", keys_b.loc[right.index].astype(str))
    # Repeated IDs are aligned deterministically without creating a Cartesian join.
    left.insert(1, "__comparison_occurrence", left.groupby("__comparison_id", sort=False).cumcount())
    right.insert(1, "__comparison_occurrence", right.groupby("__comparison_id", sort=False).cumcount())
    join_columns = {"__comparison_id", "__comparison_occurrence"}
    left = left.rename(columns={column: f"a__{column}" for column in left.columns if column not in join_columns})
    right = right.rename(columns={column: f"b__{column}" for column in right.columns if column not in join_columns})
    merged = left.merge(
        right,
        on=["__comparison_id", "__comparison_occurrence"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    truth_a_name = f"a__{truth_column_a}"
    truth_b_name = f"b__{truth_column_b}"
    score_a_name = f"a__{prediction_column_a}"
    score_b_name = f"b__{prediction_column_b}"
    truth_a = pd.to_numeric(merged[truth_a_name], errors="coerce")
    truth_b = pd.to_numeric(merged[truth_b_name], errors="coerce")
    matched = merged["_merge"].eq("both")
    truth_mismatch = matched & truth_a.notna() & truth_b.notna() & truth_a.ne(truth_b)
    merged["__truth"] = truth_a.where(~truth_a.isna(), truth_b).mask(truth_mismatch)

    evaluation_a = evaluate_binary(merged, "__truth", score_a_name, threshold_a)
    evaluation_b = evaluate_binary(merged, "__truth", score_b_name, threshold_b)
    both_evaluable = matched & ~truth_mismatch & evaluation_a.has_eval & evaluation_b.has_eval
    a_correct = both_evaluable & ~evaluation_a.is_failure
    b_correct = both_evaluable & ~evaluation_b.is_failure
    merged["__comparison_outcome"] = np.select(
        [
            merged["_merge"].eq("left_only"),
            merged["_merge"].eq("right_only"),
            truth_mismatch,
            matched & ~both_evaluable,
            a_correct & b_correct,
            a_correct & ~b_correct,
            ~a_correct & b_correct,
        ],
        [
            "only_in_a",
            "only_in_b",
            "truth_mismatch",
            "unscored",
            "both_correct",
            "a_only_correct",
            "b_only_correct",
        ],
        default="both_wrong",
    )
    merged["__prediction_disagreement"] = both_evaluable & evaluation_a.predicted_positive.ne(
        evaluation_b.predicted_positive
    )
    merged["__score_delta"] = evaluation_a.score - evaluation_b.score
    merged["__true_confidence_delta"] = (
        evaluation_a.true_class_confidence - evaluation_b.true_class_confidence
    )
    merged["__a_failure_type"] = evaluation_a.failure_type
    merged["__b_failure_type"] = evaluation_b.failure_type
    merged["__transition"] = evaluation_a.failure_type.astype(str) + " → " + evaluation_b.failure_type.astype(str)
    merged["__a_score"] = evaluation_a.score
    merged["__b_score"] = evaluation_b.score
    merged["__a_correct"] = a_correct
    merged["__b_correct"] = b_correct

    alignment = {
        "source_a_rows": int(len(frame_a)),
        "source_b_rows": int(len(frame_b)),
        "matched": int(matched.sum()),
        "only_in_a": int(merged["_merge"].eq("left_only").sum()),
        "only_in_b": int(merged["_merge"].eq("right_only").sum()),
        "missing_id_a": int(keys_a.isna().sum()),
        "missing_id_b": int(keys_b.isna().sum()),
        "truth_mismatches": int(truth_mismatch.sum()),
        "duplicate_ids_a": duplicate_ids_a,
        "duplicate_ids_b": duplicate_ids_b,
    }
    return ComparisonResult(merged, evaluation_a, evaluation_b, alignment)
