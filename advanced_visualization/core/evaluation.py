"""Framework-independent binary evaluation and grouped metric helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BinaryEvaluation:
    """Vectorized binary-classification state for one prediction column."""

    score: pd.Series
    predicted_positive: pd.Series
    actual_positive: pd.Series
    truth_valid: pd.Series
    truth_invalid: pd.Series
    has_eval: pd.Series
    is_failure: pd.Series
    failure_type: pd.Series
    confidence: pd.Series
    true_class_confidence: pd.Series


def evaluate_binary(
    frame: pd.DataFrame,
    truth_column: str | None,
    prediction_column: str | None,
    threshold: float,
    *,
    positive_value: int = 1,
    negative_value: int = 0,
    invalid_value: int = -1,
    inclusive: bool = True,
) -> BinaryEvaluation:
    """Evaluate a binary score without mutating the source dataframe."""
    index = frame.index
    if not truth_column or not prediction_column or truth_column not in frame or prediction_column not in frame:
        false = pd.Series(False, index=index, dtype=bool)
        missing = pd.Series(np.nan, index=index, dtype=float)
        return BinaryEvaluation(
            score=missing,
            predicted_positive=false,
            actual_positive=false,
            truth_valid=false,
            truth_invalid=false,
            has_eval=false,
            is_failure=false,
            failure_type=pd.Series("unscored", index=index, dtype="object"),
            confidence=missing.copy(),
            true_class_confidence=missing.copy(),
        )

    score = pd.to_numeric(frame[prediction_column], errors="coerce")
    truth = pd.to_numeric(frame[truth_column], errors="coerce")
    predicted_positive = score.ge(threshold) if inclusive else score.gt(threshold)
    actual_positive = truth.eq(positive_value)
    truth_valid = actual_positive | truth.eq(negative_value)
    truth_invalid = truth.eq(invalid_value)
    has_eval = score.notna() & truth_valid
    failure = predicted_positive.ne(actual_positive) & has_eval
    confidence = np.maximum(score, 1.0 - score)
    true_class_confidence = pd.Series(
        np.where(actual_positive, score, 1.0 - score),
        index=index,
        dtype=float,
    ).where(has_eval)
    failure_type = pd.Series(
        np.select(
            [
                ~has_eval,
                predicted_positive & ~actual_positive,
                ~predicted_positive & actual_positive,
                ~failure,
            ],
            ["unscored", "false positive", "false negative", "correct"],
            default="failure",
        ),
        index=index,
        dtype="object",
    )
    return BinaryEvaluation(
        score=score,
        predicted_positive=predicted_positive,
        actual_positive=actual_positive,
        truth_valid=truth_valid,
        truth_invalid=truth_invalid,
        has_eval=has_eval,
        is_failure=failure,
        failure_type=failure_type,
        confidence=confidence,
        true_class_confidence=true_class_confidence,
    )


def attach_binary_evaluation(frame: pd.DataFrame, evaluation: BinaryEvaluation, prefix: str = "__") -> pd.DataFrame:
    """Return a copy with conventional evaluation columns attached."""
    result = frame.copy()
    result[f"{prefix}has_eval"] = evaluation.has_eval
    result[f"{prefix}prediction_score"] = evaluation.score
    result[f"{prefix}predicted_positive"] = evaluation.predicted_positive
    result[f"{prefix}actual_positive"] = evaluation.actual_positive
    result[f"{prefix}truth_is_valid"] = evaluation.truth_valid
    result[f"{prefix}truth_is_invalid"] = evaluation.truth_invalid
    result[f"{prefix}is_failure"] = evaluation.is_failure
    result[f"{prefix}failure_type"] = evaluation.failure_type
    result[f"{prefix}confidence"] = evaluation.confidence
    result[f"{prefix}true_class_confidence"] = evaluation.true_class_confidence
    return result


def evaluation_metrics(evaluation: BinaryEvaluation, mask: pd.Series | None = None) -> dict[str, int | float]:
    """Return numeric accuracy, APCER, and BPCER metrics."""
    selected = pd.Series(True, index=evaluation.has_eval.index, dtype=bool) if mask is None else mask.reindex(
        evaluation.has_eval.index, fill_value=False
    )
    evaluable = selected & evaluation.has_eval
    attacks = evaluable & evaluation.actual_positive
    genuine = evaluable & ~evaluation.actual_positive
    false_negatives = int((attacks & ~evaluation.predicted_positive).sum())
    false_positives = int((genuine & evaluation.predicted_positive).sum())
    failures = int((evaluable & evaluation.is_failure).sum())
    rows = int(evaluable.sum())
    attack_rows = int(attacks.sum())
    genuine_rows = int(genuine.sum())
    return {
        "rows": rows,
        "correct": rows - failures,
        "failures": failures,
        "accuracy": (rows - failures) / rows if rows else 0.0,
        "apcer": false_negatives / attack_rows if attack_rows else 0.0,
        "apcer_errors": false_negatives,
        "attack_rows": attack_rows,
        "bpcer": false_positives / genuine_rows if genuine_rows else 0.0,
        "bpcer_errors": false_positives,
        "genuine_rows": genuine_rows,
    }
