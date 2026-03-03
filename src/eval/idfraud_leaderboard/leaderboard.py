"""
Simplified leaderboard that computes metrics directly from prediction DataFrames.
No info.json pipeline required - just needs pred_prob and label columns.
"""
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Container for binary classification metrics."""
    TP: int
    FP: int
    TN: int
    FN: int
    apcer: float
    bpcer: float
    acer: float
    accuracy: float

    def to_dict(self) -> Dict:
        return {
            'TP': self.TP, 'FP': self.FP, 'TN': self.TN, 'FN': self.FN,
            'apcer': self.apcer, 'bpcer': self.bpcer,
            'acer': self.acer, 'accuracy': self.accuracy
        }


def compute_metrics(df: pd.DataFrame, threshold: float = 0.5,
                    prob_col: str = 'pred_prob', label_col: str = 'label') -> MetricResult:
    """
    Compute binary classification metrics from predictions DataFrame.

    Args:
        df: DataFrame with probability predictions and ground truth labels
        threshold: Classification threshold
        prob_col: Column name for predicted probabilities
        label_col: Column name for ground truth labels (1=attack, 0=bonafide)

    Returns:
        MetricResult with all computed metrics
    """
    y_true = df[label_col].values
    y_pred = (df[prob_col].values > threshold).astype(int)

    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())

    # APCER: Attack Presentation Classification Error Rate (false accept)
    # Proportion of attack samples incorrectly classified as bonafide
    apcer = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    # BPCER: Bonafide Presentation Classification Error Rate (false reject)
    # Proportion of bonafide samples incorrectly classified as attack
    bpcer = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    # ACER: Average Classification Error Rate
    acer = (apcer + bpcer) / 2

    # Accuracy
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0.0

    return MetricResult(
        TP=TP, FP=FP, TN=TN, FN=FN,
        apcer=apcer, bpcer=bpcer, acer=acer, accuracy=accuracy
    )


def compute_metrics_multi_threshold(df: pd.DataFrame,
                                     thresholds: List[float] = [0.3, 0.5, 0.7],
                                     prob_col: str = 'pred_prob',
                                     label_col: str = 'label') -> Dict[str, MetricResult]:
    """Compute metrics at multiple thresholds."""
    return {
        f"t{t}": compute_metrics(df, threshold=t, prob_col=prob_col, label_col=label_col)
        for t in thresholds
    }


class Leaderboard:
    """
    Generate leaderboard from prediction CSVs with weighted scoring.

    Usage:
        lb = Leaderboard(weights={'apcer': 1.0, 'bpcer': 1.0})
        lb.add_model('model_v1', 'predictions_v1.csv')
        lb.add_model('model_v2', 'predictions_v2.csv')
        df = lb.build()
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None, threshold: float = 0.5):
        """
        Args:
            weights: Dict of metric_name -> weight for scoring.
                     Default: {'acer': 1.0} (lower acer = better)
            threshold: Classification threshold for predictions
        """
        self.weights = weights or {'acer': 1.0}
        self.threshold = threshold
        self.models: List[Dict] = []

    def add_model(self, name: str, predictions: pd.DataFrame | str,
                  prob_col: str = 'pred_prob', label_col: str = 'label'):
        """
        Add a model's predictions to the leaderboard.

        Args:
            name: Model identifier
            predictions: DataFrame or path to CSV with predictions
            prob_col: Column name for probabilities
            label_col: Column name for ground truth
        """
        if isinstance(predictions, str):
            predictions = pd.read_csv(predictions)

        metrics = compute_metrics(
            predictions,
            threshold=self.threshold,
            prob_col=prob_col,
            label_col=label_col
        )

        self.models.append({
            'model': name,
            **metrics.to_dict()
        })

    def _calculate_score(self, row: Dict) -> float:
        """
        Calculate weighted score. Higher score = better model.

        For error metrics (apcer, bpcer, acer): score += (1 - metric) * weight
        For accuracy: score += metric * weight
        """
        score = 0.0
        total_weight = sum(abs(w) for w in self.weights.values())

        for metric, weight in self.weights.items():
            value = row[metric]

            # Invert error metrics (lower is better -> higher score)
            if metric in ('apcer', 'bpcer', 'acer'):
                value = 1 - value

            score += value * abs(weight)

        return round(score / total_weight, 4) if total_weight > 0 else 0.0

    def build(self, sort_by: str = 'score', ascending: bool = False) -> pd.DataFrame:
        """
        Build the leaderboard DataFrame.

        Args:
            sort_by: Column to sort by (default: 'score')
            ascending: Sort order (default: False, higher is better)

        Returns:
            DataFrame with all models and their metrics, sorted by score
        """
        if not self.models:
            return pd.DataFrame()

        df = pd.DataFrame(self.models)
        df['score'] = df.apply(lambda row: self._calculate_score(row.to_dict()), axis=1)
        df = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

        # Reorder columns
        cols = ['model', 'score', 'accuracy', 'acer', 'apcer', 'bpcer', 'TP', 'FP', 'TN', 'FN']
        df = df[[c for c in cols if c in df.columns]]

        return df

    def get_best_model(self) -> Dict:
        """Return the best model's info."""
        df = self.build()
        if df.empty:
            return {}
        return df.iloc[0].to_dict()


def build_leaderboard_from_csvs(prediction_files: Dict[str, str],
                                 threshold: float = 0.5,
                                 weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Convenience function to build leaderboard from prediction CSV paths.

    Args:
        prediction_files: {'model_name': 'path/to/predictions.csv', ...}
        threshold: Classification threshold
        weights: Metric weights for scoring

    Returns:
        Leaderboard DataFrame
    """
    lb = Leaderboard(weights=weights, threshold=threshold)
    for name, path in prediction_files.items():
        lb.add_model(name, path)
    return lb.build()


if __name__ == '__main__':
    # Example usage
    import numpy as np

    # Create dummy predictions for demo
    np.random.seed(42)
    n = 1000

    # Model 1: decent model
    df1 = pd.DataFrame({
        'label': np.random.randint(0, 2, n),
        'pred_prob': np.random.uniform(0, 1, n)
    })
    # Make it slightly better than random
    df1.loc[df1['label'] == 1, 'pred_prob'] += 0.2
    df1['pred_prob'] = df1['pred_prob'].clip(0, 1)

    # Model 2: better model
    df2 = pd.DataFrame({
        'label': np.random.randint(0, 2, n),
        'pred_prob': np.random.uniform(0, 1, n)
    })
    df2.loc[df2['label'] == 1, 'pred_prob'] += 0.4
    df2['pred_prob'] = df2['pred_prob'].clip(0, 1)

    # Build leaderboard
    lb = Leaderboard(weights={'acer': 1.0, 'accuracy': 0.5}, threshold=0.5)
    lb.add_model('model_v1', df1)
    lb.add_model('model_v2', df2)

    print("=" * 70)
    print("LEADERBOARD")
    print("=" * 70)
    print(lb.build().to_string(index=False))
    print("=" * 70)
    print(f"Best model: {lb.get_best_model()['model']}")
