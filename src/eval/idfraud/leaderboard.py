import pandas as pd
import re
from pathlib import Path
from typing import Dict, Tuple
from utils.device import is_main_process
from eval.idfraud.metrics import compute_binary_metrics

DATASET_RULES = {
    'production': {
        'patterns': [
            r'batch_production_\d+.*',
            r'\d+_myekyc_.*',
        ],
        'weights': {'apcer': -1.0, 'bpcer': -1.0},
    },
    'datacollection': {
        'patterns': [
            r'batch_datacollection_\d+.*',
        ],
        'weights': {'apcer': -1.0, 'bpcer': -1.0},
    },
    'issue': {
        'patterns': [
            r'batch_issue_\d+.*',
        ],
        'weights': {'apcer': -0.0769, 'bpcer': 0},
    },
    'test_plan': {
        'patterns': [
            r'.*test_plan_subject.*',
            r'grayscale_print.*',
            r'colorprint.*',
            r'replay_.*',
        ],
        'weights': {'apcer': -1.0, 'bpcer': -1.0},
    },
}

# Special cases with exact match
SPECIAL_CASES = {
    'batch_issue_set/index_annotation_mykadfront_background.csv': {
        'weights': {'apcer': 0, 'bpcer': -0.0769},
    },
    'batch_issue_20240412_partner_both_general/index_annotation_mykadfront': {
        'weights': {'apcer': -0.0769, 'bpcer': -0.0769},
    }
}


def _classify_dataset(dataset_name: str) -> Tuple[str, Dict[str, float]]:
    """Classify dataset and return (category, weights)."""
    # Check special cases first
    if dataset_name in SPECIAL_CASES:
        return 'special', SPECIAL_CASES[dataset_name]['weights']

    # Check against pattern rules
    for category, rules in DATASET_RULES.items():
        for pattern in rules['patterns']:
            if re.match(pattern, dataset_name, re.IGNORECASE):
                return category, rules['weights']

    # Default
    return 'unknown', {'apcer': -1.0, 'bpcer': -1.0}


def calculate_metrics(df, prob_col, threshold=0.5, target_col='label'):
    """
    Calculate APCER/BPCER metrics from predictions DataFrame.

    Args:
        df: DataFrame with predictions and labels
        prob_col: Column with predicted probabilities
        threshold: Classification threshold (default 0.5)
        target_col: Column with ground truth (1=attack, 0=bonafide)

    Returns:
        Dict with TP, FP, TN, FN, apcer, bpcer, acer, accuracy
    """
    y_true = df[target_col].values
    y_pred = (df[prob_col].values > threshold).astype(int)

    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())

    apcer = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    bpcer = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    acer = (apcer + bpcer) / 2
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    return {
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'apcer': apcer, 'bpcer': bpcer, 'acer': acer, 'accuracy': accuracy
    }


def _calculate_weighted_score(metrics_per_batch):
    """
    Calculate weighted score from per-batch metrics.

    Formula: score = Σ((1 - metric) * |weight|) / Σ(|weight|)
    Higher score = better performance.
    """
    score = 0
    total_weight = 0

    for batch_name, metrics in metrics_per_batch.items():
        _, weights = _classify_dataset(batch_name)

        for metric_name in ['apcer', 'bpcer']:
            weight = weights.get(metric_name, 0)
            if weight != 0:
                total_weight += abs(weight)
                score += (1 - metrics[metric_name]) * abs(weight)

    return round(score / total_weight, 4) if total_weight > 0 else 0


def _evaluate_epoch(df, prob_col, threshold=0.5):
    """Calculate metrics per original_batch_name for one epoch."""
    metrics_per_batch = {}

    for batch_name in df['original_batch_name'].unique():
        batch_df = df[df['original_batch_name'] == batch_name]
        metrics = calculate_metrics(batch_df, prob_col, threshold)
        metrics_per_batch[batch_name] = metrics

    return metrics_per_batch


def _generate_leaderboard(df, threshold=0.5):
    """Generate leaderboard comparing all checkpoint columns."""
    ckpt_cols = sorted(
        [(int(re.search(r'ckpt(\d+)$', col).group(1)), col)
         for col in df.columns if re.match(r'^pred_prob_ckpt\d+$', col)]
    )

    results = []
    for epoch, col in ckpt_cols:
        metrics_per_batch = _evaluate_epoch(df, col, threshold)
        weighted_score = _calculate_weighted_score(metrics_per_batch)
        overall = calculate_metrics(df, col, threshold)

        results.append({
            'epoch': epoch,
            'weighted_score': weighted_score,
            'overall_apcer': overall['apcer'],
            'overall_bpcer': overall['bpcer'],
            'overall_acer': overall['acer'],
        })
        print(f"Epoch {epoch:>2}: score={weighted_score:.4f}, "
              f"APCER={overall['apcer']*100:.2f}%, BPCER={overall['bpcer']*100:.2f}%")

    return pd.DataFrame(results)


def create_leaderboard(cfg):
    if not is_main_process():
        return None
    
    """Create leaderboard from config - callable from train.py."""
    csv_path = Path(cfg.run_dir) / 'eval_predictions.csv'

    if not csv_path.exists():
        print(f"Predictions file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(df)} rows for leaderboard")

    threshold = getattr(cfg.training, 'training_threshold', 0.5)
    leaderboard = _generate_leaderboard(df, threshold=threshold)
    leaderboard_sorted = leaderboard.sort_values('weighted_score', ascending=False)

    # Save leaderboard
    output_path = csv_path.with_name('leaderboard.csv')
    leaderboard_sorted.to_csv(output_path, index=False)
    print(f"Leaderboard saved to {output_path}")

    # Best epoch summary
    best = leaderboard_sorted.iloc[0]
    print(f"\nBest epoch: {int(best['epoch'])}")
    print(f"  Weighted Score: {best['weighted_score']:.4f}")
    print(f"  Overall APCER:  {best['overall_apcer']*100:.2f}%")
    print(f"  Overall BPCER:  {best['overall_bpcer']*100:.2f}%")

    # Save breakdown for best epoch
    best_col = f"pred_prob_ckpt{int(best['epoch'])}"
    metrics_per_batch = _evaluate_epoch(df, best_col, threshold)

    breakdown = []
    for batch_name, metrics in metrics_per_batch.items():
        category, weights = _classify_dataset(batch_name)
        breakdown.append({
            'batch': batch_name,
            'category': category,
            'n_genuine': metrics['TN'] + metrics['FP'],
            'n_attack': metrics['TP'] + metrics['FN'],
            'apcer': metrics['apcer'],
            'bpcer': metrics['bpcer'],
            'apcer_weight': weights['apcer'],
            'bpcer_weight': weights['bpcer'],
        })

    breakdown_df = pd.DataFrame(breakdown).sort_values('category')
    breakdown_path = csv_path.with_name('best_ckpt_breakdown.csv')
    breakdown_df.to_csv(breakdown_path, index=False)
    print(f"Breakdown saved to {breakdown_path}")

    return leaderboard_sorted
