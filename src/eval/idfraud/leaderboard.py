"""Generate leaderboard with weighted scores across batch directories."""
import re
import argparse
import pandas as pd
from calculate import calculate_metrics

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
    'batch_issue_set': {
        'weights': {'apcer': 0, 'bpcer': -0.0769},
    },
    'batch_issue_20240412': {
        'weights': {'apcer': -0.0769, 'bpcer': -0.0769},
    }
}

def classify_dataset(dataset_name):
    """Classify dataset and return (category, weights)."""
    if dataset_name in SPECIAL_CASES:
        return 'special', SPECIAL_CASES[dataset_name]['weights']

    for category, rules in DATASET_RULES.items():
        for pattern in rules['patterns']:
            if re.match(pattern, dataset_name, re.IGNORECASE):
                return category, rules['weights']

    return 'unknown', {'apcer': 1.0, 'bpcer': 1.0}


def calculate_weighted_score(metrics_per_batch):
    """
    Calculate weighted score from per-batch metrics.

    Formula: score = Σ((1 - metric) * |weight|) / Σ(|weight|)
    """
    score = 0
    total_weight = 0

    for batch_name, metrics in metrics_per_batch.items():
        _, weights = classify_dataset(batch_name)

        for metric_name in ['apcer', 'bpcer']:
            weight = weights.get(metric_name, 0)
            if weight != 0:
                total_weight += abs(weight)
                score += (1 - metrics[metric_name]) * abs(weight)

    return round(score / total_weight, 4) if total_weight > 0 else 0


def evaluate_epoch(df, prob_col, threshold=0.5):
    """Calculate metrics per batch_directory for one epoch."""
    metrics_per_batch = {}

    for batch_name in df['batch_directory'].unique():
        batch_df = df[df['batch_directory'] == batch_name]
        metrics = calculate_metrics(batch_df, threshold, prob_col, 'label')
        metrics_per_batch[batch_name] = metrics

    return metrics_per_batch


def generate_leaderboard(df, threshold=0.5):
    """Generate leaderboard for all checkpoints."""
    # Find all pred_prob_ckpt columns
    pattern = re.compile(r'^pred_prob_ckpt(\d+)$')
    ckpt_cols = []
    for col in df.columns:
        match = pattern.match(col)
        if match:
            ckpt_cols.append((int(match.group(1)), col))
    ckpt_cols.sort(key=lambda x: x[0])

    results = []
    for epoch, col in ckpt_cols:
        metrics_per_batch = evaluate_epoch(df, col, threshold)
        weighted_score = calculate_weighted_score(metrics_per_batch)

        # Also calculate overall metrics
        overall = calculate_metrics(df, threshold, col, 'label')

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', help='Path to predictions CSV',default=)
    parser.add_argument('--threshold', '-t', type=float, default=0.5)
    parser.add_argument('--output', '-o', help='Output CSV path')
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    results_df = generate_leaderboard(df, args.threshold)

    print(f"\n{'='*60}")
    best = results_df.loc[results_df['weighted_score'].idxmax()]
    print(f"BEST: Epoch {int(best['epoch'])}, score={best['weighted_score']:.4f}")
    print(f"  APCER={best['overall_apcer']*100:.2f}%, BPCER={best['overall_bpcer']*100:.2f}%")

    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")
