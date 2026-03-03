#!/usr/bin/env python3
"""
Build leaderboard from evaluation CSV using dataset classification rules.
Preserves weighted scoring based on dataset type (production, datacollection, issue, test_plan).
"""
import os
import re
import sys
import pandas as pd
from typing import Dict, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from idfraud.calculate import calculate_metrics


# Dataset classification rules based on naming patterns
DATASET_RULES = {
    'production': {
        'patterns': [
            r'batch_production_\d+.*',
            r'\d+_myekyc_.*',
        ],
        'weights': {'apcer': 1.0, 'bpcer': 1.0},
    },
    'datacollection': {
        'patterns': [
            r'batch_datacollection_\d+.*',
        ],
        'weights': {'apcer': 1.0, 'bpcer': 1.0},
    },
    'issue': {
        'patterns': [
            r'batch_issue_\d+.*',
        ],
        'weights': {'apcer': 0.0769, 'bpcer': 0},
    },
    'test_plan': {
        'patterns': [
            r'.*test_plan_subject.*',
            r'grayscale_print.*',
            r'colorprint.*',
            r'replay_.*',
        ],
        'weights': {'apcer': 1.0, 'bpcer': 1.0},
    },
}

# Special cases with exact match
SPECIAL_CASES = {
    'batch_issue_set-index_annotation_mykadfront_background': {
        'weights': {'apcer': 0, 'bpcer': 0.0769},
    },
    'batch_issue_20240412_partner_both_general-index_annotation_mykadfront': {
        'weights': {'apcer': 0.0769, 'bpcer': 0.0769},
    }
}


def classify_dataset(dataset_name: str) -> Tuple[str, Dict[str, float]]:
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
    return 'unknown', {'apcer': 1.0, 'bpcer': 1.0}


def build_leaderboard(csv_path: str, group_by=['model', 'epoch'], threshold=0.5):
    """
    Build weighted leaderboard from evaluation CSV.

    Args:
        csv_path: Path to CSV with columns: label, pred_prob, dataset, and group_by columns
        group_by: Columns to group by (e.g., ['model', 'epoch'])
        threshold: Classification threshold

    Returns:
        DataFrame with weighted scores per model/epoch
    """
    df = pd.read_csv(csv_path)
    rows = []

    for keys, model_group in df.groupby(group_by):
        if not isinstance(keys, tuple):
            keys = (keys,)

        # Calculate weighted score across all datasets for this model
        total_weighted_score = 0.0
        total_weight = 0.0
        dataset_metrics = []

        for dataset_name, dataset_group in model_group.groupby('dataset'):
            metrics = calculate_metrics(dataset_group, threshold=threshold)
            category, weights = classify_dataset(dataset_name)

            # Weighted contribution: (1 - error_rate) * weight
            apcer_contrib = (1 - metrics['apcer']) * weights['apcer']
            bpcer_contrib = (1 - metrics['bpcer']) * weights['bpcer']

            total_weighted_score += apcer_contrib + bpcer_contrib
            total_weight += weights['apcer'] + weights['bpcer']

            dataset_metrics.append({
                'dataset': dataset_name,
                'category': category,
                **metrics
            })

        # Normalize score
        final_score = total_weighted_score / total_weight if total_weight > 0 else 0

        # Aggregate metrics across all datasets
        all_metrics = calculate_metrics(model_group, threshold=threshold)

        row = dict(zip(group_by, keys))
        row['score'] = round(final_score, 4)
        row.update(all_metrics)
        row['n_datasets'] = len(dataset_metrics)
        rows.append(row)

    leaderboard = pd.DataFrame(rows)
    leaderboard = leaderboard.sort_values('score', ascending=False).reset_index(drop=True)

    cols = group_by + ['score', 'acer', 'apcer', 'bpcer', 'accuracy', 'n_datasets', 'TP', 'FP', 'TN', 'FN']
    return leaderboard[[c for c in cols if c in leaderboard.columns]]


def build_detailed_leaderboard(csv_path: str, group_by=['model', 'epoch'], threshold=0.5):
    """Build leaderboard with per-dataset breakdown."""
    df = pd.read_csv(csv_path)
    rows = []

    for keys, model_group in df.groupby(group_by):
        if not isinstance(keys, tuple):
            keys = (keys,)

        for dataset_name, dataset_group in model_group.groupby('dataset'):
            metrics = calculate_metrics(dataset_group, threshold=threshold)
            category, weights = classify_dataset(dataset_name)

            row = dict(zip(group_by, keys))
            row['dataset'] = dataset_name
            row['category'] = category
            row['weight_apcer'] = weights['apcer']
            row['weight_bpcer'] = weights['bpcer']
            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows).sort_values(group_by + ['dataset']).reset_index(drop=True)


def print_leaderboard(lb):
    print("\n" + "=" * 80)
    print("LEADERBOARD (higher score = better)")
    print("=" * 80)
    print(lb.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', help='Path to evaluation CSV')
    parser.add_argument('--group-by', '-g', nargs='+', default=['model', 'epoch'])
    parser.add_argument('--threshold', '-t', type=float, default=0.5)
    parser.add_argument('--detailed', '-d', action='store_true', help='Show per-dataset breakdown')
    parser.add_argument('--output', '-o', help='Save leaderboard CSV')
    args = parser.parse_args()

    if args.detailed:
        lb = build_detailed_leaderboard(args.csv_path, args.group_by, args.threshold)
    else:
        lb = build_leaderboard(args.csv_path, args.group_by, args.threshold)

    print_leaderboard(lb)

    if args.output:
        lb.to_csv(args.output, index=False)
        print(f"Saved: {args.output}")
