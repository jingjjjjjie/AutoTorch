"""
Simple leaderboard generator.
Reads info.json files from eval results and creates a ranked leaderboard.

Usage:
    python leaderboard.py --results_dir ../runs/Experiment1/infer_results
"""
import os
import json
import argparse
import pandas as pd
from glob import glob


def load_eval_results(results_dir):
    """Load all info.json files from results directory."""
    results = []

    # Find all checkpoint folders
    ckpt_dirs = glob(os.path.join(results_dir, "*/"))

    for ckpt_dir in ckpt_dirs:
        ckpt_name = os.path.basename(ckpt_dir.rstrip("/"))

        # Find all dataset folders under this checkpoint
        dataset_dirs = glob(os.path.join(ckpt_dir, "*/"))

        for dataset_dir in dataset_dirs:
            dataset_name = os.path.basename(dataset_dir.rstrip("/"))
            info_path = os.path.join(dataset_dir, "info.json")

            if not os.path.exists(info_path):
                print(f"Warning: {info_path} not found, skipping")
                continue

            with open(info_path) as f:
                info = json.load(f)

            # Extract metrics (handle different threshold keys)
            metrics_dict = info.get("metrics", {})
            for threshold_key, metrics in metrics_dict.items():
                results.append({
                    "checkpoint": ckpt_name,
                    "dataset": dataset_name,
                    "threshold": threshold_key,
                    "TP": metrics.get("TP", 0),
                    "TN": metrics.get("TN", 0),
                    "FP": metrics.get("FP", 0),
                    "FN": metrics.get("FN", 0),
                    "accuracy": metrics.get("accuracy", 0),
                    "apcer": metrics.get("apcer", 0),
                    "bpcer": metrics.get("bpcer", 0),
                    "acer": metrics.get("acer", 0),
                })

    return pd.DataFrame(results)


def compute_score(df, weights=None):
    """
    Compute weighted score for ranking.

    Default: Lower ACER is better (score = 1 - acer)
    Custom weights: {"accuracy": 1, "apcer": -1, "bpcer": -1}
        - Positive weight = higher is better
        - Negative weight = lower is better
    """
    if weights is None:
        # Simple default: just use 1 - ACER (lower ACER = higher score)
        df["score"] = 1 - df["acer"]
    else:
        score = 0
        total_weight = 0
        for metric, weight in weights.items():
            if metric in df.columns:
                if weight < 0:
                    # Lower is better: invert the metric
                    score += (1 - df[metric]) * abs(weight)
                else:
                    # Higher is better
                    score += df[metric] * abs(weight)
                total_weight += abs(weight)
        df["score"] = score / total_weight if total_weight else 0

    return df


def generate_leaderboard(results_dir, output_path=None, weights=None, group_by_dataset=False):
    """Generate leaderboard from evaluation results."""

    # Load all results
    df = load_eval_results(results_dir)

    if df.empty:
        print("No results found!")
        return None

    # Compute score
    df = compute_score(df, weights)

    if group_by_dataset:
        # Average score across datasets for each checkpoint
        summary = df.groupby("checkpoint").agg({
            "accuracy": "mean",
            "apcer": "mean",
            "bpcer": "mean",
            "acer": "mean",
            "score": "mean",
        }).reset_index()
        summary = summary.sort_values("score", ascending=False)
    else:
        summary = df.sort_values("score", ascending=False)

    # Save
    if output_path:
        summary.to_csv(output_path, index=False)
        print(f"Leaderboard saved to: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate leaderboard from eval results")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing eval results")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: results_dir/leaderboard.csv)")
    parser.add_argument("--group", action="store_true", help="Group by checkpoint (average across datasets)")
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.results_dir, "leaderboard.csv")

    leaderboard = generate_leaderboard(
        results_dir=args.results_dir,
        output_path=output_path,
        group_by_dataset=args.group,
    )

    if leaderboard is not None:
        print("\n" + "=" * 70)
        print("LEADERBOARD (sorted by score, higher is better)")
        print("=" * 70)
        print(leaderboard.to_string(index=False))
        print("=" * 70)

        # Show best model
        best = leaderboard.iloc[0]
        print(f"\nBest: {best['checkpoint']} (score: {best['score']:.4f}, ACER: {best['acer']:.4f})")


if __name__ == "__main__":
    main()
