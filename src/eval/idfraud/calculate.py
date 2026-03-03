"""Calculate APCER, BPCER, ACER from predictions CSV."""
import pandas as pd


def calculate_metrics(df, threshold=0.5, prob_col='pred_prob', target_col='label'):
    """
    Calculate metrics from predictions DataFrame.

    Args:
        df: DataFrame with predictions and labels
        threshold: Classification threshold
        prob_col: Column with predicted probabilities
        label_col: Column with ground truth (1=attack, 0=bonafide)

    Returns:
        Dict with TP, FP, TN, FN, apcer, bpcer, acer, accuracy
    """
    y_true = df[target_col].values
    y_pred = (df[prob_col].values > threshold).astype(int)

    TP = int(((y_pred == 1) & (y_true == 1)).sum())  # attack correctly classified
    FP = int(((y_pred == 1) & (y_true == 0)).sum())  # bonafide misclassified as attack
    TN = int(((y_pred == 0) & (y_true == 0)).sum())  # bonafide correctly classified
    FN = int(((y_pred == 0) & (y_true == 1)).sum())  # attack misclassified as bonafide

    # APCER: Attack samples wrongly classified as bonafide / total attacks
    apcer = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    # BPCER: Bonafide samples wrongly classified as attack / total bonafide
    bpcer = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    # ACER: Average of APCER and BPCER
    acer = (apcer + bpcer) / 2

    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    return {
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'apcer': apcer, 'bpcer': bpcer, 'acer': acer, 'accuracy': accuracy
    }


def print_metrics(metrics):
    """Print metrics in a formatted way."""
    print(f"\n{'='*40}")
    print(f"  TP: {metrics['TP']:>6}  |  FP: {metrics['FP']:>6}")
    print(f"  FN: {metrics['FN']:>6}  |  TN: {metrics['TN']:>6}")
    print(f"{'='*40}")
    print(f"  APCER:    {metrics['apcer']:.4f}")
    print(f"  BPCER:    {metrics['bpcer']:.4f}")
    print(f"  ACER:     {metrics['acer']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"{'='*40}\n")


def calculate_all_checkpoints(df, threshold=0.5, label_col='label'):
    """
    Calculate metrics for all pred_prob_ckpt{epoch} columns.
    
    Returns:
        List of dicts with epoch and metrics
    """
    import re

    # Find all columns matching pred_prob_ckpt{number}
    pattern = re.compile(r'^pred_prob_ckpt(\d+)$')
    ckpt_cols = []
    for col in df.columns:
        match = pattern.match(col)
        if match:
            epoch = int(match.group(1))
            ckpt_cols.append((epoch, col))

    # Sort by epoch number
    ckpt_cols.sort(key=lambda x: x[0])

    results = []
    for epoch, col in ckpt_cols:
        metrics = calculate_metrics(df, threshold, col, label_col)
        results.append({
            'epoch': epoch,
            **metrics
        })

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', help='Path to predictions CSV')
    parser.add_argument('--threshold', '-t', type=float, default=0.5)
    parser.add_argument('--prob_col', default='pred_prob')
    parser.add_argument('--label_col', default='label')
    parser.add_argument('--output', '-o', help='Output CSV for all checkpoints metrics')
    parser.add_argument('--all_ckpts', action='store_true', help='Calculate metrics for all pred_prob_ckpt* columns')
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    if args.all_ckpts:
        results = calculate_all_checkpoints(df, args.threshold, args.label_col)
        if not results:
            print("No pred_prob_ckpt* columns found!")
        else:
            results_df = pd.DataFrame(results)
            print(results_df.to_string(index=False))

            if args.output:
                results_df.to_csv(args.output, index=False)
                print(f"\nSaved metrics to {args.output}")
    else:
        metrics = calculate_metrics(df, args.threshold, args.prob_col, args.label_col)
        print_metrics(metrics)
